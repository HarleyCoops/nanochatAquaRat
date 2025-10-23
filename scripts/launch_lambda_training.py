#!/usr/bin/env python3
"""
Automation helper to launch a Lambda Labs instance and kick off the nanochat AQuA-RAT run.

Usage example:

    python scripts/launch_lambda_training.py \
        --ssh-key-name my-key \
        --region us-west-1 \
        --instance-type gpu_1x_a10 \
        --repo-url https://github.com/your-org/nanochatAquaRat.git \
        --auto-start \
        --inject-env WANDB_API_KEY

By default the script will create cloud-init user-data that installs basic tooling,
clones the repository, copies an `.env` file when provided, and (optionally) runs
`run_aquarat_small.sh` inside a detached tmux session. The Lambda Cloud API key is
read from the `LAMBDA_API_KEY` environment variable or `--api-key`.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import textwrap
import time
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests

API_BASE = "https://cloud.lambda.ai/api/v1"
DEFAULT_PACKAGES = ["git", "curl", "tmux", "build-essential"]


def log(msg: str) -> None:
    print(f"[info] {msg}")


def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)


def error(msg: str) -> None:
    print(f"[error] {msg}", file=sys.stderr)


def shell_quote(value: str) -> str:
    """Return a shell-escaped string for safe embedding inside scripts."""
    return shlex.quote(value)


def guess_repo_url() -> Optional[str]:
    """Attempt to infer the git remote URL for the current repository."""
    try:
        completed = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    url = completed.stdout.strip()
    return url or None


def collect_env_pairs(cli_pairs: Sequence[str], inject_names: Sequence[str]) -> List[Tuple[str, str]]:
    """
    Merge KEY=VALUE pairs declared via the CLI with variables injected from the local env.
    Later occurrences with the same key take precedence.
    """
    merged: "OrderedDict[str, str]" = OrderedDict()

    for item in cli_pairs:
        if "=" not in item:
            raise ValueError(f"--env expects KEY=VALUE entries, got '{item}'")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Environment key is empty in '{item}'")
        merged[key] = value

    for name in inject_names:
        if not name:
            raise ValueError("Encountered empty --inject-env name")
        if name not in os.environ:
            raise ValueError(f"--inject-env requested '{name}' but it is not set locally")
        merged[name] = os.environ[name]

    return list(merged.items())


def build_bootstrap_script(
    repo_dir: str,
    run_script: str,
    branch: str,
    repo_url: Optional[str],
    env_file_remote: str,
    auto_start: bool,
    tmux_session: str,
) -> str:
    """Compose the bash script executed on the instance to prepare the training run."""
    lines: List[str] = [
        "#!/usr/bin/env bash",
        "set -euxo pipefail",
        f'REPO_DIR="$HOME/{repo_dir}"',
        f'RUN_SCRIPT="{run_script}"',
        f'ENV_FILE="{env_file_remote}"',
        f'AUTO_START="{1 if auto_start else 0}"',
    ]

    if repo_url:
        lines.append(f"REPO_URL={shell_quote(repo_url)}")
        lines.extend(
            [
                'if [ ! -d "$REPO_DIR/.git" ]; then',
                '  rm -rf "$REPO_DIR"',
                '  git clone "$REPO_URL" "$REPO_DIR"',
                "fi",
                'cd "$REPO_DIR"',
                "git fetch --all --prune",
                f"git switch {shell_quote(branch)}",
                "git pull --ff-only || true",
            ]
        )
    else:
        lines.extend(
            [
                'mkdir -p "$REPO_DIR"',
                'cd "$REPO_DIR"',
            ]
        )

    lines.extend(
        [
            'if [ -f "$ENV_FILE" ]; then',
            '  cp "$ENV_FILE" .env',
            "fi",
            'if [ -f "$RUN_SCRIPT" ]; then',
            '  chmod +x "$RUN_SCRIPT"',
            "else",
            '  echo "Run script $RUN_SCRIPT not found; auto-start will be skipped." >&2',
            '  AUTO_START="0"',
            "fi",
        ]
    )

    if auto_start:
        tmux_line = (
            f'tmux new -d -s {shell_quote(tmux_session)} '
            '"cd \\"$REPO_DIR\\" && bash \\"$RUN_SCRIPT\\""'
        )
        nohup_line = (
            'nohup bash -lc "cd \\"$REPO_DIR\\" && bash \\"$RUN_SCRIPT\\"" '
            '> "$HOME/nanochat-train.log" 2>&1 &'
        )
        lines.extend(
            [
                'if [ "$AUTO_START" = "1" ]; then',
                "  if command -v tmux >/dev/null 2>&1; then",
                f"    {tmux_line}",
                "  else",
                f"    {nohup_line}",
                "  fi",
                "fi",
            ]
        )

    return "\n".join(lines) + "\n"


def build_user_data(
    packages: Sequence[str],
    bootstrap_script: str,
    env_pairs: Sequence[Tuple[str, str]],
    env_file_remote: str,
) -> str:
    """Render cloud-init user-data with package installs, env file, and bootstrap script."""
    lines: List[str] = ["#cloud-config", "package_update: true", "package_upgrade: false"]

    if packages:
        lines.append("packages:")
        for package in packages:
            lines.append(f"  - {package}")

    lines.append("write_files:")
    if env_pairs:
        env_content = "\n".join(f"{key}={value}" for key, value in env_pairs) + "\n"
        lines.extend(
            [
                f"  - path: {env_file_remote}",
                "    owner: ubuntu:ubuntu",
                "    permissions: '0640'",
                "    content: |",
                textwrap.indent(env_content, "      "),
            ]
        )

    lines.extend(
        [
            "  - path: /home/ubuntu/bootstrap_nanochat.sh",
            "    owner: ubuntu:ubuntu",
            "    permissions: '0755'",
            "    content: |",
            textwrap.indent(bootstrap_script, "      "),
        ]
    )

    lines.extend(
        [
            "runcmd:",
            "  - \"su - ubuntu -c '/home/ubuntu/bootstrap_nanochat.sh'\"",
        ]
    )

    return "\n".join(lines) + "\n"


class LambdaClient:
    """Minimal wrapper around the Lambda Cloud REST API."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("Lambda Cloud API key not provided")

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

    def launch_instances(self, payload: Dict[str, object]) -> List[str]:
        response = self.session.post(
            f"{API_BASE}/instance-operations/launch",
            data=json.dumps(payload),
            timeout=60,
        )
        if response.status_code >= 400:
            raise requests.HTTPError(response.text, response=response)
        data = response.json()
        instance_ids = data["data"]["instance_ids"]
        return instance_ids

    def get_instance(self, instance_id: str) -> Optional[Dict[str, object]]:
        response = self.session.get(
            f"{API_BASE}/instances/{instance_id}",
            timeout=30,
        )
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            raise requests.HTTPError(response.text, response=response)
        return response.json()["data"]

    def wait_for_instance(
        self,
        instance_id: str,
        poll_seconds: int,
        max_wait_minutes: int,
    ) -> Dict[str, object]:
        deadline = time.time() + max_wait_minutes * 60
        last_status = "unknown"
        while time.time() < deadline:
            instance = self.get_instance(instance_id)
            if not instance:
                time.sleep(poll_seconds)
                continue

            status = str(instance.get("status", "unknown"))
            if status != last_status:
                log(f"Instance {instance_id} status: {status}")
                last_status = status

            if status == "active":
                return instance
            if status in {"terminated", "terminating", "preempted"}:
                raise RuntimeError(f"Instance {instance_id} entered terminal status '{status}'")
            if status == "unhealthy":
                warn(f"Instance {instance_id} reported unhealthy; continuing to poll")

            time.sleep(poll_seconds)

        raise TimeoutError(f"Timed out waiting for instance {instance_id} to become active")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a Lambda Labs instance and prepare the nanochat training run."
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LAMBDA_API_KEY"),
        help="Lambda Cloud API key (default: read from LAMBDA_API_KEY).",
    )
    parser.add_argument("--region", default="us-west-1", help="Lambda Cloud region name.")
    parser.add_argument(
        "--instance-type",
        default="gpu_1x_a10",
        help="Instance type to launch (see Lambda Cloud docs for the catalog).",
    )
    parser.add_argument(
        "--ssh-key-name",
        required=True,
        help="Name of the SSH key already registered with Lambda Cloud.",
    )
    parser.add_argument(
        "--quantity",
        type=int,
        default=1,
        help="Number of instances to launch (auto-start only supports 1).",
    )
    parser.add_argument("--name", help="Friendly name to assign to the instance(s).")
    parser.add_argument(
        "--repo-url",
        help="Git URL for the nanochat repository (default: auto-detect from current repo).",
    )
    parser.add_argument("--branch", default="main", help="Git branch to checkout on the instance.")
    parser.add_argument(
        "--repo-dir",
        default="nanochatAquaRat",
        help="Directory name to clone the repository into on the instance.",
    )
    parser.add_argument(
        "--run-script",
        default="run_aquarat_small.sh",
        help="Relative path to the training launch script inside the repo.",
    )
    parser.add_argument(
        "--tmux-session",
        default="nanochat-train",
        help="tmux session name used when --auto-start is active.",
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Kick off the training script automatically after provisioning completes.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Additional KEY=VALUE pairs to write into the remote .env file (repeatable).",
    )
    parser.add_argument(
        "--inject-env",
        action="append",
        default=[],
        help="Names of local environment variables whose values should populate the remote .env.",
    )
    parser.add_argument(
        "--env-file-name",
        default=".env.lambda",
        help="Filename (relative to /home/ubuntu) for the generated environment file.",
    )
    parser.add_argument(
        "--image-id",
        help="Optional image ID to use instead of the default Lambda Stack image.",
    )
    parser.add_argument(
        "--image-family",
        help="Optional image family name to use instead of the default image.",
    )
    parser.add_argument(
        "--max-wait-minutes",
        type=int,
        default=25,
        help="Maximum minutes to wait for the instance to become active.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=20,
        help="Polling interval while waiting for the instance to become active.",
    )
    parser.add_argument(
        "--skip-wait",
        action="store_true",
        help="Exit after requesting the launch without waiting for active status.",
    )
    parser.add_argument(
        "--no-user-data",
        action="store_true",
        help="Skip sending user-data; instance boots with the stock image configuration.",
    )
    parser.add_argument(
        "--print-user-data",
        action="store_true",
        help="Print the generated user-data to stdout before launching.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.image_id and args.image_family:
        error("Provide only one of --image-id or --image-family.")
        return 1

    if args.auto_start and args.quantity != 1:
        error("--auto-start currently supports a single instance (set --quantity=1).")
        return 1

    if not args.api_key:
        error("Lambda Cloud API key not provided. Use --api-key or set LAMBDA_API_KEY.")
        return 1

    repo_url = args.repo_url
    if not repo_url:
        repo_url = guess_repo_url()
        if repo_url:
            log(f"Discovered repository URL: {repo_url}")
        else:
            warn(
                "Could not auto-detect repository URL; "
                "pass --repo-url if you want the instance to clone the repo automatically."
            )

    try:
        env_pairs = collect_env_pairs(args.env, args.inject_env)
    except ValueError as exc:
        error(str(exc))
        return 1

    env_file_remote = f"/home/ubuntu/{args.env_file_name}"

    bootstrap_script = build_bootstrap_script(
        repo_dir=args.repo_dir,
        run_script=args.run_script,
        branch=args.branch,
        repo_url=repo_url,
        env_file_remote=env_file_remote,
        auto_start=args.auto_start,
        tmux_session=args.tmux_session,
    )

    user_data: Optional[str] = None
    if not args.no_user_data:
        user_data = build_user_data(
            packages=DEFAULT_PACKAGES,
            bootstrap_script=bootstrap_script,
            env_pairs=env_pairs,
            env_file_remote=env_file_remote,
        )
        if args.print_user_data:
            print(user_data)
    else:
        if args.print_user_data:
            print("# user-data disabled (--no-user-data)")

    payload: Dict[str, object] = {
        "region_name": args.region,
        "instance_type_name": args.instance_type,
        "ssh_key_names": [args.ssh_key_name],
    }
    if args.quantity:
        payload["quantity"] = args.quantity
    if args.name:
        payload["name"] = args.name
    if user_data:
        payload["user_data"] = user_data
    if args.image_id:
        payload["image"] = {"id": args.image_id}
    elif args.image_family:
        payload["image"] = {"family": args.image_family}

    client = LambdaClient(args.api_key)

    log(
        "Requesting instance launch "
        f"(region={args.region}, type={args.instance_type}, quantity={args.quantity})"
    )

    try:
        instance_ids = client.launch_instances(payload)
    except requests.HTTPError as exc:
        error(f"Instance launch failed: {exc}")
        if exc.response is not None:
            warn(f"Response content: {exc.response.text}")
        return 1

    log(f"Requested instance IDs: {', '.join(instance_ids)}")

    if args.skip_wait:
        log("Skipping wait (--skip-wait supplied).")
        return 0

    instances: List[Dict[str, object]] = []
    for instance_id in instance_ids:
        try:
            instance = client.wait_for_instance(
                instance_id=instance_id,
                poll_seconds=args.poll_seconds,
                max_wait_minutes=args.max_wait_minutes,
            )
        except (RuntimeError, TimeoutError, requests.HTTPError) as exc:
            error(f"Failed while waiting for instance {instance_id}: {exc}")
            return 1
        instances.append(instance)

    for instance in instances:
        ip = instance.get("ip") or "<pending>"
        name = instance.get("name") or instance.get("id")
        log(f"Instance {name} is active with public IP {ip}")
        if ip and ip != "<pending>":
            log(
                f"SSH command: ssh -i /path/to/key.pem ubuntu@{ip}"
            )

    if args.auto_start:
        log(
            "Auto-start enabled. Training is running inside tmux; attach with "
            f"`ssh ...` then `tmux attach -t {args.tmux_session}`."
        )
    else:
        log(
            "Auto-start disabled. After SSH'ing in, run "
            f"`cd ~/{args.repo_dir} && bash {args.run_script}`."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
