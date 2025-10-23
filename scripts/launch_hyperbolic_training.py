#!/usr/bin/env python3
"""
Automation helper to launch a Hyperbolic Labs marketplace instance and kick off the
nanochat AQuA-RAT training run.

The workflow mirrors `launch_lambda_training.py`, but uses Hyperbolic's REST API.

Example:

    python scripts/launch_hyperbolic_training.py \\
        --gpu-count 1 \\
        --region us-east \\
        --max-price 4.5 \\
        --auto-start \\
        --inject-env WANDB_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

API_BASE = "https://api.hyperbolic.xyz"
MARKETPLACE_BASE = f"{API_BASE}/v1/marketplace"
READY_STATUSES = {
    "ready",
    "running",
    "instance_running",
    "node_ready",
    "active",
    "online",
}


def log(msg: str) -> None:
    print(f"[info] {msg}")


def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)


def error(msg: str) -> None:
    print(f"[error] {msg}", file=sys.stderr)


def shell_quote(value: str) -> str:
    return shlex.quote(value)


def collect_env_pairs(cli_pairs: Sequence[str], inject_names: Sequence[str]) -> List[Tuple[str, str]]:
    """Merge KEY=VALUE pairs with env vars pulled from the local environment."""
    merged: Dict[str, str] = {}

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


class HyperbolicClient:
    def __init__(self, api_key: Optional[str]):
        if not api_key:
            raise ValueError("Hyperbolic API key is required. Pass --api-key or set HYPERBOLIC_API_KEY.")
        self.api_key = api_key

    def _headers(self, with_auth: bool = True) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if with_auth and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def list_marketplace(self) -> List[Dict[str, Any]]:
        response = requests.post(
            MARKETPLACE_BASE,
            headers=self._headers(with_auth=False),
            json={"filters": {}},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        instances = payload.get("instances")
        if instances is None:
            if isinstance(payload, list):
                instances = payload
            elif isinstance(payload, dict):
                instances = payload.get("nodes") or payload.get("data") or []
            else:
                instances = []
        return instances

    def create_instance(
        self,
        cluster_name: str,
        node_name: str,
        gpu_count: int,
        image: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "cluster_name": cluster_name,
            "node_name": node_name,
            "gpu_count": gpu_count,
        }
        if image:
            payload["image"] = image
        response = requests.post(
            f"{MARKETPLACE_BASE}/instances/create",
            headers=self._headers(),
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def list_instances(self) -> List[Dict[str, Any]]:
        response = requests.get(
            f"{MARKETPLACE_BASE}/instances",
            headers=self._headers(),
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            return payload.get("instances") or payload.get("data") or []
        if isinstance(payload, list):
            return payload
        return []

    def terminate_instance(self, instance_id: str) -> Dict[str, Any]:
        response = requests.post(
            f"{MARKETPLACE_BASE}/instances/terminate",
            headers=self._headers(),
            json={"id": instance_id},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def get_balance(self) -> Optional[float]:
        try:
            response = requests.get(
                f"{API_BASE}/billing/get_current_balance",
                headers=self._headers(),
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                return float(data.get("balance") or data.get("amount") or data.get("credits"))
        except (requests.HTTPError, ValueError, TypeError):
            warn("Unable to fetch current account balance.")
        return None


def summarize_node(node: Dict[str, Any]) -> str:
    def _gpu_models() -> str:
        gpus = []
        hardware = node.get("hardware") or {}
        for gpu in hardware.get("gpus") or []:
            model = gpu.get("model")
            ram = gpu.get("ram") or gpu.get("memory") or gpu.get("vram")
            if model and ram:
                gpus.append(f"{model} ({ram} GB)")
            elif model:
                gpus.append(model)
        return ", ".join(gpus) if gpus else "Unknown GPUs"

    price_info = (node.get("pricing") or {}).get("price") or {}
    price = price_info.get("amount")
    price_str = f"${price:.2f}/hr" if isinstance(price, (int, float)) else "n/a"
    region = ((node.get("location") or {}).get("region")) or "unknown region"
    cluster = node.get("cluster_name") or "unknown cluster"
    available = (node.get("gpus_total") or 0) - (node.get("gpus_reserved") or 0)
    supplier = node.get("supplier_id") or "unknown supplier"
    return (
        f"{node.get('id', '<unknown>')} | {cluster} | {region} | "
        f"{available}/{node.get('gpus_total', '?')} GPUs free | "
        f"{_gpu_models()} | {price_str} | supplier: {supplier}"
    )


def filter_nodes(
    nodes: Iterable[Dict[str, Any]],
    gpu_count: int,
    region: Optional[str],
    supplier: Optional[str],
    max_price: Optional[float],
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    region = region.lower() if region else None
    supplier = supplier.lower() if supplier else None

    for node in nodes:
        total = node.get("gpus_total") or 0
        reserved = node.get("gpus_reserved") or 0
        available = total - reserved
        if available < gpu_count:
            continue

        if region:
            region_value = ((node.get("location") or {}).get("region") or "").lower()
            if region not in region_value:
                continue

        if supplier:
            supplier_value = (node.get("supplier_id") or "").lower()
            if supplier not in supplier_value:
                continue

        price_info = (node.get("pricing") or {}).get("price") or {}
        price = price_info.get("amount")
        if max_price is not None and isinstance(price, (int, float)) and price > max_price:
            continue

        filtered.append(node)

    filtered.sort(
        key=lambda n: ((n.get("pricing") or {}).get("price") or {}).get("amount", float("inf"))
    )
    return filtered


def extract_instance_id(payload: Dict[str, Any], before_ids: Sequence[str], client: HyperbolicClient) -> str:
    candidates: List[str] = []
    for key in ("id", "instance_id", "instanceId"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            candidates.append(value)
    instance_obj = payload.get("instance") or payload.get("data")
    if isinstance(instance_obj, dict):
        for key in ("id", "instance_id", "instanceId"):
            value = instance_obj.get(key)
            if isinstance(value, str) and value:
                candidates.append(value)
    if candidates:
        return candidates[0]

    # Fall back to diffing current instances.
    time.sleep(3)
    current = client.list_instances()
    current_ids = {str(inst.get("id")) for inst in current if inst.get("id")}
    diff = current_ids.difference(before_ids)
    if diff:
        return diff.pop()
    raise RuntimeError("Unable to determine instance ID from API response.")


def extract_ip(instance: Dict[str, Any]) -> Optional[str]:
    network = instance.get("network") or {}
    candidates = [
        instance.get("public_ip"),
        instance.get("ip_address"),
        instance.get("ip"),
        instance.get("ipv4"),
        network.get("public_ip"),
        network.get("ip"),
        network.get("ipv4"),
    ]

    for item in instance.get("ip_addresses") or []:
        candidates.extend(
            [
                item.get("public_ip"),
                item.get("ip"),
                item.get("ipv4"),
                item.get("address"),
            ]
        )

    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def extract_ssh_port(instance: Dict[str, Any]) -> int:
    network = instance.get("network") or {}
    candidates = [
        instance.get("ssh_port"),
        network.get("ssh_port"),
        (instance.get("ssh") or {}).get("port"),
    ]

    for candidate in candidates:
        if isinstance(candidate, int):
            return candidate
        if isinstance(candidate, str) and candidate.isdigit():
            return int(candidate)
    return 22


def extract_status(instance: Dict[str, Any]) -> str:
    for key in ("status", "instance_status", "state"):
        value = instance.get(key)
        if isinstance(value, str):
            return value
    return ""


def wait_for_instance(
    client: HyperbolicClient,
    instance_id: str,
    poll_seconds: int,
    max_wait_minutes: int,
) -> Dict[str, Any]:
    log(f"Waiting for instance {instance_id} to become ready...")
    deadline = time.time() + max_wait_minutes * 60
    while time.time() < deadline:
        instances = client.list_instances()
        for instance in instances:
            identifiers = {
                str(instance.get("id")),
                str(instance.get("instance_id")),
                str(instance.get("instanceId")),
            }
            if instance_id not in identifiers:
                continue

            status = extract_status(instance).lower()
            ip = extract_ip(instance)
            if status in READY_STATUSES and ip:
                log(f"Instance is ready: status={status}, ip={ip}")
                return instance

            log(f"  status={status or '<unknown>'}; waiting for ready state...")
        time.sleep(poll_seconds)

    raise TimeoutError(f"Timed out waiting for instance {instance_id} to become ready.")


def build_env_content(pairs: Sequence[Tuple[str, str]]) -> str:
    return "\n".join(f"{key}={value}" for key, value in pairs) + ("\n" if pairs else "")


def scp(
    local_path: Path,
    remote_path: str,
    ssh_user: str,
    host: str,
    port: int,
    ssh_key: Optional[str],
) -> None:
    cmd = ["scp", "-o", "StrictHostKeyChecking=no"]
    if ssh_key:
        cmd.extend(["-i", ssh_key])
    if port != 22:
        cmd.extend(["-P", str(port)])
    cmd.extend([str(local_path), f"{ssh_user}@{host}:{remote_path}"])
    subprocess.run(cmd, check=True)


def ssh_command(
    ssh_user: str,
    host: str,
    port: int,
    ssh_key: Optional[str],
    *command: str,
) -> None:
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no"]
    if ssh_key:
        cmd.extend(["-i", ssh_key])
    if port != 22:
        cmd.extend(["-p", str(port)])
    cmd.append(f"{ssh_user}@{host}")
    if command:
        cmd.append(" ".join(command))
    subprocess.run(cmd, check=True)


def deploy_to_instance(
    instance: Dict[str, Any],
    bootstrap_script: str,
    env_pairs: Sequence[Tuple[str, str]],
    env_file_remote: str,
    ssh_user: str,
    ssh_key: Optional[str],
) -> None:
    ip = extract_ip(instance)
    if not ip:
        raise RuntimeError("Instance does not report a public IP address yet.")
    port = extract_ssh_port(instance)
    log(f"Deploying bootstrap assets to {ssh_user}@{ip}:{port} ...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        bootstrap_path = tmp_path / "bootstrap.sh"
        bootstrap_path.write_text(bootstrap_script)
        scp(bootstrap_path, "/tmp/nanochat_bootstrap.sh", ssh_user, ip, port, ssh_key)

        if env_pairs:
            env_path = tmp_path / "nanochat.env"
            env_path.write_text(build_env_content(env_pairs))
            ssh_command(ssh_user, ip, port, ssh_key, f"mkdir -p {shell_quote(str(Path(env_file_remote).parent))}")
            scp(env_path, env_file_remote, ssh_user, ip, port, ssh_key)

        log("Executing remote bootstrap script...")
        ssh_command(ssh_user, ip, port, ssh_key, "bash /tmp/nanochat_bootstrap.sh")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Hyperbolic Labs instance for AQuA-RAT training")
    parser.add_argument("--api-key", default=os.environ.get("HYPERBOLIC_API_KEY"), help="Hyperbolic API key")
    parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs to request (default: 1)")
    parser.add_argument("--region", help="Preferred region substring (case-insensitive)")
    parser.add_argument("--supplier", help="Preferred supplier substring (case-insensitive)")
    parser.add_argument("--max-price", type=float, help="Maximum hourly price in USD")
    parser.add_argument("--node-name", help="Specify node name explicitly")
    parser.add_argument("--cluster-name", help="Cluster name when using --node-name")
    parser.add_argument("--list", action="store_true", help="List available marketplace nodes and exit")

    parser.add_argument("--repo-url", help="Repository URL to clone (defaults to git remote origin)")
    parser.add_argument("--branch", default="main", help="Branch to checkout on the instance (default: main)")
    parser.add_argument("--run-script", default="run_aquarat_small.sh", help="Script to execute on the instance")
    parser.add_argument("--repo-dir", default="nanochatAquaRat", help="Directory name for the repo on the instance")

    parser.add_argument("--auto-start", action="store_true", help="Automatically run the training script")
    parser.add_argument("--tmux-session", default="training", help="tmux session name when auto-start is enabled")
    parser.add_argument("--ssh-user", default="ubuntu", help="SSH username for the instance (default: ubuntu)")
    parser.add_argument("--ssh-key", help="Path to SSH private key for scp/ssh")
    parser.add_argument("--no-deploy", action="store_true", help="Skip deployment after the instance is ready")

    parser.add_argument("--env", action="append", default=[], help="Environment variable in KEY=VALUE form")
    parser.add_argument("--inject-env", action="append", default=[], help="Environment variable name to copy from local env")

    parser.add_argument("--poll-seconds", type=int, default=20, help="Polling interval while waiting (default: 20)")
    parser.add_argument("--max-wait-minutes", type=int, default=25, help="Maximum minutes to wait for ready state")

    return parser.parse_args()


def guess_repo_url() -> Optional[str]:
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


def main() -> int:
    args = parse_args()

    try:
        env_pairs = collect_env_pairs(args.env, args.inject_env)
    except ValueError as exc:
        error(str(exc))
        return 1

    if not args.api_key:
        error("Hyperbolic API key not provided. Use --api-key or set HYPERBOLIC_API_KEY.")
        return 1

    client = HyperbolicClient(args.api_key)

    try:
        nodes = client.list_marketplace()
    except requests.HTTPError as exc:
        error(f"Failed to list marketplace nodes: {exc}")
        return 1

    if args.list:
        log("Available marketplace nodes:")
        for node in nodes:
            print(summarize_node(node))
        return 0

    selected_node: Optional[Dict[str, Any]] = None

    if args.node_name:
        for node in nodes:
            if node.get("id") == args.node_name:
                selected_node = node
                break
        if not selected_node:
            error(f"Node '{args.node_name}' not found in marketplace list.")
            return 1
        if not args.cluster_name:
            args.cluster_name = selected_node.get("cluster_name")
    else:
        filtered_nodes = filter_nodes(nodes, args.gpu_count, args.region, args.supplier, args.max_price)
        if not filtered_nodes:
            error("No marketplace nodes match the specified constraints.")
            return 1
        selected_node = filtered_nodes[0]
        log("Selected node:")
        print("  " + summarize_node(selected_node))
        args.cluster_name = selected_node.get("cluster_name")
        args.node_name = selected_node.get("id")

    if not args.cluster_name:
        error("Cluster name is required; unable to determine cluster for the selected node.")
        return 1

    repo_url = args.repo_url or guess_repo_url()
    if repo_url:
        log(f"Using repository: {repo_url}")
    else:
        warn("Could not determine repository URL. Auto-start will clone existing repo on instance if present.")

    balance = client.get_balance()
    if balance is not None:
        log(f"Current Hyperbolic balance: ${balance:.2f}")

    before_instances = client.list_instances()
    before_ids = {str(inst.get("id")) for inst in before_instances if inst.get("id")}
    log(f"Launching instance on cluster '{args.cluster_name}' node '{args.node_name}' "
        f"with {args.gpu_count} GPU(s)...")

    try:
        create_response = client.create_instance(
            cluster_name=args.cluster_name,
            node_name=args.node_name,
            gpu_count=args.gpu_count,
        )
    except requests.HTTPError as exc:
        error(f"Failed to launch instance: {exc}")
        try:
            warn(f"Response payload: {exc.response.text}")  # type: ignore[attr-defined]
        except Exception:
            pass
        return 1

    instance_id = extract_instance_id(create_response, before_ids, client)
    log(f"Instance request acknowledged with id={instance_id}")

    try:
        instance = wait_for_instance(
            client=client,
            instance_id=instance_id,
            poll_seconds=args.poll_seconds,
            max_wait_minutes=args.max_wait_minutes,
        )
    except TimeoutError as exc:
        error(str(exc))
        return 1

    ip = extract_ip(instance)
    port = extract_ssh_port(instance)
    ssh_user = args.ssh_user

    log("Instance ready. Connection details:")
    if ip:
        ssh_parts = ["ssh", "-o", "StrictHostKeyChecking=no"]
        if args.ssh_key:
            ssh_parts.extend(["-i", args.ssh_key])
        if port != 22:
            ssh_parts.extend(["-p", str(port)])
        ssh_parts.append(f"{ssh_user}@{ip}")
        print("  SSH:", " ".join(ssh_parts))
    else:
        warn("Instance IP not available; SSH command cannot be constructed.")

    if args.no_deploy:
        log("Skipping deployment (--no-deploy supplied).")
        return 0

    env_file_remote = f"/home/{ssh_user}/nanochat_aquarat.env"
    bootstrap_script = build_bootstrap_script(
        repo_dir=args.repo_dir,
        run_script=args.run_script,
        branch=args.branch,
        repo_url=repo_url,
        env_file_remote=env_file_remote,
        auto_start=args.auto_start,
        tmux_session=args.tmux_session,
    )

    try:
        deploy_to_instance(
            instance=instance,
            bootstrap_script=bootstrap_script,
            env_pairs=env_pairs,
            env_file_remote=env_file_remote,
            ssh_user=ssh_user,
            ssh_key=args.ssh_key,
        )
    except subprocess.CalledProcessError as exc:
        error(f"Deployment failed: {exc}")
        warn("You can manually SSH to the instance and run the training script.")
        return 1

    log("Deployment complete.")
    if args.auto_start:
        log("Training should now be running on the instance.")
    else:
        log("Auto-start disabled; after SSHing in, run the configured script manually.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

