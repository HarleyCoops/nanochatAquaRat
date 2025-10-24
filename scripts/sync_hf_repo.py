"""
Sync documentation assets from the GitHub repo to the Hugging Face model card.

The script keeps a local staging directory (`hf_release/`) aligned with the
files referenced from the root `README.md`, then optionally pushes the same
payload to the Hugging Face Hub in a single commit.

Usage:
    uv run python -m scripts.sync_hf_repo --repo-id HarleyCooper/nanochatAquaRat

Environment:
    - Requires `huggingface_hub` to be installed (already declared in uv.lock).
    - Expects an auth token to be available via `HF_TOKEN` or prior `huggingface-cli login`.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Iterable, List, Set

from huggingface_hub import (
    CommitOperationAdd,
    HfApi,
    create_repo,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync docs to Hugging Face model repo.")
    parser.add_argument(
        "--repo-id",
        default="HarleyCooper/nanochatAquaRat",
        help="Target Hugging Face repo id (e.g. 'namespace/name'). Defaults to HarleyCooper/nanochatAquaRat.",
    )
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=("model", "dataset", "space"),
        help="Hugging Face repository type. Defaults to 'model'.",
    )
    parser.add_argument(
        "--readme",
        default="README.md",
        help="Path to the root README that acts as the source of truth.",
    )
    parser.add_argument(
        "--release-dir",
        default="hf_release",
        help="Local staging directory that mirrors the Hugging Face card contents.",
    )
    parser.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="Additional files or globs to include besides those referenced in the README.",
    )
    parser.add_argument(
        "--no-push",
        dest="push",
        action="store_false",
        help="Skip pushing to Hugging Face (updates local staging dir only).",
    )
    parser.add_argument(
        "--message",
        default="Sync docs from GitHub",
        help="Commit message used when pushing to Hugging Face.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the operations without copying or pushing files.",
    )
    return parser.parse_args()


def _extract_relative_refs(readme_text: str) -> Set[str]:
    """
    Return all repo-relative assets referenced from markdown links/images.

    Matches `[]()` and `![]()` targets that do not start with http(s) or '#'.
    """
    pattern = re.compile(r"\]\((?!https?://)(?!\#)([^)]+)\)")

    refs: Set[str] = set()
    for match in pattern.finditer(readme_text):
        target = match.group(1).strip()
        if not target:
            continue
        # Strip query strings or anchors if present.
        target = target.split("#", 1)[0].split("?", 1)[0].strip()
        if not target:
            continue
        # Ignore mailto links.
        if target.lower().startswith("mailto:") or target.lower().startswith("javascript:"):
            continue
        if target.startswith("./"):
            target = target[2:]
        refs.add(target)
    return refs


def _expand_globs(root: Path, patterns: Iterable[str]) -> Set[str]:
    expanded: Set[str] = set()
    for pattern in patterns:
        if not pattern:
            continue
        if "*" in pattern or "?" in pattern or "[" in pattern:
            matches = list(root.glob(pattern))
            if not matches:
                LOGGER.warning("Pattern %s did not match any files", pattern)
            for match in matches:
                if match.is_file():
                    expanded.add(match.relative_to(root).as_posix())
                else:
                    for file_path in match.rglob("*"):
                        if file_path.is_file():
                            expanded.add(file_path.relative_to(root).as_posix())
        else:
            expanded.add(pattern)
    return expanded


def _copy_to_release(root: Path, release_dir: Path, files: Iterable[str], dry_run: bool = False) -> None:
    if dry_run:
        for rel_path in sorted(files):
            LOGGER.info("[DRY-RUN] Would copy %s -> %s", rel_path, release_dir / rel_path)
        return

    for rel_path in sorted(files):
        src = root / rel_path
        dst = release_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        LOGGER.info("Copied %s -> %s", src, dst)

    # Remove stale files.
    valid_rel_paths = {Path(rel_path) for rel_path in files}
    for file_path in list(release_dir.rglob("*")):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(release_dir)
        if rel not in valid_rel_paths:
            LOGGER.info("Removing stale file %s", file_path)
            file_path.unlink()
    # Clean empty directories.
    for dir_path in sorted(
        (p for p in release_dir.rglob("*") if p.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        try:
            next(dir_path.iterdir())
        except StopIteration:
            dir_path.rmdir()
            LOGGER.debug("Removed empty directory %s", dir_path)


def _ensure_files_exist(root: Path, rel_paths: Iterable[str]) -> List[str]:
    missing: List[str] = []
    for rel in rel_paths:
        if not (root / rel).is_file():
            missing.append(rel)
    return missing


def _push_to_hf(
    release_dir: Path,
    repo_id: str,
    repo_type: str,
    files: Iterable[str],
    message: str,
    dry_run: bool = False,
) -> None:
    if dry_run:
        for rel_path in sorted(files):
            LOGGER.info("[DRY-RUN] Would upload %s to %s", rel_path, repo_id)
        return

    api = HfApi()
    create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)

    operations = []
    for rel_path in sorted(files):
        local_path = release_dir / rel_path
        operations.append(
            CommitOperationAdd(
                path_in_repo=rel_path.replace(os.sep, "/"),
                path_or_fileobj=local_path,
            )
        )

    if not operations:
        LOGGER.warning("Nothing to push to %s", repo_id)
        return

    LOGGER.info("Pushing %d file(s) to %s", len(operations), repo_id)
    api.create_commit(
        repo_id=repo_id,
        repo_type=repo_type,
        operations=operations,
        commit_message=message,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    root = Path(__file__).resolve().parents[1]
    readme_path = (root / args.readme).resolve()
    release_dir = (root / args.release_dir).resolve()

    if not readme_path.is_file():
        raise FileNotFoundError(f"README not found at {readme_path}")

    readme_text = readme_path.read_text(encoding="utf-8")
    referenced_assets = _extract_relative_refs(readme_text)

    files_to_sync: Set[str] = {args.readme.replace("\\", "/")}
    files_to_sync.update(referenced_assets)
    files_to_sync.update(_expand_globs(root, args.extra))

    missing = _ensure_files_exist(root, files_to_sync)
    if missing:
        missing_list = ", ".join(sorted(missing))
        LOGGER.warning("Skipping missing file(s): %s", missing_list)
        files_to_sync.difference_update(missing)

    if not files_to_sync:
        LOGGER.info("No files to sync. Exiting.")
        return

    LOGGER.info("Preparing release directory at %s", release_dir)
    release_dir.mkdir(parents=True, exist_ok=True)
    _copy_to_release(root, release_dir, files_to_sync, dry_run=args.dry_run)

    if args.push:
        _push_to_hf(
            release_dir=release_dir,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            files=files_to_sync,
            message=args.message,
            dry_run=args.dry_run,
        )
    else:
        LOGGER.info("Push disabled; skipping Hugging Face upload step.")


if __name__ == "__main__":
    main()
