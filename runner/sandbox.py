"""Sandbox policy enforcement: detect and invalidate tampered runs."""

from __future__ import annotations

import hashlib
from pathlib import Path

SKIP_DIRS = {"__pycache__", ".pytest_cache"}


def snapshot_directory(path: Path) -> dict[str, str]:
    """Hash every file under *path* recursively.

    Returns a dict mapping relative POSIX paths to their SHA-256 hex digests.
    Skips __pycache__ and .pytest_cache directories.
    """
    snapshot: dict[str, str] = {}
    if not path.exists():
        return snapshot
    for file in sorted(path.rglob("*")):
        if file.is_dir():
            continue
        # Skip files inside excluded directories
        if any(part in SKIP_DIRS for part in file.relative_to(path).parts):
            continue
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        rel = file.relative_to(path).as_posix()
        snapshot[rel] = digest
    return snapshot


def check_policy(
    pre: dict[str, str],
    post: dict[str, str],
    allowed_files: list[str],
) -> tuple[bool, list[str]]:
    """Compare pre/post snapshots and report policy violations.

    A violation is any file that was added, modified, or deleted and whose
    relative path is NOT in *allowed_files*.

    Returns (compliant, violations) where *violations* is a list of
    human-readable descriptions.
    """
    allowed = set(allowed_files)
    violations: list[str] = []

    all_paths = set(pre) | set(post)
    for path in sorted(all_paths):
        in_pre = path in pre
        in_post = path in post

        if in_pre and not in_post:
            if path not in allowed:
                violations.append(f"deleted: {path}")
        elif not in_pre and in_post:
            if path not in allowed:
                violations.append(f"added: {path}")
        elif pre[path] != post[path]:
            if path not in allowed:
                violations.append(f"modified: {path}")

    compliant = len(violations) == 0
    return compliant, violations
