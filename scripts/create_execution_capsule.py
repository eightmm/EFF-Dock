#!/usr/bin/env python3
"""Create a read-only, per-run code capsule for queued Slurm jobs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import uuid
from pathlib import Path, PurePosixPath


def _safe_relative_path(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"capsule path must be a safe relative path: {value!r}")
    return Path(*pure.parts)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def create_capsule(
    *,
    repo_root: Path,
    output: Path,
    copy_files: list[str],
    link_roots: list[str],
    copy_overrides: dict[str, str] | None = None,
) -> dict[str, object]:
    """Atomically freeze selected files and link declared large roots."""
    repo_root = repo_root.resolve(strict=True)
    output = output.resolve(strict=False)
    capsule_root = repo_root / ".effdock_execution_capsules"
    if not _inside(output, capsule_root):
        raise ValueError(f"capsule output must be below {capsule_root}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)

    links = [_safe_relative_path(value) for value in link_roots]
    if len(set(links)) != len(links):
        raise ValueError("duplicate capsule link roots")
    for relative in links:
        if len(relative.parts) != 1:
            raise ValueError(f"link roots must be top-level paths: {relative}")
        source = repo_root / relative
        if not source.exists():
            raise FileNotFoundError(source)
        if _inside(output, source.resolve(strict=True)):
            raise ValueError(f"link root would contain the capsule output: {relative}")

    copies = [_safe_relative_path(value) for value in copy_files]
    if len(set(copies)) != len(copies):
        raise ValueError("duplicate capsule copy paths")
    for relative in copies:
        if any(relative == link or link in relative.parents for link in links):
            raise ValueError(f"copy path is already provided by a link root: {relative}")
        source = repo_root / relative
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"capsule source must be a regular file: {source}")
        if not _inside(source.resolve(strict=True), repo_root):
            raise ValueError(f"capsule source escapes repository: {source}")

    overrides: dict[Path, Path] = {}
    for destination_value, source_value in (copy_overrides or {}).items():
        destination = _safe_relative_path(destination_value)
        source_relative = _safe_relative_path(source_value)
        if destination in overrides or destination in copies:
            raise ValueError(f"duplicate capsule destination: {destination}")
        if any(destination == link or link in destination.parents for link in links):
            raise ValueError(f"copy destination is already provided by a link root: {destination}")
        source = repo_root / source_relative
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"capsule override source must be a regular file: {source}")
        if not _inside(source.resolve(strict=True), repo_root):
            raise ValueError(f"capsule override source escapes repository: {source}")
        overrides[destination] = source_relative

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.incomplete-{uuid.uuid4().hex}"
    temporary.mkdir(mode=0o700)
    try:
        for relative in copies:
            source = repo_root / relative
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination, follow_symlinks=False)
        for destination_relative, source_relative in overrides.items():
            source = repo_root / source_relative
            destination = temporary / destination_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination, follow_symlinks=False)
        link_records: dict[str, str] = {}
        for relative in links:
            target = (repo_root / relative).resolve(strict=True)
            os.symlink(target, temporary / relative, target_is_directory=target.is_dir())
            link_records[relative.as_posix()] = str(target)

        identity = {
            "schema_version": "effdock.execution_capsule.v1",
            "status": "frozen",
            "source_repo_root": str(repo_root),
            "copy_files": [path.as_posix() for path in copies],
            "copy_overrides": {
                destination.as_posix(): source.as_posix()
                for destination, source in sorted(
                    overrides.items(), key=lambda item: item[0].as_posix()
                )
            },
            "linked_roots": link_records,
        }
        identity_path = temporary / "execution_capsule.json"
        identity_path.write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        for path in sorted(temporary.rglob("*"), reverse=True):
            if path.is_symlink():
                continue
            mode = 0o555 if path.is_dir() else 0o444
            path.chmod(mode)
        temporary.chmod(0o555)
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            for path in temporary.rglob("*"):
                if not path.is_symlink():
                    path.chmod(path.stat().st_mode | stat.S_IWUSR)
            temporary.chmod(0o700)
            shutil.rmtree(temporary)
        raise
    return identity


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copy-file", action="append", default=[])
    parser.add_argument(
        "--copy-file-as",
        action="append",
        default=[],
        metavar="DESTINATION=SOURCE",
        help="copy SOURCE from the repository to a different DESTINATION in the capsule",
    )
    parser.add_argument("--link-root", action="append", default=[])
    args = parser.parse_args(argv)
    copy_overrides: dict[str, str] = {}
    for value in args.copy_file_as:
        destination, separator, source = value.partition("=")
        if not separator or not destination or not source:
            parser.error("--copy-file-as must use DESTINATION=SOURCE")
        if destination in copy_overrides:
            parser.error(f"duplicate --copy-file-as destination: {destination}")
        copy_overrides[destination] = source
    result = create_capsule(
        repo_root=args.repo_root,
        output=args.output,
        copy_files=args.copy_file,
        link_roots=args.link_root,
        copy_overrides=copy_overrides,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
