#!/usr/bin/env python3
"""Resume one artifact with concurrent HTTP range requests and atomic assembly."""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

MIN_PART_BYTES = 8 * 1024 * 1024


def split_ranges(start: int, end: int, workers: int) -> list[tuple[int, int]]:
    if start > end:
        return []
    remaining = end - start + 1
    count = min(workers, max(1, math.ceil(remaining / MIN_PART_BYTES)))
    base, extra = divmod(remaining, count)
    ranges: list[tuple[int, int]] = []
    cursor = start
    for index in range(count):
        length = base + (1 if index < extra else 0)
        ranges.append((cursor, cursor + length - 1))
        cursor += length
    return ranges


def download_range(url: str, part: Path, start: int, end: int) -> None:
    expected = end - start + 1
    actual = part.stat().st_size if part.is_file() else 0
    if actual == expected:
        print(f"range cached: {start}-{end}", flush=True)
        return
    if actual > expected:
        raise RuntimeError(
            f"cached range is larger than expected: {part} {actual} > {expected}"
        )

    failures_without_progress = 0
    incoming = part.with_name(f"{part.name}.incoming")
    while actual < expected:
        incoming.unlink(missing_ok=True)
        resume_start = start + actual
        result = subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--connect-timeout",
                "30",
                "--range",
                f"{resume_start}-{end}",
                "--output",
                str(incoming),
                url,
            ],
            check=False,
        )
        received = incoming.stat().st_size if incoming.is_file() else 0
        if received:
            with part.open("ab") as output, incoming.open("rb") as source:
                shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            incoming.unlink(missing_ok=True)
            actual = part.stat().st_size
            failures_without_progress = 0
            if actual > expected:
                raise RuntimeError(
                    f"server ignored requested range {resume_start}-{end}: "
                    f"cached {actual} bytes for expected range length {expected}"
                )
            if actual < expected:
                print(
                    f"range resume: {start}-{end} cached={actual}/{expected}",
                    flush=True,
                )
        else:
            failures_without_progress += 1

        if result.returncode != 0 or received == 0:
            if failures_without_progress >= 20:
                raise RuntimeError(
                    f"range {start}-{end} made no progress after "
                    f"{failures_without_progress} attempts"
                )
            if received == 0:
                time.sleep(2)

    incoming.unlink(missing_ok=True)
    print(f"range complete: {start}-{end}", flush=True)


def download(url: str, target: Path, expected_size: int, workers: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    prefix_size = target.stat().st_size if target.is_file() else 0
    if prefix_size == expected_size:
        print(f"download already complete: {target}", flush=True)
        return
    if prefix_size > expected_size:
        raise RuntimeError(
            f"existing target is larger than expected: {prefix_size} > {expected_size}"
        )

    ranges = split_ranges(prefix_size, expected_size - 1, workers)
    parts = [
        target.with_name(f".{target.name}.part-{start}-{end}") for start, end in ranges
    ]
    print(
        f"parallel resume: target={target} prefix={prefix_size} "
        f"remaining={expected_size - prefix_size} ranges={len(ranges)}",
        flush=True,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ranges)) as executor:
        futures = [
            executor.submit(download_range, url, part, start, end)
            for part, (start, end) in zip(parts, ranges, strict=True)
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    assembling = target.with_name(f".{target.name}.assembling")
    with assembling.open("wb") as output:
        if prefix_size:
            with target.open("rb") as prefix:
                shutil.copyfileobj(prefix, output, length=16 * 1024 * 1024)
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
        output.flush()
        os.fsync(output.fileno())

    actual_size = assembling.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"assembled size mismatch: expected {expected_size}, received {actual_size}"
        )
    os.replace(assembling, target)
    for part in parts:
        part.unlink(missing_ok=True)
    print(f"download complete: {target} ({actual_size} bytes)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("target", type=Path)
    parser.add_argument("expected_size", type=int)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.expected_size <= 0:
        parser.error("expected_size must be positive")
    if args.workers <= 0:
        parser.error("workers must be positive")
    download(args.url, args.target, args.expected_size, args.workers)


if __name__ == "__main__":
    main()
