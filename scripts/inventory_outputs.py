#!/usr/bin/env python3
"""Create a non-destructive retention inventory for ignored experiment outputs."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

TEXT_SUFFIXES = {".md", ".py", ".sh", ".sbatch", ".toml", ".yaml", ".yml", ".json", ".jsonl", ".tex"}
TRANSIENT_TOKENS = (
    "dry",
    "dryrun",
    "smoke",
    "tmp",
    "debug",
    "wrappertest",
    "foreground",
    "repro_",
)
RESULT_MARKERS = {"RESULTS.md", "aggregate.json", "summary.json", "report.json"}
DEFAULT_RETAIN = {
    "archive",
    "benchmarks",
    "eff-dock",
    "figures",
    "guidance",
    "physical",
    "presentation",
    "presentations",
}
DEFAULT_BENCHMARK_RETAIN = {
    "confidence",
    "confidence_posebusters_official",
    "guidance_budget1000_full_v2",
    "guidance_eta_cap_extension_runs",
    "guidance_sigma_sweep_eta2_runs",
    "guidance_steric_high_eta_confidence_runs",
    "logs",
}


def source_corpus(repo_root: Path) -> str:
    roots = [repo_root / name for name in ("docs", "scripts", "src", "tests")]
    roots.extend(repo_root / name for name in ("README.md", "PROJECT.md", "pyproject.toml"))
    chunks: list[str] = []
    for root in roots:
        paths = root.rglob("*") if root.is_dir() else (root,)
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if path.name == "OUTPUT_RETENTION_PLAN.md":
                continue
            try:
                chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                continue
    return "\n".join(chunks)


def collect_stats(outputs: Path) -> dict[Path, dict[str, object]]:
    """Walk outputs once and aggregate both requested directory levels."""
    mutable: dict[Path, dict[str, object]] = {}

    def targets(file_path: Path) -> tuple[Path, ...]:
        relative = file_path.relative_to(outputs)
        parts = relative.parts
        selected = [outputs / parts[0]] if parts else []
        if len(parts) >= 3 and parts[0] == "benchmarks":
            selected.append(outputs / "benchmarks" / parts[1])
        return tuple(selected)

    for root, _, filenames in os.walk(outputs):
        root_path = Path(root)
        for filename in filenames:
            file_path = root_path / filename
            try:
                stat = file_path.stat()
            except OSError:
                continue
            for target in targets(file_path):
                entry = mutable.setdefault(
                    target,
                    {
                        "bytes": 0,
                        "files": 0,
                        "newest": 0.0,
                        "result_markers": set(),
                        "failed_markers": 0,
                    },
                )
                entry["bytes"] = int(entry["bytes"]) + stat.st_size
                entry["files"] = int(entry["files"]) + 1
                entry["newest"] = max(float(entry["newest"]), stat.st_mtime)
                if filename in RESULT_MARKERS:
                    entry["result_markers"].add(filename)
                if filename.endswith(".failed") or filename == ".submission.failed":
                    entry["failed_markers"] = int(entry["failed_markers"]) + 1
    result: dict[Path, dict[str, object]] = {}
    for path, entry in mutable.items():
        newest = float(entry.pop("newest"))
        result[path] = {
            **entry,
            "newest_mtime_utc": (
                datetime.fromtimestamp(newest, timezone.utc).isoformat() if newest else None
            ),
            "result_markers": sorted(entry["result_markers"]),
        }
    return result


def classify(
    *,
    name: str,
    stats: dict[str, object],
    reference_count: int,
    explicitly_retained: bool,
    active: bool,
) -> tuple[str, str]:
    if active:
        return "retain_active", "currently active Slurm output root"
    if explicitly_retained:
        return "retain", "named canonical/user-facing output root"
    if reference_count:
        return "retain_referenced", "referenced by repository code or documentation"
    normalized = name.lower()
    if any(token in normalized for token in TRANSIENT_TOKENS):
        if int(stats["bytes"]) > 16 * 1024**2:
            return (
                "review_large_transient",
                "transient-name pattern but too large for automatic archival",
            )
        return "cleanup_candidate", "unreferenced transient-name pattern; inspect before deletion"
    if stats["result_markers"]:
        return "archive_candidate", "unreferenced but contains result artifacts"
    if int(stats["failed_markers"]):
        return "cleanup_candidate", "unreferenced failed-run markers"
    return "review", "no automatic retention decision"


def inventory_level(
    root: Path,
    *,
    repo_root: Path,
    corpus: str,
    retain_names: set[str],
    active_paths: set[Path],
    stats_by_path: dict[Path, dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((item for item in root.iterdir() if item.is_dir()), key=lambda p: p.name):
        relative = path.relative_to(repo_root)
        stats = stats_by_path.get(
            path,
            {
                "bytes": 0,
                "files": 0,
                "newest_mtime_utc": None,
                "result_markers": [],
                "failed_markers": 0,
            },
        )
        reference_count = corpus.count(str(relative))
        active = any(path.resolve() == item or path.resolve() in item.parents for item in active_paths)
        category, reason = classify(
            name=path.name,
            stats=stats,
            reference_count=reference_count,
            explicitly_retained=path.name in retain_names,
            active=active,
        )
        rows.append(
            {
                "path": str(relative),
                **stats,
                "repository_reference_count": reference_count,
                "classification": category,
                "reason": reason,
            }
        )
    return rows


def gib(value: int) -> str:
    return f"{value / 1024**3:.2f} GiB"


def render_markdown(result: dict[str, object]) -> str:
    lines = [
        "# Output retention inventory",
        "",
        f"Generated: {result['created_utc']}",
        "",
        "> Generating this inventory is non-destructive. Any previously applied "
        "recoverable archive move is listed below; no files were deleted. "
        "New cleanup candidates require a separate reference check and explicit approval.",
        "",
    ]
    archive_manifests = result.get("applied_archive_manifests", [])
    if archive_manifests:
        lines.extend(
            [
                "Applied recoverable archive operations:",
                "",
                *[f"- `{path}`" for path in archive_manifests],
                "",
            ]
        )
    for level_name in ("outputs", "outputs/benchmarks"):
        rows = result["levels"][level_name]
        counts = Counter(row["classification"] for row in rows)
        total = sum(int(row["bytes"]) for row in rows)
        lines.extend(
            [
                f"## {level_name}",
                "",
                f"Total indexed size: {gib(total)} across {len(rows)} directories.",
                "",
                "Classification counts: "
                + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())),
                "",
                "| class | size | files | references | path | reason |",
                "|---|---:|---:|---:|---|---|",
            ]
        )
        priority = sorted(
            rows,
            key=lambda row: (
                row["classification"] not in {"cleanup_candidate", "archive_candidate", "retain_active"},
                -int(row["bytes"]),
            ),
        )
        for row in priority[:40]:
            lines.append(
                f"| {row['classification']} | {gib(int(row['bytes']))} | {row['files']} | "
                f"{row['repository_reference_count']} | `{row['path']}` | {row['reason']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Retention policy",
            "",
            "1. Keep active runs, model weights, frozen input manifests, selected SDFs, exact cohort audits, official PB shards, and final aggregates.",
            "2. Archive completed but superseded runs as intact directories so hashes and relative paths remain meaningful.",
            "3. Delete only unreferenced smoke/dry/debug/failed artifacts after confirming they are not a parent of a retained report.",
            "4. Never delete raw benchmark data or training data; data remain ignored by Git as defined by the project contract.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--active-path", action="append", type=Path, default=[])
    parser.add_argument("--archive-manifest", action="append", type=Path, default=[])
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    outputs = repo_root / "outputs"
    corpus = source_corpus(repo_root)
    active_paths = {
        (path if path.is_absolute() else repo_root / path).resolve()
        for path in args.active_path
    }
    stats_by_path = collect_stats(outputs)
    levels = {
        "outputs": inventory_level(
            outputs,
            repo_root=repo_root,
            corpus=corpus,
            retain_names=DEFAULT_RETAIN,
            active_paths=active_paths,
            stats_by_path=stats_by_path,
        ),
        "outputs/benchmarks": inventory_level(
            outputs / "benchmarks",
            repo_root=repo_root,
            corpus=corpus,
            retain_names=DEFAULT_BENCHMARK_RETAIN,
            active_paths=active_paths,
            stats_by_path=stats_by_path,
        ),
    }
    result = {
        "schema_version": "effdock.output_retention_inventory.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "destructive_actions_performed": False,
        "applied_archive_manifests": [str(path) for path in args.archive_manifest],
        "active_paths": sorted(str(path.relative_to(repo_root)) for path in active_paths),
        "levels": levels,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(result), encoding="utf-8")


if __name__ == "__main__":
    main()
