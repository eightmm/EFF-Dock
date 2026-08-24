#!/usr/bin/env python3
"""Aggregate three symmetry-confidence checkpoints on the frozen refined external bank."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from effdock.workflows.evaluate import file_sha256
from effdock.workflows.posebusters_report import VALIDITY_CHECKS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_s50_symmetry_confidence_refined_external_shard import (
    EXPECTED_COUNTS,
    PROTOCOL_ID,
)

ARMS = {
    "u001500": "2af26bf66bec53676b8344e811911bbf47ee85aa6550610f35c3812b7a7f9d15",
    "u025000": "1c59034172fb925cc8a70777dcba236be349f1a1de1775d49cc17d492b17c030",
    "u050000": "fd49fa86f67187bf26d6c1bcf2daf925ba3e3b19dfeae733e57535d183280469",
}
DOCKING_SHA256 = "65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6"


def _truth(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid boolean {value!r}")


def _load_official(root: Path) -> dict[tuple[str, str, int], bool]:
    rows: dict[tuple[str, str, int], bool] = {}
    for shard in range(32):
        directory = root / f"shard-{shard:03d}-of-032"
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if summary.get("status") != "complete":
            raise ValueError(f"incomplete PoseBusters shard {shard}")
        with gzip.open(directory / "poses.csv.gz", "rt", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (row["dataset"], row["id"], int(row["pose_index"]))
                if key in rows:
                    raise ValueError(f"duplicate official row {key}")
                rows[key] = all(_truth(row[name]) for name in VALIDITY_CHECKS)
    if len(rows) != 39_300:
        raise ValueError(f"expected 39,300 official rows, got {len(rows)}")
    return rows


def _load_scores(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 100 or [int(row["pose_index"]) for row in rows] != list(range(100)):
        raise ValueError(f"invalid score inventory: {path}")
    return rows


def _stage(
    rows: list[dict[str, str]],
    official: list[bool] | None,
    *,
    stage: str,
) -> dict[str, Any]:
    prefix = "before" if stage == "step_000" else "after"
    rmsd_field = (
        "initial_symmetry_rmsd_angstrom"
        if stage == "step_000"
        else "final_symmetry_rmsd_angstrom"
    )
    scores = [float(row[f"{prefix}_confidence_rmsd"]) for row in rows]
    rmsds = [float(row[rmsd_field]) for row in rows]
    if not all(math.isfinite(value) for value in scores + rmsds):
        raise ValueError("non-finite score or RMSD")
    order = sorted(range(100), key=lambda index: (scores[index], index))
    selected = order[0]
    return {
        "selected_index": selected,
        "selected_rmsd": rmsds[selected],
        "selected_lt2": rmsds[selected] < 2.0,
        "selected_official_valid": None if official is None else official[selected],
        "selected_joint": None if official is None else official[selected] and rmsds[selected] < 2.0,
        "top5_lt2": any(rmsds[index] < 2.0 for index in order[:5]),
        "oracle_lt2": min(rmsds) < 2.0,
        "oracle_rmsd": min(rmsds),
        "k2": sum(value < 2.0 for value in rmsds),
    }


def _aggregate(rows: list[dict[str, Any]], *, stage: str) -> dict[str, Any]:
    result = {
        "complexes": len(rows),
        "selected_lt2_count": sum(bool(row[f"{stage}_selected_lt2"]) for row in rows),
        "selected_lt2_pct": 100.0 * sum(bool(row[f"{stage}_selected_lt2"]) for row in rows) / len(rows),
        "top5_lt2_pct": 100.0 * sum(bool(row[f"{stage}_top5_lt2"]) for row in rows) / len(rows),
        "oracle_lt2_pct": 100.0 * sum(bool(row[f"{stage}_oracle_lt2"]) for row in rows) / len(rows),
        "selected_mean_rmsd": statistics.fmean(float(row[f"{stage}_selected_rmsd"]) for row in rows),
        "selected_median_rmsd": statistics.median(float(row[f"{stage}_selected_rmsd"]) for row in rows),
        "mean_k2": statistics.fmean(int(row[f"{stage}_k2"]) for row in rows),
    }
    official = [row[f"{stage}_selected_official_valid"] for row in rows]
    joint = [row[f"{stage}_selected_joint"] for row in rows]
    if all(value is not None for value in official + joint):
        result["selected_official_valid_pct"] = 100.0 * sum(bool(value) for value in official) / len(rows)
        result["selected_joint_pct"] = 100.0 * sum(bool(value) for value in joint) / len(rows)
    return result


def _comparison(
    rows: list[dict[str, Any]], *, baseline: str, candidate: str, dataset: str
) -> dict[str, Any]:
    subset = [row for row in rows if row["dataset"] == dataset]
    result: dict[str, Any] = {
        "dataset": dataset,
        "baseline": baseline,
        "candidate": candidate,
        "complexes": len(subset),
    }
    for metric in ("selected_lt2", "selected_official_valid", "selected_joint"):
        left = [bool(row[f"{baseline}_step_100_{metric}"]) for row in subset]
        right = [bool(row[f"{candidate}_step_100_{metric}"]) for row in subset]
        result[metric] = {
            "delta_percentage_points": 100.0 * (sum(right) - sum(left)) / len(subset),
            "false_to_true": sum(not a and b for a, b in zip(left, right, strict=True)),
            "true_to_false": sum(a and not b for a, b in zip(left, right, strict=True)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    official_rows = _load_official(args.official_root)
    all_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for arm, expected_sha in ARMS.items():
        arm_root = args.scores_root / arm
        for shard in range(32):
            shard_path = arm_root / "shards" / f"full-shard-{shard:03d}-of-032.json"
            shard_summary = json.loads(shard_path.read_text(encoding="utf-8"))
            if (
                shard_summary.get("protocol_id") != PROTOCOL_ID
                or shard_summary.get("status") != "complete"
                or shard_summary.get("stage") != "full"
                or shard_summary.get("arm") != arm
                or shard_summary.get("checkpoint_sha256") != expected_sha
                or shard_summary.get("docking_checkpoint_sha256") != DOCKING_SHA256
                or int(shard_summary.get("num_shards", -1)) != 32
                or int(shard_summary.get("shard_index", -1)) != shard
            ):
                raise ValueError(f"invalid shard summary: {shard_path}")
        for dataset, count in EXPECTED_COUNTS.items():
            directories = sorted(path for path in (arm_root / dataset).iterdir() if path.is_dir())
            if len(directories) != count:
                raise ValueError(f"{arm}/{dataset}: expected {count} outputs, got {len(directories)}")
            for directory in directories:
                complex_id = directory.name
                summary_path = directory / "summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                inputs = summary.get("inputs", {})
                if (
                    summary.get("status") != "complete_descriptive"
                    or summary.get("dataset") != dataset
                    or summary.get("complex_id") != complex_id
                    or int(summary.get("pose_count", -1)) != 100
                    or float(summary.get("sigma", -1)) != 2.0
                    or int(summary.get("pose_batch_size", -1)) != 20
                    or inputs.get("confidence_checkpoint_sha256") != expected_sha
                    or inputs.get("docking_checkpoint_sha256") != DOCKING_SHA256
                ):
                    raise ValueError(f"invalid score summary: {summary_path}")
                scores_spec = summary["artifacts"]["scores_csv"]
                scores_path = Path(scores_spec["path"])
                if file_sha256(scores_path) != scores_spec["sha256"]:
                    raise ValueError(f"score artifact hash mismatch: {scores_path}")
                score_rows = _load_scores(scores_path)
                official = [official_rows[(dataset, complex_id, index)] for index in range(100)]
                key = (dataset, complex_id)
                row = all_rows.setdefault(key, {"dataset": dataset, "id": complex_id})
                for stage in ("step_000", "step_100"):
                    values = _stage(
                        score_rows,
                        official if stage == "step_100" else None,
                        stage=stage,
                    )
                    selected = summary["selected"][stage]
                    if int(selected["pose_index"]) != int(values["selected_index"]):
                        raise ValueError(f"selector mismatch: {summary_path} {stage}")
                    row.update({f"{arm}_{stage}_{name}": value for name, value in values.items()})

    if len(all_rows) != sum(EXPECTED_COUNTS.values()):
        raise ValueError(f"expected 393 paired complexes, got {len(all_rows)}")
    rows = [all_rows[key] for key in sorted(all_rows)]
    aggregate: list[dict[str, Any]] = []
    for arm in ARMS:
        for dataset in EXPECTED_COUNTS:
            subset = [row for row in rows if row["dataset"] == dataset]
            for stage in ("step_000", "step_100"):
                renamed = [
                    {
                        **row,
                        **{
                            f"{stage}_{name}": row[f"{arm}_{stage}_{name}"]
                            for name in (
                                "selected_rmsd", "selected_lt2", "selected_official_valid",
                                "selected_joint", "top5_lt2", "oracle_lt2", "oracle_rmsd", "k2",
                            )
                        },
                    }
                    for row in subset
                ]
                aggregate.append({"arm": arm, "dataset": dataset, "stage": stage, **_aggregate(renamed, stage=stage)})
    comparisons = [
        _comparison(rows, baseline=baseline, candidate=candidate, dataset=dataset)
        for baseline, candidate in (("u001500", "u025000"), ("u001500", "u050000"), ("u025000", "u050000"))
        for dataset in EXPECTED_COUNTS
    ]
    result = {
        "schema_version": "effdock.s50_symmetry_confidence_refined_external_report.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "complete_descriptive",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "claim_boundary": "Repeated-use Astex/PoseBusters descriptive comparison; external outcomes do not select a checkpoint.",
        "arms": ARMS,
        "docking_checkpoint_sha256": DOCKING_SHA256,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "complex_rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.link(temporary, args.output)
    temporary.unlink()


if __name__ == "__main__":
    main()
