#!/usr/bin/env python3
"""Aggregate U70k and U100k on the frozen raw/refined external bank."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from effdock.workflows.evaluate import file_sha256

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.report_s50_symmetry_confidence_refined_external import (
    DOCKING_SHA256,
    _aggregate,
    _comparison,
    _load_official,
    _load_scores,
    _stage,
)
from scripts.run_s50_raw_refined_confidence_external_shard import PROTOCOL_ID
from scripts.run_s50_symmetry_confidence_refined_external_shard import EXPECTED_COUNTS

ARMS = {
    "u070000": "ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638",
    "u100000": "2ea1aca4f1c326cd0841e76c3597e3749231854a523d1ba8bd923c6fb5a9bff8",
}


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
                    row.update(
                        {f"{arm}_{stage}_{name}": value for name, value in values.items()}
                    )

    if len(all_rows) != sum(EXPECTED_COUNTS.values()):
        raise ValueError(f"expected 393 paired complexes, got {len(all_rows)}")
    rows = [all_rows[key] for key in sorted(all_rows)]
    aggregate: list[dict[str, Any]] = []
    metric_names = (
        "selected_rmsd",
        "selected_lt2",
        "selected_official_valid",
        "selected_joint",
        "top5_lt2",
        "oracle_lt2",
        "oracle_rmsd",
        "k2",
    )
    for arm in ARMS:
        for dataset in EXPECTED_COUNTS:
            subset = [row for row in rows if row["dataset"] == dataset]
            for stage in ("step_000", "step_100"):
                renamed = [
                    {
                        **row,
                        **{
                            f"{stage}_{name}": row[f"{arm}_{stage}_{name}"]
                            for name in metric_names
                        },
                    }
                    for row in subset
                ]
                aggregate.append(
                    {
                        "arm": arm,
                        "dataset": dataset,
                        "stage": stage,
                        **_aggregate(renamed, stage=stage),
                    }
                )

    comparisons = [
        _comparison(
            rows,
            baseline="u070000",
            candidate="u100000",
            dataset=dataset,
        )
        for dataset in EXPECTED_COUNTS
    ]
    result = {
        "schema_version": "effdock.s50_raw_refined_confidence_external_report.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "complete_descriptive",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "claim_boundary": (
            "Repeated-use Astex/PoseBusters descriptive comparison; external "
            "outcomes do not select a checkpoint."
        ),
        "arms": ARMS,
        "docking_checkpoint_sha256": DOCKING_SHA256,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "complex_rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.link(temporary, args.output)
    temporary.unlink()


if __name__ == "__main__":
    main()
