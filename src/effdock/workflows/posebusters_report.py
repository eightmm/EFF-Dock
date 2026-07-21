"""Run official PoseBusters redock checks on selected PoseBusters poses."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
from posebusters import PoseBusters


def load_rows(input_dir: Path, run_name: str) -> list[dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for path in sorted(input_dir.glob(f"{run_name}*.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["id"] in rows:
                    raise ValueError(f"duplicate PoseBusters row: {row['id']}")
                rows[row["id"]] = row
    return [rows[key] for key in sorted(rows)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/benchmarks/raw"))
    parser.add_argument("--run-name", default="effdock-redock-ema-n40-s25-v1-posebusters")
    parser.add_argument(
        "--pose-dir",
        type=Path,
        default=Path("outputs/benchmarks/raw/poses/posebusters/vina"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/benchmarks/posebusters_official")
    )
    parser.add_argument("--selector", default="effdock_torch_vina_plus_dg")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args(argv)

    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    all_rows = load_rows(args.input_dir, args.run_name)
    rows = all_rows[args.shard_index :: args.num_shards]
    if not rows:
        raise ValueError("no PoseBusters rows assigned")

    buster = PoseBusters(config="redock", max_workers=0)
    results: list[dict] = []
    failures: list[dict] = []
    for index, row in enumerate(rows, start=1):
        complex_id = row["id"]
        try:
            frame = buster.bust(
                args.pose_dir / f"{complex_id}.sdf",
                Path(row["ligand_ref"]),
                Path(row["protein"]),
                full_report=False,
            )
            checks = {
                key: False if pd.isna(value) else bool(value)
                for key, value in frame.iloc[0].to_dict().items()
            }
            validity_checks = {
                key: value for key, value in checks.items() if not key.startswith("rmsd_")
            }
            results.append(
                {
                    "id": complex_id,
                    "posebusters_valid": all(validity_checks.values()),
                    **checks,
                }
            )
            print(
                f"[{index:04d}/{len(rows)}] {complex_id} valid={results[-1]['posebusters_valid']}"
            )
        except Exception as exc:
            failures.append({"id": complex_id, "error": repr(exc)})
            print(f"[{index:04d}/{len(rows)}] {complex_id} FAIL {exc!r}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"shard-{args.shard_index:03d}-of-{args.num_shards:03d}"
    csv_path = args.output_dir / f"{tag}.csv"
    if results:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    summary = {
        "posebusters_version": "0.6.5",
        "config": "redock",
        "selector": args.selector,
        "num_discovered_total": len(all_rows),
        "num_assigned": len(rows),
        "num_success": len(results),
        "num_failed": len(failures),
        "posebusters_valid_pct": (
            sum(result["posebusters_valid"] for result in results) / len(results) * 100
            if results
            else None
        ),
        "failures": failures,
        "csv": str(csv_path) if results else None,
    }
    summary_path = args.output_dir / f"{tag}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
