#!/usr/bin/env python3
"""Extract the classical no-minimization rows from the PoseBusters paper data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    ROOT
    / "outputs/external_models/inputs/posebench_native/posebusters_benchmark/"
    "vina_posebusters_benchmark_inputs.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "benchmarks/results/external_models/posebusters_classical_paper_values.json"
)
SOURCE_URL = (
    "https://zenodo.org/records/8278563/files/"
    "posebusters_paper_results.csv?download=1"
)
SOURCE_MD5 = "a7cbe725e86e412fdfeb3c3e35c566dd"

IDENTITY_COLUMNS = {
    "dataset",
    "method",
    "post-processing",
    "pdb_id",
    "ccd_id",
    "has_cofactors",
    "sequence_identity",
    "rmsd_within_threshold",
    "rmsd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-results", type=Path, required=True)
    parser.add_argument("--posebusters-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def metric(count: int, denominator: int) -> dict[str, float | int]:
    return {"count": count, "pct": 100.0 * count / denominator}


def main() -> None:
    args = parse_args()
    with args.paper_results.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("PoseBusters paper result CSV is empty")
    validity_checks = [key for key in rows[0] if key not in IDENTITY_COLUMNS]
    if len(validity_checks) != 18:
        raise ValueError(
            f"expected 18 paper-era validity columns, found {len(validity_checks)}"
        )

    with args.posebusters_manifest.open(newline="", encoding="utf-8") as handle:
        posebusters_ids = {
            row["complex_name"].upper() for row in csv.DictReader(handle)
        }
    if len(posebusters_ids) != 308:
        raise ValueError("PoseBusters v2 manifest must contain exactly 308 IDs")

    datasets: dict[str, object] = {}
    for dataset_key, source_name, denominator in (
        ("astex", "astex", 85),
        ("posebusters", "posebuster", 308),
    ):
        methods: list[dict[str, object]] = []
        for method, display in (("gold", "GOLD"), ("vina", "AutoDock Vina")):
            selected = [
                row
                for row in rows
                if row["dataset"] == source_name
                and row["method"] == method
                and row["post-processing"] == "none"
                and (
                    dataset_key == "astex"
                    or f"{row['pdb_id']}_{row['ccd_id']}".upper()
                    in posebusters_ids
                )
            ]
            if len(selected) != denominator:
                raise ValueError(
                    f"{dataset_key}/{method}: {len(selected)} rows != {denominator}"
                )
            rmsd_count = sum(
                row["rmsd_within_threshold"] == "True" for row in selected
            )
            pb_valid = [
                all(row[key] == "True" for key in validity_checks)
                for row in selected
            ]
            joint_count = sum(
                row["rmsd_within_threshold"] == "True" and valid
                for row, valid in zip(selected, pb_valid, strict=True)
            )
            methods.append(
                {
                    "method": display,
                    "family": "classical",
                    "source": "PoseBusters paper deposited result table",
                    "post_processing": "none",
                    "top1_rmsd_lt2": metric(rmsd_count, denominator),
                    "top1_pb_valid": metric(sum(pb_valid), denominator),
                    "top1_joint_rmsd_lt2_pb_valid": metric(
                        joint_count, denominator
                    ),
                }
            )
        datasets[dataset_key] = {
            "name": "Astex Diverse" if dataset_key == "astex" else "PoseBusters v2",
            "n": denominator,
            "methods": methods,
        }

    payload = {
        "schema_version": 1,
        "comparison_scope": "supplied_pocket_only",
        "provenance": {
            "paper": "Buttenschoen, Morris, and Deane, Chemical Science (2024)",
            "doi": "10.1039/D3SC04185A",
            "zenodo_record": "https://zenodo.org/records/8278563",
            "source_csv_url": SOURCE_URL,
            "source_csv_md5": SOURCE_MD5,
            "posebusters_v2_filter": str(
                args.posebusters_manifest.resolve().relative_to(ROOT.resolve())
            ),
        },
        "metric_contract": (
            "Top-1 RMSD <2 A; Joint additionally passes every validity column "
            "in the deposited paper result table. No energy minimization rows."
        ),
        "datasets": datasets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
