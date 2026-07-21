#!/usr/bin/env python3
"""Evaluate frozen cluster-free confidence filters on saved external pose sets."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pattern in paths:
        matched = sorted(glob.glob(pattern))
        if not matched:
            raise FileNotFoundError(f"no files match {pattern!r}")
        for raw_path in matched:
            path = Path(raw_path)
            with path.open() as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        row["_source"] = str(path)
                        rows.append(row)
    ids = [str(row["id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate complex IDs across pose-score inputs")
    return sorted(rows, key=lambda row: str(row["id"]))


def _arrays(row: dict[str, Any]) -> dict[str, np.ndarray]:
    scores = row["scores"]
    arrays = {
        "rmsd": np.asarray(row["rmsd"], dtype=np.float64),
        "pred": np.asarray(scores["confidence_pred_rmsd"], dtype=np.float64),
        "success": np.asarray(scores["confidence_pred_success"], dtype=np.float64),
        "atom_rmsd": np.asarray(scores["atom_rmsd"], dtype=np.float64),
        "atom_ok": np.asarray(scores["atom_ok"], dtype=np.float64),
        "clash": np.asarray(scores["pl_clash_1p6"], dtype=np.float64),
        "fast_valid": np.asarray(scores["validity_valid"], dtype=np.float64),
    }
    lengths = {len(value) for value in arrays.values()}
    if lengths != {int(row["num_samples"])}:
        raise ValueError(f"{row['id']}: score lengths {lengths} do not match num_samples")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError(f"{row['id']}: non-finite pose scores")
    return arrays


def _lowest_pred(pred: np.ndarray, mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return None
    return int(indices[np.argmin(pred[indices])])


def _strict_filter(a: dict[str, np.ndarray]) -> int:
    pred, success, atom_ok, clash = a["pred"], a["success"], a["atom_ok"], a["clash"]
    base = int(np.argmin(pred))
    within = pred <= pred[base] + 0.03
    if clash[base] > 0.0:
        physical = within & (clash <= 0.0) & (success >= success[base]) & (
            atom_ok >= atom_ok[base]
        )
        selected = _lowest_pred(pred, physical)
        if selected is not None:
            return selected
    consensus = within & (clash <= clash[base]) & (success >= success[base]) & (
        atom_ok >= atom_ok[base]
    )
    consensus[base] = False
    selected = _lowest_pred(pred, consensus)
    return base if selected is None else selected


def _atom_guard(a: dict[str, np.ndarray]) -> int:
    pred = a["pred"]
    base = int(np.argmin(pred))
    mask = (
        (pred <= pred[base] + 0.20)
        & (a["clash"] <= a["clash"][base])
        & (a["atom_rmsd"] <= a["atom_rmsd"][base])
        & (a["success"] >= a["success"][base] - 0.05)
        & (a["atom_ok"] >= a["atom_ok"][base] - 0.02)
    )
    mask[base] = False
    selected = _lowest_pred(pred, mask)
    return base if selected is None else selected


def _summarize(rows: list[dict[str, Any]], selector: str) -> dict[str, float | int]:
    rmsd = np.asarray([row[f"{selector}_rmsd"] for row in rows], dtype=np.float64)
    selected = np.asarray([row[f"{selector}_index"] for row in rows], dtype=np.int64)
    pure = np.asarray([row["pure_index"] for row in rows], dtype=np.int64)
    valid = np.asarray([row[f"{selector}_fast_valid"] for row in rows], dtype=np.float64)
    pure_success = np.asarray([row["pure_rmsd"] < 2.0 for row in rows])
    selected_success = rmsd < 2.0
    return {
        "n": int(len(rows)),
        "lt2_count": int(selected_success.sum()),
        "lt2_pct": float(selected_success.mean() * 100.0),
        "median_rmsd": float(np.median(rmsd)),
        "mean_rmsd": float(np.mean(rmsd)),
        "fast_valid_pct": float(valid.mean() * 100.0),
        "switch_count": int((selected != pure).sum()),
        "switch_pct": float((selected != pure).mean() * 100.0),
        "rescued_lt2": int((~pure_success & selected_success).sum()),
        "lost_lt2": int((pure_success & ~selected_success).sum()),
    }


def evaluate_dataset(paths: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = _load(paths)
    out: list[dict[str, Any]] = []
    for row in source:
        arrays = _arrays(row)
        indices = {
            "pure": int(np.argmin(arrays["pred"])),
            "strict": _strict_filter(arrays),
            "atom_guard": _atom_guard(arrays),
            "oracle": int(np.argmin(arrays["rmsd"])),
        }
        converted: dict[str, Any] = {
            "id": row["id"],
            "source": row["_source"],
            "num_samples": int(row["num_samples"]),
        }
        for name, index in indices.items():
            converted[f"{name}_index"] = index
            converted[f"{name}_rmsd"] = float(arrays["rmsd"][index])
            converted[f"{name}_fast_valid"] = bool(arrays["fast_valid"][index] >= 0.5)
        out.append(converted)
    return out, {
        selector: _summarize(out, selector)
        for selector in ("pure", "strict", "atom_guard", "oracle")
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--astex", action="append", required=True, help="JSONL glob")
    parser.add_argument("--posebusters", action="append", required=True, help="JSONL glob")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/eff-dock/confidence-filter-v1-external"),
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "protocol_id": "EFFDOCK-CONFIDENCE-FILTER-V1-EXTERNAL",
        "candidate_contract": "N80/S25/sigma0.5/pocket10, geometry step100000, confidence step42500",
        "selectors_frozen_before_external": True,
        "official_posebusters_pass_all_run": False,
        "datasets": {},
    }
    for dataset, paths in (("astex", args.astex), ("posebusters", args.posebusters)):
        rows, summary = evaluate_dataset(paths)
        result["datasets"][dataset] = summary
        csv_path = args.output_dir / f"{dataset}_rows.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result_path = args.output_dir / "result.json"
    with result_path.open("w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
