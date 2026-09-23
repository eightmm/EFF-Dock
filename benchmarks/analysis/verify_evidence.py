"""Verify published evidence tables against case-level records and manuscript inputs."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks.analysis.evidence import COUNTS, aggregate_confidence, aggregate_physical, require

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarks/results/paper"


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def close(left, right):
    if isinstance(left, dict):
        require(left.keys() == right.keys(), "Summary keys differ")
        for key in left:
            close(left[key], right[key])
    elif isinstance(left, list):
        require(len(left) == len(right), "Summary length differs")
        for a, b in zip(left, right, strict=True):
            close(a, b)
    elif isinstance(left, float):
        require(np.isclose(left, right, atol=1e-10, rtol=0), "Summary value differs")
    else:
        require(left == right, "Summary value differs")


def verify():
    directory = DATA / "evidence"
    data = json.loads((directory / "results.json").read_text())
    cases = read_csv(directory / "confidence_cases.csv")
    for row in cases:
        for key, value in row.items():
            if key in ("dataset", "id", "stage"):
                continue
            row[key] = (
                None
                if value == ""
                else value == "True"
                if value in ("True", "False")
                else float(value)
            )
    require(
        len(cases)
        == len({(r["dataset"], r["repeat"], r["id"], r["stage"]) for r in cases})
        == 12492,
        "Incomplete or duplicate confidence table",
    )
    original = {
        (r["dataset"], int(r["repeat"]), r["id"], r["stage"]): r
        for r in read_csv(DATA / "selected_outcomes.csv")
        if r["policy"] == "filtered"
    }
    require(len(original) == len(cases), "Missing primary outcomes")
    for r in cases:
        old = original[r["dataset"], r["repeat"], r["id"], r["stage"]]
        for old_key, new_key in (
            ("rmsd_lt2", "filtered_success"),
            ("pb_valid", "filtered_pb_valid"),
            ("joint", "filtered_joint"),
        ):
            require((old[old_key] == "True") == r[new_key], "Published outcome drift")
        require(r["filtered_regret"] >= 0 and r["baseline_regret"] >= 0, "Negative regret")
    confidence, density = aggregate_confidence(cases)
    close(confidence, data["confidence"])
    close(density, data["density"])
    pb = {}
    for row in read_csv(directory / "pb_checks.csv"):
        key = row["dataset"], int(row["repeat"]), row["id"], row["stage"], row["policy"]
        require(key not in pb, "Duplicate PB rows")
        checks = {
            k: bool(int(v))
            for k, v in row.items()
            if k not in ("dataset", "repeat", "id", "stage", "policy", "pose_index")
        }
        require(len(checks) == 27, "Missing PB checks")
        pb[key] = dict(row, pose_index=int(row["pose_index"]), checks=checks)
    require(len(pb) == 24984, "Incomplete PB table")
    close(aggregate_physical(pb), data["physical"])
    related = read_csv(directory / "relatedness.csv")
    require(
        len(related) == len({(r["dataset"], r["id"]) for r in related}) == 2082, "Relatedness IDs"
    )
    for r in related:
        expected = (
            float(r["sequence_identity"]) < 30
            and float(r["ligand_tanimoto"]) < 0.5
            and r["exact_ligand"] == "False"
        )
        require(expected == (r["stringent"] == "True"), "Subset threshold drift")
    for row in data["subsets"]:
        ids = {
            r["id"]
            for r in related
            if r["dataset"] == row["dataset"]
            and (row["subset"] == "All" or r["stringent"] == "True")
        }
        require(len(ids) == row["n"], "Subset count mismatch")
        for rep in range(3):
            group = [
                r
                for r in cases
                if r["dataset"] == row["dataset"]
                and r["id"] in ids
                and r["repeat"] == rep
                and r["stage"] == "refined"
            ]
            require(len(group) == len(ids), "Subset join mismatch")
            for metric, key in (
                ("top1", "filtered_success"),
                ("joint", "filtered_joint"),
                ("oracle", "oracle_success"),
            ):
                close(
                    100 * sum(r[key] for r in group) / len(group) if group else None,
                    row[metric]["per_repeat"][rep],
                )
    for ds, count in COUNTS.items():
        for stage in ("raw", "refined"):
            for metric in ("success_probability", "predicted_rmsd"):
                bins = [
                    r
                    for r in data["reliability"]
                    if (r["dataset"], r["stage"], r["metric"]) == (ds, stage, metric)
                ]
                require(sum(r["n"] for r in bins) == count * 300, "Calibration denominator")
                if metric == "success_probability":
                    brier = sum(r["n"] * r["mse"] for r in bins) / sum(r["n"] for r in bins)
                    close(
                        brier,
                        next(r for r in confidence if (r["dataset"], r["stage"]) == (ds, stage))[
                            "brier"
                        ]["mean"],
                    )
    baseline = read_csv(directory / "baseline_cases.csv")
    require(len(baseline) == 5895, "Baseline count mismatch")
    group = defaultdict(list)
    for row in baseline:
        group[row["dataset"], row["method"]].append(row)
    for r in json.loads((directory / "baseline_uncertainty.json").read_text())["comparisons"]:
        rows = group[r["dataset"], r["method"]]
        delta = 100 * np.mean(
            [(v["effdock_" + r["metric"]] == "True") - (v[r["metric"]] == "True") for v in rows]
        )
        for kind in ("complex_bootstrap", "pdb_bootstrap"):
            close(float(delta), r[kind]["delta_pp"])
            require(r[kind]["complexes"] == COUNTS[r["dataset"]], "Baseline denominator")
    for row in json.loads((directory / "structures.json").read_text()):
        match = next(
            r
            for r in cases
            if (r["dataset"], r["id"], r["repeat"], r["stage"])
            == ("astex", row["id"], 0, "refined")
        )
        close(row["selected_rmsd"], match["selected_rmsd"])
        require(row["selected_index"] == match["selected_index"], "Structure index mismatch")
        for key in ("reference", "selected", "comparator"):
            mol = row[key]
            if mol is not None:
                xyz = np.asarray(mol["coordinates"])
                require(
                    xyz.shape == (len(mol["elements"]), 3) and np.isfinite(xyz).all(),
                    "Structure coordinates",
                )
    metadata = json.loads((ROOT / "docs/paper/manifest.json").read_text())
    for row in metadata["numerical_inputs"]:
        require(
            hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() == row["sha256"],
            "Numerical source hash drift",
        )
    print(
        "Verified 12,492 confidence banks, 24,984 PB rows, 2,082 relatedness records, 5,895 baseline rows, calibration denominators and three structures"
    )


if __name__ == "__main__":
    verify()
