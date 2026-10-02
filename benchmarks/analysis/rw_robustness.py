"""Collect and verify completed Rw sensitivity aggregates without new inference."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarks/results/paper"
COUNTS = {"astex": 85, "posebusters": 308}
METRICS = ["rmsd_lt2", "pb_valid", "joint", "oracle"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(test, message):
    if not test:
        raise ValueError(message)


def verify(data):
    require(
        data["status"] == "complete" and data["orientation_injection"] == "rw",
        "Only complete corrected-Rw measurements can be published",
    )
    require(data["protocol"] == "EFFDOCK-RW-ROBUSTNESS-R3-V1", "Wrong sensitivity protocol")
    require(data["seeds"] == [42, 100042, 200042], "Seed inventory changed")
    require(data["new_executions"] == 27117, "Execution denominator changed")
    cs = data["conditions"]
    require(len(cs) == len({c["id"] for c in cs}) == 24, "Missing or duplicate conditions")
    actual = {(c["cutoff"], c["sigma"], c["jitter"], c["n"], c["steps"], c["guided"]) for c in cs}
    expected = {
        (cutoff, 2, jitter, 100, 10, False) for cutoff in [6, 8, 10, 12, 14] for jitter in [0, 1, 2]
    }
    expected |= {(10, sigma, jitter, 100, 10, False) for sigma in [1, 4] for jitter in [0, 1, 2]}
    expected |= {(10, 2, 0, 100, 10, True), (10, 2, 0, 40, 25, False), (10, 2, 0, 40, 25, True)}
    require(actual == expected, "Sensitivity factors changed")
    by_id = {c["id"]: c for c in cs}
    for condition in cs:
        require(type(condition["guided"]) is bool, "Invalid guidance flag")
        require(
            condition["reuse"]
            == (
                (
                    condition["cutoff"],
                    condition["sigma"],
                    condition["jitter"],
                    condition["n"],
                    condition["steps"],
                    condition["guided"],
                )
                == (10, 2, 0, 100, 10, False)
            ),
            "Invalid baseline reuse",
        )
    keys = [(r["condition"]["id"], r["dataset"], r["stage"], r["policy"]) for r in data["rows"]]
    require(len(keys) == len(set(keys)) == 192, "Missing or duplicate aggregate rows")
    repeated = data["per_repeat"]
    require(
        len(repeated) == 576
        and len(
            {(r["condition"], r["dataset"], r["repeat"], r["stage"], r["policy"]) for r in repeated}
        )
        == 576,
        "Invalid repeat rows",
    )
    for row in data["rows"]:
        require(
            row["condition"] == by_id.get(row["condition"]["id"]),
            "Row condition differs from registered inventory",
        )
        require(
            row["dataset"] in COUNTS and row["count"] == COUNTS[row["dataset"]],
            "Cohort denominator changed",
        )
        require(
            row["stage"] in ["raw", "refined"] and row["policy"] in ["baseline", "filtered"],
            "Undeclared endpoint",
        )
        group = [
            r
            for r in repeated
            if (r["condition"], r["dataset"], r["stage"], r["policy"])
            == (row["condition"]["id"], row["dataset"], row["stage"], row["policy"])
        ]
        require(len(group) == 3 and [r["repeat"] for r in group] == [0, 1, 2], "Incomplete seeds")
        require(all(r["count"] == row["count"] for r in group), "Repeat denominator changed")
        for metric in METRICS:
            values = [r[metric] for r in group]
            require(all(math.isfinite(x) and 0 <= x <= 100 for x in values), "Invalid percentage")
            require(
                all(
                    abs(x * row["count"] / 100 - round(x * row["count"] / 100)) < 1e-8
                    for x in values
                ),
                "Success percentage inconsistent with cohort size",
            )
            require(row[metric]["per_repeat"] == values, "Repeat values changed")
            require(
                abs(row[metric]["mean"] - statistics.mean(values)) < 1e-10
                and abs(row[metric]["sd"] - statistics.stdev(values)) < 1e-10,
                "Mean/SD mismatch",
            )
        for r in group:
            require(
                r["joint"] <= min(r["rmsd_lt2"], r["pb_valid"]) + 1e-9
                and r["rmsd_lt2"] <= r["oracle"] + 1e-9,
                "Endpoint subset relation violated",
            )
    baseline = [r for r in data["rows"] if r["condition"]["reuse"]]
    require(len(baseline) == 8, "Reused baseline changed")
    primary = json.loads((DATA / "benchmark_results.json").read_text())["rows"]
    for row in baseline:
        matches = [
            r
            for r in primary
            if (r["dataset"], r["stage"], r["policy"], r["n"], r["guidance"])
            == (row["dataset"], row["stage"], row["policy"], 100, "off")
        ]
        require(len(matches) == 1, "Missing primary baseline")
        for metric in ["rmsd_lt2", "pb_valid", "joint"]:
            require(
                all(
                    abs(row[metric][field] - matches[0][metric][field]) < 1e-9
                    for field in ["mean", "sd"]
                )
                and all(
                    abs(a - b) < 1e-9
                    for a, b in zip(
                        row[metric]["per_repeat"], matches[0][metric]["per_repeat"], strict=True
                    )
                ),
                "Reused baseline differs from primary results",
            )


def collect(source, output):
    source = source.resolve()
    data = json.loads(source.read_text())
    verify(data)
    # Public aggregates retain source hashes without cluster paths or per-case structures.
    portable = {
        k: data[k]
        for k in [
            "status",
            "protocol",
            "orientation_injection",
            "conditions",
            "seeds",
            "new_executions",
            "primary",
            "rows",
            "per_repeat",
            "source_inventory_sha256",
            "baseline_report_sha256",
        ]
    }
    for key in ["execution_revision", "recovery_manifest_sha256"]:
        portable[key] = data[key]
    portable["source_report_sha256"] = digest(source)
    portable["selection_ledger_hashes"] = [
        dict(
            condition=r["condition"],
            dataset=r["dataset"],
            repeat=r["repeat"],
            reused=r["reused"],
            sha256=r["selection_ledger_sha256"],
        )
        for r in data["provenance"]
    ]
    require(len(portable["selection_ledger_hashes"]) == 144, "Missing source ledger hashes")
    verify(portable)
    output.write_text(json.dumps(portable, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print("Verified complete Rw sensitivity: 24 conditions, 576 repeat rows, 192 aggregates")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=DATA / "rw_robustness.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        verify(json.loads(args.output.read_text()))
        print("Rw sensitivity statistics and reused primary baseline verified")
    elif args.input:
        collect(args.input, args.output)
    else:
        parser.error("Provide --input or --check")
