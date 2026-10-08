"""Reproduce benchmark-specific summary statistics from released case outcomes.

This checks endpoint aggregation, not molecular scoring or candidate selection.
Missing native scores stay unassigned; they are not relabelled as failures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

COUNTS = {"foldbench": 558, "openbind": 802, "phibench": 206}
METRICS = {
    "foldbench": ("rmsd_success", "lddt_pli_success", "published_success"),
    "openbind": ("rmsd_success", "lddt_pli_success", "published_success", "rmsd_pb_success"),
    "phibench": ("rmsd_success", "rmsd_pb_success"),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def flag(value: str) -> bool | None:
    require(value in {"True", "False", ""}, f"Invalid outcome flag: {value!r}")
    return None if value == "" else value == "True"


def conjunction(*values: bool | None) -> bool | None:
    if False in values:
        return False
    return None if None in values else True


def rate(values: list[bool | None]) -> dict:
    require(bool(values), "Empty cohort")
    successes = sum(value is True for value in values)
    unresolved = sum(value is None for value in values)
    determinate = len(values) - unresolved
    return {
        "successes": successes,
        "all_cases": len(values),
        "determinate_cases": determinate,
        "unresolved_cases": unresolved,
        "rate_percent": 100 * successes / len(values),
        "defined_case_rate_percent": 100 * successes / determinate if determinate else None,
    }


def compare(actual, expected, location: str) -> None:
    if isinstance(actual, dict):
        require(isinstance(expected, dict), location)
        for key, value in actual.items():
            require(key in expected, f"Missing {location}/{key}")
            compare(value, expected[key], f"{location}/{key}")
    elif isinstance(actual, float):
        require(isinstance(expected, (int, float)) and not isinstance(expected, bool) and math.isfinite(expected)
                and math.isclose(actual, expected, abs_tol=1e-9, rel_tol=0), location)
    else:
        require(actual == expected, location)


def validate_top1(row: dict) -> None:
    # OR across candidates cannot be followed by AND across endpoint totals.
    if int(row["top_k"]) != 1:
        return
    dataset = row["dataset"]
    rmsd = flag(row["rmsd_success"])
    joint = flag(row["published_success"])
    if dataset in {"foldbench", "openbind"}:
        first = rmsd if dataset == "foldbench" else flag(row["rmsd_pb_success"])
        require(joint is conjunction(first, flag(row["lddt_pli_success"])), "Top-1 same-pose conjunction mismatch")
    if dataset != "foldbench" and flag(row["rmsd_pb_success"]) is True:
        require(rmsd is True, "PoseBusters conjunction without RMSD success")


def verify(data: dict, cases: list[dict]) -> dict:
    require(data["schema"] == "effdock.published_endpoint_source_data.v1", "Unsupported schema")
    require(data["status"] == "complete_corrected_saved_pose_analysis_verified", "Incomplete source data")
    groups, ids, seen = defaultdict(list), defaultdict(set), set()
    for row in cases:
        ds = row["dataset"]
        require(ds in COUNTS, "Unknown dataset")
        repeat, prefix, top_k = (int(row[k]) for k in ("repeat", "saved_bank_prefix", "top_k"))
        require(repeat in range(3) and row["stage"] in {"raw", "refined"}, "Invalid repeat or stage")
        key = ds, row["stage"], prefix, top_k, repeat
        entity = key + (row["id"],)
        require(entity not in seen, "Duplicated case endpoint")
        seen.add(entity)
        ids[ds, repeat].add(row["id"])
        for metric in METRICS[ds]:
            flag(row[metric])
        evaluated = int(row["evaluated_candidates"])
        require(1 <= evaluated <= top_k <= prefix, "Invalid evaluated candidate count")
        require(all(0 <= int(row[k]) <= evaluated for k in ("undefined_pose_count", "partial_coverage_pose_count")), "Invalid coverage count")
        validate_top1(row)
        groups[key].append(row)
    require(len(cases) == 34692, "Incomplete case table")
    require(sum(len(value) for value in ids.values()) == data["complex_repeat_count"] == 4698, "Bank count mismatch")
    for ds, count in COUNTS.items():
        require(all(len(ids[ds, rep]) == count for rep in range(3)), "Cohort membership mismatch")
        require(ids[ds, 0] == ids[ds, 1] == ids[ds, 2], "Repeat membership differs")
    expected_groups = {
        (ds, stage, prefix, k, rep)
        for ds in COUNTS
        for stage in (("refined",) if ds == "phibench" else ("raw", "refined"))
        for prefix in ((100,) if ds == "foldbench" else (25, 100) if ds == "openbind" else (40, 100))
        for k in ((1,) if ds == "foldbench" else (1, 5, 25) if ds == "openbind" else (1, 5))
        for rep in range(3)
    }
    require(set(groups) == expected_groups, "Endpoint group mismatch")
    summaries = {(r["dataset"], r["stage"], r["saved_bank_prefix"], r["top_k"], r["endpoint"]): r
                 for r in data["summary_rows"]}
    expected_summaries = {(ds, stage, prefix, k, metric) for ds, stage, prefix, k, _ in groups for metric in METRICS[ds]}
    require(len(summaries) == len(data["summary_rows"]) == 62 and set(summaries) == expected_summaries, "Summary coverage mismatch")
    for key, summary in summaries.items():
        ds, stage, prefix, k, metric = key
        rates = []
        observed = {r["repeat"]: r for r in summary["repeats"]}
        require(len(observed) == len(summary["repeats"]) == 3 and set(observed) == {0, 1, 2}, "Summary repeats differ")
        require(summary["complexes_per_repeat"] == COUNTS[ds], "Summary cohort size differs")
        for rep in range(3):
            rows = groups[ds, stage, prefix, k, rep]
            require({r["id"] for r in rows} == ids[ds, rep], "Missing endpoint cases")
            result = rate([flag(r[metric]) for r in rows])
            result.update(undefined_pose_count=sum(int(r["undefined_pose_count"]) for r in rows),
                          partial_coverage_pose_count=sum(int(r["partial_coverage_pose_count"]) for r in rows))
            compare(result, observed[rep], str(key) + f"/repeat{rep}")
            rates.append(result["rate_percent"])
        compare(mean(rates), summary["mean_percent"], str(key) + "/mean")
        compare(stdev(rates), summary["sd_percent"], str(key) + "/sample_sd")
    conditional = data["native_both_metric_scoreable_subset"]
    conditional_keys = {(r["dataset"], r["stage"], r["metric"]) for r in conditional}
    expected_conditional = {(ds, stage, metric) for ds in ("foldbench", "openbind")
                            for stage in ("raw", "refined")
                            for metric in (("rmsd_success", "lddt_pli_success", "published_success")
                                           if ds == "foldbench" else ("rmsd_success", "rmsd_pb_success", "published_success"))}
    require(len(conditional_keys) == len(conditional) == 12 and conditional_keys == expected_conditional, "Conditional table coverage mismatch")
    for summary in conditional:
        ds, stage, metric = (summary[k] for k in ("dataset", "stage", "metric"))
        require(ds in {"foldbench", "openbind"} and summary["saved_bank_prefix"] == 100 and summary["top_k"] == 1, "Invalid scoreable subset")
        require(len(summary["repeat_results"]) == 3 and {r["repeat"] for r in summary["repeat_results"]} == {0, 1, 2}, "Conditional repeats differ")
        rates = []
        for observed in summary["repeat_results"]:
            rows = groups[ds, stage, 100, 1, observed["repeat"]]
            assigned = [r for r in rows if all(flag(r[k]) is not None for k in ("rmsd_success", "lddt_pli_success"))]
            compare({"all_complexes": len(rows), "both_metrics_assigned": len(assigned), "excluded_from_conditional_denominator": len(rows) - len(assigned)}, observed, "scoreability")
            value = 100 * sum(flag(r[metric]) is True for r in assigned) / len(assigned) if assigned else None
            compare(value, observed["rates_percent"][metric], "conditional repeat rate")
            rates.append(value)
        require(len(rates) == 3, "Conditional repeats differ")
        compare(mean(rates) if all(x is not None for x in rates) else None, summary["mean_rate_percent"], "conditional mean")
        compare(stdev(rates) if all(x is not None for x in rates) else None, summary["sd_percent"], "conditional sample_sd")
    return {"status": "passed", "complex_repeat_banks": 4698, "case_endpoint_rows": len(cases), "full_summary_rows": len(summaries), "both_metric_assigned_summary_rows": len(conditional), "scope": "Released outcome aggregation only; does not independently rerun molecular scoring, candidate selection, reference controls or literature extraction."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).resolve().parents[2] / "papers/data"
    parser.add_argument("--data", type=Path, default=default / "published_endpoints.json")
    parser.add_argument("--cases", type=Path, default=default / "published_endpoints.csv")
    args = parser.parse_args()
    with args.cases.open(newline="") as handle:
        cases = list(csv.DictReader(handle))
    print(json.dumps(verify(json.loads(args.data.read_text()), cases)))


if __name__ == "__main__":
    main()
