"""Saved-bank candidate analysis; implementation based on GPT-5.6-terra audit."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "outputs/benchmarks"
ARMS = {
    "unguided_n100_s10": (
        B / "astex_pb_unguided_r3_hostmatched_v2/n100_s10/selection/full",
        {"astex": 85, "posebusters": 308},
        100,
    ),
    "unguided_n40_s25": (
        B / "astex_pb_unguided_r3_hostmatched_v2/n40_s25/selection/full",
        {"astex": 85, "posebusters": 308},
        40,
    ),
    "guided_n100_s10": (
        B / "external_chirality_u70k_three_seed_v1/full",
        {"astex": 85, "posebusters": 308},
        100,
    ),
    "guided_n40_s25": (
        B / "astex_pb_n40_s25_r3_v1/selection/full",
        {"astex": 85, "posebusters": 308},
        40,
    ),
    "temporal_n100_s10": (
        B / "external_chirality_u70k_temporal_full_r3_v1/full",
        {"phibench": 206, "foldbench": 558, "openbind": 925},
        100,
    ),
}


def candidate_case(record, n):
    scores, rmsd, mask = (record[k] for k in ("predicted_rmsd", "symmetry_rmsd", "chirality_valid"))
    if not all(isinstance(v, list) and len(v) == n for v in (scores, rmsd, mask)):
        raise ValueError("candidate count mismatch")
    if not all(
        isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x)
        for x in scores + rmsd
    ):
        raise ValueError("nonfinite or nonnumeric vector")
    if min(rmsd) < 0 or not all(type(x) is bool for x in mask):
        raise ValueError("negative RMSD or invalid mask")
    order = sorted(range(n), key=lambda i: (scores[i], i))
    passing = [i for i in order if mask[i]]
    chosen = passing[0] if passing else order[0]
    if (
        record["baseline"] != order[0]
        or record["selected"] != chosen
        or record["fallback"] != bool(not passing)
    ):
        raise ValueError("ranking or fallback mismatch")
    if record["valid_count"] != sum(mask):
        raise ValueError("chirality count mismatch")
    for key, i in [("baseline_rmsd", order[0]), ("selected_rmsd", chosen)]:
        if not math.isclose(record[key], rmsd[i], abs_tol=1e-12, rel_tol=0):
            raise ValueError("indexed RMSD mismatch")
    hit = [x < 2 for x in rmsd]
    return dict(
        id=record["id"],
        stage=record["stage"],
        candidate_count=n,
        top1_lt2=hit[order[0]],
        top5_lt2=any(hit[i] for i in order[:5]),
        oracle_lt2=any(hit),
        filtered_top1_lt2=hit[chosen],
        candidate_lt2_count=sum(hit),
        candidate_lt2=sum(hit) / n,
        chirality_valid=sum(mask) / n,
        top1_rmsd=rmsd[order[0]],
        oracle_rmsd=min(rmsd),
        chirality_fallback=record["fallback"],
    )


def aggregate(cases, count):
    if len(cases) != count or len({r["id"] for r in cases}) != count:
        raise ValueError("cohort count/identity mismatch")
    metrics = (
        "top1_lt2",
        "top5_lt2",
        "oracle_lt2",
        "filtered_top1_lt2",
        "candidate_lt2",
        "chirality_valid",
    )
    values = {k: 100 * statistics.mean(r[k] for r in cases) for k in metrics}
    assert values["top1_lt2"] <= values["top5_lt2"] <= values["oracle_lt2"]
    values["candidate_lt2_pooled"] = (
        100
        * sum(r["candidate_lt2_count"] for r in cases)
        / sum(r["candidate_count"] for r in cases)
    )
    return values


def collect():
    rows = []
    cases = []
    sources = []
    for arm, (root, datasets, n) in ARMS.items():
        for dataset, count in datasets.items():
            repeats = {s: [] for s in ("raw", "refined")}
            id_sets = []
            for repeat in range(3):
                path = root / f"{dataset}_repeat_{repeat}/records.json"
                data = path.read_bytes()
                records = json.loads(data)
                verified_scores = {}
                for record in records:
                    summary_path = Path(record["confidence_summary"])
                    if summary_path not in verified_scores:
                        summary_bytes = summary_path.read_bytes()
                        if (
                            hashlib.sha256(summary_bytes).hexdigest()
                            != record["confidence_summary_sha256"]
                        ):
                            raise ValueError("confidence summary hash drift")
                        artifact = json.loads(summary_bytes)["artifacts"]["scores_csv"]
                        scores_bytes = Path(artifact["path"]).read_bytes()
                        if (
                            hashlib.sha256(scores_bytes).hexdigest() != record["scores_sha256"]
                            or artifact["sha256"] != record["scores_sha256"]
                        ):
                            raise ValueError("scores CSV hash drift")
                        table = list(csv.DictReader(scores_bytes.decode().splitlines()))
                        if [int(r["pose_index"]) for r in table] != list(range(n)):
                            raise ValueError("score CSV candidate index drift")
                        verified_scores[summary_path] = table
                    table = verified_scores[summary_path]
                    score_col, rmsd_col = (
                        ("before_confidence_rmsd", "initial_symmetry_rmsd_angstrom")
                        if record["stage"] == "raw"
                        else ("after_confidence_rmsd", "final_symmetry_rmsd_angstrom")
                    )
                    if [float(r[score_col]) for r in table] != record["predicted_rmsd"] or [
                        float(r[rmsd_col]) for r in table
                    ] != record["symmetry_rmsd"]:
                        raise ValueError("cached candidate vectors differ from scores CSV")
                if (
                    len(records) != count * 2
                    or len({(r["id"], r["stage"]) for r in records}) != count * 2
                ):
                    raise ValueError(f"incomplete or duplicated ledger {path}")
                sources.append(
                    dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(data).hexdigest())
                )
                stage_ids = []
                for stage in ("raw", "refined"):
                    values = [candidate_case(r, n) for r in records if r["stage"] == stage]
                    stage_ids.append({r["id"] for r in values})
                    repeats[stage].append(aggregate(values, count))
                    cases.extend(dict(arm=arm, dataset=dataset, repeat=repeat, **r) for r in values)
                if stage_ids[0] != stage_ids[1]:
                    raise ValueError("raw/refined IDs differ")
                id_sets.append(stage_ids[0])
            if not all(s == id_sets[0] for s in id_sets):
                raise ValueError("repeat IDs differ")
            for stage, reps in repeats.items():
                metrics = {
                    k: dict(
                        mean=statistics.mean(r[k] for r in reps),
                        sd=statistics.stdev(r[k] for r in reps),
                        per_repeat=[r[k] for r in reps],
                    )
                    for k in reps[0]
                }
                rows.append(
                    dict(
                        arm=arm,
                        dataset=dataset,
                        stage=stage,
                        complexes_per_repeat=count,
                        n=n,
                        **metrics,
                    )
                )
    return dict(
        rows=rows,
        sources=sources,
        case_row_count=len(cases),
        threshold="symmetry-aware no-alignment RMSD strictly <2 Angstrom",
        vector_source="saved selection ledgers; every underlying confidence summary and scores.csv hash checked and both vectors reproduced",
        official_candidate_pb=None,
        official_candidate_pb_reason="Only baseline/filtered selected poses evaluated; no PB Top5/oracle or candidate fraction inferred",
    ), cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/paper/20260919")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result, cases = collect()
    (args.output / "candidate_metrics.json").write_text(json.dumps(result, indent=2))
    with (args.output / "candidate_cases.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    print(json.dumps({"rows": len(result["rows"]), "cases": len(cases), "status": "complete"}))


if __name__ == "__main__":
    main()
