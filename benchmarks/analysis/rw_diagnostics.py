"""Saved-bank operator contrasts and eligibility-aware selector diagnostics."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from benchmarks.analysis.baseline_uncertainty import paired_interval
from benchmarks.analysis.evidence import COUNTS, require


def derive(root, study, before):
    out = root / "benchmarks/results/paper/evidence"
    with (before / "selected_outcomes.csv").open() as handle:
        historical = {
            (r["dataset"], int(r["repeat"]), r["id"]): r
            for r in csv.DictReader(handle)
            if r["stage"] == "refined" and r["policy"] == "filtered"
        }
    with (root / "benchmarks/results/paper/selected_outcomes.csv").open() as handle:
        outcomes = {
            (r["dataset"], int(r["repeat"]), r["id"], r["stage"]): r
            for r in csv.DictReader(handle)
            if r["policy"] == "filtered"
        }
    contrasts, bottlenecks = [], []
    states = [
        "generation_failure",
        "chirality_exclusion",
        "ranking_failure",
        "pb_failure",
        "success",
    ]
    for ds, count in COUNTS.items():
        current, previous = {}, {}
        counts = {s: [] for s in ("raw", "refined")}
        regrets = {s: [] for s in ("raw", "refined")}
        ids = None
        for rep in range(3):
            rows = json.loads(
                (study / "post/full" / f"rw_{ds}_repeat_{rep}/records.json").read_text()
            )
            group = {r["id"] for r in rows}
            require(len(group) == count and (ids is None or ids == group), "Rw cohort drift")
            ids = group
            for stage in ("raw", "refined"):
                per = Counter()
                regret = []
                for r in (x for x in rows if x["stage"] == stage):
                    values = np.asarray(r["symmetry_rmsd"])
                    mask = np.asarray(r["chirality_valid"], bool)
                    effective = mask if mask.any() else np.ones(100, bool)
                    full_hit = bool((values < 2).any())
                    eligible_hit = bool((values[effective] < 2).any())
                    chosen = r["selected"]
                    require(effective[chosen], "Selected pose outside effective candidate set")
                    q = outcomes[ds, rep, r["id"], stage]
                    hit, joint = q["rmsd_lt2"] == "True", q["joint"] == "True"
                    require(hit == (values[chosen] < 2), "Outcome drift")
                    label = (
                        "generation_failure"
                        if not full_hit
                        else "chirality_exclusion"
                        if not eligible_hit
                        else "ranking_failure"
                        if not hit
                        else "success"
                        if joint
                        else "pb_failure"
                    )
                    per[label] += 1
                    regret.append(float(values[chosen] - values[effective].min()))
                    if stage == "refined":
                        current[rep, r["id"]] = q
                require(
                    sum(per.values()) == count and min(regret) >= -1e-10,
                    "Invalid diagnostic partition",
                )
                counts[stage].append([100 * per[s] / count for s in states])
                regrets[stage].append(regret)
            if ds in ("astex", "posebusters"):
                files = sorted(
                    (study / "post/pb_full" / f"legacy_rt_w_{ds}_repeat_{rep}").glob(
                        "shard-*/results.json"
                    )
                )
                require(len(files) == 8, "Missing paired legacy PB")
                for f in files:
                    for r in json.loads(f.read_text())["rows"]:
                        if r["stage"] == "refined" and r["policy"] == "filtered":
                            key = rep, r["id"]
                            require(key not in previous, "Duplicate legacy outcome")
                            previous[key] = dict(
                                rmsd_lt2=r["rmsd"] < 2, joint=r["rmsd"] < 2 and r["pb_valid"]
                            )
            else:
                previous.update(
                    {
                        (rep, i): {
                            k: historical[ds, rep, i][k] == "True" for k in ("rmsd_lt2", "joint")
                        }
                        for i in ids
                    }
                )
        require(
            current.keys() == previous.keys() and len(current) == count * 3,
            "Unpaired operator contrast",
        )
        for metric in ("rmsd_lt2", "joint"):
            ordered = sorted(ids)
            vals = [
                float(
                    np.mean(
                        [
                            int(current[rep, i][metric] == "True") - int(previous[rep, i][metric])
                            for rep in range(3)
                        ]
                    )
                )
                for i in ordered
            ]
            contrasts.append(
                dict(
                    dataset=ds,
                    metric=metric,
                    reference="paired_rerun"
                    if ds in ("astex", "posebusters")
                    else "historical_bank",
                    estimate=paired_interval(vals, list(range(count)), seed=20261002),
                    previous_mean=100
                    * np.mean([previous[rep, i][metric] for rep in range(3) for i in ordered]),
                    rw_mean=100
                    * np.mean(
                        [current[rep, i][metric] == "True" for rep in range(3) for i in ordered]
                    ),
                )
            )
        for stage in ("raw", "refined"):
            a = np.asarray(counts[stage])
            flat = np.concatenate(regrets[stage])
            grid = np.unique(
                np.r_[np.linspace(0, 5, 401), np.linspace(5, max(5.0, float(flat.max())), 101)]
            )
            cdf = np.array(
                [[100 * np.mean(np.array(v) <= t) for t in grid] for v in regrets[stage]]
            )
            bottlenecks.append(
                dict(
                    dataset=ds,
                    stage=stage,
                    n=count,
                    states=states,
                    mean=a.mean(axis=0).tolist(),
                    sd=a.std(axis=0, ddof=1).tolist(),
                    per_repeat=a.tolist(),
                    effective_regret_median_per_repeat=[
                        float(np.median(v)) for v in regrets[stage]
                    ],
                    effective_regret_p90_per_repeat=[
                        float(np.quantile(v, 0.9)) for v in regrets[stage]
                    ],
                    cdf=dict(
                        rmsd_regret_angstrom=grid.tolist(),
                        mean=cdf.mean(axis=0).tolist(),
                        sd=cdf.std(axis=0, ddof=1).tolist(),
                        per_repeat=cdf.tolist(),
                    ),
                )
            )
    for filename, payload in (
        (
            "rw_operator_change.json",
            dict(
                comparisons=contrasts,
                scope="Paired complex bootstrap of three-seed average differences; fixed cohort, seeds kept together; fresh matched priors only for Astex/PoseBusters; temporal differences include historical execution variation; no multiple-testing adjustment.",
            ),
        ),
        (
            "selector_bottleneck.json",
            dict(
                rows=bottlenecks,
                orientation_injection="rw",
                scope="Effective chirality eligibility includes the frozen unfiltered fallback; PB labels refer to selected poses only.",
            ),
        ),
    ):
        (out / filename).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(
        "Derived operator-change CIs and eligibility-aware failure/regret diagnostics", flush=True
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--before", type=Path, required=True)
    args = p.parse_args()
    derive(Path(__file__).resolve().parents[2], args.study, args.before)
