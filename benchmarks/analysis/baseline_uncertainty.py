"""Paired-complex uncertainty for the frozen, locally executed comparison rows."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks.analysis.evidence import require, write_csv


def paired_interval(values, groups, seed=20260923, draws=2000):
    require(len(values) == len(groups) and len(values) > 0, "Invalid paired cohort")
    sums, sizes = [], []
    for group in sorted(set(groups)):
        selected = [v for v, g in zip(values, groups, strict=True) if g == group]
        sums.append(sum(selected))
        sizes.append(len(selected))
    sums, sizes = np.asarray(sums), np.asarray(sizes)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        idx = rng.integers(len(sums), size=len(sums))
        samples.append(100 * sums[idx].sum() / sizes[idx].sum())
    return dict(
        delta_pp=float(100 * np.mean(values)),
        ci95_pp=np.quantile(samples, [0.025, 0.975]).tolist(),
        complexes=len(values),
        groups=len(sums),
        draws=draws,
        seed=seed,
    )


def collect(root, output):
    paper = root / "benchmarks/results/paper"
    comparison = json.loads((paper / "figure_data.json").read_text())["comparison"]["rows"]
    with (paper / "selected_outcomes.csv").open() as handle:
        own = {
            (r["dataset"], int(r["repeat"]), r["id"]): r
            for r in csv.DictReader(handle)
            if r["stage"] == "refined" and r["policy"] == "filtered"
        }
    sources, cases, summaries = {}, [], []
    for row in comparison:
        if row["source_type"] != "our_run" or row["method"] == "EFF-Dock":
            continue
        ds, method = row["dataset"], row["method"]
        by_id = defaultdict(list)
        for rep, source in enumerate(row["source"]):
            base = root / source
            files = (
                [base.parent / "per_target.csv"]
                if base.suffix == ".json"
                else sorted(base.glob("shard_*/results.csv"))
            )
            require(bool(files), "Missing baseline files")
            rs = []
            for path in files:
                sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
                with path.open() as handle:
                    rs.extend(csv.DictReader(handle))
            ids = set()
            for r in rs:
                ident = r["complex_name"].lower()
                if ds == "astex":
                    ident = ident.split("_")[0]
                require(ident not in ids, "Ambiguous baseline mapping")
                ids.add(ident)
                ours = own[ds, rep, ident]
                success = float(r["top1_rmsd"]) < 2
                joint = (
                    r["top1_joint" if method == "SigmaDock" else "joint_rmsd_lt2_pb_valid"]
                    == "True"
                )
                require(not joint or success, "Inconsistent baseline labels")
                outcome = dict(
                    dataset=ds,
                    method=method,
                    repeat=rep,
                    id=ident,
                    rmsd_lt2=success,
                    joint=joint,
                    effdock_rmsd_lt2=ours["rmsd_lt2"] == "True",
                    effdock_joint=ours["joint"] == "True",
                    pdb=ours["pdb"],
                )
                cases.append(outcome)
                by_id[ident].append(outcome)
            expected_ids = {k[2] for k in own if k[:2] == (ds, rep)}
            require(ids == expected_ids and len(ids) == row["count"], "Baseline cohort mismatch")
            group = [
                r for r in cases if (r["dataset"], r["method"], r["repeat"]) == (ds, method, rep)
            ]
            for metric in ("rmsd_lt2", "joint"):
                mean = 100 * sum(r[metric] for r in group) / len(group)
                require(
                    abs(mean - row[metric]["per_repeat"][rep]) < 1e-8, "Frozen comparison changed"
                )
        for metric in ("rmsd_lt2", "joint"):
            values, pdbs = [], []
            for ident, rs in sorted(by_id.items()):
                require(len(rs) == 3, "Missing repeat")
                values.append(np.mean([int(r["effdock_" + metric]) - int(r[metric]) for r in rs]))
                pdbs.append(rs[0]["pdb"])
            summaries.append(
                dict(
                    dataset=ds,
                    method=method,
                    metric=metric,
                    complex_bootstrap=paired_interval(values, list(range(len(values)))),
                    pdb_bootstrap=paired_interval(values, pdbs),
                )
            )
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "baseline_cases.csv", cases)
    result = dict(
        comparisons=summaries,
        sources=sources,
        scope="Fixed published native-method conditions; three-repeat means paired by complex, not equal budgets or paired random seeds; percentile CIs without multiplicity adjustment.",
    )
    (output / "baseline_uncertainty.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Checked {len(cases)} baseline records and {len(summaries)} paired contrasts", flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(args.root.resolve(), args.output)
