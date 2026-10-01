"""Refresh manuscript inputs from a completed, immutable Rw study.

Only saved candidates and selected-pose PB labels are used. Historical guidance,
budget, pocket/prior and illustration data retain their own operator provenance.
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks.analysis.baseline_uncertainty import paired_interval
from benchmarks.analysis.evidence import COUNTS, collect, require, sample_cases, write_csv

CUTS = dict(heavy_atoms=(0, 20, 30, 40), rotatable_bonds=(0, 5, 10, 15), fragments=(1, 3, 5, 7))


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def stats(values):
    a = np.asarray(values, float)
    require(a.shape[0] == 3 and np.isfinite(a).all(), "Expected three finite repeats")
    return dict(
        mean=a.mean(axis=0).tolist(), sd=a.std(axis=0, ddof=1).tolist(), per_repeat=a.tolist()
    )


def candidate(record):
    from benchmarks.analysis.evidence import selected_indices

    p = np.asarray(record["predicted_rmsd"])
    r = np.asarray(record["symmetry_rmsd"])
    mask = np.asarray(record["chirality_valid"], bool)
    require(np.isfinite(p).all() and np.isfinite(r).all() and (r >= 0).all(), "Invalid vector")
    baseline, selected = selected_indices(record)
    order = np.array(sorted(range(100), key=lambda i: (p[i], i)))
    hit = r < 2
    prefix, filtered, fallback = [], [], []
    best, eligible = 0, None
    for i in range(100):
        if (p[i], i) < (p[best], best):
            best = i
        if mask[i] and (eligible is None or (p[i], i) < (p[eligible], eligible)):
            eligible = i
        prefix.append(hit[best])
        filtered.append(hit[eligible if eligible is not None else best])
        fallback.append(eligible is None)
    metrics = dict(
        top1_lt2=bool(hit[baseline]),
        top5_lt2=bool(hit[order[:5]].any()),
        oracle_lt2=bool(hit.any()),
        filtered_top1_lt2=bool(hit[selected]),
        candidate_lt2=float(hit.mean()),
        chirality_valid=float(mask.mean()),
        candidate_lt2_pooled=float(hit.mean()),
    )
    curves = dict(
        prefix_top1=prefix,
        prefix_filtered_top1=filtered,
        prefix_fallback=fallback,
        prefix_oracle=np.maximum.accumulate(hit),
        ranked_topk=np.maximum.accumulate(hit[order]),
    )
    return metrics, curves


def refresh(root, study, before):
    import torch

    data = root / "benchmarks/results/paper"
    origin = study.resolve().parents[2]
    sources = {}

    def read(path, expected=None):
        path = Path(path)
        if not path.is_absolute():
            path = root / path
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        require(expected is None or digest == expected, f"Changed input: {path}")
        rel = path.relative_to(root) if path.is_relative_to(root) else path.relative_to(origin)
        sources[str(rel)] = digest
        return json.loads(payload)

    report = read(study / "three_seed_summary.json")
    inventory = read(study / "inventory.json")
    require(report["status"] == "complete", "Incomplete Rw campaign")
    require(report["seeds"] == [42, 100042, 200042], "Seed drift")
    require({c["dataset"]: c["count"] for c in inventory["cohorts"]} == COUNTS, "Cohort drift")
    old = json.loads((before / "benchmark_results.json").read_text())
    public = json.loads((before / "figure_data.json").read_text())
    overlap = json.loads((before / "overlap_summary.json").read_text())
    annotations = {}
    for row in read_csv(before / "selected_outcomes.csv"):
        key = row["dataset"], row["id"]
        item = {k: row[k] for k in ("heavy_atoms", "rotatable_bonds", "pdb")}
        require(key not in annotations or annotations[key] == item, "Descriptor drift")
        annotations[key] = item
    related = {(r["dataset"], r["id"]): r for r in read_csv(before / "evidence/relatedness.csv")}
    rows, pb, private, candidate_rows, candidate_sources, outcomes = [], {}, {}, [], [], []
    curves, descriptors = [], {}
    for cohort in inventory["cohorts"]:
        ds, count = cohort["dataset"], cohort["count"]
        expected = {r["id"] for r in cohort["cases"]}
        candidate_repeats, curve_repeats = defaultdict(list), defaultdict(list)
        for rep in range(3):
            path = study / "post/full" / f"rw_{ds}_repeat_{rep}" / "records.json"
            records = read(path)
            require(len(records) == 2 * count, "Incomplete selection ledger")
            require(
                {(r["id"], r["stage"]) for r in records}
                == {(i, s) for i in expected for s in ("raw", "refined")},
                "Ledger identity drift",
            )
            candidate_sources.append(
                dict(path=str(path.relative_to(root)), sha256=sources[str(path.relative_to(root))])
            )
            profile = "pb_full" if ds in ("astex", "posebusters") else "pb_inchi_compat_v1"
            paths = sorted(
                (study / "post" / profile / f"rw_{ds}_repeat_{rep}").glob("shard-*/results.json")
            )
            require(len(paths) == 8, "Missing PB shards")
            for path_pb in paths:
                table = read(path_pb)
                require(
                    table["selection_sha256"] == candidate_sources[-1]["sha256"],
                    "PB ledger mismatch",
                )
                require(table["version"] == "0.6.5", "PB version drift")
                for row in table["rows"]:
                    key = ds, rep, row["id"], row["stage"], row["policy"]
                    require(key not in pb, "Duplicate PB outcome")
                    checks = {k: v for k, v in row["checks"].items() if not k.startswith("rmsd_")}
                    require(
                        len(checks) == 27 and all(type(v) is bool for v in checks.values()),
                        "PB schema drift",
                    )
                    require(all(checks.values()) == row["pb_valid"], "PB conjunction drift")
                    pb[key] = row
            for stage in ("raw", "refined"):
                metrics, sums = [], defaultdict(lambda: np.zeros(100))
                for record in (r for r in records if r["stage"] == stage):
                    ident = record["id"]
                    private[ds, rep, ident, stage] = record
                    metric, curve = candidate(record)
                    metrics.append(metric)
                    for k, v in curve.items():
                        sums[k] += v
                    for policy, selected in (
                        ("baseline", record["baseline"]),
                        ("filtered", record["selected"]),
                    ):
                        q = pb[ds, rep, ident, stage, policy]
                        require(
                            q["pose_index"] == selected
                            and abs(q["rmsd"] - record["symmetry_rmsd"][selected]) < 1e-10,
                            "PB pose mismatch",
                        )
                        hit, valid, oracle = q["rmsd"] < 2, q["pb_valid"], metric["oracle_lt2"]
                        state = (
                            "coverage_failure"
                            if not oracle
                            else "selection_failure"
                            if not hit
                            else "success"
                            if valid
                            else "physical_failure"
                        )
                        outcomes.append(
                            dict(
                                dataset=ds,
                                repeat=rep,
                                id=ident,
                                stage=stage,
                                policy=policy,
                                state=state,
                                rmsd_lt2=hit,
                                joint=hit and valid,
                                pb_valid=valid,
                                **annotations[ds, ident],
                            )
                        )
                    if stage == "refined":
                        conf = read(
                            record["confidence_summary"], record["confidence_summary_sha256"]
                        )
                        require(conf["orientation_injection"] == "rw", "Wrong confidence operator")
                        ref = read(
                            conf["inputs"]["refinement_summary"],
                            conf["inputs"]["refinement_summary_sha256"],
                        )
                        a = ref["artifacts"]["trajectory_pt"]
                        saved = torch.load(
                            a["path"], map_location="cpu", weights_only=True, mmap=True
                        )
                        ids = saved["fragment_id"]
                        heavy = int(annotations[ds, ident]["heavy_atoms"])
                        require(
                            ids.shape == (heavy,) and ids.dtype == torch.int64,
                            "Fragment shape drift",
                        )
                        unique = torch.unique(ids, sorted=True)
                        require(
                            torch.equal(unique, torch.arange(len(unique))),
                            "Noncontiguous fragments",
                        )
                        require(
                            saved["frames_pocket_centered"].shape[1:] == (100, heavy, 3),
                            "Pose trajectory shape drift",
                        )
                        descriptor = dict(
                            heavy_atoms=heavy,
                            rotatable_bonds=int(annotations[ds, ident]["rotatable_bonds"]),
                            fragments=len(unique),
                        )
                        require(
                            (ds, ident) not in descriptors or descriptors[ds, ident] == descriptor,
                            "Repeat descriptor drift",
                        )
                        descriptors[ds, ident] = descriptor
                for k in metrics[0]:
                    candidate_repeats[stage, k].append(100 * np.mean([r[k] for r in metrics]))
                for k, v in sums.items():
                    curve_repeats[stage, k].append(100 * v / count)
            print(f"Checked Rw {ds} repeat {rep}: {count} complexes", flush=True)
        for stage in ("raw", "refined"):
            candidate_rows.append(
                dict(
                    arm="rw_n100_s10",
                    dataset=ds,
                    stage=stage,
                    complexes_per_repeat=count,
                    n=100,
                    **{k: stats(candidate_repeats[stage, k]) for k in metrics[0]},
                )
            )
            curves.append(
                dict(
                    dataset=ds,
                    stage=stage,
                    complexes=count,
                    n=list(range(1, 101)),
                    **{k: stats(curve_repeats[stage, k]) for k in sums},
                )
            )
            for policy in ("baseline", "filtered"):
                values = report["results"]["rw_" + ds][stage + "_" + policy]
                rows.append(
                    dict(
                        dataset=ds,
                        n=100,
                        steps=10,
                        guidance="off",
                        stage=stage,
                        policy=policy,
                        orientation_injection="rw",
                        complexes_per_repeat=count,
                        **{
                            k: stats(values[v]["per_repeat_percent"])
                            for k, v in (
                                ("rmsd_lt2", "rmsd_success"),
                                ("pb_valid", "pb_valid"),
                                ("joint", "pb_valid_success"),
                            )
                        },
                    )
                )
    require(len(outcomes) == 24984 and len(descriptors) == 2082, "Incomplete refresh")
    write_csv(data / "selected_outcomes.csv", outcomes)
    write_csv(
        data / "complexity_descriptors.csv",
        [dict(dataset=ds, id=i, **v) for (ds, i), v in sorted(descriptors.items())],
    )
    new = dict(
        rows=rows,
        legacy_ablation_rows=old["rows"],
        robustness=old["robustness"],
        sources=[dict(path=k, sha256=v) for k, v in sources.items()],
        orientation_injection="rw",
    )
    write_json(data / "benchmark_results.json", new)
    write_json(
        data / "candidate_metrics.json",
        dict(
            rows=candidate_rows,
            sources=candidate_sources,
            case_row_count=12492,
            orientation_injection="rw",
            threshold="RMSD strictly <2 Angstrom",
            official_candidate_pb=None,
            official_candidate_pb_reason="PB evaluated only for baseline and filtered selected poses",
        ),
    )
    index = {(r["dataset"], r["repeat"], r["id"], r["stage"], r["policy"]): r for r in outcomes}
    complexity = []
    for ds in COUNTS:
        for field, cuts in CUTS.items():
            for lower, upper in zip(cuts, (*cuts[1:], None), strict=True):
                ids = sorted(
                    i
                    for (d, i), v in descriptors.items()
                    if d == ds and v[field] >= lower and (upper is None or v[field] < upper)
                )
                row = dict(
                    dataset=ds, descriptor=field, lower=lower, upper=upper, complexes=len(ids)
                )
                for metric, label in (
                    ("rmsd_lt2", "top1"),
                    ("joint", "joint"),
                    ("oracle", "oracle"),
                ):
                    values = (
                        [
                            100
                            * np.mean(
                                [
                                    min(private[ds, rep, i, "refined"]["symmetry_rmsd"]) < 2
                                    if metric == "oracle"
                                    else index[ds, rep, i, "refined", "filtered"][metric]
                                    for i in ids
                                ]
                            )
                            for rep in range(3)
                        ]
                        if ids
                        else None
                    )
                    row[label] = stats(values) if values else None
                complexity.append(row)
    public["complexity_budget"] = dict(
        complexity=complexity,
        curves=curves,
        fragment_case_rows=6246,
        fragment_source="Saved Rw refinement trajectory fragment_id; all three repeats checked",
        prefix_pb="Not evaluated; prefix success is RMSD-only",
        orientation_injection="rw",
    )
    failures = []
    for r in complexity:
        base = {k: r[k] for k in ("dataset", "descriptor", "lower", "upper", "complexes")}
        o, t, j = [r[k]["mean"] if r[k] else None for k in ("oracle", "top1", "joint")]
        failures.append(
            dict(
                base,
                coverage_failure=100 - o if o is not None else None,
                selection_failure=o - t if o is not None else None,
                physical_failure=t - j if o is not None else None,
                success=j,
                oracle_conditional_selection_percent=100 * t / o if o else None,
            )
        )
    public["complexity_failures"] = dict(rows=failures, orientation_injection="rw")
    for row in public["comparison"]["rows"]:
        if row["method"] == "EFF-Dock":
            main = next(
                r
                for r in rows
                if r["dataset"] == row["dataset"]
                and r["stage"] == "refined"
                and r["policy"] == "filtered"
            )
            for k in ("rmsd_lt2", "joint"):
                row[k] = main[k]
            row["orientation_injection"] = "rw"
            row["source"] = [str((study / "three_seed_summary.json").relative_to(root))]
    sequence = public["sequence"]["records"]
    perf = []
    for row in public["sequence_performance"]:
        ds, b = row["dataset"], row["bin"]
        bins = ["<30", "30–<70", "70–<90", "90–100"]
        ids = [
            r["id"]
            for r in sequence
            if r["dataset"] == ds
            and bins[int(np.searchsorted([30, 70, 90], r["max_sequence_identity"], side="right"))]
            == b
        ]
        require(len(ids) == row["n"] and ids, "Sequence group drift")
        val = stats(
            [
                [
                    100 * np.mean([index[ds, rep, i, "refined", "filtered"][k] for i in ids])
                    for k in ("rmsd_lt2", "joint")
                ]
                for rep in range(3)
            ]
        )
        perf.append(dict(dataset=ds, bin=b, n=len(ids), **val))
    public["sequence_performance"] = perf
    overlap_rows, overlap_reps = [], []
    for row in overlap["rows"]:
        if row["axis"] == "exact_train_pdb":
            continue
        ds, axis, stratum = row["dataset"], row["axis"], row["stratum"]
        ids = []
        for (d, i), q in related.items():
            if d != ds:
                continue
            exact, t = q["exact_ligand"] == "True", float(q["ligand_tanimoto"])
            group = (
                ("observed_overlap" if exact else "no_observed_match")
                if axis == "exact_train_ligand"
                else (
                    "exact_identity"
                    if exact
                    else "no_observed_exact_>=0.8"
                    if t >= 0.8
                    else "no_observed_exact_0.5_to_<0.8"
                    if t >= 0.5
                    else "no_observed_exact_<0.5"
                )
            )
            if group == stratum:
                ids.append(i)
        require(len(ids) == row["count_per_repeat"], "Ligand group drift")
        item = {
            k: row[k] for k in ("dataset", "stage", "policy", "axis", "stratum", "count_per_repeat")
        }
        for k in ("rmsd_lt2", "pb_valid", "joint"):
            item[k] = (
                stats(
                    [
                        100
                        * np.mean([index[ds, rep, i, row["stage"], row["policy"]][k] for i in ids])
                        for rep in range(3)
                    ]
                )
                if ids
                else None
            )
        overlap_rows.append(item)
        for rep in range(3):
            overlap_reps.append(
                dict(
                    dataset=ds,
                    repeat=rep,
                    stage=row["stage"],
                    policy=row["policy"],
                    axis=axis,
                    stratum=stratum,
                    count=len(ids),
                    **{
                        k: item[k]["per_repeat"][rep] if item[k] else None
                        for k in ("rmsd_lt2", "pb_valid", "joint")
                    },
                )
            )
    overlap.update(
        rows=overlap_rows,
        per_repeat=overlap_reps,
        orientation_injection="rw",
        pb_sources=[
            dict(path=k, sha256=v) for k, v in sources.items() if k.endswith("results.json")
        ],
    )
    write_json(data / "overlap_summary.json", overlap)
    contrasts = []
    specs = dict(
        refinement_baseline=(("raw", "baseline"), ("refined", "baseline")),
        refinement_filtered=(("raw", "filtered"), ("refined", "filtered")),
        chirality_raw=(("raw", "baseline"), ("raw", "filtered")),
        chirality_refined=(("refined", "baseline"), ("refined", "filtered")),
    )
    for ds in COUNTS:
        ids = sorted(i for d, i in descriptors if d == ds)
        for contrast, (a, b) in specs.items():
            for metric in ("rmsd_lt2", "joint"):
                delta = np.array(
                    [
                        [
                            int(index[ds, rep, i, *b][metric]) - int(index[ds, rep, i, *a][metric])
                            for rep in range(3)
                        ]
                        for i in ids
                    ]
                )
                means = delta.mean(axis=1).tolist()
                pdbs = [annotations[ds, i]["pdb"] for i in ids]
                contrasts.append(
                    dict(
                        dataset=ds,
                        contrast=contrast,
                        metric=metric,
                        complex_bootstrap=paired_interval(
                            means, list(range(len(ids))), seed=20260920
                        ),
                        exact_pdb_bootstrap=paired_interval(means, pdbs, seed=20260920)
                        if len(set(pdbs)) > 1
                        else None,
                        missing_pdb=0,
                        gains=int((delta > 0).sum()),
                        losses=int((delta < 0).sum()),
                    )
                )
    public["uncertainty"] = dict(
        contrasts=contrasts, bootstrap_seed=20260920, orientation_injection="rw"
    )
    public["provenance"]["rw_refresh"] = dict(
        protocol=report["protocol"],
        orientation_injection="rw",
        seeds=report["seeds"],
        legacy_figures=[
            "06_guidance_budget",
            "07_runtime_memory",
            "08_pocket_prior",
            "S10_fragment_trajectory",
        ],
        sources=sources,
        baseline_note="Fresh paired R^T w controls for Astex/PoseBusters; historical comparisons for temporal cohorts",
    )
    write_json(data / "figure_data.json", public)
    evidence, cases, records, checks = collect(root, data / "evidence", study=study)
    sample_cases(root, data / "evidence", cases, records, checks, evidence, study=study)
    # Native-method outcome rows are unchanged; their pairing now uses Rw outcomes.
    baseline_path = data / "evidence/baseline_cases.csv"
    base_rows = read_csv(before / "evidence/baseline_cases.csv")
    for row in base_rows:
        ours = index[row["dataset"], int(row["repeat"]), row["id"], "refined", "filtered"]
        row["effdock_rmsd_lt2"], row["effdock_joint"] = ours["rmsd_lt2"], ours["joint"]
    write_csv(baseline_path, base_rows)
    groups = defaultdict(list)
    for row in base_rows:
        groups[row["dataset"], row["method"]].append(row)
    baseline = json.loads((before / "evidence/baseline_uncertainty.json").read_text())
    for r in baseline["comparisons"]:
        gs = groups[r["dataset"], r["method"]]
        by_id = defaultdict(list)
        for v in gs:
            own = v["effdock_" + r["metric"]]
            theirs = v[r["metric"]] == "True"
            by_id[v["id"]].append(int(own) - int(theirs))
        ids = sorted(by_id)
        require(all(len(v) == 3 for v in by_id.values()), "Baseline repeat mismatch")
        means = [float(np.mean(by_id[i])) for i in ids]
        r["complex_bootstrap"] = paired_interval(means, list(range(len(ids))))
        r["pdb_bootstrap"] = paired_interval(
            means, [annotations[r["dataset"], i]["pdb"] for i in ids]
        )
    baseline["orientation_injection"] = "rw"
    write_json(data / "evidence/baseline_uncertainty.json", baseline)
    print(
        "Refreshed complete Rw numerical inputs, confidence, PB transitions and structure examples",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument(
        "--before", type=Path, required=True, help="Immutable previous numerical tables"
    )
    args = parser.parse_args()
    refresh(Path(__file__).resolve().parents[2], args.study.absolute(), args.before.absolute())
