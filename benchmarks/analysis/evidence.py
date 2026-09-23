"""Collect empirical manuscript evidence from frozen, hash-checked pose banks."""

import argparse
import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

COUNTS = dict(astex=85, posebusters=308, phibench=206, foldbench=558, openbind=925)
BANKS = {
    "astex": "astex_pb_unguided_r3_hostmatched_v2/n100_s10/selection/full",
    "posebusters": "astex_pb_unguided_r3_hostmatched_v2/n100_s10/selection/full",
    **dict.fromkeys(
        ("phibench", "foldbench", "openbind"), "external_chirality_u70k_temporal_full_r3_v1/full"
    ),
}
GROUPS = {
    "Bond geometry": ["bond_lengths", "bond_angles"],
    "Internal clash": ["internal_steric_clash"],
    "Receptor clash": ["minimum_distance_to_protein", "volume_overlap_with_protein"],
    "Stereochemistry": ["tetrahedral_chirality", "double_bond_stereochemistry"],
    "Ring planarity": ["aromatic_ring_flatness", "double_bond_flatness"],
    "Internal energy": ["internal_energy"],
}
DENSITY_LABELS = ["0", "1–5", "6–20", "21–50", "51–100"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ranking_metrics(prediction, rmsd):
    from scipy.stats import rankdata

    p, r = np.asarray(prediction, float), np.asarray(rmsd, float)
    require(p.ndim == 1 and p.shape == r.shape and len(p) > 0, "Score shape mismatch")
    require(np.isfinite(p).all() and np.isfinite(r).all() and (r >= 0).all(), "Invalid scores")
    pr, rr = rankdata(p), rankdata(r)
    rho = float(np.corrcoef(pr, rr)[0, 1]) if np.ptp(pr) and np.ptp(rr) else None
    hits = r < 2
    positives = int(hits.sum())
    if positives in (0, len(r)):
        return dict(spearman=rho, auroc=None, average_precision=None)
    score = -p
    auc = (rankdata(score)[hits].sum() - positives * (positives + 1) / 2) / (
        positives * (len(r) - positives)
    )
    order = np.argsort(-score, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(score[order])), len(score) - 1]
    tp = np.cumsum(hits[order])[ends]
    precision = tp / (ends + 1)
    recall = tp / positives
    ap = np.sum(np.diff(np.r_[0, recall]) * precision)
    return dict(spearman=rho, auroc=float(auc), average_precision=float(ap))


def selected_indices(record):
    p, r, mask = [record[k] for k in ("predicted_rmsd", "symmetry_rmsd", "chirality_valid")]
    require(len(p) == len(r) == len(mask) == 100, "Expected 100 candidates")
    require(all(type(x) is bool for x in mask), "Invalid chirality mask")
    order = sorted(range(100), key=lambda i: (p[i], i))
    passing = [i for i in order if mask[i]]
    baseline, selected = order[0], (passing or order)[0]
    require((baseline, selected) == (record["baseline"], record["selected"]), "Selector drift")
    require(record["fallback"] == (not passing), "Fallback drift")
    return baseline, selected


def stringent(ligand, sequence):
    sim = ligand["nearest_train_tanimoto"]
    require(sim is not None and np.isfinite(sim) and 0 <= sim <= 1, "Unresolved ligand similarity")
    identity = sequence["max_sequence_identity"]
    require(np.isfinite(identity) and 0 <= identity <= 100, "Unresolved sequence identity")
    return (not ligand["observed_train_ligand_identity"]) and sim < 0.5 and identity < 30


def repeat_summary(values):
    require(len(values) == 3, "Expected three repeats")
    valid = [v for v in values if v is not None]
    return dict(
        per_repeat=values,
        mean=float(np.mean(valid)) if valid else None,
        sd=float(np.std(valid, ddof=1)) if len(valid) > 1 else None,
        defined_repeats=len(valid),
    )


def write_csv(path, rows):
    require(bool(rows), f"Empty export {path}")
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def transition(before, after):
    require(len(before) == len(after), "Unpaired checks")
    b, a = np.asarray(before, bool), np.asarray(after, bool)
    return dict(
        n=len(b),
        raw_fail=int((~b).sum()),
        refined_fail=int((~a).sum()),
        rescued=int((~b & a).sum()),
        harmed=int((b & ~a).sum()),
    )


def aggregate_confidence(cases):
    out, density = [], []
    for ds in COUNTS:
        for stage in ("raw", "refined"):
            rs = [r for r in cases if r["dataset"] == ds and r["stage"] == stage]
            reps = [[r for r in rs if r["repeat"] == rep] for rep in range(3)]
            row = dict(dataset=ds, stage=stage, n=COUNTS[ds])
            for key in ("spearman", "auroc", "average_precision", "brier"):
                groups = [[r[key] for r in g if r[key] is not None] for g in reps]
                row[key] = repeat_summary([float(np.mean(g)) if g else None for g in groups])
                row[key]["eligible_per_repeat"] = [len(g) for g in groups]
            for policy in ("baseline", "filtered"):
                row[policy + "_regret"] = repeat_summary(
                    [float(np.median([r[policy + "_regret"] for r in g])) for g in reps]
                )
                eligible = [[r for r in g if r["oracle_success"]] for g in reps]
                row[policy + "_conditional_success"] = repeat_summary(
                    [
                        100 * sum(r[policy + "_success"] for r in g) / len(g) if g else None
                        for g in eligible
                    ]
                )
                row[policy + "_conditional_success"]["eligible_per_repeat"] = [
                    len(g) for g in eligible
                ]
            out.append(row)
            for i, label in enumerate(DENSITY_LABELS):
                gs = [[r for r in g if r["density_bin"] == i] for g in reps]
                density.append(
                    dict(
                        dataset=ds,
                        stage=stage,
                        bin=label,
                        counts=[len(g) for g in gs],
                        success=repeat_summary(
                            [
                                100 * sum(r["filtered_success"] for r in g) / len(g) if g else None
                                for g in gs
                            ]
                        ),
                    )
                )
    return out, density


def aggregate_physical(pb):
    all_checks = sorted(next(iter(pb.values()))["checks"])
    metrics = {**{k: [k] for k in all_checks}, **GROUPS, "All PB checks": all_checks}
    result = []
    for ds, count in COUNTS.items():
        for mode in ("primary_selection", "matched_evaluated_candidate"):
            for metric, keys in metrics.items():
                reps = []
                for rep in range(3):
                    rows = [v for k, v in pb.items() if k[0:2] == (ds, rep)]
                    indexed = {}
                    for r in rows:
                        if mode == "primary_selection" and r["policy"] != "filtered":
                            continue
                        index = (
                            (r["id"], r["pose_index"]) if mode != "primary_selection" else r["id"]
                        )
                        key = (index, r["stage"])
                        if key in indexed:
                            require(
                                indexed[key]["checks"] == r["checks"],
                                "Duplicate-pose PB disagreement",
                            )
                        indexed[key] = r
                    both = {
                        i for i, s in indexed if (i, "raw") in indexed and (i, "refined") in indexed
                    }
                    if mode == "primary_selection":
                        require(len(both) == count, "PB cohort incomplete")
                    before = [
                        all(indexed[i, "raw"]["checks"][k] for k in keys) for i in sorted(both)
                    ]
                    after = [
                        all(indexed[i, "refined"]["checks"][k] for k in keys) for i in sorted(both)
                    ]
                    reps.append(transition(before, after))
                result.append(
                    dict(
                        dataset=ds,
                        mode=mode,
                        metric=metric,
                        per_repeat=reps,
                        **{
                            k: repeat_summary(
                                [100 * r[k] / r["n"] if r["n"] else None for r in reps]
                            )
                            for k in ("raw_fail", "refined_fail", "rescued", "harmed")
                        },
                    )
                )
    return result


def collect(root, out):
    sources = {}

    def read(path, expected=None, as_csv=False):
        path = Path(path)
        if not path.is_absolute():
            path = root / path
        blob = path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        require(expected is None or digest == expected, f"Changed input: {path}")
        sources[str(path.relative_to(root))] = digest
        return list(csv.DictReader(io.StringIO(blob.decode()))) if as_csv else json.loads(blob)

    paper = "benchmarks/results/paper/"
    public = read(paper + "figure_data.json")
    seq = {(r["dataset"], r["id"]): r for r in public["sequence"]["records"]}
    ligand = {
        (r["dataset"], r["id"]): r
        for r in read("docs/paper/20260919/overlap_metrics.json")["annotations"]
    }
    old = {
        (r["dataset"], int(r["repeat"]), r["id"], r["stage"], r["policy"]): r
        for r in read(paper + "selected_outcomes.csv", as_csv=True)
    }
    manifest = read(paper + "candidate_metrics.json")
    ledger_hash = {r["path"]: r["sha256"] for r in manifest["sources"]}
    provenance = read(
        "outputs/benchmarks/external_chirality_u70k_temporal_full_r3_v1/pb_inchi_compat_v1/provenance.json"
    )
    cases, pb, related, private = [], {}, [], {}
    calibration = defaultdict(lambda: np.zeros(4))
    for ds, count in COUNTS.items():
        baseline_ids = None
        for rep in range(3):
            name = f"{ds}_repeat_{rep}"
            path = f"outputs/benchmarks/{BANKS[ds]}/{name}/records.json"
            records = read(path, ledger_hash[path])
            ids = {r["id"] for r in records}
            require(len(ids) == count and len(records) == count * 2, "Candidate cohort incomplete")
            require(len({(r["id"], r["stage"]) for r in records}) == count * 2, "Duplicate records")
            require(baseline_ids is None or ids == baseline_ids, "Repeat IDs differ")
            baseline_ids = ids
            if ds in ("astex", "posebusters"):
                pbpaths = [
                    (p, None)
                    for p in sorted(
                        (root / "outputs/benchmarks" / BANKS[ds])
                        .parent.joinpath("pb_full", name)
                        .glob("shard-*/results.json")
                    )
                ]
            else:
                pbpaths = [
                    (root / r["path"], r["sha256"])
                    for r in provenance["sources"]
                    if f"/{name}/" in r["path"]
                ]
            require(len(pbpaths) == 8, "Expected eight PB shards")
            for path_pb, digest in pbpaths:
                table = read(path_pb, digest)
                require(
                    table["selection_sha256"] == ledger_hash[path], "PB ledger identity mismatch"
                )
                for row in table["rows"]:
                    key = (ds, rep, row["id"], row["stage"], row["policy"])
                    require(key not in pb and key in old, "PB row mismatch")
                    checks = {k: v for k, v in row["checks"].items() if not k.startswith("rmsd_")}
                    require(
                        len(checks) == 27 and all(type(v) is bool for v in checks.values()),
                        "Invalid PB checks",
                    )
                    require(all(checks.values()) == row["pb_valid"], "PB conjunction mismatch")
                    require(
                        row["pb_valid"] == (old[key]["pb_valid"] == "True"), "Published PB mismatch"
                    )
                    require(
                        (row["rmsd"] < 2) == (old[key]["rmsd_lt2"] == "True"),
                        "Published RMSD mismatch",
                    )
                    pb[key] = dict(row, checks=checks)
            cache = {}
            for record in records:
                ident, stage = record["id"], record["stage"]
                baseline, selected = selected_indices(record)
                key = (ds, rep, ident, stage)
                private[key] = record
                if ident not in cache:
                    summary = read(
                        record["confidence_summary"], record["confidence_summary_sha256"]
                    )
                    artifact = summary["artifacts"]["scores_csv"]
                    require(
                        artifact["sha256"] == record["scores_sha256"], "Score identity mismatch"
                    )
                    table = read(artifact["path"], artifact["sha256"], as_csv=True)
                    require(
                        [int(r["pose_index"]) for r in table] == list(range(100)),
                        "Pose index mismatch",
                    )
                    cache[ident] = table
                table = cache[ident]
                prefix, target = ("before", "initial") if stage == "raw" else ("after", "final")
                p = np.array([float(r[f"{prefix}_confidence_rmsd"]) for r in table])
                r = np.array([float(r[f"{target}_symmetry_rmsd_angstrom"]) for r in table])
                prob = np.array([float(r[f"{prefix}_confidence_success"]) for r in table])
                require(
                    np.array_equal(p, record["predicted_rmsd"])
                    and np.array_equal(r, record["symmetry_rmsd"]),
                    "Score vectors differ",
                )
                require(
                    np.isfinite(prob).all() and (prob >= 0).all() and (prob <= 1).all(),
                    "Invalid probability",
                )
                require((p >= 0).all(), "Negative predicted RMSD")
                hits = r < 2
                chosen_pb = pb[key + ("filtered",)]
                require(chosen_pb["pose_index"] == selected, "Selected index mismatch")
                c = dict(
                    dataset=ds,
                    repeat=rep,
                    id=ident,
                    stage=stage,
                    **ranking_metrics(p, r),
                    brier=float(np.mean((prob - hits) ** 2)),
                    baseline_regret=float(r[baseline] - r.min()),
                    filtered_regret=float(r[selected] - r.min()),
                    baseline_success=bool(hits[baseline]),
                    filtered_success=bool(hits[selected]),
                    oracle_success=bool(hits.any()),
                    near_native_count=int(hits.sum()),
                    density_bin=int(np.searchsorted([0, 5, 20, 50], hits.sum(), side="left")),
                    filtered_pb_valid=chosen_pb["pb_valid"],
                    filtered_joint=bool(hits[selected] and chosen_pb["pb_valid"]),
                    selected_index=selected,
                    baseline_index=baseline,
                    oracle_index=int(np.argmin(r)),
                    selected_rmsd=float(r[selected]),
                    oracle_rmsd=float(r.min()),
                )
                cases.append(c)
                for metric, prediction, actual, bounds in (
                    ("success_probability", prob, hits, np.arange(0.1, 1, 0.1)),
                    ("predicted_rmsd", p, r, [1, 2, 3, 4, 6, 10]),
                ):
                    bins = np.searchsorted(bounds, prediction, side="right")
                    for i in np.unique(bins):
                        mask = bins == i
                        calibration[ds, stage, metric, int(i)] += [
                            int(mask.sum()),
                            float(prediction[mask].sum()),
                            float(actual[mask].sum()),
                            float(((prediction[mask] - actual[mask]) ** 2).sum()),
                        ]
            print(f"Collected {ds} repeat {rep}: {len(records)} banks", flush=True)
        for ident in sorted(baseline_ids):
            q, l = seq[ds, ident], ligand[ds, ident]
            related.append(
                dict(
                    dataset=ds,
                    id=ident,
                    sequence_identity=q["max_sequence_identity"],
                    ligand_tanimoto=l["nearest_train_tanimoto"],
                    exact_ligand=l["observed_train_ligand_identity"],
                    stringent=stringent(l, q),
                )
            )
    require(
        len(cases) == 12492 and len(pb) == 24984 and len(related) == 2082,
        "Incomplete final evidence",
    )
    confidence, density = aggregate_confidence(cases)
    subsets = []
    for ds in COUNTS:
        for label in ("All", "Stringent"):
            ids = {
                r["id"]
                for r in related
                if r["dataset"] == ds and (label == "All" or r["stringent"])
            }
            gs = [
                [
                    r
                    for r in cases
                    if r["dataset"] == ds
                    and r["repeat"] == rep
                    and r["stage"] == "refined"
                    and r["id"] in ids
                ]
                for rep in range(3)
            ]
            subsets.append(
                dict(
                    dataset=ds,
                    subset=label,
                    n=len(ids),
                    **{
                        metric: repeat_summary(
                            [100 * sum(r[key] for r in g) / len(g) if g else None for g in gs]
                        )
                        for metric, key in (
                            ("top1", "filtered_success"),
                            ("oracle", "oracle_success"),
                            ("joint", "filtered_joint"),
                        )
                    },
                )
            )
    reliability = [
        dict(
            dataset=ds,
            stage=stage,
            metric=metric,
            bin=i,
            n=int(v[0]),
            prediction=v[1] / v[0],
            observed=v[2] / v[0],
            mse=v[3] / v[0],
        )
        for (ds, stage, metric, i), v in sorted(calibration.items())
    ]
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "confidence_cases.csv", cases)
    write_csv(out / "relatedness.csv", related)
    pb_export = [
        dict(
            dataset=k[0],
            repeat=k[1],
            id=k[2],
            stage=k[3],
            policy=k[4],
            pose_index=r["pose_index"],
            **{check: int(value) for check, value in sorted(r["checks"].items())},
        )
        for k, r in sorted(pb.items())
    ]
    write_csv(out / "pb_checks.csv", pb_export)
    result = dict(
        schema="effdock.paper_evidence.v1",
        confidence=confidence,
        density=density,
        reliability=reliability,
        subsets=subsets,
        physical=aggregate_physical(pb),
        sources=sources,
        datasets=COUNTS,
    )
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    # Saved separately from publication: paths locate the few illustrative structures.
    return result, cases, private, pb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result, cases, private, pb = collect(args.root.resolve(), args.output)
    sample_cases(args.root.resolve(), args.output, cases, private, pb, result)


def binding_chain_display(pdb, reference_coordinates, cutoff=5.0):
    """Retain complete protein chains contacting the crystal ligand for display."""
    ref = np.asarray(reference_coordinates)
    records = [line for line in pdb.splitlines() if line.startswith("ATOM  ")]
    chains = set()
    for line in records:
        element = line[76:78].strip() or line[12:16].strip()[0]
        if element in ("H", "D"):
            continue
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        if np.min(np.linalg.norm(ref - xyz, axis=1)) <= cutoff:
            chains.add(line[21])
    require(bool(chains), "No ligand-contacting protein chain")
    retained = [line for line in records if line[21] in chains]
    return dict(
        pdb="\n".join(retained) + "\nEND\n",
        chains=sorted(chains),
        contact_cutoff_angstrom=cutoff,
        atom_records=len(retained),
        selection="Complete ATOM chains with a heavy atom within cutoff of a crystal ligand heavy atom; display only",
    )


def sample_cases(root, out, cases, private, pb, result):
    """Export deterministic mechanism examples in their original receptor frame."""
    from rdkit import Chem

    refined = [
        r for r in cases if r["dataset"] == "astex" and r["repeat"] == 0 and r["stage"] == "refined"
    ]
    success = [(r["selected_rmsd"], r) for r in refined if r["filtered_joint"]]
    failure = [
        (r["filtered_regret"], r)
        for r in refined
        if r["oracle_success"] and not r["filtered_success"]
    ]
    rescue = []
    for r in refined:
        raw = private["astex", 0, r["id"], "raw"]
        before = raw["symmetry_rmsd"][r["selected_index"]]
        if before >= 2 and r["filtered_joint"]:
            rescue.append((before - r["selected_rmsd"], r))
    geometries = []

    def molecule(path, index=0, expected=None):
        path = Path(path)
        if not path.is_absolute():
            path = root / path
        if expected:
            require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, "Structure changed")
        mols = list(Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False))
        require(index < len(mols) and mols[index] is not None, "Unreadable structure")
        mol = mols[index]
        heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
        mapping = {v: i for i, v in enumerate(heavy)}
        return dict(
            coordinates=mol.GetConformer().GetPositions()[heavy].tolist(),
            elements=[mol.GetAtomWithIdx(i).GetSymbol() for i in heavy],
            bonds=[
                [mapping[b.GetBeginAtomIdx()], mapping[b.GetEndAtomIdx()]]
                for b in mol.GetBonds()
                if b.GetBeginAtomIdx() in mapping and b.GetEndAtomIdx() in mapping
            ],
        )

    for title, candidates in (
        ("Successful pose", success),
        ("Refinement rescue", rescue),
        ("Selection failure", failure),
    ):
        require(bool(candidates), f"No eligible example for {title}")
        ordered = sorted(candidates, key=lambda x: (x[0], x[1]["id"]))
        effect, row = ordered[(len(ordered) - 1) // 2]
        rec = private["astex", 0, row["id"], "refined"]
        raw = private["astex", 0, row["id"], "raw"]
        conf = json.loads(Path(rec["confidence_summary"]).read_text())
        ref = json.loads(Path(conf["inputs"]["refinement_summary"]).read_text())
        inputs = ref["inputs"]
        reference = molecule(inputs["ligand_reference"], expected=inputs["ligand_reference_sha256"])
        chosen = molecule(rec["bank"], row["selected_index"], rec["bank_sha256"])
        comparator = None
        comparator_rmsd = None
        if title == "Refinement rescue":
            comparator = molecule(raw["bank"], row["selected_index"], raw["bank_sha256"])
            comparator_rmsd = raw["symmetry_rmsd"][row["selected_index"]]
        elif title == "Selection failure":
            comparator = molecule(rec["bank"], row["oracle_index"], rec["bank_sha256"])
            comparator_rmsd = row["oracle_rmsd"]
        protein = root / inputs["protein"]
        require(
            hashlib.sha256(protein.read_bytes()).hexdigest() == inputs["protein_sha256"],
            "Receptor changed",
        )
        geometries.append(
            dict(
                title=title,
                dataset="astex",
                id=row["id"],
                repeat=0,
                eligible_cases=len(ordered),
                median_order=(len(ordered) - 1) // 2,
                effect=effect,
                selected_rmsd=row["selected_rmsd"],
                comparator_rmsd=comparator_rmsd,
                selected_index=row["selected_index"],
                oracle_index=row["oracle_index"],
                pb_valid=row["filtered_pb_valid"],
                reference=reference,
                selected=chosen,
                comparator=comparator,
                protein_display=binding_chain_display(
                    protein.read_text(), reference["coordinates"]
                ),
                source_hashes=dict(
                    reference=inputs["ligand_reference_sha256"],
                    protein=inputs["protein_sha256"],
                    raw_bank=raw["bank_sha256"],
                    refined_bank=rec["bank_sha256"],
                ),
            )
        )
    (out / "structures.json").write_text(json.dumps(geometries, indent=2, allow_nan=False) + "\n")
    result["case_ids"] = [
        {k: r[k] for k in ("title", "id", "selected_rmsd", "comparator_rmsd", "eligible_cases")}
        for r in geometries
    ]
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Collected all empirical evidence and three structure examples", flush=True)


if __name__ == "__main__":
    main()
