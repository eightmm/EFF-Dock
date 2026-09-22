"""Train-only exposure audit based on GPT-5.6-sol investigation."""

import argparse
import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from effdock.workflows.benchmark_inputs import BenchmarkInputMismatchError, canonical_heavy_smiles

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "outputs/benchmarks"
EXPECTED = dict(astex=85, posebusters=308, phibench=206, foldbench=558, openbind=925)
TEMP = B / "external_chirality_u70k_temporal_full_r3_v1/pb_inchi_compat_v1/provenance.json"
MAPPINGS = {
    d: ROOT / f"data/external_benchmarks/{folder}/external/{d}_smiles.json"
    for d, folder in [
        ("phibench", "phibench206"),
        ("foldbench", "foldbench_full"),
        ("openbind", "openbind_full"),
    ]
}
MANIFESTS = {
    d: ROOT / f"data/external_benchmarks/{folder}/manifests/{d}.json"
    for d, folder in [("phibench", "phibench206"), ("foldbench", "foldbench_full")]
}
TRAIN_INDEX = (
    ROOT
    / "data/plinder_processed/.effdock_index/dcee3e1d58c79520dbbfee8b7904e7c0a1dd2196b38a2bbc0e223f94fe1844ce.json"
)


def executed_membership(split):
    """Verify the loader cache key against ordered split IDs and training filters."""
    config = dict(
        version=1,
        root=str((ROOT / "data/plinder_processed").resolve()),
        min_atoms=5,
        max_atoms=120,
        max_frags=30,
        min_protein_res=50,
    )
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for sample_id in split["train"]:
        digest.update(b"\0")
        digest.update(sample_id.encode())
    if digest.hexdigest() != TRAIN_INDEX.stem:
        raise ValueError("executed loader cache key drift")
    if sha(TRAIN_INDEX) != "21b194112242d0645cd61faeddec09d3188c0c5e15b66ab0ffee4c6ccf2a02b4":
        raise ValueError("executed loader membership hash drift")
    cached = json.loads(TRAIN_INDEX.read_text())
    ids = cached["sample_ids"]
    members = set(ids)
    if len(ids) != 47277 or len(members) != len(ids) or not members <= set(split["train"]):
        raise ValueError("executed membership coverage mismatch")
    if ids != [s for s in split["train"] if s in members]:
        raise ValueError("executed loader order mismatch")
    return members


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def classify(exact, tanimoto):
    if exact:
        return "exact_identity"
    if tanimoto is None:
        return "missing"
    if tanimoto < 0.5:
        return "no_observed_exact_<0.5"
    if tanimoto < 0.8:
        return "no_observed_exact_0.5_to_<0.8"
    return "no_observed_exact_>=0.8"


def inputs():
    frozen = ROOT / "benchmarks/inputs/guidance_budget1000_full_inputs.json"
    d = json.loads(frozen.read_text())
    mappings = {
        ds: {k.lower(): v["smiles"] for k, v in d["datasets"][ds]["ligands"].items()}
        for ds in ("astex", "posebusters")
    }
    mappings.update(
        {
            ds: {k.lower(): v for k, v in json.loads(p.read_text()).items()}
            for ds, p in MAPPINGS.items()
        }
    )
    pdb = {ds: {k: k[:4] for k in mappings[ds]} for ds in ("astex", "posebusters")}
    pdb["openbind"] = {k: None for k in mappings["openbind"]}
    for ds, p in MANIFESTS.items():
        pdb[ds] = {
            r["id"].lower(): r["pdb_id"].lower() for r in json.loads(p.read_text())["records"]
        }
    for ds, n in EXPECTED.items():
        if len(mappings[ds]) != n or set(mappings[ds]) != set(pdb[ds]):
            raise ValueError("external mapping coverage mismatch")
    return mappings, pdb, [frozen, *MAPPINGS.values(), *MANIFESTS.values()]


def training():
    split_path = ROOT / "data/splits/plinder.json"
    pool_path = ROOT / "data/plinder_pool.parquet"
    split = json.loads(split_path.read_text())
    train = set(split["train"])
    val = set(split["val"])
    if len(train) != 47310 or len(val) != 1076 or train & val:
        raise ValueError("preserved split drift")
    train = executed_membership(split)
    rows = pq.read_table(
        pool_path,
        columns=[
            "system_id",
            "ligand_instance_chain",
            "entry_pdb_id",
            "ligand_rdkit_canonical_smiles",
        ],
    ).to_pylist()
    keyed = {}
    membership = train | val
    for row in rows:
        key = f"{row['system_id']}__{row['ligand_instance_chain']}".replace("/", "_")
        if key in membership:
            if key in keyed and keyed[key] != row:
                raise ValueError("conflicting pool metadata")
            keyed[key] = row
    if (train | val) - keyed.keys():
        raise ValueError("missing training pool identities")
    cache = {}

    def index(keys):
        by_smi = defaultdict(set)
        by_pdb = set()
        invalid = []
        for key in sorted(keys):
            row = keyed[key]
            raw = row["ligand_rdkit_canonical_smiles"]
            pdb = str(row["entry_pdb_id"]).lower()
            by_pdb.add(pdb)
            if raw not in cache:
                try:
                    cache[raw] = canonical_heavy_smiles(raw)
                except BenchmarkInputMismatchError:
                    cache[raw] = None
            canonical = cache[raw]
            if not canonical:
                invalid.append(
                    dict(
                        sample_key=key,
                        smiles=raw,
                        pdb_id=pdb,
                        reason="canonical_heavy_smiles_parse_failure",
                    )
                )
                continue
            by_smi[canonical].add(pdb)
        return by_smi, by_pdb, invalid

    return index(train), index(val), [split_path, pool_path, TRAIN_INDEX]


def annotate():
    mappings, pdbs, paths = inputs()
    (train, train_pdb, train_invalid), (val, _, val_invalid), training_paths = training()
    print(f"Train index complete: {len(train)} unique ligands", flush=True)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False
    )
    canonicals = sorted(train)
    fps = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in canonicals]
    cache = {}
    annotations = []
    for dataset, mapping in mappings.items():
        for cid, raw in sorted(mapping.items()):
            canonical = canonical_heavy_smiles(raw)
            if canonical not in cache:
                similarities = DataStructs.BulkTanimotoSimilarity(
                    generator.GetFingerprint(Chem.MolFromSmiles(canonical)), fps
                )
                i = max(range(len(similarities)), key=lambda j: (similarities[j], -j))
                cache[canonical] = (float(similarities[i]), canonicals[i])
            similarity, neighbor = cache[canonical]
            exact = canonical in train
            if exact:
                neighbor = canonical
            pdb = pdbs[dataset][cid]
            annotations.append(
                dict(
                    dataset=dataset,
                    id=cid,
                    observed_train_ligand_identity=exact,
                    exact_train_ligand_identity=(
                        True if exact else None if train_invalid else False
                    ),
                    observed_validation_ligand_identity=(canonical in val),
                    validation_only_ligand_identity=(
                        False if exact or canonical not in val else None if train_invalid else True
                    ),
                    nearest_train_tanimoto=similarity,
                    similarity_stratum=classify(exact, similarity),
                    canonical_heavy_isomeric_smiles=canonical,
                    nearest_train_canonical_smiles=neighbor,
                    benchmark_pdb_id=pdb,
                    exact_train_pdb_status=(
                        "missing"
                        if pdb is None
                        else "exact_match"
                        if pdb in train_pdb
                        else "no_exact_match"
                    ),
                    exact_train_pdb_and_ligand=(
                        None if pdb is None else pdb in train.get(canonical, set())
                    ),
                )
            )
        print(f"Annotated {dataset}: {len(mapping)}", flush=True)
    metadata = dict(
        train_members=47277,
        preserved_split_train_members=47310,
        validation_members=1076,
        unique_train_ligands=len(train),
        train_invalid_smiles=train_invalid,
        validation_invalid_smiles=val_invalid,
        parseable_train_members=47277 - len(train_invalid),
        missing_training_policy="No-match refers only to parseable train ligands; unresolved train ligands are not assumed non-overlap. Tanimoto is nearest parseable-train similarity.",
        executed_filtered_train_count=47277,
        exact_filtered_membership_audited=True,
        fingerprint=dict(radius=2, bits=2048, include_chirality=False),
        rdkit=rdBase.rdkitVersion,
        protein_pocket_nn="not_available; exact PDB accession is not homology",
        membership_caveat="Executed 47,277-system S50 fine-tuning loader membership verified by cache-key reconstruction and content hash; not a union of all predecessor pretraining exposure.",
    )
    return annotations, metadata, paths + training_paths


def selected_sources():
    base = B / "astex_pb_unguided_r3_hostmatched_v2/n100_s10/selection/pb_full"
    paths = list(base.glob("*_repeat_*/shard-*/results.json"))
    if len(paths) != 48:
        raise ValueError("missing AX/PB selected PB shards")
    for p in sorted(paths):
        yield p, "official_redock", sha(p)
    prov = json.loads(TEMP.read_text())
    if (
        len(prov["sources"]) != 72
        or sum(s["profile"] == "official_redock" for s in prov["sources"]) != 69
    ):
        raise ValueError("temporal provenance drift")
    for s in prov["sources"]:
        p = ROOT / s["path"]
        if sha(p) != s["sha256"]:
            raise ValueError("temporal PB source hash drift")
        yield p, s["profile"], s["sha256"]


def stratify(annotations):
    by_id = {(r["dataset"], r["id"]): r for r in annotations}
    cells = defaultdict(list)
    coverage = defaultdict(set)
    sources = []
    for path, profile, digest in selected_sources():
        ds, rep = re.search(
            r"/(astex|posebusters|phibench|foldbench|openbind)_repeat_(\d+)/", str(path)
        ).groups()
        rep = int(rep)
        sources.append(dict(path=str(path.relative_to(ROOT)), sha256=digest, profile=profile))
        for r in json.loads(path.read_text())["rows"]:
            key = (ds, rep, r["stage"], r["policy"])
            cid = r["id"].lower()
            if cid in coverage[key]:
                raise ValueError("duplicate PB ID")
            coverage[key].add(cid)
            ann = by_id[(ds, cid)]
            hit = float(r["rmsd"] < 2)
            valid = float(r["pb_valid"])
            values = (hit, valid, hit * valid)
            for axis, label in [
                ("ligand_similarity", ann["similarity_stratum"]),
                (
                    "exact_train_ligand",
                    "observed_overlap"
                    if ann["observed_train_ligand_identity"]
                    else "no_observed_match",
                ),
                ("exact_train_pdb", ann["exact_train_pdb_status"]),
            ]:
                cells[(*key, axis, label)].append(values)
    for ds, n in EXPECTED.items():
        ids = {cid for d, cid in by_id if d == ds}
        for rep in range(3):
            for stage in ("raw", "refined"):
                for policy in ("baseline", "filtered"):
                    if coverage[(ds, rep, stage, policy)] != ids or len(ids) != n:
                        raise ValueError("selected cohort incompleteness")
    repeats = []
    groups = defaultdict(list)
    for (ds, rep, stage, policy, axis, label), values in sorted(cells.items()):
        row = dict(
            dataset=ds,
            repeat=rep,
            stage=stage,
            policy=policy,
            axis=axis,
            stratum=label,
            count=len(values),
            **{
                k: 100 * statistics.mean(v[i] for v in values)
                for i, k in enumerate(("rmsd_lt2", "pb_valid", "joint"))
            },
        )
        repeats.append(row)
        groups[(ds, stage, policy, axis, label)].append(row)
    summaries = []
    for key, rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: r["repeat"])
        if [r["repeat"] for r in rs] != [0, 1, 2] or len({r["count"] for r in rs}) != 1:
            raise ValueError("stratum coverage drift")
        row = dict(zip(("dataset", "stage", "policy", "axis", "stratum"), key))
        row["count_per_repeat"] = rs[0]["count"]
        row.update(
            {
                k: dict(
                    mean=statistics.mean(r[k] for r in rs),
                    sd=statistics.stdev(r[k] for r in rs),
                    per_repeat=[r[k] for r in rs],
                )
                for k in ("rmsd_lt2", "pb_valid", "joint")
            }
        )
        summaries.append(row)
    return repeats, summaries, sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/paper")
    args = p.parse_args()
    annotations, metadata, paths = annotate()
    repeats, summary, pb_sources = stratify(annotations)
    sources = [
        dict(path=str(p.relative_to(ROOT)), sha256=sha(p))
        for p in [*paths, TEMP, Path(__file__), ROOT / "src/effdock/workflows/benchmark_inputs.py"]
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "overlap_metrics.json").write_text(
        json.dumps(
            dict(
                metadata=metadata,
                annotations=annotations,
                per_repeat=repeats,
                rows=summary,
                sources=sources,
                pb_sources=pb_sources,
            ),
            indent=2,
        )
    )
    print(
        json.dumps(
            dict(status="complete", annotations=len(annotations), stratified_rows=len(summary))
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
