#!/usr/bin/env python
"""Generate train/val splits for EFFDock PLINDER processed dataset.

Policy
------
1. Source rows = pool parquet ∩ processed sample dirs (meta.pt exists).
2. Strictly drop rows whose canonical SMILES appears in a frozen external
   benchmark set.
3. Group split on `pocket_fident__70__community` (community level holdout
   prevents pocket-cluster leakage). Rows with NaN community → train.
4. Target val fraction ≈ 5% (sample-count) — caps at 6% by stopping when
   crossed.
5. Enforce SMILES disjoint: any SMILES that appears in both buckets after
   the group split is forced to train (the larger bucket); val rows with
   that SMILES are dropped.
6. Writes:
     data/splits/train.txt   one sample_key per line
     data/splits/val.txt     one sample_key per line
     data/splits/manifest.json   counts + RNG seed + provenance
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def sample_key(system_id: str, instance_chain: str) -> str:
    return f"{system_id}__{instance_chain}".replace("/", "_")


def canon_smiles(smi: str) -> str | None:
    if not isinstance(smi, str) or not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_external_smiles(ext_dir: Path) -> set[str]:
    smis: set[str] = set()
    required = ("astex_smiles.json", "pb_smiles.json")
    optional = ("phibench_smiles.json", "foldbench_smiles.json", "openbind_smiles.json")
    missing = [name for name in required if not (ext_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "strict split requires frozen SMILES for every benchmark; missing: "
            + ", ".join(str(ext_dir / name) for name in missing)
        )
    mapping_names = [*required, *(name for name in optional if (ext_dir / name).exists())]
    for name in mapping_names:
        p = ext_dir / name
        with open(p) as f:
            d = json.load(f)
        for v in d.values():
            s = v.get("smiles") if isinstance(v, dict) else v
            c = canon_smiles(s)
            if c:
                smis.add(c)
    log.info(
        "external test canonical SMILES: %d from %s",
        len(smis),
        ", ".join(mapping_names),
    )
    return smis


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-parquet", type=Path, default=Path("data/plinder_pool.parquet"))
    ap.add_argument("--processed-dir", type=Path, default=Path("data/plinder_processed"))
    ap.add_argument("--external-dir", type=Path, default=Path("data/external_test"))
    ap.add_argument("--output-dir", dest="out_dir", type=Path, default=Path("data/splits"))
    ap.add_argument("--val-fraction", dest="val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # 1) pool ∩ processed
    df = pd.read_parquet(args.pool_parquet)
    df["sample_key"] = [sample_key(s, c) for s, c in zip(df.system_id, df.ligand_instance_chain)]
    log.info("pool rows: %d  uniq_keys=%d", len(df), df.sample_key.nunique())

    processed = {
        p.name for p in args.processed_dir.iterdir() if p.is_dir() and (p / "meta.pt").exists()
    }
    log.info("processed sample dirs (meta.pt): %d", len(processed))
    df = df[df.sample_key.isin(processed)].reset_index(drop=True)
    log.info("intersect (pool ∩ processed): %d", len(df))

    # 2) strict external-benchmark ligand exclusion
    ext_smis = load_external_smiles(args.external_dir)
    df["smiles_canon"] = df.ligand_rdkit_canonical_smiles.map(canon_smiles)
    n_missing = df.smiles_canon.isna().sum()
    if n_missing:
        log.warning("rows with un-parseable SMILES (kept, treated as unique): %d", n_missing)
    smi_match = df.smiles_canon.isin(ext_smis) & df.smiles_canon.notna()
    leak_mask = smi_match
    log.info("[strict] drop external canonical-SMILES matches: %d rows", int(leak_mask.sum()))
    df = df[~leak_mask].reset_index(drop=True)

    # 3) group split on pocket70 community
    nan_mask = df["pocket_fident__70__community"].isna()
    df_nan = df[nan_mask].reset_index(drop=True)  # always train
    df_grp = df[~nan_mask].reset_index(drop=True)
    log.info("rows with NaN pocket70 (→ train): %d", len(df_nan))

    group_counts = df_grp["pocket_fident__70__community"].value_counts()
    groups = group_counts.index.to_numpy()
    sizes = group_counts.values
    perm = rng.permutation(len(groups))
    groups, sizes = groups[perm], sizes[perm]

    target_val = args.val_frac * (len(df_grp) + len(df_nan))
    val_groups: set[str] = set()
    val_count = 0
    for g, s in zip(groups, sizes):
        if val_count >= target_val:
            break
        val_groups.add(g)
        val_count += s
    log.info(
        "val groups picked: %d  (%d rows ≈ %.2f%%)",
        len(val_groups),
        val_count,
        100 * val_count / max(len(df), 1),
    )

    val_mask = df_grp["pocket_fident__70__community"].isin(val_groups)
    df_val = df_grp[val_mask].reset_index(drop=True)
    df_tr = pd.concat([df_grp[~val_mask], df_nan], ignore_index=True)

    # 4) SMILES disjoint enforcement (val SMILES ∩ train SMILES → drop from val)
    tr_smis = set(df_tr.smiles_canon.dropna().unique())
    overlap = df_val.smiles_canon.isin(tr_smis) & df_val.smiles_canon.notna()
    log.info("val rows sharing SMILES with train (dropped from val): %d", int(overlap.sum()))
    df_val = df_val[~overlap].reset_index(drop=True)

    train_keys = sorted(df_tr.sample_key.tolist())
    val_keys = sorted(df_val.sample_key.tolist())

    # assertion: disjoint sample keys
    assert set(train_keys).isdisjoint(val_keys), "train/val sample_key overlap"

    (args.out_dir / "train.txt").write_text("\n".join(train_keys) + "\n")
    (args.out_dir / "val.txt").write_text("\n".join(val_keys) + "\n")

    # Trainer expects a single JSON keyed by split (see effdock.training.trainer
    # `EFFDockDataset(split_file=..., split_key="train"|"val")`). Emit it
    # alongside the plain text files so both consumers work.
    with open(args.out_dir / "plinder.json", "w") as f:
        json.dump({"train": train_keys, "val": val_keys}, f)

    manifest = {
        "seed": args.seed,
        "val_frac_target": args.val_frac,
        "leak_policy": "strict_external_canonical_smiles",
        "pool_parquet": str(args.pool_parquet),
        "processed_dir": str(args.processed_dir),
        "external_dir": str(args.external_dir),
        "n_train": len(train_keys),
        "n_val": len(val_keys),
        "val_frac_actual": len(val_keys) / max(len(train_keys) + len(val_keys), 1),
        "n_val_pocket70_groups": len(val_groups),
        "n_pool_rows": int(len(df) + int(leak_mask.sum())),
        "n_external_smiles_dropped": int(leak_mask.sum()),
        "n_smiles_overlap_dropped_from_val": int(overlap.sum()),
        "uniq_smiles_train": int(df_tr.smiles_canon.nunique()),
        "uniq_smiles_val": int(df_val.smiles_canon.nunique()),
        "uniq_pocket70_train": int(df_tr["pocket_fident__70__community"].nunique()),
        "uniq_pocket70_val": int(df_val["pocket_fident__70__community"].nunique()),
    }
    with open(args.out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("=" * 60)
    log.info(
        "train: %d  val: %d  (val_frac=%.4f)",
        len(train_keys),
        len(val_keys),
        manifest["val_frac_actual"],
    )
    log.info(
        "uniq SMILES  train=%d  val=%d", manifest["uniq_smiles_train"], manifest["uniq_smiles_val"]
    )
    log.info(
        "uniq pocket70  train=%d  val=%d",
        manifest["uniq_pocket70_train"],
        manifest["uniq_pocket70_val"],
    )
    log.info("manifest -> %s", args.out_dir / "manifest.json")


if __name__ == "__main__":
    main()
