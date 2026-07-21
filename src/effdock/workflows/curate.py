#!/usr/bin/env python
"""Filter PLINDER plindex → curated EFFDock training pool.

Policy:
Process the FULL plindex (ignore official train/val/test/removed split).
Custom train/val/test split is generated separately by a downstream script
that enforces SMILES-disjoint sets.

Hard filters (drop):
  - is_ion | is_artifact | is_invalid | is_covalent
  - entry_resolution > 3.5 (NaN OK → keep CryoEM)
  - resolved_heavy / heavy < 0.9
  - rscc < 0.7 (NaN OK)

PoseBusters / structural quality (drop):
  - ligand_posebusters_volume_overlap_protein ≥ 0.1   (ligand-protein interpenetration)
  - ligand_posebusters_internal_energy fail
      EXCEPT when `ligand_is_cofactor=True`. Audit shows PB's conformer-energy
      heuristic drops HEM (15k), CLA (22k), BCR (6.6k), BCL (3.2k) — legitimate
      docking targets (heme, chlorophylls, carotenoids).
  - system_ligand_validation_average_rsr > 0.3
  - system_fraction_atoms_with_crystal_contacts ≥ 0.3
  - ligand_num_missing_pli_interface_residues > 0

Size caps (EFFDock build pipeline alignment, max_atoms=120):
  - ligand_num_heavy_atoms > 120
  - ligand_num_rot_bonds > 30

Pocket interaction:
  - system_proper_num_interactions < 3

NaN policy: missing quality metrics PASS the gate (lenient). Strict-quality
variant would require explicit numeric thresholds with NaN→fail.

Representative dedup
--------------------
Each (entry_pdb_id, smiles) pair, then each (pocket_fident_70_cluster, smiles)
pair, is collapsed to a single representative row. The survivor is selected by
sorting on:

  1. biounit_priority asc            (system_biounit_id == "1" wins, others penalised)
  2. system_proper_num_interactions  desc
  3. rscc                            desc  (NaN → -1, pushed last)
  4. rsr                             asc   (NaN → 999, pushed last)
  5. entry_resolution                asc   (NaN → 999, pushed last)
  6. system_id, ligand_instance_chain asc  (deterministic tie-breakers)

Then:
  (a) drop_duplicates(entry_pdb_id, ligand_rdkit_canonical_smiles)
      → same PDB + same ligand = 1 row (assembly/chain copies collapsed).
  (b) drop_duplicates(pocket_fident__70__community, ligand_rdkit_canonical_smiles)
      → same pocket cluster + same ligand = 1 row (cross-PDB redundancy collapsed).
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Columns retained in the pool parquet — keep audit metrics so quality
# decisions can be re-evaluated downstream without re-reading plindex.
OUT_COLS = [
    # identity
    "system_id",
    "system_id_no_biounit",
    "system_biounit_id",
    "ligand_id",
    "ligand_instance_chain",
    "ligand_unique_ccd_code",
    "entry_pdb_id",
    "entry_resolution",
    # ligand chemistry
    "ligand_num_heavy_atoms",
    "ligand_num_resolved_heavy_atoms",
    "ligand_num_rot_bonds",
    "ligand_molecular_weight",
    "ligand_is_cofactor",
    "ligand_is_kinase_inhibitor",
    "ligand_is_lipinski",
    "ligand_is_fragment",
    "ligand_rdkit_canonical_smiles",
    "ligand_smiles",
    # pocket / interaction
    "system_proper_num_pocket_residues",
    "system_proper_num_interactions",
    "ligand_num_missing_pli_interface_residues",
    "pocket_fident__70__community",
    "pocket_fident__50__community",
    # audit (quality metrics used in the filter — kept for downstream review)
    "system_ligand_validation_average_rscc",
    "system_ligand_validation_average_rsr",
    "system_fraction_atoms_with_crystal_contacts",
    "ligand_posebusters_volume_overlap_protein",
    "ligand_posebusters_internal_energy",
]

# Hard-filter inputs not retained in output (drop semantics already applied).
DROP_FLAG_COLS = ["ligand_is_ion", "ligand_is_artifact", "ligand_is_invalid", "ligand_is_covalent"]
NEEDED_COLS = sorted(set(OUT_COLS + DROP_FLAG_COLS))


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df[
        ~df.ligand_is_ion & ~df.ligand_is_artifact & ~df.ligand_is_invalid & ~df.ligand_is_covalent
    ]
    log.info("  drop ion/artifact/invalid/covalent: %d → %d", n0, len(df))

    df = df[df.system_proper_num_interactions >= 3]
    log.info("  + interactions ≥ 3: %d", len(df))

    df = df[(df.entry_resolution.isna()) | (df.entry_resolution <= 3.5)]
    log.info("  + entry_resolution ≤ 3.5 (NaN ok): %d", len(df))

    resolved_frac = df.ligand_num_resolved_heavy_atoms / df.ligand_num_heavy_atoms.replace(0, pd.NA)
    df = df[resolved_frac.fillna(0) >= 0.9]
    log.info("  + resolved/heavy ≥ 0.9: %d", len(df))

    df = df[df.system_ligand_validation_average_rscc.fillna(1.0) >= 0.7]
    log.info("  + rscc ≥ 0.7 (NaN ok): %d", len(df))

    df = df[df.ligand_posebusters_volume_overlap_protein.fillna(0) < 0.1]
    log.info("  + pb_volume_overlap_protein < 0.1 (NaN ok): %d", len(df))

    # internal_energy: cofactors exempted (heme/CLA/BCR/BCL macrocycles fail PB heuristic).
    # Use .eq(True) instead of astype(bool) so a stray "False" string would not flip to True.
    ie_ok = df.ligand_posebusters_internal_energy.fillna(True).eq(True)
    df = df[ie_ok | df.ligand_is_cofactor.fillna(False).eq(True)]
    log.info("  + pb_internal_energy_ok OR is_cofactor: %d", len(df))

    df = df[df.system_ligand_validation_average_rsr.fillna(0) <= 0.3]
    log.info("  + rsr ≤ 0.3 (NaN ok): %d", len(df))

    df = df[df.system_fraction_atoms_with_crystal_contacts.fillna(0) < 0.3]
    log.info("  + crystal_contacts < 0.3 (NaN ok): %d", len(df))

    df = df[df.ligand_num_missing_pli_interface_residues.fillna(0) == 0]
    log.info("  + no missing pocket interface residues (NaN ok): %d", len(df))

    df = df[df.ligand_num_heavy_atoms <= 120]
    log.info("  + heavy_atoms ≤ 120: %d", len(df))

    df = df[df.ligand_num_rot_bonds <= 30]
    log.info("  + rot_bonds ≤ 30: %d", len(df))

    # Deterministic sort with explicit NaN handling: NaN values are pushed last
    # via sentinel fills so the dedup survivor is reproducible across runs.
    df = df.assign(
        _biounit_priority=(df.system_biounit_id != "1").astype(int),
        _rscc_sort=df.system_ligand_validation_average_rscc.fillna(-1.0),
        _rsr_sort=df.system_ligand_validation_average_rsr.fillna(999.0),
        _res_sort=df.entry_resolution.fillna(999.0),
    )
    df = df.sort_values(
        [
            "_biounit_priority",
            "system_proper_num_interactions",
            "_rscc_sort",
            "_rsr_sort",
            "_res_sort",
            "system_id",
            "ligand_instance_chain",
        ],
        ascending=[True, False, False, True, True, True, True],
    )
    df = df.drop_duplicates(subset=["entry_pdb_id", "ligand_rdkit_canonical_smiles"], keep="first")
    log.info("  + dedup (pdb, smiles): %d", len(df))
    df = df.drop_duplicates(
        subset=["pocket_fident__70__community", "ligand_rdkit_canonical_smiles"], keep="first"
    )
    log.info(
        "  + dedup (pocket70, smiles): %d  uniq_smi=%d  uniq_pocket=%d",
        len(df),
        df.ligand_rdkit_canonical_smiles.nunique(),
        df["pocket_fident__70__community"].nunique(),
    )
    return df.drop(
        columns=["_biounit_priority", "_rscc_sort", "_rsr_sort", "_res_sort"]
    ).reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--plindex", type=Path, default=Path("archive/data_plinder_meta/plinder_plindex.parquet")
    )
    ap.add_argument("--out", type=Path, default=Path("data/plinder_pool.parquet"))
    args = ap.parse_args(argv)

    log.info("Loading plindex…")
    plindex = pd.read_parquet(args.plindex, columns=NEEDED_COLS)
    log.info(
        "plindex full: rows=%d uniq_smi=%d",
        len(plindex),
        plindex.ligand_rdkit_canonical_smiles.nunique(),
    )

    pool = apply_filters(plindex)
    out = pool[OUT_COLS]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    log.info(
        "Wrote %s  rows=%d uniq_smi=%d uniq_pocket70=%d biounit_not_1=%d",
        args.out,
        len(out),
        out.ligand_rdkit_canonical_smiles.nunique(),
        out.pocket_fident__70__community.nunique(),
        (out.system_biounit_id != "1").sum(),
    )


if __name__ == "__main__":
    main()
