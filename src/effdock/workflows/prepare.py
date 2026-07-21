#!/usr/bin/env python
"""Build EFFDock processed dataset from PLINDER systems.

Each row in the curated pool parquet (output of ``scripts/filter_plinder.py``)
maps to one (system, ligand) → ``data/plinder_processed/{system_id}__{instance_chain}/``
sample containing ``protein.pt``, ``ligand.pt``, ``meta.pt``.

PLINDER on-disk layout:
    ~/.local/share/plinder/{release}/{iteration}/systems/{system_id}/
        receptor.pdb
        ligand_files/{instance_chain}.sdf
"""

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch

from effdock.preprocess.fragments import decompose_fragments
from effdock.preprocess.ligand import featurize_ligand, load_molecule
from effdock.preprocess.protein import parse_pocket_atoms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SCHEMA_VERSION = 2  # PLINDER provenance


def sample_key(system_id: str, instance_chain: str) -> str:
    return f"{system_id}__{instance_chain}".replace("/", "_")


def process_sample(row: dict, plinder_root: Path, out_dir: Path) -> dict:
    system_id = row["system_id"]
    instance_chain = row["ligand_instance_chain"]
    key = sample_key(system_id, instance_chain)
    status = {
        "key": key,
        "system_id": system_id,
        "ligand_chain": instance_chain,
        "success": False,
        "reason": "",
    }

    sys_dir = plinder_root / "systems" / system_id
    pdb_path = sys_dir / "receptor.pdb"
    sdf_path = sys_dir / "ligand_files" / f"{instance_chain}.sdf"
    if not pdb_path.exists():
        status["reason"] = "missing_receptor_pdb"
        return status
    if not sdf_path.exists():
        status["reason"] = "missing_ligand_sdf"
        return status

    # Skip if already done (idempotent)
    out_complex = out_dir / key
    if (out_complex / "meta.pt").exists():
        status["success"] = True
        status["reason"] = "already_done"
        return status

    # --- Ligand --------------------------------------------------------
    mol, used_mol2, sanitize_ok = load_molecule(sdf_path, None)
    if mol is None:
        status["reason"] = "ligand_parse_failed"
        return status

    lig_data = featurize_ligand(mol)
    if lig_data is None:
        status["reason"] = "ligand_featurize_failed"
        return status

    frag_data = decompose_fragments(mol, lig_data["atom_coords"])
    if frag_data is None:
        status["reason"] = "fragment_decomposition_failed"
        return status

    # NOTE: dg_bounds intentionally dropped (was dense [N,N] float32 → 3.4GB total
    # disk waste; never consumed — distance_geometry_loss recomputes from runtime
    # coordinates). schema_version=2 omits it.

    lig_data["fragment_id"] = frag_data["fragment_id"]
    lig_data["frag_centers"] = frag_data["frag_centers"]
    lig_data["frag_local_coords"] = frag_data["frag_local_coords"]
    lig_data["frag_sizes"] = frag_data["frag_sizes"]
    lig_data["tri_edge_index"] = frag_data["tri_edge_index"]
    lig_data["tri_edge_ref_dist"] = frag_data["tri_edge_ref_dist"]
    lig_data["fragment_adj_index"] = frag_data["fragment_adj_index"]
    lig_data["cut_bond_index"] = frag_data["cut_bond_index"]

    # --- Protein ------------------------------------------------------
    prot_data = parse_pocket_atoms(pdb_path)
    if prot_data is None:
        status["reason"] = "protein_parse_failed"
        return status

    # Pocket center: residues within 8Å of THIS ligand only (multi-ligand safe)
    pres_coords = prot_data["pres_coords"]
    pres_to_lig = torch.cdist(pres_coords, lig_data["atom_coords"])
    pocket_mask = pres_to_lig.min(dim=1).values <= 8.0
    pocket_center = (
        pres_coords[pocket_mask].mean(dim=0)
        if pocket_mask.any()
        else lig_data["atom_coords"].mean(dim=0)
    )
    n_pocket_res = int(pocket_mask.sum().item())

    # --- Meta ---------------------------------------------------------
    n_atoms = lig_data["atom_coords"].shape[0]
    meta = {
        "pdb_id": key,
        "plinder_system_id": system_id,
        "plinder_ligand_chain": instance_chain,
        "plinder_ccd_code": row.get("ligand_unique_ccd_code", ""),
        "is_cofactor": bool(row.get("ligand_is_cofactor", False)),
        "is_kinase_inhibitor": bool(row.get("ligand_is_kinase_inhibitor", False)),
        "pocket_center": pocket_center,
        "num_pocket_res": torch.tensor(n_pocket_res, dtype=torch.int64),
        "num_res": torch.tensor(prot_data["pres_coords"].shape[0], dtype=torch.int64),
        "num_atom": torch.tensor(n_atoms, dtype=torch.int64),
        "num_frag": torch.tensor(frag_data["n_frags"], dtype=torch.int64),
        "num_prot_atom": torch.tensor(prot_data["patom_coords"].shape[0], dtype=torch.int64),
        "ligand_sanitize_ok": torch.tensor(sanitize_ok, dtype=torch.bool),
        "schema_version": SCHEMA_VERSION,
        "source": "plinder_2024_06_v2",
    }

    out_complex.mkdir(parents=True, exist_ok=True)
    torch.save(prot_data, out_complex / "protein.pt")
    torch.save(lig_data, out_complex / "ligand.pt")
    torch.save(meta, out_complex / "meta.pt")

    status["success"] = True
    status["num_res"] = prot_data["pres_coords"].shape[0]
    status["num_prot_atom"] = prot_data["patom_coords"].shape[0]
    status["num_atom"] = n_atoms
    status["num_frag"] = frag_data["n_frags"]
    status["is_cofactor"] = meta["is_cofactor"]
    return status


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filtered_parquet", type=Path, default=Path("data/plinder_pool.parquet"))
    ap.add_argument(
        "--plinder_root", type=Path, default=Path.home() / ".local/share/plinder/2024-06/v2"
    )
    ap.add_argument("--out_dir", type=Path, default=Path("data/plinder_processed"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--limit", type=int, default=None, help="Process only first N rows (for testing)."
    )
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.filtered_parquet)
    if args.limit:
        df = df.head(args.limit)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Loaded %d (system,ligand) rows from %s", len(df), args.filtered_parquet)
    log.info("PLINDER root: %s", args.plinder_root)
    log.info("Output:       %s", args.out_dir)

    rows = df.to_dict("records")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_sample, r, args.plinder_root, args.out_dir): r for r in rows}
        for i, fut in enumerate(as_completed(futures)):
            try:
                results.append(fut.result())
            except Exception as e:
                row = futures[fut]
                results.append(
                    {
                        "key": sample_key(row["system_id"], row["ligand_instance_chain"]),
                        "success": False,
                        "reason": f"exception: {e}",
                    }
                )
            if (i + 1) % 1000 == 0:
                ok = sum(1 for r in results if r["success"])
                log.info("  progress: %d/%d, success=%d", i + 1, len(rows), ok)

    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    log.info("=" * 60)
    log.info(
        "Done. success=%d / %d (%.1f%%)",
        len(successes),
        len(results),
        100 * len(successes) / max(len(results), 1),
    )
    if failures:
        from collections import Counter

        reasons = Counter(r.get("reason", "unknown") for r in failures)
        for reason, c in sorted(reasons.items(), key=lambda x: -x[1]):
            log.info("  %s: %d", reason, c)

    if successes:
        for label, key in [
            ("Prot residues", "num_res"),
            ("Lig atoms", "num_atom"),
            ("Fragments", "num_frag"),
        ]:
            vals = sorted([r[key] for r in successes if key in r])
            if vals:
                log.info(
                    "  %-15s min=%d med=%d max=%d", label, vals[0], vals[len(vals) // 2], vals[-1]
                )
        cof = sum(1 for r in successes if r.get("is_cofactor"))
        log.info("  cofactors: %d (%.1f%%)", cof, 100 * cof / len(successes))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": "plinder_2024_06_v2",
        "filtered_parquet": str(args.filtered_parquet),
        "total": len(results),
        "success": len(successes),
        "failed": len(failures),
        "keys": sorted([r["key"] for r in successes]),
    }
    with open(args.out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("manifest -> %s", args.out_dir / "manifest.json")


if __name__ == "__main__":
    main()
