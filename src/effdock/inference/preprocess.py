"""Pocket-aware ligand/protein preprocessing for inference.

Parses a protein PDB + an RDKit ligand mol, crops the protein around a
pocket center, and builds the unified graph that the main model consumes.
Mirrors the data pipeline used at training time.
"""

from __future__ import annotations

from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops

from effdock.data.dataset import crop_to_nearest_residues, crop_to_pocket
from effdock.preprocess.fragments import decompose_fragments
from effdock.preprocess.graph import build_static_complex_graph
from effdock.preprocess.ligand import featurize_ligand, load_molecule
from effdock.preprocess.protein import parse_pocket_atoms


# ---------------------------------------------------------------------------
# Ligand loading (SMILES / SDF / MOL2)
# ---------------------------------------------------------------------------
def load_ligand(ligand_input: str) -> tuple[Chem.Mol, bool]:
    """Load ligand from SMILES / SDF / MOL2. Returns (mol, has_pose)."""
    path = Path(ligand_input)

    if path.suffix.lower() == ".sdf" and path.exists():
        mol, _, _ = load_molecule(path)
        assert mol is not None, f"Failed to parse SDF: {ligand_input}"
        return mol, True

    if path.suffix.lower() == ".mol2" and path.exists():
        mol = Chem.MolFromMol2File(str(path), sanitize=False)
        assert mol is not None, f"Failed to parse MOL2: {ligand_input}"
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
        frags = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumAtoms())
        assert mol.GetNumConformers() > 0, "MOL2 has no 3D conformer"
        return mol, True

    # SMILES → ETKDG conformer
    mol = Chem.MolFromSmiles(ligand_input)
    assert mol is not None, f"Invalid SMILES: {ligand_input}"
    mol = Chem.AddHs(mol)
    status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    assert status == 0, f"3D embedding failed: {ligand_input}"
    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    mol = Chem.RemoveHs(mol)
    return mol, False


# ---------------------------------------------------------------------------
# Pocket center derivation
# ---------------------------------------------------------------------------
def derive_pocket_center(
    prot_data: dict[str, torch.Tensor],
    ligand_coords: torch.Tensor,
    cutoff: float = 8.0,
) -> torch.Tensor:
    """Pocket center = centroid of residue virtual nodes within cutoff of ligand.

    Matches the training-time PLINDER preprocessing behavior.
    """
    pres = prot_data["pres_coords"]
    dmat = torch.cdist(pres, ligand_coords)
    mask = dmat.min(dim=1).values <= cutoff
    if mask.any():
        return pres[mask].mean(dim=0)
    return ligand_coords.mean(dim=0)


def derive_contact_atom_center(
    prot_data: dict[str, torch.Tensor],
    ligand_coords: torch.Tensor,
    cutoff: float = 6.0,
) -> torch.Tensor:
    """Pocket center = centroid of protein atoms contacting ligand atoms."""
    patom = prot_data["patom_coords"]
    dmat = torch.cdist(patom, ligand_coords)
    mask = dmat.min(dim=1).values <= cutoff
    if mask.any():
        return patom[mask].mean(dim=0)
    return ligand_coords.mean(dim=0)


# ---------------------------------------------------------------------------
# Preprocessing bundle
# ---------------------------------------------------------------------------
def preprocess_complex(
    protein_pdb: Path,
    mol: Chem.Mol,
    pocket_center: torch.Tensor | None = None,
    pocket_cutoff: float = 8.0,
    crop_ref_coords: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], dict, dict]:
    """Parse protein + ligand, crop pocket, build unified graph.

    Args:
        protein_pdb: Full protein or pocket PDB.
        mol: RDKit mol with 3D conformer.
        pocket_center: [3] reference. If None, derived from ligand coords.
        pocket_cutoff: Residue-aware cutoff (Å).
        crop_ref_coords: optional [N, 3] coordinates used only for protein
            cropping. Keeps the prior center unchanged while allowing
            ligand-contact pocket crops for benchmark redocking.
    """
    lig_data = featurize_ligand(mol)
    assert lig_data is not None, "Ligand featurization failed"

    frag_data = decompose_fragments(mol, lig_data["atom_coords"])
    assert frag_data is not None, "Fragment decomposition failed"

    for k in (
        "fragment_id",
        "frag_centers",
        "frag_local_coords",
        "frag_sizes",
        "tri_edge_index",
        "tri_edge_ref_dist",
        "fragment_adj_index",
        "cut_bond_index",
    ):
        lig_data[k] = frag_data[k]

    prot_data = parse_pocket_atoms(protein_pdb)
    assert prot_data is not None, f"Protein parsing failed: {protein_pdb}"

    if pocket_center is None:
        pocket_center = derive_pocket_center(
            prot_data,
            lig_data["atom_coords"],
            cutoff=pocket_cutoff,
        )
    else:
        pocket_center = pocket_center.to(torch.float32)

    crop_ref = crop_ref_coords.to(torch.float32) if crop_ref_coords is not None else pocket_center
    cropped = crop_to_pocket(prot_data, crop_ref, cutoff=pocket_cutoff)
    assert cropped is not None, f"No protein residues within {pocket_cutoff}Å of crop reference"

    graph = build_static_complex_graph(lig_data, cropped)

    meta = {
        "pocket_center": pocket_center,
        "num_frag": frag_data["n_frags"],
        "num_atom": lig_data["atom_coords"].shape[0],
    }
    return graph, lig_data, meta


# ---------------------------------------------------------------------------
# Preprocessed-data helpers (PDBbind-style processed tensors)
# ---------------------------------------------------------------------------
def load_processed(pdb_dir: Path) -> tuple[dict, dict, dict] | None:
    """Load ``protein.pt``, ``ligand.pt``, ``meta.pt`` from a processed complex dir.

    Returns ``None`` on any I/O failure so callers can skip missing complexes.
    """
    try:
        prot = torch.load(pdb_dir / "protein.pt", map_location="cpu", weights_only=True)
        lig = torch.load(pdb_dir / "ligand.pt", map_location="cpu", weights_only=True)
        meta = torch.load(pdb_dir / "meta.pt", map_location="cpu", weights_only=True)
    except Exception:
        return None
    return prot, lig, meta


def build_inference_bundle(
    prot: dict,
    lig: dict,
    meta: dict,
    pocket_cutoff: float = 8.0,
) -> tuple[dict, dict, dict] | None:
    """Build graph + lig_data + inference meta from ALREADY-featurized tensors.

    Skips the raw-PDB/RDKit path (which :func:`preprocess_complex` performs) and
    reuses the stored ligand features + crystal pocket center.  The protein is
    cropped around the pocket center rather than the crystal ligand coordinates
    so evaluation matches prospective inference, where the
    ligand pose is not known before sampling.
    """
    pocket_center = meta["pocket_center"]
    cropped = crop_to_pocket(prot, pocket_center, cutoff=pocket_cutoff)
    if cropped is None:
        cropped = crop_to_pocket(prot, pocket_center, cutoff=pocket_cutoff + 5.0)
    if cropped is None:
        cropped = crop_to_nearest_residues(prot, pocket_center, max_residues=32)
    if cropped is None:
        return None
    try:
        graph = build_static_complex_graph(lig, cropped)
    except Exception:
        return None
    inf_meta = {
        "pocket_center": pocket_center,
        "num_frag": int(lig["frag_sizes"].shape[0]),
        "num_atom": int(lig["atom_coords"].shape[0]),
    }
    return graph, lig, inf_meta


__all__ = [
    "load_ligand",
    "derive_pocket_center",
    "derive_contact_atom_center",
    "preprocess_complex",
    "load_processed",
    "build_inference_bundle",
]
