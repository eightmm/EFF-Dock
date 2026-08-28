"""Benchmark evaluation helpers.

Shared utilities for docking benchmarks (Astex / PoseBusters):
  - Dataset/complex discovery (auto-detect protein + ligand files)
  - Robust RDKit loaders for SDF / MOL2 with sanitization fallbacks
  - MMFF94s local refinement tethered to predicted heavy-atom positions
  - Symmetry-aware pose RMSD (RDKit CalcRMS with index fallback)
  - Atom matching between crystal and docking mols (strict → no-charge → MCS)
  - Pose selection + aggregate stats
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS, rdMolAlign, rdmolops

REFINE_METHODS = ("none", "mmff")

# SELECT_METHODS is a *label registry* used to bucket reported metrics.
# The selection logic does NOT all live in ``select_pose`` — only "oracle"
# is purely RMSD-based. "vina" requires its own scoring
# pass before picking the best; callers must use ``select_by_score``
# directly with the score array (e.g. Vina free energy).
SELECT_METHODS = ("oracle", "vina")
V2_IDS_PATH = Path("data/external_test/posebusters_v2_ids.txt")


# ---------------------------------------------------------------------------
# Dataset auto-detection
# ---------------------------------------------------------------------------
def detect_complex_files(complex_dir: Path, pdb_id: str) -> tuple[Path, Path, str] | None:
    """Return (protein_pdb, ligand_file, format) for a complex dir.

    Prefers ``{id}_protein.pdb`` (full protein) and only falls back to
    ``{id}_pocket.pdb`` when the full protein is absent.  Some Astex
    pocket.pdb files are truncated/empty (e.g. 1q1g: 21 atoms, 1u1c: 0),
    which mispositions the pocket center.
    """
    prot_pdb = complex_dir / f"{pdb_id}_protein.pdb"
    pocket_pdb = complex_dir / f"{pdb_id}_pocket.pdb"
    lig_sdf = complex_dir / f"{pdb_id}_ligand.sdf"
    lig_mol2 = complex_dir / f"{pdb_id}_ligand.mol2"

    for prot in (prot_pdb, pocket_pdb):
        if not prot.exists():
            continue
        if lig_sdf.exists():
            return prot, lig_sdf, "sdf"
        if lig_mol2.exists():
            return prot, lig_mol2, "mol2"
    return None


def detect_dataset_name(data_dir: Path) -> str:
    name = data_dir.name.lower()
    if "posebusters" in name:
        return "posebusters"
    if "astex" in name:
        return "astex"
    return data_dir.name


# ---------------------------------------------------------------------------
# Robust RDKit loaders
# ---------------------------------------------------------------------------
_RELAXED_SANITIZE = (
    Chem.SanitizeFlags.SANITIZE_FINDRADICALS
    | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
    | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
    | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
    | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
)


def _finalize_mol(mol: Chem.Mol, src: str) -> Chem.Mol:
    frags = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumAtoms())
    assert mol.GetNumConformers() > 0, f"No 3D conformer in {src}"
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return mol


def load_sdf_robust(sdf_path: Path) -> Chem.Mol:
    suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=True, removeHs=True)
    mol = next(suppl)
    if mol is not None:
        frags = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumAtoms())
        return mol

    suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
    mol = next(suppl)
    assert mol is not None, f"RDKit cannot parse {sdf_path}"
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    Chem.SanitizeMol(mol, sanitizeOps=_RELAXED_SANITIZE)
    mol = Chem.RemoveHs(mol, sanitize=False)
    return _finalize_mol(mol, str(sdf_path))


def load_mol2_robust(mol2_path: Path) -> Chem.Mol:
    mol = Chem.MolFromMol2File(str(mol2_path), sanitize=True)
    if mol is not None:
        mol = Chem.RemoveHs(mol)
        frags = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumAtoms())
        return mol

    mol = Chem.MolFromMol2File(str(mol2_path), sanitize=False, removeHs=False)
    assert mol is not None, f"RDKit cannot parse {mol2_path}"
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    Chem.SanitizeMol(mol, sanitizeOps=_RELAXED_SANITIZE)
    mol = Chem.RemoveHs(mol, sanitize=False)
    return _finalize_mol(mol, str(mol2_path))


def load_ligand(path: Path, fmt: str) -> Chem.Mol:
    if fmt == "sdf":
        return load_sdf_robust(path)
    if fmt == "mol2":
        return load_mol2_robust(path)
    raise ValueError(f"Unknown ligand format: {fmt}")


# ---------------------------------------------------------------------------
# Refinement
# ---------------------------------------------------------------------------
def mmff_refine(
    mol: Chem.Mol,
    pred_pos: torch.Tensor,
    pocket_center: torch.Tensor,
    max_iters: int = 200,
) -> torch.Tensor:
    """Relax a predicted pose with MMFF94s, tethering heavy atoms.

    Unconstrained minimization pulls the molecule back to its gas-phase
    minimum and can translate it tens of Å; tethering heavy atoms keeps the
    binding-pose translation fixed while removing local geometric strain.
    """
    try:
        mol_copy = Chem.RWMol(mol)
        mol_copy.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol_copy)
        conf = mol_copy.GetConformer()
        pos_abs = pred_pos + pocket_center
        n_heavy = mol_copy.GetNumAtoms()
        for j in range(n_heavy):
            conf.SetAtomPosition(j, pos_abs[j].tolist())

        mol_h = Chem.AddHs(mol_copy, addCoords=True, addResidueInfo=False)

        props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94s")
        if props is None:
            return pred_pos
        ff = AllChem.MMFFGetMoleculeForceField(mol_h, props, confId=0)
        if ff is None:
            return pred_pos

        for j in range(n_heavy):
            ff.MMFFAddPositionConstraint(j, 0.5, 50.0)

        ff.Minimize(maxIts=max_iters)

        refined = torch.zeros_like(pred_pos)
        conf = mol_h.GetConformer()
        for j in range(n_heavy):
            p = conf.GetAtomPosition(j)
            refined[j] = torch.tensor([p.x, p.y, p.z])
        return refined - pocket_center
    except Exception:
        return pred_pos


def apply_refinement(
    method: str,
    poses: list[torch.Tensor],
    mol: Chem.Mol,
    pocket_center: torch.Tensor,
) -> list[torch.Tensor]:
    if method == "none":
        return poses
    return [mmff_refine(mol, pos, pocket_center) for pos in poses]


# ---------------------------------------------------------------------------
# RMSD + matching
# ---------------------------------------------------------------------------
def compute_rmsd(pred: torch.Tensor, ref: torch.Tensor) -> float:
    """Index-wise RMSD (thin float wrapper around :func:`metrics.ligand_rmsd`)."""
    from effdock.evaluation.metrics import ligand_rmsd

    return float(ligand_rmsd(pred, ref))


def compute_centroid_dist(pred: torch.Tensor, ref: torch.Tensor) -> float:
    from effdock.evaluation.metrics import centroid_distance

    return float(centroid_distance(pred, ref))


def compute_pose_rmsd_with_method(
    pose: torch.Tensor,
    ref_pos: torch.Tensor,
    pocket_center: torch.Tensor,
    dock_idx: list[int],
    mol_dock: Chem.Mol,
    mol_ref: Chem.Mol,
) -> tuple[float, str]:
    """Return pose RMSD and the exact symmetry/fallback calculation path.

    Uses RDKit ``rdMolAlign.CalcRMS`` when topology matches; otherwise falls
    back to index-based RMSD on the matched atom subset.
    """
    if len(dock_idx) == mol_dock.GetNumAtoms() == mol_ref.GetNumAtoms():
        try:
            mol_pose = Chem.RWMol(mol_dock)
            conf = mol_pose.GetConformer()
            pose_abs = pose + pocket_center
            for i in range(mol_dock.GetNumAtoms()):
                conf.SetAtomPosition(i, pose_abs[i].tolist())
            return (
                float(rdMolAlign.CalcRMS(mol_pose, mol_ref)),
                "rdkit_calc_rms_symmetry_no_align",
            )
        except Exception:
            pass
    dock_idx_t = torch.as_tensor(dock_idx, dtype=torch.long)
    return (
        compute_rmsd(pose.index_select(0, dock_idx_t), ref_pos),
        "mapped_index_fallback",
    )


def compute_pose_rmsd(
    pose: torch.Tensor,
    ref_pos: torch.Tensor,
    pocket_center: torch.Tensor,
    dock_idx: list[int],
    mol_dock: Chem.Mol,
    mol_ref: Chem.Mol,
) -> float:
    """Compatibility wrapper returning only the pose RMSD value."""
    value, _ = compute_pose_rmsd_with_method(
        pose,
        ref_pos,
        pocket_center,
        dock_idx,
        mol_dock,
        mol_ref,
    )
    return value


def _strip_charges(mol: Chem.Mol) -> Chem.Mol:
    m = Chem.RWMol(mol)
    for atom in m.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNumRadicalElectrons(0)
    try:
        Chem.SanitizeMol(
            m,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
        )
    except Exception:
        pass
    return m


def match_atoms(mol_ref: Chem.Mol, mol_dock: Chem.Mol) -> tuple[list[int], list[int], str]:
    """Map atoms between crystal and docking mols.

    Returns ``(dock_indices, ref_indices, method)``; both lists have equal
    length (``dock_indices[i]`` ↔ ``ref_indices[i]``). Falls back from strict
    → charge-agnostic → MCS. Empty lists signal no overlap.
    """
    match = mol_ref.GetSubstructMatch(mol_dock)
    if len(match) == mol_dock.GetNumAtoms():
        return list(range(mol_dock.GetNumAtoms())), list(match), "strict"

    ref2, dock2 = _strip_charges(mol_ref), _strip_charges(mol_dock)
    match = ref2.GetSubstructMatch(dock2)
    if len(match) == dock2.GetNumAtoms():
        return list(range(mol_dock.GetNumAtoms())), list(match), "nocharges"

    mcs = rdFMCS.FindMCS(
        [ref2, dock2],
        timeout=5,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        matchValences=False,
        ringMatchesRingOnly=False,
    )
    if mcs.numAtoms == 0:
        return [], [], "fail"
    patt = Chem.MolFromSmarts(mcs.smartsString)
    ref_m = ref2.GetSubstructMatch(patt)
    dock_m = dock2.GetSubstructMatch(patt)
    if len(ref_m) == len(dock_m) == mcs.numAtoms:
        return list(dock_m), list(ref_m), f"mcs({mcs.numAtoms}/{mol_dock.GetNumAtoms()})"
    return [], [], "fail"


# ---------------------------------------------------------------------------
# Selection + stats
# ---------------------------------------------------------------------------
def select_by_score(scores) -> int:
    """Generic argmin over a score array (lower = better).

    Use for any score-based selector such as Vina free energy. Callers are responsible for sign-flipping
    when a higher score means a better pose.
    """
    return int(np.argmin(scores))


def select_pose(method: str, rmsds: list[float]) -> int:
    """RMSD-based oracle selection only.

    Vina selection requires an external free-energy score array; use ``select_by_score`` for
    those instead of overloading this function.
    """
    if method == "oracle":
        return select_by_score(rmsds)
    raise ValueError(
        f"select_pose(method='{method}') is not RMSD-based. "
        f"Use select_by_score(score_array) with the appropriate score "
        f"(for example, Vina energy) instead."
    )


def compute_stats(rmsds: np.ndarray) -> dict:
    return {
        "mean_rmsd": float(rmsds.mean()),
        "median_rmsd": float(np.median(rmsds)),
        "std_rmsd": float(rmsds.std()),
        "pct_lt_1A": float((rmsds < 1.0).mean() * 100),
        "pct_lt_2A": float((rmsds < 2.0).mean() * 100),
        "pct_lt_3A": float((rmsds < 3.0).mean() * 100),
        "pct_lt_5A": float((rmsds < 5.0).mean() * 100),
    }


__all__ = [
    "REFINE_METHODS",
    "SELECT_METHODS",
    "V2_IDS_PATH",
    "detect_complex_files",
    "detect_dataset_name",
    "load_sdf_robust",
    "load_mol2_robust",
    "load_ligand",
    "mmff_refine",
    "apply_refinement",
    "compute_rmsd",
    "compute_centroid_dist",
    "compute_pose_rmsd",
    "match_atoms",
    "select_pose",
    "select_by_score",
    "compute_stats",
]
