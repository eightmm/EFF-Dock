"""Ligand preprocessing: SDF/MOL2 → atom/bond feature tensors."""

import logging
import os
from pathlib import Path

import torch
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, rdDistGeom, rdmolops

log = logging.getLogger(__name__)


# Element vocab: common drug-like elements + OTHER
ELEMENT_VOCAB: dict[int, int] = {
    6: 0,  # C
    7: 1,  # N
    8: 2,  # O
    16: 3,  # S
    15: 4,  # P
    9: 5,  # F
    17: 6,  # Cl
    35: 7,  # Br
    53: 8,  # I
    5: 9,  # B
    14: 10,  # Si
    34: 11,  # Se
}
OTHER_ELEMENT_IDX = 12
NUM_ELEMENTS = 13

# Hybridization vocab (UNSPECIFIED kept as a distinct slot so it doesn't
# collapse onto OTHER, which hides sanitize-degraded atoms)
HYBRIDIZATION_MAP: dict[Chem.rdchem.HybridizationType, int] = {
    Chem.rdchem.HybridizationType.SP: 0,
    Chem.rdchem.HybridizationType.SP2: 1,
    Chem.rdchem.HybridizationType.SP3: 2,
    Chem.rdchem.HybridizationType.SP3D: 3,
    Chem.rdchem.HybridizationType.SP3D2: 4,
    Chem.rdchem.HybridizationType.UNSPECIFIED: 5,
}
OTHER_HYBRID_IDX = 6
NUM_HYBRIDIZATIONS = 7

# Bond type vocab
BOND_TYPE_MAP: dict[Chem.rdchem.BondType, int] = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}
OTHER_BOND_IDX = 4
NUM_BOND_TYPES = 5

# Chirality vocab
CHIRALITY_MAP: dict[Chem.rdchem.ChiralType, int] = {
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 0,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
    Chem.rdchem.ChiralType.CHI_OTHER: 3,
}

# Bond stereo vocab
BOND_STEREO_MAP: dict[Chem.rdchem.BondStereo, int] = {
    Chem.rdchem.BondStereo.STEREONONE: 0,
    Chem.rdchem.BondStereo.STEREOANY: 1,
    Chem.rdchem.BondStereo.STEREOZ: 2,
    Chem.rdchem.BondStereo.STEREOE: 3,
    Chem.rdchem.BondStereo.STEREOCIS: 4,
    Chem.rdchem.BondStereo.STEREOTRANS: 5,
}

_FDEF_PATH = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
_FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(_FDEF_PATH)

# Map RDKit canonical pharmacophore families to our per-atom feature keys.
# Halogen is element-based (not a fdef family) and handled separately.
FAMILY_TO_FEATURE: dict[str, str] = {
    "Donor": "atom_is_donor",
    "Acceptor": "atom_is_acceptor",
    "PosIonizable": "atom_is_positive",
    "NegIonizable": "atom_is_negative",
    "Hydrophobe": "atom_is_hydrophobe",
}

HALOGEN_ATOMIC_NUMS: set[int] = {9, 17, 35, 53}


def _get_atom_valence(atom: Chem.Atom, explicit: bool) -> int:
    """Return atom valence while supporting old/new RDKit APIs."""
    try:
        which = Chem.rdchem.ValenceType.EXPLICIT if explicit else Chem.rdchem.ValenceType.IMPLICIT
        return int(atom.GetValence(which))
    except Exception:
        if explicit:
            return int(atom.GetExplicitValence())
        return int(atom.GetImplicitValence())


def compute_atom_pharmacophore_features(mol: Chem.Mol) -> dict[str, torch.Tensor]:
    """Compute per-atom pharmacophore flags from RDKit BaseFeatures.fdef.

    Uses RDKit's canonical feature factory (same as pharmacophore search),
    covering donor/acceptor/positive/negative/hydrophobe families. Halogen is
    computed directly from atomic numbers.

    On feature-factory failure (e.g., incomplete sanitization), all pharmacophore
    flags fall back to False; halogen is still computed. Callers should check
    ``sanitize_ok`` in meta before trusting these features.
    """
    n_atoms = mol.GetNumAtoms()
    result: dict[str, torch.Tensor] = {
        key: torch.zeros(n_atoms, dtype=torch.bool) for key in FAMILY_TO_FEATURE.values()
    }
    result["atom_is_halogen"] = torch.zeros(n_atoms, dtype=torch.bool)

    try:
        features = _FEATURE_FACTORY.GetFeaturesForMol(mol)
    except Exception:
        features = []

    for feat in features:
        key = FAMILY_TO_FEATURE.get(feat.GetFamily())
        if key is None:
            continue
        for atom_idx in feat.GetAtomIds():
            result[key][atom_idx] = True

    for i in range(n_atoms):
        if mol.GetAtomWithIdx(i).GetAtomicNum() in HALOGEN_ATOMIC_NUMS:
            result["atom_is_halogen"][i] = True

    return result


def load_molecule(
    sdf_path: Path | None, mol2_path: Path | None = None
) -> tuple[Chem.Mol | None, bool, bool]:
    """Load molecule from SDF, fallback to MOL2.

    Returns:
        mol: Sanitized RDKit mol with hydrogens removed, or None on failure.
        used_mol2_fallback: True if SDF failed (or was missing) and MOL2 was used.
        sanitize_ok: True only if full sanitization succeeded. False means
            partial sanitization (properties skipped) — downstream chemistry
            features (aromatic, hybridization, conjugated, pharmacophore) may
            be unreliable and the caller should record this flag in meta.
    """
    if sdf_path is not None and sdf_path.exists():
        mol, sanitize_ok = _try_load_sdf(sdf_path)
    else:
        mol, sanitize_ok = None, False
    used_fallback = False

    if mol is None and mol2_path is not None and mol2_path.exists():
        raw = Chem.MolFromMol2File(str(mol2_path), sanitize=False)
        if raw is not None:
            mol, sanitize_ok = _try_sanitize(raw)
            used_fallback = True

    if mol is None:
        return None, used_fallback, sanitize_ok

    # Assign stereo from 3D coords BEFORE removing hydrogens — E/Z and chiral
    # tags that depend on H positions would otherwise be lost.
    try:
        Chem.AssignStereochemistryFrom3D(mol)
    except Exception:
        sanitize_ok = False

    mol = Chem.RemoveHs(mol)
    mol = _keep_largest_component(mol, sdf_path)

    if mol.GetNumConformers() == 0:
        return None, used_fallback, sanitize_ok
    conf = mol.GetConformer()
    if not conf.Is3D():
        return None, used_fallback, sanitize_ok

    return mol, used_fallback, sanitize_ok


def featurize_ligand(mol: Chem.Mol) -> dict[str, torch.Tensor] | None:
    """Extract minimal atom/bond features from RDKit mol.

    Returns dict with:
        atom_coords: [N_atom, 3] float32
        atom_element: [N_atom] int64
        atom_charge: [N_atom] int8
        atom_aromatic: [N_atom] bool
        atom_hybridization: [N_atom] int8
        atom_num_rings: [N_atom] int8  (membership = num_rings > 0)
        bond_index: [2, E_dir] int64 (directed, E_dir = 2 * N_bonds)
        bond_type: [E_dir] int8
        bond_conjugated: [E_dir] bool
        bond_in_ring: [E_dir] bool

    Returns None if molecule has < 2 atoms.
    """
    n_atoms = mol.GetNumAtoms()
    if n_atoms < 2:
        return None

    conf = mol.GetConformer()
    pharmacophore_features = compute_atom_pharmacophore_features(mol)

    # Ensure ring info is populated even if the mol came through the relaxed
    # sanitize path (which skips property setting / ring perception).
    try:
        Chem.GetSSSR(mol)
    except Exception:
        pass

    # Atom features
    coords = []
    elements = []
    charges = []
    aromatics = []
    hybridizations = []
    degrees = []
    implicit_valences = []
    explicit_valences = []
    num_rings_list = []
    chiralities = []

    ring_info = mol.GetRingInfo()

    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        pos = conf.GetAtomPosition(i)

        coords.append([pos.x, pos.y, pos.z])
        elements.append(ELEMENT_VOCAB.get(atom.GetAtomicNum(), OTHER_ELEMENT_IDX))
        charges.append(atom.GetFormalCharge())
        aromatics.append(atom.GetIsAromatic())
        hybridizations.append(HYBRIDIZATION_MAP.get(atom.GetHybridization(), OTHER_HYBRID_IDX))
        try:
            nr = ring_info.NumAtomRings(i)
        except Exception:
            nr = 0
        degrees.append(atom.GetDegree())
        implicit_valences.append(_get_atom_valence(atom, explicit=False))
        explicit_valences.append(_get_atom_valence(atom, explicit=True))
        num_rings_list.append(nr)
        chiralities.append(CHIRALITY_MAP.get(atom.GetChiralTag(), 0))

    # Bond features (directed: each bond stored twice)
    src_list = []
    dst_list = []
    bond_types = []
    bond_conj = []
    bond_rings = []
    bond_stereos = []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = BOND_TYPE_MAP.get(bond.GetBondType(), OTHER_BOND_IDX)
        conj = bond.GetIsConjugated()
        in_ring = bond.IsInRing()
        stereo = BOND_STEREO_MAP.get(bond.GetStereo(), 0)

        # Forward
        src_list.append(i)
        dst_list.append(j)
        bond_types.append(bt)
        bond_conj.append(conj)
        bond_rings.append(in_ring)
        bond_stereos.append(stereo)

        # Backward
        src_list.append(j)
        dst_list.append(i)
        bond_types.append(bt)
        bond_conj.append(conj)
        bond_rings.append(in_ring)
        bond_stereos.append(stereo)

    coords_t = torch.tensor(coords, dtype=torch.float32)
    if not torch.isfinite(coords_t).all():
        return None
    if any(v < 0 or v > 120 for v in explicit_valences + implicit_valences):
        return None

    return {
        "atom_coords": coords_t,
        "atom_element": torch.tensor(elements, dtype=torch.int64),
        "atom_charge": torch.tensor(charges, dtype=torch.int8),
        "atom_aromatic": torch.tensor(aromatics, dtype=torch.bool),
        "atom_hybridization": torch.tensor(hybridizations, dtype=torch.int8),
        "atom_degree": torch.tensor(degrees, dtype=torch.int8),
        "atom_implicit_valence": torch.tensor(implicit_valences, dtype=torch.int8),
        "atom_explicit_valence": torch.tensor(explicit_valences, dtype=torch.int8),
        "atom_num_rings": torch.tensor(num_rings_list, dtype=torch.int8),
        "atom_chirality": torch.tensor(chiralities, dtype=torch.int8),
        **pharmacophore_features,
        "bond_index": torch.tensor([src_list, dst_list], dtype=torch.int64),
        "bond_type": torch.tensor(bond_types, dtype=torch.int8),
        "bond_conjugated": torch.tensor(bond_conj, dtype=torch.bool),
        "bond_in_ring": torch.tensor(bond_rings, dtype=torch.bool),
        "bond_stereo": torch.tensor(bond_stereos, dtype=torch.int8),
    }


def compute_dg_bounds(mol: Chem.Mol) -> torch.Tensor | None:
    """Compute raw distance geometry bounds matrix.

    Returns [N, N] float32 tensor where:
        upper triangle (i < j) = upper bound
        lower triangle (i > j) = lower bound
        diagonal = 0

    Raw values from RDKit (triangle-inequality smoothed), no margin applied.
    Margin should be applied at training time in the loss function.
    """
    try:
        bounds = rdDistGeom.GetMoleculeBoundsMatrix(mol)
    except Exception:
        return None

    return torch.from_numpy(bounds).float()


def _try_load_sdf(path: Path) -> tuple[Chem.Mol | None, bool]:
    """Load SDF and sanitize. Warns if the file contains multiple molecules.

    Returns (mol, sanitize_ok). ``mol`` is None on total failure.
    """
    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    try:
        n_entries = len(supplier)
    except Exception:
        n_entries = -1
    if n_entries > 1:
        log.warning(
            "SDF %s contains %d entries; using the first valid molecule.",
            path,
            n_entries,
        )

    for mol in supplier:
        if mol is not None:
            return _try_sanitize(mol)
    return None, False


def _try_sanitize(mol: Chem.Mol) -> tuple[Chem.Mol | None, bool]:
    """Sanitize mol. Returns (mol, sanitize_ok).

    - Full success: (mol, True)
    - Partial success (properties skipped): (mol, False) — downstream chemistry
      features may be unreliable.
    - Total failure: (None, False).
    """
    try:
        Chem.SanitizeMol(mol)
        return mol, True
    except Exception:
        pass

    try:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
        )
        return mol, False
    except Exception:
        return None, False


def _keep_largest_component(mol: Chem.Mol, source: Path | None = None) -> Chem.Mol:
    """Keep only the largest connected component, warning on multi-fragment."""
    frags = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) <= 1:
        return mol
    sizes = [m.GetNumAtoms() for m in frags]
    log.warning(
        "Multi-fragment molecule%s: %d components with sizes %s; keeping largest.",
        f" in {source}" if source is not None else "",
        len(frags),
        sizes,
    )
    return max(frags, key=lambda m: m.GetNumAtoms())
