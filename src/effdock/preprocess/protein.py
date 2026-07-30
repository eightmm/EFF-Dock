"""Protein preprocessing: token-based pocket parsing with canonical bonds.

Design choices (protein is static, so we avoid RDKit sanitize entirely):
- Line-based PDB parse; water/nucleic-acid/altloc/hydrogen filtered out.
- Per-atom identity collapses into a single (residue, atom_name) token; that
  token implicitly encodes element, hybridization, aromaticity, pharmacophore,
  and backbone/sidechain for all standard amino acids.
- Protein intra bonds are rebuilt deterministically from a hardcoded topology
  table per standard AA + peptide bonds (C(i) to N(i+1) within the same chain)
  + disulfide bonds (CYS SG - CYS SG within 2.5 Å).
- Metal ions get per-element tokens and ``is_metal=True``.
"""

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Residue-level constants
# ---------------------------------------------------------------------------

AA3_TO_IDX: dict[str, int] = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLN": 5,
    "GLU": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
}
UNK_RES_IDX = 20
METAL_RES_IDX = 21
NUM_RES_TYPES = 22

# PTM / variant residues → parent amino acid
PTM_MAPPING: dict[str, str] = {
    "MSE": "MET",
    "SEP": "SER",
    "TPO": "THR",
    "PTR": "TYR",
    "HID": "HIS",
    "HIE": "HIS",
    "HIP": "HIS",
    "CYX": "CYS",
    "CYM": "CYS",
    "CME": "CYS",
    "OCS": "CYS",
    "ASH": "ASP",
    "GLH": "GLU",
    "KCX": "LYS",
    "LLP": "LYS",
    "MLZ": "LYS",
    "M3L": "LYS",
}

# Residues we intentionally drop
WATER_RESIDUES: set[str] = {"HOH", "WAT", "DOD", "H2O", "TIP3", "TIP4"}
NUCLEIC_ACID_RESIDUES: set[str] = {
    "DA",
    "DC",
    "DG",
    "DT",
    "DU",
    "A",
    "C",
    "G",
    "T",
    "U",
    "RA",
    "RC",
    "RG",
    "RT",
    "RU",
}

# Metal elements kept as ions (both residue-name and element)
METAL_ELEMENTS: set[str] = {
    "ZN",
    "MG",
    "CA",
    "FE",
    "MN",
    "CU",
    "CO",
    "NI",
    "NA",
    "K",
    "LI",
    "CS",
    "RB",
    "SR",
    "BA",
    "HG",
    "CD",
    "PT",
    "PD",
    "AU",
    "AG",
}

BACKBONE_ATOMS: set[str] = {"N", "CA", "C", "O"}

# ---------------------------------------------------------------------------
# Per-atom pharmacophore lookup (mirror of ligand-side features)
# Backbone is handled via universal rules at runtime:
#   - backbone N → donor (PRO excluded — cyclic, no NH)
#   - backbone O → acceptor (universal)
# Side-chain atoms below carry their own flags.
# ---------------------------------------------------------------------------
SC_DONOR_ATOMS: set[tuple[str, str]] = {
    ("ARG", "NE"),
    ("ARG", "NH1"),
    ("ARG", "NH2"),
    ("ASN", "ND2"),
    ("GLN", "NE2"),
    ("HIS", "ND1"),
    ("HIS", "NE2"),  # both possible tautomers
    ("LYS", "NZ"),
    ("SER", "OG"),
    ("THR", "OG1"),
    ("TYR", "OH"),
    ("TRP", "NE1"),
    ("CYS", "SG"),  # weak
}

SC_ACCEPTOR_ATOMS: set[tuple[str, str]] = {
    ("ASN", "OD1"),
    ("ASP", "OD1"),
    ("ASP", "OD2"),
    ("GLN", "OE1"),
    ("GLU", "OE1"),
    ("GLU", "OE2"),
    ("HIS", "ND1"),
    ("HIS", "NE2"),
    ("SER", "OG"),
    ("THR", "OG1"),
    ("TYR", "OH"),
    ("MET", "SD"),
    ("CYS", "SG"),  # weak
}

SC_POSITIVE_ATOMS: set[tuple[str, str]] = {
    ("ARG", "NE"),
    ("ARG", "NH1"),
    ("ARG", "NH2"),
    ("ARG", "CZ"),
    ("LYS", "NZ"),
    ("HIS", "ND1"),
    ("HIS", "NE2"),  # only ~10% protonated at pH 7.4 but kept for completeness
}

SC_NEGATIVE_ATOMS: set[tuple[str, str]] = {
    ("ASP", "OD1"),
    ("ASP", "OD2"),
    ("GLU", "OE1"),
    ("GLU", "OE2"),
}

SC_HYDROPHOBIC_ATOMS: set[tuple[str, str]] = {
    ("ALA", "CB"),
    ("VAL", "CG1"),
    ("VAL", "CG2"),
    ("LEU", "CG"),
    ("LEU", "CD1"),
    ("LEU", "CD2"),
    ("ILE", "CB"),
    ("ILE", "CG1"),
    ("ILE", "CG2"),
    ("ILE", "CD1"),
    ("MET", "CG"),
    ("MET", "SD"),
    ("MET", "CE"),
    ("PRO", "CB"),
    ("PRO", "CG"),
    ("PRO", "CD"),
    ("PHE", "CG"),
    ("PHE", "CD1"),
    ("PHE", "CD2"),
    ("PHE", "CE1"),
    ("PHE", "CE2"),
    ("PHE", "CZ"),
    ("TRP", "CG"),
    ("TRP", "CD2"),
    ("TRP", "CE2"),
    ("TRP", "CE3"),
    ("TRP", "CZ2"),
    ("TRP", "CZ3"),
    ("TRP", "CH2"),
    ("TYR", "CG"),
    ("TYR", "CD1"),
    ("TYR", "CD2"),
    ("TYR", "CE1"),
    ("TYR", "CE2"),
    ("CYS", "CB"),
}


def _patom_pharmacophore(res_name: str, atom_name: str) -> tuple[bool, bool, bool, bool, bool]:
    """Return (donor, acceptor, positive, negative, hydrophobic) for one heavy atom.

    Backbone N → donor (except PRO).  Backbone O → acceptor (universal).
    Side-chain atoms use ``SC_*`` lookup tables. Metal atoms are positive.
    """
    is_donor = is_acceptor = is_positive = is_negative = is_hydrophobic = False
    if atom_name == "N" and res_name != "PRO":
        is_donor = True
    if atom_name == "O" or atom_name == "OXT":
        is_acceptor = True
    key = (res_name, atom_name)
    if key in SC_DONOR_ATOMS:
        is_donor = True
    if key in SC_ACCEPTOR_ATOMS:
        is_acceptor = True
    if key in SC_POSITIVE_ATOMS:
        is_positive = True
    if key in SC_NEGATIVE_ATOMS:
        is_negative = True
    if key in SC_HYDROPHOBIC_ATOMS:
        is_hydrophobic = True
    return is_donor, is_acceptor, is_positive, is_negative, is_hydrophobic


# ---------------------------------------------------------------------------
# Canonical heavy-atom topology for the 20 standard amino acids
# ---------------------------------------------------------------------------

STANDARD_AA_ATOMS: dict[str, list[str]] = {
    "ALA": ["N", "CA", "C", "O", "CB"],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "GLY": ["N", "CA", "C", "O"],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
    "TRP": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
    ],
    "TYR": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "OH",
    ],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
}

STANDARD_AA_BONDS: dict[str, list[tuple[str, str]]] = {
    "ALA": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB")],
    "ARG": [
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD"),
        ("CD", "NE"),
        ("NE", "CZ"),
        ("CZ", "NH1"),
        ("CZ", "NH2"),
    ],
    "ASN": [
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "OD1"),
        ("CG", "ND2"),
    ],
    "ASP": [
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "OD1"),
        ("CG", "OD2"),
    ],
    "CYS": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "SG")],
    "GLN": [
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD"),
        ("CD", "OE1"),
        ("CD", "NE2"),
    ],
    "GLU": [
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD"),
        ("CD", "OE1"),
        ("CD", "OE2"),
    ],
    "GLY": [("N", "CA"), ("CA", "C"), ("C", "O")],
    "HIS": [
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "ND1"),
        ("CG", "CD2"),
        ("ND1", "CE1"),
        ("CD2", "NE2"),
        ("CE1", "NE2"),
    ],
    "ILE": [
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("CA", "CB"),
        ("CB", "CG1"),
        ("CB", "CG2"),
        ("CG1", "CD1"),
    ],
    "LEU": [
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD1"),
        ("CG", "CD2"),
    ],
    "LYS": [
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD"),
        ("CD", "CE"),
        ("CE", "NZ"),
    ],
    "MET": [
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "SD"),
        ("SD", "CE"),
    ],
    "PHE": [
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD1"),
        ("CG", "CD2"),
        ("CD1", "CE1"),
        ("CD2", "CE2"),
        ("CE1", "CZ"),
        ("CE2", "CZ"),
    ],
    "PRO": [
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD"),
        ("CD", "N"),
    ],
    "SER": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "OG")],
    "THR": [
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("CA", "CB"),
        ("CB", "OG1"),
        ("CB", "CG2"),
    ],
    "TRP": [
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD1"),
        ("CG", "CD2"),
        ("CD1", "NE1"),
        ("CD2", "CE2"),
        ("CD2", "CE3"),
        ("NE1", "CE2"),
        ("CE2", "CZ2"),
        ("CZ2", "CH2"),
        ("CH2", "CZ3"),
        ("CZ3", "CE3"),
    ],
    "TYR": [
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD1"),
        ("CG", "CD2"),
        ("CD1", "CE1"),
        ("CD2", "CE2"),
        ("CE1", "CZ"),
        ("CE2", "CZ"),
        ("CZ", "OH"),
    ],
    "VAL": [
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("CA", "CB"),
        ("CB", "CG1"),
        ("CB", "CG2"),
    ],
}

# ---------------------------------------------------------------------------
# Build (residue, atom) → token table
# ---------------------------------------------------------------------------

RES_ATOM_TOKEN: dict[tuple[str, str], int] = OrderedDict()
for _aa in [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]:
    for _atom in STANDARD_AA_ATOMS[_aa]:
        RES_ATOM_TOKEN[(_aa, _atom)] = len(RES_ATOM_TOKEN)

# Shared specials
OXT_TOKEN = len(RES_ATOM_TOKEN)
RES_ATOM_TOKEN[("ANY", "OXT")] = OXT_TOKEN

# UNK residue backbone fallbacks
for _atom in ["N", "CA", "C", "O", "CB"]:
    RES_ATOM_TOKEN[("UNK", _atom)] = len(RES_ATOM_TOKEN)

# Catchall for unknown (res, atom) pairs
UNK_ATOM_TOKEN = len(RES_ATOM_TOKEN)

# Per-element metal tokens
_METAL_TOKEN_ORDER: list[str] = [
    "ZN",
    "MG",
    "CA",
    "FE",
    "MN",
    "CU",
    "CO",
    "NI",
    "NA",
    "K",
]
METAL_ATOM_TOKENS: dict[str, int] = {}
for _idx, _elem in enumerate(_METAL_TOKEN_ORDER):
    METAL_ATOM_TOKENS[_elem] = UNK_ATOM_TOKEN + 1 + _idx
METAL_OTHER_TOKEN = UNK_ATOM_TOKEN + 1 + len(_METAL_TOKEN_ORDER)

NUM_ATOM_TOKENS = METAL_OTHER_TOKEN + 1


def _get_res_atom_token(res_name: str, atom_name: str) -> int:
    """Resolve (res, atom) → token with graceful fallbacks."""
    key = (res_name, atom_name)
    if key in RES_ATOM_TOKEN:
        return RES_ATOM_TOKEN[key]
    if atom_name == "OXT":
        return OXT_TOKEN
    unk_key = ("UNK", atom_name)
    if unk_key in RES_ATOM_TOKEN:
        return RES_ATOM_TOKEN[unk_key]
    return UNK_ATOM_TOKEN


# ---------------------------------------------------------------------------
# Line-based PDB parsing
# ---------------------------------------------------------------------------


@dataclass
class ParsedAtom:
    record_type: str
    atom_name: str
    res_name: str  # post-PTM mapping for AAs, original for metals
    chain: str
    res_num: int
    icode: str
    coords: tuple[float, float, float]
    element: str
    is_metal: bool
    raw_res_name: str | None = None


def _compute_pseudo_cb(
    n: tuple[float, float, float],
    ca: tuple[float, float, float],
    c: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Idealized CB placement for glycine (Rosetta-style tetrahedral geometry).

    Given N, CA, C backbone positions, returns a pseudo-CB coordinate that sits
    where the CB would be if the residue were not GLY. Used as a surrogate
    virtual-node position so every residue has a consistent ``pres_coords``.
    """
    b = (ca[0] - n[0], ca[1] - n[1], ca[2] - n[2])
    c_vec = (c[0] - ca[0], c[1] - ca[1], c[2] - ca[2])
    a = (
        b[1] * c_vec[2] - b[2] * c_vec[1],
        b[2] * c_vec[0] - b[0] * c_vec[2],
        b[0] * c_vec[1] - b[1] * c_vec[0],
    )
    # Rosetta idealized CB coefficients
    return (
        -0.58273431 * a[0] + 0.56802827 * b[0] - 0.54067466 * c_vec[0] + ca[0],
        -0.58273431 * a[1] + 0.56802827 * b[1] - 0.54067466 * c_vec[1] + ca[1],
        -0.58273431 * a[2] + 0.56802827 * b[2] - 0.54067466 * c_vec[2] + ca[2],
    )


def _infer_element(atom_name: str, raw_element_col: str) -> str:
    if raw_element_col:
        return raw_element_col.strip().upper()
    stripped = atom_name.lstrip("0123456789").upper()
    if len(stripped) >= 2 and stripped[:2] in METAL_ELEMENTS:
        return stripped[:2]
    return stripped[:1] if stripped else ""


def _parse_pdb_lines(pdb_path: Path) -> list[ParsedAtom]:
    """Line-level PDB parse with filtering (HOH/NA/altloc/hydrogens)."""
    atoms: list[ParsedAtom] = []
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            altloc = line[16]
            if altloc not in (" ", "A"):
                continue

            res_name_raw = line[17:20].strip()
            atom_name = line[12:16].strip()

            raw_element = line[76:78] if len(line) >= 78 else ""
            element = _infer_element(atom_name, raw_element)
            if element == "H":
                continue

            if res_name_raw in WATER_RESIDUES:
                continue
            if res_name_raw in NUCLEIC_ACID_RESIDUES:
                continue

            # Metal ion: treat residue as metal
            is_metal = res_name_raw in METAL_ELEMENTS or element in METAL_ELEMENTS

            if not is_metal:
                res_name = PTM_MAPPING.get(res_name_raw, res_name_raw)
            else:
                res_name = res_name_raw

            chain = line[21]
            try:
                res_num = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            icode = line[26].strip()

            atoms.append(
                ParsedAtom(
                    record_type=line[0:6].strip(),
                    atom_name=atom_name,
                    res_name=res_name,
                    chain=chain,
                    res_num=res_num,
                    icode=icode,
                    coords=(x, y, z),
                    element=element,
                    is_metal=is_metal,
                    raw_res_name=res_name_raw,
                )
            )
    return atoms


# ---------------------------------------------------------------------------
# Public parsers
# ---------------------------------------------------------------------------


def parse_pocket_pdb(pdb_path: Path) -> dict[str, torch.Tensor] | None:
    """Parse pocket PDB file, extract CA coords and residue types.

    Kept for backward compatibility with tests / scripts that only need
    residue-level CA features.

    Returns dict with:
        res_coords: [N_res, 3] float32
        res_type:   [N_res]    int64 (0-19, 20=UNK)

    Returns None when no valid residues are found.
    """
    ca_coords: list[list[float]] = []
    res_types: list[int] = []
    seen: set[tuple[str, int, str]] = set()

    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue

            res_name = line[17:20].strip()
            if res_name in WATER_RESIDUES or res_name in NUCLEIC_ACID_RESIDUES:
                continue

            mapped = PTM_MAPPING.get(res_name, res_name)
            if mapped not in AA3_TO_IDX:
                continue

            try:
                res_num = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            icode = line[26].strip()
            chain = line[21]

            key = (chain, res_num, icode)
            if key in seen:
                continue
            seen.add(key)

            ca_coords.append([x, y, z])
            res_types.append(AA3_TO_IDX[mapped])

    if not ca_coords:
        return None

    return {
        "res_coords": torch.tensor(ca_coords, dtype=torch.float32),
        "res_type": torch.tensor(res_types, dtype=torch.int64),
    }


def parse_pocket_atoms(
    pdb_path: Path,
    ligand_coords: torch.Tensor | None = None,
    cutoff: float = 8.0,
) -> dict[str, torch.Tensor] | None:
    """Parse pocket heavy atoms using canonical AA topology (no RDKit).

    Args:
        pdb_path: Pocket PDB file.
        ligand_coords: ``[N_lig, 3]`` ligand crystal coordinates (optional).
        cutoff: Residue-aware 8Å distance filter from any ligand atom.

    Returns a dict with per-atom and per-residue tensors, or None on failure.
    """
    raw_atoms = _parse_pdb_lines(pdb_path)
    if not raw_atoms:
        return None

    # Residue-aware cutoff
    coords_all = torch.tensor([a.coords for a in raw_atoms], dtype=torch.float32)
    if ligand_coords is not None and ligand_coords.shape[0] > 0:
        dmat = torch.cdist(coords_all, ligand_coords)
        in_range = dmat.min(dim=1).values <= cutoff
        keys = [(a.chain, a.res_num, a.icode) for a in raw_atoms]
        active: set[tuple[str, int, str]] = set()
        for idx, ok in enumerate(in_range.tolist()):
            if ok:
                active.add(keys[idx])
        atoms = [a for a, k in zip(raw_atoms, keys) if k in active]
    else:
        atoms = raw_atoms

    if not atoms:
        return None

    # Local residue indexing (ordered by first occurrence)
    residue_to_local: dict[tuple[str, int, str], int] = OrderedDict()
    for a in atoms:
        key = (a.chain, a.res_num, a.icode)
        if key not in residue_to_local:
            residue_to_local[key] = len(residue_to_local)

    patom_residue_id = torch.tensor(
        [residue_to_local[(a.chain, a.res_num, a.icode)] for a in atoms],
        dtype=torch.int64,
    )

    tokens: list[int] = []
    is_backbone: list[bool] = []
    is_metal: list[bool] = []
    is_donor: list[bool] = []
    is_acceptor: list[bool] = []
    is_positive: list[bool] = []
    is_negative: list[bool] = []
    is_hydrophobic: list[bool] = []
    for a in atoms:
        if a.is_metal:
            tokens.append(METAL_ATOM_TOKENS.get(a.element, METAL_OTHER_TOKEN))
            # Treat metal ions as positive cations (Zn2+, Mg2+, etc.)
            d, ac, p, n, h = False, False, True, False, False
        else:
            tokens.append(_get_res_atom_token(a.res_name, a.atom_name))
            d, ac, p, n, h = _patom_pharmacophore(a.res_name, a.atom_name)
        is_backbone.append(a.atom_name in BACKBONE_ATOMS)
        is_metal.append(a.is_metal)
        is_donor.append(d)
        is_acceptor.append(ac)
        is_positive.append(p)
        is_negative.append(n)
        is_hydrophobic.append(h)

    patom_token = torch.tensor(tokens, dtype=torch.int64)
    patom_is_backbone = torch.tensor(is_backbone, dtype=torch.bool)
    patom_is_metal = torch.tensor(is_metal, dtype=torch.bool)
    patom_is_donor = torch.tensor(is_donor, dtype=torch.bool)
    patom_is_acceptor = torch.tensor(is_acceptor, dtype=torch.bool)
    patom_is_positive = torch.tensor(is_positive, dtype=torch.bool)
    patom_is_negative = torch.tensor(is_negative, dtype=torch.bool)
    patom_is_hydrophobic = torch.tensor(is_hydrophobic, dtype=torch.bool)
    patom_coords = torch.tensor([a.coords for a in atoms], dtype=torch.float32)

    # Per-residue virtual nodes (one per residue).
    # Placement rule:
    #   - standard AA with CB present → CB atom position
    #   - GLY → pseudo-CB from backbone (N, CA, C) via Rosetta-ideal geometry
    #   - metal ion residue → the metal atom position itself
    # ``pres_atom_index`` always refers to a real atom in patom_* space:
    #   - non-GLY AA: CB atom index
    #   - GLY: CA atom index (pseudo-CB is virtual — pres_coords carries it)
    #   - metal: the metal atom index
    pres_coords_list: list[list[float]] = []
    pres_res_type_list: list[int] = []
    pres_atom_index_list: list[int] = []
    pres_is_pseudo_list: list[bool] = []

    atom_lookup: dict[tuple[tuple[str, int, str], str], int] = {}
    for idx, a in enumerate(atoms):
        atom_lookup[((a.chain, a.res_num, a.icode), a.atom_name)] = idx

    for key in residue_to_local:
        ca_idx = atom_lookup.get((key, "CA"))
        cb_idx = atom_lookup.get((key, "CB"))

        # Metal residue: the (single) metal atom is the virtual node.
        metal_idx = None
        if ca_idx is None and cb_idx is None:
            for (rk, _), idx in atom_lookup.items():
                if rk == key and atoms[idx].is_metal:
                    metal_idx = idx
                    break
        if metal_idx is not None:
            a = atoms[metal_idx]
            pres_coords_list.append(list(a.coords))
            pres_res_type_list.append(METAL_RES_IDX)
            pres_atom_index_list.append(metal_idx)
            pres_is_pseudo_list.append(False)
            continue

        # Standard AA with CB atom present.
        if cb_idx is not None:
            a = atoms[cb_idx]
            res_type_int = AA3_TO_IDX.get(a.res_name, UNK_RES_IDX)
            pres_coords_list.append(list(a.coords))
            pres_res_type_list.append(res_type_int)
            pres_atom_index_list.append(cb_idx)
            pres_is_pseudo_list.append(False)
            continue

        # GLY / CB missing: compute pseudo-CB from backbone if possible.
        n_idx = atom_lookup.get((key, "N"))
        c_idx = atom_lookup.get((key, "C"))
        if ca_idx is not None and n_idx is not None and c_idx is not None:
            pseudo = _compute_pseudo_cb(
                atoms[n_idx].coords,
                atoms[ca_idx].coords,
                atoms[c_idx].coords,
            )
            ca_atom = atoms[ca_idx]
            res_type_int = AA3_TO_IDX.get(ca_atom.res_name, UNK_RES_IDX)
            pres_coords_list.append(list(pseudo))
            pres_res_type_list.append(res_type_int)
            pres_atom_index_list.append(ca_idx)  # real atom sentinel → CA
            pres_is_pseudo_list.append(True)
            continue

        # Fall back to CA (or skip if not even CA is present)
        if ca_idx is not None:
            a = atoms[ca_idx]
            res_type_int = AA3_TO_IDX.get(a.res_name, UNK_RES_IDX)
            pres_coords_list.append(list(a.coords))
            pres_res_type_list.append(res_type_int)
            pres_atom_index_list.append(ca_idx)
            pres_is_pseudo_list.append(True)

    if not pres_coords_list:
        return None

    pres_coords = torch.tensor(pres_coords_list, dtype=torch.float32)
    pres_residue_type = torch.tensor(pres_res_type_list, dtype=torch.int64)
    pres_atom_index = torch.tensor(pres_atom_index_list, dtype=torch.int64)
    pres_is_pseudo = torch.tensor(pres_is_pseudo_list, dtype=torch.bool)

    # Canonical protein bonds
    pbond_src, pbond_dst = _build_protein_bonds(atoms, atom_lookup)
    if pbond_src:
        pbond_index = torch.tensor([pbond_src, pbond_dst], dtype=torch.int64)
    else:
        pbond_index = torch.zeros(2, 0, dtype=torch.int64)

    return {
        "patom_coords": patom_coords,
        "patom_token": patom_token,
        "patom_residue_id": patom_residue_id,
        "patom_is_backbone": patom_is_backbone,
        "patom_is_metal": patom_is_metal,
        # Per-atom pharmacophore (NEW in schema_version=2): mirrors ligand-side
        # donor/acceptor/positive/negative/hydrophobe features so the model can
        # learn polar/charged contacts directly instead of inferring them from
        # the (residue, atom) token alone.
        "patom_is_donor": patom_is_donor,
        "patom_is_acceptor": patom_is_acceptor,
        "patom_is_positive": patom_is_positive,
        "patom_is_negative": patom_is_negative,
        "patom_is_hydrophobic": patom_is_hydrophobic,
        "pbond_index": pbond_index,
        "pres_coords": pres_coords,
        "pres_residue_type": pres_residue_type,
        "pres_atom_index": pres_atom_index,
        "pres_is_pseudo": pres_is_pseudo,
    }


def _build_protein_bonds(
    atoms: list[ParsedAtom],
    atom_lookup: dict[tuple[tuple[str, int, str], str], int],
) -> tuple[list[int], list[int]]:
    """Build bidirectional bond index from:

    1. Intra-residue canonical topology (``STANDARD_AA_BONDS``)
    2. Peptide bonds ``C(i) — N(i+1)`` (same chain, consecutive resnum, d < 2.0Å)
    3. Disulfide bonds ``CYS SG — CYS SG`` (d < 2.5Å)
    """
    src: list[int] = []
    dst: list[int] = []

    def add_bond(i: int, j: int) -> None:
        src.extend([i, j])
        dst.extend([j, i])

    # Group atom indices per residue key
    residues: dict[tuple[str, int, str], dict[str, int]] = OrderedDict()
    for idx, a in enumerate(atoms):
        residues.setdefault((a.chain, a.res_num, a.icode), {})[a.atom_name] = idx

    # (1) Intra-residue canonical bonds
    for atom_map in residues.values():
        any_idx = next(iter(atom_map.values()))
        res_name = atoms[any_idx].res_name
        bonds = STANDARD_AA_BONDS.get(res_name)
        if bonds is None:
            continue
        for a_name, b_name in bonds:
            ai = atom_map.get(a_name)
            bi = atom_map.get(b_name)
            if ai is not None and bi is not None:
                add_bond(ai, bi)

    # (2) Peptide bonds: C(i) — N(i+1) on same chain with consecutive numbers
    sorted_keys = sorted(residues.keys(), key=lambda k: (k[0], k[1], k[2]))
    for i in range(len(sorted_keys) - 1):
        cur = sorted_keys[i]
        nxt = sorted_keys[i + 1]
        if cur[0] != nxt[0]:
            continue
        if nxt[1] - cur[1] != 1:
            continue
        c_idx = atom_lookup.get((cur, "C"))
        n_idx = atom_lookup.get((nxt, "N"))
        if c_idx is None or n_idx is None:
            continue
        c_pos = atoms[c_idx].coords
        n_pos = atoms[n_idx].coords
        d2 = (c_pos[0] - n_pos[0]) ** 2 + (c_pos[1] - n_pos[1]) ** 2 + (c_pos[2] - n_pos[2]) ** 2
        if d2 < 4.0:  # 2.0 Å² cutoff
            add_bond(c_idx, n_idx)

    # (3) Disulfide bonds between CYS SGs
    sg_atoms = [
        (idx, a) for idx, a in enumerate(atoms) if a.res_name == "CYS" and a.atom_name == "SG"
    ]
    for i in range(len(sg_atoms)):
        for j in range(i + 1, len(sg_atoms)):
            ii, ai = sg_atoms[i]
            jj, aj = sg_atoms[j]
            d2 = (
                (ai.coords[0] - aj.coords[0]) ** 2
                + (ai.coords[1] - aj.coords[1]) ** 2
                + (ai.coords[2] - aj.coords[2]) ** 2
            )
            if d2 < 6.25:  # 2.5 Å² cutoff
                add_bond(ii, jj)

    return src, dst
