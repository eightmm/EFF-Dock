"""Shared node / edge type vocabulary for the unified protein-ligand graph.

Single source of truth for integer type IDs used by both preprocessing
(``build_static_complex_graph``) and the model (``EFFDock``).  Keep
changes here in lock-step: any new edge/node type must be appended and the
``NUM_*`` counters incremented.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------
NTYPE_LIG_ATOM = 0
NTYPE_FRAGMENT = 1
NTYPE_PROT_ATOM = 2
NTYPE_PROT_RES = 3
NUM_NODE_TYPES = 4

# String-keyed dict for builder code readability
NODE_TYPES: dict[str, int] = {
    "ligand_atom": NTYPE_LIG_ATOM,
    "ligand_fragment": NTYPE_FRAGMENT,
    "protein_atom": NTYPE_PROT_ATOM,
    "protein_res": NTYPE_PROT_RES,
}


# ---------------------------------------------------------------------------
# Edge types
# ---------------------------------------------------------------------------
ETYPE_LIG_BOND = 0
ETYPE_LIG_TRI = 1
ETYPE_LIG_CUT = 2
ETYPE_LIG_ATOM_FRAG = 3
ETYPE_LIG_FRAG_FRAG = 4
ETYPE_PROT_BOND = 5
ETYPE_PROT_ATOM_RES = 6
ETYPE_PROT_RES_RES = 7
ETYPE_PROT_RES_FRAG = 8
ETYPE_DYNAMIC_CONTACT = 9
NUM_EDGE_TYPES = 10

EDGE_TYPES: dict[str, int] = {
    "ligand_bond": ETYPE_LIG_BOND,
    "ligand_tri": ETYPE_LIG_TRI,
    "ligand_cut": ETYPE_LIG_CUT,
    "ligand_atom_frag": ETYPE_LIG_ATOM_FRAG,
    "ligand_frag_frag": ETYPE_LIG_FRAG_FRAG,
    "protein_bond": ETYPE_PROT_BOND,
    "protein_atom_res": ETYPE_PROT_ATOM_RES,
    "protein_res_res": ETYPE_PROT_RES_RES,
    "protein_res_frag": ETYPE_PROT_RES_FRAG,
    "dynamic_contact": ETYPE_DYNAMIC_CONTACT,
}

DYNAMIC_EDGE_TYPES: set[int] = {
    ETYPE_LIG_FRAG_FRAG,
    ETYPE_PROT_RES_FRAG,
}


__all__ = [
    "NTYPE_LIG_ATOM",
    "NTYPE_FRAGMENT",
    "NTYPE_PROT_ATOM",
    "NTYPE_PROT_RES",
    "NUM_NODE_TYPES",
    "NODE_TYPES",
    "ETYPE_LIG_BOND",
    "ETYPE_LIG_TRI",
    "ETYPE_LIG_CUT",
    "ETYPE_LIG_ATOM_FRAG",
    "ETYPE_LIG_FRAG_FRAG",
    "ETYPE_PROT_BOND",
    "ETYPE_PROT_ATOM_RES",
    "ETYPE_PROT_RES_RES",
    "ETYPE_PROT_RES_FRAG",
    "ETYPE_DYNAMIC_CONTACT",
    "NUM_EDGE_TYPES",
    "EDGE_TYPES",
    "DYNAMIC_EDGE_TYPES",
]
