"""Tests for preprocessing pipeline: protein, ligand, fragments."""

from pathlib import Path

import pytest
import torch
from rdkit import Chem

from effdock.inference.preprocess import load_ligand
from effdock.preprocess.fragments import decompose_fragments
from effdock.preprocess.graph import build_static_complex_graph
from effdock.preprocess.ligand import (
    featurize_ligand,
    ligand_graph_identity,
    load_molecule,
)
from effdock.preprocess.protein import AA3_TO_IDX, parse_pocket_atoms, parse_pocket_pdb

# ─── Fixtures ─────────────────────────────────────────────────────────

SAMPLE_PDB = """\
ATOM      1  N   ALA A  10      10.000  10.000  10.000  1.00 20.00           N
ATOM      2  CA  ALA A  10      11.000  10.000  10.000  1.00 20.00           C
ATOM      3  C   ALA A  10      12.000  10.000  10.000  1.00 20.00           C
ATOM      4  N   GLY A  15      15.000  12.000  10.000  1.00 20.00           N
ATOM      5  CA  GLY A  15      16.000  12.000  10.000  1.00 20.00           C
ATOM      6  C   GLY A  15      17.000  12.000  10.000  1.00 20.00           C
ATOM      7  N   MET A  20      20.000  15.000  10.000  1.00 20.00           N
ATOM      8  CA  MET A  20      21.000  15.000  10.000  1.00 20.00           C
HETATM    9  CA  MSE A  25      25.000  18.000  10.000  1.00 20.00          SE
HETATM   10  O   HOH A 100      30.000  30.000  30.000  1.00 20.00           O
END
"""

CHIRAL_MOL2 = """\
@<TRIPOS>MOLECULE
chiral_halocarbon
5 4 0 0 0
SMALL
NO_CHARGES

@<TRIPOS>ATOM
      1 C1          0.0000    0.0000    0.0000 C.3       1 LIG       0.0000
      2 F1          1.0000    1.0000    1.0000 F         1 LIG       0.0000
      3 CL1        -1.0000   -1.0000    1.0000 Cl        1 LIG       0.0000
      4 BR1        -1.0000    1.0000   -1.0000 Br        1 LIG       0.0000
      5 H1          1.0000   -1.0000   -1.0000 H         1 LIG       0.0000
@<TRIPOS>BOND
     1    1    2 1
     2    1    3 1
     3    1    4 1
     4    1    5 1
"""


def _make_mol_manual_coords(smiles: str, coords_3d: list[list[float]]) -> Chem.Mol:
    """Create mol with manually specified 3D coordinates (no embedding needed)."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.RemoveHs(mol)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords_3d):
        conf.SetAtomPosition(i, (x, y, z))
    conf.Set3D(True)
    mol.AddConformer(conf, assignId=True)
    return mol


def _write_sdf(mol: Chem.Mol, path: Path) -> None:
    """Write mol to SDF file."""
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


# ─── Manual coordinate sets ──────────────────────────────────────────

# Benzene: 6 carbons in hexagonal ring
BENZENE_COORDS = [
    [1.4, 0.0, 0.0],
    [0.7, 1.2, 0.0],
    [-0.7, 1.2, 0.0],
    [-1.4, 0.0, 0.0],
    [-0.7, -1.2, 0.0],
    [0.7, -1.2, 0.0],
]

# Two phenyl rings connected by -CH2-CH2- linker (14 heavy atoms)
# Ring1(6) + CH2(1) + CH2(1) + Ring2(6)
DIPHENYLETHANE_COORDS = [
    # Ring 1
    [0.0, 1.4, 0.0],
    [1.2, 0.7, 0.0],
    [1.2, -0.7, 0.0],
    [0.0, -1.4, 0.0],
    [-1.2, -0.7, 0.0],
    [-1.2, 0.7, 0.0],
    # CH2 linker
    [0.0, 2.9, 0.0],
    [0.0, 4.4, 0.0],
    # Ring 2
    [0.0, 5.9, 0.0],
    [1.2, 6.6, 0.0],
    [1.2, 8.0, 0.0],
    [0.0, 8.7, 0.0],
    [-1.2, 8.0, 0.0],
    [-1.2, 6.6, 0.0],
]


# ─── Protein Tests ────────────────────────────────────────────────────


class TestProteinParsing:
    def test_parse_pocket_pdb(self, tmp_path: Path):
        pdb_file = tmp_path / "pocket.pdb"
        pdb_file.write_text(SAMPLE_PDB)

        result = parse_pocket_pdb(pdb_file)
        assert result is not None

        # Should find 4 CAs: ALA, GLY, MET, MSE(→MET)
        assert result["res_coords"].shape == (4, 3)
        assert result["res_type"].shape == (4,)

        # Check types
        assert result["res_type"][0].item() == AA3_TO_IDX["ALA"]
        assert result["res_type"][1].item() == AA3_TO_IDX["GLY"]
        assert result["res_type"][2].item() == AA3_TO_IDX["MET"]
        assert result["res_type"][3].item() == AA3_TO_IDX["MET"]  # MSE → MET

        # Check coords
        assert result["res_coords"][0, 0].item() == pytest.approx(11.0)  # ALA CA x

    def test_empty_pdb(self, tmp_path: Path):
        pdb_file = tmp_path / "empty.pdb"
        pdb_file.write_text("END\n")
        assert parse_pocket_pdb(pdb_file) is None

    def test_water_only_pdb(self, tmp_path: Path):
        pdb_file = tmp_path / "water.pdb"
        pdb_file.write_text("HETATM    1  O   HOH A   1      10.0  10.0  10.0  1.00 20.00\nEND\n")
        assert parse_pocket_pdb(pdb_file) is None

    def test_duplicate_residue_altloc(self, tmp_path: Path):
        """Only first CA per residue should be kept."""
        # PDB format: columns are fixed width
        # 1-6: record, 7-11: serial, 12: blank, 13-16: name, 17: altLoc, 18-20: resName
        pdb_text = (
            "ATOM      1  CA AALA A  10      10.000  10.000  10.000  1.00 20.00           C\n"
            "ATOM      2  CA BALA A  10      11.000  11.000  11.000  1.00 20.00           C\n"
            "END\n"
        )
        pdb_file = tmp_path / "altloc.pdb"
        pdb_file.write_text(pdb_text)

        result = parse_pocket_pdb(pdb_file)
        assert result is not None
        assert result["res_coords"].shape == (1, 3)
        assert result["res_coords"][0, 0].item() == pytest.approx(10.0)


# ─── Ligand Tests ─────────────────────────────────────────────────────


class TestLigandParsing:
    def test_load_molecule_sdf(self, tmp_path: Path):
        mol = _make_mol_manual_coords("c1ccccc1", BENZENE_COORDS)
        sdf_path = tmp_path / "test.sdf"
        _write_sdf(mol, sdf_path)

        loaded, used_fallback, sanitize_ok = load_molecule(sdf_path)
        assert loaded is not None
        assert not used_fallback
        assert sanitize_ok
        assert loaded.GetNumAtoms() == 6

    def test_docking_and_trace_mol2_loaders_have_identity_parity(
        self,
        tmp_path: Path,
    ):
        mol2_path = tmp_path / "chiral.mol2"
        mol2_path.write_text(CHIRAL_MOL2)
        docking_mol, has_pose = load_ligand(str(mol2_path))
        trace_mol, _, sanitize_ok = load_molecule(None, mol2_path)

        assert has_pose
        assert trace_mol is not None
        assert sanitize_ok
        fragment_id = torch.zeros(docking_mol.GetNumAtoms(), dtype=torch.long)
        assert ligand_graph_identity(
            docking_mol,
            fragment_id,
        ) == ligand_graph_identity(
            trace_mol,
            fragment_id,
        )

    def test_featurize_benzene(self):
        mol = _make_mol_manual_coords("c1ccccc1", BENZENE_COORDS)
        result = featurize_ligand(mol)
        assert result is not None

        # 6 atoms
        assert result["atom_coords"].shape == (6, 3)
        assert result["atom_element"].shape == (6,)
        assert (result["atom_element"] == 0).all()  # all carbon
        assert (result["atom_aromatic"]).all()  # all aromatic
        assert (result["atom_num_rings"] > 0).all()  # all in ring

        # 6 bonds → 12 directed edges
        assert result["bond_index"].shape[1] == 12
        assert result["bond_type"].shape == (12,)

    def test_featurize_charged_molecule(self):
        # Glycine zwitterion: [NH3+]CC([O-])=O → 5 heavy atoms
        coords = [[0, 0, 0], [1.5, 0, 0], [3.0, 0, 0], [3.7, 1.2, 0], [3.7, -1.2, 0]]
        mol = _make_mol_manual_coords("[NH3+]CC([O-])=O", coords)
        result = featurize_ligand(mol)
        assert result is not None
        assert result["atom_charge"].abs().sum() > 0

    def test_single_atom_returns_none(self):
        mol = Chem.MolFromSmiles("[Na+]")
        assert featurize_ligand(mol) is None

    def test_bond_stereo_range_matches_model_embedding(self):
        """Regression: BOND_STEREO_MAP has values 0..5; model embedding must
        accommodate that range. Previously NUM_BOND_STEREO=4 truncated CIS/TRANS
        stereos onto out-of-range indices and silently corrupted the embedding
        lookup (after the +1 N/A shift).
        """
        from effdock.models.effdock import EFFDockInteractionLayer
        from effdock.preprocess.ligand import BOND_STEREO_MAP

        max_stereo = max(BOND_STEREO_MAP.values())
        assert EFFDockInteractionLayer.NUM_BOND_STEREO >= max_stereo + 1, (
            f"NUM_BOND_STEREO={EFFDockInteractionLayer.NUM_BOND_STEREO} "
            f"cannot encode BOND_STEREO_MAP max index {max_stereo}"
        )

    def test_featurize_stereo_double_bond(self):
        """trans-2-butene: C=C bond should carry STEREOE (value 3 in BOND_STEREO_MAP)."""
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("C/C=C/C")
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=0) != 0:
            pytest.skip("RDKit embed failed")
        mol = Chem.RemoveHs(mol)
        Chem.AssignStereochemistryFrom3D(mol)

        result = featurize_ligand(mol)
        assert result is not None
        # STEREOE=3, STEREOZ=2, STEREOCIS=4, STEREOTRANS=5 — any of these proves
        # featurizer can emit values that NUM_BOND_STEREO=4 would have truncated.
        assert (result["bond_stereo"] >= 2).any()


# ─── Fragment Tests ───────────────────────────────────────────────────


class TestFragmentDecomposition:
    def test_benzene_single_fragment(self):
        """Benzene has no rotatable bonds → 1 fragment."""
        mol = _make_mol_manual_coords("c1ccccc1", BENZENE_COORDS)
        coords = torch.tensor(BENZENE_COORDS, dtype=torch.float32)

        result = decompose_fragments(mol, coords)
        assert result is not None
        assert result["n_frags"] == 1
        assert (result["fragment_id"] == 0).all()

    def test_diphenylethane_fragments(self):
        """c1ccc(CCc2ccccc2)cc1: two rings + CH2CH2 linker."""
        mol = _make_mol_manual_coords("c1ccc(CCc2ccccc2)cc1", DIPHENYLETHANE_COORDS)
        coords = torch.tensor(DIPHENYLETHANE_COORDS, dtype=torch.float32)

        result = decompose_fragments(mol, coords)
        assert result is not None
        assert result["n_frags"] >= 1
        assert result["fragment_id"].shape[0] == 14
        assert result["frag_centers"].shape == (result["n_frags"], 3)
        assert result["frag_local_coords"].shape == (14, 3)

    def test_local_coords_centroid_property(self):
        """Local coords should sum to ~0 per fragment."""
        mol = _make_mol_manual_coords("c1ccc(CCc2ccccc2)cc1", DIPHENYLETHANE_COORDS)
        coords = torch.tensor(DIPHENYLETHANE_COORDS, dtype=torch.float32)

        result = decompose_fragments(mol, coords)
        assert result is not None

        for f in range(result["n_frags"]):
            mask = result["fragment_id"] == f
            local = result["frag_local_coords"][mask]
            assert local.mean(dim=0).abs().max() < 1e-5

    def test_reconstruction_from_local_coords(self):
        """x_global = R_frag @ x_local + T_frag, with R=I at crystal pose."""
        mol = _make_mol_manual_coords("c1ccc(CCc2ccccc2)cc1", DIPHENYLETHANE_COORDS)
        coords = torch.tensor(DIPHENYLETHANE_COORDS, dtype=torch.float32)

        result = decompose_fragments(mol, coords)
        assert result is not None

        reconstructed = result["frag_local_coords"] + result["frag_centers"][result["fragment_id"]]
        assert torch.allclose(reconstructed, coords, atol=1e-5)

    def test_fragment_id_valid_range(self):
        """All fragment IDs should be in [0, n_frags)."""
        mol = _make_mol_manual_coords("c1ccc(CCc2ccccc2)cc1", DIPHENYLETHANE_COORDS)
        coords = torch.tensor(DIPHENYLETHANE_COORDS, dtype=torch.float32)

        result = decompose_fragments(mol, coords)
        assert result is not None
        assert result["fragment_id"].min() >= 0
        assert result["fragment_id"].max() < result["n_frags"]

    def test_frag_sizes_match(self):
        """Sum of frag_sizes should equal total atoms."""
        mol = _make_mol_manual_coords("c1ccc(CCc2ccccc2)cc1", DIPHENYLETHANE_COORDS)
        coords = torch.tensor(DIPHENYLETHANE_COORDS, dtype=torch.float32)

        result = decompose_fragments(mol, coords)
        assert result is not None
        assert result["frag_sizes"].sum().item() == 14

    def test_triangulation_edge_filtering(self):
        """Verify that neighbor-neighbor pairs (torsion-dependent) are excluded."""
        # Ethane-like: C1-C2-C3-C4 (actually 4 atoms)
        # 0-1 (cut) 1-2, 2-3... wait.
        # Let's use a simple 4-atom chain: A-B-C-D where B-C is rotatable.
        # Frag1: {A, B}, Frag2: {C, D}
        # Cut bond: (1, 2)
        # Neighbors of 1 in Frag1: {0}
        # Neighbors of 2 in Frag2: {3}
        # Triangulation edges should be: (1, 2), (1, 3), (0, 2)
        # (0, 3) should be EXCLUDED as it depends on torsion 0-1-2-3.
        coords = [[0, 0, 0], [1.5, 0, 0], [3.0, 0, 0], [4.5, 0, 0]]
        mol = _make_mol_manual_coords("CCCC", coords)
        # Ensure B-C is rotatable and cut
        # Decompose will cut B-C (1-2) if it's not in ring, etc.
        res = decompose_fragments(mol, torch.tensor(coords, dtype=torch.float32))
        assert res is not None
        assert res["n_frags"] == 2

        # Check tri_edge_index
        tri_edges = res["tri_edge_index"]
        # Convert to set of frozen sets for easy comparison
        edge_sets = {frozenset(tri_edges[:, k].tolist()) for k in range(tri_edges.shape[1])}

        # Expected:
        # (1, 2) - invariant (bond length)
        # (1, 3) - invariant (bond angle)
        # (0, 2) - invariant (bond angle)
        assert frozenset([1, 2]) in edge_sets
        assert frozenset([1, 3]) in edge_sets
        assert frozenset([0, 2]) in edge_sets

        # EXCLUDED:
        # (0, 3) - variant (torsion)
        assert frozenset([0, 3]) not in edge_sets


class TestGraphFeatures:
    def test_graph_carries_model_input_features(self, tmp_path: Path):
        """Graph keeps available chemistry flags for the docking model."""
        mol = _make_mol_manual_coords(
            "ClCC([O-])=O",
            [
                [0.0, 0.0, 0.0],
                [1.7, 0.0, 0.0],
                [3.1, 0.0, 0.0],
                [3.8, 1.1, 0.0],
                [3.8, -1.1, 0.0],
            ],
        )
        lig = featurize_ligand(mol)
        assert lig is not None
        frag = decompose_fragments(mol, lig["atom_coords"])
        assert frag is not None
        lig.update(frag)

        pdb_file = tmp_path / "pocket.pdb"
        pdb_file.write_text(E2E_POCKET_PDB)
        prot = parse_pocket_atoms(pdb_file)
        assert prot is not None

        graph = build_static_complex_graph(lig, prot)
        total_nodes = int(graph["num_nodes"])
        for key in (
            "node_atom_degree",
            "node_atom_implicit_valence",
            "node_atom_explicit_valence",
            "node_atom_chirality",
            "node_is_halogen",
            "node_patom_is_backbone",
            "node_patom_is_metal",
            "node_pres_is_pseudo",
        ):
            assert graph[key].shape == (total_nodes,)

        atom_lo, atom_hi = graph["lig_atom_slice"].tolist()
        prot_lo, prot_hi = graph["prot_atom_slice"].tolist()
        pres_lo, pres_hi = graph["prot_res_slice"].tolist()

        torch.testing.assert_close(
            graph["node_atom_degree"][atom_lo:atom_hi],
            lig["atom_degree"],
        )
        assert graph["node_is_halogen"][atom_lo:atom_hi].any()
        torch.testing.assert_close(
            graph["node_patom_is_backbone"][prot_lo:prot_hi],
            prot["patom_is_backbone"],
        )
        torch.testing.assert_close(
            graph["node_patom_is_metal"][prot_lo:prot_hi],
            prot["patom_is_metal"],
        )
        torch.testing.assert_close(
            graph["node_pres_is_pseudo"][pres_lo:pres_hi],
            prot["pres_is_pseudo"],
        )


# ─── Integration Test ─────────────────────────────────────────────────

# Pocket PDB placed near diphenylethane ligand (coords ~0-9 Å) so the
# 8 Å residue-level cutoff keeps all residues.
E2E_POCKET_PDB = """\
ATOM      1  N   ALA A   1       3.000   3.000   0.500  1.00 20.00           N
ATOM      2  CA  ALA A   1       4.500   3.000   0.500  1.00 20.00           C
ATOM      3  C   ALA A   1       5.500   3.000   0.500  1.00 20.00           C
ATOM      4  O   ALA A   1       5.500   4.200   0.500  1.00 20.00           O
ATOM      5  CB  ALA A   1       4.500   1.500   0.500  1.00 20.00           C
ATOM      6  N   LEU A   2       6.700   3.000   0.500  1.00 20.00           N
ATOM      7  CA  LEU A   2       7.500   4.200   0.500  1.00 20.00           C
ATOM      8  C   LEU A   2       8.500   4.200   0.500  1.00 20.00           C
ATOM      9  O   LEU A   2       8.500   5.400   0.500  1.00 20.00           O
ATOM     10  CB  LEU A   2       7.500   5.700   0.500  1.00 20.00           C
ATOM     11  CG  LEU A   2       7.500   7.200   0.500  1.00 20.00           C
ATOM     12  CD1 LEU A   2       6.300   7.900   0.500  1.00 20.00           C
ATOM     13  CD2 LEU A   2       8.700   7.900   0.500  1.00 20.00           C
END
"""
