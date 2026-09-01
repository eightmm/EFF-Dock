from __future__ import annotations

import pytest
import torch

Chem = pytest.importorskip("rdkit.Chem")

from effdock.guidance.errors import UnsupportedPhysicalChemistryError  # noqa: E402
from effdock.guidance.interaction import (  # noqa: E402
    InteractionEnergyConfig,
    interaction_contact_stats,
    interaction_energy,
)
from effdock.guidance.system import build_physical_system  # noqa: E402

DTYPE = torch.float64


def _mol(smiles: str, coords: torch.Tensor) -> "Chem.Mol":
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    assert mol.GetNumAtoms() == coords.shape[0]
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index, point in enumerate(coords.tolist()):
        conformer.SetAtomPosition(index, point)
    conformer.Set3D(True)
    mol.AddConformer(conformer)
    return mol


def _pdb_line(
    record: str,
    serial: int,
    atom_name: str,
    residue: str,
    residue_number: int,
    coord: torch.Tensor,
    element: str,
    *,
    altloc: str = " ",
    occupancy: float = 1.0,
) -> str:
    return (
        f"{record:<6}{serial:>5} {atom_name:^4}{altloc:1}{residue:>3} A"
        f"{residue_number:>4}    {float(coord[0]):>8.3f}"
        f"{float(coord[1]):>8.3f}{float(coord[2]):>8.3f}"
        f"{occupancy:>6.2f} 20.00          {element:>2}\n"
    )


def _build(
    mol: "Chem.Mol",
    ligand: torch.Tensor,
    receptor,
    *,
    fragment_id: torch.Tensor | None = None,
):
    if fragment_id is None:
        fragment_id = torch.arange(ligand.shape[0], dtype=torch.long)
    return build_physical_system(
        mol,
        receptor,
        fragment_id=fragment_id,
        near_coords=ligand,
    ).to(torch.device("cpu"), DTYPE)


def test_strict_mg_profile_uses_retained_water_and_one_o_vacancy(tmp_path) -> None:
    # Five fixed octahedral directions leave +x open for the ligand oxygen.
    fixed = (
        torch.tensor(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=DTYPE,
        )
        * 2.13
    )
    ligand = torch.tensor(
        [
            [3.55, 0.0, 0.0],
            [2.13, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ],
        dtype=DTYPE,
    )
    mol = _mol("COS", ligand)
    lines = [_pdb_line("HETATM", 1, "MG", "MG", 1, torch.zeros(3), "MG")]
    for index, coord in enumerate(fixed[:4], start=2):
        lines.append(
            _pdb_line(
                "ATOM",
                index,
                "OD1",
                "ASP",
                index,
                coord,
                "O",
            )
        )
    lines.append(
        _pdb_line(
            "HETATM",
            20,
            "O",
            "HOH",
            20,
            fixed[4],
            "O",
        )
    )
    lines.append("END\n")
    receptor = tmp_path / "mg_one_vacancy.pdb"
    receptor.write_text("".join(lines))

    system = _build(
        mol,
        ligand,
        receptor,
        fragment_id=torch.zeros(3, dtype=torch.long),
    )
    topology = system.interaction_topology
    assert topology is not None
    assert topology.metal_atomic_number.tolist() == [12]
    assert topology.metal_fixed_coordination.tolist() == [5]
    assert topology.metal_target_coordination.tolist() == [6]
    assert topology.metal_attraction_enabled.tolist() == [True]
    assert topology.metal_ligand_donor_allowed.tolist() == [[False, True, False]]
    torch.testing.assert_close(
        topology.metal_vacant_direction[0],
        torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE),
        atol=2e-3,
        rtol=0.0,
    )

    metal_energy = interaction_energy(ligand, system)[
        "interaction_metal_coordination"
    ]
    assert metal_energy < 0
    differentiable = ligand.clone().requires_grad_(True)
    differentiable_energy = interaction_energy(differentiable, system)[
        "interaction_metal_coordination"
    ]
    differentiable_energy.backward()
    assert differentiable.grad is not None
    assert bool(torch.isfinite(differentiable.grad).all())
    stats = interaction_contact_stats(ligand, system)["metal_coordination"]
    assert stats["metal_site_count"] == 1
    assert stats["attraction_enabled"] == [True]
    assert stats["top_donor_pairs"][0]["metal"]["profile"].startswith("MG:+2:")

    # Sulfur is typed as a possible generic metal donor, but Mg's site-specific
    # O-only mask must route it back through non-donor clash repulsion.
    sulfur_clash = ligand.clone()
    sulfur_clash[2] = torch.tensor([0.1, 0.0, 0.0], dtype=DTYPE)
    clash_stats = interaction_contact_stats(
        sulfur_clash,
        system,
    )["metal_coordination"]
    assert clash_stats["non_donor_repulsion_kcal_mol"] > 0
    assert clash_stats["top_non_donor_repulsions"][0]["ligand_atom"]["index"] == 2

    unsupported_shell = tmp_path / "mg_with_close_histidine_n.pdb"
    unsupported_shell.write_text(
        "".join(
            lines[:-1]
            + [
                _pdb_line(
                    "ATOM",
                    30,
                    "ND1",
                    "HIS",
                    30,
                    torch.tensor([1.5, 1.5, 0.0]),
                    "N",
                ),
                "END\n",
            ]
        )
    )
    with pytest.raises(
        UnsupportedPhysicalChemistryError,
        match="outside the attractive profile",
    ) as exc_info:
        _build(
            mol,
            ligand,
            unsupported_shell,
            fragment_id=torch.zeros(3, dtype=torch.long),
        )
    assert exc_info.value.code == "unsupported_metal_profile"


def test_ca_profile_is_automatic_repulsion_only_and_default_on(tmp_path) -> None:
    ligand = torch.tensor([[0.1, 0.0, 0.0]], dtype=DTYPE)
    mol = _mol("C", ligand)
    receptor = tmp_path / "calcium.pdb"
    receptor.write_text(
        "".join(
            [
                _pdb_line("HETATM", 1, "CA", "CA", 1, torch.zeros(3), "CA"),
                _pdb_line(
                    "ATOM",
                    2,
                    "OD1",
                    "ASP",
                    2,
                    torch.tensor([4.0, 0.0, 0.0]),
                    "O",
                ),
                "END\n",
            ]
        )
    )
    system = _build(mol, ligand, receptor)
    topology = system.interaction_topology
    assert topology is not None
    assert topology.metal_atomic_number.tolist() == [20]
    assert topology.metal_attraction_enabled.tolist() == [False]
    assert topology.metal_ligand_donor_allowed.tolist() == [[False, False, False]]
    assert any("attraction_disabled" in item for item in topology.metal_typing_exclusion_labels)

    metal_energy = interaction_energy(ligand, system)[
        "interaction_metal_coordination"
    ]
    assert metal_energy > 0
    with pytest.raises(
        UnsupportedPhysicalChemistryError,
        match="requires explicit metal_coordination activation",
    ) as exc_info:
        interaction_energy(
            ligand,
            system,
            InteractionEnergyConfig(active_terms=("hydrophobic",)),
        )
    assert exc_info.value.code == "required_interaction_term_inactive"


def test_independent_profiled_metals_are_kept_but_nearby_cluster_fails(tmp_path) -> None:
    ligand = torch.tensor(
        [[0.2, 0.0, 0.0], [10.2, 0.0, 0.0]],
        dtype=DTYPE,
    )
    mol = _mol("C.C", ligand)

    def receptor_text(second_x: float) -> str:
        return "".join(
            [
                _pdb_line("HETATM", 1, "CA", "CA", 1, torch.zeros(3), "CA"),
                _pdb_line(
                    "HETATM",
                    2,
                    "FE",
                    "FE",
                    2,
                    torch.tensor([second_x, 0.0, 0.0]),
                    "FE",
                ),
                _pdb_line(
                    "ATOM",
                    3,
                    "OD1",
                    "ASP",
                    3,
                    torch.tensor([0.0, 4.0, 0.0]),
                    "O",
                ),
                _pdb_line(
                    "ATOM",
                    4,
                    "OD1",
                    "ASP",
                    4,
                    torch.tensor([second_x, 4.0, 0.0]),
                    "O",
                ),
                "END\n",
            ]
        )

    independent = tmp_path / "independent_metals.pdb"
    independent.write_text(receptor_text(10.0))
    independent_system = _build(mol, ligand, independent)
    topology = independent_system.interaction_topology
    assert topology is not None
    assert topology.metal_atomic_number.tolist() == [20, 26]
    assert topology.metal_attraction_enabled.tolist() == [False, False]
    assert interaction_energy(ligand, independent_system)[
        "interaction_metal_coordination"
    ] > 0

    clustered = tmp_path / "clustered_metals.pdb"
    clustered.write_text(receptor_text(4.0))
    with pytest.raises(
        UnsupportedPhysicalChemistryError,
        match="nearby-metal",
    ) as exc_info:
        _build(mol, ligand, clustered)
    assert exc_info.value.code == "unsupported_metal_profile"


@pytest.mark.parametrize(
    ("residue", "occupancy", "expected_text"),
    [
        ("HEM", 1.0, "matching PDB residue and element identity"),
        ("FE", 0.5, "full-occupancy"),
    ],
)
def test_ambiguous_standalone_metal_identity_fails_closed(
    tmp_path,
    residue: str,
    occupancy: float,
    expected_text: str,
) -> None:
    ligand = torch.tensor([[0.2, 0.0, 0.0]], dtype=DTYPE)
    mol = _mol("C", ligand)
    receptor = tmp_path / f"ambiguous_{residue}_{occupancy}.pdb"
    receptor.write_text(
        "".join(
            [
                _pdb_line(
                    "HETATM",
                    1,
                    "FE",
                    residue,
                    1,
                    torch.zeros(3),
                    "FE",
                    occupancy=occupancy,
                ),
                _pdb_line(
                    "ATOM",
                    2,
                    "OD1",
                    "ASP",
                    2,
                    torch.tensor([4.0, 0.0, 0.0]),
                    "O",
                ),
                "END\n",
            ]
        )
    )
    with pytest.raises(
        UnsupportedPhysicalChemistryError,
        match=expected_text,
    ) as exc_info:
        _build(mol, ligand, receptor)
    assert exc_info.value.code == "unsupported_metal_profile"


def test_nonprimary_only_metal_altloc_cannot_disappear_in_parser(tmp_path) -> None:
    ligand = torch.tensor([[0.2, 0.0, 0.0]], dtype=DTYPE)
    mol = _mol("C", ligand)
    receptor = tmp_path / "hidden_altloc_b_iron.pdb"
    receptor.write_text(
        "".join(
            [
                _pdb_line(
                    "HETATM",
                    1,
                    "FE",
                    "FE",
                    1,
                    torch.zeros(3),
                    "FE",
                    altloc="B",
                ),
                _pdb_line(
                    "ATOM",
                    2,
                    "OD1",
                    "ASP",
                    2,
                    torch.tensor([4.0, 0.0, 0.0]),
                    "O",
                ),
                "END\n",
            ]
        )
    )
    with pytest.raises(
        UnsupportedPhysicalChemistryError,
        match="hidden by normalization",
    ) as exc_info:
        _build(mol, ligand, receptor)
    assert exc_info.value.code == "unsupported_metal_profile"


def test_blank_element_protein_ca_is_not_misread_as_calcium(tmp_path) -> None:
    ligand = torch.tensor([[0.2, 0.0, 0.0]], dtype=DTYPE)
    mol = _mol("C", ligand)
    receptor = tmp_path / "protein_alpha_carbon_altloc.pdb"
    receptor.write_text(
        "".join(
            [
                _pdb_line(
                    "ATOM",
                    1,
                    "CA",
                    "ALA",
                    1,
                    torch.tensor([1.0, 0.0, 0.0]),
                    "",
                    altloc="B",
                ),
                _pdb_line(
                    "ATOM",
                    2,
                    "OD1",
                    "ASP",
                    2,
                    torch.tensor([4.0, 0.0, 0.0]),
                    "O",
                ),
                "END\n",
            ]
        )
    )
    system = _build(mol, ligand, receptor)
    assert system.interaction_topology is not None
    assert system.interaction_topology.metal_coords.shape == (0, 3)
