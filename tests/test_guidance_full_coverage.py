from __future__ import annotations

from dataclasses import replace

import pytest
import torch

Chem = pytest.importorskip("rdkit.Chem")

from effdock.guidance.errors import UnsupportedPhysicalChemistryError  # noqa: E402
from effdock.guidance.interaction import interaction_energy  # noqa: E402
from effdock.guidance.physical import PhysicalEnergyConfig, physical_energy  # noqa: E402
from effdock.guidance.runtime import guidance_energy  # noqa: E402
from effdock.guidance.system import (  # noqa: E402
    build_physical_system,
    receptor_policy_identity,
)
from effdock.preprocess.protein import (  # noqa: E402
    UNK_ATOM_TOKEN,
    parse_pocket_atoms,
)

DTYPE = torch.float64


def _molecule() -> tuple["Chem.Mol", torch.Tensor]:
    coords = torch.tensor(
        [[0.10, 0.0, 0.0], [1.55, 0.0, 0.0], [2.80, 0.0, 0.0]],
        dtype=DTYPE,
    )
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index, coord in enumerate(coords.tolist()):
        conformer.SetAtomPosition(index, coord)
    conformer.Set3D(True)
    mol.AddConformer(conformer)
    return mol, coords


def _pdb_line(
    record: str,
    serial: int,
    atom_name: str,
    residue: str,
    residue_number: int,
    coord: tuple[float, float, float],
    element: str,
    *,
    altloc: str = " ",
    occupancy: float = 1.0,
) -> str:
    return (
        f"{record:<6}{serial:>5} {atom_name:^4}{altloc:1}{residue:>3} A"
        f"{residue_number:>4}    {coord[0]:>8.3f}{coord[1]:>8.3f}{coord[2]:>8.3f}"
        f"{occupancy:>6.2f} 20.00          {element:>2}\n"
    )


def _protein_atom(serial: int = 1) -> str:
    return _pdb_line("ATOM", serial, "CA", "ALA", 1, (0.0, 4.0, 0.0), "C")


def _build(mol: "Chem.Mol", coords: torch.Tensor, receptor, *, policy: str):
    return build_physical_system(
        mol,
        receptor,
        fragment_id=torch.zeros(coords.shape[0], dtype=torch.long),
        near_coords=coords,
        receptor_policy=policy,
    ).to(torch.device("cpu"), DTYPE)


def test_plain_protein_is_energy_and_gradient_equivalent_between_policies(tmp_path) -> None:
    mol, coords = _molecule()
    receptor = tmp_path / "plain.pdb"
    receptor.write_text(_protein_atom() + "END\n")

    strict = _build(mol, coords, receptor, policy="fail_closed")
    geometry = _build(mol, coords, receptor, policy="geometry_only")

    assert strict.topology.reference_sha256() == geometry.topology.reference_sha256()
    assert strict.interaction_topology is not None
    assert geometry.interaction_topology is not None
    assert (
        strict.interaction_topology.reference_sha256()
        == geometry.interaction_topology.reference_sha256()
    )
    assert not geometry.geometry_obstacle_coords.numel()
    assert geometry.receptor_provenance["metal_fallbacks"] == []
    assert geometry.receptor_policy_identity == receptor_policy_identity("geometry_only")

    strict_coords = coords.clone().requires_grad_(True)
    geometry_coords = coords.clone().requires_grad_(True)
    strict_energy = guidance_energy(strict_coords, strict)
    geometry_energy = guidance_energy(geometry_coords, geometry)
    assert strict_energy.keys() == geometry_energy.keys()
    for name in strict_energy:
        torch.testing.assert_close(strict_energy[name], geometry_energy[name], rtol=0, atol=0)
    strict_gradient = torch.autograd.grad(strict_energy["total"], strict_coords)[0]
    geometry_gradient = torch.autograd.grad(geometry_energy["total"], geometry_coords)[0]
    torch.testing.assert_close(strict_gradient, geometry_gradient, rtol=0, atol=0)


def test_gol_and_heme_are_explicit_repulsion_only_geometry(tmp_path) -> None:
    mol, coords = _molecule()
    receptor = tmp_path / "gol_heme.pdb"
    receptor.write_text(
        _protein_atom()
        + _pdb_line("HETATM", 2, "C1", "GOL", 2, (0.0, 0.0, 0.0), "C")
        + _pdb_line("HETATM", 3, "O1", "GOL", 2, (1.3, 0.0, 0.0), "O")
        + _pdb_line("HETATM", 4, "FE", "HEM", 3, (2.4, 0.0, 0.0), "FE")
        + _pdb_line("HETATM", 5, "NA", "HEM", 3, (2.4, 1.5, 0.0), "N")
        + _pdb_line("HETATM", 6, "C1", "HEM", 3, (2.4, -1.5, 0.0), "C")
        + "END\n"
    )

    with pytest.raises(UnsupportedPhysicalChemistryError):
        _build(mol, coords, receptor, policy="fail_closed")
    system = _build(mol, coords, receptor, policy="geometry_only")
    topology = system.interaction_topology
    assert topology is not None

    assert system.geometry_obstacle_coords.shape == (4, 3)
    assert system.geometry_obstacle_is_generic.tolist() == [False] * 4
    assert set(system.geometry_obstacle_kinds) == {"effff_v2_repulsion_only"}
    assert topology.metal_atomic_number.tolist() == [26]
    assert topology.metal_attraction_enabled.tolist() == [False]
    assert topology.metal_ligand_donor_allowed.tolist() == [[False, False, False]]
    assert topology.zinc_coords.shape == (0, 3)
    fallback = system.receptor_provenance["metal_fallbacks"][0]
    assert fallback["code"] == "unsupported_metal_profile"
    assert "matching PDB residue" in fallback["message"]

    differentiable = coords.clone().requires_grad_(True)
    components = physical_energy(differentiable, system)
    assert "receptor_geometry_obstacle_uff_repulsive" in components
    assert not any("obstacle" in name and "attractive" in name for name in components)
    components["total"].backward()
    assert differentiable.grad is not None
    assert bool(torch.isfinite(differentiable.grad).all())


@pytest.mark.parametrize(
    ("element", "atomic_number"),
    [("NA", 11), ("K", 19), ("V", 23), ("MO", 42)],
)
def test_unregistered_metal_elements_downgrade_with_exact_provenance(
    tmp_path,
    element: str,
    atomic_number: int,
) -> None:
    mol, coords = _molecule()
    receptor = tmp_path / f"{element.lower()}.pdb"
    receptor.write_text(
        _protein_atom()
        + _pdb_line("HETATM", 2, element, element, 2, (0.0, 0.0, 0.0), element)
        + "END\n"
    )

    with pytest.raises(UnsupportedPhysicalChemistryError) as exc_info:
        _build(mol, coords, receptor, policy="fail_closed")
    assert exc_info.value.code == "unsupported_metal_profile"

    system = _build(mol, coords, receptor, policy="geometry_only")
    topology = system.interaction_topology
    assert topology is not None
    assert topology.metal_atomic_number.tolist() == [atomic_number]
    assert topology.metal_attraction_enabled.tolist() == [False]
    provenance = system.receptor_provenance["metal_fallbacks"]
    assert provenance == [
        {
            "metal_site": topology.metal_site_labels[0],
            "element": element,
            "action": "bounded_all_ligand_repulsion_only",
            "vacant_direction_semantics": (
                "inactive_unit_placeholder_required_by_interaction_v1_tensor_contract"
            ),
            "code": exc_info.value.code,
            "message": str(exc_info.value),
            "details": exc_info.value.details,
        }
    ]

    overlap = coords.clone()
    overlap[0] = topology.metal_coords[0]
    differentiable = overlap.requires_grad_(True)
    metal = interaction_energy(differentiable, system)["interaction_metal_coordination"]
    assert torch.isfinite(metal)
    assert float(metal.detach()) > 0
    gradient = torch.autograd.grad(metal, differentiable)[0]
    assert bool(torch.isfinite(gradient).all())


def test_xe_uses_bounded_generic_steric_without_force_field_claim(tmp_path) -> None:
    mol, coords = _molecule()
    receptor = tmp_path / "xe.pdb"
    receptor.write_text(
        _protein_atom()
        + _pdb_line("HETATM", 2, "XE", "XE", 2, (0.0, 0.0, 0.0), "XE")
        + "END\n"
    )

    with pytest.raises(UnsupportedPhysicalChemistryError) as exc_info:
        _build(mol, coords, receptor, policy="fail_closed")
    assert exc_info.value.code == "active_nonprotein_residue"

    system = _build(mol, coords, receptor, policy="geometry_only")
    assert system.geometry_obstacle_atomic_numbers.tolist() == [54]
    assert system.geometry_obstacle_is_generic.tolist() == [True]
    assert system.geometry_obstacle_uff_x.tolist() == [0.0]
    assert system.geometry_obstacle_uff_d.tolist() == [0.0]
    assert system.geometry_obstacle_kinds == ("generic_bounded_steric_v1",)
    assert (
        system.receptor_policy_identity["generic_obstacle_parameter_claim"]
        == "internal geometry-only diagnostic prior"
    )

    overlap = coords.clone()
    overlap[0] = system.geometry_obstacle_coords[0]
    differentiable = overlap.requires_grad_(True)
    components = physical_energy(differentiable, system)
    generic = components["receptor_geometry_obstacle_generic_repulsive"]
    assert torch.isfinite(generic)
    assert 0 < float(generic.detach()) <= (
        PhysicalEnergyConfig().generic_obstacle_repulsion_max * coords.shape[0]
    )
    gradient = torch.autograd.grad(components["total"], differentiable)[0]
    assert bool(torch.isfinite(gradient).all())


@pytest.mark.parametrize(("element", "expected_code"), [("ZN", "unsupported_zinc_site"), ("MG", "unsupported_metal_profile")])
def test_invalid_attractive_metal_site_downgrades_without_attraction(
    tmp_path,
    element: str,
    expected_code: str,
) -> None:
    mol, coords = _molecule()
    receptor = tmp_path / f"invalid_{element.lower()}.pdb"
    receptor.write_text(
        _protein_atom()
        + _pdb_line("HETATM", 2, element, element, 2, (0.0, 0.0, 0.0), element)
        + "END\n"
    )

    with pytest.raises(UnsupportedPhysicalChemistryError) as exc_info:
        _build(mol, coords, receptor, policy="fail_closed")
    assert exc_info.value.code == expected_code

    system = _build(mol, coords, receptor, policy="geometry_only")
    topology = system.interaction_topology
    assert topology is not None
    assert topology.metal_attraction_enabled.tolist() == [False]
    assert topology.metal_fixed_coordination.tolist() == [0]
    assert topology.metal_target_coordination.tolist() == [0]
    assert topology.metal_ligand_r0.tolist() == [[0.0, 0.0, 0.0]]
    assert topology.metal_ligand_donor_allowed.tolist() == [[False, False, False]]
    assert topology.zinc_coords.shape == (0, 3)
    fallback = system.receptor_provenance["metal_fallbacks"][0]
    assert fallback["code"] == expected_code

    differentiable = coords.clone().requires_grad_(True)
    energy = interaction_energy(differentiable, system)["interaction_metal_coordination"]
    assert torch.isfinite(energy)
    assert float(energy.detach()) >= 0
    gradient = torch.autograd.grad(energy, differentiable)[0]
    assert bool(torch.isfinite(gradient).all())

    rotated_placeholder = replace(
        topology,
        metal_vacant_direction=torch.tensor([[0.0, 1.0, 0.0]], dtype=DTYPE),
    )
    rotated_system = replace(system, interaction_topology=rotated_placeholder)
    rotated_energy = interaction_energy(coords, rotated_system)[
        "interaction_metal_coordination"
    ]
    torch.testing.assert_close(rotated_energy, energy.detach(), rtol=0, atol=0)


def test_geometry_policy_traces_filtered_water_and_nucleic_records(tmp_path) -> None:
    mol, coords = _molecule()
    receptor = tmp_path / "filtered.pdb"
    receptor.write_text(
        _protein_atom()
        + _pdb_line("HETATM", 2, "O", "HOH", 2, (0.0, 1.0, 0.0), "O")
        + _pdb_line("HETATM", 3, "P", "DA", 3, (0.0, 2.0, 0.0), "P")
        + "END\n"
    )
    system = _build(mol, coords, receptor, policy="geometry_only")
    assert system.receptor_provenance["filtered_records"] == {
        "water_heavy_atoms": 1,
        "nucleic_acid_heavy_atoms": 1,
        "nonprimary_nonmetal_altloc_heavy_atoms": 0,
        "nonprimary_nonmetal_altloc_records": [],
        "policy": (
            "shared parser filters water, nucleic-acid, and non-primary "
            "alternate-location records"
        ),
    }
    assert system.geometry_obstacle_coords.shape == (0, 3)


def test_v_mo_are_guidance_local_metals_without_model_feature_change(tmp_path) -> None:
    mol, coords = _molecule()
    receptor = tmp_path / "v_mo.pdb"
    receptor.write_text(
        _protein_atom()
        + _pdb_line("HETATM", 2, "V", "V", 2, (0.0, 0.0, 0.0), "V")
        + _pdb_line("HETATM", 3, "MO", "MO", 3, (2.0, 0.0, 0.0), "MO")
        + "END\n"
    )

    features = parse_pocket_atoms(receptor)
    assert features is not None
    assert features["patom_is_metal"].tolist() == [False, False, False]
    assert features["patom_is_positive"].tolist() == [False, False, False]
    assert features["patom_token"][-2:].tolist() == [UNK_ATOM_TOKEN, UNK_ATOM_TOKEN]
    for name in (
        "patom_is_backbone",
        "patom_is_donor",
        "patom_is_acceptor",
        "patom_is_negative",
        "patom_is_hydrophobic",
    ):
        assert features[name][-2:].tolist() == [False, False]
    assert features["pres_coords"].shape == (1, 3)

    with pytest.raises(UnsupportedPhysicalChemistryError) as exc_info:
        _build(mol, coords, receptor, policy="fail_closed")
    assert exc_info.value.code == "unsupported_metal_profile"

    system = _build(mol, coords, receptor, policy="geometry_only")
    topology = system.interaction_topology
    assert topology is not None
    assert topology.metal_atomic_number.tolist() == [23, 42]
    assert topology.metal_attraction_enabled.tolist() == [False, False]
    assert system.geometry_obstacle_coords.shape == (0, 3)
    assert [item["element"] for item in system.receptor_provenance["metal_fallbacks"]] == [
        "V",
        "MO",
    ]


def test_primary_and_nonprimary_metal_altlocs_form_one_fallback_site(tmp_path) -> None:
    mol, coords = _molecule()
    receptor = tmp_path / "zn_a_b.pdb"
    receptor.write_text(
        _protein_atom()
        + _pdb_line(
            "HETATM",
            2,
            "ZN",
            "ZN",
            2,
            (0.0, 0.0, 0.0),
            "ZN",
            altloc="A",
            occupancy=0.6,
        )
        + _pdb_line(
            "HETATM",
            3,
            "ZN",
            "ZN",
            2,
            (0.2, 0.0, 0.0),
            "ZN",
            altloc="B",
            occupancy=0.4,
        )
        + "END\n"
    )
    system = _build(mol, coords, receptor, policy="geometry_only")
    topology = system.interaction_topology
    assert topology is not None
    assert topology.metal_coords.shape == (1, 3)
    assert topology.metal_attraction_enabled.tolist() == [False]
    assert system.receptor_provenance["metal_fallback_count"] == 1
    assert system.receptor_provenance["metal_fallbacks"][0]["details"] == {
        "metal_site": "A:ZN2:ZN",
        "matching_record_count": 2,
    }


def test_b_c_only_metal_altloc_group_is_one_deterministic_site(tmp_path) -> None:
    mol, coords = _molecule()
    receptor = tmp_path / "zn_b_c.pdb"
    receptor.write_text(
        _protein_atom()
        + _pdb_line(
            "HETATM",
            2,
            "ZN",
            "ZN",
            2,
            (0.3, 0.0, 0.0),
            "ZN",
            altloc="B",
            occupancy=0.5,
        )
        + _pdb_line(
            "HETATM",
            3,
            "ZN",
            "ZN",
            2,
            (0.7, 0.0, 0.0),
            "ZN",
            altloc="C",
            occupancy=0.5,
        )
        + "END\n"
    )

    with pytest.raises(UnsupportedPhysicalChemistryError):
        _build(mol, coords, receptor, policy="fail_closed")
    first = _build(mol, coords, receptor, policy="geometry_only")
    second = _build(mol, coords, receptor, policy="geometry_only")
    first_topology = first.interaction_topology
    second_topology = second.interaction_topology
    assert first_topology is not None and second_topology is not None
    assert first_topology.metal_coords.shape == (1, 3)
    assert first_topology.metal_site_labels == ("A:ZN2:ZN:altloc_group=B,C",)
    torch.testing.assert_close(
        first_topology.metal_coords,
        torch.tensor([[0.3, 0.0, 0.0]], dtype=DTYPE),
        rtol=0,
        atol=0,
    )
    assert first_topology.reference_sha256() == second_topology.reference_sha256()
    fallback = first.receptor_provenance["metal_fallbacks"][0]
    assert fallback["details"]["selected_line_number"] == 2
    assert [record["altloc"] for record in fallback["details"]["records"]] == ["B", "C"]


def test_nonprimary_nonmetal_altloc_records_are_explicitly_traced(tmp_path) -> None:
    mol, coords = _molecule()
    receptor = tmp_path / "gol_altloc.pdb"
    receptor.write_text(
        _protein_atom()
        + _pdb_line(
            "HETATM",
            2,
            "C1",
            "GOL",
            2,
            (0.0, 0.0, 0.0),
            "C",
            altloc="B",
            occupancy=0.5,
        )
        + _pdb_line(
            "HETATM",
            3,
            "C1",
            "GOL",
            2,
            (0.2, 0.0, 0.0),
            "C",
            altloc="C",
            occupancy=0.5,
        )
        + "END\n"
    )
    system = _build(mol, coords, receptor, policy="geometry_only")
    filtered = system.receptor_provenance["filtered_records"]
    assert filtered["nonprimary_nonmetal_altloc_heavy_atoms"] == 2
    assert [record["line_number"] for record in filtered["nonprimary_nonmetal_altloc_records"]] == [
        2,
        3,
    ]
    assert [record["altloc"] for record in filtered["nonprimary_nonmetal_altloc_records"]] == [
        "B",
        "C",
    ]
    assert system.geometry_obstacle_coords.shape == (0, 3)
