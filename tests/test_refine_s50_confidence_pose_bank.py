from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import torch
from rdkit import Chem

from effdock.guidance import build_physical_system
from effdock.preprocess.protein import (
    METAL_ATOM_TOKENS,
    METAL_OTHER_TOKEN,
    RES_ATOM_TOKEN,
    UNK_ATOM_TOKEN,
)
from scripts.calibrate_s50_refinement_budget import _compare, _config
from scripts.refine_s50_confidence_pose_bank import (
    PROTOCOL_ID,
    REFINEMENT_BATCH_SIZE,
    _ordered_ids_sha256,
    _protein_pdb_text,
    _refinement_config,
    _selected_records,
    _semantic_implementation_contract,
)


def test_processed_protein_reconstruction_preserves_standard_atom_identity() -> None:
    protein = {
        "patom_coords": torch.tensor(
            [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
            dtype=torch.float32,
        ),
        "patom_token": torch.tensor(
            [
                RES_ATOM_TOKEN[("ALA", "N")],
                RES_ATOM_TOKEN[("ALA", "CA")],
                RES_ATOM_TOKEN[("ALA", "OXT")]
                if ("ALA", "OXT") in RES_ATOM_TOKEN
                else RES_ATOM_TOKEN[("ANY", "OXT")],
            ],
            dtype=torch.long,
        ),
        "patom_residue_id": torch.tensor([0, 0, 0], dtype=torch.long),
    }
    text = _protein_pdb_text(protein)
    lines = text.splitlines()
    assert len(lines) == 4
    assert " ALA A   1" in lines[0]
    assert lines[0].endswith(" N")
    assert lines[1].endswith(" C")
    assert "OXT" in lines[2]
    assert lines[2].endswith(" O")
    assert lines[-1] == "END"


def test_processed_protein_reconstruction_keeps_unknown_residue_backbone() -> None:
    protein = {
        "patom_coords": torch.tensor(
            [[0.0, 0.0, 0.0], [1.3, 0.0, 0.0], [2.6, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        "patom_token": torch.tensor(
            [
                RES_ATOM_TOKEN[("UNK", "N")],
                RES_ATOM_TOKEN[("UNK", "CA")],
                RES_ATOM_TOKEN[("UNK", "C")],
            ],
            dtype=torch.long,
        ),
        "patom_residue_id": torch.tensor([0, 0, 0], dtype=torch.long),
    }

    lines = _protein_pdb_text(protein).splitlines()
    assert all(" UNK A   1" in line for line in lines[:3])
    assert lines[0].endswith(" N")
    assert lines[1].endswith(" C")
    assert lines[2].endswith(" C")


def test_processed_protein_reconstruction_maps_catchall_to_generic_obstacle() -> None:
    protein = {
        "patom_coords": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        "patom_token": torch.tensor([UNK_ATOM_TOKEN], dtype=torch.long),
        "patom_residue_id": torch.tensor([0], dtype=torch.long),
    }

    line = _protein_pdb_text(protein).splitlines()[0]
    assert line.startswith("HETATM")
    assert " GEO A   1" in line
    assert line.endswith(" X")


def test_repeated_catchalls_in_one_residue_have_unique_obstacle_labels() -> None:
    protein = {
        "patom_coords": torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.3, 0.0, 0.0],
                [2.6, 0.0, 0.0],
                [3.2, 1.0, 0.0],
                [0.0, 2.0, 0.0],
                [1.0, 2.0, 0.0],
                [2.0, 2.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "patom_token": torch.tensor(
            [
                RES_ATOM_TOKEN[("ALA", "N")],
                RES_ATOM_TOKEN[("ALA", "CA")],
                RES_ATOM_TOKEN[("ALA", "C")],
                RES_ATOM_TOKEN[("ALA", "O")],
                UNK_ATOM_TOKEN,
                UNK_ATOM_TOKEN,
                METAL_OTHER_TOKEN,
            ],
            dtype=torch.long,
        ),
        "patom_residue_id": torch.tensor([0, 0, 0, 0, 1, 1, 1]),
    }
    text = _protein_pdb_text(protein)
    obstacle_names = [line[12:16].strip() for line in text.splitlines()[4:7]]
    assert len(set(obstacle_names)) == 3

    molecule = Chem.MolFromSmiles("CC")
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    conformer.SetAtomPosition(0, (0.0, 0.0, 0.0))
    conformer.SetAtomPosition(1, (1.5, 0.0, 0.0))
    molecule.AddConformer(conformer)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb") as handle:
        handle.write(text)
        handle.flush()
        system = build_physical_system(
            molecule,
            Path(handle.name),
            fragment_id=torch.tensor([0, 0]),
            near_coords=torch.zeros((1, 3)),
            receptor_policy="geometry_only",
        )

    assert system.geometry_obstacle_coords.shape == (3, 3)
    assert len(set(system.geometry_obstacle_labels)) == 3
    assert bool(system.geometry_obstacle_is_generic.all())


def test_processed_protein_reconstruction_covers_entire_frozen_token_vocabulary() -> None:
    tokens = sorted(
        {
            *RES_ATOM_TOKEN.values(),
            UNK_ATOM_TOKEN,
            *METAL_ATOM_TOKENS.values(),
            METAL_OTHER_TOKEN,
        }
    )
    protein = {
        "patom_coords": torch.stack(
            [torch.tensor([float(index), 0.0, 0.0]) for index in range(len(tokens))]
        ),
        "patom_token": torch.tensor(tokens, dtype=torch.long),
        "patom_residue_id": torch.arange(len(tokens), dtype=torch.long),
    }

    lines = _protein_pdb_text(protein).splitlines()
    assert len(lines) == len(tokens) + 1
    assert lines[-1] == "END"


def test_selected_records_uses_frozen_one_based_split_index_striding() -> None:
    bank = {
        "records": [
            {"sample_key": f"id{i}", "split": "train", "split_index": i} for i in range(1, 8)
        ]
    }
    assert [
        row["sample_key"]
        for row in _selected_records(bank, split="train", shard_index=1, num_shards=3)
    ] == ["id2", "id5"]


def test_ordered_id_digest_is_order_sensitive() -> None:
    assert _ordered_ids_sha256(["a", "b"]) != _ordered_ids_sha256(["b", "a"])


def test_implementation_contract_ignores_only_observational_pyproject_digest() -> None:
    baseline = {
        "guidance": {
            "schema_version": "effdock.guidance_implementation.v1",
            "sha256": "1" * 64,
            "files": ["guidance/runtime.py"],
            "runtime_versions": {"torch_runtime": "2.10.0+cu130"},
            "project_inputs": {
                "pyproject.toml": "2" * 64,
                "uv.lock": "3" * 64,
            },
        },
        "parameters": {"sha256": "4" * 64},
        "torch": "2.10.0+cu130",
    }
    packaging_edit = copy.deepcopy(baseline)
    packaging_edit["guidance"]["sha256"] = "5" * 64
    packaging_edit["guidance"]["project_inputs"]["pyproject.toml"] = "6" * 64

    assert _semantic_implementation_contract(
        packaging_edit
    ) == _semantic_implementation_contract(baseline)

    for path, changed_value in (
        (("guidance", "files"), ["guidance/changed.py"]),
        (("guidance", "runtime_versions"), {"torch_runtime": "changed"}),
        (("guidance", "project_inputs", "uv.lock"), "7" * 64),
        (("parameters",), {"sha256": "8" * 64}),
        (("torch",), "changed"),
    ):
        changed = copy.deepcopy(baseline)
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = changed_value
        assert _semantic_implementation_contract(
            changed
        ) != _semantic_implementation_contract(baseline)


def test_refined_bank_v2_uses_calibrated_safe_ceiling_and_batch() -> None:
    config = _refinement_config()

    assert PROTOCOL_ID == "EFFDOCK-S50-REFINED-POSE-BANK-V2"
    assert REFINEMENT_BATCH_SIZE == 20
    assert config.max_steps == 100
    assert config.convergence_displacement_angstrom == 0.01
    assert config.convergence_patience == 5
    assert config.convergence_energy_absolute_kcal_mol is None
    assert config.convergence_energy_relative is None


def test_adaptive_budget_uses_requested_displacement_and_energy_stops() -> None:
    config = _config(75, adaptive=True)
    rescue = _config(90, adaptive=True)
    displacement_only = _config(100, adaptive=True, energy_stop=False)

    assert config.max_steps == 75
    assert config.convergence_displacement_angstrom == 0.01
    assert config.convergence_patience == 5
    assert config.convergence_energy_absolute_kcal_mol == 0.02
    assert config.convergence_energy_relative == 0.001
    assert config.convergence_energy_min_steps == 25
    assert rescue.max_steps == 90
    assert rescue.convergence_displacement_angstrom == 0.01
    assert displacement_only.max_steps == 100
    assert displacement_only.convergence_displacement_angstrom == 0.01
    assert displacement_only.convergence_energy_absolute_kcal_mol is None
    assert displacement_only.convergence_energy_relative is None


def test_budget_comparison_requires_every_physical_gate() -> None:
    baseline = {
        "elapsed_seconds": 100.0,
        "coords": torch.zeros((2, 2, 3)),
        "energies": [1.0, 2.0],
        "statuses": ["max_steps", "max_steps"],
    }
    passing = {
        "elapsed_seconds": 75.0,
        "coords": torch.full((2, 2, 3), 0.01),
        "energies": [1.1, 2.1],
        "statuses": ["converged_displacement", "max_steps"],
    }
    failing = {**passing, "energies": [10.0, 20.0]}

    assert _compare(passing, baseline)["passed"] is True
    assert _compare(failing, baseline)["passed"] is False


def test_budget_comparison_supports_ligands_with_different_atom_counts() -> None:
    baseline = {
        "elapsed_seconds": 100.0,
        "coords": [torch.zeros((2, 2, 3)), torch.zeros((2, 5, 3))],
        "energies": [1.0, 2.0, 3.0, 4.0],
        "statuses": ["max_steps"] * 4,
    }
    candidate = {
        "elapsed_seconds": 75.0,
        "coords": [torch.full((2, 2, 3), 0.01), torch.full((2, 5, 3), 0.01)],
        "energies": [1.1, 2.1, 3.1, 4.1],
        "statuses": ["converged_displacement"] * 4,
    }

    assert _compare(candidate, baseline)["passed"] is True
