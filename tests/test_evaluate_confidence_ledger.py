from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from rdkit import Chem

from effdock.confidence import runtime as confidence_runtime
from effdock.confidence import selectors as confidence_selectors
from effdock.inference import preprocess as inference_preprocess
from effdock.workflows import evaluate
from effdock.workflows.evaluate import (
    CONFIDENCE_SCORE_LEDGER_FIELDS,
    ComplexInput,
    candidate_ensemble_sha256,
    confidence_score_ledger_json,
    evaluate_one,
    main,
    pose_diversity_metrics,
    select_confidence_cluster_free,
    summarize_rows,
)


def _score(offset: float = 0.0) -> dict[str, float]:
    return {
        "confidence_rmsd": 1.0 + offset,
        "confidence_success_logit": 0.2 + offset,
        "confidence_success": 0.55 + offset,
        "confidence_atom_rmsd": 1.1 + offset,
        "confidence_atom_q90": 1.8 + offset,
        "confidence_atom_ok": 0.7 + offset,
        "pl_clash_1p6": 0.0 + offset,
    }


def test_candidate_ensemble_hash_is_stable_and_order_sensitive() -> None:
    first = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    second = torch.tensor([[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]])
    expected = candidate_ensemble_sha256([first, second])

    assert candidate_ensemble_sha256([first.to(torch.float64), second]) == expected
    assert candidate_ensemble_sha256([second, first]) != expected

    changed = second.clone()
    changed[0, 0] += 0.01
    assert candidate_ensemble_sha256([first, changed]) != expected


def test_candidate_ensemble_hash_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="poses must be non-empty"):
        candidate_ensemble_sha256([])


def test_confidence_score_ledger_projects_finite_selector_inputs() -> None:
    first = _score()
    first["confidence_top8_pl_clash1p6_w0.5"] = float("inf")
    ledger = json.loads(confidence_score_ledger_json([first, _score(0.1)]))

    assert len(ledger) == 2
    assert tuple(sorted(ledger[0])) == tuple(sorted(CONFIDENCE_SCORE_LEDGER_FIELDS))
    assert "confidence_top8_pl_clash1p6_w0.5" not in ledger[0]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_confidence_score_ledger_rejects_non_finite_values(invalid: float) -> None:
    score = _score()
    score["confidence_rmsd"] = invalid
    with pytest.raises(ValueError, match="must be finite"):
        confidence_score_ledger_json([score])


def test_confidence_score_ledger_requires_every_field() -> None:
    score = _score()
    del score["pl_clash_1p6"]
    with pytest.raises(ValueError, match="lacks required field"):
        confidence_score_ledger_json([score])


def test_cluster_free_selector_is_pure_argmin_plus_fixed_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_pairwise(*_args, **_kwargs):
        raise AssertionError("pairwise pose clustering must not run")

    monkeypatch.setattr(confidence_selectors, "_pairwise_pose_rmsd", reject_pairwise)
    poses = [torch.tensor([[float(index), 0.0, 0.0]]) for index in range(3)]
    scores = [_score(0.5), _score(-0.6), _score(0.2)]
    graph = {
        "node_type": torch.tensor([2]),
        "node_coords": torch.tensor([[100.0, 100.0, 100.0]]),
    }

    selected = select_confidence_cluster_free(
        poses,
        scores,
        graph,
        torch.zeros(3),
    )

    assert selected == {"confidence": 1, "confidence_filter": 1}
    assert [score["pl_clash_1p6"] for score in scores] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "selector_profile, extra_args, expected",
    [
        (
            "confidence_cluster_free",
            ["--no-confidence"],
            "requires --confidence-checkpoint",
        ),
        (
            "confidence_cluster_free",
            ["--trajectory-dir", "traj"],
            "does not support --trajectory-dir",
        ),
        (
            "candidate_only",
            ["--trajectory-dir", "traj"],
            "does not support --trajectory-dir",
        ),
    ],
)
def test_cluster_free_cli_rejects_incompatible_options(
    selector_profile: str,
    extra_args: list[str],
    expected: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--dataset",
                "astex",
                "--data-dir",
                "data",
                "--pocket-centers",
                "centers.json",
                "--selector-profile",
                selector_profile,
                *extra_args,
            ]
        )
    assert expected in capsys.readouterr().err


def _single_atom_mol() -> Chem.Mol:
    mol = Chem.MolFromSmiles("C")
    assert mol is not None
    conformer = Chem.Conformer(1)
    conformer.SetAtomPosition(0, (0.0, 0.0, 0.0))
    mol.AddConformer(conformer)
    return mol


def test_cluster_free_profile_omits_legacy_scoring_and_pose_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol = _single_atom_mol()
    protein = tmp_path / "protein.pdb"
    reference = tmp_path / "reference.sdf"
    protein.write_text("END\n")
    reference.write_text("reference\n")
    item = ComplexInput(
        complex_id="case",
        protein=protein,
        ligand_ref=reference,
        ligand_format="sdf",
        smiles="C",
        pocket_center=(0.0, 0.0, 0.0),
    )
    poses = [
        torch.tensor([[3.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[2.0, 0.0, 0.0]]),
    ]
    graph = {
        "node_type": torch.tensor([2]),
        "node_coords": torch.tensor([[100.0, 100.0, 100.0]]),
    }
    lig_data = {"fragment_id": torch.tensor([0]), "frag_sizes": torch.tensor([1])}
    meta = {"pocket_center": torch.zeros(3), "num_frag": 1}

    monkeypatch.setattr(evaluate, "load_ref_ligand", lambda *_args: mol)
    monkeypatch.setattr(inference_preprocess, "load_ligand", lambda *_args, **_kwargs: (mol, {}))
    monkeypatch.setattr(evaluate, "match_atoms", lambda *_args: ([0], [0], "identity"))
    monkeypatch.setattr(
        evaluate,
        "full_heavy_atom_mapping_metadata",
        lambda *_args: {"accepted": True, "relation": "exact_graph"},
    )
    monkeypatch.setattr(
        evaluate,
        "preprocess_complex",
        lambda *_args, **_kwargs: (graph, lig_data, meta),
    )
    monkeypatch.setattr(
        evaluate,
        "sample_unified",
        lambda *_args, **_kwargs: [{"atom_pos_pred": pose} for pose in poses],
    )
    monkeypatch.setattr(evaluate, "apply_refinement", lambda _mode, values, *_args: values)
    monkeypatch.setattr(
        evaluate,
        "compute_pose_rmsd_with_method",
        lambda pose, *_args: (
            float(pose[0, 0]),
            "rdkit_calc_rms_symmetry_no_align",
        ),
    )
    monkeypatch.setattr(
        confidence_runtime,
        "score_poses_with_confidence",
        lambda *_args, **_kwargs: [_score(0.5), _score(-0.6), _score(0.2)],
    )
    monkeypatch.setattr(
        confidence_runtime,
        "sample_sigmas",
        lambda results, _sigma: torch.ones(len(results)),
    )

    def reject_legacy(*_args, **_kwargs):
        raise AssertionError("legacy selector/scorer must not run")

    monkeypatch.setattr(evaluate, "score_poses", reject_legacy)
    monkeypatch.setattr(evaluate, "select_by_score", reject_legacy)
    monkeypatch.setattr(confidence_selectors, "select_confidence_poses", reject_legacy)
    monkeypatch.setattr(
        evaluate,
        "build_protein_vina_inputs",
        lambda *_args, **_kwargs: {
            "coords": torch.tensor([[100.0, 100.0, 100.0]]),
            "atomic_nums": torch.tensor([6]),
        },
    )
    monkeypatch.setattr(evaluate, "ligand_bounds", lambda _mol: {})
    monkeypatch.setattr(evaluate, "vdw_radii", lambda values: torch.ones(len(values)))
    monkeypatch.setattr(
        evaluate,
        "check_validity",
        lambda *_args, **_kwargs: {"valid": True, "bond_lengths": True},
    )

    pose_dir = tmp_path / "poses"
    row = evaluate_one(
        torch.nn.Identity(),
        item,
        dataset="astex",
        confidence_model=torch.nn.Identity(),
        device=torch.device("cpu"),
        num_samples=3,
        num_steps=1,
        sigma=0.5,
        sigma_list=[],
        sigma_counts=[],
        center_jitter_sigma=0.0,
        pocket_cutoff=10.0,
        pose_objective="linear_fm",
        score_rot_sigma_max=float(torch.pi),
        score_alpha_min=0.0,
        time_schedule="uniform",
        schedule_power=1.0,
        vina_guidance_scale=0.0,
        vina_guidance_start_t=0.5,
        vina_guidance_ramp_power=1.0,
        vina_guidance_max_force=10.0,
        vina_guidance_max_velocity=5.0,
        vina_guidance_max_angular_velocity=5.0,
        vina_guidance_protein_shell=18.0,
        vina_guidance_w_strain=1.0,
        unified_guidance_scale=0.0,
        unified_guidance_start_t=0.5,
        unified_guidance_ramp_power=1.0,
        unified_guidance_max_force=20.0,
        unified_guidance_max_velocity=5.0,
        unified_guidance_max_angular_velocity=5.0,
        unified_guidance_max_atom_displacement=0.25,
        unified_guidance_max_backtracks=8,
        unified_guidance_protein_shell=18.0,
        unified_guidance_receptor_policy="geometry_only",
        unified_guidance_mode="normalized_drift",
        prior_pool_size=0,
        seed=42,
        refine="none",
        pose_dir=pose_dir,
        trajectory_dir=None,
        require_full_ligand_atom_mapping=False,
        selector_profile="confidence_cluster_free",
    )

    assert row["selector_profile"] == "confidence_cluster_free"
    assert row["confidence_index"] == 1
    assert row["confidence_filter_index"] == 1
    assert "first_rmsd" in row and "oracle_rmsd" in row
    assert not any(key.startswith("vina_") for key in row)
    assert not any(key.startswith("confidence_final_") for key in row)
    assert set(json.loads(row["saved_pose_sha256_json"])) == {
        "confidence",
        "confidence_filter",
    }
    assert sorted(path.parent.name for path in pose_dir.rglob("*.sdf")) == [
        "all_poses",
        "confidence",
        "confidence_filter",
    ]
    all_poses_path = pose_dir / "all_poses" / "case.sdf"
    all_poses = [mol for mol in Chem.SDMolSupplier(str(all_poses_path), removeHs=False)]
    assert len(all_poses) == 3
    assert [mol.GetIntProp("sample_index") for mol in all_poses] == [0, 1, 2]
    assert [mol.GetDoubleProp("confidence_rmsd") for mol in all_poses] == [1.5, 0.4, 1.2]
    assert [mol.GetDoubleProp("confidence_pred_rmsd") for mol in all_poses] == [
        1.5,
        0.4,
        1.2,
    ]
    assert row["all_poses_count"] == 3
    assert row["all_poses_sdf"] == str(all_poses_path)
    assert row["all_poses_sdf_sha256"] == evaluate.file_sha256(all_poses_path)
    cluster_free_fast_valid = json.loads(row["candidate_fast_valid_json"])
    assert cluster_free_fast_valid == [True, True, True]
    assert len(cluster_free_fast_valid) == row["num_samples"]
    assert sum(cluster_free_fast_valid) == row["num_fast_valid_candidates"]
    cluster_free_rmsd_methods = json.loads(row["candidate_rmsd_method_json"])
    assert cluster_free_rmsd_methods == ["rdkit_calc_rms_symmetry_no_align"] * 3
    assert row["num_mapped_index_rmsd_fallback_candidates"] == 0
    assert set(summarize_rows([row])) == {
        "first",
        "confidence",
        "confidence_filter",
        "oracle",
        "candidate_set",
    }
    assert row["ligand_conformer_seed"] == 42


def test_pose_diversity_uses_heavy_atoms_and_stable_rounding() -> None:
    editable = Chem.RWMol()
    carbon = editable.AddAtom(Chem.Atom(6))
    hydrogen = editable.AddAtom(Chem.Atom(1))
    editable.AddBond(carbon, hydrogen, Chem.BondType.SINGLE)
    mol = editable.GetMol()
    poses = [
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        torch.tensor([[0.00049, 0.0, 0.0], [100.0, 0.0, 0.0]]),
        torch.tensor([[0.00051, 0.0, 0.0], [-100.0, 0.0, 0.0]]),
    ]

    metrics = pose_diversity_metrics(poses, mol)

    assert metrics["diversity_heavy_atom_count"] == 1
    assert metrics["coordinate_unique_count"] == 2
    assert metrics["pairwise_heavy_atom_rmsd_mean"] == pytest.approx(0.00034)
    assert metrics["pairwise_heavy_atom_rmsd_median"] == pytest.approx(0.00049)
    assert metrics["pairwise_heavy_atom_rmsd_ge2_fraction"] == 0.0
    assert metrics["nearest_neighbor_heavy_atom_rmsd_median"] == pytest.approx(
        0.00002,
        rel=1e-5,
    )
    assert metrics["c2_connected_component_count"] == 1

    strict_boundary = pose_diversity_metrics(
        [
            torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            torch.tensor([[2.0, 0.0, 0.0], [-10.0, 0.0, 0.0]]),
            torch.tensor([[3.9, 0.0, 0.0], [20.0, 0.0, 0.0]]),
        ],
        mol,
    )
    assert strict_boundary["c2_connected_component_count"] == 2


def test_candidate_only_preserves_candidate_metrics_without_selector_scoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol = _single_atom_mol()
    protein = tmp_path / "protein.pdb"
    reference = tmp_path / "reference.sdf"
    protein.write_text("END\n")
    reference.write_text("reference\n")
    item = ComplexInput(
        complex_id="case",
        protein=protein,
        ligand_ref=reference,
        ligand_format="sdf",
        smiles="C",
        pocket_center=(0.0, 0.0, 0.0),
    )
    poses = [
        torch.tensor([[3.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[2.0, 0.0, 0.0]]),
    ]
    graph = {
        "node_type": torch.tensor([2]),
        "node_coords": torch.tensor([[100.0, 100.0, 100.0]]),
    }
    lig_data = {"fragment_id": torch.tensor([0]), "frag_sizes": torch.tensor([1])}
    meta = {"pocket_center": torch.zeros(3), "num_frag": 1}
    conformer_seeds: list[int] = []

    monkeypatch.setattr(evaluate, "load_ref_ligand", lambda *_args: mol)

    def load_ligand(_input: str, *, random_seed: int):
        conformer_seeds.append(random_seed)
        return mol, {}

    monkeypatch.setattr(inference_preprocess, "load_ligand", load_ligand)
    monkeypatch.setattr(evaluate, "match_atoms", lambda *_args: ([0], [0], "identity"))
    monkeypatch.setattr(
        evaluate,
        "full_heavy_atom_mapping_metadata",
        lambda *_args: {"accepted": True, "relation": "exact_graph"},
    )
    monkeypatch.setattr(
        evaluate,
        "preprocess_complex",
        lambda *_args, **_kwargs: (graph, lig_data, meta),
    )
    monkeypatch.setattr(
        evaluate,
        "sample_unified",
        lambda *_args, **_kwargs: [{"atom_pos_pred": pose} for pose in poses],
    )
    monkeypatch.setattr(evaluate, "apply_refinement", lambda _mode, values, *_args: values)
    monkeypatch.setattr(
        evaluate,
        "compute_pose_rmsd_with_method",
        lambda pose, *_args: (
            float(pose[0, 0]),
            (
                "mapped_index_fallback"
                if float(pose[0, 0]) == 2.0
                else "rdkit_calc_rms_symmetry_no_align"
            ),
        ),
    )

    def reject_selector_scoring(*_args, **_kwargs):
        raise AssertionError("candidate_only must not run Vina or confidence selection")

    monkeypatch.setattr(evaluate, "score_poses", reject_selector_scoring)
    monkeypatch.setattr(
        confidence_runtime,
        "score_poses_with_confidence",
        reject_selector_scoring,
    )
    monkeypatch.setattr(
        evaluate,
        "build_protein_vina_inputs",
        lambda *_args, **_kwargs: {
            "coords": torch.tensor([[100.0, 100.0, 100.0]]),
            "atomic_nums": torch.tensor([6]),
        },
    )
    monkeypatch.setattr(evaluate, "ligand_bounds", lambda _mol: {})
    monkeypatch.setattr(evaluate, "vdw_radii", lambda values: torch.ones(len(values)))
    monkeypatch.setattr(
        evaluate,
        "check_validity",
        lambda pose, *_args, **_kwargs: {
            "valid": float(pose[0, 0]) != 2.0,
            "bond_lengths": True,
        },
    )

    pose_dir = tmp_path / "poses"
    row = evaluate_one(
        torch.nn.Identity(),
        item,
        dataset="astex",
        confidence_model=torch.nn.Identity(),
        device=torch.device("cpu"),
        num_samples=3,
        num_steps=1,
        sigma=0.5,
        sigma_list=[],
        sigma_counts=[],
        center_jitter_sigma=0.0,
        pocket_cutoff=10.0,
        pose_objective="linear_fm",
        score_rot_sigma_max=float(torch.pi),
        score_alpha_min=0.0,
        time_schedule="uniform",
        schedule_power=1.0,
        vina_guidance_scale=0.0,
        vina_guidance_start_t=0.5,
        vina_guidance_ramp_power=1.0,
        vina_guidance_max_force=10.0,
        vina_guidance_max_velocity=5.0,
        vina_guidance_max_angular_velocity=5.0,
        vina_guidance_protein_shell=18.0,
        vina_guidance_w_strain=1.0,
        unified_guidance_scale=0.0,
        unified_guidance_start_t=0.5,
        unified_guidance_ramp_power=1.0,
        unified_guidance_max_force=20.0,
        unified_guidance_max_velocity=5.0,
        unified_guidance_max_angular_velocity=5.0,
        unified_guidance_max_atom_displacement=0.25,
        unified_guidance_max_backtracks=8,
        unified_guidance_protein_shell=18.0,
        unified_guidance_receptor_policy="geometry_only",
        unified_guidance_mode="normalized_drift",
        prior_pool_size=0,
        seed=42,
        refine="none",
        pose_dir=pose_dir,
        trajectory_dir=None,
        require_full_ligand_atom_mapping=False,
        selector_profile="candidate_only",
        ligand_conformer_seed=0,
    )

    assert conformer_seeds == [0]
    assert row["sampling_seed"] == 42
    assert row["ligand_conformer_seed"] == 0
    assert row["selector_profile"] == "candidate_only"
    assert row["selected_index"] == row["first_index"] == 0
    assert row["selected_rmsd"] == row["first_rmsd"] == 3.0
    assert row["selected_fast_valid"]
    assert json.loads(row["candidate_rmsds_json"]) == [3.0, 1.0, 2.0]
    candidate_rmsd_methods = json.loads(row["candidate_rmsd_method_json"])
    assert candidate_rmsd_methods == [
        "rdkit_calc_rms_symmetry_no_align",
        "rdkit_calc_rms_symmetry_no_align",
        "mapped_index_fallback",
    ]
    assert len(candidate_rmsd_methods) == row["num_samples"]
    assert row["num_mapped_index_rmsd_fallback_candidates"] == candidate_rmsd_methods.count(
        "mapped_index_fallback"
    )
    candidate_fast_valid = json.loads(row["candidate_fast_valid_json"])
    assert candidate_fast_valid == [True, True, False]
    assert len(candidate_fast_valid) == row["num_samples"]
    assert sum(candidate_fast_valid) == row["num_fast_valid_candidates"]
    assert row["num_rmsd_lt2_candidates"] == 1
    assert row["num_fast_valid_candidates"] == 2
    assert row["num_fast_valid_rmsd_lt2_candidates"] == 1
    assert row["fast_valid_oracle_index"] == 1
    assert row["fast_valid_oracle_rmsd"] == 1.0
    assert row["joint_fast_valid_and_rmsd_lt2"]
    assert row["coordinate_unique_count"] == 3
    assert row["pairwise_heavy_atom_rmsd_mean"] == pytest.approx(4.0 / 3.0)
    assert row["pairwise_heavy_atom_rmsd_median"] == 1.0
    assert row["pairwise_heavy_atom_rmsd_ge2_fraction"] == pytest.approx(1.0 / 3.0)
    assert row["nearest_neighbor_heavy_atom_rmsd_median"] == 1.0
    assert row["c2_connected_component_count"] == 1
    assert row["pose_diversity_round_decimals"] == 3
    assert "candidate_ensemble_sha256" in row
    assert not any(key.startswith("vina_") for key in row)
    assert not any(key.startswith("confidence_") for key in row)
    assert set(json.loads(row["saved_pose_sha256_json"])) == {"selected"}
    assert sorted(path.parent.name for path in pose_dir.rglob("*.sdf")) == [
        "all_poses",
        "selected",
    ]
    all_poses_path = pose_dir / "all_poses" / "case.sdf"
    all_poses = [mol for mol in Chem.SDMolSupplier(str(all_poses_path), removeHs=False)]
    assert len(all_poses) == 3
    assert [mol.GetIntProp("sample_index") for mol in all_poses] == [0, 1, 2]
    assert set(summarize_rows([row])) == {
        "selected",
        "first",
        "oracle",
        "candidate_set",
    }
