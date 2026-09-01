from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

Chem = pytest.importorskip("rdkit.Chem")

sys.path.insert(0, str(Path(__file__).parents[1]))

from effdock.workflows.guidance_budget_posebusters_report import (  # noqa: E402
    VALIDITY_CHECKS,
)
from effdock.workflows.guidance_pl_valid import (  # noqa: E402
    COFACTOR_AND_WATER_CHECKS,
    PL_VALIDITY_CHECKS,
    is_pl_valid,
)
from scripts.evaluate_guidance_sdf_post_refinement_posebusters import (  # noqa: E402
    validate_pb_row,
)
from scripts.report_guidance_sdf_post_refinement_full import (  # noqa: E402
    _decomposition_row,
    _stage_metrics,
)
from scripts.report_s50_symmetry_confidence_refined_external import (  # noqa: E402
    _comparison,
    _stage,
)
from scripts.run_guidance_sdf_post_refinement import (  # noqa: E402
    _graph_signature,
    _load_pose_batch,
    _select_record,
    _tensor_sha256,
)
from scripts.run_guidance_sdf_post_refinement_posebusters_shard import (  # noqa: E402
    _validated_checks,
)
from scripts.run_s50_symmetry_confidence_refined_external_shard import (  # noqa: E402
    _records as _external_records,
)
from scripts.score_guidance_sdf_post_refinement_confidence import (  # noqa: E402
    _chunk_ranges,
    _load_refinement,
    _select_index,
)


def test_select_record_is_exact_and_eta_typed() -> None:
    manifest = {
        "records": [
            {"dataset": "posebusters", "id": "a", "eta": 0.0},
            {"dataset": "posebusters", "id": "a", "eta": 3.0},
        ]
    }
    assert _select_record(manifest, dataset="posebusters", eta=0.0, complex_id="A")[
        "eta"
    ] == 0.0
    with pytest.raises(ValueError, match="found 0"):
        _select_record(manifest, dataset="astex", eta=0.0, complex_id="a")


def test_graph_signature_detects_atom_order_and_connectivity() -> None:
    left = Chem.MolFromSmiles("CCO")
    same = Chem.MolFromSmiles("CCO")
    reordered = Chem.RenumberAtoms(left, [2, 1, 0])
    assert _graph_signature(left) == _graph_signature(same)
    assert _graph_signature(left) != _graph_signature(reordered)


def test_tensor_hash_binds_shape_dtype_and_values() -> None:
    value = torch.arange(12, dtype=torch.float32).view(2, 2, 3)
    assert _tensor_sha256(value) == _tensor_sha256(value.clone())
    assert _tensor_sha256(value) != _tensor_sha256(value + 1)
    assert _tensor_sha256(value) != _tensor_sha256(value.reshape(1, 4, 3))


def test_pose_batch_loader_reads_record_after_64k_boundary(tmp_path: Path) -> None:
    template = Chem.MolFromSmiles("CC")
    conformer = Chem.Conformer(template.GetNumAtoms())
    conformer.SetAtomPosition(0, (0.0, 0.0, 0.0))
    conformer.SetAtomPosition(1, (1.5, 0.0, 0.0))
    template.AddConformer(conformer)
    mol_block = Chem.MolToMolBlock(template)

    def record(index: int, *, padding: int = 0) -> str:
        properties = f">  <sample_index>\n{index}\n\n"
        if padding:
            properties += f">  <padding>\n{'x' * padding}\n\n"
        return f"{mol_block}{properties}$$$$\n"

    first_without_padding = record(0)
    padding_header_bytes = len(">  <padding>\n\n\n".encode())
    padding = 65_537 - len(first_without_padding.encode()) - padding_header_bytes
    assert padding > 0
    first = record(0, padding=padding)
    assert len(first.encode()) == 65_537
    path = tmp_path / "boundary.sdf"
    path.write_text(
        first + "".join(record(index) for index in range(1, 100)),
        encoding="utf-8",
    )

    coordinates, properties = _load_pose_batch(path, template)
    assert coordinates.shape == (100, template.GetNumAtoms(), 3)
    assert [row["sample_index"] for row in properties] == [
        str(index) for index in range(100)
    ]


def test_validate_pb_row_keeps_rmsd_separate_from_validity() -> None:
    raw = {key: True for key in VALIDITY_CHECKS}
    raw["rmsd_≤_2å"] = False
    rmsd_check, checks = validate_pb_row(raw, label="test")
    assert rmsd_check == "rmsd_≤_2å"
    assert all(checks[key] for key in VALIDITY_CHECKS)
    assert not checks[rmsd_check]


def test_pure_confidence_selection_is_minimum_predicted_rmsd() -> None:
    scores = [{"confidence_rmsd": 3.0} for _ in range(100)]
    scores[41]["confidence_rmsd"] = 0.5
    assert _select_index(scores) == 41


def test_pure_confidence_selection_uses_stable_index_tie_break() -> None:
    scores = [{"confidence_rmsd": 3.0} for _ in range(100)]
    scores[9]["confidence_rmsd"] = 0.5
    scores[4]["confidence_rmsd"] = 0.5
    assert _select_index(scores) == 4


def test_confidence_chunk_ranges_preserve_all_pose_indices() -> None:
    ranges = _chunk_ranges(100, 20)
    assert ranges == [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    assert [index for start, stop in ranges for index in range(start, stop)] == list(range(100))


def test_refinement_loader_rejects_failed_pose(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(
        '{"protocol_id":"EFFDOCK-GUIDANCE-SDF-POST-REFINEMENT-V1",'
        '"status":"complete_descriptive","counts":{"poses":100,"failed":1},'
        '"poses":[{"status":"nonfinite_energy"}]}'
    )
    with pytest.raises(ValueError, match="unusable terminal poses"):
        _load_refinement(path)


def test_refinement_loader_accepts_explicit_finite_line_search_terminal(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    poses = [{"status": "line_search_failed"}] + [{"status": "max_steps"}] * 99
    path.write_text(
        json.dumps(
            {
                "protocol_id": "EFFDOCK-GUIDANCE-SDF-POST-REFINEMENT-V1",
                "status": "complete_descriptive",
                "counts": {"poses": 100, "failed": 0, "line_search_failed": 1},
                "poses": poses,
            }
        )
    )
    assert _load_refinement(path)["counts"]["line_search_failed"] == 1


def test_refinement_loader_accepts_energy_plateau_terminal(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    poses = [{"status": "converged_energy_plateau"}] * 100
    path.write_text(
        json.dumps(
            {
                "protocol_id": "EFFDOCK-GUIDANCE-SDF-POST-REFINEMENT-V1",
                "status": "complete_descriptive",
                "counts": {"poses": 100, "failed": 0},
                "poses": poses,
            }
        )
    )
    assert len(_load_refinement(path)["poses"]) == 100


def test_pl_valid_excludes_only_cofactor_and_water_checks() -> None:
    assert len(VALIDITY_CHECKS) == 27
    assert len(PL_VALIDITY_CHECKS) == 21
    checks = {key: True for key in VALIDITY_CHECKS}
    for key in COFACTOR_AND_WATER_CHECKS:
        checks[key] = False
    assert is_pl_valid(checks)
    checks["minimum_distance_to_protein"] = False
    assert not is_pl_valid(checks)


def test_refined_pb_schema_keeps_rmsd_outside_validity() -> None:
    raw = {key: True for key in VALIDITY_CHECKS}
    raw["rmsd_≤_2å"] = False
    name, checks = _validated_checks(raw, "test")
    assert name == "rmsd_≤_2å"
    assert is_pl_valid(checks)
    assert not checks[name]


def test_stage_metrics_does_not_filter_confidence_selection() -> None:
    checks = [{key: True for key in VALIDITY_CHECKS} for _ in range(100)]
    checks[7]["minimum_distance_to_protein"] = False
    rmsds = [3.0] * 100
    rmsds[7] = 1.0
    metrics = _stage_metrics(rmsds, checks, selected_index=7)
    assert metrics["selected_index"] == 7
    assert metrics["selected_rmsd_lt2"]
    assert not metrics["selected_pl_valid"]
    assert not metrics["selected_joint"]
    assert metrics["pl_valid_oracle_lt2"] is False


def test_effect_decomposition_is_additive() -> None:
    row = _decomposition_row(
        dataset="astex",
        metric="selected_joint",
        unit="percentage_points",
        before=20.0,
        after_fixed=70.0,
        after_reselected=75.0,
    )
    assert row["refinement_contribution"] == 50.0
    assert row["reselection_contribution"] == 5.0
    assert row["total_change"] == 55.0
    assert row["total_change"] == (
        row["refinement_contribution"] + row["reselection_contribution"]
    )


def test_external_refined_records_are_exact_and_strided() -> None:
    records = [
        {"dataset": "astex", "id": f"a{i:03d}", "eta": 2.0} for i in range(85)
    ] + [
        {"dataset": "posebusters", "id": f"p{i:03d}", "eta": 2.0}
        for i in range(308)
    ]
    assigned = _external_records(
        {"records": records}, shard_index=0, num_shards=32, smoke=False
    )
    assert len(assigned) == 13
    assert assigned == sorted(records, key=lambda row: (row["dataset"], row["id"]))[::32]


def test_external_stage_uses_stable_confidence_order() -> None:
    rows = []
    for index in range(100):
        rows.append(
            {
                "before_confidence_rmsd": "2.0",
                "after_confidence_rmsd": "2.0",
                "initial_symmetry_rmsd_angstrom": "3.0",
                "final_symmetry_rmsd_angstrom": "3.0",
            }
        )
    rows[7]["after_confidence_rmsd"] = "0.5"
    rows[7]["final_symmetry_rmsd_angstrom"] = "1.0"
    metrics = _stage(rows, [True] * 100, stage="step_100")
    assert metrics["selected_index"] == 7
    assert metrics["selected_lt2"]
    assert metrics["selected_joint"]


def test_external_checkpoint_comparison_is_paired() -> None:
    rows = [
        {
            "dataset": "astex",
            "u001500_step_100_selected_lt2": False,
            "u025000_step_100_selected_lt2": True,
            "u001500_step_100_selected_official_valid": True,
            "u025000_step_100_selected_official_valid": True,
            "u001500_step_100_selected_joint": False,
            "u025000_step_100_selected_joint": True,
        },
        {
            "dataset": "astex",
            "u001500_step_100_selected_lt2": True,
            "u025000_step_100_selected_lt2": False,
            "u001500_step_100_selected_official_valid": True,
            "u025000_step_100_selected_official_valid": False,
            "u001500_step_100_selected_joint": True,
            "u025000_step_100_selected_joint": False,
        },
    ]
    result = _comparison(rows, baseline="u001500", candidate="u025000", dataset="astex")
    assert result["selected_lt2"]["false_to_true"] == 1
    assert result["selected_lt2"]["true_to_false"] == 1
    assert result["selected_lt2"]["delta_percentage_points"] == 0.0
