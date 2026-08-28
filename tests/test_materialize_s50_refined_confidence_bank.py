from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts import materialize_s50_refined_confidence_bank as materialize
from scripts.materialize_s50_refined_confidence_bank import (
    REFINED_MANIFEST_SCHEMA,
    REFINED_PROTOCOL_ID,
    MaterializeError,
    _load_contracts,
)


def _write_json(path: Path, value: dict[str, object]) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_refined_materializer_accepts_current_v2_coordinate_contract(tmp_path: Path) -> None:
    raw = _write_json(
        tmp_path / "raw.json",
        {
            "protocol_id": "EFFDOCK-S50-CONFIDENCE-TRAINING-BANK-V1",
            "status": "complete",
            "records": [{"sample_key": "sample"}],
        },
    )
    refined = _write_json(
        tmp_path / "refined.json",
        {
            "schema_version": REFINED_MANIFEST_SCHEMA,
            "protocol_id": REFINED_PROTOCOL_ID,
            "status": "complete",
            "records": [{"sample_key": "sample"}],
        },
    )
    frozen = _write_json(
        tmp_path / "frozen.json",
        {"records": [{"sample_key": "sample", "status": "eligible"}]},
    )

    loaded, refined_by_id, frozen_by_id = _load_contracts(raw, refined, frozen)

    assert loaded["status"] == "complete"
    assert set(refined_by_id) == {"sample"}
    assert set(frozen_by_id) == {"sample"}


def test_refined_materializer_rejects_superseded_v1_manifest(tmp_path: Path) -> None:
    raw = _write_json(
        tmp_path / "raw.json",
        {
            "protocol_id": "EFFDOCK-S50-CONFIDENCE-TRAINING-BANK-V1",
            "status": "complete",
            "records": [{"sample_key": "sample"}],
        },
    )
    refined = _write_json(
        tmp_path / "refined.json",
        {
            "schema_version": "effdock.s50_refined_pose_bank.manifest.v1",
            "protocol_id": "EFFDOCK-S50-REFINED-POSE-BANK-V1",
            "status": "complete",
            "records": [{"sample_key": "sample"}],
        },
    )
    frozen = _write_json(
        tmp_path / "frozen.json",
        {"records": [{"sample_key": "sample", "status": "eligible"}]},
    )

    with pytest.raises(MaterializeError, match="refined coordinate bank is incomplete"):
        _load_contracts(raw, refined, frozen)


def test_materializer_adds_exact_crystal_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_path = tmp_path / "raw.pt"
    refined_path = tmp_path / "refined.pt"
    poses = torch.ones(100, 2, 3)
    raw_payload = {
        "pid": "sample",
        "system_id": "system",
        "pose_atom_coords": torch.zeros_like(poses),
    }
    torch.save(raw_payload, raw_path)
    torch.save(
        {
            "pose_atom_coords_refined": poses,
            "source_pt_sha256": materialize.file_sha256(raw_path),
            "refined_pose_ensemble_sha256": "e" * 64,
        },
        refined_path,
    )
    graph = {"node_coords": torch.zeros(1, 3)}
    monkeypatch.setattr(
        materialize,
        "_prepare_runtime_input",
        lambda *_args: (object(), object(), graph, object(), {"pocket_center": torch.zeros(3)}, object()),
    )
    monkeypatch.setattr(
        materialize,
        "_extract_hidden_chunked",
        lambda _model, _graph, _ligand, _meta, values, **_kwargs: {
            "h_lig_node": torch.zeros(values.shape[0], 3, 8),
            "lig_node_type": torch.zeros(3, dtype=torch.long),
        },
    )
    monkeypatch.setattr(
        materialize,
        "_fixed_labels",
        lambda values, *_args: (
            torch.zeros(2, 3),
            torch.ones(values.shape[0], 2),
            torch.ones(values.shape[0]),
        ),
    )
    monkeypatch.setattr(materialize, "_center_graph", lambda value, _center: value)

    payload = materialize._materialize_one(
        {
            "sample_key": "sample",
            "split": "train",
            "pt_path": str(raw_path),
            "pt_sha256": materialize.file_sha256(raw_path),
        },
        {
            "pt_path": str(refined_path),
            "pt_sha256": materialize.file_sha256(refined_path),
            "refined_pose_ensemble_sha256": "e" * 64,
        },
        {"input_to_reference": [0, 1]},
        model=torch.nn.Identity(),
        device=torch.device("cpu"),
        processed_root=tmp_path,
        checkpoint=tmp_path / "checkpoint.pt",
        checkpoint_sha256="a" * 64,
        config=tmp_path / "config.yaml",
        config_sha256="b" * 64,
    )

    assert payload["crystal_anchor_pose_atom_coords"].shape == (1, 2, 3)
    assert payload["crystal_anchor_h_lig_node"].shape == (1, 3, 8)
    assert payload["crystal_anchor_atom_disp"].shape == (1, 2)
    assert payload["crystal_anchor_pose_rmsd"].tolist() == [0.0]
    assert torch.equal(
        payload["crystal_anchor_pose_atom_coords"][0],
        payload["lig_atom_coords_crystal_centered"],
    )
    assert payload["crystal_anchor_rmsd_method"] == "exact_mapped_reference_zero"
