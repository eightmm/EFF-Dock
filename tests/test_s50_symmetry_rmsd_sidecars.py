from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts import build_s50_symmetry_rmsd_sidecars as sidecars


def _write_json(path: Path, payload: dict) -> str:
    data = (json.dumps(payload, sort_keys=True) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _record(sample_key: str, split_index: int) -> tuple[dict, dict]:
    frozen = {
        "sample_key": sample_key,
        "system_id": f"system-{split_index}",
        "split": "train",
        "status": "eligible",
        "split_index": split_index,
        "global_index": split_index + 100,
        "sampling_seed": split_index + 42,
    }
    bank = {
        **frozen,
        "status": "complete",
        "pose_count": 100,
        "pt_path": f"/{sample_key}.pt",
        "pt_sha256": "a" * 64,
        "pose_ensemble_sha256": "b" * 64,
    }
    return frozen, bank


def test_load_contract_requires_exact_frozen_inventory(tmp_path: Path) -> None:
    frozen_a, bank_a = _record("a", 1)
    frozen_b, bank_b = _record("b", 2)
    input_path = tmp_path / "input.json"
    bank_path = tmp_path / "bank.json"
    input_sha = _write_json(
        input_path, {"status": "complete", "records": [frozen_b, frozen_a]}
    )
    bank_sha = _write_json(
        bank_path,
        {
            "status": "complete",
            "pose_tag": "s50_n100_s10_sig2_latep3_pc10_rdkitseed0",
            "records": [bank_b, bank_a],
        },
    )

    records, frozen, _ = sidecars._load_contract(
        input_path, input_sha, bank_path, bank_sha, split="train"
    )
    assert [record["sample_key"] for record in records] == ["a", "b"]
    assert set(frozen) == {"a", "b"}

    bank_b["sampling_seed"] += 1
    bank_sha = _write_json(
        bank_path,
        {
            "status": "complete",
            "pose_tag": "s50_n100_s10_sig2_latep3_pc10_rdkitseed0",
            "records": [bank_a, bank_b],
        },
    )
    with pytest.raises(sidecars.SidecarContractError, match="sampling_seed mismatch"):
        sidecars._load_contract(input_path, input_sha, bank_path, bank_sha, split="train")


def test_load_contract_accepts_explicit_refined_pose_tag(tmp_path: Path) -> None:
    frozen, bank = _record("refined", 1)
    input_path = tmp_path / "input.json"
    bank_path = tmp_path / "bank.json"
    input_sha = _write_json(input_path, {"status": "complete", "records": [frozen]})
    refined_tag = "s50_n100_s10_sig2_latep3_pc10_rdkitseed0_refine100"
    bank_sha = _write_json(
        bank_path,
        {"status": "complete", "pose_tag": refined_tag, "records": [bank]},
    )

    records, _, _ = sidecars._load_contract(
        input_path,
        input_sha,
        bank_path,
        bank_sha,
        split="train",
        expected_pose_tag=refined_tag,
    )
    assert [record["sample_key"] for record in records] == ["refined"]


def test_label_digest_and_atomic_publication_are_fail_closed(tmp_path: Path) -> None:
    labels = torch.arange(100, dtype=torch.float32)
    digest = sidecars._tensor_digest("sample", labels)
    assert digest == sidecars._tensor_digest("sample", labels.clone())
    assert digest != sidecars._tensor_digest("other", labels)

    output = tmp_path / "label.bin"
    sidecars._atomic_write_bytes(output, b"first")
    assert output.read_bytes() == b"first"
    with pytest.raises(sidecars.SidecarContractError, match="refusing to overwrite"):
        sidecars._atomic_write_bytes(output, b"second")
