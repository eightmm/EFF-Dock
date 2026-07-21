from __future__ import annotations

import numpy as np
import pytest

from effdock.workflows.benchmark_data import strip_reference_ligand
from effdock.workflows.evaluate import ComplexInput, shard_complexes, summarize_rows


def _atom_line(record: str, serial: int, atom: str, residue: str, number: int, xyz) -> str:
    x, y, z = xyz
    return (
        f"{record:<6}{serial:5d} {atom:^4s} {residue:>3s} A{number:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C  \n"
    )


def test_strip_reference_ligand_handles_atom_encoded_peptide() -> None:
    ligand = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    pdb = "".join(
        [
            _atom_line("ATOM", 1, "C1", "LIG", 9, ligand[0]),
            _atom_line("ATOM", 2, "C2", "LIG", 9, ligand[1]),
            _atom_line("ATOM", 3, "C3", "LIG", 9, ligand[2]),
            _atom_line("ATOM", 4, "CA", "ALA", 10, [10.0, 0.0, 0.0]),
            "END\n",
        ]
    )
    cleaned, removed = strip_reference_ligand(pdb, ligand)
    assert removed == ["A:9:LIG"]
    assert "LIG" not in cleaned
    assert "ALA" in cleaned


def test_strip_reference_ligand_fails_without_coordinate_match() -> None:
    pdb = "".join(
        _atom_line("HETATM", i, f"C{i}", "LIG", 9, [10.0 + i, 0.0, 0.0]) for i in range(1, 4)
    )
    with pytest.raises(ValueError, match="no RCSB residue matched"):
        strip_reference_ligand(pdb, np.zeros((3, 3)))


def test_shard_complexes_is_disjoint_and_complete() -> None:
    complexes = [ComplexInput(str(i), None, None, "sdf", None, (0.0, 0.0, 0.0)) for i in range(9)]
    shards = [shard_complexes(complexes, index, 4) for index in range(4)]
    observed = [item.complex_id for shard in shards for item in shard]
    assert sorted(observed, key=int) == [str(i) for i in range(9)]
    assert len(observed) == len(set(observed))


def test_summarize_rows_keeps_selector_metrics_separate() -> None:
    rows = [
        {
            "first_rmsd": 3.0,
            "vina_rmsd": 1.5,
            "oracle_rmsd": 0.8,
            "confidence_rmsd": 1.2,
            "confidence_final_rmsd": 1.1,
            "first_fast_valid": False,
            "vina_fast_valid": True,
            "oracle_fast_valid": True,
            "confidence_fast_valid": True,
            "confidence_final_fast_valid": True,
        },
        {
            "first_rmsd": 1.0,
            "vina_rmsd": 2.5,
            "oracle_rmsd": 1.0,
            "confidence_rmsd": 2.2,
            "confidence_final_rmsd": 1.8,
            "first_fast_valid": True,
            "vina_fast_valid": False,
            "oracle_fast_valid": True,
            "confidence_fast_valid": False,
            "confidence_final_fast_valid": True,
        },
    ]
    stats = summarize_rows(rows)
    assert stats["first"]["pct_lt_2A"] == 50.0
    assert stats["vina"]["pct_lt_2A"] == 50.0
    assert stats["oracle"]["pct_lt_2A"] == 100.0
    assert stats["confidence"]["pct_lt_2A"] == 50.0
    assert stats["confidence_final"]["pct_lt_2A"] == 100.0
    assert stats["oracle"]["fast_valid_pct"] == 100.0
