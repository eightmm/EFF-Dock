from __future__ import annotations

import json
from pathlib import Path

from effdock.workflows.benchmark_inputs import load_benchmark_inputs
from effdock.workflows.external_benchmark_data import (
    PhiCandidate,
    _ccd_from_cif,
    select_foldbench_postcut,
    select_openbind_clean,
    sequence_diverse_representatives,
)


def test_ccd_cif_fallback_uses_heavy_atom_order(tmp_path: Path) -> None:
    path = tmp_path / "TST.cif"
    path.write_text(
        """data_TST
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
_chem_comp_atom.pdbx_aromatic_flag
C1 C 0 N
O1 O 0 N
H1 H 0 N
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
C1 O1 SING N
C1 H1 SING N
loop_
_pdbx_chem_comp_descriptor.comp_id
_pdbx_chem_comp_descriptor.type
_pdbx_chem_comp_descriptor.program
_pdbx_chem_comp_descriptor.program_version
_pdbx_chem_comp_descriptor.descriptor
TST SMILES_CANONICAL CACTVS 1.0 CO
"""
    )
    record = _ccd_from_cif(path)
    assert record["ref_atom_name_chars"] == ["C1", "O1"]
    assert record["ref_mol"].GetNumAtoms() == 2
    assert record["ref_mol"].GetNumBonds() == 1
    assert record["canonical_smiles"] == "CO"


def test_openbind_clean_cohort_requires_all_four_filters() -> None:
    base = {
        "covalent": "False",
        "pb_valid_prepared": "True",
        "pb_valid_ref": "True",
        "suspected_artefact": "False",
    }
    rows = [
        {"id": "keep", **base},
        {"id": "covalent", **base, "covalent": "True"},
        {"id": "bad-prepared", **base, "pb_valid_prepared": "False"},
        {"id": "bad-ref", **base, "pb_valid_ref": "False"},
        {"id": "artefact", **base, "suspected_artefact": "True"},
    ]
    assert [row["id"] for row in select_openbind_clean(rows)] == ["keep"]


def test_foldbench_postcut_is_strict() -> None:
    rows = [
        {"pdb_id": "before-assembly1"},
        {"pdb_id": "cutoff-assembly1"},
        {"pdb_id": "after-assembly1"},
    ]
    dates = {
        "before": {"initial_release_date": "2024-06-29T00:00:00Z"},
        "cutoff": {"initial_release_date": "2024-06-30T00:00:00Z"},
        "after": {"initial_release_date": "2024-07-01T00:00:00Z"},
    }
    assert select_foldbench_postcut(rows, dates) == [rows[-1]]


def test_phibench_sequence_components_are_deterministic() -> None:
    candidates = [
        PhiCandidate("z.pkl.gz", "z", "9zzz", "AAAA"),
        PhiCandidate("a.pkl.gz", "a", "9aaa", "AAAA"),
        PhiCandidate("b.pkl.gz", "b", "9bbb", "TTTT"),
    ]
    selected, audit = sequence_diverse_representatives(candidates, threshold=0.995)
    assert [item.complex_id for item in selected] == ["a", "b"]
    assert audit["source_candidates"] == 3
    assert audit["exact_sequence_representatives"] == 2
    assert audit["selected_components"] == 2


def test_external_legacy_mappings_are_loadable(tmp_path: Path) -> None:
    for dataset in ("phibench", "foldbench", "openbind"):
        (tmp_path / f"{dataset}_smiles.json").write_text(json.dumps({"one": "CCO"}))
        mapping, identity = load_benchmark_inputs(dataset, tmp_path)
        assert mapping == {"one": "CCO"}
        assert identity["dataset"] == dataset
        assert identity["count"] == 1
