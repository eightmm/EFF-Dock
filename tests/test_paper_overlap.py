import pytest

from effdock.workflows.benchmark_inputs import canonical_heavy_smiles
from scripts.collect_paper_overlap import classify, executed_membership


def test_similarity_boundaries_and_exact_priority():
    assert classify(False, 0.499) == "no_observed_exact_<0.5"
    assert classify(False, 0.5) == "no_observed_exact_0.5_to_<0.8"
    assert classify(False, 0.799) == "no_observed_exact_0.5_to_<0.8"
    assert classify(False, 0.8) == "no_observed_exact_>=0.8"
    assert classify(True, 1) == "exact_identity"
    assert classify(False, None) == "missing"


def test_explicit_hydrogen_normalization():
    assert canonical_heavy_smiles("[H]OC") == canonical_heavy_smiles("CO")


def test_stereochemistry_retained_in_exact_identity():
    assert canonical_heavy_smiles("N[C@H](C)C(=O)O") != canonical_heavy_smiles("N[C@@H](C)C(=O)O")


def test_loader_membership_rejects_changed_split_before_reading_cache():
    with pytest.raises(ValueError, match="cache key drift"):
        executed_membership({"train": ["unrelated_system"]})
