import pytest

from benchmarks.analysis.evidence import (
    ranking_metrics,
    repeat_summary,
    selected_indices,
    stringent,
    transition,
)


def test_tied_classifier_scores_use_grouped_precision_and_midrank_auc():
    result = ranking_metrics([1, 1, 1, 1], [0.5, 1, 4, 5])
    assert result == dict(spearman=None, auroc=0.5, average_precision=0.5)
    perfect = ranking_metrics([0.1, 0.2, 3, 4], [0.5, 1, 4, 5])
    assert perfect["auroc"] == perfect["average_precision"] == 1


def test_single_class_is_explicitly_ineligible_and_two_angstrom_is_failure():
    assert ranking_metrics([1, 2], [2, 3])["auroc"] is None
    result = ranking_metrics([1, 2], [1.999, 2])
    assert result["auroc"] == result["average_precision"] == 1
    with pytest.raises(ValueError, match="Invalid scores"):
        ranking_metrics([1, float("nan")], [1, 3])


def test_chirality_fallback_preserves_original_selector_and_ties():
    record = dict(
        predicted_rmsd=[1.0] * 100,
        symmetry_rmsd=[3.0] * 100,
        chirality_valid=[False] * 100,
        baseline=0,
        selected=0,
        fallback=True,
    )
    assert selected_indices(record) == (0, 0)
    record.update(chirality_valid=[False, True] + [False] * 98, selected=1, fallback=False)
    assert selected_indices(record) == (0, 1)


def test_stringent_is_conjunction_with_strict_boundaries_and_no_imputation():
    ligand = dict(nearest_train_tanimoto=0.49, observed_train_ligand_identity=False)
    assert stringent(ligand, dict(max_sequence_identity=29.9))
    assert not stringent(ligand, dict(max_sequence_identity=30))
    assert not stringent(dict(ligand, nearest_train_tanimoto=0.5), dict(max_sequence_identity=0))
    assert not stringent(
        dict(ligand, observed_train_ligand_identity=True), dict(max_sequence_identity=0)
    )
    with pytest.raises(ValueError):
        stringent(dict(ligand, nearest_train_tanimoto=None), dict(max_sequence_identity=0))


def test_empty_repeat_cells_remain_missing_and_transitions_include_harm():
    assert repeat_summary([None, None, None])["mean"] is None
    assert transition([False, True, True, False], [True, False, True, False]) == dict(
        n=4, raw_fail=2, refined_fail=2, rescued=1, harmed=1
    )


def test_paired_bootstrap_preserves_group_weights():
    from benchmarks.analysis.baseline_uncertainty import paired_interval

    result = paired_interval([1.0, 1.0, -1.0], ["one", "one", "one"], draws=20)
    assert result["groups"] == 1
    assert result["complexes"] == 3
    assert result["delta_pp"] == pytest.approx(100 / 3)
    assert result["ci95_pp"] == pytest.approx([100 / 3, 100 / 3])
