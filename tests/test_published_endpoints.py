import pytest

from benchmarks.analysis.published_endpoints import conjunction, flag, rate, validate_top1


def test_unassigned_scores_preserve_endpoint_decidability():
    assert conjunction(None, False) is False
    assert conjunction(None, True) is None
    assert conjunction(True, True) is True
    assert flag("") is None
    with pytest.raises(ValueError):
        flag("nan")


def test_full_cohort_and_conditional_rates_differ():
    result = rate([True, False, None])
    assert result["rate_percent"] == pytest.approx(100 / 3)
    assert result["defined_case_rate_percent"] == 50
    assert result["unresolved_cases"] == 1
    assert rate([None])["defined_case_rate_percent"] is None


def test_topk_conjunction_cannot_combine_different_poses():
    row = {"dataset": "foldbench", "top_k": "5", "rmsd_success": "True",
           "lddt_pli_success": "True", "published_success": "False"}
    validate_top1(row)
    with pytest.raises(ValueError):
        validate_top1({**row, "top_k": "1"})
    validate_top1({**row, "top_k": "1", "rmsd_success": "", "lddt_pli_success": "False"})
