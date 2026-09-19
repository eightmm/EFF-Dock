import pytest

from scripts.collect_paper_candidate_metrics import aggregate, candidate_case


def record():
    return dict(
        id="a",
        stage="raw",
        predicted_rmsd=[1.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        symmetry_rmsd=[2.0, 3.0, 3.0, 3.0, 3.0, 1.0],
        chirality_valid=[True] * 6,
        baseline=0,
        selected=0,
        fallback=False,
        valid_count=6,
        baseline_rmsd=2.0,
        selected_rmsd=2.0,
    )


def test_threshold_tie_and_oracle():
    r = candidate_case(record(), 6)
    assert not r["top1_lt2"] and not r["top5_lt2"] and r["oracle_lt2"]
    assert r["candidate_lt2"] == pytest.approx(1 / 6)
    assert aggregate([r], 1)["oracle_lt2"] == 100


def test_empty_chirality_fallback():
    r = record()
    r.update(chirality_valid=[False] * 6, valid_count=0, fallback=True)
    assert candidate_case(r, 6)["chirality_fallback"]


@pytest.mark.parametrize("change", ["nan", "length", "rank", "rmsd", "mask"])
def test_bad_record(change):
    r = record()
    if change == "nan":
        r["predicted_rmsd"][0] = float("nan")
    if change == "length":
        r["symmetry_rmsd"].pop()
    if change == "rank":
        r["baseline"] = 1
    if change == "rmsd":
        r["baseline_rmsd"] = 1
    if change == "mask":
        r["chirality_valid"][0] = 1
    with pytest.raises(ValueError):
        candidate_case(r, 6)


def test_duplicate_ids():
    r = candidate_case(record(), 6)
    with pytest.raises(ValueError):
        aggregate([r, r], 2)
