import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_external_model_manifest_has_explicit_pipeline_components() -> None:
    manifest = json.loads((ROOT / "configs/external_models.json").read_text())
    required = {
        "id",
        "method",
        "site_information",
        "site_engine",
        "pose_engine",
        "scoring_engine",
        "refinement",
        "family",
    }
    arms = manifest["pipeline_arms"]

    assert len({arm["id"] for arm in arms}) == len(arms)
    assert all(required <= arm.keys() for arm in arms)
    assert {arm["site_information"] for arm in arms} <= {"pocket_supplied", "blind_site"}
    assert {arm["family"] for arm in arms} <= {"Classical", "Hybrid", "DL"}


def test_primary_comparison_is_supplied_pocket_only() -> None:
    manifest = json.loads((ROOT / "configs/external_models.json").read_text())

    assert manifest["comparison_scope"] == "supplied_pocket_only"

    primary = [arm for arm in manifest["pipeline_arms"] if arm.get("primary_comparison", True)]
    excluded = [arm for arm in manifest["pipeline_arms"] if not arm.get("primary_comparison", True)]

    assert primary
    assert all(arm["site_information"] == "pocket_supplied" for arm in primary)
    assert all(arm.get("exclusion_reason") for arm in excluded)


def test_public_u70k_figure_source_has_exact_full_denominator_rates() -> None:
    source = json.loads(
        (ROOT / "docs/results/external_models/effdock_u70k_benchmark.json").read_text()
    )

    assert source["comparison_scope"] == "supplied_pocket_only"
    for dataset_key, expected_n in (("astex", 85), ("posebusters", 308)):
        dataset = source["datasets"][dataset_key]
        assert dataset["n"] == expected_n
        for metric in (
            "raw_top1_lt2",
            "refined_top1_lt2",
            "refined_oracle_lt2",
            "refined_pb_valid",
            "refined_joint_lt2_pb_valid",
        ):
            row = dataset[metric]
            assert abs(row["pct"] - 100.0 * row["count"] / expected_n) < 1e-12


def test_literature_figure_source_keeps_excluded_methods_out_of_values() -> None:
    source = json.loads((ROOT / "docs/LITERATURE_RMSD_COMPARISON.json").read_text())

    assert source["comparison_scope"] == "supplied_pocket_only"
    for dataset_key in ("astex", "posebusters"):
        dataset = source[dataset_key]
        admitted = {row["method"] for row in dataset["values"]}
        excluded = {row["method"] for row in dataset["excluded_non_pocket"]}
        assert admitted.isdisjoint(excluded)


def test_public_executed_reruns_are_exact_three_repeat_aggregates() -> None:
    source = json.loads(
        (ROOT / "docs/results/external_models/pocket_only_executed_reruns.json").read_text()
    )

    assert source["comparison_scope"] == "supplied_pocket_only"
    expected_methods = {
        "SigmaDock",
        "DiffDock-Pocket",
        "RLDiff RL++",
        "DiffBindFR + MDN/EC",
        "Vina",
    }
    for dataset_key, expected_n in (("astex", 85), ("posebusters", 308)):
        dataset = source["datasets"][dataset_key]
        assert dataset["n"] == expected_n
        assert {row["method"] for row in dataset["methods"]} == expected_methods
        for row in dataset["methods"]:
            assert row["repeat_count"] == 3
            assert all(len(values) == 3 for values in row["coverage"].values())
            for metric_name in ("top1", "oracle"):
                metric = row[metric_name]
                assert len(metric["values"]) == 3
                assert metric["mean"] == statistics.mean(metric["values"])
                assert metric["std"] == statistics.stdev(metric["values"])


def test_external_model_family_is_derived_from_runtime_components() -> None:
    manifest = json.loads((ROOT / "configs/external_models.json").read_text())

    for arm in manifest["pipeline_arms"]:
        learned = (
            arm["site_engine"] in {"learned_pocket_prediction", "joint_learned_pose"}
            or arm["pose_engine"].startswith("learned_")
            or arm["scoring_engine"].startswith("learned_")
            or arm["refinement"] == "learned_refinement"
        )
        explicit_nonlearned_runtime = (
            arm["pose_engine"] == "classical_search"
            or arm["scoring_engine"] == "classical_energy"
            or arm["refinement"]
            in {"geometry_projection", "classical_minimization", "in_repo_physical_gradient"}
        )

        if arm["family"] == "Classical":
            assert not learned, arm["id"]
        elif arm["family"] == "DL":
            assert learned and not explicit_nonlearned_runtime, arm["id"]
        else:
            assert learned and explicit_nonlearned_runtime, arm["id"]


def test_external_model_sources_and_install_sets_are_pinned() -> None:
    manifest = json.loads((ROOT / "configs/external_models.json").read_text())
    default_targets = set(manifest["install_targets"])
    optional_targets = set(manifest["optional_install_targets"])

    assert default_targets.isdisjoint(optional_targets)
    assert "posebench-diffdock" not in default_targets
    assert "posebench-fabind" not in default_targets
    assert "posebench-dynamicbind" not in default_targets
    assert "posebench-flowdock" not in default_targets
    assert "posebench-diffdock" in optional_targets
    assert "posebench-fabind" in optional_targets
    assert "posebench-dynamicbind" in optional_targets
    assert "posebench-flowdock" in optional_targets
    for source in manifest["sources"].values():
        assert source["url"].startswith("https://")
        assert len(source["commit"]) == 40
        int(source["commit"], 16)


def test_cuda_build_compatibility_is_explicitly_pinned() -> None:
    manifest = json.loads((ROOT / "configs/external_models.json").read_text())
    override = manifest["environment_compatibility_overrides"]["posebench-diffdock"][0]

    assert set(override["packages"]) == {"gcc_linux-64", "gxx_linux-64"}
    assert override["version"] == "11"
    assert override["resolved_distribution"] == "conda-forge"
