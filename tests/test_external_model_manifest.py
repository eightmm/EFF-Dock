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
        (ROOT / "benchmarks/results/external_models/effdock_u70k_benchmark.json").read_text()
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


def test_public_foldbench_558_ledger_has_exact_rates_and_claim_boundary() -> None:
    source = json.loads(
        (ROOT / "benchmarks/results/external_models/foldbench_pocket_558.json").read_text()
    )

    assert source["comparison_scope"] == "holo_receptor_crystal_pocket_redocking"
    assert source["directly_comparable_to_foldbench_source_native"] is False
    for dataset in source["slices"].values():
        denominator = dataset["n"]
        for metric in (
            "raw_top1_lt2",
            "refined_top1_lt2",
            "refined_oracle_lt2",
            "refined_pb_valid",
            "refined_joint_lt2_pb_valid",
        ):
            row = dataset[metric]
            assert abs(row["pct"] - 100.0 * row["count"] / denominator) < 1e-12


def test_public_phibench_top5_ledger_has_exact_rates_and_claim_boundary() -> None:
    source = json.loads(
        (ROOT / "benchmarks/results/external_models/phibench_u70k_top5.json").read_text()
    )

    assert source["comparison_scope"] == "holo_receptor_crystal_pocket_redocking"
    assert source["directly_comparable_to_phibench_source_native"] is False
    results = source["results"]
    assert results["n"] == 203
    for metric, row in results.items():
        if metric == "n":
            continue
        assert abs(row["pct"] - 100.0 * row["count"] / results["n"]) < 1e-12
    assert results["refined_top1_lt2"]["count"] == 131
    assert results["refined_top5_lt2"]["count"] == 156
    assert results["refined_top1_joint_lt2_pb_valid"]["count"] == 120
    assert results["refined_top5_joint_lt2_pb_valid"]["count"] == 150


def test_temporal_comparison_views_are_explicitly_non_rankable() -> None:
    source = json.loads(
        (ROOT / "benchmarks/results/external_models/temporal_literature.json").read_text()
    )

    comparison = source["comparison_views"]
    assert comparison["direct_ranking_permitted"] is False
    assert comparison["foldbench_protocol_matrix"]["directly_comparable"] is False
    assert all(
        row["directly_comparable"] is False
        for row in comparison["phibench_pocket_guided"]
    )

    public = json.loads(
        (ROOT / "benchmarks/results/external_models/foldbench_pocket_558.json").read_text()
    )
    matrix = comparison["foldbench_protocol_matrix"]
    assert matrix["effdock_panel"]["endpoints"][
        "refined_symmetry_lrmsd_lt_2_pct"
    ] == public["slices"]["all_558"]["refined_top1_lt2"]["pct"]

    phibench_public = json.loads(
        (ROOT / "benchmarks/results/external_models/phibench_u70k_top5.json").read_text()
    )
    top5 = next(
        row
        for row in comparison["phibench_pocket_guided"]
        if row["method"] == "EFF-Dock U70k"
        and row["endpoint"] == "refined confidence Top-5"
    )
    assert top5["rmsd_lt_2_pct"] == phibench_public["results"]["refined_top5_lt2"][
        "pct"
    ]
    assert top5["joint_pb_valid_pct"] == phibench_public["results"][
        "refined_top5_joint_lt2_pb_valid"
    ]["pct"]

    reported = {
        row["model"]: row["success_pct"]
        for row in source["foldbench_source_native"]["after_2023_01_full"]
    }
    assert reported == {
        "AlphaFold 3": 64.9,
        "Boltz-1": 55.04,
        "Chai-1": 51.23,
        "HelixFold 3": 51.82,
        "Protenix": 50.7,
        "OpenFold 3 preview": 44.49,
    }


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
        (ROOT / "benchmarks/results/external_models/pocket_only_executed_reruns.json").read_text()
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
