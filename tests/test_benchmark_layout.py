from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_code_has_one_canonical_home() -> None:
    benchmark_root = ROOT / "benchmarks"
    external_root = benchmark_root / "external_models"

    assert (benchmark_root / "README.md").is_file()
    assert (external_root / "README.md").is_file()
    assert (external_root / "models.json").is_file()
    assert (external_root / "slurm").is_dir()
    assert (benchmark_root / "figures" / "README.md").is_file()
    assert (benchmark_root / "results" / "README.md").is_file()


def test_external_model_config_resolves_to_manifest() -> None:
    alias = ROOT / "configs" / "external_models.json"
    assert alias.is_symlink()
    assert alias.resolve() == ROOT / "benchmarks" / "external_models" / "models.json"
