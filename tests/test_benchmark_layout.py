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


def test_legacy_benchmark_paths_are_symlink_aliases() -> None:
    aliases = {
        ROOT / "scripts" / "external_models": ROOT / "benchmarks" / "external_models",
        ROOT / "scripts" / "figures": ROOT / "benchmarks" / "figures",
        ROOT / "configs" / "external_models.json": (
            ROOT / "benchmarks" / "external_models" / "models.json"
        ),
    }

    for alias, canonical in aliases.items():
        assert alias.is_symlink(), alias
        assert alias.resolve() == canonical.resolve()


def test_external_slurm_compatibility_aliases_are_complete() -> None:
    canonical_dir = ROOT / "benchmarks" / "external_models" / "slurm"
    legacy_dir = ROOT / "scripts" / "slurm"
    canonical_names = {path.name for path in canonical_dir.iterdir() if path.is_file()}

    assert canonical_names
    for name in canonical_names:
        alias = legacy_dir / name
        assert alias.is_symlink(), alias
        assert alias.resolve() == (canonical_dir / name).resolve()
