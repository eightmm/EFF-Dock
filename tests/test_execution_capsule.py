from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from create_execution_capsule import create_capsule  # noqa: E402


def _capsule_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    for directory in (".venv", "data", "weights", "outputs", "src/effdock"):
        (repo / directory).mkdir(parents=True, exist_ok=True)
    (repo / "src/effdock/runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    return repo


def test_execution_capsule_freezes_code_and_links_declared_large_roots(
    tmp_path: Path,
) -> None:
    repo = _capsule_repo(tmp_path)
    output = repo / ".effdock_execution_capsules/test/run"
    result = create_capsule(
        repo_root=repo,
        output=output,
        copy_files=["src/effdock/runtime.py"],
        link_roots=[".venv", "data", "weights", "outputs"],
    )

    copied = output / "src/effdock/runtime.py"
    assert copied.read_text(encoding="utf-8") == "VALUE = 1\n"
    (repo / "src/effdock/runtime.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert copied.read_text(encoding="utf-8") == "VALUE = 1\n"
    assert stat.S_IMODE(copied.stat().st_mode) == 0o444
    assert stat.S_IMODE((output / "src").stat().st_mode) == 0o555
    assert (output / "data").is_symlink()
    assert (output / "data").resolve() == (repo / "data").resolve()
    assert (output / "outputs").resolve() == (repo / "outputs").resolve()
    assert result["status"] == "frozen"
    assert (output / "execution_capsule.json").is_file()


def test_execution_capsule_rejects_copy_inside_linked_root(tmp_path: Path) -> None:
    repo = _capsule_repo(tmp_path)
    (repo / "data/input.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="already provided by a link root"):
        create_capsule(
            repo_root=repo,
            output=repo / ".effdock_execution_capsules/test/run",
            copy_files=["data/input.json"],
            link_roots=["data"],
        )


def test_execution_capsule_rejects_link_root_that_contains_output(
    tmp_path: Path,
) -> None:
    repo = _capsule_repo(tmp_path)
    (repo / ".effdock_execution_capsules").mkdir()
    with pytest.raises(ValueError, match="would contain the capsule output"):
        create_capsule(
            repo_root=repo,
            output=repo / ".effdock_execution_capsules/test/run",
            copy_files=["src/effdock/runtime.py"],
            link_roots=[".effdock_execution_capsules"],
        )


def test_execution_capsule_can_freeze_an_explicit_historical_input(
    tmp_path: Path,
) -> None:
    repo = _capsule_repo(tmp_path)
    (repo / "pyproject.toml").write_text("current\n", encoding="utf-8")
    (repo / "frozen_pyproject.toml").write_text("historical\n", encoding="utf-8")
    output = repo / ".effdock_execution_capsules/test/run"
    result = create_capsule(
        repo_root=repo,
        output=output,
        copy_files=["src/effdock/runtime.py"],
        link_roots=["data"],
        copy_overrides={"pyproject.toml": "frozen_pyproject.toml"},
    )

    assert (output / "pyproject.toml").read_text(encoding="utf-8") == "historical\n"
    assert result["copy_overrides"] == {
        "pyproject.toml": "frozen_pyproject.toml"
    }


def test_execution_capsule_rejects_override_destination_collision(tmp_path: Path) -> None:
    repo = _capsule_repo(tmp_path)
    with pytest.raises(ValueError, match="duplicate capsule destination"):
        create_capsule(
            repo_root=repo,
            output=repo / ".effdock_execution_capsules/test/run",
            copy_files=["src/effdock/runtime.py"],
            link_roots=[],
            copy_overrides={"src/effdock/runtime.py": "src/effdock/runtime.py"},
        )


def test_slurm_launchers_and_gpu_stages_use_frozen_capsule_provenance() -> None:
    root = Path(__file__).resolve().parents[1]
    launchers = (
        root / "scripts/slurm/submit_plinder_guidance_validation.sh",
        root / "scripts/slurm/submit_guidance_eta_sweep_confidence_standalone_pb.sh",
    )
    for launcher in launchers:
        text = launcher.read_text(encoding="utf-8")
        assert "scripts/create_execution_capsule.py" in text
        assert "EFFDOCK_REPO_DIR=$execution_root_abs" in text
        assert "PYTHONPATH=$execution_root_abs/src" in text
        assert "execution_capsule_identity_sha256" in text

    gpu_stages = (
        root / "scripts/slurm/plinder_guidance_sampling.sbatch",
        root / "scripts/slurm/guidance_eta_sweep_confidence_standalone_array.sbatch",
    )
    for stage in gpu_stages:
        text = stage.read_text(encoding="utf-8")
        assert "EFFDOCK_GIT_COMMIT" in text
        assert "EFFDOCK_GIT_DIFF_SHA256" in text
        assert "git rev-parse HEAD" not in text
        assert "git diff --no-ext-diff" not in text


def test_eta_gpu_stages_use_the_frozen_dual_partition_resource_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    stages = (
        root / "scripts/slurm/plinder_guidance_sampling.sbatch",
        root / "scripts/slurm/guidance_eta_sweep_confidence_standalone_array.sbatch",
    )
    for stage in stages:
        text = stage.read_text(encoding="utf-8")
        assert "#SBATCH --partition=6000ada,heavy" in text
        assert "6000ada|heavy)" in text
        assert '$(wc -l <<< "$gpu_inventory") -ne 1' in text
        assert '"$gpu_name" != *"RTX 6000 Ada"*' in text
        assert '"$gpu_name" != *"H100"*' in text
        assert '"$gpu_name" != *"RTX PRO 6000"*' in text
        assert '"$gpu_memory_mib" -lt 48000' in text
        assert "gpu_runtime partition=%s name=%s memory_mib=%s" in text

    launchers = {
        root / "scripts/slurm/submit_plinder_guidance_validation.sh": 2,
        root / "scripts/slurm/submit_guidance_eta_sweep_confidence_standalone_pb.sh": 3,
    }
    for launcher, expected_gpu_stages in launchers.items():
        text = launcher.read_text(encoding="utf-8")
        assert "gpu_partitions=6000ada,heavy" in text
        assert '"gpu_partitions=$gpu_partitions"' in text
        assert text.count('"$gpu_partitions"') == expected_gpu_stages


def test_gpu_summary_producers_record_the_actual_slurm_partition() -> None:
    root = Path(__file__).resolve().parents[1]
    producers = (
        root / "scripts/run_plinder_guidance_validation.py",
        root / "src/effdock/workflows/evaluate.py",
    )
    for producer in producers:
        text = producer.read_text(encoding="utf-8")
        assert '"slurm_partition": os.environ.get("SLURM_JOB_PARTITION")' in text
