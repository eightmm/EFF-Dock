from __future__ import annotations

import subprocess
import sys
from importlib import import_module
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from plinder_guidance_audit import (  # noqa: E402
    _validate_cuda_runtime,
    inspect_incomplete_attempts,
)
from plinder_guidance_common import ids_sha256  # noqa: E402
from run_plinder_guidance_posebusters import (  # noqa: E402
    publish_posebusters_attempt,
    reserve_posebusters_attempt,
)

from effdock.workflows.evaluate import sorted_id_sha256  # noqa: E402


def _publish_locks(root: Path, count: int) -> None:
    root.mkdir()
    for shard in range(count):
        (root / f".shard-{shard:03d}-of-{count:03d}.publish.lock").touch()


def test_common_id_hash_matches_evaluator_contract() -> None:
    ids = ["complex-b", "complex-a"]
    assert ids_sha256(ids) == sorted_id_sha256(sorted(ids))
    with pytest.raises(ValueError, match="duplicates"):
        ids_sha256(["complex-a", "complex-a"])


@pytest.mark.parametrize(
    ("partition", "gpu_name"),
    (
        ("6000ada", "NVIDIA RTX 6000 Ada Generation"),
        ("heavy", "NVIDIA H100 80GB HBM3"),
        ("heavy", "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"),
    ),
)
def test_plinder_cuda_audit_accepts_declared_gpu_inventory(partition: str, gpu_name: str) -> None:
    runtime = {
        "device": "cuda",
        "slurm_partition": partition,
        "gpu": gpu_name,
        "gpu_total_memory_bytes": 48_000 * 1024**2,
        "cuda_max_memory_allocated_bytes": 1024,
    }
    assert _validate_cuda_runtime(runtime, label="test") is runtime


def test_plinder_cuda_audit_rejects_unknown_or_undersized_gpu() -> None:
    runtime = {
        "device": "cuda",
        "slurm_partition": "heavy",
        "gpu": "NVIDIA A5000",
        "gpu_total_memory_bytes": 48_000 * 1024**2,
        "cuda_max_memory_allocated_bytes": 1024,
    }
    with pytest.raises(ValueError, match="sampling GPU is not allowed"):
        _validate_cuda_runtime(runtime, label="test")
    runtime["gpu"] = "NVIDIA H100 80GB HBM3"
    runtime["slurm_partition"] = "test"
    with pytest.raises(ValueError, match="slurm_partition"):
        _validate_cuda_runtime(runtime, label="test")
    runtime["slurm_partition"] = "6000ada"
    with pytest.raises(ValueError, match="sampling GPU is not allowed"):
        _validate_cuda_runtime(runtime, label="test")
    runtime["slurm_partition"] = "heavy"
    runtime["gpu_total_memory_bytes"] = 47_999 * 1024**2
    with pytest.raises(ValueError, match="memory headroom"):
        _validate_cuda_runtime(runtime, label="test")


def test_incomplete_inventory_accepts_only_locks_and_named_stale_attempts(
    tmp_path: Path,
) -> None:
    incomplete = tmp_path / ".incomplete"
    _publish_locks(incomplete, 2)
    stale = incomplete / "shard-001-of-002.attempt-a1b2c3d4"
    stale.mkdir()
    (stale / "partial.txt").write_text("preserved failure\n")

    recovered = inspect_incomplete_attempts(incomplete, num_shards=2)
    assert [record["name"] for record in recovered] == [stale.name]
    assert recovered[0]["shard_index"] == 1
    assert recovered[0]["file_count"] == 1

    unexpected = incomplete / "unbound-output"
    unexpected.mkdir()
    with pytest.raises(ValueError, match="unexpected incomplete-attempt entry"):
        inspect_incomplete_attempts(incomplete, num_shards=2)


def test_incomplete_inventory_rejects_nonempty_publish_lock(tmp_path: Path) -> None:
    incomplete = tmp_path / ".incomplete"
    _publish_locks(incomplete, 1)
    lock = incomplete / ".shard-000-of-001.publish.lock"
    lock.write_text("not a lock-only sentinel")
    with pytest.raises(ValueError, match="zero-byte regular file"):
        inspect_incomplete_attempts(incomplete, num_shards=1)


def test_posebusters_attempt_is_hidden_until_atomic_publish(tmp_path: Path) -> None:
    attempt = reserve_posebusters_attempt(
        tmp_path, mode="smoke", eta=0.0, shard_index=0, num_shards=1
    )
    (attempt.attempt_dir / "results.csv").write_text("id\n")
    assert not attempt.final_dir.exists()
    published = publish_posebusters_attempt(attempt)
    assert published == attempt.final_dir
    assert (published / "results.csv").read_text() == "id\n"
    assert attempt.publish_lock.is_file()
    assert attempt.publish_lock.stat().st_size == 0


def _download_job_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    script = repo / "scripts/slurm/download_plinder_guidance_validation.sbatch"
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env bash\n")
    return repo.resolve(), script.resolve()


def _sacct_record(verifier, repo: Path, **overrides: str) -> str:
    fields = {
        "JobIDRaw": "49788",
        "JobName": "effdock-plinder-guidance-data",
        "State": "COMPLETED",
        "ExitCode": "0:0",
        "Partition": "cpu_only",
        "QOS": "long",
        "AllocCPUS": "8",
        "ReqCPUS": "8",
        "ReqMem": "32G",
        "Timelimit": "3-00:00:00",
        "SubmitLine": " ".join(
            (
                "sbatch",
                "--parsable",
                "--partition=cpu_only",
                "--qos=long",
                "--cpus-per-task=8",
                "--mem=32G",
                "--time=3-00:00:00",
                f"--export=ALL,EFFDOCK_REPO_DIR={repo}",
                "scripts/slurm/download_plinder_guidance_validation.sbatch",
            )
        ),
        "WorkDir": str(repo),
    }
    fields.update(overrides)
    return "|".join(fields[name] for name in verifier.SACCT_FIELDS)


def _purged_scontrol(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args,
        1,
        stdout="",
        stderr="slurm_load_jobs error: Invalid job id specified\n",
    )


def test_download_job_binding_checks_exact_live_slurm_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = import_module("verify_plinder_download_job")
    repo, script = _download_job_repo(tmp_path)
    record = " ".join(
        (
            "JobId=49788",
            "JobName=effdock-plinder-guidance-data",
            "JobState=RUNNING",
            "Partition=cpu_only",
            "QOS=long",
            "BatchFlag=1",
            "CPUs/Task=8",
            "MinMemoryNode=32G",
            "TimeLimit=3-00:00:00",
            f"Command={script}",
            f"WorkDir={repo}",
            "ExitCode=0:0",
        )
    )
    calls: list[list[str]] = []

    def scheduler(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout=record + "\n", stderr="")

    monkeypatch.setattr(verifier, "_run_scheduler_command", scheduler)
    output = tmp_path / "binding.json"
    result = verifier.verify("49788", repo_root=repo, output=output)
    assert result["status"] == "verified"
    assert result["job_state_at_binding"] == "RUNNING"
    assert result["scheduler_record_source"] == "scontrol"
    assert result["scheduler_dependency_required"] is True
    assert calls == [["scontrol", "show", "job", "-o", "49788"]]
    assert output.is_file()


def test_download_job_binding_accepts_strict_completed_sacct_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = import_module("verify_plinder_download_job")
    repo, _ = _download_job_repo(tmp_path)
    calls: list[list[str]] = []

    def scheduler(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[0] == "scontrol":
            return _purged_scontrol(args)
        return subprocess.CompletedProcess(
            args,
            0,
            stdout=_sacct_record(verifier, repo) + "\n",
            stderr="",
        )

    monkeypatch.setattr(verifier, "_run_scheduler_command", scheduler)
    output = tmp_path / "binding.json"
    result = verifier.verify("49788", repo_root=repo, output=output)

    assert result["status"] == "verified"
    assert result["job_state_at_binding"] == "COMPLETED"
    assert result["exit_code_at_binding"] == "0:0"
    assert result["scheduler_record_source"] == "sacct"
    assert result["scheduler_dependency_required"] is False
    assert calls[0] == ["scontrol", "show", "job", "-o", "49788"]
    assert calls[1][0:6] == ["sacct", "-X", "-j", "49788", "-n", "-P"]
    assert output.is_file()


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"JobName": "wrong-download"}, "sacct.JobName mismatch"),
        ({"State": "RUNNING"}, "sacct.State mismatch"),
        ({"ExitCode": "1:0"}, "sacct.ExitCode mismatch"),
        ({"ReqCPUS": "4"}, "sacct.ReqCPUS mismatch"),
        ({"ReqMem": "16G"}, "sacct.ReqMem mismatch"),
        ({"SubmitLine": "sbatch wrong.sbatch"}, "sacct.SubmitLine mismatch"),
    ),
)
def test_download_job_binding_rejects_inexact_completed_sacct_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override: dict[str, str],
    message: str,
) -> None:
    verifier = import_module("verify_plinder_download_job")
    repo, _ = _download_job_repo(tmp_path)

    def scheduler(args: list[str]) -> subprocess.CompletedProcess[str]:
        if args[0] == "scontrol":
            return _purged_scontrol(args)
        return subprocess.CompletedProcess(
            args,
            0,
            stdout=_sacct_record(verifier, repo, **override) + "\n",
            stderr="",
        )

    monkeypatch.setattr(verifier, "_run_scheduler_command", scheduler)
    with pytest.raises(ValueError, match=message):
        verifier.verify("49788", repo_root=repo, output=tmp_path / "binding.json")


def test_download_job_binding_does_not_fallback_on_scontrol_transport_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = import_module("verify_plinder_download_job")
    repo, _ = _download_job_repo(tmp_path)
    calls: list[list[str]] = []

    def scheduler(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(
            args,
            1,
            stdout="",
            stderr="slurm_load_jobs error: Unable to contact slurm controller\n",
        )

    monkeypatch.setattr(verifier, "_run_scheduler_command", scheduler)
    with pytest.raises(RuntimeError, match="without proving.*purged"):
        verifier.verify("49788", repo_root=repo, output=tmp_path / "binding.json")
    assert len(calls) == 1


def test_plinder_launcher_routes_completed_binding_without_old_afterok() -> None:
    launcher = (
        Path(__file__).resolve().parents[1] / "scripts/slurm/submit_plinder_guidance_validation.sh"
    ).read_text(encoding="utf-8")
    assert 'raw_gate_job=$(submit_job "$raw_gate_dependency"' in launcher
    assert 'raw_gate_job=$(submit_job "$raw_download_job"' not in launcher
    assert "raw_download_dependency_mode=completed_cache_reuse_no_afterok" in launcher
    assert '"recovery_provenance=$recovery_provenance"' in launcher
