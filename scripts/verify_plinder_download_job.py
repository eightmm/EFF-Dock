#!/usr/bin/env python3
"""Bind a PLINDER raw download to exact live or completed Slurm metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from plinder_guidance_common import file_sha256, write_json_noreplace

EXPECTED_JOB_NAME = "effdock-plinder-guidance-data"
EXPECTED_PARTITION = "cpu_only"
EXPECTED_QOS = "long"
EXPECTED_CPUS = "8"
EXPECTED_MEMORY = "32G"
EXPECTED_TIMELIMIT = "3-00:00:00"
SACCT_FIELDS = (
    "JobIDRaw",
    "JobName",
    "State",
    "ExitCode",
    "Partition",
    "QOS",
    "AllocCPUS",
    "ReqCPUS",
    "ReqMem",
    "Timelimit",
    "SubmitLine",
    "WorkDir",
)
FAILED_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}


def _parse_record(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line.strip().split():
        if "=" not in token:
            raise ValueError(f"unexpected scontrol token: {token!r}")
        key, value = token.split("=", 1)
        if not key or key in fields:
            raise ValueError(f"duplicate or empty scontrol key: {key!r}")
        fields[key] = value
    return fields


def _parse_sacct_record(line: str) -> dict[str, str]:
    values = line.rstrip("\n").split("|")
    if len(values) != len(SACCT_FIELDS):
        raise ValueError(
            "sacct record field count mismatch: "
            f"expected={len(SACCT_FIELDS)} observed={len(values)}"
        )
    return dict(zip(SACCT_FIELDS, values, strict=True))


def _require_exact_fields(
    fields: dict[str, str], expected: dict[str, str], *, source: str
) -> None:
    for key, value in expected.items():
        if fields.get(key) != value:
            raise ValueError(
                f"download dependency {source}.{key} mismatch: expected={value!r} "
                f"observed={fields.get(key)!r}"
            )


def _expected_submit_line(repo_root: Path) -> str:
    return " ".join(
        (
            "sbatch",
            "--parsable",
            f"--partition={EXPECTED_PARTITION}",
            f"--qos={EXPECTED_QOS}",
            f"--cpus-per-task={EXPECTED_CPUS}",
            f"--mem={EXPECTED_MEMORY}",
            f"--time={EXPECTED_TIMELIMIT}",
            f"--export=ALL,EFFDOCK_REPO_DIR={repo_root}",
            "scripts/slurm/download_plinder_guidance_validation.sbatch",
        )
    )


def _verify_scontrol_record(
    line: str, *, job_id: str, repo_root: Path, download_script: Path
) -> dict[str, object]:
    fields = _parse_record(line)
    _require_exact_fields(
        fields,
        {
            "JobId": job_id,
            "JobName": EXPECTED_JOB_NAME,
            "Partition": EXPECTED_PARTITION,
            "QOS": EXPECTED_QOS,
            "BatchFlag": "1",
            "CPUs/Task": EXPECTED_CPUS,
            "MinMemoryNode": EXPECTED_MEMORY,
            "TimeLimit": EXPECTED_TIMELIMIT,
            "Command": str(download_script),
            "WorkDir": str(repo_root),
        },
        source="scontrol",
    )
    state = fields.get("JobState")
    if not state or any(state.startswith(failed) for failed in FAILED_STATES):
        raise ValueError(f"download dependency is not admissible: JobState={state!r}")
    if state == "COMPLETED" and fields.get("ExitCode") != "0:0":
        raise ValueError("completed download dependency has a nonzero exit code")
    return {
        "state": state,
        "exit_code": fields.get("ExitCode"),
        "command": fields["Command"],
        "cpus_per_task": fields["CPUs/Task"],
        "memory": fields["MinMemoryNode"],
        "scheduler_dependency_required": state != "COMPLETED",
    }


def _verify_sacct_record(
    line: str, *, job_id: str, repo_root: Path, download_script: Path
) -> dict[str, object]:
    fields = _parse_sacct_record(line)
    _require_exact_fields(
        fields,
        {
            "JobIDRaw": job_id,
            "JobName": EXPECTED_JOB_NAME,
            "State": "COMPLETED",
            "ExitCode": "0:0",
            "Partition": EXPECTED_PARTITION,
            "QOS": EXPECTED_QOS,
            "AllocCPUS": EXPECTED_CPUS,
            "ReqCPUS": EXPECTED_CPUS,
            "ReqMem": EXPECTED_MEMORY,
            "Timelimit": EXPECTED_TIMELIMIT,
            "SubmitLine": _expected_submit_line(repo_root),
            "WorkDir": str(repo_root),
        },
        source="sacct",
    )
    return {
        "state": fields["State"],
        "exit_code": fields["ExitCode"],
        "command": str(download_script),
        "cpus_per_task": fields["ReqCPUS"],
        "memory": fields["ReqMem"],
        "scheduler_dependency_required": False,
    }


def _run_scheduler_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def verify(job_id: str, *, repo_root: Path, output: Path) -> dict[str, object]:
    if re.fullmatch(r"[0-9]+", job_id) is None:
        raise ValueError("download job ID must be numeric")
    repo_root = repo_root.resolve()
    download_script = repo_root / "scripts/slurm/download_plinder_guidance_validation.sbatch"
    if not download_script.is_file():
        raise FileNotFoundError(download_script)
    scontrol = _run_scheduler_command(["scontrol", "show", "job", "-o", job_id])
    if scontrol.returncode == 0:
        lines = [line for line in scontrol.stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            raise ValueError(
                f"scontrol must return exactly one job record, got {len(lines)}"
            )
        line = lines[0]
        source = "scontrol"
        normalized = _verify_scontrol_record(
            line,
            job_id=job_id,
            repo_root=repo_root,
            download_script=download_script,
        )
    else:
        diagnostic = "\n".join((scontrol.stdout, scontrol.stderr))
        if "Invalid job id specified" not in diagnostic:
            raise RuntimeError(
                "scontrol failed without proving that the completed job was purged: "
                f"returncode={scontrol.returncode} stderr={scontrol.stderr.strip()!r}"
            )
        sacct = _run_scheduler_command(
            [
                "sacct",
                "-X",
                "-j",
                job_id,
                "-n",
                "-P",
                f"--format={','.join(SACCT_FIELDS)}",
            ]
        )
        if sacct.returncode != 0:
            raise RuntimeError(
                "sacct fallback failed: "
                f"returncode={sacct.returncode} stderr={sacct.stderr.strip()!r}"
            )
        lines = [line for line in sacct.stdout.splitlines() if line.strip()]
        if len(lines) != 1:
            raise ValueError(
                f"sacct must return exactly one allocation record, got {len(lines)}"
            )
        line = lines[0]
        source = "sacct"
        normalized = _verify_sacct_record(
            line,
            job_id=job_id,
            repo_root=repo_root,
            download_script=download_script,
        )

    result: dict[str, object] = {
        "schema_version": "effdock.plinder_download_slurm_binding.v2",
        "status": "verified",
        "job_id": job_id,
        "job_name": EXPECTED_JOB_NAME,
        "job_state_at_binding": normalized["state"],
        "partition": EXPECTED_PARTITION,
        "qos": EXPECTED_QOS,
        "command": normalized["command"],
        "work_dir": str(repo_root),
        "cpus_per_task": normalized["cpus_per_task"],
        "memory": normalized["memory"],
        "exit_code_at_binding": normalized["exit_code"],
        "download_script_sha256": file_sha256(download_script),
        "scheduler_record_source": source,
        "scheduler_dependency_required": normalized[
            "scheduler_dependency_required"
        ],
        "scheduler_record": line,
        "scheduler_record_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(),
    }
    result[f"{source}_record"] = line
    result[f"{source}_record_sha256"] = result["scheduler_record_sha256"]
    write_json_noreplace(output, result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = verify(args.job_id, repo_root=args.repo_root, output=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
