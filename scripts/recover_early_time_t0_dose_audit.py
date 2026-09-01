#!/usr/bin/env python
"""Run the frozen paired-dose audit with scoped post-launch provenance checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

EXPECTED_PRODUCER_CODE_SHA256 = (
    "6b41a2f744b9f9678f870f4121160479ed559546306b134ca808fae40664e3cc"
)
EXPECTED_CURRENT_CODE_SHA256 = (
    "c372f0b1794f57a30bc429b9cffd48a90d4bc0f457800ac8aba637c2c89a2e31"
)
EXPECTED_DEPENDENCY_SHA256 = {
    "scripts/audit_early_time_t0_dose_10k.py": (
        "11df01d88d0dcf30d886c0d961cb8f2b644c74c9b4b55fbc0145dc4e6691041e"
    ),
    "scripts/audit_early_time_ft_50k.py": (
        "6ee4b5b94f196f20b62fff9d11e3b1a0f6cc3dbdb9c4fb939781c99b8d5208b9"
    ),
    "src/effdock/checkpoint.py": (
        "02e159681a1224205a9ce678fea816e7134f3c62f08cc7b0ee22ecb7f20a0ec1"
    ),
    "src/effdock/__init__.py": (
        "79ba578ff487840d906011acb308dc7506ec98f3d2e800700f3fc95a42f4a19e"
    ),
    "pyproject.toml": "2719e6215eb4ffbb7778e167a354d3f6294e7a6cb8698d44b055ea43edc90230",
    "uv.lock": "f767f067aa0101a57375cc3885d771c09d09efba3469d84d850da80c135224c4",
}
EXPECTED_ALLOWED_LATE_SOURCE_SHA256 = {
    "src/effdock/evaluation/fragment_geometry.py": (
        "4040b9922c8b09db4ae165b18792e62863c20e981bc3e8b69594d2f437cc13c9"
    ),
    "src/effdock/inference/preprocess.py": (
        "26e373eb01158eb104226eff6ea2264e502a0879d428dfbfb8b77556d100dee6"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_inventory_sha256() -> str:
    """Reproduce the registered sha256sum-of-sha256sum source identity."""
    digest = hashlib.sha256()
    paths = sorted(Path("src/effdock").rglob("*.py"), key=lambda path: path.as_posix())
    paths.extend((Path("pyproject.toml"), Path("uv.lock")))
    for path in paths:
        digest.update(f"{_sha256(path)}  {path.as_posix()}\n".encode("utf-8"))
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} does not contain a JSON object")
    return value


def _atomic_write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _verify_recovery_provenance(
    *,
    control_dir: Path,
    treatment_dir: Path,
    expected_producer_array_job_id: str,
    original_audit_job_id: str,
    original_failure_message: str,
) -> dict[str, Any]:
    dependency_hashes = {path: _sha256(Path(path)) for path in EXPECTED_DEPENDENCY_SHA256}
    if dependency_hashes != EXPECTED_DEPENDENCY_SHA256:
        raise RuntimeError("recovery audit dependency hash mismatch")

    identities = {
        "control": _read_json(control_dir / "run_identity.json"),
        "treatment": _read_json(treatment_dir / "run_identity.json"),
    }
    for branch, identity in identities.items():
        if identity.get("status") != "completed":
            raise RuntimeError(f"{branch} producer identity is not completed")
        if identity.get("producer_array_job_id") != expected_producer_array_job_id:
            raise RuntimeError(f"{branch} producer array identity mismatch")
        if identity.get("code_sha256") != EXPECTED_PRODUCER_CODE_SHA256:
            raise RuntimeError(f"{branch} frozen producer code hash mismatch")

    starts = {
        branch: datetime.fromisoformat(str(identity["started_at"]))
        for branch, identity in identities.items()
    }
    earliest_start_timestamp = min(value.timestamp() for value in starts.values())
    late_sources = {
        path.as_posix(): {
            "mtime": datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(),
            "sha256": _sha256(path),
        }
        for path in sorted(Path("src/effdock").rglob("*.py"), key=lambda item: item.as_posix())
        if path.stat().st_mtime > earliest_start_timestamp
    }
    late_source_hashes = {path: record["sha256"] for path, record in late_sources.items()}
    if late_source_hashes != EXPECTED_ALLOWED_LATE_SOURCE_SHA256:
        raise RuntimeError(
            "post-producer source inventory differs from the registered recovery scope: "
            f"{sorted(late_source_hashes)}"
        )

    current_code_sha256 = _source_inventory_sha256()
    if current_code_sha256 != EXPECTED_CURRENT_CODE_SHA256:
        raise RuntimeError("current live source hash differs from the frozen recovery identity")

    return {
        "original_audit": {
            "job_id": original_audit_job_id,
            "status": "failed_before_python_audit",
            "failure_message": original_failure_message,
        },
        "producer": {
            "array_job_id": expected_producer_array_job_id,
            "frozen_code_sha256": EXPECTED_PRODUCER_CODE_SHA256,
            "started_at": {branch: value.isoformat() for branch, value in starts.items()},
        },
        "recovery_scope": {
            "reason": (
                "replace only the over-broad aggregate live-tree freshness gate with exact "
                "hashes for the unchanged checkpoint-audit dependency closure"
            ),
            "current_live_code_sha256": current_code_sha256,
            "dependency_sha256": dependency_hashes,
            "post_producer_source_changes": late_sources,
            "post_producer_changes_are_outside_training_and_audit_import_paths": True,
            "claim_boundary": (
                "the original wrapper did not pass; the unchanged Python audit is rerun "
                "inside this separately identified recovery envelope"
            ),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--treatment-dir", type=Path, required=True)
    parser.add_argument("--control-config", type=Path, required=True)
    parser.add_argument("--treatment-config", type=Path, required=True)
    parser.add_argument("--common-init", type=Path, required=True)
    parser.add_argument("--expected-producer-array-job-id", required=True)
    parser.add_argument("--original-audit-job-id", required=True)
    parser.add_argument("--original-failure-message", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite recovery output: {args.output}")
    provenance = _verify_recovery_provenance(
        control_dir=args.control_dir,
        treatment_dir=args.treatment_dir,
        expected_producer_array_job_id=args.expected_producer_array_job_id,
        original_audit_job_id=args.original_audit_job_id,
        original_failure_message=args.original_failure_message,
    )
    if args.verify_only:
        _atomic_write_json(
            {"schema_version": 1, "status": "verified_only", "provenance": provenance},
            args.output,
        )
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, inner_name = tempfile.mkstemp(
        prefix="paired-dose-inner-audit-",
        suffix=".json",
        dir=args.output.parent,
    )
    os.close(descriptor)
    inner_path = Path(inner_name)
    command = [
        sys.executable,
        "scripts/audit_early_time_t0_dose_10k.py",
        "--control-dir",
        str(args.control_dir),
        "--treatment-dir",
        str(args.treatment_dir),
        "--control-config",
        str(args.control_config),
        "--treatment-config",
        str(args.treatment_config),
        "--common-init",
        str(args.common_init),
        "--expected-producer-array-job-id",
        args.expected_producer_array_job_id,
        "--output",
        str(inner_path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        audit_result = _read_json(inner_path)
        audit_result_sha256 = _sha256(inner_path)
        post_provenance = _verify_recovery_provenance(
            control_dir=args.control_dir,
            treatment_dir=args.treatment_dir,
            expected_producer_array_job_id=args.expected_producer_array_job_id,
            original_audit_job_id=args.original_audit_job_id,
            original_failure_message=args.original_failure_message,
        )
        if post_provenance != provenance:
            raise RuntimeError("recovery provenance changed while the audit was running")
        result = {
            "schema_version": 1,
            "status": "recovery_audit_passed",
            "provenance": provenance,
            "inner_audit_sha256": audit_result_sha256,
            "inner_audit_stdout_sha256": hashlib.sha256(
                completed.stdout.encode("utf-8")
            ).hexdigest(),
            "audit": audit_result,
        }
        _atomic_write_json(result, args.output)
    except subprocess.CalledProcessError as error:
        inner_result = _read_json(inner_path) if inner_path.stat().st_size else None
        _atomic_write_json(
            {
                "schema_version": 1,
                "status": "recovery_audit_failed",
                "provenance": provenance,
                "subprocess": {
                    "returncode": error.returncode,
                    "stdout": error.stdout,
                    "stderr": error.stderr,
                },
                "inner_audit": inner_result,
            },
            args.output,
        )
        print(error.stderr, file=sys.stderr, end="")
        return error.returncode
    finally:
        inner_path.unlink(missing_ok=True)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
