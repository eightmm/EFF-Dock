#!/usr/bin/env bash
#SBATCH --job-name=effdock-eta-v2-report-recover
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"

output_root=${EFFDOCK_OUTPUT_ROOT:?EFFDOCK_OUTPUT_ROOT is required}
source_snapshot=${EFFDOCK_PARENT_SOURCE_SNAPSHOT:?EFFDOCK_PARENT_SOURCE_SNAPSHOT is required}
source_manifest=${EFFDOCK_PARENT_SOURCE_MANIFEST:?EFFDOCK_PARENT_SOURCE_MANIFEST is required}
expected_manifest_sha256=${EFFDOCK_PARENT_SOURCE_MANIFEST_SHA256:?EFFDOCK_PARENT_SOURCE_MANIFEST_SHA256 is required}
cohort_audit=${EFFDOCK_COHORT_MANIFEST:-$output_root/audit/combined.json}
expected_audit_sha256=dac7903488ccd36552a9bca134e37e633e3f07166d94f0389837012081ff3048
expected_implementation_sha256=d726ddc4cb89b495f0495aa059faf9efdf33ee76c42c4b14e71356068935c0a5

source_snapshot=$(readlink -f "$source_snapshot")
source_manifest=$(readlink -f "$source_manifest")
for required in "$source_snapshot/src/effdock" "$source_snapshot/pyproject.toml" \
  "$source_snapshot/uv.lock" "$source_manifest" "$cohort_audit" \
  "$output_root/raw" "$output_root/posebusters_official"; do
  if [[ ! -e "$required" ]]; then
    echo "missing parent-report recovery input: $required" >&2
    exit 2
  fi
done

read -r observed_manifest_sha256 _ < <(sha256sum "$source_manifest")
if [[ "$observed_manifest_sha256" != "$expected_manifest_sha256" ]]; then
  echo "historical source manifest SHA-256 mismatch" >&2
  exit 2
fi
if ! (cd "$source_snapshot" && sha256sum --check --quiet "$source_manifest"); then
  echo "historical source snapshot verification failed" >&2
  exit 2
fi
read -r observed_audit_sha256 _ < <(sha256sum "$cohort_audit")
if [[ "$observed_audit_sha256" != "$expected_audit_sha256" ]]; then
  echo "frozen parent audit SHA-256 mismatch" >&2
  exit 2
fi

snapshot_pythonpath="$source_snapshot/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
PYTHONPATH="$snapshot_pythonpath" .venv/bin/python -c \
  'import json, sys; from pathlib import Path; import effdock; from effdock.guidance.provenance import guidance_implementation_identity; root=Path(sys.argv[1]).resolve(); imported=Path(effdock.__file__).resolve(); expected=sys.argv[2]; identity=guidance_implementation_identity(); assert imported.is_relative_to(root / "src"), (imported, root); assert identity["sha256"] == expected, identity["sha256"]; print(json.dumps({"effdock_import": str(imported), "guidance_implementation": identity}, sort_keys=True))' \
  "$source_snapshot" "$expected_implementation_sha256"

job_id=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
recovery_root="$output_root/recovery"
temporary_root="$recovery_root/parent-report-$job_id.tmp"
aggregate_output="$output_root/aggregate.json"
posebusters_output="$output_root/posebusters_aggregate.json"
provenance_output="$recovery_root/parent-report-$job_id.json"
for final_output in "$aggregate_output" "$posebusters_output" "$provenance_output"; do
  if [[ -e "$final_output" ]]; then
    echo "refusing to overwrite parent-report recovery output: $final_output" >&2
    exit 2
  fi
done
mkdir -p "$recovery_root"
mkdir "$temporary_root"

PYTHONPATH="$snapshot_pythonpath" .venv/bin/python -m \
  effdock.workflows.guidance_eta_sweep_report \
  --input-dir "$output_root/raw" \
  --cohort-audit "$cohort_audit" \
  --output "$temporary_root/aggregate.json"
PYTHONPATH="$snapshot_pythonpath" .venv/bin/python -m \
  effdock.workflows.guidance_eta_sweep_posebusters_report \
  --input-dir "$output_root/posebusters_official" \
  --sampling-dir "$output_root/raw" \
  --cohort-audit "$cohort_audit" \
  --output "$temporary_root/posebusters_aggregate.json"

.venv/bin/python -c \
  'import hashlib, json, sys; from pathlib import Path
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
aggregate_path=Path(sys.argv[1]); pb_path=Path(sys.argv[2]); provenance_path=Path(sys.argv[3]); source_snapshot=Path(sys.argv[4]); source_manifest=Path(sys.argv[5]); manifest_sha=sys.argv[6]; audit_path=Path(sys.argv[7]); implementation_sha=sys.argv[8]; job_id=sys.argv[9]; script_path=Path(sys.argv[10])
aggregate=json.loads(aggregate_path.read_text()); pb=json.loads(pb_path.read_text())
assert aggregate["protocol_id"] == "EFFDOCK-UNIFIED-GUIDANCE-ETA-SWEEP-V2"
assert aggregate["status"] == "complete_strict_full_cohort_paired_descriptive_eta_sweep"
assert aggregate["sampling_inventory"] == {"run_cells": 16, "shards_per_cell": 8, "total_shard_tasks": 128}
assert aggregate["guidance_implementation"]["sha256"] == implementation_sha
assert aggregate["datasets"]["astex"]["coverage"]["count"] == 85
assert aggregate["datasets"]["posebusters"]["coverage"]["count"] == 308
assert pb["protocol_id"] == "EFFDOCK-UNIFIED-GUIDANCE-ETA-SWEEP-V2"
assert pb["status"] == "complete_strict_full_cohort_paired_official_posebusters_eta_sweep"
assert pb["official_inventory"] == {"run_cells": 16, "shards_per_cell": 8, "total_shard_tasks": 128}
assert pb["datasets"]["astex"]["coverage"]["count"] == 85
assert pb["datasets"]["posebusters"]["coverage"]["count"] == 308
record={"schema_version":"effdock.parent_report_recovery.v1","status":"passed","slurm_job_id":job_id,"source_snapshot":str(source_snapshot),"source_manifest":str(source_manifest),"source_manifest_sha256":manifest_sha,"guidance_implementation_sha256":implementation_sha,"cohort_audit":str(audit_path),"cohort_audit_sha256":digest(audit_path),"recovery_script":str(script_path),"recovery_script_sha256":digest(script_path),"aggregate_sha256":digest(aggregate_path),"posebusters_aggregate_sha256":digest(pb_path)}
provenance_path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False)+"\n")' \
  "$temporary_root/aggregate.json" \
  "$temporary_root/posebusters_aggregate.json" \
  "$temporary_root/provenance.json" \
  "$source_snapshot" "$source_manifest" "$expected_manifest_sha256" \
  "$cohort_audit" "$expected_implementation_sha256" "$job_id" \
  "$repo_dir/scripts/slurm/guidance_eta_sweep_report_recovery.sbatch"

mv "$temporary_root/aggregate.json" "$aggregate_output"
mv "$temporary_root/posebusters_aggregate.json" "$posebusters_output"
mv "$temporary_root/provenance.json" "$provenance_output"
rmdir "$temporary_root"
