#!/usr/bin/env bash
# Submit confidence replay -> sentinel identity gates -> official PB -> report.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

if [[ $# -ne 2 ]]; then
  echo "usage: $0 outputs/benchmarks/guidance_eta_sweep_v2_runs/RUN_ID PARENT_REPORT_JOB" >&2
  exit 2
fi
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is not available" >&2
  exit 2
fi

output_root=$1
parent_report_job=$2
output_prefix=outputs/benchmarks/guidance_eta_sweep_v2_runs/
frozen_output_root=${output_prefix}20260801T102903Z
frozen_parent_report_job=47341
run_component=${output_root#"$output_prefix"}
if [[ "$run_component" == "$output_root" || "$run_component" == */* ]] \
  || [[ ! "$run_component" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "output root must be $output_prefix followed by one safe run ID" >&2
  exit 2
fi
if [[ "$output_root" != "$frozen_output_root" ]]; then
  echo "confidence protocol is frozen to parent run: $frozen_output_root" >&2
  exit 2
fi
if [[ "$parent_report_job" != "$frozen_parent_report_job" ]]; then
  echo "parent report job must be frozen job $frozen_parent_report_job" >&2
  exit 2
fi
parent_job_info=$(scontrol show job "$parent_report_job" -o)
expected_parent_command="$repo_root/scripts/slurm/guidance_eta_sweep_parent_artifact_gate.sbatch"
if [[ "$parent_job_info" != *"JobName=effdock-eta-v2-parent-gate"* \
  || "$parent_job_info" != *"Command=$expected_parent_command"* \
  || "$parent_job_info" != *"WorkDir=$repo_root"* ]]; then
  echo "job $parent_report_job is not the frozen eta-v2 parent artifact gate" >&2
  exit 2
fi
if [[ ! -d "$output_root/raw" || ! -d "$output_root/smoke/raw" ]]; then
  echo "parent eta-sweep raw outputs are missing: $output_root" >&2
  exit 2
fi

extension_root="$output_root/confidence_selector_replay"
if ! mkdir "$extension_root"; then
  echo "refusing to reuse confidence extension output: $extension_root" >&2
  exit 2
fi
reservation_committed=0
submitted_jobs=()
cleanup_reservation() {
  if [[ "$reservation_committed" -eq 0 ]]; then
    for job_id in "${submitted_jobs[@]}"; do
      scancel "$job_id" 2>/dev/null || true
    done
    printf 'status=launcher_failed\ncancelled_jobs=%s\n' \
      "${submitted_jobs[*]:-none}" > "$extension_root/.submission.failed"
  fi
}
trap cleanup_reservation EXIT

mkdir -p outputs/benchmarks/logs
mapfile -t package_files < <(rg --files src/effdock | sort)
code_files=(
  "${package_files[@]}"
  pyproject.toml
  uv.lock
  configs/train.yaml
  docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json
  data/external_test/astex_reference_pocket_centers.json
  data/external_test/posebusters_reference_pocket_centers.json
  "$output_root/audit/combined.json"
  "$output_root/aggregate.json"
  "$output_root/posebusters_aggregate.json"
  "$output_root/recovery/parent-report-47280.json"
  "$output_root/recovery/parent_source_d726.manifest.sha256"
  scripts/slurm/guidance_eta_sweep_report_recovery.sbatch
  scripts/slurm/guidance_eta_sweep_parent_artifact_gate.sbatch
  scripts/slurm/guidance_eta_sweep_confidence_array.sbatch
  scripts/slurm/guidance_eta_sweep_confidence_identity.sbatch
  scripts/slurm/guidance_eta_sweep_confidence_posebusters_smoke.sbatch
  scripts/slurm/guidance_eta_sweep_confidence_posebusters_array.sbatch
  scripts/slurm/guidance_eta_sweep_confidence_report.sbatch
  scripts/slurm/submit_guidance_eta_sweep_confidence_pb.sh
)
code_manifest="$extension_root/execution_manifest.sha256"
sha256sum "${code_files[@]}" > "$code_manifest"
read -r code_manifest_sha256 _ < <(sha256sum "$code_manifest")
base_export="ALL,EFFDOCK_REPO_DIR=$repo_root,EFFDOCK_OUTPUT_ROOT=$output_root"
base_export+=",EFFDOCK_COHORT_MANIFEST=,EFFDOCK_CONFIDENCE_CODE_MANIFEST=$code_manifest"
base_export+=",EFFDOCK_CONFIDENCE_CODE_MANIFEST_SHA256=$code_manifest_sha256"

submit_job() {
  local dependency=$1
  local script=$2
  local export_spec=$3
  local array_spec=${4:-}
  local raw
  local args=(--parsable --export="$export_spec" --dependency="afterok:$dependency")
  if [[ -n "$array_spec" ]]; then
    args+=(--array="$array_spec")
  fi
  if ! raw=$(sbatch "${args[@]}" "$script"); then
    echo "failed to submit $script" >&2
    return 1
  fi
  local job_id=${raw%%;*}
  if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
    echo "invalid sbatch --parsable response for $script: $raw" >&2
    return 1
  fi
  printf '%s' "$job_id"
}

smoke_export="$base_export,EFFDOCK_CONFIDENCE_SMOKE_GRID=1"
full_export="$base_export,EFFDOCK_CONFIDENCE_SMOKE_GRID=0"
smoke_job=$(submit_job "$parent_report_job" \
  scripts/slurm/guidance_eta_sweep_confidence_array.sbatch "$smoke_export" "0-15%8")
submitted_jobs+=("$smoke_job")
smoke_identity_job=$(submit_job "$smoke_job" \
  scripts/slurm/guidance_eta_sweep_confidence_identity.sbatch "$smoke_export")
submitted_jobs+=("$smoke_identity_job")
posebusters_smoke_job=$(submit_job "$smoke_identity_job" \
  scripts/slurm/guidance_eta_sweep_confidence_posebusters_smoke.sbatch "$smoke_export")
submitted_jobs+=("$posebusters_smoke_job")
sampling_job=$(submit_job "$posebusters_smoke_job" \
  scripts/slurm/guidance_eta_sweep_confidence_array.sbatch "$full_export" "0-127%8")
submitted_jobs+=("$sampling_job")
identity_job=$(submit_job "$sampling_job" \
  scripts/slurm/guidance_eta_sweep_confidence_identity.sbatch "$full_export")
submitted_jobs+=("$identity_job")
posebusters_job=$(submit_job "$identity_job" \
  scripts/slurm/guidance_eta_sweep_confidence_posebusters_array.sbatch \
  "$full_export" "0-255%16")
submitted_jobs+=("$posebusters_job")
report_job=$(submit_job "$posebusters_job" \
  scripts/slurm/guidance_eta_sweep_confidence_report.sbatch "$full_export")
submitted_jobs+=("$report_job")

printf '%s\n' \
  'protocol_id=EFFDOCK-UNIFIED-GUIDANCE-ETA-SWEEP-CONFIDENCE-PB-V1' \
  "execution_manifest=$code_manifest" \
  "execution_manifest_sha256=$code_manifest_sha256" \
  "parent_report_job=$parent_report_job" \
  "smoke_job=$smoke_job" \
  "smoke_identity_job=$smoke_identity_job" \
  "posebusters_smoke_job=$posebusters_smoke_job" \
  "sampling_job=$sampling_job" \
  "identity_job=$identity_job" \
  "posebusters_job=$posebusters_job" \
  "report_job=$report_job" > "$extension_root/.submission"
reservation_committed=1
trap - EXIT

printf 'output_root=%s\n' "$output_root"
printf 'protocol_id=EFFDOCK-UNIFIED-GUIDANCE-ETA-SWEEP-CONFIDENCE-PB-V1\n'
printf 'smoke_job=%s dependency=afterok:%s tasks=16\n' "$smoke_job" "$parent_report_job"
printf 'smoke_identity_job=%s dependency=afterok:%s\n' "$smoke_identity_job" "$smoke_job"
printf 'posebusters_smoke_job=%s dependency=afterok:%s selectors=2\n' \
  "$posebusters_smoke_job" "$smoke_identity_job"
printf 'sampling_job=%s dependency=afterok:%s tasks=128\n' "$sampling_job" "$posebusters_smoke_job"
printf 'identity_job=%s dependency=afterok:%s\n' "$identity_job" "$sampling_job"
printf 'posebusters_job=%s dependency=afterok:%s tasks=256\n' "$posebusters_job" "$identity_job"
printf 'report_job=%s dependency=afterok:%s\n' "$report_job" "$posebusters_job"
printf 'monitor: squeue -j %s,%s,%s,%s,%s,%s,%s,%s\n' \
  "$parent_report_job" "$smoke_job" "$smoke_identity_job" \
  "$posebusters_smoke_job" "$sampling_job" "$identity_job" \
  "$posebusters_job" "$report_job"
