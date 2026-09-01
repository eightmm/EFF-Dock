#!/usr/bin/env bash
# Submit fresh audit -> fixed-ID smoke -> full eta sweep -> official checks -> reports.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [outputs/benchmarks/guidance_eta_sweep_v2_runs/RUN_ID]" >&2
  exit 2
fi
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is not available" >&2
  exit 2
fi

run_id=$(date -u +%Y%m%dT%H%M%SZ)
output_prefix=outputs/benchmarks/guidance_eta_sweep_v2_runs/
output_root=${1:-$output_prefix$run_id}
run_component=${output_root#"$output_prefix"}
if [[ "$run_component" == "$output_root" || "$run_component" == */* ]] \
  || [[ ! "$run_component" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "output root must be $output_prefix followed by one safe run ID" >&2
  exit 2
fi
output_parent=${output_root%/*}
mkdir -p "$output_parent"
if ! mkdir "$output_root"; then
  echo "refusing to reuse an existing or concurrently created output root: $output_root" >&2
  exit 2
fi
mkdir -p outputs/benchmarks/logs

base_export="ALL,EFFDOCK_REPO_DIR=$repo_root,EFFDOCK_OUTPUT_ROOT=$output_root"
base_export+=",EFFDOCK_COHORT_MANIFEST=,EFFDOCK_ONLY_ID=,EFFDOCK_DATASET=,EFFDOCK_ETA="

submit_job() {
  local dependency=$1
  local script=$2
  local export_spec=$3
  local array_spec=${4:-}
  local raw
  local args=(--parsable --export="$export_spec")
  if [[ -n "$dependency" ]]; then
    args+=(--dependency="afterok:$dependency")
  fi
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

audit_export="$base_export,EFFDOCK_SMOKE_GRID=0"
smoke_export="$base_export,EFFDOCK_SMOKE_GRID=1"
full_export="$base_export,EFFDOCK_SMOKE_GRID=0"

audit_job=$(submit_job "" scripts/slurm/guidance_direct_drift_audit.sbatch "$audit_export")
merge_job=$(
  submit_job "$audit_job" scripts/slurm/guidance_direct_drift_merge_audit.sbatch \
    "$audit_export"
)
smoke_job=$(
  submit_job "$merge_job" scripts/slurm/guidance_eta_sweep_array.sbatch \
    "$smoke_export" "0-15%8"
)
sampling_job=$(
  submit_job "$smoke_job" scripts/slurm/guidance_eta_sweep_array.sbatch \
    "$full_export" "0-127%8"
)
posebusters_job=$(
  submit_job "$sampling_job" scripts/slurm/guidance_eta_sweep_posebusters_array.sbatch \
    "$full_export" "0-127%16"
)
report_job=$(
  submit_job "$posebusters_job" scripts/slurm/guidance_eta_sweep_report.sbatch \
    "$full_export"
)

printf 'output_root=%s\n' "$output_root"
printf 'audit_job=%s tasks=2\n' "$audit_job"
printf 'merge_job=%s dependency=afterok:%s\n' "$merge_job" "$audit_job"
printf 'smoke_job=%s dependency=afterok:%s tasks=16 mapping=2_datasets_x_8_eta\n' \
  "$smoke_job" "$merge_job"
printf 'sampling_job=%s dependency=afterok:%s tasks=128 mapping=2_datasets_x_8_eta_x_8_shards\n' \
  "$sampling_job" "$smoke_job"
printf 'posebusters_job=%s dependency=afterok:%s tasks=128\n' \
  "$posebusters_job" "$sampling_job"
printf 'report_job=%s dependency=afterok:%s\n' "$report_job" "$posebusters_job"
printf 'smoke IDs: astex=1jje posebusters=7b2c_tp7\n'
printf 'monitor: squeue -j %s,%s,%s,%s,%s,%s\n' \
  "$audit_job" "$merge_job" "$smoke_job" "$sampling_job" "$posebusters_job" "$report_job"
