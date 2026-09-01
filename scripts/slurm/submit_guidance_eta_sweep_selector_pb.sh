#!/usr/bin/env bash
# Submit the additive selector-PoseBusters extension after the parent report.

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
run_component=${output_root#"$output_prefix"}
if [[ "$run_component" == "$output_root" || "$run_component" == */* ]] \
  || [[ ! "$run_component" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "output root must be $output_prefix followed by one safe run ID" >&2
  exit 2
fi
if [[ "$output_root" != "$frozen_output_root" ]]; then
  echo "selector protocol is frozen to parent run: $frozen_output_root" >&2
  exit 2
fi
if [[ ! -d "$output_root" || ! -d "$output_root/raw" ]]; then
  echo "parent eta-sweep output root is incomplete: $output_root" >&2
  exit 2
fi
for output_path in \
  "$output_root/posebusters_official_selectors" \
  "$output_root/posebusters_first_aggregate.json" \
  "$output_root/posebusters_vina_aggregate.json" \
  "$output_root/posebusters_selector_comparison.json"; do
  if [[ -e "$output_path" ]]; then
    echo "refusing to reuse selector-extension output: $output_path" >&2
    exit 2
  fi
done
reservation_dir="$output_root/posebusters_official_selectors"
if ! mkdir "$reservation_dir"; then
  echo "failed to reserve selector-extension output: $reservation_dir" >&2
  exit 2
fi
reservation_committed=0
cleanup_reservation() {
  if [[ "$reservation_committed" -eq 0 ]]; then
    rmdir "$reservation_dir" 2>/dev/null || true
  fi
}
trap cleanup_reservation EXIT
if [[ ! "$parent_report_job" =~ ^[0-9]+$ ]]; then
  echo "parent report job must be a numeric Slurm job ID" >&2
  exit 2
fi

mkdir -p outputs/benchmarks/logs
export_spec="ALL,EFFDOCK_REPO_DIR=$repo_root,EFFDOCK_OUTPUT_ROOT=$output_root"
export_spec+=",EFFDOCK_COHORT_MANIFEST=,EFFDOCK_ONLY_ID=,EFFDOCK_DATASET=,EFFDOCK_ETA="

submit_job() {
  local dependency=$1
  local script=$2
  local array_spec=${3:-}
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

posebusters_job=$(
  submit_job "$parent_report_job" \
    scripts/slurm/guidance_eta_sweep_selector_posebusters_array.sbatch "0-255%16"
)
reservation_committed=1
printf 'protocol_id=%s\nparent_report_job=%s\nselector_posebusters_job=%s\n' \
  EFFDOCK-UNIFIED-GUIDANCE-ETA-SWEEP-SELECTOR-PB-V1 \
  "$parent_report_job" "$posebusters_job" > "$reservation_dir/.submission"
report_job=$(
  submit_job "$posebusters_job" scripts/slurm/guidance_eta_sweep_selector_report.sbatch
)
printf 'selector_report_job=%s\n' "$report_job" >> "$reservation_dir/.submission"
trap - EXIT

printf 'output_root=%s\n' "$output_root"
printf 'protocol_id=EFFDOCK-UNIFIED-GUIDANCE-ETA-SWEEP-SELECTOR-PB-V1\n'
printf 'parent_report_job=%s\n' "$parent_report_job"
printf 'selector_posebusters_job=%s dependency=afterok:%s tasks=256 mapping=2_selectors_x_2_datasets_x_8_eta_x_8_shards\n' \
  "$posebusters_job" "$parent_report_job"
printf 'selector_report_job=%s dependency=afterok:%s\n' "$report_job" "$posebusters_job"
printf 'monitor: squeue -j %s,%s,%s\n' \
  "$parent_report_job" "$posebusters_job" "$report_job"
