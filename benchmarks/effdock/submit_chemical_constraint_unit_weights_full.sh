#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

run_id=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "unsafe run ID" >&2; exit 2; }
output_root="$repo_root/outputs/benchmarks/chemical_constraint_unit_weights_full_runs/$run_id"
mkdir -p "${output_root%/*}" outputs/benchmarks/logs
mkdir "$output_root" || { echo "refusing to reuse output root: $output_root" >&2; exit 2; }
export_spec="ALL,EFFDOCK_REPO_DIR=$repo_root,EFFDOCK_OUTPUT_ROOT=$output_root"

submit() {
  local dependency=$1 script=$2 raw
  local args=(--parsable --export="$export_spec")
  [[ -z "$dependency" ]] || args+=(--dependency="afterok:$dependency")
  raw=$(sbatch "${args[@]}" "$script")
  printf '%s' "${raw%%;*}"
}

audit=$(submit "" benchmarks/effdock/slurm/chemical_constraint_unit_weights_full_audit.sbatch)
merge=$(submit "$audit" benchmarks/effdock/slurm/chemical_constraint_unit_weights_full_merge.sbatch)
sampling=$(submit "$merge" benchmarks/effdock/slurm/chemical_constraint_unit_weights_full_sampling.sbatch)
posebusters=$(submit "$sampling" benchmarks/effdock/slurm/chemical_constraint_unit_weights_full_posebusters.sbatch)
report=$(submit "$posebusters" benchmarks/effdock/slurm/chemical_constraint_unit_weights_full_report.sbatch)

printf 'output_root=%s\n' "$output_root"
printf 'audit_job=%s\nmerge_job=%s\nsampling_job=%s\nposebusters_job=%s\nreport_job=%s\n' \
  "$audit" "$merge" "$sampling" "$posebusters" "$report"
