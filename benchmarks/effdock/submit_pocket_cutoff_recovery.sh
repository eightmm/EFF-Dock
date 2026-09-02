#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$repo_root"
[[ $# -eq 3 ]] || { echo "usage: $0 OUTPUT_ROOT EXECUTION_ROOT SMOKE_JOB_ID" >&2; exit 2; }
output_root=$(readlink -f "$1")
execution_root=$(readlink -f "$2")
smoke_job=$3
[[ "$smoke_job" =~ ^[0-9]+$ ]] || { echo "invalid smoke job ID" >&2; exit 2; }
archive_root="$output_root/recovery_archive/generation_job_63380"
task_file="$archive_root/failed_task_ids.txt"
[[ -f "$task_file" ]] || { echo "missing failed-task inventory: $task_file" >&2; exit 2; }
[[ -x .venv/bin/python && -f "$execution_root/benchmarks/effdock/slurm/pocket_cutoff_generation.sbatch" ]]
failed_tasks=$(tr -d '[:space:]' < "$task_file")
[[ "$failed_tasks" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "invalid failed task list" >&2; exit 2; }

base_export="ALL,EFFDOCK_REPO_DIR=$execution_root,EFFDOCK_RUNTIME_VENV=$repo_root/.venv,PYTHONPATH=$execution_root/src,PYTHONDONTWRITEBYTECODE=1,EFFDOCK_OUTPUT_ROOT=$output_root"
submitted=()
committed=0
cleanup() {
  code=$?
  trap - EXIT
  if [[ "$committed" -eq 0 ]]; then
    for job in "${submitted[@]}"; do scancel "$job" 2>/dev/null || true; done
  fi
  exit "$code"
}
trap cleanup EXIT
generation_job=$(sbatch --parsable --partition=6000ada,heavy \
  --dependency="afterok:$smoke_job" --array="$failed_tasks%8" \
  --export="$base_export" benchmarks/effdock/slurm/pocket_cutoff_generation_recovery.sbatch)
generation_job=${generation_job%%;*}
submitted+=("$generation_job")
manifest_job=$(sbatch --parsable --partition=cpu_only --dependency="afterok:$generation_job" \
  --array=0-11%12 --export="$base_export" \
  "$execution_root/benchmarks/effdock/slurm/pocket_cutoff_manifest.sbatch")
manifest_job=${manifest_job%%;*}
submitted+=("$manifest_job")
refinement_job=$(sbatch --parsable --partition=6000ada,heavy --dependency="afterok:$manifest_job" \
  --array=0-383%8 --export="$base_export,EFFDOCK_STAGE=refinement" \
  "$execution_root/benchmarks/effdock/slurm/pocket_cutoff_refine_confidence.sbatch")
refinement_job=${refinement_job%%;*}
submitted+=("$refinement_job")
confidence_job=$(sbatch --parsable --partition=6000ada,heavy --dependency="afterok:$refinement_job" \
  --array=0-383%8 --export="$base_export,EFFDOCK_STAGE=confidence" \
  "$execution_root/benchmarks/effdock/slurm/pocket_cutoff_refine_confidence.sbatch")
confidence_job=${confidence_job%%;*}
submitted+=("$confidence_job")
posebusters_job=$(sbatch --parsable --partition=cpu_only --dependency="afterok:$confidence_job" \
  --array=0-191%12 --export="$base_export" \
  "$execution_root/benchmarks/effdock/slurm/pocket_cutoff_selected_posebusters.sbatch")
posebusters_job=${posebusters_job%%;*}
submitted+=("$posebusters_job")
report_job=$(sbatch --parsable --partition=cpu_only --dependency="afterok:$posebusters_job" \
  --export="$base_export" "$execution_root/benchmarks/effdock/slurm/pocket_cutoff_report.sbatch")
report_job=${report_job%%;*}
submitted+=("$report_job")

recovery_file="$output_root/.submission.recovery-63380"
printf '%s\n' \
  'status=submitted' 'source_generation_job=63380' \
  "recovery_smoke_job=$smoke_job" \
  "failed_task_ids=$failed_tasks" 'gpu_partitions=6000ada,heavy' \
  "generation_job=$generation_job" "manifest_job=$manifest_job" \
  "refinement_job=$refinement_job" "confidence_job=$confidence_job" \
  "posebusters_job=$posebusters_job" "report_job=$report_job" \
  "submitted_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$recovery_file"
committed=1
trap - EXIT
printf 'generation=%s manifest=%s refinement=%s confidence=%s posebusters=%s report=%s\n' \
  "$generation_job" "$manifest_job" "$refinement_job" "$confidence_job" \
  "$posebusters_job" "$report_job"
