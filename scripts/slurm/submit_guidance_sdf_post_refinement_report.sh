#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
[[ $# -ge 3 && $# -le 5 ]] || {
  echo "usage: $0 RUN_ID INPUT_ROOT PB_JOB_ID_OR_NONE [CONFIDENCE_JOB_ID_OR_NONE] [OUTPUT_DIR]" >&2
  exit 2
}
run_id=$1
input_root=$2
pb_job_id=$3
confidence_job_id=${4:-}
report_output_dir=${5:-"$input_root/report"}
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "unsafe run ID" >&2; exit 2; }
if [[ "$pb_job_id" != none ]]; then
  [[ "$pb_job_id" =~ ^[0-9]+$ ]] || { echo "invalid dependency job ID" >&2; exit 2; }
fi
if [[ -n "$confidence_job_id" && "$confidence_job_id" != none ]]; then
  [[ "$confidence_job_id" =~ ^[0-9]+$ ]] || {
    echo "invalid confidence dependency job ID" >&2
    exit 2
  }
fi
[[ -d "$input_root" ]] || { echo "missing input root" >&2; exit 2; }
[[ ! -e "$report_output_dir" ]] || { echo "report output already exists" >&2; exit 2; }

protocol_id=EFFDOCK-GUIDANCE-SDF-POST-REFINEMENT-REPORT-V2
mapfile -t package_files < <(rg --files src/effdock | sort)
copy_files=(
  "${package_files[@]}" pyproject.toml uv.lock
  scripts/report_guidance_sdf_post_refinement_full.py
  scripts/slurm/guidance_sdf_post_refinement_report.sbatch
  scripts/slurm/submit_guidance_sdf_post_refinement_report.sh
  scripts/create_execution_capsule.py
)
capsule_args=()
for path in "${copy_files[@]}"; do capsule_args+=(--copy-file "$path"); done
execution_root=".effdock_execution_capsules/$protocol_id/$run_id"
.venv/bin/python scripts/create_execution_capsule.py \
  --repo-root "$repo_root" --output "$repo_root/$execution_root" \
  --link-root .venv --link-root outputs \
  "${capsule_args[@]}" >/dev/null
execution_root_abs=$(readlink -f "$execution_root")
base_export="ALL,EFFDOCK_REPO_DIR=$execution_root_abs,PYTHONPATH=$execution_root_abs/src,PYTHONDONTWRITEBYTECODE=1,EFFDOCK_INPUT_ROOT=$input_root,EFFDOCK_REPORT_OUTPUT_DIR=$report_output_dir"
dependency_ids=()
[[ "$pb_job_id" == none ]] || dependency_ids+=("$pb_job_id")
[[ -z "$confidence_job_id" || "$confidence_job_id" == none ]] || dependency_ids+=("$confidence_job_id")
dependency_args=()
if [[ "${#dependency_ids[@]}" -gt 0 ]]; then
  dependency=$(IFS=:; echo "${dependency_ids[*]}")
  dependency_args=(--dependency="afterok:$dependency")
fi
raw=$(sbatch --parsable --partition=cpu_only "${dependency_args[@]}" \
  --export="$base_export" "$execution_root_abs/scripts/slurm/guidance_sdf_post_refinement_report.sbatch")
job_id=${raw%%;*}
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "invalid sbatch response" >&2; exit 2; }
printf '%s\n' \
  'status=submitted' "protocol_id=$protocol_id" "run_id=$run_id" \
  "input_root=$input_root" "output_root=$report_output_dir" \
  "dependency_pb_job=$pb_job_id" \
  "dependency_confidence_job=${confidence_job_id:-none}" "report_job=$job_id" \
  "execution_root=$execution_root_abs" "submitted_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$input_root/.report_submission.$run_id"
printf 'report_job=%s\nexecution_root=%s\n' "$job_id" "$execution_root_abs"
