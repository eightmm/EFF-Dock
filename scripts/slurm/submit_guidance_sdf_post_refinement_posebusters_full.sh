#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
[[ $# -eq 3 ]] || {
  echo "usage: $0 RUN_ID INPUT_ROOT REFINEMENT_JOB_ID_OR_NONE" >&2
  exit 2
}
run_id=$1
input_root=$2
refinement_job_id=$3
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "unsafe run ID" >&2; exit 2; }
if [[ "$refinement_job_id" != none ]]; then
  [[ "$refinement_job_id" =~ ^[0-9]+$ ]] || {
    echo "invalid dependency job ID" >&2
    exit 2
  }
fi
[[ -d "$input_root" ]] || { echo "missing input root" >&2; exit 2; }

protocol_id=EFFDOCK-GUIDANCE-SDF-POST-REFINEMENT-PB-V1
source_manifest=${EFFDOCK_SOURCE_MANIFEST:-outputs/benchmarks/guidance_all_pose_pb_eta_runs/20260811T042451Z/manifest.json}
source_eta=${EFFDOCK_SOURCE_ETA:-0}
mapfile -t package_files < <(rg --files src/effdock | sort)
copy_files=(
  "${package_files[@]}"
  pyproject.toml uv.lock
  scripts/run_guidance_sdf_post_refinement_posebusters_shard.py
  scripts/slurm/guidance_sdf_post_refinement_posebusters_full_array.sbatch
  scripts/slurm/submit_guidance_sdf_post_refinement_posebusters_full.sh
  scripts/create_execution_capsule.py
)
capsule_args=()
for path in "${copy_files[@]}"; do capsule_args+=(--copy-file "$path"); done
execution_root=".effdock_execution_capsules/$protocol_id/$run_id"
.venv/bin/python scripts/create_execution_capsule.py \
  --repo-root "$repo_root" --output "$repo_root/$execution_root" \
  --link-root .venv --link-root data --link-root outputs \
  "${capsule_args[@]}" >/dev/null
execution_root_abs=$(readlink -f "$execution_root")
pb_output_root="$input_root/posebusters_step100"
if [[ -n "${EFFDOCK_RESUME_PB_OUTPUT_ROOT:-}" ]]; then
  [[ -d "$pb_output_root" ]] || { echo "resume PB output root is missing" >&2; exit 2; }
else
  mkdir "$pb_output_root" || { echo "refusing existing PB output root" >&2; exit 2; }
fi
base_export="ALL,EFFDOCK_REPO_DIR=$execution_root_abs,PYTHONPATH=$execution_root_abs/src,PYTHONDONTWRITEBYTECODE=1,EFFDOCK_INPUT_ROOT=$input_root,EFFDOCK_PB_OUTPUT_ROOT=$pb_output_root,EFFDOCK_SOURCE_MANIFEST=$source_manifest,EFFDOCK_SOURCE_ETA=$source_eta"
dependency_args=()
[[ "$refinement_job_id" == none ]] || dependency_args=(--dependency="afterok:$refinement_job_id")
raw=$(sbatch --parsable --partition=cpu_only "${dependency_args[@]}" \
  --array=0-31%16 --export="$base_export" \
  "$execution_root_abs/scripts/slurm/guidance_sdf_post_refinement_posebusters_full_array.sbatch")
job_id=${raw%%;*}
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "invalid sbatch response" >&2; exit 2; }
printf '%s\n' \
  'status=submitted' "protocol_id=$protocol_id" "run_id=$run_id" \
  "input_root=$input_root" "output_root=$pb_output_root" \
  "source_manifest=$source_manifest" "source_eta=$source_eta" \
  "dependency_refinement_job=$refinement_job_id" "posebusters_array_job=$job_id" \
  "execution_root=$execution_root_abs" "submitted_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$input_root/.posebusters_submission"
printf 'pb_output_root=%s\nposebusters_array_job=%s\nexecution_root=%s\n' \
  "$pb_output_root" "$job_id" "$execution_root_abs"
