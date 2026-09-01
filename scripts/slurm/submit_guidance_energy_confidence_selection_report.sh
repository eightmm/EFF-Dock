#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
[[ $# -eq 3 ]] || { echo "usage: $0 RUN_ID INPUT_ROOT OUTPUT_DIR" >&2; exit 2; }
run_id=$1
input_root=$2
output_dir=$3
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "unsafe run ID" >&2
  exit 2
}
[[ -d "$input_root" ]] || { echo "missing input root" >&2; exit 2; }
[[ ! -e "$output_dir" ]] || { echo "output already exists" >&2; exit 2; }

for required in \
  .venv/bin/python \
  docs/GUIDANCE_ENERGY_CONFIDENCE_SELECTION_PROTOCOL.md \
  scripts/report_guidance_energy_confidence_selection.py \
  scripts/report_guidance_sdf_post_refinement_full.py; do
  [[ -e "$required" ]] || { echo "missing input $required" >&2; exit 2; }
done

protocol_id=EFFDOCK-GUIDANCE-ENERGY-CONFIDENCE-SELECTION-V1
mapfile -t package_files < <(rg --files src/effdock | sort)
copy_files=(
  "${package_files[@]}" pyproject.toml uv.lock
  docs/GUIDANCE_ENERGY_CONFIDENCE_SELECTION_PROTOCOL.md
  scripts/report_guidance_energy_confidence_selection.py
  scripts/report_guidance_sdf_post_refinement_full.py
  scripts/slurm/guidance_energy_confidence_selection_report.sbatch
  scripts/slurm/submit_guidance_energy_confidence_selection_report.sh
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
base_export="ALL,EFFDOCK_REPO_DIR=$execution_root_abs,PYTHONPATH=$execution_root_abs/src,PYTHONDONTWRITEBYTECODE=1,EFFDOCK_INPUT_ROOT=$input_root,EFFDOCK_REPORT_OUTPUT_DIR=$output_dir"
raw=$(sbatch --parsable --partition=cpu_only --export="$base_export" \
  "$execution_root_abs/scripts/slurm/guidance_energy_confidence_selection_report.sbatch")
job_id=${raw%%;*}
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "invalid sbatch response" >&2; exit 2; }
printf '%s\n' \
  'status=submitted' "protocol_id=$protocol_id" "run_id=$run_id" \
  "input_root=$input_root" "output_root=$output_dir" "report_job=$job_id" \
  "execution_root=$execution_root_abs" \
  "submitted_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$input_root/.energy_confidence_selection_submission.$run_id"
printf 'report_job=%s\noutput_root=%s\nexecution_root=%s\n' \
  "$job_id" "$output_dir" "$execution_root_abs"
