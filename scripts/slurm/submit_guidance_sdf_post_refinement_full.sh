#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
[[ $# -le 2 ]] || { echo "usage: $0 [SAFE_RUN_ID] [full|confidence-only]" >&2; exit 2; }
run_id=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
mode=${2:-full}
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "unsafe run ID" >&2; exit 2; }
[[ "$mode" == full || "$mode" == confidence-only ]] || { echo "invalid mode" >&2; exit 2; }

protocol_id=EFFDOCK-GUIDANCE-SDF-POST-REFINEMENT-CONFIDENCE-V2
output_root=${EFFDOCK_RESUME_OUTPUT_ROOT:-"outputs/benchmarks/guidance_sdf_post_refinement_runs/$run_id/full"}
source_manifest=${EFFDOCK_SOURCE_MANIFEST:-outputs/benchmarks/guidance_all_pose_pb_eta_runs/20260811T042451Z/manifest.json}
source_eta=${EFFDOCK_SOURCE_ETA:-0}
refinement_protocol_file=${EFFDOCK_REFINEMENT_PROTOCOL_FILE:-docs/GUIDANCE_SDF_POST_REFINEMENT_PROTOCOL.md}
refinement_partition=${EFFDOCK_REFINEMENT_PARTITION:-6000ada,heavy,test}
mkdir -p outputs/benchmarks/guidance_sdf_post_refinement_runs outputs/benchmarks/logs
if [[ -n "${EFFDOCK_RESUME_OUTPUT_ROOT:-}" ]]; then
  [[ -d "$output_root" ]] || { echo "resume output root does not exist" >&2; exit 2; }
else
  mkdir -p "$(dirname "$output_root")"
  mkdir "$output_root" || { echo "refusing existing output root" >&2; exit 2; }
fi

for required in \
  .venv/bin/python \
  "$source_manifest" \
  docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json \
  "$refinement_protocol_file" \
  weights/effdock_geometry_ft_100k_best.pt \
  weights/effdock_confidence_extmatch_n80_s25_step42500.pt; do
  [[ -e "$required" ]] || { echo "missing input $required" >&2; exit 2; }
done

mapfile -t package_files < <(rg --files src/effdock | sort)
copy_files=(
  "${package_files[@]}"
  pyproject.toml uv.lock configs/train.yaml
  docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json
  docs/GUIDANCE_SDF_POST_REFINEMENT_PROTOCOL.md
  docs/GUIDANCE_SIGMA2_ETA2_REFINEMENT_PROTOCOL.md
  scripts/run_guidance_sdf_post_refinement.py
  scripts/score_guidance_sdf_post_refinement_confidence.py
  scripts/run_guidance_sdf_post_refinement_shard.py
  scripts/slurm/guidance_sdf_post_refinement_full_array.sbatch
  scripts/slurm/submit_guidance_sdf_post_refinement_full.sh
  scripts/create_execution_capsule.py
)
capsule_args=()
for path in "${copy_files[@]}"; do capsule_args+=(--copy-file "$path"); done
execution_root=".effdock_execution_capsules/$protocol_id/$run_id"
.venv/bin/python scripts/create_execution_capsule.py \
  --repo-root "$repo_root" --output "$repo_root/$execution_root" \
  --link-root .venv --link-root data --link-root outputs --link-root weights \
  "${capsule_args[@]}" >/dev/null
execution_root_abs=$(readlink -f "$execution_root")

git_commit=$(git rev-parse HEAD)
read -r git_diff_sha256 _ < <(git diff --no-ext-diff | sha256sum)
printf '%s\n' \
  'status=submitting' \
  "protocol_id=$protocol_id" \
  "run_id=$run_id" \
  "output_root=$output_root" \
  'datasets=astex:85,posebusters:308' \
  'poses_per_complex=100' \
  'refinement_steps=100' \
  "source_manifest=$source_manifest" \
  "source_eta=$source_eta" \
  "refinement_protocol_file=$refinement_protocol_file" \
  "energy_convergence_absolute_kcal_mol=${EFFDOCK_ENERGY_CONVERGENCE_ABSOLUTE_KCAL_MOL:-disabled}" \
  "energy_convergence_relative=${EFFDOCK_ENERGY_CONVERGENCE_RELATIVE:-disabled}" \
  "energy_convergence_patience=${EFFDOCK_ENERGY_CONVERGENCE_PATIENCE:-5}" \
  "energy_convergence_min_steps=${EFFDOCK_ENERGY_CONVERGENCE_MIN_STEPS:-20}" \
  "mode=$mode" \
  'selector=argmin_frozen_confidence_rmsd' \
  "confidence_sigma=${EFFDOCK_CONFIDENCE_SIGMA:-0.5}" \
  'confidence_pose_batch_size=20' \
  'historical_selector_match=diagnostic_only' \
  'confidence_partition=6000ada' \
  "git_commit=$git_commit" \
  "git_diff_sha256=$git_diff_sha256" \
  "execution_root=$execution_root_abs" > "$output_root/.submission.pending"

base_export="ALL,EFFDOCK_REPO_DIR=$execution_root_abs,PYTHONPATH=$execution_root_abs/src,PYTHONDONTWRITEBYTECODE=1,EFFDOCK_OUTPUT_ROOT=$repo_root/$output_root,EFFDOCK_SOURCE_MANIFEST=$source_manifest,EFFDOCK_SOURCE_ETA=$source_eta,EFFDOCK_REFINEMENT_PROTOCOL_FILE=$refinement_protocol_file,EFFDOCK_ENERGY_CONVERGENCE_ABSOLUTE_KCAL_MOL=${EFFDOCK_ENERGY_CONVERGENCE_ABSOLUTE_KCAL_MOL:-},EFFDOCK_ENERGY_CONVERGENCE_RELATIVE=${EFFDOCK_ENERGY_CONVERGENCE_RELATIVE:-},EFFDOCK_ENERGY_CONVERGENCE_PATIENCE=${EFFDOCK_ENERGY_CONVERGENCE_PATIENCE:-5},EFFDOCK_ENERGY_CONVERGENCE_MIN_STEPS=${EFFDOCK_ENERGY_CONVERGENCE_MIN_STEPS:-20},EFFDOCK_CONFIDENCE_SIGMA=${EFFDOCK_CONFIDENCE_SIGMA:-0.5}"
refinement_job_id=not_submitted
dependency_args=()
if [[ "$mode" == full ]]; then
  raw=$(sbatch --parsable --job-name=effdock-sdf-refine-full \
    --partition="$refinement_partition" --array=0-31%12 \
    --export="$base_export,EFFDOCK_STAGE=refinement" \
    "$execution_root_abs/scripts/slurm/guidance_sdf_post_refinement_full_array.sbatch")
  refinement_job_id=${raw%%;*}
  [[ "$refinement_job_id" =~ ^[0-9]+$ ]] || { echo "invalid sbatch response: $raw" >&2; exit 2; }
  dependency_args=(--dependency="afterok:$refinement_job_id")
else
  refinement_count=$(find "$output_root/refinement" -name summary.json -type f | wc -l)
  [[ "$refinement_count" -eq 393 ]] || {
    echo "confidence-only requires 393 refinement summaries, got $refinement_count" >&2
    exit 2
  }
fi
raw=$(sbatch --parsable --job-name=effdock-sdf-conf20-full \
  --partition=6000ada "${dependency_args[@]}" --array=0-31%12 \
  --export="$base_export,EFFDOCK_STAGE=confidence" \
  "$execution_root_abs/scripts/slurm/guidance_sdf_post_refinement_full_array.sbatch")
confidence_job_id=${raw%%;*}
[[ "$confidence_job_id" =~ ^[0-9]+$ ]] || { echo "invalid sbatch response: $raw" >&2; exit 2; }
printf '%s\n' "refinement_array_job=$refinement_job_id" \
  "confidence_array_job=$confidence_job_id" \
  "submitted_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >> "$output_root/.submission.pending"
sed -i 's/^status=submitting$/status=submitted/' "$output_root/.submission.pending"
submission_path="$output_root/.submission"
[[ -z "${EFFDOCK_RESUME_OUTPUT_ROOT:-}" ]] || submission_path="$output_root/.submission.resume.$run_id"
mv "$output_root/.submission.pending" "$submission_path"
printf 'output_root=%s\nrefinement_array_job=%s\nconfidence_array_job=%s\nexecution_root=%s\n' \
  "$repo_root/$output_root" "$refinement_job_id" "$confidence_job_id" "$execution_root_abs"
