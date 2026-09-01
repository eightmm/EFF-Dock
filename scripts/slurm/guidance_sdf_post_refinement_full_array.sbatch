#!/usr/bin/env bash
#SBATCH --job-name=effdock-sdf-stage
#SBATCH --partition=6000ada,heavy,test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-31%12
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err

set -euo pipefail
repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
cd "${repo_dir:?EFFDOCK_REPO_DIR or SLURM_SUBMIT_DIR is required}"
: "${EFFDOCK_OUTPUT_ROOT:?set EFFDOCK_OUTPUT_ROOT}"
: "${EFFDOCK_STAGE:?set EFFDOCK_STAGE}"
task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
task_count=${SLURM_ARRAY_TASK_COUNT:?SLURM_ARRAY_TASK_COUNT is required}
[[ "$task_count" -eq 32 ]] || { echo "full run requires 32 shards" >&2; exit 2; }
mkdir -p outputs/benchmarks/logs
task_cpus=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS="$task_cpus" MKL_NUM_THREADS="$task_cpus" OPENBLAS_NUM_THREADS="$task_cpus"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
source_manifest=${EFFDOCK_SOURCE_MANIFEST:-outputs/benchmarks/guidance_all_pose_pb_eta_runs/20260811T042451Z/manifest.json}
source_eta=${EFFDOCK_SOURCE_ETA:-0}
protocol_file=${EFFDOCK_REFINEMENT_PROTOCOL_FILE:-docs/GUIDANCE_SDF_POST_REFINEMENT_PROTOCOL.md}
energy_args=()
if [[ -n "${EFFDOCK_ENERGY_CONVERGENCE_ABSOLUTE_KCAL_MOL:-}" ]]; then
  energy_args+=(
    --energy-convergence-absolute-kcal-mol
    "$EFFDOCK_ENERGY_CONVERGENCE_ABSOLUTE_KCAL_MOL"
  )
fi
if [[ -n "${EFFDOCK_ENERGY_CONVERGENCE_RELATIVE:-}" ]]; then
  energy_args+=(
    --energy-convergence-relative
    "$EFFDOCK_ENERGY_CONVERGENCE_RELATIVE"
  )
fi
energy_args+=(
  --energy-convergence-patience "${EFFDOCK_ENERGY_CONVERGENCE_PATIENCE:-5}"
  --energy-convergence-min-steps "${EFFDOCK_ENERGY_CONVERGENCE_MIN_STEPS:-20}"
)
.venv/bin/python scripts/run_guidance_sdf_post_refinement_shard.py \
  --manifest "$source_manifest" \
  --benchmark-input-manifest docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json \
  --protocol-file "$protocol_file" \
  --external-dir data/external_test \
  --output-root "$EFFDOCK_OUTPUT_ROOT" \
  --num-shards 32 --shard-index "$task_id" --stage "$EFFDOCK_STAGE" \
  --eta "$source_eta" \
  --confidence-sigma "${EFFDOCK_CONFIDENCE_SIGMA:-0.5}" \
  "${energy_args[@]}"
