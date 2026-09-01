#!/usr/bin/env bash
#SBATCH --job-name=effdock-cut-stage
#SBATCH --partition=6000ada,heavy,test
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err

set -euo pipefail
cd "${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:?}}"
: "${EFFDOCK_OUTPUT_ROOT:?set EFFDOCK_OUTPUT_ROOT}"
: "${EFFDOCK_STAGE:?set EFFDOCK_STAGE}"
runtime_venv=${EFFDOCK_RUNTIME_VENV:-.venv}
cutoffs=(6 8 10 12)
num_shards=32
task_id=${SLURM_ARRAY_TASK_ID:?}
[[ "${SLURM_ARRAY_TASK_COUNT:?}" -eq 384 ]] || { echo "stage requires 384 tasks" >&2; exit 2; }
shard_index=$((task_id % num_shards))
repeat_index=$(((task_id / num_shards) % 3))
cutoff=${cutoffs[$((task_id / (num_shards * 3)))]}
condition_root="$EFFDOCK_OUTPUT_ROOT/cutoff_$(printf '%02d' "$cutoff")/repeat_$repeat_index"
"$runtime_venv/bin/python" scripts/run_guidance_sdf_post_refinement_shard.py \
  --manifest "$condition_root/manifest.json" \
  --benchmark-input-manifest docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json \
  --protocol-file docs/GUIDANCE_SIGMA2_ETA2_REFINEMENT_PROTOCOL.md \
  --external-dir data/external_test --output-root "$condition_root/full" \
  --num-shards "$num_shards" --shard-index "$shard_index" \
  --stage "$EFFDOCK_STAGE" --eta 2 \
  --energy-convergence-absolute-kcal-mol 0.02 \
  --energy-convergence-relative 0.001 \
  --energy-convergence-patience 5 --energy-convergence-min-steps 25 \
  --confidence-sigma 2 --pocket-cutoff 10 \
  --docking-checkpoint weights/effdock_docking_early_time_t0p10_50k.pt \
  --confidence-checkpoint weights/effdock_confidence_s50_raw_refined_u70k.pt
