#!/usr/bin/env bash
#SBATCH --job-name=effdock-ext-rmsd-r3
#SBATCH --partition=cpu_only
#SBATCH --qos=short
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=external_models/logs/%x-%A_%a.out
#SBATCH --error=external_models/logs/%x-%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

models=(diffdock_pocket rldiff_rlpp diffbindfr posebench_vina)
datasets=(astex_diverse posebusters_benchmark)
task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
model_index=$((task_id / 6))
dataset_index=$(((task_id % 6) / 3))
repeat_index=$((task_id % 3))
model="${models[$model_index]}"
dataset="${datasets[$dataset_index]}"
seed="$repeat_index"
if [[ "$model" == "posebench_vina" ]]; then
  seed=$((repeat_index + 1))
fi

output_dir="${OUTPUT_DIR:-$SLURM_SUBMIT_DIR/outputs/external_models/evaluation/official_repeat_rmsd_20260901}"
output_dir="$output_dir/$model/$dataset/repeat_$repeat_index"

uv run python scripts/external_models/evaluate_native_outputs.py \
  --model "$model" \
  --dataset "$dataset" \
  --official-seed "$seed" \
  --output-dir "$output_dir"
