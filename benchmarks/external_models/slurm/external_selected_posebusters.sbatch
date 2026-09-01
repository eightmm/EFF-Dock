#!/usr/bin/env bash
#SBATCH --job-name=effdock-ext-pb-r3
#SBATCH --partition=cpu_only
#SBATCH --qos=short
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=external_models/logs/%x-%A_%a.out
#SBATCH --error=external_models/logs/%x-%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

models=(diffdock_pocket rldiff_rlpp diffbindfr)
datasets=(astex_diverse posebusters_benchmark)
num_shards="${NUM_SHARDS:-12}"
task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
tasks_per_model=$((2 * 3 * num_shards))
tasks_per_dataset=$((3 * num_shards))
model_index=$((task_id / tasks_per_model))
dataset_index=$(((task_id % tasks_per_model) / tasks_per_dataset))
repeat_index=$(((task_id % tasks_per_dataset) / num_shards))
shard_index=$((task_id % num_shards))
model="${models[$model_index]}"
dataset="${datasets[$dataset_index]}"

args=(
  --model "$model"
  --dataset "$dataset"
  --repeat-index "$repeat_index"
  --num-shards "$num_shards"
  --shard-index "$shard_index"
)
if [[ -n "${LIMIT:-}" ]]; then
  args+=(--limit "$LIMIT")
fi
if [[ -n "${OUTPUT_ROOT:-}" ]]; then
  args+=(--output-root "$OUTPUT_ROOT")
fi

uv run python benchmarks/external_models/evaluate_selected_posebusters.py "${args[@]}"
