#!/usr/bin/env bash
#SBATCH --job-name=effdock-ext-rmsd
#SBATCH --partition=cpu_only
#SBATCH --qos=short
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=external_models/logs/%x-%A_%a.out
#SBATCH --error=external_models/logs/%x-%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

arms=(
  diffdock_pocket rldiff rldiff_rlpp surfdock diffbindfr interformer
  posebench_diffdock posebench_fabind posebench_dynamicbind posebench_vina
  sigmadock
)
datasets=(astex_diverse posebusters_benchmark)
task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
model="${arms[$((task_id / 2))]}"
dataset="${datasets[$((task_id % 2))]}"
output_dir="${OUTPUT_DIR:-$SLURM_SUBMIT_DIR/outputs/external_models/evaluation/common_rmsd_v1}"

uv run python scripts/external_models/evaluate_native_outputs.py \
  --model "$model" \
  --dataset "$dataset" \
  --output-dir "$output_dir"
