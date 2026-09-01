#!/usr/bin/env bash
#SBATCH --job-name=effdock-confidence-prepare
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"
mkdir -p outputs/slurm

num_shards=${EFFDOCK_NUM_SHARDS:-1}
shard_index=${SLURM_ARRAY_TASK_ID:-0}
limit=${EFFDOCK_CONFIDENCE_PREPARE_LIMIT:-1000}
start=$((shard_index * limit))
split=${EFFDOCK_CONFIDENCE_PREPARE_SPLIT:-train}
checkpoint=${EFFDOCK_DOCKING_CHECKPOINT:-weights/effdock_docking_early_time_t0p10_50k.pt}
config=${EFFDOCK_DOCKING_CONFIG:-configs/train.yaml}
pose_tag=${EFFDOCK_CONFIDENCE_POSE_TAG:-s50_n100_s10_sig2_pc10}
num_samples=${EFFDOCK_CONFIDENCE_PREPARE_NUM_SAMPLES:-100}
num_steps=${EFFDOCK_CONFIDENCE_PREPARE_NUM_STEPS:-10}
sigma=${EFFDOCK_CONFIDENCE_PREPARE_SIGMA:-2.0}
pocket_cutoff=${EFFDOCK_CONFIDENCE_PREPARE_POCKET_CUTOFF:-10}
output_args=()
if [[ -n "${EFFDOCK_CONFIDENCE_PREPARE_OUTPUT_ROOT:-}" ]]; then
  output_args=(--output_root "$EFFDOCK_CONFIDENCE_PREPARE_OUTPUT_ROOT")
fi

if (( shard_index >= num_shards )); then
  echo "array index ${shard_index} is outside EFFDOCK_NUM_SHARDS=${num_shards}"
  exit 2
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
uv run eff-dock confidence prepare \
  --checkpoint "$checkpoint" \
  --config "$config" \
  --split "$split" \
  --start "$start" \
  --limit "$limit" \
  --num_samples "$num_samples" \
  --num_steps "$num_steps" \
  --sigma "$sigma" \
  --pocket_cutoff "$pocket_cutoff" \
  --seed 42 \
  --pose_tag "$pose_tag" \
  --manifest_suffix "shard${shard_index}" \
  "${output_args[@]}"
