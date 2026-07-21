#!/usr/bin/env bash
#SBATCH --job-name=effdock-confidence-train
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
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

config=${EFFDOCK_CONFIDENCE_CONFIG:-configs/train_confidence.yaml}
args=(--config "$config")
if [[ "${EFFDOCK_CONFIDENCE_SMOKE:-0}" == "1" ]]; then
  args+=(
    --total_steps 1
    --train_limit 1
    --val_limit 1
    --eval_every 1
    --eval_complexes 1
    --save_every 1
    --num_workers 0
    --max_train_poses_per_complex 2
    --max_val_poses_per_complex 2
    --no-use_muon
    --no-use_wandb
    --out_dir outputs/eff-dock/confidence-smoke
  )
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
uv run eff-dock confidence train "${args[@]}"
