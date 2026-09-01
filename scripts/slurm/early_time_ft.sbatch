#!/usr/bin/env bash
#SBATCH --job-name=effdock-early-time-ft
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --partition=6000ada
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"
mkdir -p outputs/slurm

config=configs/train_early_time_ft.yaml
checkpoint=weights/effdock_geometry_ft_100k_best.pt
output_dir=outputs/eff-dock/early-time-ft-t0p10-v1-retry1-20260813
expected_checkpoint_sha256=6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db

for required in "$config" "$checkpoint" .venv/bin/torchrun .venv/bin/eff-dock; do
  [[ -e "$required" ]] || { echo "missing required path: $required" >&2; exit 2; }
done
actual_checkpoint_sha256=$(sha256sum "$checkpoint" | awk '{print $1}')
[[ "$actual_checkpoint_sha256" == "$expected_checkpoint_sha256" ]] || {
  echo "docking checkpoint hash mismatch" >&2
  exit 2
}
[[ ! -e "$output_dir/checkpoints/latest.pt" ]] || {
  echo "refusing to overwrite completed output: $output_dir/checkpoints/latest.pt" >&2
  exit 2
}

export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export EFFDOCK_DDP_TIMEOUT_MIN=30

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
.venv/bin/torchrun --standalone --nproc_per_node=4 \
  .venv/bin/eff-dock train \
  --config "$config" \
  --init-from "$checkpoint"
