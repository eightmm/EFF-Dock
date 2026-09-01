#!/usr/bin/env bash
#SBATCH --job-name=effdock-early-time-ddp-smoke
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --partition=6000ada
#SBATCH --qos=veryshort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:4

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"
mkdir -p outputs/slurm

config=${EFFDOCK_SMOKE_CONFIG:-configs/train_early_time_ft.yaml}
checkpoint=${EFFDOCK_SMOKE_CHECKPOINT:-weights/effdock_geometry_ft_100k_best.pt}
expected_checkpoint_sha256=${EFFDOCK_SMOKE_CHECKPOINT_SHA256:-6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db}
smoke_world_size=${EFFDOCK_SMOKE_WORLD_SIZE:-4}

if ! [[ "$smoke_world_size" =~ ^[1-9][0-9]*$ ]]; then
  echo "EFFDOCK_SMOKE_WORLD_SIZE must be a positive integer" >&2
  exit 2
fi

for required in "$config" "$checkpoint" .venv/bin/torchrun scripts/early_time_training_smoke.py; do
  [[ -e "$required" ]] || { echo "missing required path: $required" >&2; exit 2; }
done
actual_checkpoint_sha256=$(sha256sum "$checkpoint" | awk '{print $1}')
[[ "$actual_checkpoint_sha256" == "$expected_checkpoint_sha256" ]] || {
  echo "docking checkpoint hash mismatch" >&2
  exit 2
}

export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export EFFDOCK_DDP_TIMEOUT_MIN=30

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
.venv/bin/torchrun --standalone --nproc_per_node="$smoke_world_size" \
  scripts/early_time_training_smoke.py \
  --config "$config" \
  --checkpoint "$checkpoint" \
  --exercise-rollout

resume_checkpoint=outputs/eff-dock/early-time-ft-smoke-${SLURM_JOB_ID}/checkpoints/latest.pt
.venv/bin/torchrun --standalone --nproc_per_node="$smoke_world_size" \
  scripts/early_time_training_smoke.py \
  --config "$config" \
  --checkpoint "$checkpoint" \
  --exercise-rollout \
  --resume-only \
  --resume-checkpoint "$resume_checkpoint"
