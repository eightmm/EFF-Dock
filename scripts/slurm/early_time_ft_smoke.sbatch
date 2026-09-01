#!/usr/bin/env bash
#SBATCH --job-name=effdock-early-time-smoke
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --partition=6000ada
#SBATCH --qos=veryshort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"
mkdir -p outputs/slurm

config=configs/train_early_time_ft.yaml
checkpoint=weights/effdock_geometry_ft_100k_best.pt
expected_checkpoint_sha256=6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db
actual_checkpoint_sha256=$(sha256sum "$checkpoint" | awk '{print $1}')
[[ "$actual_checkpoint_sha256" == "$expected_checkpoint_sha256" ]] || {
  echo "docking checkpoint hash mismatch" >&2
  exit 2
}

export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
.venv/bin/python scripts/early_time_training_smoke.py \
  --config "$config" \
  --checkpoint "$checkpoint"
