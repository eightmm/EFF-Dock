#!/usr/bin/env bash
#SBATCH --job-name=effdock-cut-gen-retry
#SBATCH --partition=6000ada,heavy
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err

set -euo pipefail
: "${EFFDOCK_REPO_DIR:?set EFFDOCK_REPO_DIR}"
gpu_memory_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | tr -d '[:space:]')
[[ "$gpu_memory_mib" =~ ^[0-9]+$ && "$gpu_memory_mib" -ge 48000 ]] || {
  echo "pocket-cutoff recovery requires one GPU with at least 48000 MiB" >&2
  exit 2
}
export PYTORCH_ALLOC_CONF=expandable_segments:True
# The frozen generation script validates the original 192-cell task domain.
export SLURM_ARRAY_TASK_COUNT=192
exec bash "$EFFDOCK_REPO_DIR/benchmarks/effdock/slurm/pocket_cutoff_generation.sbatch"
