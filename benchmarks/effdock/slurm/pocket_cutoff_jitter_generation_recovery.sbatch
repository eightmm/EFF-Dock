#!/usr/bin/env bash
#SBATCH --job-name=effdock-jitter-retry
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
: "${EFFDOCK_REPO_DIR:?}"
gpu_memory_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | tr -d '[:space:]')
[[ "$gpu_memory_mib" =~ ^[0-9]+$ && "$gpu_memory_mib" -ge 48000 ]] || exit 2
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Preserve the frozen task-to-condition mapping for a sparse recovery array.
export SLURM_ARRAY_TASK_COUNT=384
exec bash "$EFFDOCK_REPO_DIR/benchmarks/effdock/slurm/pocket_cutoff_jitter_generation.sbatch"
