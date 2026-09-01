#!/usr/bin/env bash
#SBATCH --job-name=effdock-ref-energy
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1

set -euo pipefail
repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
cd "${repo_dir:?EFFDOCK_REPO_DIR or SLURM_SUBMIT_DIR is required}"
summary=${EFFDOCK_SUMMARY:?EFFDOCK_SUMMARY is required}
output=${EFFDOCK_OUTPUT:?EFFDOCK_OUTPUT is required}
task_cpus=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS="$task_cpus" MKL_NUM_THREADS="$task_cpus" OPENBLAS_NUM_THREADS="$task_cpus"
mkdir -p outputs/benchmarks/logs
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits

.venv/bin/python scripts/analyze_guidance_refinement_energy.py \
  --summary "$summary" \
  --benchmark-input-manifest docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json \
  --external-dir data/external_test \
  --device cuda \
  --batch-size 10 \
  --output "$output"
