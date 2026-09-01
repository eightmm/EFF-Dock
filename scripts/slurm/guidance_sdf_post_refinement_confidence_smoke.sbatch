#!/usr/bin/env bash
#SBATCH --job-name=effdock-sdf-conf
#SBATCH --partition=6000ada
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/benchmarks/logs/effdock-sdf-conf-%j.out
#SBATCH --error=outputs/benchmarks/logs/effdock-sdf-conf-%j.err

set -euo pipefail
repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
cd "${repo_dir:?EFFDOCK_REPO_DIR or SLURM_SUBMIT_DIR is required}"
: "${EFFDOCK_REFINEMENT_SUMMARY:?set EFFDOCK_REFINEMENT_SUMMARY}"
: "${EFFDOCK_CONFIDENCE_OUTPUT_DIR:?set EFFDOCK_CONFIDENCE_OUTPUT_DIR}"
mkdir -p outputs/benchmarks/logs
task_cpus=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS="$task_cpus" MKL_NUM_THREADS="$task_cpus" OPENBLAS_NUM_THREADS="$task_cpus"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
.venv/bin/python scripts/score_guidance_sdf_post_refinement_confidence.py \
  --refinement-summary "${EFFDOCK_REFINEMENT_SUMMARY}" \
  --benchmark-input-manifest docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json \
  --external-dir data/external_test \
  --config configs/train.yaml \
  --docking-checkpoint weights/effdock_geometry_ft_100k_best.pt \
  --confidence-checkpoint weights/effdock_confidence_extmatch_n80_s25_step42500.pt \
  --output-dir "${EFFDOCK_CONFIDENCE_OUTPUT_DIR}" \
  --sigma 0.5 --pose-batch-size 20 --device cuda
