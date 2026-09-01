#!/usr/bin/env bash
#SBATCH --job-name=effdock-sdf-pb
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/benchmarks/logs/effdock-sdf-pb-%j.out
#SBATCH --error=outputs/benchmarks/logs/effdock-sdf-pb-%j.err

set -euo pipefail

: "${EFFDOCK_REFINEMENT_SUMMARY:?set EFFDOCK_REFINEMENT_SUMMARY}"
: "${EFFDOCK_PB_OUTPUT_DIR:?set EFFDOCK_PB_OUTPUT_DIR}"

task_cpus=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS="$task_cpus"
export MKL_NUM_THREADS="$task_cpus"

.venv/bin/python scripts/evaluate_guidance_sdf_post_refinement_posebusters.py \
  --refinement-summary "${EFFDOCK_REFINEMENT_SUMMARY}" \
  --output-dir "${EFFDOCK_PB_OUTPUT_DIR}" \
  --workers "$task_cpus"
