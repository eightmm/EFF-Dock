#!/usr/bin/env bash
#SBATCH --job-name=effdock-u50-pb-before
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-31%8
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err

set -euo pipefail
: "${EFFDOCK_OUTPUT_ROOT:?}"
: "${EFFDOCK_SOURCE_REPORT:?}"
: "${EFFDOCK_SCORES_ROOT:?}"
: "${EFFDOCK_REFINEMENT_ROOT:?}"
: "${EFFDOCK_DATASET:?}"
: "${EFFDOCK_EXPECTED_SCRIPT_SHA256:?}"
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$SLURM_SUBMIT_DIR"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
read -r actual _ < <(sha256sum scripts/evaluate_s50_u50_refinement_validity.py)
[[ "$actual" == "$EFFDOCK_EXPECTED_SCRIPT_SHA256" ]] || exit 2
.venv/bin/python scripts/evaluate_s50_u50_refinement_validity.py shard \
  --source-report "$EFFDOCK_SOURCE_REPORT" \
  --dataset "$EFFDOCK_DATASET" \
  --scores-root "$EFFDOCK_SCORES_ROOT" \
  --refinement-root "$EFFDOCK_REFINEMENT_ROOT" \
  --output-root "$EFFDOCK_OUTPUT_ROOT/shards" \
  --num-shards 32 \
  --shard-index "${SLURM_ARRAY_TASK_ID:?}"
