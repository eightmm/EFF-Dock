#!/usr/bin/env bash
#SBATCH --job-name=effdock-u50-pb-report
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
: "${EFFDOCK_OUTPUT_ROOT:?}"
: "${EFFDOCK_SOURCE_REPORT:?}"
: "${EFFDOCK_SCORES_ROOT:?}"
: "${EFFDOCK_STEP100_OFFICIAL_ROOT:?}"
: "${EFFDOCK_DATASET:?}"
: "${EFFDOCK_EXPECTED_SCRIPT_SHA256:?}"
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$SLURM_SUBMIT_DIR"
export PYTHONDONTWRITEBYTECODE=1
read -r actual _ < <(sha256sum scripts/evaluate_s50_u50_refinement_validity.py)
[[ "$actual" == "$EFFDOCK_EXPECTED_SCRIPT_SHA256" ]] || exit 2
.venv/bin/python scripts/evaluate_s50_u50_refinement_validity.py report \
  --source-report "$EFFDOCK_SOURCE_REPORT" \
  --dataset "$EFFDOCK_DATASET" \
  --scores-root "$EFFDOCK_SCORES_ROOT" \
  --step100-official-root "$EFFDOCK_STEP100_OFFICIAL_ROOT" \
  --shards-root "$EFFDOCK_OUTPUT_ROOT/shards" \
  --output "$EFFDOCK_OUTPUT_ROOT/report.json"
