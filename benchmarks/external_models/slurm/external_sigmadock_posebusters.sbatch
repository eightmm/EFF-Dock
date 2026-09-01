#!/usr/bin/env bash
#SBATCH --job-name=effdock-ext-sigma-pb
#SBATCH --partition=cpu_only
#SBATCH --qos=long
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=others/logs/%x-%A_%a.out
#SBATCH --error=others/logs/%x-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
: "${MODEL_ROOT:?Set MODEL_ROOT to the directory containing seed_*}"
: "${EXPECTED_TARGETS:?Set EXPECTED_TARGETS}"

SEED="${SEED:-${SLURM_ARRAY_TASK_ID:?Set SEED or submit an array}}"
SEED_DIR="$MODEL_ROOT/seed_$SEED"
PREDICTION_PATH="$SEED_DIR/predictions.pt"
OUTPUT_PATH="$SEED_DIR/posebusters.pt"
MODEL_RUNNER="$SLURM_SUBMIT_DIR/scripts/others/run_model.sh"

test -f "$PREDICTION_PATH"
test -f "$SEED_DIR/rescoring.pt"
if test -f "$OUTPUT_PATH"; then
  echo "SigmaDock PoseBusters already complete: $OUTPUT_PATH"
  exit 0
fi

"$MODEL_RUNNER" sigmadock python \
  scripts/external_models/postprocess_sigmadock_posebusters.py \
  --prediction-path "$PREDICTION_PATH" \
  --output-path "$OUTPUT_PATH" \
  --expected-targets "$EXPECTED_TARGETS" \
  --config redock
