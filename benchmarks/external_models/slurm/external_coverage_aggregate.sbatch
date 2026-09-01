#!/usr/bin/env bash
#SBATCH --job-name=effdock-ext-cover
#SBATCH --partition=cpu_only
#SBATCH --qos=short
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=external_models/logs/%x-%j.out
#SBATCH --error=external_models/logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
: "${EXPECTED_CSV:?Set EXPECTED_CSV}"
: "${RUN_ROOT:?Set RUN_ROOT}"
: "${OUTPUT_JSON:?Set OUTPUT_JSON}"

aggregate_args=(
  --expected-csv "$EXPECTED_CSV"
  --run-root "$RUN_ROOT"
  --output-json "$OUTPUT_JSON"
)
if test "${STRICT:-0}" = 1; then
  aggregate_args+=(--strict)
fi
if test "${ALLOW_NATIVE_UNSUPPORTED:-0}" = 1; then
  aggregate_args+=(--allow-native-unsupported)
fi
.venv/bin/python scripts/external_models/aggregate_inference_coverage.py \
  "${aggregate_args[@]}"
