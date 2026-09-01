#!/usr/bin/env bash
#SBATCH --job-name=effdock-ext-sigma-agg
#SBATCH --partition=cpu_only
#SBATCH --qos=long
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=others/logs/%x-%j.out
#SBATCH --error=others/logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
: "${RUN_ROOT:?Set RUN_ROOT}"
: "${EXPECTED_TARGETS:?Set EXPECTED_TARGETS}"
: "${OUTPUT_JSON:?Set OUTPUT_JSON}"

mkdir -p others/logs
python3 scripts/external_models/aggregate_sigmadock_coverage.py \
  --run-root "$RUN_ROOT" \
  --expected-seeds 40 \
  --expected-targets "$EXPECTED_TARGETS" \
  --output-json "$OUTPUT_JSON"
