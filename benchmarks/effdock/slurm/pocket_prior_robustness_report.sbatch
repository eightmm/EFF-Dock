#!/usr/bin/env bash
#SBATCH --job-name=effdock-pocket-prior-report
#SBATCH --partition=cpu_only
#SBATCH --qos=short
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err

set -euo pipefail
cd "${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:?}}"
: "${EFFDOCK_OUTPUT_ROOT:?set EFFDOCK_OUTPUT_ROOT}"
: "${EFFDOCK_BASELINE_REPORT:?set EFFDOCK_BASELINE_REPORT}"
runtime_venv=${EFFDOCK_RUNTIME_VENV:-.venv}
"$runtime_venv/bin/python" benchmarks/effdock/report_pocket_prior_robustness_extension.py \
  --output-root "$EFFDOCK_OUTPUT_ROOT" --baseline-report "$EFFDOCK_BASELINE_REPORT" \
  --output "$EFFDOCK_OUTPUT_ROOT/report.json"
