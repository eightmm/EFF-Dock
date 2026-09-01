#!/usr/bin/env bash
#SBATCH --job-name=effdock-cut-report
#SBATCH --partition=cpu_only
#SBATCH --qos=short
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err

set -euo pipefail
cd "${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:?}}"
: "${EFFDOCK_OUTPUT_ROOT:?set EFFDOCK_OUTPUT_ROOT}"
runtime_venv=${EFFDOCK_RUNTIME_VENV:-.venv}
"$runtime_venv/bin/python" benchmarks/effdock/report_pocket_cutoff_robustness.py \
  --output-root "$EFFDOCK_OUTPUT_ROOT" --output "$EFFDOCK_OUTPUT_ROOT/report.json"
