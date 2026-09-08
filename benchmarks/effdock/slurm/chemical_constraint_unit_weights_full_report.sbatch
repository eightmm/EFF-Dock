#!/usr/bin/env bash
#SBATCH --job-name=effdock-chemical-full-report
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
cd "${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:?}}"
: "${EFFDOCK_OUTPUT_ROOT:?set EFFDOCK_OUTPUT_ROOT}"

.venv/bin/python benchmarks/effdock/report_chemical_constraint_unit_weights_full.py \
  --output-root "$EFFDOCK_OUTPUT_ROOT" \
  --output "$EFFDOCK_OUTPUT_ROOT/aggregate.json"
