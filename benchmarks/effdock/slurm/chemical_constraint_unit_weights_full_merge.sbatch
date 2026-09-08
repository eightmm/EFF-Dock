#!/usr/bin/env bash
#SBATCH --job-name=effdock-chemical-full-merge
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00

set -euo pipefail
cd "${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:?}}"
: "${EFFDOCK_OUTPUT_ROOT:?set EFFDOCK_OUTPUT_ROOT}"

.venv/bin/python -m effdock.workflows.guidance_coverage_audit \
  --merge-audit "$EFFDOCK_OUTPUT_ROOT/audit/astex.json" \
  --merge-audit "$EFFDOCK_OUTPUT_ROOT/audit/posebusters.json" \
  --require-complete-success \
  --protocol-id EFFDOCK-UNIFIED-GUIDANCE-BUDGET1000-FULL-V2 \
  --output "$EFFDOCK_OUTPUT_ROOT/audit/combined.json"
