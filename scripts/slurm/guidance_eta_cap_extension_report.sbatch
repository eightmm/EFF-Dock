#!/usr/bin/env bash
#SBATCH --job-name=effdock-eta-cap-report
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
cd "${repo_dir:?repository is required}"
.venv/bin/python scripts/report_guidance_eta_cap_extension.py \
  --output-root "${EFFDOCK_OUTPUT_ROOT:?output root is required}" \
  --reference-root "${EFFDOCK_ETA_CAP_REFERENCE_ROOT:?reference root is required}" \
  --output-json "$EFFDOCK_OUTPUT_ROOT/report/aggregate.json" \
  --output-md "$EFFDOCK_OUTPUT_ROOT/report/RESULTS.md"
