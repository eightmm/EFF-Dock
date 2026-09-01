#!/usr/bin/env bash
#SBATCH --job-name=effdock-eta-cap-audit
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
.venv/bin/python scripts/audit_guidance_eta_cap_extension.py \
  --output-root "${EFFDOCK_OUTPUT_ROOT:?output root is required}" \
  --mode "${EFFDOCK_ETA_CAP_AUDIT_MODE:?audit mode is required}" \
  --input-manifest docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json \
  --output "$EFFDOCK_OUTPUT_ROOT/audit/$EFFDOCK_ETA_CAP_AUDIT_MODE.json"
