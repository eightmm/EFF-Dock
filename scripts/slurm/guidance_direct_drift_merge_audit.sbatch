#!/usr/bin/env bash
#SBATCH --job-name=effdock-direct-audit-merge
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"

output_root=${EFFDOCK_OUTPUT_ROOT:-outputs/benchmarks/guidance_direct_drift_v1}
audit_root="$output_root/audit"
.venv/bin/python -m effdock.workflows.guidance_coverage_audit \
  --merge-audit "$audit_root/astex.json" \
  --merge-audit "$audit_root/posebusters.json" \
  --require-complete-success \
  --protocol-id EFFDOCK-UNIFIED-GUIDANCE-BUDGET1000-FULL-V2 \
  --output "$audit_root/combined.json"
