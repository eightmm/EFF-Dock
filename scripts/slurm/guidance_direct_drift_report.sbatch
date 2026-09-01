#!/usr/bin/env bash
#SBATCH --job-name=effdock-direct-report
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"

output_root=${EFFDOCK_OUTPUT_ROOT:-outputs/benchmarks/guidance_direct_drift_v1}
.venv/bin/python -m effdock.workflows.guidance_direct_drift_report \
  --input-dir "$output_root/raw" \
  --cohort-audit "$output_root/audit/combined.json" \
  --output "$output_root/aggregate.json"
.venv/bin/python -m effdock.workflows.guidance_direct_drift_posebusters_report \
  --input-dir "$output_root/posebusters_official" \
  --cohort-audit "$output_root/audit/combined.json" \
  --output "$output_root/posebusters_aggregate.json"
