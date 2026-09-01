#!/usr/bin/env bash
#SBATCH --job-name=effdock-fk-sde-report
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err
#SBATCH --partition=cpu_only
#SBATCH --qos=veryshort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"

output_root=${EFFDOCK_FK_SDE_OUTPUT_ROOT:?EFFDOCK_FK_SDE_OUTPUT_ROOT is required}
.venv/bin/python -m effdock.workflows.fk_sde_astex_report report \
  --input-root "$output_root/raw" \
  --cohort-manifest docs/GUIDANCE_BUDGET1000_FULL_COHORT.json \
  --output-json "$output_root/report.json" \
  --output-markdown "$output_root/RESULTS.md"
