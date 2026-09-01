#!/usr/bin/env bash
#SBATCH --job-name=effdock-eta-v2-report
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

output_root=${EFFDOCK_OUTPUT_ROOT:-outputs/benchmarks/guidance_eta_sweep_v2}
cohort_audit=${EFFDOCK_COHORT_MANIFEST:-$output_root/audit/combined.json}
.venv/bin/python -m effdock.workflows.guidance_eta_sweep_report \
  --input-dir "$output_root/raw" \
  --cohort-audit "$cohort_audit" \
  --output "$output_root/aggregate.json"
.venv/bin/python -m effdock.workflows.guidance_eta_sweep_posebusters_report \
  --input-dir "$output_root/posebusters_official" \
  --sampling-dir "$output_root/raw" \
  --cohort-audit "$cohort_audit" \
  --output "$output_root/posebusters_aggregate.json"
