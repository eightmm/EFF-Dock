#!/usr/bin/env bash
# Strict additive reports for first/vina/oracle selected-pose comparison.

#SBATCH --job-name=effdock-eta-v2-selector-report
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
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"

output_root=${EFFDOCK_OUTPUT_ROOT:?EFFDOCK_OUTPUT_ROOT is required}
cohort_audit=${EFFDOCK_COHORT_MANIFEST:-$output_root/audit/combined.json}
selector_root="$output_root/posebusters_official_selectors"

for selector in first vina; do
  .venv/bin/python -m effdock.workflows.guidance_eta_sweep_selector_report \
    --input-dir "$selector_root/$selector" \
    --sampling-dir "$output_root/raw" \
    --cohort-audit "$cohort_audit" \
    --selector "$selector" \
    --output "$output_root/posebusters_${selector}_aggregate.json"
done

.venv/bin/python -m effdock.workflows.guidance_eta_sweep_selector_comparison_report \
  --oracle-input-dir "$output_root/posebusters_official" \
  --selector-input-root "$selector_root" \
  --sampling-dir "$output_root/raw" \
  --cohort-audit "$cohort_audit" \
  --oracle-report "$output_root/posebusters_aggregate.json" \
  --first-report "$output_root/posebusters_first_aggregate.json" \
  --vina-report "$output_root/posebusters_vina_aggregate.json" \
  --parent-sampling-report "$output_root/aggregate.json" \
  --output "$output_root/posebusters_selector_comparison.json"
