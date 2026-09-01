#!/usr/bin/env bash
#SBATCH --job-name=effdock-fk-sde-pb-manifest
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err
#SBATCH --partition=cpu_only
#SBATCH --qos=veryshort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"

output_root=${EFFDOCK_FK_SDE_PB_OUTPUT_ROOT:?EFFDOCK_FK_SDE_PB_OUTPUT_ROOT is required}
mode=${EFFDOCK_FK_SDE_PB_MODE:?EFFDOCK_FK_SDE_PB_MODE is required}
case "$mode" in
  smoke)
    input_root="$output_root/smoke/raw"
    output="$output_root/smoke/manifest.json"
    ;;
  full)
    input_root="$output_root/raw"
    output="$output_root/manifest.json"
    ;;
  *) echo "manifest mode must be smoke or full" >&2; exit 2 ;;
esac
if [[ -e "$output" ]]; then
  echo "refusing to overwrite manifest: $output" >&2
  exit 2
fi

.venv/bin/python -m effdock.workflows.fk_sde_posebusters_report manifest \
  --input-root "$input_root" \
  --cohort-manifest docs/GUIDANCE_BUDGET1000_FULL_COHORT.json \
  --mode "$mode" \
  --output "$output"
