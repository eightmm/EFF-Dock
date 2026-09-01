#!/usr/bin/env bash
#SBATCH --job-name=effdock-fk-sde-pb-smokeaudit
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err
#SBATCH --partition=cpu_only
#SBATCH --qos=veryshort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"

output_root=${EFFDOCK_FK_SDE_PB_OUTPUT_ROOT:?EFFDOCK_FK_SDE_PB_OUTPUT_ROOT is required}
manifest="$output_root/smoke/manifest.json"
output="$output_root/smoke/audit.json"
read -r manifest_sha256 _ < <(sha256sum "$manifest")
.venv/bin/python -m effdock.workflows.fk_sde_posebusters_report smoke-audit \
  --manifest "$manifest" \
  --manifest-sha256 "$manifest_sha256" \
  --official-root "$output_root/official" \
  --output "$output"
