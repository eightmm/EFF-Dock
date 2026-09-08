#!/usr/bin/env bash
#SBATCH --job-name=effdock-chemical-unit1-report
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
fi
cd "$repo_dir"
mkdir -p outputs/benchmarks/logs

.venv/bin/python benchmarks/effdock/report_chemical_constraint_guidance_smoke.py \
  --input-dir outputs/benchmarks/chemical_constraint_unit_weights_v1/raw \
  --run-prefix effdock-chemical-unit1 \
  --arms physical_interaction_1,all_three_1 \
  --output outputs/benchmarks/chemical_constraint_unit_weights_v1/report.json
