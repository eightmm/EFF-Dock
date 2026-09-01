#!/usr/bin/env bash
#SBATCH --job-name=effdock-early-t0p10-audit
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
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
mkdir -p outputs/slurm

output_dir=outputs/eff-dock/early-time-t0p10-50k-v1-20260813
.venv/bin/python scripts/audit_early_time_ft_50k.py \
  --output-dir "$output_dir" \
  --config configs/train_early_time_ft_50k.yaml \
  --output "$output_dir/final_audit.json" \
  --expected-steps 50000 \
  --rollout-every 5000 \
  --expected-val-samples 1076 \
  --expected-world-size 4
