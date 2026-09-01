#!/usr/bin/env bash
#SBATCH --job-name=effdock-s50-receptor-audit
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --partition=cpu_only
#SBATCH --qos=veryshort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

repo_dir=$(cd "${SLURM_SUBMIT_DIR:?}" && pwd -P)
bank_manifest="$repo_dir/outputs/eff-dock/s50-confidence-matched-runs/14cb2840f040b05127610a1f3dccd8aa5b2573c41156ff446898e2d212f1164b/full/bank_manifest.json"
output="$repo_dir/outputs/analysis/s50_receptor_graph_duplicate_audit_v1.json"
ledger="$repo_dir/outputs/analysis/s50_receptor_graph_duplicate_audit_v1.jsonl"

[[ -f "$bank_manifest" ]] || { echo "missing bank manifest: $bank_manifest" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "refusing to overwrite: $output" >&2; exit 2; }
mkdir -p "$(dirname "$output")"

export PYTHONPATH="$repo_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$repo_dir"
oms run-ledger \
  --note "Outcome-blind S50 receptor exact-coordinate duplication audit" \
  --file "$ledger" \
  --no-gate \
  --reason "Separate duplicate-inflated graphs from genuinely large pockets before changing the frozen training cohort" \
  -- .venv/bin/python scripts/audit_s50_receptor_graph_duplicates.py \
    --bank-manifest "$bank_manifest" \
    --pool-parquet "$repo_dir/data/plinder_pool.parquet" \
    --processed-root "$repo_dir/data/plinder_processed" \
    --output "$output" \
    --workers 8 \
    --pocket-cutoff 10
