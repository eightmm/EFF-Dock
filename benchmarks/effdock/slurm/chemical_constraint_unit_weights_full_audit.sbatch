#!/usr/bin/env bash
#SBATCH --job-name=effdock-chemical-full-audit
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2

set -euo pipefail
cd "${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:?}}"
: "${EFFDOCK_OUTPUT_ROOT:?set EFFDOCK_OUTPUT_ROOT}"
mkdir -p "$EFFDOCK_OUTPUT_ROOT/audit" outputs/benchmarks/logs

case "${SLURM_ARRAY_TASK_ID:?}" in
  0)
    dataset=astex
    data_dir=data/external_benchmarks/data/astex_diverse_set
    centers=data/external_test/astex_reference_pocket_centers.json
    ;;
  1)
    dataset=posebusters
    data_dir=data/external_benchmarks/data/posebusters_benchmark_set
    centers=data/external_test/posebusters_reference_pocket_centers.json
    ;;
  *) exit 2 ;;
esac

.venv/bin/python -m effdock.workflows.guidance_coverage_audit \
  --dataset "$dataset" \
  --data-dir "$data_dir" \
  --external-dir data/external_test \
  --benchmark-input-manifest docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json \
  --pocket-centers "$centers" \
  --receptor-policy geometry_only \
  --seed 42 \
  --pocket-cutoff 10 \
  --legacy-eligibility-manifest docs/GUIDANCE_BUDGET1000_ELIGIBILITY.json \
  --protocol-id EFFDOCK-UNIFIED-GUIDANCE-BUDGET1000-FULL-V2 \
  --require-complete-success \
  --output "$EFFDOCK_OUTPUT_ROOT/audit/$dataset.json"
