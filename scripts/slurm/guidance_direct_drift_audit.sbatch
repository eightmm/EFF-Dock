#!/usr/bin/env bash
#SBATCH --job-name=effdock-direct-audit
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"
mkdir -p outputs/benchmarks/logs

output_root=${EFFDOCK_OUTPUT_ROOT:-outputs/benchmarks/guidance_direct_drift_v1}
audit_root="$output_root/audit"
mkdir -p "$audit_root"
benchmark_input_manifest=docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json
expected_input_sha256=99f15f557644cc51c3dd1f559b0dd97dd4259c1de3e1403fb761b7c7e079f668
read -r actual_input_sha256 _ < <(sha256sum "$benchmark_input_manifest")
if [[ "$actual_input_sha256" != "$expected_input_sha256" ]]; then
  echo "frozen benchmark-input manifest SHA-256 mismatch" >&2
  exit 2
fi

case "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}" in
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
  *)
    echo "invalid audit task: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac

output="$audit_root/$dataset.json"
echo "dataset=$dataset output=$output"
git rev-parse HEAD
git diff --no-ext-diff | sha256sum

.venv/bin/python -m effdock.workflows.guidance_coverage_audit \
  --dataset "$dataset" \
  --data-dir "$data_dir" \
  --external-dir data/external_test \
  --benchmark-input-manifest "$benchmark_input_manifest" \
  --pocket-centers "$centers" \
  --receptor-policy geometry_only \
  --seed 42 \
  --pocket-cutoff 10.0 \
  --legacy-eligibility-manifest docs/GUIDANCE_BUDGET1000_ELIGIBILITY.json \
  --protocol-id EFFDOCK-UNIFIED-GUIDANCE-BUDGET1000-FULL-V2 \
  --require-complete-success \
  --output "$output"
