#!/usr/bin/env bash
#SBATCH --job-name=effdock-direct-b1000
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err
#SBATCH --partition=6000ada
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-47%8

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"
mkdir -p outputs/benchmarks/logs

protocol_id=EFFDOCK-UNIFIED-GUIDANCE-DIRECT-DRIFT-BUDGET1000-V1
checkpoint=weights/effdock_geometry_ft_100k_best.pt
config=configs/train.yaml
benchmark_input_manifest=docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json
expected_input_sha256=99f15f557644cc51c3dd1f559b0dd97dd4259c1de3e1403fb761b7c7e079f668
output_root=${EFFDOCK_OUTPUT_ROOT:-outputs/benchmarks/guidance_direct_drift_v1}
raw_root="$output_root/raw"
cohort_audit=${EFFDOCK_COHORT_MANIFEST:-$output_root/audit/combined.json}

task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
dataset_index=$((task_id / 24))
within_dataset=$((task_id % 24))
condition_index=$((within_dataset / 8))
shard_index=$((within_dataset % 8))
num_shards=8
steps=(10 20 25)
poses=(100 50 40)
num_steps=${steps[$condition_index]}
num_samples=${poses[$condition_index]}

case "$dataset_index" in
  0)
    dataset=astex
    expected_count=85
    data_dir=data/external_benchmarks/data/astex_diverse_set
    centers=data/external_test/astex_reference_pocket_centers.json
    ;;
  1)
    dataset=posebusters
    expected_count=308
    data_dir=data/external_benchmarks/data/posebusters_benchmark_set
    centers=data/external_test/posebusters_reference_pocket_centers.json
    ;;
  *)
    echo "invalid dataset index: $dataset_index" >&2
    exit 2
    ;;
esac

for required in .venv/bin/eff-dock "$checkpoint" "$config" "$centers" \
  "$benchmark_input_manifest" "$cohort_audit"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 2
  fi
done
read -r actual_input_sha256 _ < <(sha256sum "$benchmark_input_manifest")
if [[ "$actual_input_sha256" != "$expected_input_sha256" ]]; then
  echo "frozen benchmark-input manifest SHA-256 mismatch" >&2
  exit 2
fi
.venv/bin/python -c \
  'import sys; from pathlib import Path; from effdock.workflows.guidance_budget_full_report import validate_full_cohort_audit_for_dataset; validate_full_cohort_audit_for_dataset(Path(sys.argv[1]), sys.argv[2])' \
  "$cohort_audit" "$dataset"

only_id_args=()
if [[ -n "${EFFDOCK_ONLY_ID:-}" ]]; then
  if [[ "${SLURM_ARRAY_TASK_COUNT:-0}" -ne 1 ]]; then
    echo "EFFDOCK_ONLY_ID requires one explicit array task" >&2
    exit 2
  fi
  only_id_args=(--only-id "$EFFDOCK_ONLY_ID")
  num_shards=1
  shard_index=0
  raw_root="$output_root/smoke/$EFFDOCK_ONLY_ID"
fi

echo "protocol=$protocol_id dataset=$dataset cell=N${num_samples}/S${num_steps} shard=$shard_index/$num_shards"
git rev-parse HEAD
git diff --no-ext-diff | sha256sum
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

common_args=(
  --dataset "$dataset"
  --data-dir "$data_dir"
  --external-dir data/external_test
  --benchmark-input-manifest "$benchmark_input_manifest"
  --require-full-ligand-atom-mapping
  --pocket-centers "$centers"
  --checkpoint "$checkpoint"
  --config "$config"
  --no-confidence
  --output-dir "$raw_root"
  --protocol-id "$protocol_id"
  --num-samples "$num_samples"
  --num-steps "$num_steps"
  --prior-pool-size 100
  --sigma 0.5
  --time-schedule late
  --schedule-power 3.0
  --pocket-cutoff 10.0
  --center-jitter-sigma 0.0
  --unified-guidance-mode normalized_drift
  --unified-guidance-receptor-policy geometry_only
  --unified-guidance-start-t 0.5
  --unified-guidance-ramp-power 1.0
  --unified-guidance-max-force 20.0
  --unified-guidance-max-velocity 5.0
  --unified-guidance-max-angular-velocity 5.0
  --unified-guidance-max-atom-displacement 0.25
  --unified-guidance-max-backtracks 8
  --unified-guidance-protein-shell 18.0
  --seed 42
  --num-shards "$num_shards"
  --shard-index "$shard_index"
  --expected-discovered-count "$expected_count"
  --require-complete-success
  --save-selected-poses
  --eligibility-manifest "$cohort_audit"
)

for arm in unguided direct; do
  if [[ "$arm" == direct ]]; then
    guidance_scale=0.1
  else
    guidance_scale=0.0
  fi
  run_name="effdock-guidance-direct-drift-v1-${dataset}-n${num_samples}-s${num_steps}-${arm}"
  echo "run=$run_name budget=$((num_samples * num_steps)) scale=$guidance_scale"
  .venv/bin/eff-dock evaluate \
    "${common_args[@]}" \
    --run-name "$run_name" \
    --unified-guidance-scale "$guidance_scale" \
    "${only_id_args[@]}"
done
