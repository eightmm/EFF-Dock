#!/usr/bin/env bash
#SBATCH --job-name=effdock-guidance-b1000
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-47%8

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"
mkdir -p outputs/benchmarks/logs

protocol_id=EFFDOCK-UNIFIED-GUIDANCE-BUDGET1000-V1
eligibility=docs/GUIDANCE_BUDGET1000_ELIGIBILITY.json
checkpoint=weights/effdock_geometry_ft_100k_best.pt
config=configs/train.yaml
output_root=${EFFDOCK_OUTPUT_ROOT:-outputs/benchmarks/guidance_budget1000_v1/raw}

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
    data_dir=data/external_benchmarks/data/astex_diverse_set
    centers=data/external_test/astex_reference_pocket_centers.json
    ;;
  1)
    dataset=posebusters
    data_dir=data/external_benchmarks/data/posebusters_benchmark_set
    centers=data/external_test/posebusters_reference_pocket_centers.json
    ;;
  *)
    echo "invalid dataset index: $dataset_index" >&2
    exit 2
    ;;
esac

mapfile -t eligible_ids < <(
  .venv/bin/python -c \
    'import json,sys; d=json.load(open(sys.argv[1])); print(*d["datasets"][sys.argv[2]]["eligible_ids"], sep="\n")' \
    "$eligibility" "$dataset"
)
if [[ ${#eligible_ids[@]} -eq 0 ]]; then
  echo "eligibility cohort is empty for $dataset" >&2
  exit 2
fi

only_id_args=()
if [[ -n "${EFFDOCK_ONLY_ID:-}" ]]; then
  only_id_args=(--only-id "$EFFDOCK_ONLY_ID")
  num_shards=1
  shard_index=0
else
  for complex_id in "${eligible_ids[@]}"; do
    only_id_args+=(--only-id "$complex_id")
  done
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

common_args=(
  --dataset "$dataset"
  --data-dir "$data_dir"
  --external-dir data/external_test
  --pocket-centers "$centers"
  --checkpoint "$checkpoint"
  --config "$config"
  --no-confidence
  --output-dir "$output_root"
  --protocol-id "$protocol_id"
  --eligibility-manifest "$eligibility"
  --num-samples "$num_samples"
  --num-steps "$num_steps"
  --prior-pool-size 100
  --sigma 0.5
  --time-schedule late
  --schedule-power 3.0
  --pocket-cutoff 10.0
  --center-jitter-sigma 0.0
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
  --save-selected-poses
)

for arm in unguided guided; do
  if [[ "$arm" == guided ]]; then
    guidance_scale=0.1
  else
    guidance_scale=0.0
  fi
  run_name="effdock-guidance-budget1000-v1-${dataset}-n${num_samples}-s${num_steps}-${arm}"
  echo "run=$run_name shard=$shard_index/$num_shards budget=$((num_samples * num_steps))"
  .venv/bin/eff-dock evaluate \
    "${common_args[@]}" \
    --run-name "$run_name" \
    --unified-guidance-scale "$guidance_scale" \
    "${only_id_args[@]}"
done
