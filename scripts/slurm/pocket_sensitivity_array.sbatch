#!/usr/bin/env bash
#SBATCH --job-name=effdock-pocket-sens
#SBATCH --output=outputs/benchmarks/pocket_sensitivity_n80_s25_v2/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/pocket_sensitivity_n80_s25_v2/logs/%x-%A_%a.err
#SBATCH --partition=6000ada
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=03:30:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-23%8
#SBATCH --requeue

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"

protocol_id=EFFDOCK-POCKET-SENSITIVITY-N80-S25-V2
output_root=${EFFDOCK_OUTPUT_ROOT:-outputs/benchmarks/pocket_sensitivity_n80_s25_v2}
mkdir -p "$output_root/logs" "$output_root/raw"

cutoffs=(6 8 10 12)
jitters=(0 1 2)
conditions_per_dataset=12
dataset_index=$((SLURM_ARRAY_TASK_ID / conditions_per_dataset))
condition_index=$((SLURM_ARRAY_TASK_ID % conditions_per_dataset))
cutoff=${cutoffs[$((condition_index / 3))]}
jitter=${jitters[$((condition_index % 3))]}

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

run_suffix=${EFFDOCK_RUN_SUFFIX:-}
run_name="effdock-pocket-sens-n80-s25-v2-${dataset}-pc${cutoff}-cj${jitter}${run_suffix}"
limit_args=()
if [[ -n "${EFFDOCK_LIMIT:-}" ]]; then
  limit_args=(--limit "$EFFDOCK_LIMIT")
fi
only_id_args=()
if [[ -n "${EFFDOCK_ONLY_IDS:-}" ]]; then
  IFS=',' read -r -a only_ids <<< "$EFFDOCK_ONLY_IDS"
  for complex_id in "${only_ids[@]}"; do
    only_id_args+=(--only-id "$complex_id")
  done
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
.venv/bin/eff-dock evaluate \
  --dataset "$dataset" \
  --data-dir "$data_dir" \
  --external-dir data/external_test \
  --pocket-centers "$centers" \
  --checkpoint weights/effdock_geometry_ft_100k_best.pt \
  --confidence-checkpoint weights/effdock_confidence_extmatch_n80_s25_step42500.pt \
  --config configs/train.yaml \
  --output-dir "$output_root/raw" \
  --run-name "$run_name" \
  --protocol-id "$protocol_id" \
  --num-samples 80 \
  --num-steps 25 \
  --sigma 0.5 \
  --time-schedule late \
  --schedule-power 3.0 \
  --pocket-cutoff "$cutoff" \
  --center-jitter-sigma "$jitter" \
  --seed 42 \
  --num-shards 1 \
  --shard-index 0 \
  "${limit_args[@]}" \
  "${only_id_args[@]}"
