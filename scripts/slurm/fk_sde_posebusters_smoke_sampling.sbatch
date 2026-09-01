#!/usr/bin/env bash
#SBATCH --job-name=effdock-fk-sde-pb-smoke
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err
#SBATCH --partition=6000ada,heavy
#SBATCH --qos=veryshort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:1

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"

output_root=${EFFDOCK_FK_SDE_PB_OUTPUT_ROOT:?EFFDOCK_FK_SDE_PB_OUTPUT_ROOT is required}
mkdir -p "$output_root/smoke/raw" "$output_root/.task_reservations"
if ! mkdir "$output_root/.task_reservations/smoke-sampling"; then
  echo "refusing duplicate smoke-sampling reservation" >&2
  exit 2
fi

for required in \
  .venv/bin/eff-dock \
  configs/train.yaml \
  weights/effdock_geometry_ft_100k_best.pt \
  weights/effdock_confidence_extmatch_n80_s25_step42500.pt \
  docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json \
  docs/GUIDANCE_BUDGET1000_FULL_COHORT.json \
  data/external_test/posebusters_reference_pocket_centers.json \
  data/external_benchmarks/data/posebusters_benchmark_set; do
  if [[ ! -e "$required" ]]; then
    echo "missing frozen smoke input: $required" >&2
    exit 2
  fi
done

gpu_inventory=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
if [[ -z "$gpu_inventory" || $(wc -l <<< "$gpu_inventory") -ne 1 ]]; then
  echo "smoke requires exactly one visible CUDA device" >&2
  exit 2
fi
IFS=, read -r gpu_name gpu_memory_mib <<< "$gpu_inventory"
gpu_memory_mib=${gpu_memory_mib//[[:space:]]/}
if [[ ! "$gpu_memory_mib" =~ ^[0-9]+$ || "$gpu_memory_mib" -lt 48000 ]]; then
  echo "smoke requires at least 48000 MiB visible memory" >&2
  exit 2
fi
printf 'protocol=%s smoke_id=%s gpu=%s memory_mib=%s\n' \
  EFFDOCK-FK-TRANSLATION-SDE-POSEBUSTERS-V1 7b2c_tp7 "$gpu_name" "$gpu_memory_mib"

run_arm() {
  local arm=$1
  local slug=$2
  shift 2
  mkdir -p "$output_root/smoke/raw/$arm"
  .venv/bin/eff-dock evaluate \
    --dataset posebusters \
    --data-dir data/external_benchmarks/data/posebusters_benchmark_set \
    --external-dir data/external_test \
    --benchmark-input-manifest docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json \
    --eligibility-manifest docs/GUIDANCE_BUDGET1000_FULL_COHORT.json \
    --require-full-ligand-atom-mapping \
    --pocket-centers data/external_test/posebusters_reference_pocket_centers.json \
    --checkpoint weights/effdock_geometry_ft_100k_best.pt \
    --confidence-checkpoint weights/effdock_confidence_extmatch_n80_s25_step42500.pt \
    --selector-profile confidence_cluster_free \
    --config configs/train.yaml \
    --device cuda \
    --protocol-id EFFDOCK-FK-TRANSLATION-SDE-POSEBUSTERS-V1 \
    --output-dir "$output_root/smoke/raw/$arm" \
    --run-name "effdock-fk-sde-posebusters-v1-smoke-$slug" \
    --num-samples 40 \
    --num-steps 25 \
    --prior-pool-size 100 \
    --sigma 0.5 \
    --time-schedule late \
    --schedule-power 3.0 \
    --pocket-cutoff 10.0 \
    --center-jitter-sigma 0.0 \
    --seed 42 \
    --num-shards 1 \
    --shard-index 0 \
    --only-id 7b2c_tp7 \
    --expected-discovered-count 308 \
    --require-complete-success \
    --save-selected-poses \
    --refine none \
    --unified-guidance-receptor-policy geometry_only \
    --fk-constraint-beta 0.01 \
    --fk-resample-times 0.3,0.6,0.8 \
    --fk-resample-method systematic \
    "$@"
}

export PYTHONHASHSEED=0
run_arm fk_ode fk-ode
run_arm fk_sde fk-sde --translation-sde-base-sigma 0.3
printf 'status=complete\n' > "$output_root/.task_reservations/smoke-sampling/status"
