#!/usr/bin/env bash
#SBATCH --job-name=effdock-fk-sde-astex
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err
#SBATCH --partition=6000ada
#SBATCH --qos=veryshort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-31%8

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"

output_root=${EFFDOCK_FK_SDE_OUTPUT_ROOT:?EFFDOCK_FK_SDE_OUTPUT_ROOT is required}
task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
arm_index=$((task_id / 8))
shard_index=$((task_id % 8))

case "$arm_index" in
  0) arm=ode; slug=ode; arm_args=() ;;
  1) arm=sde; slug=sde; arm_args=(--translation-sde-base-sigma 0.3) ;;
  2)
    arm=fk_ode
    slug=fk-ode
    arm_args=(
      --fk-constraint-beta 0.01
      --fk-resample-times 0.3,0.6,0.8
      --fk-resample-method systematic
    )
    ;;
  3)
    arm=fk_sde
    slug=fk-sde
    arm_args=(
      --translation-sde-base-sigma 0.3
      --fk-constraint-beta 0.01
      --fk-resample-times 0.3,0.6,0.8
      --fk-resample-method systematic
    )
    ;;
  *) echo "invalid FK-SDE arm index: $arm_index" >&2; exit 2 ;;
esac

for required in \
  .venv/bin/eff-dock \
  configs/train.yaml \
  weights/effdock_geometry_ft_100k_best.pt \
  weights/effdock_confidence_extmatch_n80_s25_step42500.pt \
  docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json \
  docs/GUIDANCE_BUDGET1000_FULL_COHORT.json \
  data/external_test/astex_reference_pocket_centers.json \
  data/external_benchmarks/data/astex_diverse_set; do
  if [[ ! -e "$required" ]]; then
    echo "missing frozen FK-SDE Astex input: $required" >&2
    exit 2
  fi
done

mkdir -p "$output_root/raw/$arm" "$output_root/tasks"
if ! mkdir "$output_root/tasks/$task_id"; then
  echo "refusing duplicate FK-SDE task reservation: $task_id" >&2
  exit 2
fi

gpu_inventory=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
if [[ -z "$gpu_inventory" || $(wc -l <<< "$gpu_inventory") -ne 1 ]]; then
  echo "FK-SDE Astex task requires exactly one visible CUDA device" >&2
  exit 2
fi
printf 'protocol=%s arm=%s shard=%s/8 gpu=%s\n' \
  EFFDOCK-FK-TRANSLATION-SDE-ASTEX-V1 "$arm" "$shard_index" "$gpu_inventory"

export PYTHONHASHSEED=0
.venv/bin/eff-dock evaluate \
  --dataset astex \
  --data-dir data/external_benchmarks/data/astex_diverse_set \
  --external-dir data/external_test \
  --benchmark-input-manifest docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json \
  --eligibility-manifest docs/GUIDANCE_BUDGET1000_FULL_COHORT.json \
  --require-full-ligand-atom-mapping \
  --pocket-centers data/external_test/astex_reference_pocket_centers.json \
  --checkpoint weights/effdock_geometry_ft_100k_best.pt \
  --confidence-checkpoint weights/effdock_confidence_extmatch_n80_s25_step42500.pt \
  --selector-profile confidence_cluster_free \
  --config configs/train.yaml \
  --device cuda \
  --protocol-id EFFDOCK-FK-TRANSLATION-SDE-ASTEX-V1 \
  --output-dir "$output_root/raw/$arm" \
  --run-name "effdock-fk-sde-astex-v1-$slug" \
  --num-samples 40 \
  --num-steps 25 \
  --prior-pool-size 100 \
  --sigma 0.5 \
  --time-schedule late \
  --schedule-power 3.0 \
  --pocket-cutoff 10.0 \
  --center-jitter-sigma 0.0 \
  --seed 42 \
  --num-shards 8 \
  --shard-index "$shard_index" \
  --expected-discovered-count 85 \
  --require-complete-success \
  --save-selected-poses \
  --refine none \
  --unified-guidance-receptor-policy geometry_only \
  "${arm_args[@]}"

printf 'status=complete\n' > "$output_root/tasks/$task_id/status"
