#!/usr/bin/env bash
#SBATCH --job-name=effdock-pocket-prior-gen
#SBATCH --partition=6000ada,heavy
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err

set -euo pipefail
cd "${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:?}}"
: "${EFFDOCK_OUTPUT_ROOT:?set EFFDOCK_OUTPUT_ROOT}"
runtime_venv=${EFFDOCK_RUNTIME_VENV:-.venv}
source benchmarks/effdock/slurm/pocket_prior_robustness_common.sh

task_id=${SLURM_ARRAY_TASK_ID:?}
task_count=${SLURM_ARRAY_TASK_COUNT:?}
smoke=${EFFDOCK_SMOKE:-0}
num_shards=8
if [[ "$smoke" == 1 ]]; then
  [[ "$task_count" -eq 2 ]] || { echo "smoke requires two tasks" >&2; exit 2; }
  smoke_conditions=(6 24)
  effdock_resolve_pocket_prior_condition "${smoke_conditions[$task_id]}"
  EFFDOCK_CONDITION_REPEAT=0
  dataset=astex
  expected_count=85
  data_dir=data/external_benchmarks/data/astex_diverse_set
  centers=data/external_test/astex_reference_pocket_centers.json
  shard_index=0
  active_shards=1
  only_id_args=(--only-id 1jje)
  condition_root=$(effdock_pocket_prior_condition_root "$EFFDOCK_OUTPUT_ROOT/smoke")
else
  [[ "$task_count" -eq 432 ]] || { echo "full generation requires 432 tasks" >&2; exit 2; }
  shard_index=$((task_id % num_shards))
  dataset_index=$(((task_id / num_shards) % 2))
  condition_index=$((task_id / (num_shards * 2)))
  effdock_resolve_pocket_prior_condition "$condition_index"
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
    *) exit 2 ;;
  esac
  active_shards=$num_shards
  only_id_args=()
  condition_root=$(effdock_pocket_prior_condition_root "$EFFDOCK_OUTPUT_ROOT")
fi

gpu_inventory=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
[[ $(wc -l <<< "$gpu_inventory") -eq 1 ]] || { echo "exactly one visible GPU is required" >&2; exit 2; }
IFS=, read -r gpu_name gpu_memory_mib <<< "$gpu_inventory"
gpu_memory_mib=${gpu_memory_mib//[[:space:]]/}
[[ "$gpu_memory_mib" =~ ^[0-9]+$ && "$gpu_memory_mib" -ge 48000 ]] || {
  echo "at least 48000 MiB GPU memory required, got $gpu_name/$gpu_memory_mib" >&2
  exit 2
}

checkpoint=weights/effdock_docking_early_time_t0p10_50k.pt
confidence=weights/effdock_confidence_s50_raw_refined_u70k.pt
eligibility=outputs/benchmarks/guidance_eta_sweep_v2_runs/20260801T102903Z/audit/combined.json
for path in "$runtime_venv/bin/eff-dock" "$checkpoint" "$confidence" "$eligibility" \
  configs/train.yaml docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json; do
  [[ -f "$path" ]] || { echo "missing input $path" >&2; exit 2; }
done

reservation="$EFFDOCK_OUTPUT_ROOT/.task_reservations/generation-${smoke}-${task_id}"
mkdir -p "$EFFDOCK_OUTPUT_ROOT/.task_reservations" "$condition_root/raw" outputs/benchmarks/logs
mkdir "$reservation" || { echo "duplicate task reservation: $reservation" >&2; exit 2; }
base_seed=$((42 + EFFDOCK_CONDITION_REPEAT * 100000))
run_name=$(printf 'effdock-pocket-prior-v1-%s-c%02d-s%02d-j%02d-r%d-n100-s10' \
  "$dataset" "$EFFDOCK_CONDITION_CUTOFF" "$EFFDOCK_CONDITION_SIGMA" \
  "$EFFDOCK_CONDITION_JITTER" "$EFFDOCK_CONDITION_REPEAT")

"$runtime_venv/bin/eff-dock" evaluate \
  --dataset "$dataset" --data-dir "$data_dir" --external-dir data/external_test \
  --benchmark-input-manifest docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json \
  --require-full-ligand-atom-mapping --pocket-centers "$centers" \
  --checkpoint "$checkpoint" --no-confidence \
  --selector-profile candidate_only --config configs/train.yaml \
  --device cuda --output-dir "$condition_root/raw" \
  --protocol-id EFFDOCK-POCKET-PRIOR-ROBUSTNESS-EXTENSION-V1 \
  --run-name "$run_name" --num-samples 100 --num-steps 10 \
  --prior-pool-size 100 --sigma "$EFFDOCK_CONDITION_SIGMA" \
  --time-schedule late --schedule-power 3 \
  --pocket-cutoff "$EFFDOCK_CONDITION_CUTOFF" \
  --center-jitter-sigma "$EFFDOCK_CONDITION_JITTER" --refine none \
  --unified-guidance-mode normalized_drift --unified-guidance-scale 2 \
  --unified-guidance-receptor-policy geometry_only \
  --unified-guidance-start-t 0.5 --unified-guidance-ramp-power 1 \
  --unified-guidance-max-force 20 --unified-guidance-max-velocity 5 \
  --unified-guidance-max-angular-velocity 5 \
  --unified-guidance-max-atom-displacement 0.25 \
  --unified-guidance-max-backtracks 8 --unified-guidance-protein-shell 18 \
  --seed "$base_seed" --ligand-conformer-seed 0 \
  --num-shards "$active_shards" --shard-index "$shard_index" \
  --expected-discovered-count "$expected_count" --require-complete-success \
  --save-selected-poses --eligibility-manifest "$eligibility" \
  "${only_id_args[@]}"
