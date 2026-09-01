#!/usr/bin/env bash
#SBATCH --job-name=effdock-eta-cap
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err
#SBATCH --partition=6000ada,heavy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-31%4

set -euo pipefail
repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
cd "${repo_dir:?EFFDOCK_REPO_DIR or SLURM_SUBMIT_DIR is required}"
mkdir -p outputs/benchmarks/logs
protocol_id=EFFDOCK-UNIFIED-GUIDANCE-ETA-CAP-EXTENSION-V1
etas=(2.5 3.0)
eta_tags=(eta2500 eta3000)
eta_count=${#etas[@]}
num_shards=8
task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
task_count=${SLURM_ARRAY_TASK_COUNT:?SLURM_ARRAY_TASK_COUNT is required}
output_root=${EFFDOCK_OUTPUT_ROOT:?EFFDOCK_OUTPUT_ROOT is required}
mode=full
raw_root="$output_root/raw"
only_id=""

if [[ "${EFFDOCK_ETA_CAP_SMOKE:-0}" == 1 ]]; then
  expected_tasks=$((2 * eta_count))
  [[ "$task_count" -eq "$expected_tasks" ]] || { echo "smoke requires $expected_tasks tasks" >&2; exit 2; }
  dataset_index=$((task_id / eta_count))
  eta_index=$((task_id % eta_count))
  shard_index=0; active_shards=1; mode=smoke; raw_root="$output_root/smoke/raw"
  case "$dataset_index" in
    0) dataset=astex; only_id=1jje ;;
    1) dataset=posebusters; only_id=7b2c_tp7 ;;
    *) echo "invalid smoke task $task_id" >&2; exit 2 ;;
  esac
else
  expected_tasks=$((2 * eta_count * num_shards))
  [[ "$task_count" -eq "$expected_tasks" ]] || { echo "full run requires $expected_tasks tasks" >&2; exit 2; }
  shard_index=$((task_id % num_shards))
  eta_index=$(((task_id / num_shards) % eta_count))
  dataset_index=$((task_id / (num_shards * eta_count)))
  active_shards=$num_shards
  case "$dataset_index" in 0) dataset=astex ;; 1) dataset=posebusters ;; *) exit 2 ;; esac
fi

eta=${etas[$eta_index]}; eta_tag=${eta_tags[$eta_index]}
case "$dataset" in
  astex) expected_count=85; data_dir=data/external_benchmarks/data/astex_diverse_set; centers=data/external_test/astex_reference_pocket_centers.json ;;
  posebusters) expected_count=308; data_dir=data/external_benchmarks/data/posebusters_benchmark_set; centers=data/external_test/posebusters_reference_pocket_centers.json ;;
esac
checkpoint=weights/effdock_geometry_ft_100k_best.pt
confidence_checkpoint=weights/effdock_confidence_extmatch_n80_s25_step42500.pt
config=configs/train.yaml
benchmark_input_manifest=docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json
cohort_audit=${EFFDOCK_COHORT_MANIFEST:?EFFDOCK_COHORT_MANIFEST is required}
for required in .venv/bin/eff-dock "$checkpoint" "$confidence_checkpoint" "$config" "$centers" "$benchmark_input_manifest" "$cohort_audit"; do
  [[ -e "$required" ]] || { echo "missing $required" >&2; exit 2; }
done
mkdir -p "$output_root/.task_reservations"
mkdir "$output_root/.task_reservations/sampling-${mode}-${task_id}" || { echo "duplicate task reservation" >&2; exit 2; }
gpu_inventory=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
[[ $(wc -l <<< "$gpu_inventory") -eq 1 ]] || { echo "one visible GPU required" >&2; exit 2; }
IFS=, read -r gpu_name gpu_memory_mib <<< "$gpu_inventory"; gpu_memory_mib=${gpu_memory_mib//[[:space:]]/}
[[ "$gpu_memory_mib" =~ ^[0-9]+$ && "$gpu_memory_mib" -ge 48000 ]] || { echo "at least 48000 MiB GPU required" >&2; exit 2; }
run_name="effdock-guidance-eta-cap-extension-v1-${dataset}-n100-s10-${eta_tag}"
only_id_args=(); [[ -z "$only_id" ]] || only_id_args=(--only-id "$only_id")
echo "protocol=$protocol_id mode=$mode dataset=$dataset eta=$eta sigma=0.5 shard=$shard_index/$active_shards gpu=$gpu_name"
.venv/bin/eff-dock evaluate \
  --dataset "$dataset" --data-dir "$data_dir" --external-dir data/external_test \
  --benchmark-input-manifest "$benchmark_input_manifest" --require-full-ligand-atom-mapping \
  --pocket-centers "$centers" --checkpoint "$checkpoint" --confidence-checkpoint "$confidence_checkpoint" \
  --selector-profile confidence_cluster_free --config "$config" --device cuda --output-dir "$raw_root" \
  --protocol-id "$protocol_id" --run-name "$run_name" --num-samples 100 --num-steps 10 \
  --prior-pool-size 100 --sigma 0.5 --time-schedule late --schedule-power 3.0 \
  --pocket-cutoff 10.0 --center-jitter-sigma 0.0 --refine none \
  --unified-guidance-mode normalized_drift --unified-guidance-scale "$eta" \
  --unified-guidance-receptor-policy geometry_only --unified-guidance-start-t 0.5 \
  --unified-guidance-ramp-power 1.0 --unified-guidance-max-force 20.0 \
  --unified-guidance-max-velocity 5.0 --unified-guidance-max-angular-velocity 5.0 \
  --unified-guidance-max-atom-displacement 0.25 --unified-guidance-max-backtracks 8 \
  --unified-guidance-protein-shell 18.0 --seed 42 --num-shards "$active_shards" \
  --shard-index "$shard_index" --expected-discovered-count "$expected_count" \
  --require-complete-success --save-selected-poses --eligibility-manifest "$cohort_audit" \
  "${only_id_args[@]}"
