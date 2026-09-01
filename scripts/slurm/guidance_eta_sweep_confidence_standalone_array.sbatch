#!/usr/bin/env bash
#SBATCH --job-name=effdock-eta-conf-1pass
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err
#SBATCH --partition=6000ada,heavy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-127%8

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"
mkdir -p outputs/benchmarks/logs
# shellcheck source=guidance_eta_sweep_confidence_standalone_profile.sh
source scripts/slurm/guidance_eta_sweep_confidence_standalone_profile.sh

verify_manifest() {
  local label=$1
  local manifest=$2
  local expected_sha256=$3
  local actual_sha256
  if [[ ! -f "$manifest" ]]; then
    echo "missing frozen $label manifest: $manifest" >&2
    exit 2
  fi
  read -r actual_sha256 _ < <(sha256sum "$manifest")
  if [[ "$actual_sha256" != "$expected_sha256" ]] \
    || ! sha256sum --check --quiet "$manifest"; then
    echo "frozen $label manifest verification failed" >&2
    exit 2
  fi
}

execution_manifest=${EFFDOCK_STANDALONE_EXECUTION_MANIFEST:?EFFDOCK_STANDALONE_EXECUTION_MANIFEST is required}
execution_manifest_sha256=${EFFDOCK_STANDALONE_EXECUTION_MANIFEST_SHA256:?EFFDOCK_STANDALONE_EXECUTION_MANIFEST_SHA256 is required}
frozen_inputs_manifest=${EFFDOCK_STANDALONE_FROZEN_INPUTS_MANIFEST:?EFFDOCK_STANDALONE_FROZEN_INPUTS_MANIFEST is required}
frozen_inputs_manifest_sha256=${EFFDOCK_STANDALONE_FROZEN_INPUTS_MANIFEST_SHA256:?EFFDOCK_STANDALONE_FROZEN_INPUTS_MANIFEST_SHA256 is required}
verify_manifest execution "$execution_manifest" "$execution_manifest_sha256"
verify_manifest inputs "$frozen_inputs_manifest" "$frozen_inputs_manifest_sha256"

checkpoint=weights/effdock_geometry_ft_100k_best.pt
confidence_checkpoint=weights/effdock_confidence_extmatch_n80_s25_step42500.pt
config=configs/train.yaml
benchmark_input_manifest=docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json
cohort_audit=${EFFDOCK_COHORT_MANIFEST:?EFFDOCK_COHORT_MANIFEST is required}
output_root=${EFFDOCK_OUTPUT_ROOT:?EFFDOCK_OUTPUT_ROOT is required}
output_prefix_slash="$output_prefix/"
run_component=${output_root#"$output_prefix_slash"}
if [[ "$run_component" == "$output_root" || "$run_component" == */* ]] \
  || [[ ! "$run_component" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "unsafe standalone output root: $output_root" >&2
  exit 2
fi

task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
task_count=${SLURM_ARRAY_TASK_COUNT:?SLURM_ARRAY_TASK_COUNT is required}
only_id=""
raw_root="$output_root/raw"
mode=full
stress_id=${EFFDOCK_STANDALONE_STRESS_ID:-}

if [[ -n "$stress_id" ]]; then
  if [[ "$standalone_profile" != steric_high_eta_v1 || "$stress_id" != 8f4j_pho ]]; then
    echo "the frozen high-eta stress task must be steric_high_eta_v1/8f4j_pho" >&2
    exit 2
  fi
  if [[ "$task_count" -ne 1 || "$task_id" -ne 0 ]]; then
    echo "standalone stress requires the exact one-task array" >&2
    exit 2
  fi
  dataset=posebusters
  eta_index=$posebusters_smoke_eta_index
  shard_index=0
  num_shards=1
  only_id=$stress_id
  raw_root="$output_root/stress/raw"
  mode=stress
elif [[ "${EFFDOCK_STANDALONE_SMOKE_GRID:-0}" == 1 ]]; then
  if [[ "$task_count" -ne "$smoke_task_count" \
    || "$task_id" -lt 0 || "$task_id" -ge "$smoke_task_count" ]]; then
    echo "standalone smoke requires exactly $smoke_task_count tasks" >&2
    exit 2
  fi
  dataset_index=$((task_id / eta_count))
  eta_index=$((task_id % eta_count))
  shard_index=0
  num_shards=1
  raw_root="$output_root/smoke/raw"
  mode=smoke
  case "$dataset_index" in
    0) dataset=astex; only_id=1jje ;;
    1) dataset=posebusters; only_id=7b2c_tp7 ;;
    *) echo "invalid standalone-smoke task: $task_id" >&2; exit 2 ;;
  esac
else
  if [[ "$task_count" -ne "$sampling_task_count" \
    || "$task_id" -lt 0 || "$task_id" -ge "$sampling_task_count" ]]; then
    echo "standalone full sampling requires exactly $sampling_task_count tasks" >&2
    exit 2
  fi
  shard_index=$((task_id % num_shards))
  eta_index=$(((task_id / num_shards) % eta_count))
  dataset_index=$((task_id / (num_shards * eta_count)))
  case "$dataset_index" in
    0) dataset=astex ;;
    1) dataset=posebusters ;;
    *) echo "invalid dataset index: $dataset_index" >&2; exit 2 ;;
  esac
fi

eta=${eta_values[$eta_index]}
eta_tag=${eta_tags[$eta_index]}
case "$dataset" in
  astex)
    expected_count=85
    data_dir=data/external_benchmarks/data/astex_diverse_set
    centers=data/external_test/astex_reference_pocket_centers.json
    ;;
  posebusters)
    expected_count=308
    data_dir=data/external_benchmarks/data/posebusters_benchmark_set
    centers=data/external_test/posebusters_reference_pocket_centers.json
    ;;
  *) echo "dataset must be astex or posebusters, got $dataset" >&2; exit 2 ;;
esac

for required in .venv/bin/eff-dock "$checkpoint" "$confidence_checkpoint" \
  "$config" "$centers" "$benchmark_input_manifest" "$cohort_audit"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 2
  fi
done

run_name="${run_name_prefix}-${dataset}-n100-s10-${eta_tag}"
reservation_root="$output_root/.task_reservations"
mkdir -p "$reservation_root"
if ! mkdir "$reservation_root/sampling-${mode}-${task_id}"; then
  echo "refusing to rerun or overwrite standalone sampling task $mode/$task_id" >&2
  exit 2
fi

only_id_args=()
if [[ -n "$only_id" ]]; then
  only_id_args=(--only-id "$only_id")
fi

slurm_partition=${SLURM_JOB_PARTITION:-}
case "$slurm_partition" in
  6000ada|heavy) ;;
  *)
    echo "GPU sampling must run in partition 6000ada or heavy; got ${slurm_partition:-unset}" >&2
    exit 2
    ;;
esac
gpu_inventory=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
if [[ -z "$gpu_inventory" || $(wc -l <<< "$gpu_inventory") -ne 1 ]]; then
  echo "GPU sampling requires exactly one visible CUDA device" >&2
  exit 2
fi
IFS=, read -r gpu_name gpu_memory_mib <<< "$gpu_inventory"
gpu_memory_mib=${gpu_memory_mib//[[:space:]]/}
case "$slurm_partition" in
  6000ada)
    if [[ "$gpu_name" != *"RTX 6000 Ada"* ]]; then
      echo "6000ada GPU sampling requires an RTX 6000 Ada; got $gpu_name" >&2
      exit 2
    fi
    ;;
  heavy)
    if [[ "$gpu_name" != *"H100"* && "$gpu_name" != *"RTX PRO 6000"* ]]; then
      echo "heavy GPU sampling requires H100 or RTX PRO 6000; got $gpu_name" >&2
      exit 2
    fi
    ;;
esac
if [[ ! "$gpu_memory_mib" =~ ^[0-9]+$ || "$gpu_memory_mib" -lt 48000 ]]; then
  echo "GPU sampling requires at least 48000 MiB visible memory; got ${gpu_memory_mib:-unset}" >&2
  exit 2
fi

echo "protocol=$protocol_id mode=$mode dataset=$dataset eta=$eta run=$run_name shard=$shard_index/$num_shards only_id=${only_id:-none}"
printf 'gpu_runtime partition=%s name=%s memory_mib=%s\n' \
  "$slurm_partition" "$gpu_name" "$gpu_memory_mib"
printf 'frozen_git_commit=%s\n' "${EFFDOCK_GIT_COMMIT:?EFFDOCK_GIT_COMMIT is required}"
printf 'frozen_git_diff_sha256=%s\n' \
  "${EFFDOCK_GIT_DIFF_SHA256:?EFFDOCK_GIT_DIFF_SHA256 is required}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

.venv/bin/eff-dock evaluate \
  --dataset "$dataset" \
  --data-dir "$data_dir" \
  --external-dir data/external_test \
  --benchmark-input-manifest "$benchmark_input_manifest" \
  --require-full-ligand-atom-mapping \
  --pocket-centers "$centers" \
  --checkpoint "$checkpoint" \
  --confidence-checkpoint "$confidence_checkpoint" \
  --selector-profile confidence_cluster_free \
  --config "$config" \
  --device cuda \
  --output-dir "$raw_root" \
  --protocol-id "$protocol_id" \
  --run-name "$run_name" \
  --num-samples 100 \
  --num-steps 10 \
  --prior-pool-size 100 \
  --sigma 0.5 \
  --time-schedule late \
  --schedule-power 3.0 \
  --pocket-cutoff 10.0 \
  --center-jitter-sigma 0.0 \
  --refine none \
  --unified-guidance-mode normalized_drift \
  --unified-guidance-scale "$eta" \
  --unified-guidance-receptor-policy geometry_only \
  --unified-guidance-start-t 0.5 \
  --unified-guidance-ramp-power 1.0 \
  --unified-guidance-max-force 20.0 \
  --unified-guidance-max-velocity 5.0 \
  --unified-guidance-max-angular-velocity 5.0 \
  --unified-guidance-max-atom-displacement 0.25 \
  --unified-guidance-max-backtracks 8 \
  --unified-guidance-protein-shell 18.0 \
  --seed 42 \
  --num-shards "$num_shards" \
  --shard-index "$shard_index" \
  --expected-discovered-count "$expected_count" \
  --require-complete-success \
  --save-selected-poses \
  --eligibility-manifest "$cohort_audit" \
  "${only_id_args[@]}"
