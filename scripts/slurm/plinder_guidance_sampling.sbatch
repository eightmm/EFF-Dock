#!/usr/bin/env bash
#SBATCH --job-name=effdock-plinder-guidance
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err
#SBATCH --partition=6000ada,heavy
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-159%4

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  echo "EFFDOCK_REPO_DIR or SLURM_SUBMIT_DIR is required" >&2
  exit 2
fi
cd "$repo_dir"
# shellcheck source=plinder_guidance_validation_profile.sh
source scripts/slurm/plinder_guidance_validation_profile.sh
verify_plinder_execution_contract
verify_plinder_raw_gate

output_root=${EFFDOCK_OUTPUT_ROOT:?EFFDOCK_OUTPUT_ROOT is required}
raw_root=${EFFDOCK_PLINDER_RAW_ROOT:?EFFDOCK_PLINDER_RAW_ROOT is required}
mode=${EFFDOCK_PLINDER_MODE:?EFFDOCK_PLINDER_MODE is required}
require_safe_plinder_output_root "$output_root"
task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
task_count=${SLURM_ARRAY_TASK_COUNT:?SLURM_ARRAY_TASK_COUNT is required}

case "$mode" in
  smoke)
    if [[ "$task_count" -ne "$smoke_sampling_tasks" \
      || "$task_id" -lt 0 || "$task_id" -ge "$smoke_sampling_tasks" ]]; then
      echo "PLINDER smoke requires exactly $smoke_sampling_tasks all-eta tasks" >&2
      exit 2
    fi
    eta_index=$task_id
    shard_index=0
    num_shards=1
    subset_args=(--only-id "$smoke_id")
    ;;
  full)
    if [[ "$task_count" -ne "$full_sampling_tasks" \
      || "$task_id" -lt 0 || "$task_id" -ge "$full_sampling_tasks" ]]; then
      echo "PLINDER full sampling requires exactly $full_sampling_tasks tasks" >&2
      exit 2
    fi
    eta_index=$((task_id / full_num_shards))
    shard_index=$((task_id % full_num_shards))
    num_shards=$full_num_shards
    subset_args=()
    ;;
  *) echo "EFFDOCK_PLINDER_MODE must be smoke or full" >&2; exit 2 ;;
esac
eta=${eta_values[$eta_index]}

slurm_partition=${SLURM_JOB_PARTITION:-}
case "$slurm_partition" in
  6000ada|heavy) ;;
  *)
    echo "PLINDER ODE sampling requires partition 6000ada or heavy; got ${slurm_partition:-unset}" >&2
    exit 2
    ;;
esac
if [[ "${SLURM_CPUS_PER_TASK:-}" != 4 ]]; then
  echo "PLINDER ODE sampling requires exactly 4 CPUs per task" >&2
  exit 2
fi
gpu_inventory=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
if [[ -z "$gpu_inventory" || $(wc -l <<< "$gpu_inventory") -ne 1 ]]; then
  echo "PLINDER ODE sampling requires exactly one visible GPU" >&2
  exit 2
fi
IFS=, read -r gpu_name gpu_memory_mib <<< "$gpu_inventory"
gpu_memory_mib=${gpu_memory_mib//[[:space:]]/}
case "$slurm_partition" in
  6000ada)
    if [[ "$gpu_name" != *"RTX 6000 Ada"* ]]; then
      echo "PLINDER ODE sampling on 6000ada requires RTX 6000 Ada, got $gpu_name" >&2
      exit 2
    fi
    ;;
  heavy)
    if [[ "$gpu_name" != *"H100"* && "$gpu_name" != *"RTX PRO 6000"* ]]; then
      echo "PLINDER ODE sampling on heavy requires H100 or RTX PRO 6000, got $gpu_name" >&2
      exit 2
    fi
    ;;
esac
if [[ ! "$gpu_memory_mib" =~ ^[0-9]+$ || "$gpu_memory_mib" -lt 48000 ]]; then
  echo "PLINDER ODE sampling requires at least 48000 MiB, got $gpu_memory_mib" >&2
  exit 2
fi
for required in \
  .venv/bin/python \
  scripts/run_plinder_guidance_validation.py \
  data/splits/plinder.json \
  data/plinder_processed \
  weights/effdock_geometry_ft_100k_best.pt \
  weights/effdock_confidence_extmatch_n80_s25_step42500.pt; do
  if [[ ! -e "$required" ]]; then
    echo "missing PLINDER sampling input: $required" >&2
    exit 2
  fi
done

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "protocol=$protocol_id mode=$mode eta=$eta shard=$shard_index/$num_shards id=${subset_args[*]:-all}"
printf 'gpu_runtime partition=%s name=%s memory_mib=%s\n' \
  "$slurm_partition" "$gpu_name" "$gpu_memory_mib"
printf 'frozen_git_commit=%s\n' "${EFFDOCK_GIT_COMMIT:?EFFDOCK_GIT_COMMIT is required}"
printf 'frozen_git_diff_sha256=%s\n' \
  "${EFFDOCK_GIT_DIFF_SHA256:?EFFDOCK_GIT_DIFF_SHA256 is required}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

.venv/bin/python scripts/run_plinder_guidance_validation.py \
  --raw-root "$raw_root" \
  --processed-root data/plinder_processed \
  --output-root "$output_root/sampling" \
  --run-id "$mode" \
  --eta "$eta" \
  --num-shards "$num_shards" \
  --shard-index "$shard_index" \
  "${subset_args[@]}"
