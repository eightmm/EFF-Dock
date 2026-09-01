#!/usr/bin/env bash
#SBATCH --job-name=effdock-plinder-pb
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err
#SBATCH --partition=cpu_only
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --array=0-159%16

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

if [[ "${SLURM_JOB_PARTITION:-}" != "cpu_only" ]]; then
  echo "official PoseBusters must run on cpu_only" >&2
  exit 2
fi
output_root=${EFFDOCK_OUTPUT_ROOT:?EFFDOCK_OUTPUT_ROOT is required}
mode=${EFFDOCK_PLINDER_MODE:?EFFDOCK_PLINDER_MODE is required}
require_safe_plinder_output_root "$output_root"
task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
task_count=${SLURM_ARRAY_TASK_COUNT:?SLURM_ARRAY_TASK_COUNT is required}

case "$mode" in
  smoke)
    if [[ "$task_count" -ne "$smoke_pb_tasks" \
      || "$task_id" -lt 0 || "$task_id" -ge "$smoke_pb_tasks" ]]; then
      echo "PLINDER PoseBusters smoke requires exactly $smoke_pb_tasks tasks" >&2
      exit 2
    fi
    eta_index=$task_id
    shard_index=0
    num_shards=1
    ;;
  full)
    if [[ "$task_count" -ne "$full_pb_tasks" \
      || "$task_id" -lt 0 || "$task_id" -ge "$full_pb_tasks" ]]; then
      echo "PLINDER full PoseBusters requires exactly $full_pb_tasks tasks" >&2
      exit 2
    fi
    eta_index=$((task_id / full_num_shards))
    shard_index=$((task_id % full_num_shards))
    num_shards=$full_num_shards
    ;;
  *) echo "EFFDOCK_PLINDER_MODE must be smoke or full" >&2; exit 2 ;;
esac
eta=${eta_values[$eta_index]}
audit="$output_root/audit/$mode.json"
if [[ ! -f "$audit" ]]; then
  echo "missing passed PLINDER sampling audit: $audit" >&2
  exit 2
fi
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "protocol=$protocol_id mode=$mode eta=$eta selector=confidence shard=$shard_index/$num_shards"

.venv/bin/python scripts/run_plinder_guidance_posebusters.py \
  --sampling-root "$output_root/sampling" \
  --audit "$audit" \
  --split-file data/splits/plinder.json \
  --output-root "$output_root/posebusters" \
  --mode "$mode" \
  --eta "$eta" \
  --num-shards "$num_shards" \
  --shard-index "$shard_index"
