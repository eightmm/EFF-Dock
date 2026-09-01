#!/usr/bin/env bash
#SBATCH --job-name=effdock-sigma-pb
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err
#SBATCH --partition=cpu_only,test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=8G
#SBATCH --time=01:30:00

set -euo pipefail
repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
cd "${repo_dir:?EFFDOCK_REPO_DIR or SLURM_SUBMIT_DIR is required}"

sweep_root=${EFFDOCK_SIGMA_SWEEP_ROOT:?EFFDOCK_SIGMA_SWEEP_ROOT is required}
reference_root=${EFFDOCK_SIGMA_REFERENCE_ROOT:?EFFDOCK_SIGMA_REFERENCE_ROOT is required}
mode=${EFFDOCK_PB_MODE:?EFFDOCK_PB_MODE is required}
task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
task_count=${SLURM_ARRAY_TASK_COUNT:?SLURM_ARRAY_TASK_COUNT is required}
sigmas=(0.5 1.0 2.0 3.0 4.0)
sigma_tags=(sigma0500 sigma1000 sigma2000 sigma3000 sigma4000)
selectors=(confidence)

if [[ "$mode" == smoke ]]; then
  [[ "$task_count" -eq 10 ]] || { echo "smoke requires exactly 10 tasks" >&2; exit 2; }
  sigma_index=$((task_id % 5))
  dataset_index=$(((task_id / 5) % 2))
  selector_index=0
  num_shards=1
  shard_index=0
elif [[ "$mode" == full ]]; then
  [[ "$task_count" -eq 80 ]] || { echo "full requires exactly 80 tasks" >&2; exit 2; }
  shard_index=$((task_id % 8))
  sigma_index=$(((task_id / 8) % 5))
  dataset_index=$(((task_id / 40) % 2))
  selector_index=0
  num_shards=8
else
  echo "EFFDOCK_PB_MODE must be smoke or full" >&2
  exit 2
fi

sigma=${sigmas[$sigma_index]}
sigma_tag=${sigma_tags[$sigma_index]}
selector=${selectors[$selector_index]}
case "$dataset_index" in
  0) dataset=astex; expected_count=85; smoke_id=1jje ;;
  1) dataset=posebusters; expected_count=308; smoke_id=7b2c_tp7 ;;
  *) echo "invalid dataset index" >&2; exit 2 ;;
esac

if [[ "$sigma" == 0.5 ]]; then
  sampling_dir="$reference_root/raw"
  run_name="effdock-guidance-steric-high-eta-v1-${dataset}-n100-s10-eta2000"
else
  sampling_dir="$sweep_root/raw"
  run_name="effdock-guidance-sigma-sweep-eta2-v1-${dataset}-n100-s10-${sigma_tag}"
fi
pose_dir="$sampling_dir/poses/$run_name/$dataset/$selector"
if [[ "$mode" == smoke ]]; then
  output_dir="$sweep_root/sigmadock_posebusters/smoke/$selector/$run_name"
  only_id_args=(--only-id "$smoke_id")
else
  output_dir="$sweep_root/sigmadock_posebusters/official/$selector/$run_name"
  only_id_args=()
fi

reservation="$sweep_root/.task_reservations/sigmadock-pb-${mode}-${task_id}"
mkdir -p "$sweep_root/.task_reservations"
mkdir "$reservation" || { echo "refusing duplicate task reservation: $reservation" >&2; exit 2; }

echo "mode=$mode dataset=$dataset sigma=$sigma selector=$selector shard=$shard_index/$num_shards"
.venv/bin/python -m effdock.workflows.posebusters_report \
  --input-dir "$sampling_dir" \
  --run-name "$run_name" \
  --pose-dir "$pose_dir" \
  --output-dir "$output_dir" \
  --selector "$selector" \
  --num-shards "$num_shards" \
  --shard-index "$shard_index" \
  --expected-discovered-count "$expected_count" \
  --require-input-hashes \
  --require-complete-success \
  "${only_id_args[@]}"
