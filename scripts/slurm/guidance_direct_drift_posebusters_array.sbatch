#!/usr/bin/env bash
#SBATCH --job-name=effdock-direct-pb
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-95%16

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"
mkdir -p outputs/benchmarks/logs

output_root=${EFFDOCK_OUTPUT_ROOT:-outputs/benchmarks/guidance_direct_drift_v1}
raw_root="$output_root/raw"
official_root="$output_root/posebusters_official"
task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
shard_index=$((task_id % 8))
run_index=$((task_id / 8))
arm_index=$((run_index % 2))
condition_index=$(((run_index / 2) % 3))
dataset_index=$((run_index / 6))
steps=(10 20 25)
poses=(100 50 40)
arms=(unguided direct)
num_steps=${steps[$condition_index]}
num_samples=${poses[$condition_index]}
arm=${arms[$arm_index]}

case "$dataset_index" in
  0) dataset=astex; expected_count=85 ;;
  1) dataset=posebusters; expected_count=308 ;;
  *) echo "invalid dataset index: $dataset_index" >&2; exit 2 ;;
esac

run_name="effdock-guidance-direct-drift-v1-${dataset}-n${num_samples}-s${num_steps}-${arm}"
pose_dir="$raw_root/poses/$run_name/$dataset/oracle"
output_dir="$official_root/$run_name"
echo "run=$run_name shard=$shard_index/8 pose_dir=$pose_dir"

.venv/bin/python -m effdock.workflows.posebusters_report \
  --input-dir "$raw_root" \
  --num-shards 8 \
  --shard-index "$shard_index" \
  --run-name "$run_name" \
  --pose-dir "$pose_dir" \
  --output-dir "$output_dir" \
  --selector oracle \
  --expected-discovered-count "$expected_count" \
  --require-input-hashes \
  --require-complete-success
