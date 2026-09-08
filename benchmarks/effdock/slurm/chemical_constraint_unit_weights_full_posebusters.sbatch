#!/usr/bin/env bash
#SBATCH --job-name=effdock-chemical-full-pb
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --array=0-31%16

set -euo pipefail
cd "${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:?}}"
: "${EFFDOCK_OUTPUT_ROOT:?set EFFDOCK_OUTPUT_ROOT}"

task_id=${SLURM_ARRAY_TASK_ID:?}
shard_index=$((task_id % 8))
arm_index=$(((task_id / 8) % 2))
dataset_index=$((task_id / 16))
arms=(physical_interaction_1 all_three_1)
arm=${arms[$arm_index]}
case "$dataset_index" in
  0) dataset=astex; expected_count=85 ;;
  1) dataset=posebusters; expected_count=308 ;;
  *) exit 2 ;;
esac

run_name="effdock-chemical-unit1-full-${dataset}-${arm}"
pose_dir="$EFFDOCK_OUTPUT_ROOT/raw/poses/$run_name/$dataset/confidence"
output_dir="$EFFDOCK_OUTPUT_ROOT/posebusters_official/$run_name"
.venv/bin/python -m effdock.workflows.posebusters_report \
  --input-dir "$EFFDOCK_OUTPUT_ROOT/raw" \
  --num-shards 8 \
  --shard-index "$shard_index" \
  --run-name "$run_name" \
  --pose-dir "$pose_dir" \
  --output-dir "$output_dir" \
  --selector confidence \
  --expected-discovered-count "$expected_count" \
  --require-input-hashes \
  --require-complete-success
