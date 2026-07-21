#!/usr/bin/env bash
#SBATCH --job-name=effdock-pb-valid
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"
mkdir -p outputs/benchmarks/logs
: "${EFFDOCK_NUM_SHARDS:?set EFFDOCK_NUM_SHARDS}"

run_name=${EFFDOCK_PB_RUN_NAME:-effdock-redock-ema-n40-s25-v1-posebusters}
pose_dir=${EFFDOCK_PB_POSE_DIR:-outputs/benchmarks/raw/poses/posebusters/vina}
output_dir=${EFFDOCK_PB_OUTPUT_DIR:-outputs/benchmarks/posebusters_official}
selector=${EFFDOCK_PB_SELECTOR:-effdock_torch_vina_plus_dg}

uv run python -m effdock.workflows.posebusters_report \
  --num-shards "$EFFDOCK_NUM_SHARDS" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --run-name "$run_name" \
  --pose-dir "$pose_dir" \
  --output-dir "$output_dir" \
  --selector "$selector"
