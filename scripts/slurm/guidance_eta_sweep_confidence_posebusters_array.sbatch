#!/usr/bin/env bash
#SBATCH --job-name=effdock-eta-conf-pb
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --array=0-255%16

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"
mkdir -p outputs/benchmarks/logs

code_manifest=${EFFDOCK_CONFIDENCE_CODE_MANIFEST:?EFFDOCK_CONFIDENCE_CODE_MANIFEST is required}
expected_code_manifest_sha256=${EFFDOCK_CONFIDENCE_CODE_MANIFEST_SHA256:?EFFDOCK_CONFIDENCE_CODE_MANIFEST_SHA256 is required}
read -r actual_code_manifest_sha256 _ < <(sha256sum "$code_manifest")
if [[ "$actual_code_manifest_sha256" != "$expected_code_manifest_sha256" ]] \
  || ! sha256sum --check --quiet "$code_manifest"; then
  echo "frozen confidence execution manifest verification failed" >&2
  exit 2
fi

task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
task_count=${SLURM_ARRAY_TASK_COUNT:?SLURM_ARRAY_TASK_COUNT is required}
if [[ "$task_count" -ne 256 || "$task_id" -lt 0 || "$task_id" -ge 256 ]]; then
  echo "confidence PoseBusters requires the exact 0-255 array" >&2
  exit 2
fi

selectors=(confidence confidence_filter)
eta_values=(0.0 0.025 0.05 0.1 0.2 0.3 0.4 0.5)
eta_tags=(eta0000 eta0025 eta0050 eta0100 eta0200 eta0300 eta0400 eta0500)
shard_index=$((task_id % 8))
eta_index=$(((task_id / 8) % 8))
dataset_index=$(((task_id / 64) % 2))
selector_index=$((task_id / 128))
selector=${selectors[$selector_index]}
eta=${eta_values[$eta_index]}
eta_tag=${eta_tags[$eta_index]}
case "$dataset_index" in
  0) dataset=astex; expected_count=85 ;;
  1) dataset=posebusters; expected_count=308 ;;
  *) echo "invalid dataset index: $dataset_index" >&2; exit 2 ;;
esac

output_root=${EFFDOCK_OUTPUT_ROOT:?EFFDOCK_OUTPUT_ROOT is required}
extension_root="$output_root/confidence_selector_replay"
raw_root="$extension_root/raw"
official_root="$extension_root/posebusters_official"
run_name="effdock-guidance-direct-drift-eta-sweep-v2-${dataset}-n100-s10-${eta_tag}"
pose_dir="$raw_root/poses/$run_name/$dataset/$selector"
output_dir="$official_root/$selector/$run_name"

echo "run=$run_name selector=$selector shard=$shard_index/8 pose_dir=$pose_dir"
.venv/bin/python -m effdock.workflows.posebusters_report \
  --input-dir "$raw_root" \
  --num-shards 8 \
  --shard-index "$shard_index" \
  --run-name "$run_name" \
  --pose-dir "$pose_dir" \
  --output-dir "$output_dir" \
  --selector "$selector" \
  --expected-discovered-count "$expected_count" \
  --require-input-hashes \
  --require-complete-success
.venv/bin/python -m effdock.workflows.guidance_eta_sweep_confidence_binding \
  --sampling-dir "$raw_root" \
  --official-dir "$output_dir" \
  --run-name "$run_name" \
  --protocol-id EFFDOCK-UNIFIED-GUIDANCE-ETA-SWEEP-CONFIDENCE-PB-V1 \
  --dataset "$dataset" \
  --eta "$eta" \
  --selector "$selector" \
  --shard-index "$shard_index" \
  --num-shards 8
