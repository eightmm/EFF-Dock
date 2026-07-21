#!/usr/bin/env bash
#SBATCH --job-name=effdock-conf-filter-v1
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --partition=heavy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h100:1

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"
mkdir -p outputs/slurm

output_dir=${EFFDOCK_FILTER_OUTPUT_DIR:-outputs/eff-dock/confidence-filter-v1}
train_limit=${EFFDOCK_FILTER_TRAIN_LIMIT:-1024}
val_limit=${EFFDOCK_FILTER_VAL_LIMIT:-0}
num_workers=${EFFDOCK_FILTER_NUM_WORKERS:-4}
reuse_cache=${EFFDOCK_FILTER_REUSE_CACHE:-0}

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

command=(
  uv run python -m effdock.workflows.tune_confidence_filter
  --config configs/train_confidence.yaml
  --checkpoint weights/effdock_confidence_extmatch_n80_s25_step42500.pt
  --output-dir "$output_dir"
  --train-limit "$train_limit"
  --num-workers "$num_workers"
)
if [[ "$val_limit" -gt 0 ]]; then
  command+=(--val-limit "$val_limit")
fi
if [[ "$reuse_cache" == "1" ]]; then
  if [[ -f "$output_dir/result.json" ]]; then
    cp "$output_dir/result.json" "$output_dir/result_strict_both.json"
  fi
  command+=(--reuse-cache)
fi

oms research-runner \
  --question "Can a cluster-free confidence/physical filter improve top-1 pose selection?" \
  --hypothesis "Fixed near-tie head-consensus and clash guards improve PLINDER validation top-1 RMSD <2A." \
  --prediction "+1.0pp validation <2A with non-worse median RMSD and <=35% switches" \
  --baseline "pure predicted-RMSD head at retained confidence step 42500" \
  --metric "top-1 RMSD <2A on PLINDER train-fit and frozen validation-confirm" \
  --success "+1.0pp on train and validation, non-worse median, finite full coverage" \
  --change "cluster-free fixed-score filter; no pairwise pose distance or candidate-density features" \
  --metrics "$output_dir/result.json" \
  --no-gate \
  --reason "focused tests and workflow help/syntax checks passed before submission" \
  -- \
  "${command[@]}"
