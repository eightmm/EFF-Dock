#!/usr/bin/env bash
#SBATCH --job-name=effdock-jitter-manifest
#SBATCH --partition=cpu_only
#SBATCH --qos=short
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err

set -euo pipefail
cd "${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:?}}"
: "${EFFDOCK_OUTPUT_ROOT:?set EFFDOCK_OUTPUT_ROOT}"
runtime_venv=${EFFDOCK_RUNTIME_VENV:-.venv}
cutoffs=(6 8 10 12)
jitters=(1 2)
task_id=${SLURM_ARRAY_TASK_ID:?}
[[ "${SLURM_ARRAY_TASK_COUNT:?}" -eq 24 ]] || { echo "manifest requires 24 tasks" >&2; exit 2; }
repeat_index=$((task_id % 3))
jitter=${jitters[$(((task_id / 3) % 2))]}
cutoff=${cutoffs[$((task_id / 6))]}
condition_root="$EFFDOCK_OUTPUT_ROOT/jitter_$(printf '%02d' "$jitter")/cutoff_$(printf '%02d' "$cutoff")/repeat_$repeat_index"
"$runtime_venv/bin/python" benchmarks/effdock/build_pocket_cutoff_manifest.py \
  --condition-root "$condition_root" --cutoff "$cutoff" --jitter "$jitter" \
  --repeat-index "$repeat_index" --output "$condition_root/manifest.json"
