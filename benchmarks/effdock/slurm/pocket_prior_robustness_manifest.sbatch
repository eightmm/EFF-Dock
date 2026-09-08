#!/usr/bin/env bash
#SBATCH --job-name=effdock-pocket-prior-manifest
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
source benchmarks/effdock/slurm/pocket_prior_robustness_common.sh
task_id=${SLURM_ARRAY_TASK_ID:?}
[[ "${SLURM_ARRAY_TASK_COUNT:?}" -eq 27 ]] || { echo "manifest requires 27 tasks" >&2; exit 2; }
effdock_resolve_pocket_prior_condition "$task_id"
condition_root=$(effdock_pocket_prior_condition_root "$EFFDOCK_OUTPUT_ROOT")
"$runtime_venv/bin/python" benchmarks/effdock/build_pocket_cutoff_manifest.py \
  --condition-root "$condition_root" --cutoff "$EFFDOCK_CONDITION_CUTOFF" \
  --sigma "$EFFDOCK_CONDITION_SIGMA" --jitter "$EFFDOCK_CONDITION_JITTER" \
  --repeat-index "$EFFDOCK_CONDITION_REPEAT" --extension \
  --output "$condition_root/manifest.json"
