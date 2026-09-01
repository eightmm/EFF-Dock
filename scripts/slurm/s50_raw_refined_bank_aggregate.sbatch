#!/usr/bin/env bash
#SBATCH --job-name=effdock-s50-mix-bank-agg
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --partition=cpu_only
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:00

set -euo pipefail
repo_dir=$(cd "${SLURM_SUBMIT_DIR:?}" && pwd -P)
cd "$repo_dir"
worker=${EFFDOCK_MIX_BANK_WORKER:?}
read -r actual _ < <(sha256sum "$worker")
[[ "$actual" == "${EFFDOCK_MIX_BANK_WORKER_SHA:?}" ]]
read -r actual _ < <(sha256sum "${EFFDOCK_MIX_BUILDER:?}")
[[ "$actual" == "${EFFDOCK_MIX_BUILDER_SHA:?}" ]]
refined_manifest=${EFFDOCK_MIX_REFINED_MANIFEST:?}
[[ -f "$refined_manifest.sha256" ]]
export PYTHONPATH="$repo_dir:$repo_dir/src"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
.venv/bin/python "$worker" aggregate \
  --raw-bank-manifest "${EFFDOCK_MIX_RAW_BANK:?}" \
  --refined-manifest "$refined_manifest" \
  --input-manifest "${EFFDOCK_MIX_INPUT_MANIFEST:?}" \
  --output-root "${EFFDOCK_MIX_BANK_OUTPUT_ROOT:?}/full" \
  --train-shards 128 --val-shards 8
