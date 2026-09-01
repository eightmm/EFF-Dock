#!/usr/bin/env bash
#SBATCH --job-name=effdock-plinder-guidance-data
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err
#SBATCH --partition=cpu_only
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  echo "EFFDOCK_REPO_DIR or SLURM_SUBMIT_DIR is required" >&2
  exit 2
fi
cd "$repo_dir"

if [[ ! -f PROJECT.md || ! -f data/splits/plinder.json ]]; then
  echo "not an EFF-Dock repository or frozen PLINDER split is missing: $repo_dir" >&2
  exit 2
fi
if [[ ! -x .venv/bin/python ]]; then
  echo "missing executable project environment: .venv/bin/python" >&2
  exit 2
fi
if [[ ! -f scripts/download_plinder_guidance_validation.py ]]; then
  echo "missing PLINDER guidance-validation download helper" >&2
  exit 2
fi

mkdir -p outputs/benchmarks/logs outputs/benchmarks/plinder_guidance_validation
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK is required}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

echo "repo_commit=$(git rev-parse HEAD)"
echo "repo_diff_sha256=$(git diff --no-ext-diff | sha256sum | cut -d' ' -f1)"
echo "python=$(.venv/bin/python --version 2>&1)"

.venv/bin/python scripts/download_plinder_guidance_validation.py \
  --split-file data/splits/plinder.json \
  --parallelism "$SLURM_CPUS_PER_TASK" \
  --output outputs/benchmarks/plinder_guidance_validation/raw_download_manifest.json
