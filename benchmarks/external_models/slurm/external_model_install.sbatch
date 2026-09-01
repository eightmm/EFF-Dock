#!/usr/bin/env bash
#SBATCH --job-name=effdock-ext-install
#SBATCH --output=external_models/logs/%x-%j.out
#SBATCH --error=external_models/logs/%x-%j.err
#SBATCH --partition=cpu_only
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
model=${MODEL:-}
if [[ -z "$repo_dir" || -z "$model" ]]; then
  echo "EFFDOCK_REPO_DIR/SLURM_SUBMIT_DIR and MODEL are required" >&2
  exit 2
fi

cd "$repo_dir"
mkdir -p external_models/logs

echo "model=$model"
echo "host=$(hostname)"
echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "effdock_commit=$(git rev-parse HEAD)"
runtime_dir="${SLURM_TMPDIR:-/tmp}/effdock-ext-install-${SLURM_JOB_ID:-manual}"
mkdir -p "$runtime_dir"
cp scripts/external_models/install_model.sh "$runtime_dir/install_model.sh"
EFFDOCK_REPO_ROOT="$repo_dir" bash "$runtime_dir/install_model.sh" "$model"
echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
