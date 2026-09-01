#!/usr/bin/env bash
#SBATCH --job-name=effdock-ext-fabind-weight
#SBATCH --output=external_models/logs/%x-%j.out
#SBATCH --error=external_models/logs/%x-%j.err
#SBATCH --partition=cpu_only
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  echo "EFFDOCK_REPO_DIR/SLURM_SUBMIT_DIR is required" >&2
  exit 2
fi

cd "$repo_dir"
mkdir -p external_models/logs
runtime_dir="${SLURM_TMPDIR:-/tmp}/effdock-ext-fabind-${SLURM_JOB_ID:-manual}"
mkdir -p "$runtime_dir"
cp scripts/external_models/download_fabind_artifact.sh "$runtime_dir/download_fabind_artifact.sh"
cp scripts/external_models/parallel_download.py "$runtime_dir/parallel_download.py"

echo "host=$(hostname)"
echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "effdock_commit=$(git rev-parse HEAD)"
EFFDOCK_REPO_ROOT="$repo_dir" \
EFFDOCK_PARALLEL_DOWNLOADER="$runtime_dir/parallel_download.py" \
  bash "$runtime_dir/download_fabind_artifact.sh"
echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
