#!/usr/bin/env bash
#SBATCH --job-name=effdock-others-uv
#SBATCH --output=others/logs/%x-%j.out
#SBATCH --error=others/logs/%x-%j.err
#SBATCH --partition=cpu_only
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00

set -euo pipefail

repo_root=${EFFDOCK_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
model=${MODEL:-}
if [[ -z "$repo_root" || -z "$model" ]]; then
  echo "EFFDOCK_REPO_ROOT/SLURM_SUBMIT_DIR and MODEL are required" >&2
  exit 2
fi

cd "$repo_root"
mkdir -p others/logs "others/$model/logs"
echo "model=$model"
echo "host=$(hostname)"
echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "effdock_commit=$(git rev-parse HEAD)"

oms run-ledger \
  --file "others/$model/logs/install-ledger.jsonl" \
  --note "model-local uv sync: $model" \
  -- bash scripts/others/sync_model.sh "$model"

echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
