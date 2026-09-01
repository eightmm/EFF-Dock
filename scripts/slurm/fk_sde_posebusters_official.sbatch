#!/usr/bin/env bash
#SBATCH --job-name=effdock-fk-sde-pb-official
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err
#SBATCH --partition=cpu_only
#SBATCH --qos=veryshort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --array=0-15%16

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"

output_root=${EFFDOCK_FK_SDE_PB_OUTPUT_ROOT:?EFFDOCK_FK_SDE_PB_OUTPUT_ROOT is required}
mode=${EFFDOCK_FK_SDE_PB_MODE:?EFFDOCK_FK_SDE_PB_MODE is required}
shard_index=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
case "$mode" in
  smoke)
    manifest="$output_root/smoke/manifest.json"
    num_shards=1
    ;;
  full)
    manifest="$output_root/manifest.json"
    num_shards=16
    ;;
  *) echo "official mode must be smoke or full" >&2; exit 2 ;;
esac
if [[ "$shard_index" -lt 0 || "$shard_index" -ge "$num_shards" ]]; then
  echo "official shard index is outside the frozen mode grid" >&2
  exit 2
fi
read -r manifest_sha256 _ < <(sha256sum "$manifest")

.venv/bin/python -m effdock.workflows.fk_sde_posebusters_report score \
  --manifest "$manifest" \
  --manifest-sha256 "$manifest_sha256" \
  --output-root "$output_root/official" \
  --num-shards "$num_shards" \
  --shard-index "$shard_index" \
  --workers 4
