#!/usr/bin/env bash
#SBATCH --job-name=effdock-eta-conf-pb-smoke
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"

code_manifest=${EFFDOCK_CONFIDENCE_CODE_MANIFEST:?EFFDOCK_CONFIDENCE_CODE_MANIFEST is required}
expected_code_manifest_sha256=${EFFDOCK_CONFIDENCE_CODE_MANIFEST_SHA256:?EFFDOCK_CONFIDENCE_CODE_MANIFEST_SHA256 is required}
read -r actual_code_manifest_sha256 _ < <(sha256sum "$code_manifest")
if [[ "$actual_code_manifest_sha256" != "$expected_code_manifest_sha256" ]] \
  || ! sha256sum --check --quiet "$code_manifest"; then
  echo "frozen confidence execution manifest verification failed" >&2
  exit 2
fi

output_root=${EFFDOCK_OUTPUT_ROOT:?EFFDOCK_OUTPUT_ROOT is required}
extension_root="$output_root/confidence_selector_replay"
raw_root="$extension_root/smoke/raw"
run_name=effdock-guidance-direct-drift-eta-sweep-v2-posebusters-n100-s10-eta0500

for selector in confidence confidence_filter; do
  pose_dir="$raw_root/poses/$run_name/posebusters/$selector"
  output_dir="$extension_root/smoke/posebusters_official/$selector/$run_name"
  .venv/bin/python -m effdock.workflows.posebusters_report \
    --input-dir "$raw_root" \
    --num-shards 1 \
    --shard-index 0 \
    --run-name "$run_name" \
    --pose-dir "$pose_dir" \
    --output-dir "$output_dir" \
    --selector "$selector" \
    --expected-discovered-count 1 \
    --require-input-hashes \
    --require-complete-success
done
