#!/usr/bin/env bash
#SBATCH --job-name=effdock-eta-conf-pb-1pass
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --array=0-255%16

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"
mkdir -p outputs/benchmarks/logs
# shellcheck source=guidance_eta_sweep_confidence_standalone_profile.sh
source scripts/slurm/guidance_eta_sweep_confidence_standalone_profile.sh

verify_manifest() {
  local label=$1
  local manifest=$2
  local expected_sha256=$3
  local actual_sha256
  if [[ ! -f "$manifest" ]]; then
    echo "missing frozen $label manifest: $manifest" >&2
    exit 2
  fi
  read -r actual_sha256 _ < <(sha256sum "$manifest")
  if [[ "$actual_sha256" != "$expected_sha256" ]] \
    || ! sha256sum --check --quiet "$manifest"; then
    echo "frozen $label manifest verification failed" >&2
    exit 2
  fi
}

execution_manifest=${EFFDOCK_STANDALONE_EXECUTION_MANIFEST:?EFFDOCK_STANDALONE_EXECUTION_MANIFEST is required}
execution_manifest_sha256=${EFFDOCK_STANDALONE_EXECUTION_MANIFEST_SHA256:?EFFDOCK_STANDALONE_EXECUTION_MANIFEST_SHA256 is required}
frozen_inputs_manifest=${EFFDOCK_STANDALONE_FROZEN_INPUTS_MANIFEST:?EFFDOCK_STANDALONE_FROZEN_INPUTS_MANIFEST is required}
frozen_inputs_manifest_sha256=${EFFDOCK_STANDALONE_FROZEN_INPUTS_MANIFEST_SHA256:?EFFDOCK_STANDALONE_FROZEN_INPUTS_MANIFEST_SHA256 is required}
verify_manifest execution "$execution_manifest" "$execution_manifest_sha256"
verify_manifest inputs "$frozen_inputs_manifest" "$frozen_inputs_manifest_sha256"

task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
task_count=${SLURM_ARRAY_TASK_COUNT:?SLURM_ARRAY_TASK_COUNT is required}
if [[ "$task_count" -ne "$posebusters_task_count" \
  || "$task_id" -lt 0 || "$task_id" -ge "$posebusters_task_count" ]]; then
  echo "standalone PoseBusters requires exactly $posebusters_task_count tasks" >&2
  exit 2
fi

selectors=(confidence confidence_filter)
shard_index=$((task_id % num_shards))
eta_index=$(((task_id / num_shards) % eta_count))
dataset_index=$(((task_id / (num_shards * eta_count)) % dataset_count))
selector_index=$((task_id / (num_shards * eta_count * dataset_count)))
selector=${selectors[$selector_index]}
eta=${eta_values[$eta_index]}
eta_tag=${eta_tags[$eta_index]}
case "$dataset_index" in
  0) dataset=astex; expected_count=85 ;;
  1) dataset=posebusters; expected_count=308 ;;
  *) echo "invalid dataset index: $dataset_index" >&2; exit 2 ;;
esac

output_root=${EFFDOCK_OUTPUT_ROOT:?EFFDOCK_OUTPUT_ROOT is required}
output_prefix_slash="$output_prefix/"
run_component=${output_root#"$output_prefix_slash"}
if [[ "$run_component" == "$output_root" || "$run_component" == */* ]] \
  || [[ ! "$run_component" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "unsafe standalone output root: $output_root" >&2
  exit 2
fi
sampling_dir="$output_root/raw"
official_root="$output_root/posebusters_official"
audit="$output_root/audit/full.json"
run_name="${run_name_prefix}-${dataset}-n100-s10-${eta_tag}"
pose_dir="$sampling_dir/poses/$run_name/$dataset/$selector"
output_dir="$official_root/$selector/$run_name"

if [[ ! -f "$audit" ]]; then
  echo "missing passed standalone full audit: $audit" >&2
  exit 2
fi
.venv/bin/python -c \
  'import json,sys; a=json.load(open(sys.argv[1])); ok=a.get("protocol_id")==sys.argv[2] and a.get("audit_contract")==sys.argv[3] and a.get("schema_version")==sys.argv[4] and a.get("mode")=="fresh_one_pass_characterization" and a.get("run_scope")=="full" and a.get("status")=="passed" and a.get("parent_compared") is False and a.get("deterministic_replay_claim") is False; raise SystemExit(0 if ok else "standalone full audit did not pass the frozen parent-free contract")' \
  "$audit" "$protocol_id" "$audit_contract" "$audit_schema_version"

mkdir -p "$output_root/.task_reservations"
if ! mkdir "$output_root/.task_reservations/posebusters-full-${task_id}"; then
  echo "refusing to rerun or overwrite standalone PoseBusters task $task_id" >&2
  exit 2
fi

echo "protocol=$protocol_id run=$run_name selector=$selector shard=$shard_index/$num_shards pose_dir=$pose_dir"
.venv/bin/python -m effdock.workflows.posebusters_report \
  --input-dir "$sampling_dir" \
  --num-shards "$num_shards" \
  --shard-index "$shard_index" \
  --run-name "$run_name" \
  --pose-dir "$pose_dir" \
  --output-dir "$output_dir" \
  --selector "$selector" \
  --expected-discovered-count "$expected_count" \
  --require-input-hashes \
  --require-complete-success
.venv/bin/python -m effdock.workflows.guidance_eta_sweep_confidence_standalone_binding \
  --sampling-dir "$sampling_dir" \
  --official-dir "$output_dir" \
  --run-name "$run_name" \
  --protocol-id "$protocol_id" \
  --dataset "$dataset" \
  --eta "$eta" \
  --selector "$selector" \
  --shard-index "$shard_index" \
  --num-shards "$num_shards" \
  --integrity-audit "$audit" \
  --profile "$standalone_profile"
