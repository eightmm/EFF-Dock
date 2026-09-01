#!/usr/bin/env bash
#SBATCH --job-name=effdock-pl-k2-paired
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err
#SBATCH --partition=6000ada
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%4

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"
mkdir -p outputs/benchmarks/logs

runner=scripts/run_plinder_checkpoint_paired_validation.py
evaluator=src/effdock/workflows/evaluate.py
benchmark_helper=src/effdock/evaluation/benchmark.py
protocol=docs/EARLY_TIME_SAMPLER_PLINDER_K2_GATE_PROTOCOL.md
split=data/splits/plinder.json
pool=data/plinder_pool.parquet
config=configs/train.yaml
raw_gate=outputs/benchmarks/plinder_guidance_validation_runs/20260804T042517Z/raw_gate/verified.json
conformer_audit=outputs/analysis/rdkit_fragment_geometry_v2/val1076_seed0_heavy_only.json

protocol_sha256=0250853ae0793db288be2a6a8dc775db391d25aae32835b65b061782f34ab518
split_sha256=3ac570bf08bced053f1ce040b57efca27c3be616f29a82cd66ef887c08860e6b
pool_sha256=0ff455da77ce5540b839918cccb96f45414e91efff6272d7da3a65337ab1fe91
config_sha256=39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec
raw_gate_sha256=1ac146cfbec49ebfd1eb4452219320f134b0261bc8dc1bc196bcdab91b60f546
conformer_audit_sha256=d30f7380186d914b60964e120280dd84470b0f67b5a8aa9548e499af0aa942bf
runner_sha256=f003512bb600f342ad0e299185d314dd2e42616374c3003a5a9a6b8316ec5554
evaluator_sha256=0cf1b0e96edfc06467a15cbe2a6f0aaed1ee62d729219caab93b134519ea07dc
benchmark_helper_sha256=ca98e8fe121ee82b72f1b1a5f72f890ea6a3143d35247026a117365855dad401

s25_checkpoint=outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step25000_ema_inference.pt
s50_checkpoint=outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step50000_ema_common_init.pt
plus10k_checkpoint=outputs/eff-dock/early-time-t0-dose-control-t0p10-10k-v1-20260814/checkpoints/parent50k_plus10k_t0p10_ema_inference.pt
s25_sha256=c343ebc34cea3395762cd82e1c54b8c7b847dc04c4fa9e80b9813a864cafa0e1
s50_sha256=65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6
plus10k_sha256=0a48577379e286c584abd8c652d079b09dd6fff3c06a1a2f433d617ab0cd6074

verify_sha256() {
  local label=$1
  local path=$2
  local expected=$3
  local actual
  if [[ ! -f "$path" ]]; then
    echo "missing $label: $path" >&2
    exit 2
  fi
  read -r actual _ < <(sha256sum "$path")
  if [[ "$actual" != "$expected" ]]; then
    echo "$label SHA-256 mismatch: expected=$expected actual=$actual path=$path" >&2
    exit 2
  fi
}

if [[ ! -x .venv/bin/python ]]; then
  echo "missing .venv/bin/python; run uv sync before submission" >&2
  exit 2
fi
verify_sha256 protocol "$protocol" "$protocol_sha256"
verify_sha256 split "$split" "$split_sha256"
verify_sha256 pool "$pool" "$pool_sha256"
verify_sha256 config "$config" "$config_sha256"
verify_sha256 raw-gate "$raw_gate" "$raw_gate_sha256"
verify_sha256 conformer-audit "$conformer_audit" "$conformer_audit_sha256"
verify_sha256 runner "$runner" "$runner_sha256"
verify_sha256 evaluator "$evaluator" "$evaluator_sha256"
verify_sha256 benchmark-helper "$benchmark_helper" "$benchmark_helper_sha256"
verify_sha256 s25-ema "$s25_checkpoint" "$s25_sha256"
verify_sha256 s50-ema "$s50_checkpoint" "$s50_sha256"
verify_sha256 parent50k-plus10k-t0p10-ema "$plus10k_checkpoint" "$plus10k_sha256"

stage=${EFFDOCK_STAGE:?EFFDOCK_STAGE must be smoke, pilot, or full}
run_id=${EFFDOCK_RUN_ID:?EFFDOCK_RUN_ID is required}
eligibility_manifest=${EFFDOCK_ELIGIBILITY_MANIFEST:?EFFDOCK_ELIGIBILITY_MANIFEST is required}
eligibility_sha256=${EFFDOCK_ELIGIBILITY_SHA256:?EFFDOCK_ELIGIBILITY_SHA256 is required}
raw_root=${EFFDOCK_PLINDER_RAW_ROOT:-${XDG_DATA_HOME:-$HOME/.local/share}/plinder/2024-06/v2}
processed_root=${EFFDOCK_PROCESSED_ROOT:-data/plinder_processed}
output_prefix=outputs/benchmarks/early_time_sampler_plinder_k2_paired_runs

if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "unsafe run ID: $run_id" >&2
  exit 2
fi
verify_sha256 eligibility-manifest "$eligibility_manifest" "$eligibility_sha256"

task_id=${SLURM_ARRAY_TASK_ID:?submit every stage as a Slurm array}
task_count=${SLURM_ARRAY_TASK_COUNT:?SLURM_ARRAY_TASK_COUNT is required}
array_job_id=${SLURM_ARRAY_JOB_ID:?SLURM_ARRAY_JOB_ID is required}
case "$stage" in
  smoke|pilot)
    if [[ "$task_count" -ne 1 || "$task_id" -ne 0 ]]; then
      echo "$stage requires sbatch --array=0" >&2
      exit 2
    fi
    num_shards=1
    shard_index=0
    ;;
  full)
    if [[ "$task_count" -ne 8 || "$task_id" -lt 0 || "$task_id" -ge 8 ]]; then
      echo "full requires sbatch --array=0-7%4" >&2
      exit 2
    fi
    num_shards=8
    shard_index=$task_id
    ;;
  *)
    echo "EFFDOCK_STAGE must be smoke, pilot, or full; got $stage" >&2
    exit 2
    ;;
esac

if [[ ${SLURM_JOB_PARTITION:-} != "6000ada" ]]; then
  echo "frozen paired sampling requires partition 6000ada" >&2
  exit 2
fi
gpu_inventory=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
if [[ -z "$gpu_inventory" || $(wc -l <<< "$gpu_inventory") -ne 1 ]]; then
  echo "paired sampling requires exactly one visible CUDA GPU" >&2
  exit 2
fi
IFS=, read -r gpu_name gpu_memory_mib <<< "$gpu_inventory"
gpu_memory_mib=${gpu_memory_mib//[[:space:]]/}
if [[ ! "$gpu_memory_mib" =~ ^[0-9]+$ || "$gpu_memory_mib" -lt 40000 ]]; then
  echo "paired sampling requires at least 40000 MiB GPU memory" >&2
  echo "gpu=${gpu_name:-unset} memory_mib=${gpu_memory_mib:-unset}" >&2
  exit 2
fi

.venv/bin/python "$runner" verify \
  --raw-root "$raw_root" \
  --processed-root "$processed_root" \
  --eligibility-manifest "$eligibility_manifest" \
  --eligibility-manifest-sha256 "$eligibility_sha256"

run_root="$output_prefix/$run_id"
if [[ "$stage" == "pilot" || "$stage" == "full" ]]; then
  smoke_summary="$run_root/smoke/shard-000-of-001/paired_summary.json"
  .venv/bin/python "$runner" verify-stage \
    --stage smoke \
    --summary "$smoke_summary" \
    --eligibility-manifest-sha256 "$eligibility_sha256"
fi
if [[ "$stage" == "full" ]]; then
  pilot_summary="$run_root/pilot/shard-000-of-001/paired_summary.json"
  .venv/bin/python "$runner" verify-stage \
    --stage pilot \
    --summary "$pilot_summary" \
    --eligibility-manifest-sha256 "$eligibility_sha256"
fi

# Bind each stage to one array job and reserve each task before any model load.
reservation_root="$run_root/.task_reservations"
mkdir -p "$reservation_root"
array_marker="$reservation_root/${stage}.array_job_id"
marker_candidate="$reservation_root/.${stage}.${array_job_id}.${task_id}.tmp"
printf '%s\n' "$array_job_id" > "$marker_candidate"
if ! ln "$marker_candidate" "$array_marker" 2>/dev/null; then
  existing_array_job_id=
  if [[ -f "$array_marker" ]]; then
    read -r existing_array_job_id < "$array_marker" || true
  fi
  if [[ "$existing_array_job_id" != "$array_job_id" ]]; then
    rm -f "$marker_candidate"
    echo "refusing to reuse $stage outputs from another Slurm array" >&2
    exit 2
  fi
fi
rm -f "$marker_candidate"
task_reservation="$reservation_root/${stage}.task-${task_id}"
if ! mkdir "$task_reservation"; then
  echo "refusing to rerun reserved stage task: $stage/$task_id" >&2
  exit 2
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
printf 'protocol_id=%s protocol_sha256=%s\n' \
  EFFDOCK-EARLY-TIME-SAMPLER-PLINDER-K2-GATE-V1 "$protocol_sha256"
printf 'stage=%s shard=%s/%s eligibility_sha256=%s\n' \
  "$stage" "$shard_index" "$num_shards" "$eligibility_sha256"
printf 'gpu=%s memory_mib=%s partition=%s git_commit=%s\n' \
  "$gpu_name" "$gpu_memory_mib" "${SLURM_JOB_PARTITION}" "$(git rev-parse HEAD)"

.venv/bin/python "$runner" run-shard \
  --stage "$stage" \
  --raw-root "$raw_root" \
  --processed-root "$processed_root" \
  --eligibility-manifest "$eligibility_manifest" \
  --eligibility-manifest-sha256 "$eligibility_sha256" \
  --output-root "$output_prefix" \
  --run-id "$run_id" \
  --num-shards "$num_shards" \
  --shard-index "$shard_index"
