#!/usr/bin/env bash
#SBATCH --job-name=effdock-t0p10-paired
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err
#SBATCH --partition=6000ada,heavy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%4

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"
mkdir -p outputs/benchmarks/logs

protocol_id=EFFDOCK-EARLY-TIME-T0P10-50K-EXTERNAL-PAIRED-V1
protocol_path=docs/EARLY_TIME_T0P10_50K_EXTERNAL_PAIRED_PROTOCOL.md
protocol_sha256=976a292470fa03bef85ac6e8a711a74e1ad32f4b1ca870fa69dcaa3c89c1d203
evaluator_path=src/effdock/workflows/evaluate.py
evaluator_sha256=91db3ffd2953a04a46b2af2e8854a1fdf78328cb83d88c83aa9f3199e706734c
source_inventory_sha256=a3c7b5da8898f4057fcf0a71771738495ddec0ca37213c43c04a51e3aa980a35
config=configs/train.yaml
config_sha256=39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec
benchmark_input_manifest=docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json
benchmark_input_manifest_sha256=99f15f557644cc51c3dd1f559b0dd97dd4259c1de3e1403fb761b7c7e079f668
cohort_audit=${EFFDOCK_COHORT_MANIFEST:-docs/GUIDANCE_BUDGET1000_FULL_COHORT.json}
cohort_audit_sha256=d7321f847c8d6d08950e02d5f41ff42b62fd29ccea78072f27078aa039791c45

current_raw_checkpoint=weights/effdock_geometry_ft_100k_best.pt
current_raw_sha256=6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db
parent_ema_checkpoint=outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/parent_step100000_ema_inference.pt
parent_ema_sha256=166d92a7f74015b0011451ad70c71601d72769da00ce1206c8a6a27832e40d97
t0p10_50k_ema_checkpoint=outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step50000_ema_common_init.pt
t0p10_50k_ema_sha256=65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6

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
    echo "$label SHA-256 mismatch" >&2
    echo "path=$path expected=$expected actual=$actual" >&2
    exit 2
  fi
}

if [[ ! -x .venv/bin/eff-dock || ! -x .venv/bin/python ]]; then
  echo "missing .venv EFF-Dock executables; run uv sync first" >&2
  exit 2
fi
verify_sha256 protocol "$protocol_path" "$protocol_sha256"
verify_sha256 evaluator "$evaluator_path" "$evaluator_sha256"
verify_sha256 config "$config" "$config_sha256"
verify_sha256 benchmark-input-manifest \
  "$benchmark_input_manifest" "$benchmark_input_manifest_sha256"
verify_sha256 full-cohort-audit "$cohort_audit" "$cohort_audit_sha256"
verify_sha256 current_raw-checkpoint "$current_raw_checkpoint" "$current_raw_sha256"
verify_sha256 parent_ema-checkpoint "$parent_ema_checkpoint" "$parent_ema_sha256"
verify_sha256 t0p10_50k_ema-checkpoint \
  "$t0p10_50k_ema_checkpoint" "$t0p10_50k_ema_sha256"
actual_source_inventory_sha256=$(
  {
    find src/effdock -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum
    sha256sum pyproject.toml uv.lock
  } | sha256sum | awk '{print $1}'
)
if [[ "$actual_source_inventory_sha256" != "$source_inventory_sha256" ]]; then
  echo "frozen EFF-Dock source inventory SHA-256 mismatch" >&2
  echo "expected=$source_inventory_sha256 actual=$actual_source_inventory_sha256" >&2
  exit 2
fi

output_prefix=outputs/benchmarks/early_time_t0p10_50k_external_paired_runs
output_root=${EFFDOCK_OUTPUT_ROOT:?EFFDOCK_OUTPUT_ROOT is required}
output_prefix_slash="$output_prefix/"
run_component=${output_root#"$output_prefix_slash"}
if [[ "$run_component" == "$output_root" || "$run_component" == */* ]] \
  || [[ ! "$run_component" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "unsafe output root; expected $output_prefix/<one-safe-run-name>" >&2
  echo "got: $output_root" >&2
  exit 2
fi

task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
task_count=${SLURM_ARRAY_TASK_COUNT:?SLURM_ARRAY_TASK_COUNT is required}
array_job_id=${SLURM_ARRAY_JOB_ID:?SLURM_ARRAY_JOB_ID is required}
smoke=${EFFDOCK_SMOKE:-0}
stage_dataset=${EFFDOCK_STAGE_DATASET:-}
case "$smoke" in
  0|1) ;;
  *) echo "EFFDOCK_SMOKE must be 0 or 1" >&2; exit 2 ;;
esac

mode=full
only_id=
num_samples=100
num_steps=10
prior_pool_size=100
num_shards=8
if [[ "$smoke" == 1 ]]; then
  if [[ -n "$stage_dataset" ]]; then
    echo "EFFDOCK_STAGE_DATASET must be unset for the two-dataset smoke" >&2
    exit 2
  fi
  mode=smoke
  if [[ "$task_count" -ne 2 || "$task_id" -lt 0 || "$task_id" -ge 2 ]]; then
    echo "smoke requires exactly --array=0-1 (two dataset tasks)" >&2
    exit 2
  fi
  dataset_index=$task_id
  shard_index=0
  num_shards=1
  num_samples=2
  num_steps=2
  prior_pool_size=2
  raw_root="$output_root/smoke/raw"
else
  if [[ "$task_count" -ne 8 || "$task_id" -lt 0 || "$task_id" -ge 8 ]]; then
    echo "staged full run requires exactly --array=0-7 (eight shards)" >&2
    exit 2
  fi
  case "$stage_dataset" in
    astex) dataset_index=0 ;;
    posebusters) dataset_index=1 ;;
    *)
      echo "full run requires EFFDOCK_STAGE_DATASET=astex or posebusters" >&2
      echo "Astex must complete before submitting the PoseBusters stage" >&2
      exit 2
      ;;
  esac
  shard_index=$task_id
  raw_root="$output_root/raw"
fi

case "$dataset_index" in
  0)
    dataset=astex
    data_dir=data/external_benchmarks/data/astex_diverse_set
    centers=data/external_test/astex_reference_pocket_centers.json
    centers_sha256=1ac4d8629a7ee2adb785173db56fb69ec4140d68e3057631ae10df6ef88d0d85
    expected_count=85
    if [[ "$mode" == smoke ]]; then
      only_id=1jje
    fi
    ;;
  1)
    dataset=posebusters
    data_dir=data/external_benchmarks/data/posebusters_benchmark_set
    centers=data/external_test/posebusters_reference_pocket_centers.json
    centers_sha256=2d3db55c8cc75650cff85d8e3c12445fb8f45fbe2673d8bbc32045ee8c0f6ad0
    expected_count=308
    if [[ "$mode" == smoke ]]; then
      only_id=7b2c_tp7
    fi
    ;;
  *) echo "invalid dataset index: $dataset_index" >&2; exit 2 ;;
esac

if [[ ! -d "$data_dir" ]]; then
  echo "missing benchmark data directory: $data_dir" >&2
  exit 2
fi
verify_sha256 "$dataset-pocket-centers" "$centers" "$centers_sha256"
.venv/bin/python -c \
  '
import json
import sys
from pathlib import Path

from effdock.workflows.benchmark_inputs import (
    file_sha256,
    load_benchmark_inputs,
    sorted_id_sha256,
)
from effdock.workflows.evaluate import discover_complexes, load_pocket_centers

cohort_path = Path(sys.argv[1])
dataset = sys.argv[2]
data_dir = Path(sys.argv[3])
external_dir = Path(sys.argv[4])
input_manifest = Path(sys.argv[5])
centers_path = Path(sys.argv[6])
expected_count = int(sys.argv[7])
raw = json.loads(cohort_path.read_text())
entry = raw.get("datasets", {}).get(dataset)
if not isinstance(entry, dict):
    raise ValueError(f"{cohort_path}: missing dataset {dataset}")
mapping, input_identity = load_benchmark_inputs(dataset, external_dir, input_manifest)
expected_ids = sorted(mapping)
discovered = discover_complexes(
    dataset,
    data_dir,
    external_dir,
    load_pocket_centers(centers_path),
    input_manifest,
)
discovered_by_id = {item.complex_id: item for item in discovered}
if len(expected_ids) != expected_count or sorted(discovered_by_id) != expected_ids:
    raise ValueError(f"{dataset}: current discovery differs from frozen input cohort")
expected_ids_sha256 = sorted_id_sha256(expected_ids)
for name in ("ids", "audited_ids", "success_ids"):
    if entry.get(name) != expected_ids or entry.get(f"{name}_sha256") != expected_ids_sha256:
        raise ValueError(f"{cohort_path}:{dataset}.{name} differs from frozen inputs")
for name in ("discovered", "audited", "success"):
    if int(entry.get(name, -1)) != expected_count:
        raise ValueError(f"{cohort_path}:{dataset}.{name} count mismatch")
if (
    entry.get("complete") is not True
    or int(entry.get("failed", -1)) != 0
    or entry.get("failed_ids") != []
    or entry.get("failure_codes") != {}
):
    raise ValueError(f"{cohort_path}:{dataset} is not a complete zero-failure cohort")
inputs = entry.get("inputs")
if not isinstance(inputs, dict) or inputs.get("benchmark_input_identity") != input_identity:
    raise ValueError(f"{cohort_path}:{dataset} benchmark input identity mismatch")
complexes = entry.get("complexes")
if not isinstance(complexes, dict) or sorted(complexes) != expected_ids:
    raise ValueError(f"{cohort_path}:{dataset} per-complex inventory mismatch")
for complex_id in expected_ids:
    record = complexes[complex_id]
    item = discovered_by_id[complex_id]
    if record.get("status") != "success":
        raise ValueError(f"{cohort_path}:{dataset}/{complex_id} was not audited successfully")
    if record.get("protein_sha256") != file_sha256(item.protein):
        raise ValueError(f"{cohort_path}:{dataset}/{complex_id} protein hash mismatch")
    if record.get("ligand_reference_sha256") != file_sha256(item.ligand_ref):
        raise ValueError(f"{cohort_path}:{dataset}/{complex_id} reference hash mismatch")
    expected_ligand_identity = input_identity["per_id"][complex_id]["sha256"]
    if record.get("ligand_input_identity_sha256") != expected_ligand_identity:
        raise ValueError(f"{cohort_path}:{dataset}/{complex_id} ligand input mismatch")
print(
    f"paired cohort gate passed: dataset={dataset} count={expected_count} "
    f"ids_sha256={expected_ids_sha256}"
)
' \
  "$cohort_audit" "$dataset" "$data_dir" data/external_test \
  "$benchmark_input_manifest" "$centers" "$expected_count"

slurm_partition=${SLURM_JOB_PARTITION:?SLURM_JOB_PARTITION is required}
if [[ "$mode" == full ]]; then
  case "$slurm_partition" in
    6000ada|heavy) ;;
    *)
      echo "full sampling requires partition 6000ada or heavy; got $slurm_partition" >&2
      exit 2
      ;;
  esac
  minimum_gpu_memory_mib=48000
else
  case "$slurm_partition" in
    test|6000ada|heavy) ;;
    *)
      echo "smoke sampling requires partition test, 6000ada, or heavy; got $slurm_partition" >&2
      exit 2
      ;;
  esac
  minimum_gpu_memory_mib=20000
fi

gpu_inventory=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
if [[ -z "$gpu_inventory" || $(wc -l <<< "$gpu_inventory") -ne 1 ]]; then
  echo "paired sampling requires exactly one visible CUDA device" >&2
  exit 2
fi
IFS=, read -r gpu_name gpu_memory_mib <<< "$gpu_inventory"
gpu_memory_mib=${gpu_memory_mib//[[:space:]]/}
if [[ ! "$gpu_memory_mib" =~ ^[0-9]+$ ]] \
  || [[ "$gpu_memory_mib" -lt "$minimum_gpu_memory_mib" ]]; then
  echo "$mode sampling requires at least $minimum_gpu_memory_mib MiB GPU memory" >&2
  echo "gpu=${gpu_name:-unset} memory_mib=${gpu_memory_mib:-unset}" >&2
  exit 2
fi

# Bind each output mode/dataset stage to one Slurm array. This permits the
# registered Astex-then-PoseBusters full sequence while refusing stage reuse.
mkdir -p "$output_root"
array_marker="$output_root/.${mode}_${dataset}_array_job_id"
marker_candidate="$output_root/.${mode}_${dataset}_array_job_id.${array_job_id}.${task_id}.tmp"
printf '%s\n' "$array_job_id" > "$marker_candidate"
if ln "$marker_candidate" "$array_marker" 2>/dev/null; then
  :
else
  existing_array_job_id=
  if [[ -f "$array_marker" ]]; then
    read -r existing_array_job_id < "$array_marker" || true
  fi
  if [[ "$existing_array_job_id" != "$array_job_id" ]]; then
    rm -f "$marker_candidate"
    echo "refusing to reuse $mode/$dataset output stage from another Slurm array" >&2
    echo "root=$output_root existing=${existing_array_job_id:-invalid} current=$array_job_id" >&2
    exit 2
  fi
fi
rm -f "$marker_candidate"

reservation_root="$output_root/.task_reservations"
mkdir -p "$reservation_root"
reservation="$reservation_root/sampling-${mode}-${dataset}-${task_id}"
if ! mkdir "$reservation"; then
  echo "refusing to rerun or overwrite paired sampling task $mode/$dataset/$task_id" >&2
  exit 2
fi

arms=(current_raw parent_ema t0p10_50k_ema)
checkpoints=(
  "$current_raw_checkpoint"
  "$parent_ema_checkpoint"
  "$t0p10_50k_ema_checkpoint"
)
checkpoint_sha256s=(
  "$current_raw_sha256"
  "$parent_ema_sha256"
  "$t0p10_50k_ema_sha256"
)
if [[ "$mode" == smoke ]]; then
  # Same checkpoint/settings replay measures the CUDA numerical K2 floor.
  arms+=(current_raw_replay)
  checkpoints+=("$current_raw_checkpoint")
  checkpoint_sha256s+=("$current_raw_sha256")
fi

only_id_args=()
if [[ -n "$only_id" ]]; then
  # Discovery/count/seed assignment happen before this filter in the evaluator.
  only_id_args=(--only-id "$only_id")
fi

printf 'protocol_id=%s protocol_path=%s protocol_sha256=%s\n' \
  "$protocol_id" "$protocol_path" "$protocol_sha256"
printf 'evaluator_sha256=%s source_inventory_sha256=%s\n' \
  "$evaluator_sha256" "$source_inventory_sha256"
printf 'mode=%s dataset=%s shard=%s/%s only_id=%s N=%s S=%s prior=%s\n' \
  "$mode" "$dataset" "$shard_index" "$num_shards" "${only_id:-none}" \
  "$num_samples" "$num_steps" "$prior_pool_size"
printf 'output_root=%s raw_root=%s cohort=%s\n' \
  "$output_root" "$raw_root" "$cohort_audit"
printf 'gpu partition=%s name=%s memory_mib=%s minimum_mib=%s\n' \
  "$slurm_partition" "$gpu_name" "$gpu_memory_mib" "$minimum_gpu_memory_mib"
printf 'git_commit=%s\n' "$(git rev-parse HEAD)"

common_args=(
  --dataset "$dataset"
  --data-dir "$data_dir"
  --external-dir data/external_test
  --benchmark-input-manifest "$benchmark_input_manifest"
  --require-full-ligand-atom-mapping
  --pocket-centers "$centers"
  --config "$config"
  --device cuda
  --no-confidence
  --output-dir "$raw_root"
  --protocol-id "$protocol_id"
  --num-samples "$num_samples"
  --num-steps "$num_steps"
  --prior-pool-size "$prior_pool_size"
  --sigma 2.0
  --time-schedule late
  --schedule-power 3.0
  --pocket-cutoff 10.0
  --center-jitter-sigma 0.0
  --refine none
  --vina-guidance-scale 0.0
  --unified-guidance-scale 0.0
  --fk-constraint-beta 0.0
  --fk-resample-translation-jitter 0.0
  --fk-resample-rotation-jitter 0.0
  --translation-sde-base-sigma 0.0
  --seed 42
  --num-shards "$num_shards"
  --shard-index "$shard_index"
  --expected-discovered-count "$expected_count"
  --require-complete-success
  --save-selected-poses
  --eligibility-manifest "$cohort_audit"
  "${only_id_args[@]}"
)

for arm_index in "${!arms[@]}"; do
  arm=${arms[$arm_index]}
  checkpoint=${checkpoints[$arm_index]}
  checkpoint_sha256=${checkpoint_sha256s[$arm_index]}
  if [[ "$mode" == smoke ]]; then
    run_name="effdock-early-time-t0p10-50k-external-paired-smoke-${dataset}-${arm}"
  else
    run_name="effdock-early-time-t0p10-50k-external-paired-${dataset}-${arm}"
  fi
  if [[ "$num_shards" -gt 1 ]]; then
    result_tag=$(printf '%s.shard-%03d-of-%03d' "$run_name" "$shard_index" "$num_shards")
  else
    result_tag=$run_name
  fi
  if [[ -e "$raw_root/$result_tag.csv" || -e "$raw_root/$result_tag.summary.json" ]]; then
    echo "refusing to overwrite existing arm output: $raw_root/$result_tag" >&2
    exit 2
  fi
  printf 'arm=%s checkpoint=%s checkpoint_sha256=%s run_name=%s\n' \
    "$arm" "$checkpoint" "$checkpoint_sha256" "$run_name"
  .venv/bin/eff-dock evaluate \
    "${common_args[@]}" \
    --checkpoint "$checkpoint" \
    --run-name "$run_name"
done
