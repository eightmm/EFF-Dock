#!/usr/bin/env bash
#SBATCH --job-name=effdock-s50-refine-bank
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err
#SBATCH --partition=6000ada
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:1

set -euo pipefail

die() { echo "$*" >&2; exit 2; }
verify_sha256() {
  local label=$1 path=$2 expected=$3 actual
  [[ -f "$path" ]] || die "missing $label: $path"
  read -r actual _ < <(sha256sum "$path")
  [[ "$actual" == "$expected" ]] \
    || die "$label SHA mismatch: expected=$expected actual=$actual"
}

repo_dir=$(cd "${SLURM_SUBMIT_DIR:?}" && pwd -P)
cd "$repo_dir"
scope=${EFFDOCK_REFINE_SCOPE:?EFFDOCK_REFINE_SCOPE must be smoke or full}
case "$scope" in smoke|full) ;; *) die "invalid refinement scope: $scope" ;; esac

source_root="$repo_dir/outputs/eff-dock/s50-confidence-matched-runs/14cb2840f040b05127610a1f3dccd8aa5b2573c41156ff446898e2d212f1164b"
content_id=c14df954c757ee092f8299224412c52f4305730348c797416fff4e73a3076050
output_root="$repo_dir/outputs/eff-dock/s50-refined-pose-runs/$content_id"
raw_bank="$source_root/full/bank_manifest.json"
input_manifest="$source_root/frozen/input_manifest.json"
worker="$repo_dir/scripts/refine_s50_confidence_pose_bank.py"
protocol="$repo_dir/docs/S50_REFINED_POSE_BANK_PROTOCOL.md"
verify_sha256 worker "$worker" 2a411292ce29404f9d1c4d5a282db85a346e8f482f52c232d4f007e3e6d34b54
verify_sha256 protocol "$protocol" e73199e0c1ccc343f4c6d258513034a24dc6387c7f13cd37c6af8aec7fb2185b
verify_sha256 raw-bank "$raw_bank" d45e36f3f2d75fb8ba9553715e1ec45031e4ad31881631d8632b6c19999f2d2b
verify_sha256 input-manifest "$input_manifest" 6a991e964715f9ddb51ce48ccbea5a948eb3b2d3c51de1f26929389365aec5f3
[[ -x .venv/bin/python ]] || die "missing .venv Python"
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) -eq 1 ]] \
  || die "refinement task requires one visible GPU"

export PYTHONPATH="$repo_dir:$repo_dir/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
expected_implementation=56e26c3621c84ce4dc0d84e7b627a33941503a4dd646c0ff247ef35e13bf390e
expected_parameters=212a3314a9744729bd98cb4539fed61459619ddaa9b62de4455a2d363a8d4835
actual_implementation=$(
  .venv/bin/python -c \
    'from effdock.guidance.provenance import guidance_implementation_identity; print(guidance_implementation_identity()["sha256"])'
)
actual_parameters=$(
  .venv/bin/python -c \
    'from effdock.guidance.parameterization import guidance_parameter_identity; print(guidance_parameter_identity()["sha256"])'
)
[[ "$actual_implementation" == "$expected_implementation" ]] \
  || die "guidance implementation identity mismatch"
[[ "$actual_parameters" == "$expected_parameters" ]] \
  || die "guidance parameter identity mismatch"
[[ -f "$output_root/CONTENT_IDENTITY.sha256" ]] || die "missing content identity"
read -r recorded_content_id < "$output_root/CONTENT_IDENTITY.sha256"
[[ "$recorded_content_id" == "$content_id" ]] || die "content identity mismatch"
preflight="$output_root/preflight.json"
preflight_sidecar="$preflight.sha256"
[[ -f "$preflight_sidecar" ]] || die "missing label-free preflight sidecar"
read -r preflight_sha _ < "$preflight_sidecar"
verify_sha256 preflight "$preflight" "$preflight_sha"
.venv/bin/python -c \
  'import json,sys; x=json.load(open(sys.argv[1])); assert x["protocol_id"] == "EFFDOCK-S50-REFINED-POSE-BANK-V2"; assert x["status"] == "complete_label_free"; assert x["record_count"] == 44127; assert len(x["supported_and_observed_atomic_numbers"]) == 33' \
  "$preflight"

if [[ "$scope" == smoke ]]; then
  [[ ${SLURM_ARRAY_TASK_ID:-0} -eq 0 ]] || die "smoke must be a single task"
  smoke_root="$output_root/smoke"
  [[ ! -e "$smoke_root/SMOKE_COMPLETE" ]] || die "refusing to reuse completed smoke"
  .venv/bin/python "$worker" generate-shard \
    --raw-bank-manifest "$raw_bank" \
    --input-manifest "$input_manifest" \
    --output-root "$smoke_root" \
    --split train --shard-index 0 --num-shards 128 \
    --only-id '1h08__1__1.A__1.B_1.C__1.C' \
    --only-id '1wnu__1__1.A__1.C_1.D__1.D' \
    --only-id '1k2o__1__1.A__1.E_1.F_1.G__1.F' \
    --only-id '1l6g__1__1.A_1.B__1.C__1.C' \
    --device cuda
  .venv/bin/python "$worker" generate-shard \
    --raw-bank-manifest "$raw_bank" \
    --input-manifest "$input_manifest" \
    --output-root "$smoke_root" \
    --split val --shard-index 0 --num-shards 8 --limit 2 --device cuda
  printf 'protocol=%s\nworker_sha256=%s\njob_id=%s\n' \
    EFFDOCK-S50-REFINED-POSE-BANK-V2 \
    2a411292ce29404f9d1c4d5a282db85a346e8f482f52c232d4f007e3e6d34b54 \
    "${SLURM_JOB_ID:-unknown}" > "$smoke_root/SMOKE_COMPLETE.tmp"
  ln "$smoke_root/SMOKE_COMPLETE.tmp" "$smoke_root/SMOKE_COMPLETE"
  rm -f "$smoke_root/SMOKE_COMPLETE.tmp"
  exit 0
fi

[[ -f "$output_root/smoke/SMOKE_COMPLETE" ]] || die "missing completed refinement smoke"
task_id=${SLURM_ARRAY_TASK_ID:?full refinement requires array task ID}
if (( task_id < 128 )); then
  split=train
  shard_index=$task_id
  num_shards=128
elif (( task_id < 136 )); then
  split=val
  shard_index=$((task_id - 128))
  num_shards=8
else
  die "array task $task_id is outside 0..135"
fi
.venv/bin/python "$worker" generate-shard \
  --raw-bank-manifest "$raw_bank" \
  --input-manifest "$input_manifest" \
  --output-root "$output_root/full" \
  --split "$split" --shard-index "$shard_index" --num-shards "$num_shards" \
  --device cuda
