#!/usr/bin/env bash
#SBATCH --job-name=effdock-s50-sym-labels
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err
#SBATCH --partition=cpu_only
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00

set -euo pipefail

die() { echo "$*" >&2; exit 2; }
verify_sha256() {
  local label=$1 path=$2 expected=$3 actual
  [[ -f "$path" && ! -L "$path" ]] || die "missing/nonregular $label: $path"
  read -r actual _ < <(sha256sum "$path")
  [[ "$actual" == "$expected" ]] || die "$label SHA-256 mismatch"
}

repo_dir=$(cd "${SLURM_SUBMIT_DIR:?}" && pwd -P)
cd "$repo_dir"
stage=${EFFDOCK_SYM_LABEL_STAGE:?EFFDOCK_SYM_LABEL_STAGE must be smoke, full, or aggregate}
case "$stage" in smoke|full|aggregate) ;; *) die "invalid stage: $stage" ;; esac

worker="$repo_dir/scripts/build_s50_symmetry_rmsd_sidecars.py"
builder="$repo_dir/scripts/prepare_s50_confidence_training_bank.py"
protocol="$repo_dir/docs/S50_SYMMETRY_LABEL_SIDECAR_PROTOCOL.md"
source_root="$repo_dir/outputs/eff-dock/s50-confidence-matched-runs/14cb2840f040b05127610a1f3dccd8aa5b2573c41156ff446898e2d212f1164b"
input_manifest="$source_root/frozen/input_manifest.json"
bank_manifest="$source_root/full/bank_manifest.json"
output_root="$repo_dir/outputs/eff-dock/s50-confidence-symmetry-labels/41c5ff349ba95e64b45ac43356b970dc037ebf31898c6533e61a2360880f031f"

worker_sha=6ef49bb7e12f66d0e7d3becea77f5994ee5243cf9ccded62bba357751b78e060
builder_sha=0d1fb50e98345d0712862ff8a262fb8e08d8a2d231d2b0288facb6053ae19d51
protocol_sha=47465aed0068dfbbfce062c59ad3095f97c4614f5a8f20d422a67674980a1933
input_sha=6a991e964715f9ddb51ce48ccbea5a948eb3b2d3c51de1f26929389365aec5f3
bank_sha=d45e36f3f2d75fb8ba9553715e1ec45031e4ad31881631d8632b6c19999f2d2b
content_id=41c5ff349ba95e64b45ac43356b970dc037ebf31898c6533e61a2360880f031f

verify_sha256 worker "$worker" "$worker_sha"
verify_sha256 frozen-builder "$builder" "$builder_sha"
verify_sha256 protocol "$protocol" "$protocol_sha"
verify_sha256 input-manifest "$input_manifest" "$input_sha"
verify_sha256 bank-manifest "$bank_manifest" "$bank_sha"
[[ -x "$repo_dir/.venv/bin/python" ]] || die "missing .venv Python"
[[ -f "$output_root/CONTENT_IDENTITY.sha256" ]] || die "missing output content identity"
read -r recorded_content_id < "$output_root/CONTENT_IDENTITY.sha256"
[[ "$recorded_content_id" == "$content_id" ]] || die "output content identity mismatch"

export PYTHONPATH="$repo_dir:$repo_dir/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

common=(
  --input-manifest "$input_manifest"
  --expected-input-manifest-sha256 "$input_sha"
  --bank-manifest "$bank_manifest"
  --expected-bank-manifest-sha256 "$bank_sha"
  --split train
  --num-shards 128
)

case "$stage" in
  smoke)
    [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || die "smoke must not be an array task"
    [[ ! -e "$output_root/smoke" ]] || die "refusing to reuse smoke output"
    "$repo_dir/.venv/bin/python" "$worker" generate-shard \
      "${common[@]}" --output-root "$output_root/smoke" --shard-index 0 --limit 2
    ;;
  full)
    task_id=${SLURM_ARRAY_TASK_ID:?full stage requires an array task}
    (( task_id >= 0 && task_id < 128 )) || die "array task must be in 0..127"
    [[ -f "$output_root/smoke/shards/train/shard-000-of-128.summary.json" ]] \
      || die "missing successful real-payload smoke"
    mkdir -p "$output_root/full/.reservations"
    mkdir "$output_root/full/.reservations/shard-$(printf '%03d' "$task_id")" \
      || die "shard reservation already exists"
    "$repo_dir/.venv/bin/python" "$worker" generate-shard \
      "${common[@]}" --output-root "$output_root/full" --shard-index "$task_id"
    ;;
  aggregate)
    [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || die "aggregate must not be an array task"
    "$repo_dir/.venv/bin/python" "$worker" aggregate \
      "${common[@]}" --output-root "$output_root/full" \
      --output-manifest "$output_root/full/manifest.json"
    read -r manifest_sha _ < <(sha256sum "$output_root/full/manifest.json")
    printf '%s  %s\n' "$manifest_sha" "$output_root/full/manifest.json" \
      > "$output_root/full/manifest.json.sha256.tmp"
    ln "$output_root/full/manifest.json.sha256.tmp" "$output_root/full/manifest.json.sha256"
    rm -f "$output_root/full/manifest.json.sha256.tmp"
    ;;
esac
