#!/usr/bin/env bash
#SBATCH --job-name=effdock-s50-refsym-labels
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
stage=${EFFDOCK_REFINED_SYM_LABEL_STAGE:?stage must be smoke, full, or aggregate}
case "$stage" in smoke|full|aggregate) ;; *) die "invalid stage: $stage" ;; esac

worker="$repo_dir/scripts/build_s50_symmetry_rmsd_sidecars.py"
builder="$repo_dir/scripts/prepare_s50_confidence_training_bank.py"
protocol="$repo_dir/docs/S50_REFINED_SYMMETRY_LABEL_SIDECAR_PROTOCOL.md"
source_root="$repo_dir/outputs/eff-dock/s50-confidence-matched-runs/14cb2840f040b05127610a1f3dccd8aa5b2573c41156ff446898e2d212f1164b"
refined_root="$repo_dir/outputs/eff-dock/s50-refined-pose-runs/207611a7fed8805b7f52cdf4648da2ea03e7d1e699c7d47106b632215634fdc2/full"
input_manifest="$source_root/frozen/input_manifest.json"
bank_manifest="$refined_root/confidence_bank/bank_manifest.json"
bank_sidecar="$bank_manifest.sha256"
output_root="$refined_root/symmetry_labels_v1"
pose_tag=s50_n100_s10_sig2_latep3_pc10_rdkitseed0_refine100

verify_sha256 worker "$worker" f426df3f6783e491e4faa114a8a0a70d18d89e16d1c08716016440a23361d2a2
verify_sha256 builder "$builder" 0d1fb50e98345d0712862ff8a262fb8e08d8a2d231d2b0288facb6053ae19d51
verify_sha256 protocol "$protocol" d5b1caf9199a38f6d8361a65addadddfcd4a99e2b492005c976793b6386a6905
verify_sha256 input-manifest "$input_manifest" 6a991e964715f9ddb51ce48ccbea5a948eb3b2d3c51de1f26929389365aec5f3
[[ -f "$bank_sidecar" ]] || die "missing sealed refined bank sidecar"
read -r bank_sha bank_path < "$bank_sidecar"
[[ "$bank_path" == "$bank_manifest" ]] || die "refined bank sidecar path mismatch"
verify_sha256 refined-bank "$bank_manifest" "$bank_sha"
[[ -x .venv/bin/python ]] || die "missing .venv Python"

export PYTHONPATH="$repo_dir:$repo_dir/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
common=(
  --input-manifest "$input_manifest"
  --expected-input-manifest-sha256 6a991e964715f9ddb51ce48ccbea5a948eb3b2d3c51de1f26929389365aec5f3
  --bank-manifest "$bank_manifest"
  --expected-bank-manifest-sha256 "$bank_sha"
  --expected-pose-tag "$pose_tag"
  --split train
  --num-shards 128
)

case "$stage" in
  smoke)
    [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || die "smoke must not be an array"
    [[ ! -e "$output_root/smoke" ]] || die "refusing to reuse smoke output"
    .venv/bin/python "$worker" generate-shard "${common[@]}" \
      --output-root "$output_root/smoke" --shard-index 0 --limit 2
    ;;
  full)
    task_id=${SLURM_ARRAY_TASK_ID:?full stage requires array task ID}
    (( task_id >= 0 && task_id < 128 )) || die "array task must be 0..127"
    [[ -f "$output_root/smoke/shards/train/shard-000-of-128.summary.json" ]] \
      || die "missing successful refined-label smoke"
    mkdir -p "$output_root/full/.reservations"
    mkdir "$output_root/full/.reservations/shard-$(printf '%03d' "$task_id")" \
      || die "shard reservation already exists"
    .venv/bin/python "$worker" generate-shard "${common[@]}" \
      --output-root "$output_root/full" --shard-index "$task_id"
    ;;
  aggregate)
    [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || die "aggregate must not be an array"
    .venv/bin/python "$worker" aggregate "${common[@]}" \
      --output-root "$output_root/full" \
      --output-manifest "$output_root/full/manifest.json"
    read -r manifest_sha _ < <(sha256sum "$output_root/full/manifest.json")
    printf '%s  %s\n' "$manifest_sha" "$output_root/full/manifest.json" \
      > "$output_root/full/manifest.json.sha256.tmp"
    ln "$output_root/full/manifest.json.sha256.tmp" \
      "$output_root/full/manifest.json.sha256"
    rm -f "$output_root/full/manifest.json.sha256.tmp"
    ;;
esac
