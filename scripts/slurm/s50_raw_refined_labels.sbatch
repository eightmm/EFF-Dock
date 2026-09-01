#!/usr/bin/env bash
#SBATCH --job-name=effdock-s50-mix-labels
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
repo_dir=$(cd "${SLURM_SUBMIT_DIR:?}" && pwd -P)
cd "$repo_dir"
stage=${EFFDOCK_MIX_LABEL_STAGE:?stage must be smoke, full, or aggregate}
case "$stage" in smoke|full|aggregate) ;; *) die "invalid stage: $stage" ;; esac
worker=${EFFDOCK_MIX_LABEL_WORKER:?}
read -r actual _ < <(sha256sum "$worker")
[[ "$actual" == "${EFFDOCK_MIX_LABEL_WORKER_SHA:?}" ]] || die "label worker SHA mismatch"
read -r actual _ < <(sha256sum "${EFFDOCK_MIX_BUILDER:?}")
[[ "$actual" == "${EFFDOCK_MIX_BUILDER_SHA:?}" ]] || die "builder SHA mismatch"
bank_manifest=${EFFDOCK_MIX_BANK_OUTPUT_ROOT:?}/full/bank_manifest.json
[[ -f "$bank_manifest.sha256" ]] || die "missing bank manifest sidecar"
read -r bank_sha bank_path < "$bank_manifest.sha256"
[[ "$bank_path" == "$bank_manifest" ]] || die "bank sidecar path mismatch"
[[ $(sha256sum "$bank_manifest" | cut -d' ' -f1) == "$bank_sha" ]] \
  || die "bank manifest SHA mismatch"
output_root=${EFFDOCK_MIX_LABEL_OUTPUT_ROOT:?}
export PYTHONPATH="$repo_dir:$repo_dir/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
common=(
  --input-manifest "${EFFDOCK_MIX_INPUT_MANIFEST:?}"
  --expected-input-manifest-sha256 "${EFFDOCK_MIX_INPUT_MANIFEST_SHA:?}"
  --bank-manifest "$bank_manifest" --expected-bank-manifest-sha256 "$bank_sha"
  --expected-pose-tag s50_n100_s10_sig2_latep3_pc10_rdkitseed0_refine100
  --split train --num-shards 128
)
case "$stage" in
  smoke)
    [[ ! -e "$output_root/smoke" ]] || die "refusing to reuse label smoke"
    .venv/bin/python "$worker" generate-shard "${common[@]}" \
      --output-root "$output_root/smoke" --shard-index 0 --limit 2
    ;;
  full)
    task_id=${SLURM_ARRAY_TASK_ID:?full labels require array task ID}
    (( task_id >= 0 && task_id < 128 )) || die "array task must be 0..127"
    [[ -f "$output_root/smoke/shards/train/shard-000-of-128.summary.json" ]] \
      || die "missing label smoke"
    mkdir -p "$output_root/full/.reservations"
    mkdir "$output_root/full/.reservations/shard-$(printf '%03d' "$task_id")" \
      || die "label shard reservation already exists"
    .venv/bin/python "$worker" generate-shard "${common[@]}" \
      --output-root "$output_root/full" --shard-index "$task_id"
    ;;
  aggregate)
    .venv/bin/python "$worker" aggregate "${common[@]}" \
      --output-root "$output_root/full" --output-manifest "$output_root/full/manifest.json"
    sha=$(sha256sum "$output_root/full/manifest.json" | cut -d' ' -f1)
    printf '%s  %s\n' "$sha" "$output_root/full/manifest.json" \
      > "$output_root/full/manifest.json.sha256.tmp"
    ln "$output_root/full/manifest.json.sha256.tmp" \
      "$output_root/full/manifest.json.sha256"
    rm -f "$output_root/full/manifest.json.sha256.tmp"
    ;;
esac
