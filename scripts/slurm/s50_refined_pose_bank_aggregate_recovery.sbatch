#!/usr/bin/env bash
#SBATCH --job-name=effdock-s50-refine-agg-recovery
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --partition=cpu_only
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:00

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
source_root="$repo_dir/outputs/eff-dock/s50-confidence-matched-runs/14cb2840f040b05127610a1f3dccd8aa5b2573c41156ff446898e2d212f1164b"
run_root="$repo_dir/outputs/eff-dock/s50-refined-pose-runs/c14df954c757ee092f8299224412c52f4305730348c797416fff4e73a3076050"
worker="$repo_dir/scripts/refine_s50_confidence_pose_bank.py"
protocol="$repo_dir/docs/S50_REFINED_POSE_BANK_PROTOCOL.md"
preflight="$run_root/preflight.json"
raw_bank="$source_root/full/bank_manifest.json"
input_manifest="$source_root/frozen/input_manifest.json"

verify_sha256 worker "$worker" c1b548307d5e1af5d200ef28a5d0fa89ccb48536661fed61aecfd102968b9d55
verify_sha256 protocol "$protocol" c3e34daca4402cac077488cb31c9ea1c88aaf7ae191b04712ce828c949aa25b0
verify_sha256 preflight "$preflight" be82dd962c2a7318da77823a374c36ffef9b73ea978a5a0657f16ee8009324cb
verify_sha256 raw-bank "$raw_bank" d45e36f3f2d75fb8ba9553715e1ec45031e4ad31881631d8632b6c19999f2d2b
verify_sha256 input-manifest "$input_manifest" 6a991e964715f9ddb51ce48ccbea5a948eb3b2d3c51de1f26929389365aec5f3
[[ ! -e "$run_root/full/manifest.json" ]] || die "refusing to overwrite sealed manifest"
[[ -x .venv/bin/python ]] || die "missing .venv Python"

export PYTHONPATH="$repo_dir:$repo_dir/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

.venv/bin/python "$worker" aggregate \
  --raw-bank-manifest "$raw_bank" \
  --input-manifest "$input_manifest" \
  --output-root "$run_root/full" \
  --train-shards 128 --val-shards 8
