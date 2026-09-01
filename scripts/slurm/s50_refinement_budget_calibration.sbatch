#!/usr/bin/env bash
#SBATCH --job-name=effdock-s50-refine-cal
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --partition=6000ada
#SBATCH --qos=veryshort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
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
source_root="$repo_dir/outputs/eff-dock/s50-confidence-matched-runs/14cb2840f040b05127610a1f3dccd8aa5b2573c41156ff446898e2d212f1164b"
content_id=284796a567613b4f1b1a4d5183cec5c0d4dbc042ac0ccc03de6ce5383289e2a8
output_root="$repo_dir/outputs/eff-dock/s50-refinement-calibration-runs/$content_id"
report="$output_root/report.json"
worker="$repo_dir/scripts/calibrate_s50_refinement_budget.py"
protocol="$repo_dir/docs/S50_REFINEMENT_BUDGET_CALIBRATION_PROTOCOL.md"
raw_bank="$source_root/full/bank_manifest.json"
input_manifest="$source_root/frozen/input_manifest.json"

verify_sha256 worker "$worker" 4c9bb73411c3a9d871f9d891cdaee1c4226c2a532da85b944ce306360b11e817
verify_sha256 protocol "$protocol" 7402b98075633978454c82a5d3d175e82d2eff72f314f8b6c866361a4115fc25
verify_sha256 raw-bank "$raw_bank" d45e36f3f2d75fb8ba9553715e1ec45031e4ad31881631d8632b6c19999f2d2b
verify_sha256 input-manifest "$input_manifest" 6a991e964715f9ddb51ce48ccbea5a948eb3b2d3c51de1f26929389365aec5f3
[[ -x .venv/bin/python ]] || die "missing .venv Python"
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) -eq 1 ]] \
  || die "calibration requires exactly one visible GPU"
[[ ! -e "$report" ]] || die "refusing to overwrite $report"
mkdir -p "$output_root"

export PYTHONPATH="$repo_dir:$repo_dir/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

.venv/bin/python "$worker" \
  --raw-bank-manifest "$raw_bank" \
  --input-manifest "$input_manifest" \
  --output "$report" \
  --sample-id '1h08__1__1.A__1.B_1.C__1.C' \
  --sample-id '3o7u__1__1.A__1.C_1.E__1.E' \
  --sample-id '1a5s__1__1.A__1.C__1.C' \
  --arm baseline_100 --arm batch20_100 --arm displacement_100_b20 \
  --batch-size 10 --device cuda
