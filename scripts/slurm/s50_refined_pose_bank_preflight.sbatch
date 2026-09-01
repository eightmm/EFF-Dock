#!/usr/bin/env bash
#SBATCH --job-name=effdock-s50-refine-preflight
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
  [[ "$actual" == "$expected" ]] || die "$label SHA mismatch"
}

repo_dir=$(cd "${SLURM_SUBMIT_DIR:?}" && pwd -P)
cd "$repo_dir"
source_root="$repo_dir/outputs/eff-dock/s50-confidence-matched-runs/14cb2840f040b05127610a1f3dccd8aa5b2573c41156ff446898e2d212f1164b"
content_id=c14df954c757ee092f8299224412c52f4305730348c797416fff4e73a3076050
output_root="$repo_dir/outputs/eff-dock/s50-refined-pose-runs/$content_id"
worker="$repo_dir/scripts/refine_s50_confidence_pose_bank.py"
protocol="$repo_dir/docs/S50_REFINED_POSE_BANK_PROTOCOL.md"
raw_bank="$source_root/full/bank_manifest.json"
input_manifest="$source_root/frozen/input_manifest.json"
calibration="$repo_dir/outputs/eff-dock/s50-refinement-calibration-runs/284796a567613b4f1b1a4d5183cec5c0d4dbc042ac0ccc03de6ce5383289e2a8/report.json"
verify_sha256 worker "$worker" 2a411292ce29404f9d1c4d5a282db85a346e8f482f52c232d4f007e3e6d34b54
verify_sha256 protocol "$protocol" e73199e0c1ccc343f4c6d258513034a24dc6387c7f13cd37c6af8aec7fb2185b
verify_sha256 raw-bank "$raw_bank" d45e36f3f2d75fb8ba9553715e1ec45031e4ad31881631d8632b6c19999f2d2b
verify_sha256 input-manifest "$input_manifest" 6a991e964715f9ddb51ce48ccbea5a948eb3b2d3c51de1f26929389365aec5f3
verify_sha256 calibration "$calibration" 3c7417e379bbb51273fda291cb958c1c74eeeeadad4d2e324353daf1b3065784
[[ -x .venv/bin/python ]] || die "missing .venv Python"
.venv/bin/python -c \
  'import json,sys; x=json.load(open(sys.argv[1])); assert x["status"] == "complete"; assert x["selected_arm"] == "displacement_100_b20"; assert x["comparisons"]["displacement_100_b20"]["passed"] is True' \
  "$calibration"

export PYTHONPATH="$repo_dir:$repo_dir/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
[[ ! -e "$output_root" ]] || die "refusing to reuse content root: $output_root"
mkdir -p "$output_root"
printf '%s\n' "$content_id" > "$output_root/CONTENT_IDENTITY.sha256"
.venv/bin/python "$worker" preflight \
  --raw-bank-manifest "$raw_bank" \
  --input-manifest "$input_manifest" \
  --output "$output_root/preflight.json"
