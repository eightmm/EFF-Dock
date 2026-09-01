#!/usr/bin/env bash
#SBATCH --job-name=effdock-s50-conf-report
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

die() {
  echo "$*" >&2
  exit 2
}

verify_sha256() {
  local label=$1 path=$2 expected=$3 actual
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || die "$label expected SHA-256 is invalid"
  [[ -f "$path" ]] || die "missing $label: $path"
  read -r actual _ < <(sha256sum "$path")
  [[ "$actual" == "$expected" ]] \
    || die "$label SHA-256 mismatch: expected=$expected actual=$actual path=$path"
}

read_sha_sidecar() {
  local label=$1 path=$2 sidecar=$3 expected declared
  [[ -f "$sidecar" ]] || die "missing $label SHA sidecar: $sidecar"
  read -r expected declared < "$sidecar"
  verify_sha256 "$label" "$path" "$expected"
  printf '%s' "$expected"
}

seal_sha_sidecar() {
  local path=$1 sidecar=$2 digest temporary
  [[ -f "$path" ]] || die "cannot seal missing file: $path"
  [[ ! -e "$sidecar" ]] || die "refusing to overwrite SHA sidecar: $sidecar"
  read -r digest _ < <(sha256sum "$path")
  temporary="${sidecar}.tmp.${SLURM_JOB_ID:-$$}"
  printf '%s  %s\n' "$digest" "$path" > "$temporary"
  ln "$temporary" "$sidecar" || { rm -f "$temporary"; die "failed to seal $sidecar"; }
  rm -f "$temporary"
}

verify_runtime_environment() {
  local expected=${EFFDOCK_ENVIRONMENT_SHA256:?EFFDOCK_ENVIRONMENT_SHA256 is required}
  local snapshot="$run_root/RUNTIME_ENVIRONMENT.json"
  local probe=scripts/capture_s50_runtime_environment.py
  local actual
  verify_sha256 runtime-environment "$snapshot" "$expected"
  [[ -x .venv/bin/python ]] || die "missing capsule .venv/bin/python"
  [[ -f "$probe" ]] || die "missing runtime-environment probe: $probe"
  read -r actual _ < <(.venv/bin/python "$probe" | sha256sum)
  [[ "$actual" == "$expected" ]] \
    || die "linked .venv runtime changed: expected=$expected actual=$actual"
}

repo_dir=${EFFDOCK_REPO_DIR:?EFFDOCK_REPO_DIR must name the execution capsule}
repo_dir=$(cd "$repo_dir" && pwd -P)
cd "$repo_dir"
run_root=${EFFDOCK_RUN_ROOT:?EFFDOCK_RUN_ROOT is required}
run_root=$(readlink -f "$run_root")
content_sha256=${EFFDOCK_CONTENT_SHA256:?EFFDOCK_CONTENT_SHA256 is required}
read -r recorded_content_sha256 < "$run_root/CONTENT_IDENTITY.sha256"
[[ "$recorded_content_sha256" == "$content_sha256" ]] || die "content identity mismatch"
verify_runtime_environment

protocol=docs/S50_MATCHED_CONFIDENCE_TRAINING_PROTOCOL.md
protocol_sha256=${EFFDOCK_PROTOCOL_SHA256:?EFFDOCK_PROTOCOL_SHA256 is required}
reporter=scripts/report_s50_confidence_training.py
reporter_sha256=${EFFDOCK_REPORTER_SHA256:?EFFDOCK_REPORTER_SHA256 is required}
verify_sha256 protocol "$protocol" "$protocol_sha256"
verify_sha256 reporter "$reporter" "$reporter_sha256"

input_manifest="$run_root/frozen/input_manifest.json"
input_sha256=$(read_sha_sidecar \
  input-manifest "$input_manifest" "$input_manifest.sha256")
bank_manifest="$run_root/full/bank_manifest.json"
bank_sha256=$(read_sha_sidecar \
  full-bank-manifest "$bank_manifest" "$bank_manifest.sha256")
training_dir="$run_root/training"
read_sha_sidecar training-metrics \
  "$training_dir/metrics.json" "$training_dir/metrics.json.sha256" >/dev/null
output="$training_dir/selection_report.json"
[[ ! -e "$output" ]] || die "refusing to overwrite selection report"
mkdir -p "$run_root/ledgers"

export PYTHONPATH="$repo_dir/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
command=(
  .venv/bin/python "$reporter"
  --training-dir "$training_dir"
  --bank-manifest "$bank_manifest"
  --expected-bank-manifest-sha256 "$bank_sha256"
  --frozen-input-manifest "$input_manifest"
  --expected-frozen-input-manifest-sha256 "$input_sha256"
  --output "$output"
)

oms research-runner \
  --question "Does sampler-matched confidence adaptation recover more fixed S50 sub-2-A candidates than its own U0 warm start?" \
  --hypothesis "S50/N100/S10/sigma-2 matched confidence training improves pure predicted-RMSD Top-1 selection." \
  --prediction "The best fixed 5k look improves Top-1 SR2 by at least 3 percentage points and has paired cluster-bootstrap CI lower bound above zero." \
  --baseline "U0 weights from retained confidence step42500 SHA e31fde6f351284205c78f7a1510002779c43312e94d9f82003d47a14d72bc78f on the identical validation bank" \
  --metric "sample-weighted eval_top1_lt2 against symmetry-aware no-alignment RMSD on the exact 1035 eligible PLINDER validation samples" \
  --success "delta >= +3.0 pp and PLINDER-system cluster-bootstrap 95% CI lower bound > 0; otherwise retain the old confidence checkpoint" \
  --change "confidence training bank matched to the frozen S50 sampler; sampler, architecture, loss, selector, and validation bank fixed" \
  --file "$run_root/ledgers/training-report.jsonl" \
  --metrics "$output" \
  --no-gate \
  --reason "all scientific and integrity gates are implemented by the hash-pinned report command" \
  -- "${command[@]}"

seal_sha_sidecar "$output" "$output.sha256"
printf 'report=%s content_sha256=%s\n' "$output" "$content_sha256"
