#!/usr/bin/env bash
# Submit the reviewed, content-addressed S50 confidence adaptation DAG.

set -euo pipefail

die() {
  echo "$*" >&2
  exit 2
}

usage() {
  echo "usage: $0" >&2
}

train_world_size=4
if [[ $# -ne 0 ]]; then
  usage
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$repo_root"
oms snapshot --cluster --check >/dev/null

protocol_id=EFFDOCK-S50-MATCHED-CONFIDENCE-TRAIN-VAL-V1
split_sha256=3ac570bf08bced053f1ce040b57efca27c3be616f29a82cd66ef887c08860e6b
pool_sha256=0ff455da77ce5540b839918cccb96f45414e91efff6272d7da3a65337ab1fe91
docking_config_sha256=39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec
s50_checkpoint_sha256=65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6
confidence_init_sha256=e31fde6f351284205c78f7a1510002779c43312e94d9f82003d47a14d72bc78f
val_bank_sha256=928b7219ed1ef8375c1ee52470f6ef606b8fca4d5bf4ea5c51355e8332e29a4b

declare -A frozen_inputs=(
  [data/splits/plinder.json]="$split_sha256"
  [data/plinder_pool.parquet]="$pool_sha256"
  [configs/train.yaml]="$docking_config_sha256"
  [outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step50000_ema_common_init.pt]="$s50_checkpoint_sha256"
  [weights/effdock_confidence_extmatch_n80_s25_step42500.pt]="$confidence_init_sha256"
  [outputs/benchmarks/early_time_sampler_s50_confidence_runs/frozen_inputs/label_free_bank.v2.json]="$val_bank_sha256"
)
for path in "${!frozen_inputs[@]}"; do
  [[ -f "$path" ]] || die "missing frozen input: $path"
  read -r actual _ < <(sha256sum "$path")
  [[ "$actual" == "${frozen_inputs[$path]}" ]] \
    || die "frozen input hash mismatch: $path"
done
[[ -d data/plinder_processed ]] || die "missing data/plinder_processed"

required_files=(
  pyproject.toml
  uv.lock
  configs/train.yaml
  configs/train_confidence_s50_matched.yaml
  docs/S50_MATCHED_CONFIDENCE_TRAINING_PROTOCOL.md
  scripts/capture_s50_runtime_environment.py
  scripts/create_execution_capsule.py
  scripts/prepare_s50_confidence_training_bank.py
  scripts/report_s50_confidence_training.py
  scripts/slurm/s50_confidence_bank_control.sbatch
  scripts/slurm/s50_confidence_bank_array.sbatch
  scripts/slurm/s50_confidence_train_matched.sbatch
  scripts/slurm/s50_confidence_training_report.sbatch
  scripts/slurm/submit_s50_confidence_training.sh
)
for path in "${required_files[@]}"; do
  [[ -f "$path" ]] || die "missing reviewed runtime file: $path"
done
[[ -x .venv/bin/python && -x .venv/bin/torchrun && -x .venv/bin/eff-dock ]] \
  || die "missing project virtual-environment entry points"
command -v scontrol >/dev/null || die "missing scontrol for held-DAG release"

runtime_environment_json=$(.venv/bin/python scripts/capture_s50_runtime_environment.py)
runtime_environment_sha256=$(printf '%s\n' "$runtime_environment_json" | sha256sum | awk '{print $1}')
[[ "$runtime_environment_sha256" =~ ^[0-9a-f]{64}$ ]] \
  || die "invalid runtime-environment identity"

for script in scripts/slurm/s50_confidence_bank_control.sbatch \
  scripts/slurm/s50_confidence_bank_array.sbatch \
  scripts/slurm/s50_confidence_train_matched.sbatch \
  scripts/slurm/s50_confidence_training_report.sbatch \
  scripts/slurm/submit_s50_confidence_training.sh; do
  bash -n "$script"
done
.venv/bin/python -m py_compile \
  scripts/capture_s50_runtime_environment.py \
  scripts/prepare_s50_confidence_training_bank.py \
  scripts/report_s50_confidence_training.py
.venv/bin/python -m pytest -q \
  tests/test_prepare_s50_confidence_training_bank.py \
  tests/test_confidence_bank_training.py \
  tests/test_report_s50_confidence_training.py

mapfile -t package_files < <(rg --files src/effdock -g '*.py' | sort)
copy_files=("${package_files[@]}" "${required_files[@]}")
mapfile -t copy_files < <(printf '%s\n' "${copy_files[@]}" | sort -u)
runtime_sha256=$(
  {
    printf 'EFFDOCK_S50_MATCHED_CONFIDENCE_CAPSULE_V1\0'
    for path in "${copy_files[@]}"; do
      printf '%s\0' "$path"
      sha256sum "$path"
    done
  } | sha256sum | awk '{print $1}'
)
content_sha256=$(
  printf '%s\0%s\0%s\0%s\0%s\0%s\0%s\0%s\0%s\0%s\0' \
    "$protocol_id" "$runtime_sha256" "$split_sha256" "$pool_sha256" \
    "$docking_config_sha256" "$s50_checkpoint_sha256" \
    "$confidence_init_sha256" "$val_bank_sha256" "$train_world_size" \
    "$runtime_environment_sha256" \
    | sha256sum | awk '{print $1}'
)

run_root_rel="outputs/eff-dock/s50-confidence-matched-runs/$content_sha256"
run_root="$repo_root/$run_root_rel"
execution_root_rel=".effdock_execution_capsules/$protocol_id/$content_sha256"
execution_root="$repo_root/$execution_root_rel"
[[ ! -e "$run_root" && ! -L "$run_root" ]] \
  || die "refusing to reuse content root: $run_root"
[[ ! -e "$execution_root" && ! -L "$execution_root" ]] \
  || die "refusing to reuse execution capsule: $execution_root"
mkdir -p "$(dirname "$run_root")" outputs/slurm
mkdir "$run_root"
printf '%s\n' "$content_sha256" > "$run_root/CONTENT_IDENTITY.sha256"
printf '%s\n' "$runtime_environment_json" > "$run_root/RUNTIME_ENVIRONMENT.json"
read -r sealed_environment_sha256 _ < <(sha256sum "$run_root/RUNTIME_ENVIRONMENT.json")
[[ "$sealed_environment_sha256" == "$runtime_environment_sha256" ]] \
  || die "runtime-environment snapshot changed while sealing"
printf '%s  %s\n' "$runtime_environment_sha256" \
  "$run_root/RUNTIME_ENVIRONMENT.json" > "$run_root/RUNTIME_ENVIRONMENT.sha256"
chmod 0444 "$run_root/CONTENT_IDENTITY.sha256" \
  "$run_root/RUNTIME_ENVIRONMENT.json" "$run_root/RUNTIME_ENVIRONMENT.sha256"

capsule_args=()
for path in "${copy_files[@]}"; do capsule_args+=(--copy-file "$path"); done
.venv/bin/python scripts/create_execution_capsule.py \
  --repo-root "$repo_root" \
  --output "$execution_root" \
  --link-root .venv \
  --link-root data \
  --link-root weights \
  --link-root outputs \
  "${capsule_args[@]}" >/dev/null

sealed_runtime_sha256=$(
  {
    printf 'EFFDOCK_S50_MATCHED_CONFIDENCE_CAPSULE_V1\0'
    for path in "${copy_files[@]}"; do
      printf '%s\0' "$path"
      (cd "$execution_root" && sha256sum "$path")
    done
  } | sha256sum | awk '{print $1}'
)
[[ "$sealed_runtime_sha256" == "$runtime_sha256" ]] \
  || die "execution capsule runtime changed while sealing: expected=$runtime_sha256 actual=$sealed_runtime_sha256"

for path in "${!frozen_inputs[@]}"; do
  capsule_path="$execution_root/$path"
  [[ -f "$capsule_path" ]] || die "execution capsule is missing frozen input: $path"
  read -r actual _ < <(sha256sum "$capsule_path")
  [[ "$actual" == "${frozen_inputs[$path]}" ]] \
    || die "execution capsule frozen-input hash mismatch: $path"
done
[[ -d "$execution_root/data/plinder_processed" ]] \
  || die "execution capsule is missing processed PLINDER root"
[[ -x "$execution_root/.venv/bin/python" \
   && -x "$execution_root/.venv/bin/torchrun" \
   && -x "$execution_root/.venv/bin/eff-dock" ]] \
  || die "execution capsule is missing virtual-environment entry points"

read -r builder_sha256 _ < <(
  sha256sum "$execution_root/scripts/prepare_s50_confidence_training_bank.py"
)
read -r train_config_sha256 _ < <(
  sha256sum "$execution_root/configs/train_confidence_s50_matched.yaml"
)
read -r protocol_sha256 _ < <(
  sha256sum "$execution_root/docs/S50_MATCHED_CONFIDENCE_TRAINING_PROTOCOL.md"
)
read -r reporter_sha256 _ < <(
  sha256sum "$execution_root/scripts/report_s50_confidence_training.py"
)

git_commit=$(git rev-parse HEAD)
read -r git_diff_sha256 _ < <(git diff --no-ext-diff | sha256sum)
printf '%s\n' \
  'status=submitting' \
  "protocol_id=$protocol_id" \
  "content_sha256=$content_sha256" \
  "runtime_sha256=$runtime_sha256" \
  "run_root=$run_root" \
  "execution_root=$execution_root" \
  "builder_sha256=$builder_sha256" \
  "train_config_sha256=$train_config_sha256" \
  "protocol_sha256=$protocol_sha256" \
  "reporter_sha256=$reporter_sha256" \
  "runtime_environment_sha256=$runtime_environment_sha256" \
  "train_world_size=$train_world_size" \
  "git_commit=$git_commit" \
  "git_diff_sha256=$git_diff_sha256" \
  > "$run_root/.submission.pending"

base_export="ALL,EFFDOCK_REPO_DIR=$execution_root,EFFDOCK_RUN_ROOT=$run_root"
base_export+=",EFFDOCK_CONTENT_SHA256=$content_sha256,EFFDOCK_BUILDER_SHA256=$builder_sha256"
base_export+=",EFFDOCK_TRAIN_CONFIG_SHA256=$train_config_sha256"
base_export+=",EFFDOCK_PROTOCOL_SHA256=$protocol_sha256,EFFDOCK_REPORTER_SHA256=$reporter_sha256"
base_export+=",EFFDOCK_ENVIRONMENT_SHA256=$runtime_environment_sha256"

submitted=()
committed=0
cleanup() {
  code=$?
  trap - EXIT
  if [[ "$committed" -eq 0 ]]; then
    for job in "${submitted[@]}"; do scancel "$job" 2>/dev/null || true; done
  fi
  exit "$code"
}
trap cleanup EXIT

submit() {
  local dependency=$1 script=$2 export_spec=$3 array=$4 partition=$5 gres=$6
  shift 6
  local args raw job
  args=(--parsable --partition="$partition" --export="$export_spec")
  [[ -z "$dependency" ]] || args+=(--dependency="afterok:$dependency")
  [[ -z "$array" ]] || args+=(--array="$array")
  [[ -z "$gres" ]] || args+=(--gres="$gres")
  args+=("$@")
  raw=$(sbatch "${args[@]}" "$script")
  job=${raw%%;*}
  [[ "$job" =~ ^[0-9]+$ ]] || die "invalid sbatch response: $raw"
  printf '%s' "$job"
}

submit_held_root() {
  local script=$1 export_spec=$2 raw job
  raw=$(sbatch --parsable --hold --partition=cpu_only \
    --export="$export_spec" "$script")
  job=${raw%%;*}
  [[ "$job" =~ ^[0-9]+$ ]] || die "invalid held sbatch response: $raw"
  printf '%s' "$job"
}

control="$execution_root/scripts/slurm/s50_confidence_bank_control.sbatch"
array_script="$execution_root/scripts/slurm/s50_confidence_bank_array.sbatch"
train_script="$execution_root/scripts/slurm/s50_confidence_train_matched.sbatch"
report_script="$execution_root/scripts/slurm/s50_confidence_training_report.sbatch"

preflight_job=$(submit_held_root "$control" "$base_export,EFFDOCK_STAGE=freeze")
submitted+=("$preflight_job")
smoke_train_bank_job=$(submit "$preflight_job" "$array_script" \
  "$base_export,EFFDOCK_BANK_SCOPE=smoke,EFFDOCK_BANK_SPLIT=train" '0' 6000ada gpu:1)
submitted+=("$smoke_train_bank_job")
smoke_val_bank_job=$(submit "$preflight_job" "$array_script" \
  "$base_export,EFFDOCK_BANK_SCOPE=smoke,EFFDOCK_BANK_SPLIT=val" '0' 6000ada gpu:1)
submitted+=("$smoke_val_bank_job")
smoke_aggregate_job=$(submit "$smoke_train_bank_job:$smoke_val_bank_job" "$control" \
  "$base_export,EFFDOCK_STAGE=aggregate-smoke" '' cpu_only '')
submitted+=("$smoke_aggregate_job")
smoke_training_job=$(submit "$smoke_aggregate_job" "$train_script" \
  "$base_export,EFFDOCK_TRAIN_SCOPE=smoke,EFFDOCK_TRAIN_WORLD_SIZE=4" '' heavy gpu:4 \
  --cpus-per-task=16 --mem=200G --qos=normal --time=04:00:00)
submitted+=("$smoke_training_job")

full_train_bank_job=$(submit "$smoke_training_job" "$array_script" \
  "$base_export,EFFDOCK_BANK_SCOPE=full,EFFDOCK_BANK_SPLIT=train" '0-127%8' 6000ada gpu:1)
submitted+=("$full_train_bank_job")
full_val_bank_job=$(submit "$smoke_training_job" "$array_script" \
  "$base_export,EFFDOCK_BANK_SCOPE=full,EFFDOCK_BANK_SPLIT=val" '0-7%8' 6000ada gpu:1)
submitted+=("$full_val_bank_job")
full_aggregate_job=$(submit "$full_train_bank_job:$full_val_bank_job" "$control" \
  "$base_export,EFFDOCK_STAGE=aggregate-full" '' cpu_only '')
submitted+=("$full_aggregate_job")

full_training_job=$(submit "$full_aggregate_job" "$train_script" \
  "$base_export,EFFDOCK_TRAIN_SCOPE=full,EFFDOCK_TRAIN_WORLD_SIZE=4" '' heavy gpu:4 \
  --qos=long --time=2-23:59:00)
submitted+=("$full_training_job")
report_job=$(submit "$full_training_job" "$report_script" "$base_export" '' cpu_only '')
submitted+=("$report_job")

printf '%s\n' \
  "preflight_job=$preflight_job" \
  "smoke_train_bank_job=$smoke_train_bank_job" \
  "smoke_val_bank_job=$smoke_val_bank_job" \
  "smoke_aggregate_job=$smoke_aggregate_job" \
  "smoke_training_job=$smoke_training_job" \
  "full_train_bank_job=$full_train_bank_job" \
  "full_val_bank_job=$full_val_bank_job" \
  "full_aggregate_job=$full_aggregate_job" \
  "full_training_job=$full_training_job" \
  "report_job=$report_job" \
  "preflight_initial_state=held" \
  "release_after_atomic_metadata_commit=true" \
  "submitted_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >> "$run_root/.submission.pending"
sed -i 's/^status=submitting$/status=submitted/' "$run_root/.submission.pending"
mv "$run_root/.submission.pending" "$run_root/.submission"
scontrol release "$preflight_job"
committed=1
trap - EXIT

printf 'run_root=%s\ncontent_sha256=%s\n' "$run_root" "$content_sha256"
printf 'preflight=%s smoke_bank=%s,%s smoke_aggregate=%s smoke_training=%s\n' \
  "$preflight_job" "$smoke_train_bank_job" "$smoke_val_bank_job" \
  "$smoke_aggregate_job" "$smoke_training_job"
printf 'full_bank=%s,%s full_aggregate=%s full_training=%s report=%s\n' \
  "$full_train_bank_job" "$full_val_bank_job" "$full_aggregate_job" \
  "$full_training_job" "$report_job"
