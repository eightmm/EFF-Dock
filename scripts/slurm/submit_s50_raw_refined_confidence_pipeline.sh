#!/usr/bin/env bash
set -euo pipefail

die() { echo "$*" >&2; exit 2; }
[[ $# -eq 2 ]] || die "usage: $0 REFINED_RUN_ROOT REFINED_AGGREGATE_JOB_ID"
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$repo_dir"
refined_run_root=$(readlink -f "$1")
producer_job=$2
[[ "$producer_job" =~ ^[0-9]+$ ]] || die "aggregate job ID must be numeric"
[[ -d "$refined_run_root/full" ]] || die "missing refined full root"

source_root="$repo_dir/outputs/eff-dock/s50-confidence-matched-runs/14cb2840f040b05127610a1f3dccd8aa5b2573c41156ff446898e2d212f1164b"
raw_bank="$source_root/full/bank_manifest.json"
input_manifest="$source_root/frozen/input_manifest.json"
filtered_split="$source_root/full/filtered_split.json"
raw_target="$repo_dir/outputs/eff-dock/s50-confidence-symmetry-labels/41c5ff349ba95e64b45ac43356b970dc037ebf31898c6533e61a2360880f031f/full/manifest.json"
init_checkpoint="$repo_dir/outputs/eff-dock/s50-confidence-symmetry-training/88509b2189ab619c91feee843e4d90388b85b36b181532ca0a54ebf271f69b97/full/latest.pt"
sampler_checkpoint="$repo_dir/outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step50000_ema_common_init.pt"
sampler_config="$repo_dir/configs/train.yaml"
train_config="$repo_dir/configs/train_confidence_s50_raw_refined_10k.yaml"
protocol="$repo_dir/docs/S50_RAW_REFINED_CONFIDENCE_FINETUNE_PROTOCOL.md"
for path in "$raw_bank" "$input_manifest" "$filtered_split" "$raw_target" \
  "$init_checkpoint" "$sampler_checkpoint" "$sampler_config" "$train_config" "$protocol"; do
  [[ -f "$path" && ! -L "$path" ]] || die "missing/nonregular frozen input: $path"
done

mapfile -t package_files < <(
  find src/effdock -type f ! -path '*/__pycache__/*' | sort
)
copy_files=(
  "${package_files[@]}"
  scripts/materialize_s50_refined_confidence_bank.py
  scripts/build_s50_symmetry_rmsd_sidecars.py
  scripts/prepare_s50_confidence_training_bank.py
  scripts/create_execution_capsule.py
  scripts/slurm/s50_raw_refined_bank.sbatch
  scripts/slurm/s50_raw_refined_bank_aggregate.sbatch
  scripts/slurm/s50_raw_refined_labels.sbatch
  scripts/slurm/s50_raw_refined_confidence_train.sbatch
  scripts/slurm/submit_s50_raw_refined_confidence_pipeline.sh
  configs/train_confidence_s50_raw_refined_10k.yaml
  configs/train.yaml
  docs/S50_RAW_REFINED_CONFIDENCE_FINETUNE_PROTOCOL.md
  pyproject.toml
  uv.lock
)
identity_input="EFFDOCK-S50-RAW-REFINED-CONFIDENCE-PIPELINE-V2"
for path in "${copy_files[@]}"; do
  [[ -f "$path" ]] || die "missing runtime file: $path"
  identity_input+=$'\n'"$path=$(sha256sum "$path" | cut -d' ' -f1)"
done
for path in "$raw_bank" "$input_manifest" "$filtered_split" "$raw_target" \
  "$init_checkpoint" "$sampler_checkpoint"; do
  identity_input+=$'\n'"$(basename "$path")=$(sha256sum "$path" | cut -d' ' -f1)"
done
identity_input+=$'\n'"refined_content_id=$(basename "$refined_run_root")"
content_id=$(printf '%s\n' "$identity_input" | sha256sum | cut -d' ' -f1)
run_root="$repo_dir/outputs/eff-dock/s50-raw-refined-confidence-runs/$content_id"
capsule="$repo_dir/.effdock_execution_capsules/s50-raw-refined-confidence-$content_id"
[[ ! -e "$run_root" ]] || die "refusing to reuse run root: $run_root"
[[ ! -e "$capsule" ]] || die "refusing to reuse execution capsule: $capsule"
capsule_args=()
for path in "${copy_files[@]}"; do capsule_args+=(--copy-file "$path"); done
.venv/bin/python scripts/create_execution_capsule.py \
  --repo-root "$repo_dir" --output "$capsule" \
  --link-root .venv --link-root data --link-root outputs --link-root weights \
  "${capsule_args[@]}" >/dev/null
for path in "${copy_files[@]}"; do
  [[ $(sha256sum "$path" | cut -d' ' -f1) == \
      $(sha256sum "$capsule/$path" | cut -d' ' -f1) ]] \
    || die "execution capsule copy drift: $path"
done
capsule_manifest_sha=$(sha256sum "$capsule/execution_capsule.json" | cut -d' ' -f1)
mkdir -p "$run_root"
printf '%s\n' "$identity_input" > "$run_root/CONTENT_IDENTITY.txt"
printf '%s\n' "$content_id" > "$run_root/CONTENT_IDENTITY.sha256"
printf '%s  %s\n' "$capsule_manifest_sha" "$capsule/execution_capsule.json" \
  > "$run_root/EXECUTION_CAPSULE.sha256"

worker="$capsule/scripts/materialize_s50_refined_confidence_bank.py"
label_worker="$capsule/scripts/build_s50_symmetry_rmsd_sidecars.py"
builder="$capsule/scripts/prepare_s50_confidence_training_bank.py"
train_config="$capsule/configs/train_confidence_s50_raw_refined_10k.yaml"
protocol="$capsule/docs/S50_RAW_REFINED_CONFIDENCE_FINETUNE_PROTOCOL.md"
refined_manifest="$refined_run_root/full/manifest.json"
bank_output="$run_root/refined_confidence_bank"
label_output="$run_root/refined_symmetry_labels"
train_output="$run_root/training"

sha() { sha256sum "$1" | cut -d' ' -f1; }
base_export="EFFDOCK_MIX_BANK_WORKER=$worker"
base_export+=",EFFDOCK_MIX_BANK_WORKER_SHA=$(sha "$worker")"
base_export+=",EFFDOCK_MIX_LABEL_WORKER=$label_worker"
base_export+=",EFFDOCK_MIX_LABEL_WORKER_SHA=$(sha "$label_worker")"
base_export+=",EFFDOCK_MIX_BUILDER=$builder,EFFDOCK_MIX_BUILDER_SHA=$(sha "$builder")"
base_export+=",EFFDOCK_MIX_RAW_BANK=$raw_bank,EFFDOCK_MIX_RAW_BANK_SHA=$(sha "$raw_bank")"
base_export+=",EFFDOCK_MIX_RAW_TARGET=$raw_target,EFFDOCK_MIX_RAW_TARGET_SHA=$(sha "$raw_target")"
base_export+=",EFFDOCK_MIX_INPUT_MANIFEST=$input_manifest,EFFDOCK_MIX_INPUT_MANIFEST_SHA=$(sha "$input_manifest")"
base_export+=",EFFDOCK_MIX_FILTERED_SPLIT=$filtered_split,EFFDOCK_MIX_FILTERED_SPLIT_SHA=$(sha "$filtered_split")"
base_export+=",EFFDOCK_MIX_REFINED_MANIFEST=$refined_manifest"
base_export+=",EFFDOCK_MIX_BANK_OUTPUT_ROOT=$bank_output,EFFDOCK_MIX_LABEL_OUTPUT_ROOT=$label_output"
base_export+=",EFFDOCK_MIX_TRAIN_OUTPUT_ROOT=$train_output"
base_export+=",EFFDOCK_MIX_SAMPLER_CHECKPOINT=$sampler_checkpoint,EFFDOCK_MIX_SAMPLER_CHECKPOINT_SHA=$(sha "$sampler_checkpoint")"
base_export+=",EFFDOCK_MIX_SAMPLER_CONFIG=$capsule/configs/train.yaml,EFFDOCK_MIX_SAMPLER_CONFIG_SHA=$(sha "$capsule/configs/train.yaml")"
base_export+=",EFFDOCK_MIX_INIT_CHECKPOINT=$init_checkpoint,EFFDOCK_MIX_INIT_CHECKPOINT_SHA=$(sha "$init_checkpoint")"
base_export+=",EFFDOCK_MIX_TRAIN_CONFIG=$train_config,EFFDOCK_MIX_TRAIN_CONFIG_SHA=$(sha "$train_config")"
base_export+=",EFFDOCK_MIX_TRAINER_SHA=$(sha "$capsule/src/effdock/workflows/train_confidence.py")"
base_export+=",EFFDOCK_MIX_DATASET_SHA=$(sha "$capsule/src/effdock/confidence/dataset.py")"
base_export+=",EFFDOCK_MIX_PROTOCOL_SHA=$(sha "$protocol")"

submitted=()
cleanup() {
  status=$?
  if (( status != 0 )); then
    for job in "${submitted[@]}"; do scancel "$job" >/dev/null 2>&1 || true; done
  fi
  exit "$status"
}
trap cleanup EXIT
submit() {
  last_job=$(sbatch --parsable --chdir="$capsule" "$@")
  last_job=${last_job%%;*}
  [[ "$last_job" =~ ^[0-9]+$ ]] || die "unexpected sbatch job ID: $last_job"
  submitted+=("$last_job")
}

submit --hold --dependency="afterok:$producer_job" \
  --export="ALL,$base_export,EFFDOCK_MIX_BANK_STAGE=smoke" \
  scripts/slurm/s50_raw_refined_bank.sbatch
bank_smoke=$last_job
submit --dependency="afterok:$bank_smoke" --array=0-135%8 \
  --export="ALL,$base_export,EFFDOCK_MIX_BANK_STAGE=full" \
  scripts/slurm/s50_raw_refined_bank.sbatch
bank_full=$last_job
submit --dependency="afterok:$bank_full" \
  --export="ALL,$base_export" scripts/slurm/s50_raw_refined_bank_aggregate.sbatch
bank_agg=$last_job
submit --dependency="afterok:$bank_agg" \
  --export="ALL,$base_export,EFFDOCK_MIX_LABEL_STAGE=smoke" \
  scripts/slurm/s50_raw_refined_labels.sbatch
label_smoke=$last_job
submit --dependency="afterok:$label_smoke" --array=0-127%16 \
  --export="ALL,$base_export,EFFDOCK_MIX_LABEL_STAGE=full" \
  scripts/slurm/s50_raw_refined_labels.sbatch
label_full=$last_job
submit --dependency="afterok:$label_full" \
  --export="ALL,$base_export,EFFDOCK_MIX_LABEL_STAGE=aggregate" \
  scripts/slurm/s50_raw_refined_labels.sbatch
label_agg=$last_job
submit --dependency="afterok:$label_agg" \
  --export="ALL,$base_export,EFFDOCK_MIX_TRAIN_STAGE=smoke" \
  scripts/slurm/s50_raw_refined_confidence_train.sbatch
train_smoke=$last_job
submit --dependency="afterok:$train_smoke" \
  --export="ALL,$base_export,EFFDOCK_MIX_TRAIN_STAGE=full" \
  scripts/slurm/s50_raw_refined_confidence_train.sbatch
train_full=$last_job

.venv/bin/python - "$run_root/SUBMISSION.json" <<PY
import json, sys
from pathlib import Path
payload = {
  "status": "submitted_held_then_released",
  "content_id": "$content_id",
  "execution_capsule": "$capsule",
  "execution_capsule_manifest_sha256": "$capsule_manifest_sha",
  "producer_refined_aggregate_job": "$producer_job",
  "jobs": {
    "bank_smoke": "$bank_smoke", "bank_full": "$bank_full", "bank_aggregate": "$bank_agg",
    "label_smoke": "$label_smoke", "label_full": "$label_full", "label_aggregate": "$label_agg",
    "train_smoke": "$train_smoke", "train_full": "$train_full"
  }
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
scontrol release "$bank_smoke"
trap - EXIT
printf 'run_root=%s\n' "$run_root"
printf 'bank_smoke=%s bank_full=%s bank_agg=%s\n' "$bank_smoke" "$bank_full" "$bank_agg"
printf 'label_smoke=%s label_full=%s label_agg=%s\n' "$label_smoke" "$label_full" "$label_agg"
printf 'train_smoke=%s train_full=%s\n' "$train_smoke" "$train_full"
