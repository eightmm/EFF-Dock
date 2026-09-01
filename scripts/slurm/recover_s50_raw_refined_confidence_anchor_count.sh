#!/usr/bin/env bash
set -euo pipefail

die() { echo "$*" >&2; exit 2; }
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$repo_dir"

producer_job=58532
content_id=309bb1a7645f8d09a796e45559d018c969d077a937ecb225ea53314a7143c804
run_root="$repo_dir/outputs/eff-dock/s50-raw-refined-confidence-runs/$content_id"
capsule="$repo_dir/.effdock_execution_capsules/s50-raw-refined-confidence-$content_id-anchor-count-b42b2904"
refined_run_root="$repo_dir/outputs/eff-dock/s50-refined-pose-runs/c14df954c757ee092f8299224412c52f4305730348c797416fff4e73a3076050"
refined_manifest="$refined_run_root/full/manifest.json"
recovery_record="$run_root/RECOVERY_SUBMISSION_AFTER_58539.json"

[[ -d "$run_root" && -d "$capsule" ]] || die "missing frozen run root/capsule"
[[ $(cat "$run_root/CONTENT_IDENTITY.sha256") == "$content_id" ]] \
  || die "content identity mismatch"
[[ $(sha256sum "$capsule/execution_capsule.json" | cut -d' ' -f1) == \
    e89780fe7b96b4193e5717de9fb1e70c348cfd4aa8e347da1b318d747ed1f821 ]] \
  || die "execution capsule mismatch"
[[ $(sha256sum "$capsule/RECOVERY_AMENDMENT.json" | cut -d' ' -f1) == \
    de99a5e56568027e63b56cfbd69d7d72e5f3484e4fd27ea21421021b982f815b ]] \
  || die "recovery amendment mismatch"
[[ $(sha256sum "$capsule/scripts/prepare_s50_confidence_training_bank.py" | cut -d' ' -f1) == \
    b42b29044705ae4e34a3c85982a55b9d0ac45c598e903be6af59d7f19a8de3c0 ]] \
  || die "crystal-anchor count fix mismatch"
[[ -f "$refined_manifest.sha256" ]] || die "missing refined manifest sidecar"
read -r refined_sha refined_path < "$refined_manifest.sha256"
[[ "$refined_path" == "$refined_manifest" ]] || die "refined manifest sidecar path mismatch"
[[ "$refined_sha" == 684beabb7f86ecdfe3a5576f60d6cde7435c050a5bd5b166b1e5bb5e8f29579b ]] \
  || die "unexpected refined manifest identity"
[[ $(sha256sum "$refined_manifest" | cut -d' ' -f1) == "$refined_sha" ]] \
  || die "refined manifest changed"
[[ ! -e "$recovery_record" ]] || die "refusing to overwrite recovery record"
for path in refined_confidence_bank refined_symmetry_labels training; do
  [[ ! -e "$run_root/$path" ]] || die "recovery output already exists: $run_root/$path"
done

source_root="$repo_dir/outputs/eff-dock/s50-confidence-matched-runs/14cb2840f040b05127610a1f3dccd8aa5b2573c41156ff446898e2d212f1164b"
raw_bank="$source_root/full/bank_manifest.json"
input_manifest="$source_root/frozen/input_manifest.json"
filtered_split="$source_root/full/filtered_split.json"
raw_target="$repo_dir/outputs/eff-dock/s50-confidence-symmetry-labels/41c5ff349ba95e64b45ac43356b970dc037ebf31898c6533e61a2360880f031f/full/manifest.json"
init_checkpoint="$repo_dir/outputs/eff-dock/s50-confidence-symmetry-training/88509b2189ab619c91feee843e4d90388b85b36b181532ca0a54ebf271f69b97/full/latest.pt"
sampler_checkpoint="$repo_dir/outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step50000_ema_common_init.pt"
worker="$capsule/scripts/materialize_s50_refined_confidence_bank.py"
label_worker="$capsule/scripts/build_s50_symmetry_rmsd_sidecars.py"
builder="$capsule/scripts/prepare_s50_confidence_training_bank.py"
train_config="$capsule/configs/train_confidence_s50_raw_refined_10k.yaml"
protocol="$capsule/docs/S50_RAW_REFINED_CONFIDENCE_FINETUNE_PROTOCOL.md"
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
  last_job=$(cd "$capsule" && sbatch --parsable "$@")
  last_job=${last_job%%;*}
  [[ "$last_job" =~ ^[0-9]+$ ]] || die "unexpected sbatch job ID: $last_job"
  submitted+=("$last_job")
}

# The completed producer has already aged out of slurmctld even though sacct
# retains its COMPLETED/0:0 record.  The exact sealed manifest and sidecar above
# are the recovery prerequisite, so do not attach an invalid historical job ID.
submit --hold \
  --export="ALL,$base_export,EFFDOCK_MIX_BANK_STAGE=smoke" \
  scripts/slurm/s50_raw_refined_bank.sbatch
bank_smoke=$last_job
submit --dependency="afterok:$bank_smoke" --array=0-135%8 \
  --export="ALL,$base_export,EFFDOCK_MIX_BANK_STAGE=full" \
  scripts/slurm/s50_raw_refined_bank.sbatch
bank_full=$last_job
submit --dependency="afterok:$bank_full" --export="ALL,$base_export" \
  scripts/slurm/s50_raw_refined_bank_aggregate.sbatch
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

.venv/bin/python - "$recovery_record" <<PY
import json, sys
from pathlib import Path
payload = {
    "status": "recovery_submitted_held_then_released",
    "superseded_jobs": ["58539", "58540", "58541", "58542", "58543", "58544", "58545", "58546"],
    "producer_refined_aggregate_job": "$producer_job",
    "refined_manifest_sha256": "$refined_sha",
    "jobs": {
        "bank_smoke": "$bank_smoke", "bank_full": "$bank_full", "bank_aggregate": "$bank_agg",
        "label_smoke": "$label_smoke", "label_full": "$label_full", "label_aggregate": "$label_agg",
        "train_smoke": "$train_smoke", "train_full": "$train_full",
    },
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
scontrol release "$bank_smoke"
trap - EXIT
printf 'bank_smoke=%s bank_full=%s bank_agg=%s\n' "$bank_smoke" "$bank_full" "$bank_agg"
printf 'label_smoke=%s label_full=%s label_agg=%s\n' "$label_smoke" "$label_full" "$label_agg"
printf 'train_smoke=%s train_full=%s\n' "$train_smoke" "$train_full"
