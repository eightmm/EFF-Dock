#!/usr/bin/env bash
# Submit the frozen PLINDER raw-gate -> ODE/guidance -> official-PB chain.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
# shellcheck source=plinder_guidance_validation_profile.sh
source scripts/slurm/plinder_guidance_validation_profile.sh

# SBATCH_* variables override both script directives and explicit expectations.
sanitized_sbatch_vars=()
while IFS= read -r variable_name; do
  if [[ "$variable_name" == SBATCH_* ]]; then
    sanitized_sbatch_vars+=("$variable_name")
    unset "$variable_name"
  fi
done < <(compgen -A variable)
sanitized_sbatch_env=none
if [[ ${#sanitized_sbatch_vars[@]} -gt 0 ]]; then
  sanitized_sbatch_env=$(IFS=,; printf '%s' "${sanitized_sbatch_vars[*]}")
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 RAW_DOWNLOAD_JOB_ID [SAFE_RUN_ID]" >&2
  exit 2
fi
raw_download_job=$1
if [[ ! "$raw_download_job" =~ ^[0-9]+$ ]]; then
  echo "RAW_DOWNLOAD_JOB_ID must be numeric" >&2
  exit 2
fi
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is not available" >&2
  exit 2
fi

run_id=${2:-$(date -u +%Y%m%dT%H%M%SZ)}
if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "run ID must be one safe path component: $run_id" >&2
  exit 2
fi
output_root="$plinder_output_prefix/$run_id"
mkdir -p "$plinder_output_prefix" outputs/benchmarks/logs
if ! mkdir "$output_root"; then
  echo "refusing to reuse an existing or concurrently created output root: $output_root" >&2
  exit 2
fi

reservation_committed=0
submitted_jobs=()
cleanup_reservation() {
  local exit_code=$?
  trap - EXIT
  if [[ "$reservation_committed" -eq 0 ]]; then
    for job_id in "${submitted_jobs[@]}"; do
      scancel "$job_id" 2>/dev/null || true
    done
    printf '%s\n' \
      'status=launcher_failed' \
      "protocol_id=$protocol_id" \
      "run_id=$run_id" \
      'recovery_policy=failed_attempts_preserved_manual_same_run_shard_retry' \
      "cancelled_jobs=${submitted_jobs[*]:-none}" \
      "failed_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      > "$output_root/.submission.failed"
  fi
  exit "$exit_code"
}
trap cleanup_reservation EXIT

raw_root=${EFFDOCK_PLINDER_RAW_ROOT:-${XDG_DATA_HOME:-$HOME/.local/share}/plinder/2024-06/v2}
raw_manifest=${EFFDOCK_PLINDER_RAW_MANIFEST:-outputs/benchmarks/plinder_guidance_validation/raw_download_manifest.json}
raw_gate="$output_root/raw_gate/verified.json"
raw_gate_sidecar="$output_root/raw_gate/verified.sha256"
gpu_partitions=6000ada,heavy
checkpoint=weights/effdock_geometry_ft_100k_best.pt
confidence_checkpoint=weights/effdock_confidence_extmatch_n80_s25_step42500.pt
config=configs/train.yaml
split=data/splits/plinder.json
processed_manifest=data/plinder_processed/manifest.json
split_manifest=data/splits/manifest.json
protocol_doc=docs/PLINDER_GUIDANCE_VALIDATION_PROTOCOL.md
raw_download_job_binding="$output_root/raw_download_job.json"

.venv/bin/python scripts/verify_plinder_download_job.py \
  --job-id "$raw_download_job" \
  --repo-root "$repo_root" \
  --output "$raw_download_job_binding" >/dev/null
read -r raw_download_job_binding_sha256 _ < <(sha256sum "$raw_download_job_binding")
read -r raw_download_scheduler_source raw_download_dependency_required \
  raw_download_job_state raw_download_exit_code < <(
  .venv/bin/python -c \
    'import json,sys; from pathlib import Path; p=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")); required=p.get("scheduler_dependency_required"); type(required) is bool or sys.exit("invalid scheduler_dependency_required type"); print(p.get("scheduler_record_source"),str(required).lower(),p.get("job_state_at_binding"),p.get("exit_code_at_binding"))' \
    "$raw_download_job_binding"
)
case "$raw_download_scheduler_source" in
  scontrol|sacct) ;;
  *)
    echo "unexpected raw-download scheduler source: $raw_download_scheduler_source" >&2
    exit 2
    ;;
esac
case "$raw_download_dependency_required" in
  true)
    if [[ "$raw_download_scheduler_source" != "scontrol" \
      || "$raw_download_job_state" == "COMPLETED" ]]; then
      echo "invalid live raw-download dependency binding" >&2
      exit 2
    fi
    raw_gate_dependency=$raw_download_job
    raw_gate_dependency_record="afterok:$raw_download_job"
    raw_download_dependency_mode=live_job_afterok
    raw_download_cache_reuse=false
    recovery_provenance=none
    ;;
  false)
    if [[ "$raw_download_job_state" != "COMPLETED" \
      || "$raw_download_exit_code" != "0:0" ]]; then
      echo "completed raw-download reuse requires COMPLETED with exit 0:0" >&2
      exit 2
    fi
    raw_gate_dependency=
    raw_gate_dependency_record=none
    raw_download_dependency_mode=completed_cache_reuse_no_afterok
    raw_download_cache_reuse=true
    recovery_provenance="raw_download_job_${raw_download_job}_completed_0:0_verified_by_${raw_download_scheduler_source}"
    ;;
  *)
    echo "invalid scheduler_dependency_required value: $raw_download_dependency_required" >&2
    exit 2
    ;;
esac

verify_known_sha256() {
  local path=$1
  local expected=$2
  local actual
  if [[ ! -f "$path" ]]; then
    echo "missing frozen PLINDER input: $path" >&2
    exit 2
  fi
  read -r actual _ < <(sha256sum "$path")
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch for $path: expected=$expected actual=$actual" >&2
    exit 2
  fi
}
verify_known_sha256 "$checkpoint" \
  6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db
verify_known_sha256 "$confidence_checkpoint" \
  e31fde6f351284205c78f7a1510002779c43312e94d9f82003d47a14d72bc78f
verify_known_sha256 "$config" \
  39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec
verify_known_sha256 "$split" \
  3ac570bf08bced053f1ce040b57efca27c3be616f29a82cd66ef887c08860e6b
verify_known_sha256 "$processed_manifest" \
  e415df12e50df680eeb17665b9409a7879db3a186ffa39d741ca1b8fc8cdea0a
verify_known_sha256 "$split_manifest" \
  23d635ee7a4384d361636a9382cfffe4b4e194cfc4a5f3847905ec3c0335b6e9

frozen_input_files=(
  "$checkpoint"
  "$confidence_checkpoint"
  "$config"
  "$split"
  "$processed_manifest"
  "$split_manifest"
  "$protocol_doc"
  "$raw_download_job_binding"
)
frozen_inputs_manifest="$output_root/frozen_inputs_manifest.sha256"
sha256sum "${frozen_input_files[@]}" > "$frozen_inputs_manifest"
read -r frozen_inputs_manifest_sha256 _ < <(sha256sum "$frozen_inputs_manifest")

mapfile -t package_files < <(rg --files src/effdock | sort)
execution_files=(
  "${package_files[@]}"
  pyproject.toml
  uv.lock
  "${frozen_input_files[@]}"
  scripts/download_plinder_guidance_validation.py
  scripts/verify_plinder_download_job.py
  scripts/run_plinder_guidance_validation.py
  scripts/plinder_guidance_common.py
  scripts/preflight_plinder_guidance_raw.py
  scripts/plinder_guidance_audit.py
  scripts/run_plinder_guidance_posebusters.py
  scripts/aggregate_plinder_guidance_validation.py
  scripts/create_execution_capsule.py
  scripts/slurm/download_plinder_guidance_validation.sbatch
  scripts/slurm/plinder_guidance_validation_profile.sh
  scripts/slurm/plinder_guidance_raw_gate.sbatch
  scripts/slurm/plinder_guidance_sampling.sbatch
  scripts/slurm/plinder_guidance_audit.sbatch
  scripts/slurm/plinder_guidance_posebusters.sbatch
  scripts/slurm/plinder_guidance_report.sbatch
  scripts/slurm/submit_plinder_guidance_validation.sh
)
capsule_copy_files=(
  "${package_files[@]}"
  pyproject.toml
  uv.lock
  "$config"
  "$protocol_doc"
  scripts/download_plinder_guidance_validation.py
  scripts/verify_plinder_download_job.py
  scripts/run_plinder_guidance_validation.py
  scripts/plinder_guidance_common.py
  scripts/preflight_plinder_guidance_raw.py
  scripts/plinder_guidance_audit.py
  scripts/run_plinder_guidance_posebusters.py
  scripts/aggregate_plinder_guidance_validation.py
  scripts/create_execution_capsule.py
  scripts/slurm/download_plinder_guidance_validation.sbatch
  scripts/slurm/plinder_guidance_validation_profile.sh
  scripts/slurm/plinder_guidance_raw_gate.sbatch
  scripts/slurm/plinder_guidance_sampling.sbatch
  scripts/slurm/plinder_guidance_audit.sbatch
  scripts/slurm/plinder_guidance_posebusters.sbatch
  scripts/slurm/plinder_guidance_report.sbatch
  scripts/slurm/submit_plinder_guidance_validation.sh
)
execution_root=".effdock_execution_capsules/$protocol_id/$run_id"
capsule_args=()
for capsule_file in "${capsule_copy_files[@]}"; do
  capsule_args+=(--copy-file "$capsule_file")
done
.venv/bin/python scripts/create_execution_capsule.py \
  --repo-root "$repo_root" \
  --output "$repo_root/$execution_root" \
  --link-root .venv \
  --link-root data \
  --link-root weights \
  --link-root outputs \
  "${capsule_args[@]}" >/dev/null
execution_root_abs=$(readlink -f "$execution_root")
capsule_identity="$execution_root_abs/execution_capsule.json"
read -r capsule_identity_sha256 _ < <(sha256sum "$capsule_identity")
execution_files+=(execution_capsule.json)
execution_manifest="$output_root/execution_manifest.sha256"
(
  cd "$execution_root_abs"
  sha256sum "${execution_files[@]}"
) > "$repo_root/$execution_manifest"
read -r execution_manifest_sha256 _ < <(sha256sum "$execution_manifest")
git_commit=$(git rev-parse HEAD)
read -r git_diff_sha256 _ < <(git diff --no-ext-diff | sha256sum)

printf '%s\n' \
  'status=submitting' \
  "protocol_id=$protocol_id" \
  'claim_scope=guidance_development_not_untouched_confirmation' \
  'automatic_eta_selection=false' \
  "run_id=$run_id" \
  "output_root=$output_root" \
  "raw_download_job=$raw_download_job" \
  "raw_download_job_binding=$raw_download_job_binding" \
  "raw_download_job_binding_sha256=$raw_download_job_binding_sha256" \
  "raw_download_scheduler_source=$raw_download_scheduler_source" \
  "raw_download_scheduler_dependency_required=$raw_download_dependency_required" \
  "raw_download_job_state_at_binding=$raw_download_job_state" \
  "raw_download_exit_code_at_binding=$raw_download_exit_code" \
  "raw_download_dependency_mode=$raw_download_dependency_mode" \
  "raw_gate_dependency=$raw_gate_dependency_record" \
  "raw_download_cache_reuse=$raw_download_cache_reuse" \
  "recovery_provenance=$recovery_provenance" \
  "raw_root=$raw_root" \
  "raw_manifest=$raw_manifest" \
  "raw_gate=$raw_gate" \
  "raw_gate_sidecar=$raw_gate_sidecar" \
  "deferred_frozen_raw_inputs_manifest=$raw_gate_sidecar" \
  'deferred_frozen_raw_inputs_contents=raw_manifest_sha256_plus_475_archive_integrity_plus_1076_asset_ledger' \
  "git_commit=$git_commit" \
  "git_diff_sha256=$git_diff_sha256" \
  "execution_root=$execution_root_abs" \
  "execution_capsule_identity=$capsule_identity" \
  "execution_capsule_identity_sha256=$capsule_identity_sha256" \
  'execution_capsule_code_mutable=false' \
  'execution_capsule_linked_roots=.venv,data,weights,outputs' \
  'sampling_budget=N100/S10' \
  'eta_values=0,0.5,1,1.5,2' \
  'primary_selector=confidence' \
  "gpu_partitions=$gpu_partitions" \
  'gpu_request=gpu:1' \
  'gpu_minimum_visible_memory_mib=48000' \
  'recovery_policy=failed_attempts_preserved_manual_same_run_shard_retry' \
  "frozen_inputs_manifest=$frozen_inputs_manifest" \
  "frozen_inputs_manifest_sha256=$frozen_inputs_manifest_sha256" \
  "execution_manifest=$execution_manifest" \
  "execution_manifest_sha256=$execution_manifest_sha256" \
  "sanitized_sbatch_env=$sanitized_sbatch_env" \
  > "$output_root/.submission.pending"

base_export="ALL,EFFDOCK_REPO_DIR=$execution_root_abs,EFFDOCK_LIVE_REPO_DIR=$repo_root"
base_export+=",EFFDOCK_OUTPUT_ROOT=$output_root,PYTHONPATH=$execution_root_abs/src"
base_export+=",PYTHONDONTWRITEBYTECODE=1,EFFDOCK_GIT_COMMIT=$git_commit"
base_export+=",EFFDOCK_GIT_DIFF_SHA256=$git_diff_sha256"
base_export+=",EFFDOCK_PLINDER_RAW_ROOT=$raw_root"
base_export+=",EFFDOCK_PLINDER_RAW_MANIFEST=$raw_manifest"
base_export+=",EFFDOCK_PLINDER_RAW_GATE=$raw_gate"
base_export+=",EFFDOCK_PLINDER_RAW_GATE_SIDECAR=$raw_gate_sidecar"
base_export+=",EFFDOCK_PLINDER_EXECUTION_MANIFEST=$execution_manifest"
base_export+=",EFFDOCK_PLINDER_EXECUTION_MANIFEST_SHA256=$execution_manifest_sha256"
base_export+=",EFFDOCK_PLINDER_FROZEN_INPUTS_MANIFEST=$frozen_inputs_manifest"
base_export+=",EFFDOCK_PLINDER_FROZEN_INPUTS_MANIFEST_SHA256=$frozen_inputs_manifest_sha256"

submit_job() {
  local dependency=$1
  local script=$2
  local export_spec=$3
  local array_spec=$4
  local partition=$5
  local qos=$6
  local cpus=$7
  local memory=$8
  local walltime=$9
  local gres=${10}
  local raw
  local args=(
    --parsable
    --partition="$partition"
    --qos="$qos"
    --nodes=1
    --ntasks=1
    --cpus-per-task="$cpus"
    --mem="$memory"
    --time="$walltime"
    --export="$export_spec"
    --kill-on-invalid-dep=yes
  )
  if [[ -n "$dependency" ]]; then
    args+=(--dependency="afterok:$dependency")
  fi
  if [[ -n "$array_spec" ]]; then
    args+=(--array="$array_spec")
  fi
  if [[ -n "$gres" ]]; then
    args+=(--gres="$gres")
  fi
  if ! raw=$(sbatch "${args[@]}" "$script"); then
    echo "failed to submit $script" >&2
    return 1
  fi
  local job_id=${raw%%;*}
  if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
    echo "invalid sbatch --parsable response for $script: $raw" >&2
    return 1
  fi
  printf '%s' "$job_id"
}

raw_gate_job=$(submit_job "$raw_gate_dependency" \
  "$execution_root_abs/scripts/slurm/plinder_guidance_raw_gate.sbatch" \
  "$base_export,EFFDOCK_PLINDER_MODE=raw_gate" "" cpu_only long 8 32G 12:00:00 "")
submitted_jobs+=("$raw_gate_job")
smoke_job=$(submit_job "$raw_gate_job" \
  "$execution_root_abs/scripts/slurm/plinder_guidance_sampling.sbatch" \
  "$base_export,EFFDOCK_PLINDER_MODE=smoke" "0-4%4" "$gpu_partitions" long 4 64G 12:00:00 gpu:1)
submitted_jobs+=("$smoke_job")
smoke_audit_job=$(submit_job "$smoke_job" \
  "$execution_root_abs/scripts/slurm/plinder_guidance_audit.sbatch" \
  "$base_export,EFFDOCK_PLINDER_MODE=smoke" "" cpu_only long 4 24G 02:00:00 "")
submitted_jobs+=("$smoke_audit_job")
posebusters_smoke_job=$(submit_job "$smoke_audit_job" \
  "$execution_root_abs/scripts/slurm/plinder_guidance_posebusters.sbatch" \
  "$base_export,EFFDOCK_PLINDER_MODE=smoke" "0-4%5" cpu_only long 2 12G 01:00:00 "")
submitted_jobs+=("$posebusters_smoke_job")
sampling_job=$(submit_job "$posebusters_smoke_job" \
  "$execution_root_abs/scripts/slurm/plinder_guidance_sampling.sbatch" \
  "$base_export,EFFDOCK_PLINDER_MODE=full" "0-159%4" "$gpu_partitions" long 4 64G 12:00:00 gpu:1)
submitted_jobs+=("$sampling_job")
audit_job=$(submit_job "$sampling_job" \
  "$execution_root_abs/scripts/slurm/plinder_guidance_audit.sbatch" \
  "$base_export,EFFDOCK_PLINDER_MODE=full" "" cpu_only long 4 24G 02:00:00 "")
submitted_jobs+=("$audit_job")
posebusters_job=$(submit_job "$audit_job" \
  "$execution_root_abs/scripts/slurm/plinder_guidance_posebusters.sbatch" \
  "$base_export,EFFDOCK_PLINDER_MODE=full" "0-159%16" cpu_only long 2 12G 03:00:00 "")
submitted_jobs+=("$posebusters_job")
report_job=$(submit_job "$posebusters_job" \
  "$execution_root_abs/scripts/slurm/plinder_guidance_report.sbatch" \
  "$base_export,EFFDOCK_PLINDER_MODE=full" "" cpu_only long 4 32G 02:00:00 "")
submitted_jobs+=("$report_job")

printf '%s\n' \
  "raw_gate_job=$raw_gate_job" \
  "smoke_job=$smoke_job" \
  "smoke_audit_job=$smoke_audit_job" \
  "posebusters_smoke_job=$posebusters_smoke_job" \
  "sampling_job=$sampling_job" \
  "audit_job=$audit_job" \
  "posebusters_job=$posebusters_job" \
  "report_job=$report_job" \
  "submitted_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >> "$output_root/.submission.pending"
sed -i 's/^status=submitting$/status=submitted/' "$output_root/.submission.pending"
mv "$output_root/.submission.pending" "$output_root/.submission"

reservation_committed=1
trap - EXIT

printf 'output_root=%s\n' "$output_root"
printf 'raw_gate_job=%s dependency=%s raw_download_job=%s cache_reuse=%s checks=475_md5_crc32c_zipcrc_plus_1076_assets\n' \
  "$raw_gate_job" "$raw_gate_dependency_record" "$raw_download_job" \
  "$raw_download_cache_reuse"
printf 'smoke_job=%s dependency=afterok:%s tasks=5 mapping=largest_id_x_5_eta\n' \
  "$smoke_job" "$raw_gate_job"
printf 'smoke_audit_job=%s dependency=afterok:%s\n' "$smoke_audit_job" "$smoke_job"
printf 'posebusters_smoke_job=%s dependency=afterok:%s tasks=5 selector=confidence\n' \
  "$posebusters_smoke_job" "$smoke_audit_job"
printf 'sampling_job=%s dependency=afterok:%s tasks=160 concurrency=4 mapping=5_eta_x_32_shards\n' \
  "$sampling_job" "$posebusters_smoke_job"
printf 'audit_job=%s dependency=afterok:%s denominator=1076\n' "$audit_job" "$sampling_job"
printf 'posebusters_job=%s dependency=afterok:%s tasks=160 concurrency=16 selector=confidence\n' \
  "$posebusters_job" "$audit_job"
printf 'report_job=%s dependency=afterok:%s auto_eta_selection=false\n' \
  "$report_job" "$posebusters_job"
monitor_csv=$(IFS=,; printf '%s' "${submitted_jobs[*]}")
printf 'monitor: squeue -j %s\n' "$monitor_csv"
