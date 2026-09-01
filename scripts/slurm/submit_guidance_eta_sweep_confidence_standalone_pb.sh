#!/usr/bin/env bash
# Submit one fresh, parent-free confidence eta-sweep characterization.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
# shellcheck source=guidance_eta_sweep_confidence_standalone_profile.sh
source scripts/slurm/guidance_eta_sweep_confidence_standalone_profile.sh
launcher_path=${EFFDOCK_STANDALONE_LAUNCHER:-scripts/slurm/submit_guidance_eta_sweep_confidence_standalone_pb.sh}

# SBATCH_* environment variables override directives embedded in an sbatch file.
# Remove inherited submission overrides so the frozen per-stage resources below
# cannot silently change when this launcher is called from another shell/job.
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

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [SAFE_RUN_ID]" >&2
  exit 2
fi
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is not available" >&2
  exit 2
fi

run_id=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "run ID must be one safe path component: $run_id" >&2
  exit 2
fi
output_root="$output_prefix/$run_id"
mkdir -p "$output_prefix" outputs/benchmarks/logs
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
      "sweep_profile=$standalone_profile" \
      "run_id=$run_id" \
      "cancelled_jobs=${submitted_jobs[*]:-none}" \
      "failed_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      > "$output_root/.submission.failed"
  fi
  exit "$exit_code"
}
trap cleanup_reservation EXIT

checkpoint=weights/effdock_geometry_ft_100k_best.pt
confidence_checkpoint=weights/effdock_confidence_extmatch_n80_s25_step42500.pt
config=configs/train.yaml
benchmark_input_manifest=docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json
astex_centers=data/external_test/astex_reference_pocket_centers.json
posebusters_centers=data/external_test/posebusters_reference_pocket_centers.json
cohort_audit=outputs/benchmarks/guidance_eta_sweep_v2_runs/20260801T102903Z/audit/combined.json
gpu_partitions=6000ada,heavy

verify_known_sha256() {
  local path=$1
  local expected=$2
  local actual
  if [[ ! -f "$path" ]]; then
    echo "missing frozen input: $path" >&2
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
verify_known_sha256 "$benchmark_input_manifest" \
  99f15f557644cc51c3dd1f559b0dd97dd4259c1de3e1403fb761b7c7e079f668
verify_known_sha256 "$astex_centers" \
  1ac4d8629a7ee2adb785173db56fb69ec4140d68e3057631ae10df6ef88d0d85
verify_known_sha256 "$posebusters_centers" \
  2d3db55c8cc75650cff85d8e3c12445fb8f45fbe2673d8bbc32045ee8c0f6ad0
verify_known_sha256 "$cohort_audit" \
  dac7903488ccd36552a9bca134e37e633e3f07166d94f0389837012081ff3048

profile_identity=""
if [[ "$standalone_profile" == steric_high_eta_v1 ]]; then
  profile_identity="$output_root/profile_identity.json"
  .venv/bin/python -m effdock.workflows.guidance_steric_high_eta_preflight \
    --output "$profile_identity" >/dev/null
fi

frozen_input_files=(
  "$checkpoint"
  "$confidence_checkpoint"
  "$config"
  "$benchmark_input_manifest"
  "$astex_centers"
  "$posebusters_centers"
  "$cohort_audit"
)
frozen_inputs_manifest="$output_root/frozen_inputs_manifest.sha256"
sha256sum "${frozen_input_files[@]}" > "$frozen_inputs_manifest"
read -r frozen_inputs_manifest_sha256 _ < <(sha256sum "$frozen_inputs_manifest")

mapfile -t package_files < <(rg --files src/effdock | sort)
execution_files=(
  "${package_files[@]}"
  pyproject.toml
  uv.lock
  "$config"
  "$benchmark_input_manifest"
  "$astex_centers"
  "$posebusters_centers"
  "$cohort_audit"
  "$protocol_doc"
  scripts/slurm/guidance_eta_sweep_confidence_standalone_profile.sh
  scripts/slurm/guidance_eta_sweep_confidence_standalone_array.sbatch
  scripts/slurm/guidance_eta_sweep_confidence_standalone_audit.sbatch
  scripts/slurm/guidance_eta_sweep_confidence_standalone_posebusters_smoke.sbatch
  scripts/slurm/guidance_steric_high_eta_stress_audit.sbatch
  scripts/slurm/guidance_eta_sweep_confidence_standalone_posebusters_array.sbatch
  scripts/slurm/guidance_eta_sweep_confidence_standalone_report.sbatch
  scripts/slurm/submit_guidance_eta_sweep_confidence_standalone_pb.sh
  scripts/slurm/submit_guidance_steric_high_eta_confidence_pb.sh
  scripts/create_execution_capsule.py
)
if [[ -n "$profile_identity" ]]; then
  execution_files+=("$profile_identity")
fi
capsule_copy_files=(
  "${package_files[@]}"
  pyproject.toml
  uv.lock
  "$config"
  "$benchmark_input_manifest"
  "$protocol_doc"
  scripts/slurm/guidance_eta_sweep_confidence_standalone_profile.sh
  scripts/slurm/guidance_eta_sweep_confidence_standalone_array.sbatch
  scripts/slurm/guidance_eta_sweep_confidence_standalone_audit.sbatch
  scripts/slurm/guidance_eta_sweep_confidence_standalone_posebusters_smoke.sbatch
  scripts/slurm/guidance_steric_high_eta_stress_audit.sbatch
  scripts/slurm/guidance_eta_sweep_confidence_standalone_posebusters_array.sbatch
  scripts/slurm/guidance_eta_sweep_confidence_standalone_report.sbatch
  scripts/slurm/submit_guidance_eta_sweep_confidence_standalone_pb.sh
  scripts/slurm/submit_guidance_steric_high_eta_confidence_pb.sh
  scripts/create_execution_capsule.py
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
  "sweep_profile=$standalone_profile" \
  'mode=fresh_one_pass_characterization' \
  'parent_compared=false' \
  'deterministic_replay_claim=false' \
  "run_id=$run_id" \
  "output_root=$output_root" \
  "launcher=$launcher_path" \
  "git_commit=$git_commit" \
  "git_diff_sha256=$git_diff_sha256" \
  "execution_root=$execution_root_abs" \
  "execution_capsule_identity=$capsule_identity" \
  "execution_capsule_identity_sha256=$capsule_identity_sha256" \
  'execution_capsule_code_mutable=false' \
  'execution_capsule_linked_roots=.venv,data,weights,outputs' \
  "config=$config" \
  "docking_checkpoint=$checkpoint" \
  "confidence_checkpoint=$confidence_checkpoint" \
  'sampling_seed=42' \
  'sampling_budget=N100/S10' \
  "gpu_partitions=$gpu_partitions" \
  'gpu_request=gpu:1' \
  'gpu_minimum_visible_memory_mib=48000' \
  'slurm_log_root=outputs/benchmarks/logs' \
  "cohort_audit=$cohort_audit" \
  "frozen_inputs_manifest=$frozen_inputs_manifest" \
  "frozen_inputs_manifest_sha256=$frozen_inputs_manifest_sha256" \
  "execution_manifest=$execution_manifest" \
  "execution_manifest_sha256=$execution_manifest_sha256" \
  "profile_identity=${profile_identity:-none}" \
  "sanitized_sbatch_env=$sanitized_sbatch_env" \
  > "$output_root/.submission.pending"

base_export="ALL,EFFDOCK_REPO_DIR=$execution_root_abs,EFFDOCK_LIVE_REPO_DIR=$repo_root"
base_export+=",EFFDOCK_OUTPUT_ROOT=$output_root,PYTHONPATH=$execution_root_abs/src"
base_export+=",PYTHONDONTWRITEBYTECODE=1,EFFDOCK_GIT_COMMIT=$git_commit"
base_export+=",EFFDOCK_GIT_DIFF_SHA256=$git_diff_sha256"
base_export+=",EFFDOCK_STANDALONE_PROFILE=$standalone_profile"
base_export+=",EFFDOCK_COHORT_MANIFEST=$cohort_audit"
base_export+=",EFFDOCK_STANDALONE_EXECUTION_MANIFEST=$execution_manifest"
base_export+=",EFFDOCK_STANDALONE_EXECUTION_MANIFEST_SHA256=$execution_manifest_sha256"
base_export+=",EFFDOCK_STANDALONE_FROZEN_INPUTS_MANIFEST=$frozen_inputs_manifest"
base_export+=",EFFDOCK_STANDALONE_FROZEN_INPUTS_MANIFEST_SHA256=$frozen_inputs_manifest_sha256"

submit_job() {
  local dependency=$1
  local script=$2
  local export_spec=$3
  local array_spec=${4:-}
  local partition=${5:?submission partition is required}
  local raw
  local args=(--parsable --partition="$partition" --export="$export_spec")
  if [[ -n "$dependency" ]]; then
    args+=(--dependency="afterok:$dependency")
  fi
  if [[ -n "$array_spec" ]]; then
    args+=(--array="$array_spec")
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

smoke_export="$base_export,EFFDOCK_STANDALONE_SMOKE_GRID=1,EFFDOCK_STANDALONE_STRESS_ID="
full_export="$base_export,EFFDOCK_STANDALONE_SMOKE_GRID=0,EFFDOCK_STANDALONE_STRESS_ID="

smoke_job=$(submit_job "" \
  "$execution_root_abs/scripts/slurm/guidance_eta_sweep_confidence_standalone_array.sbatch" \
  "$smoke_export" "$smoke_array_spec" "$gpu_partitions")
submitted_jobs+=("$smoke_job")
smoke_audit_job=$(submit_job "$smoke_job" \
  "$execution_root_abs/scripts/slurm/guidance_eta_sweep_confidence_standalone_audit.sbatch" \
  "$smoke_export" "" cpu_only)
submitted_jobs+=("$smoke_audit_job")
posebusters_smoke_job=$(submit_job "$smoke_audit_job" \
  "$execution_root_abs/scripts/slurm/guidance_eta_sweep_confidence_standalone_posebusters_smoke.sbatch" \
  "$smoke_export" "" cpu_only)
submitted_jobs+=("$posebusters_smoke_job")
sampling_dependency=$posebusters_smoke_job
stress_job=""
stress_audit_job=""
if [[ "$standalone_profile" == steric_high_eta_v1 ]]; then
  stress_export="$base_export,EFFDOCK_STANDALONE_SMOKE_GRID=0,EFFDOCK_STANDALONE_STRESS_ID=8f4j_pho"
  stress_job=$(submit_job "$posebusters_smoke_job" \
    "$execution_root_abs/scripts/slurm/guidance_eta_sweep_confidence_standalone_array.sbatch" \
    "$stress_export" "0" "$gpu_partitions")
  submitted_jobs+=("$stress_job")
  stress_audit_job=$(submit_job "$stress_job" \
    "$execution_root_abs/scripts/slurm/guidance_steric_high_eta_stress_audit.sbatch" \
    "$stress_export" "" cpu_only)
  submitted_jobs+=("$stress_audit_job")
  sampling_dependency=$stress_audit_job
fi
sampling_job=$(submit_job "$sampling_dependency" \
  "$execution_root_abs/scripts/slurm/guidance_eta_sweep_confidence_standalone_array.sbatch" \
  "$full_export" "$sampling_array_spec" "$gpu_partitions")
submitted_jobs+=("$sampling_job")
audit_job=$(submit_job "$sampling_job" \
  "$execution_root_abs/scripts/slurm/guidance_eta_sweep_confidence_standalone_audit.sbatch" \
  "$full_export" "" cpu_only)
submitted_jobs+=("$audit_job")
posebusters_job=$(submit_job "$audit_job" \
  "$execution_root_abs/scripts/slurm/guidance_eta_sweep_confidence_standalone_posebusters_array.sbatch" \
  "$full_export" "$posebusters_array_spec" cpu_only)
submitted_jobs+=("$posebusters_job")
report_job=$(submit_job "$posebusters_job" \
  "$execution_root_abs/scripts/slurm/guidance_eta_sweep_confidence_standalone_report.sbatch" \
  "$full_export" "" cpu_only)
submitted_jobs+=("$report_job")

printf '%s\n' \
  "smoke_job=$smoke_job" \
  "smoke_audit_job=$smoke_audit_job" \
  "posebusters_smoke_job=$posebusters_smoke_job" \
  "stress_job=${stress_job:-none}" \
  "stress_audit_job=${stress_audit_job:-none}" \
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
printf 'protocol_id=%s\n' "$protocol_id"
printf 'sweep_profile=%s\n' "$standalone_profile"
printf 'smoke_job=%s tasks=%s mapping=2_datasets_x_%s_eta dependency=none\n' \
  "$smoke_job" "$smoke_task_count" "$eta_count"
printf 'smoke_audit_job=%s dependency=afterok:%s\n' "$smoke_audit_job" "$smoke_job"
printf 'posebusters_smoke_job=%s dependency=afterok:%s selectors=2\n' \
  "$posebusters_smoke_job" "$smoke_audit_job"
if [[ -n "$stress_job" ]]; then
  printf 'stress_job=%s dependency=afterok:%s id=8f4j_pho eta=2.0\n' \
    "$stress_job" "$posebusters_smoke_job"
  printf 'stress_audit_job=%s dependency=afterok:%s\n' \
    "$stress_audit_job" "$stress_job"
fi
printf 'sampling_job=%s dependency=afterok:%s tasks=%s mapping=2_datasets_x_%s_eta_x_8_shards\n' \
  "$sampling_job" "$sampling_dependency" "$sampling_task_count" "$eta_count"
printf 'audit_job=%s dependency=afterok:%s coverage=85_astex_plus_308_posebusters\n' \
  "$audit_job" "$sampling_job"
printf 'posebusters_job=%s dependency=afterok:%s tasks=%s mapping=2_selectors_x_2_datasets_x_%s_eta_x_8_shards\n' \
  "$posebusters_job" "$audit_job" "$posebusters_task_count" "$eta_count"
printf 'report_job=%s dependency=afterok:%s\n' "$report_job" "$posebusters_job"
monitor_jobs=("$smoke_job" "$smoke_audit_job" "$posebusters_smoke_job")
if [[ -n "$stress_job" ]]; then
  monitor_jobs+=("$stress_job" "$stress_audit_job")
fi
monitor_jobs+=("$sampling_job" "$audit_job" "$posebusters_job" "$report_job")
monitor_csv=$(IFS=,; printf '%s' "${monitor_jobs[*]}")
printf 'monitor: squeue -j %s\n' "$monitor_csv"
