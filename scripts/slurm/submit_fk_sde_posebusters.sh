#!/usr/bin/env bash
# Submit the frozen smoke -> paired sampling -> all-pose official PB chain.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [SAFE_RUN_ID]" >&2
  exit 2
fi
run_id=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "unsafe FK-SDE PoseBusters run ID" >&2
  exit 2
fi

protocol_id=EFFDOCK-FK-TRANSLATION-SDE-POSEBUSTERS-V1
output_root="outputs/benchmarks/fk_sde_posebusters_runs/$run_id"
mkdir -p outputs/benchmarks/fk_sde_posebusters_runs outputs/benchmarks/logs
if ! mkdir "$output_root"; then
  echo "refusing to reuse FK-SDE PoseBusters output root: $output_root" >&2
  exit 2
fi

submitted=()
committed=0
cleanup() {
  local code=$?
  trap - EXIT
  if [[ "$committed" -eq 0 ]]; then
    for job in "${submitted[@]}"; do
      scancel "$job" 2>/dev/null || true
    done
  fi
  exit "$code"
}
trap cleanup EXIT

for required in \
  .venv/bin/eff-dock \
  pyproject.toml \
  uv.lock \
  configs/train.yaml \
  weights/effdock_geometry_ft_100k_best.pt \
  weights/effdock_confidence_extmatch_n80_s25_step42500.pt \
  docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json \
  docs/GUIDANCE_BUDGET1000_FULL_COHORT.json \
  docs/FK_SDE_POSEBUSTERS_PROTOCOL.md \
  data/external_test/posebusters_reference_pocket_centers.json \
  data/external_benchmarks/data/posebusters_benchmark_set; do
  if [[ ! -e "$required" ]]; then
    echo "missing frozen FK-SDE PoseBusters input: $required" >&2
    exit 2
  fi
done

mapfile -t package_files < <(rg --files src/effdock | sort)
copy_files=(
  "${package_files[@]}"
  pyproject.toml
  uv.lock
  configs/train.yaml
  docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json
  docs/GUIDANCE_BUDGET1000_FULL_COHORT.json
  docs/FK_SDE_POSEBUSTERS_PROTOCOL.md
  scripts/create_execution_capsule.py
  scripts/slurm/fk_sde_posebusters_sampling_task.sbatch
  scripts/slurm/fk_sde_posebusters_sampling.sbatch
  scripts/slurm/fk_sde_posebusters_smoke_sampling.sbatch
  scripts/slurm/fk_sde_posebusters_manifest.sbatch
  scripts/slurm/fk_sde_posebusters_official.sbatch
  scripts/slurm/fk_sde_posebusters_smoke_audit.sbatch
  scripts/slurm/fk_sde_posebusters_report.sbatch
  scripts/slurm/submit_fk_sde_posebusters.sh
)
capsule_args=()
for path in "${copy_files[@]}"; do
  capsule_args+=(--copy-file "$path")
done
execution_root=".effdock_execution_capsules/$protocol_id/$run_id"
.venv/bin/python scripts/create_execution_capsule.py \
  --repo-root "$repo_root" \
  --output "$repo_root/$execution_root" \
  --link-root .venv \
  --link-root data \
  --link-root weights \
  --link-root outputs \
  "${capsule_args[@]}" >/dev/null
execution_root_abs=$(readlink -f "$execution_root")
output_root_abs=$(readlink -f "$output_root")
git_commit=$(git rev-parse HEAD)
read -r git_diff_sha256 _ < <(git diff --no-ext-diff | sha256sum)

printf '%s\n' \
  'status=submitting' \
  "protocol_id=$protocol_id" \
  "run_id=$run_id" \
  "output_root=$output_root_abs" \
  'dataset=posebusters' \
  'complexes=308' \
  'arms=fk_ode,fk_sde' \
  'sampling_budget=N40/S25' \
  'translation_sde_base_sigma=0.3' \
  'fk_beta=0.01' \
  'fk_times=0.3,0.6,0.8' \
  'official_posebusters_version=0.6.5' \
  'official_pose_rows=24640' \
  "git_commit=$git_commit" \
  "git_diff_sha256=$git_diff_sha256" \
  "execution_root=$execution_root_abs" > "$output_root/.submission.pending"

base_export="ALL,EFFDOCK_REPO_DIR=$execution_root_abs"
base_export+=",PYTHONPATH=$execution_root_abs/src,PYTHONDONTWRITEBYTECODE=1"
base_export+=",EFFDOCK_FK_SDE_PB_OUTPUT_ROOT=$output_root_abs"

submit() {
  local dependency=$1
  local mode=$2
  local array_override=$3
  local script=$4
  local export_spec=$base_export
  local raw
  local args=(--parsable)
  if [[ -n "$mode" ]]; then
    export_spec+=",EFFDOCK_FK_SDE_PB_MODE=$mode"
  fi
  args+=(--export="$export_spec")
  if [[ -n "$dependency" ]]; then
    args+=(--dependency="afterok:$dependency")
  fi
  if [[ -n "$array_override" ]]; then
    args+=(--array="$array_override")
  fi
  raw=$(sbatch "${args[@]}" "$script")
  local job=${raw%%;*}
  if [[ ! "$job" =~ ^[0-9]+$ ]]; then
    echo "invalid sbatch response for $script: $raw" >&2
    return 1
  fi
  printf '%s' "$job"
}

smoke_sampling_job=$(submit "" "" "" \
  "$execution_root_abs/scripts/slurm/fk_sde_posebusters_smoke_sampling.sbatch")
submitted+=("$smoke_sampling_job")
smoke_manifest_job=$(submit "$smoke_sampling_job" smoke "" \
  "$execution_root_abs/scripts/slurm/fk_sde_posebusters_manifest.sbatch")
submitted+=("$smoke_manifest_job")
smoke_official_job=$(submit "$smoke_manifest_job" smoke 0 \
  "$execution_root_abs/scripts/slurm/fk_sde_posebusters_official.sbatch")
submitted+=("$smoke_official_job")
smoke_audit_job=$(submit "$smoke_official_job" "" "" \
  "$execution_root_abs/scripts/slurm/fk_sde_posebusters_smoke_audit.sbatch")
submitted+=("$smoke_audit_job")
sampling_job=$(submit "$smoke_audit_job" "" "" \
  "$execution_root_abs/scripts/slurm/fk_sde_posebusters_sampling.sbatch")
submitted+=("$sampling_job")
manifest_job=$(submit "$sampling_job" full "" \
  "$execution_root_abs/scripts/slurm/fk_sde_posebusters_manifest.sbatch")
submitted+=("$manifest_job")
official_job=$(submit "$manifest_job" full "" \
  "$execution_root_abs/scripts/slurm/fk_sde_posebusters_official.sbatch")
submitted+=("$official_job")
report_job=$(submit "$official_job" "" "" \
  "$execution_root_abs/scripts/slurm/fk_sde_posebusters_report.sbatch")
submitted+=("$report_job")

printf '%s\n' \
  "smoke_sampling_job=$smoke_sampling_job" \
  "smoke_manifest_job=$smoke_manifest_job dependency=afterok:$smoke_sampling_job" \
  "smoke_official_job=$smoke_official_job dependency=afterok:$smoke_manifest_job" \
  "smoke_audit_job=$smoke_audit_job dependency=afterok:$smoke_official_job" \
  "sampling_job=$sampling_job dependency=afterok:$smoke_audit_job" \
  "manifest_job=$manifest_job dependency=afterok:$sampling_job" \
  "official_job=$official_job dependency=afterok:$manifest_job" \
  "report_job=$report_job dependency=afterok:$official_job" \
  "submitted_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$output_root/.submission.pending"
sed -i 's/^status=submitting$/status=submitted/' "$output_root/.submission.pending"
mv "$output_root/.submission.pending" "$output_root/.submission"
committed=1
trap - EXIT

printf 'output_root=%s\nsmoke_sampling_job=%s\nsampling_job=%s\nofficial_job=%s\nreport_job=%s\n' \
  "$output_root_abs" "$smoke_sampling_job" "$sampling_job" "$official_job" "$report_job"
