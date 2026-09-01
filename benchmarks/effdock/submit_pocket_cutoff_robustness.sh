#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$repo_root"
[[ $# -le 1 ]] || { echo "usage: $0 [SAFE_RUN_ID]" >&2; exit 2; }
run_id=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "unsafe run ID" >&2; exit 2; }

protocol_id=EFFDOCK-POCKET-CUTOFF-ROBUSTNESS-V1
output_root="$repo_root/outputs/benchmarks/effdock_pocket_cutoff_robustness_runs/$run_id"
execution_root="$repo_root/.effdock_execution_capsules/$protocol_id/$run_id"
[[ ! -e "$output_root" && ! -e "$execution_root" ]] || {
  echo "refusing to reuse output or execution root" >&2
  exit 2
}

for required in \
  .venv/bin/eff-dock \
  weights/effdock_docking_early_time_t0p10_50k.pt \
  weights/effdock_confidence_s50_raw_refined_u70k.pt \
  docs/EFFDOCK_POCKET_CUTOFF_ROBUSTNESS_PROTOCOL.md \
  docs/GUIDANCE_SIGMA2_ETA2_REFINEMENT_PROTOCOL.md \
  docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json \
  outputs/benchmarks/guidance_eta_sweep_v2_runs/20260801T102903Z/audit/combined.json; do
  [[ -f "$required" ]] || { echo "missing required input $required" >&2; exit 2; }
done
read -r docking_sha _ < <(sha256sum weights/effdock_docking_early_time_t0p10_50k.pt)
read -r confidence_sha _ < <(sha256sum weights/effdock_confidence_s50_raw_refined_u70k.pt)
[[ "$docking_sha" == 65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6 ]]
[[ "$confidence_sha" == ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638 ]]

mapfile -t package_files < <(rg --files src/effdock | sort)
mapfile -t benchmark_files < <(rg --files benchmarks/effdock | sort)
copy_files=(
  "${package_files[@]}" "${benchmark_files[@]}"
  pyproject.toml uv.lock configs/train.yaml
  docs/EFFDOCK_POCKET_CUTOFF_ROBUSTNESS_PROTOCOL.md
  docs/GUIDANCE_SIGMA2_ETA2_REFINEMENT_PROTOCOL.md
  docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json
  scripts/run_guidance_sdf_post_refinement.py
  scripts/run_guidance_sdf_post_refinement_shard.py
  scripts/score_guidance_sdf_post_refinement_confidence.py
  scripts/create_execution_capsule.py
)
capsule_args=()
for path in "${copy_files[@]}"; do capsule_args+=(--copy-file "$path"); done
.venv/bin/python scripts/create_execution_capsule.py \
  --repo-root "$repo_root" --output "$execution_root" \
  --link-root .venv --link-root data --link-root outputs --link-root weights \
  "${capsule_args[@]}" >/dev/null

mkdir -p "$output_root" outputs/benchmarks/logs
execution_root_abs=$(readlink -f "$execution_root")
protocol_sha=$(sha256sum docs/EFFDOCK_POCKET_CUTOFF_ROBUSTNESS_PROTOCOL.md | cut -d' ' -f1)
git_commit=$(git rev-parse HEAD)
read -r git_diff_sha256 _ < <(git diff --no-ext-diff | sha256sum)
printf '%s\n' \
  'status=submitting' "protocol_id=$protocol_id" "run_id=$run_id" \
  "output_root=$output_root" 'datasets=astex:85,posebusters:308' \
  'docking_cutoffs_angstrom=6,8,10,12' 'repeats=3' \
  'sampling=N100/S10,sigma2,late-power3,eta2-normalized-drift' \
  'refinement=adaptive100,abs0.02,rel0.001,patience5,min25,crop10' \
  'confidence=U70k,crop10,chunk20,pure-predicted-RMSD' \
  'validity=PoseBusters-0.6.5-redock-selected-pose' \
  "docking_checkpoint_sha256=$docking_sha" \
  "confidence_checkpoint_sha256=$confidence_sha" \
  "protocol_sha256=$protocol_sha" "git_commit=$git_commit" \
  "git_diff_sha256=$git_diff_sha256" "execution_root=$execution_root_abs" \
  > "$output_root/.submission.pending"

base_export="ALL,EFFDOCK_REPO_DIR=$execution_root_abs,EFFDOCK_RUNTIME_VENV=$repo_root/.venv,PYTHONPATH=$execution_root_abs/src,PYTHONDONTWRITEBYTECODE=1,EFFDOCK_OUTPUT_ROOT=$output_root"
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
  local dependency=$1 partition=$2 array=$3 export_spec=$4 script=$5 raw job
  args=(--parsable --partition="$partition" --export="$export_spec")
  [[ -z "$dependency" ]] || args+=(--dependency="afterok:$dependency")
  [[ -z "$array" ]] || args+=(--array="$array")
  raw=$(sbatch "${args[@]}" "$execution_root_abs/$script")
  job=${raw%%;*}
  [[ "$job" =~ ^[0-9]+$ ]] || { echo "invalid sbatch response: $raw" >&2; return 1; }
  printf '%s' "$job"
}

smoke_job=$(submit '' '6000ada,heavy,test' '0-1%2' "$base_export" \
  benchmarks/effdock/slurm/pocket_cutoff_end_to_end_smoke.sbatch); submitted+=("$smoke_job")
generation_job=$(submit "$smoke_job" '6000ada,heavy,test' '0-191%12' "$base_export" \
  benchmarks/effdock/slurm/pocket_cutoff_generation.sbatch); submitted+=("$generation_job")
manifest_job=$(submit "$generation_job" cpu_only '0-11%12' "$base_export" \
  benchmarks/effdock/slurm/pocket_cutoff_manifest.sbatch); submitted+=("$manifest_job")
refinement_job=$(submit "$manifest_job" '6000ada,heavy,test' '0-383%12' \
  "$base_export,EFFDOCK_STAGE=refinement" \
  benchmarks/effdock/slurm/pocket_cutoff_refine_confidence.sbatch); submitted+=("$refinement_job")
confidence_job=$(submit "$refinement_job" '6000ada,heavy,test' '0-383%12' \
  "$base_export,EFFDOCK_STAGE=confidence" \
  benchmarks/effdock/slurm/pocket_cutoff_refine_confidence.sbatch); submitted+=("$confidence_job")
posebusters_job=$(submit "$confidence_job" cpu_only '0-191%12' "$base_export" \
  benchmarks/effdock/slurm/pocket_cutoff_selected_posebusters.sbatch); submitted+=("$posebusters_job")
report_job=$(submit "$posebusters_job" cpu_only '' "$base_export" \
  benchmarks/effdock/slurm/pocket_cutoff_report.sbatch); submitted+=("$report_job")

printf '%s\n' \
  "smoke_job=$smoke_job" "generation_job=$generation_job" \
  "manifest_job=$manifest_job" "refinement_job=$refinement_job" \
  "confidence_job=$confidence_job" "posebusters_job=$posebusters_job" \
  "report_job=$report_job" "submitted_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >> "$output_root/.submission.pending"
sed -i 's/^status=submitting$/status=submitted/' "$output_root/.submission.pending"
mv "$output_root/.submission.pending" "$output_root/.submission"
committed=1
trap - EXIT
printf 'output_root=%s\nsmoke=%s generation=%s manifest=%s refinement=%s confidence=%s posebusters=%s report=%s\n' \
  "$output_root" "$smoke_job" "$generation_job" "$manifest_job" \
  "$refinement_job" "$confidence_job" "$posebusters_job" "$report_job"
