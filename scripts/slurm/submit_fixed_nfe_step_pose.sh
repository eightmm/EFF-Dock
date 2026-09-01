#!/usr/bin/env bash
set -euo pipefail

die() { echo "$*" >&2; exit 2; }
verify_sha() {
  local path=$1 expected=$2 actual
  [[ -f "$path" ]] || die "missing frozen input: $path"
  actual=$(sha256sum "$path" | cut -d' ' -f1)
  [[ "$actual" == "$expected" ]] || die "SHA-256 mismatch: $path"
}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
[[ $# -le 1 ]] || die "usage: $0 [SAFE_RUN_ID]"
run_id=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "unsafe run ID"

protocol_id=EFFDOCK-FIXED-NFE-STEP-POSE-V1
output_root_rel="outputs/benchmarks/fixed_nfe_step_pose_runs/$run_id"
output_root="$repo_root/$output_root_rel"
sampling_capsule="$repo_root/.effdock_execution_capsules/EFFDOCK-UNIFIED-GUIDANCE-SIGMA-SWEEP-ETA2-V1/20260809T031535Z"
refinement_capsule="$repo_root/.effdock_execution_capsules/EFFDOCK-GUIDANCE-SDF-POST-REFINEMENT-CONFIDENCE-V2/sigma2-eta2-adaptive-recovery-20260819T112108Z"
confidence_capsule="$repo_root/.effdock_execution_capsules/EFFDOCK-S50-SYMMETRY-CONFIDENCE-REFINED-EXTERNAL-V1/85574152669f7d6a8fa6d60ba2ad7e2e4e9b37a5a5840ac57eeeffe9011100c9"
n100_manifest="$repo_root/outputs/benchmarks/guidance_sigma2_eta2_refinement_runs/20260819T060555Z/manifest.json"
n100_scores_root="$repo_root/outputs/benchmarks/s50_symmetry_confidence_refined_external_runs/85574152669f7d6a8fa6d60ba2ad7e2e4e9b37a5a5840ac57eeeffe9011100c9/full/u050000"
n100_report="$repo_root/outputs/benchmarks/s50_symmetry_confidence_refined_external_runs/85574152669f7d6a8fa6d60ba2ad7e2e4e9b37a5a5840ac57eeeffe9011100c9/report.json"
cohort_audit="$repo_root/outputs/benchmarks/guidance_eta_sweep_v2_runs/20260801T102903Z/audit/combined.json"
docking_checkpoint="$repo_root/outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step50000_ema_common_init.pt"
confidence_checkpoint="$repo_root/outputs/eff-dock/s50-confidence-symmetry-training/88509b2189ab619c91feee843e4d90388b85b36b181532ca0a54ebf271f69b97/full/latest.pt"

verify_sha "$sampling_capsule/execution_capsule.json" 62c698577a3b4ad407b9926ec922dae201fbce45e189e6aae2b83c4d4fe0cb35
verify_sha "$refinement_capsule/execution_capsule.json" 4891267ff04d52915be1f3c39a9a78ffa82a19e87589c28971f3cf7d63becb75
verify_sha "$n100_manifest" 9e8be4d47dba8e346a6900b6bf02f5b853a93141f571f5c65b4c719de632d695
verify_sha "$n100_report" 501d2010a4df65fb0d9779e66113c7f3f423cd418d4ac683d903ec9b3fe1590a
verify_sha "$cohort_audit" dac7903488ccd36552a9bca134e37e633e3f07166d94f0389837012081ff3048
verify_sha "$docking_checkpoint" 65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6
verify_sha "$confidence_checkpoint" fd49fa86f67187bf26d6c1bcf2daf925ba3e3b19dfeae733e57535d183280469
verify_sha docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json 99f15f557644cc51c3dd1f559b0dd97dd4259c1de3e1403fb761b7c7e079f668
verify_sha weights/effdock_geometry_ft_100k_best.pt 6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db
verify_sha weights/effdock_confidence_extmatch_n80_s25_step42500.pt e31fde6f351284205c78f7a1510002779c43312e94d9f82003d47a14d72bc78f
confidence_src_sha=$(cd "$confidence_capsule" && find src/effdock -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
[[ "$confidence_src_sha" == ffb5d1190064fad81ff45a538105d5c20cf50a455aca3d813fadd6e3ca5fd17e ]] \
  || die "confidence capsule source identity mismatch"
[[ -d "$n100_scores_root" ]] || die "missing reused N100/S10 U50 score root"
[[ -x .venv/bin/python && -x .venv/bin/eff-dock ]] || die "missing synchronized .venv"

copy_files=(
  pyproject.toml
  uv.lock
  docs/FIXED_NFE_STEP_POSE_PROTOCOL.md
  docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json
  scripts/build_fixed_nfe_step_pose_manifest.py
  scripts/report_fixed_nfe_step_pose.py
  scripts/run_fixed_nfe_step_pose_refinement.py
  scripts/run_fixed_nfe_step_pose_confidence.py
  scripts/run_fixed_nfe_step_pose_stage.py
  scripts/run_guidance_sdf_post_refinement.py
  scripts/score_guidance_sdf_post_refinement_confidence.py
  scripts/slurm/fixed_nfe_step_pose_sampling.sbatch
  scripts/slurm/fixed_nfe_step_pose_manifest.sbatch
  scripts/slurm/fixed_nfe_step_pose_stage.sbatch
  scripts/slurm/fixed_nfe_step_pose_report.sbatch
  scripts/slurm/submit_fixed_nfe_step_pose.sh
)
capsule_args=()
for path in "${copy_files[@]}"; do
  [[ -f "$path" ]] || die "missing control file: $path"
  capsule_args+=(--copy-file "$path")
done
control_capsule_rel=".effdock_execution_capsules/$protocol_id/$run_id"
control_capsule="$repo_root/$control_capsule_rel"
.venv/bin/python scripts/create_execution_capsule.py \
  --repo-root "$repo_root" \
  --output "$control_capsule" \
  --link-root .venv --link-root data --link-root outputs --link-root weights \
  "${capsule_args[@]}" >/dev/null
control_capsule_sha=$(sha256sum "$control_capsule/execution_capsule.json" | cut -d' ' -f1)

mkdir -p outputs/benchmarks/fixed_nfe_step_pose_runs outputs/benchmarks/logs
mkdir "$output_root" || die "refusing to reuse output root: $output_root"
submitted=()
committed=0
cleanup() {
  local code=$?
  trap - EXIT
  if [[ "$committed" -eq 0 ]]; then
    local job
    for job in "${submitted[@]}"; do
      scancel "$job" 2>/dev/null || true
    done
  fi
  exit "$code"
}
trap cleanup EXIT

git_commit=$(git rev-parse HEAD)
git_diff_sha256=$(git diff --no-ext-diff | sha256sum | cut -d' ' -f1)
protocol_sha256=$(sha256sum docs/FIXED_NFE_STEP_POSE_PROTOCOL.md | cut -d' ' -f1)
printf '%s\n' \
  'status=submitting' \
  "protocol_id=$protocol_id" \
  "run_id=$run_id" \
  "output_root=$output_root" \
  'datasets=astex:85,posebusters:308' \
  'new_arm=s25_n40' \
  'reused_arm=s10_n100' \
  'learned_pose_steps_per_arm=1000' \
  'sigma=2.0' \
  'eta=2.0' \
  'refinement=adaptive100' \
  'selector=u050000_argmin_predicted_rmsd' \
  'seed=42_plus_global_complex_offset' \
  'seed_replicates=1' \
  'sampling_hardware=test:gpu:a5000:1' \
  "sampling_capsule=$sampling_capsule" \
  "refinement_capsule=$refinement_capsule" \
  "confidence_capsule=$confidence_capsule" \
  "control_capsule=$control_capsule" \
  "control_capsule_sha256=$control_capsule_sha" \
  "protocol_sha256=$protocol_sha256" \
  "git_commit=$git_commit" \
  "git_diff_sha256=$git_diff_sha256" > "$output_root/.submission.pending"

base_export="ALL,EFFDOCK_OUTPUT_ROOT=$output_root,EFFDOCK_CONTROL_CAPSULE=$control_capsule"
base_export+=",EFFDOCK_SAMPLING_CAPSULE=$sampling_capsule,EFFDOCK_REFINEMENT_CAPSULE=$refinement_capsule"
base_export+=",EFFDOCK_CONFIDENCE_CAPSULE=$confidence_capsule,EFFDOCK_COHORT_AUDIT=$cohort_audit"
base_export+=",EFFDOCK_N100_MANIFEST=$n100_manifest,EFFDOCK_N100_SCORES_ROOT=$n100_scores_root"
base_export+=",EFFDOCK_N100_REPORT=$n100_report,EFFDOCK_DOCKING_CHECKPOINT=$docking_checkpoint"
base_export+=",EFFDOCK_CONFIDENCE_CHECKPOINT=$confidence_checkpoint"

submit() {
  local dependency=$1 partition=$2 script=$3 array=${4:-} extra_export=${5:-}
  local args=(--parsable --partition="$partition" --export="$base_export$extra_export")
  [[ -z "$dependency" ]] || args+=(--dependency="afterok:$dependency")
  [[ -z "$array" ]] || args+=(--array="$array")
  local raw job
  raw=$(sbatch "${args[@]}" "$control_capsule/$script")
  job=${raw%%;*}
  [[ "$job" =~ ^[0-9]+$ ]] || die "invalid sbatch response: $raw"
  printf '%s' "$job"
}

smoke_sampling=$(submit '' test scripts/slurm/fixed_nfe_step_pose_sampling.sbatch '0-1%2' ',EFFDOCK_RUN_MODE=smoke')
submitted+=("$smoke_sampling")
smoke_manifest=$(submit "$smoke_sampling" cpu_only scripts/slurm/fixed_nfe_step_pose_manifest.sbatch '' ',EFFDOCK_RUN_MODE=smoke')
submitted+=("$smoke_manifest")
smoke_refinement=$(submit "$smoke_manifest" '6000ada,heavy,test' scripts/slurm/fixed_nfe_step_pose_stage.sbatch '0-1%2' ',EFFDOCK_RUN_MODE=smoke,EFFDOCK_STAGE=refinement')
submitted+=("$smoke_refinement")
smoke_confidence=$(submit "$smoke_refinement" 6000ada scripts/slurm/fixed_nfe_step_pose_stage.sbatch '0-1%2' ',EFFDOCK_RUN_MODE=smoke,EFFDOCK_STAGE=confidence')
submitted+=("$smoke_confidence")
full_sampling=$(submit "$smoke_confidence" test scripts/slurm/fixed_nfe_step_pose_sampling.sbatch '0-15%8' ',EFFDOCK_RUN_MODE=full')
submitted+=("$full_sampling")
full_manifest=$(submit "$full_sampling" cpu_only scripts/slurm/fixed_nfe_step_pose_manifest.sbatch '' ',EFFDOCK_RUN_MODE=full')
submitted+=("$full_manifest")
full_refinement=$(submit "$full_manifest" '6000ada,heavy,test' scripts/slurm/fixed_nfe_step_pose_stage.sbatch '0-31%12' ',EFFDOCK_RUN_MODE=full,EFFDOCK_STAGE=refinement')
submitted+=("$full_refinement")
full_confidence=$(submit "$full_refinement" 6000ada scripts/slurm/fixed_nfe_step_pose_stage.sbatch '0-31%12' ',EFFDOCK_RUN_MODE=full,EFFDOCK_STAGE=confidence')
submitted+=("$full_confidence")
report=$(submit "$full_confidence" cpu_only scripts/slurm/fixed_nfe_step_pose_report.sbatch '' '')
submitted+=("$report")

printf '%s\n' \
  "smoke_sampling_job=$smoke_sampling" \
  "smoke_manifest_job=$smoke_manifest" \
  "smoke_refinement_job=$smoke_refinement" \
  "smoke_confidence_job=$smoke_confidence" \
  "full_sampling_job=$full_sampling" \
  "full_manifest_job=$full_manifest" \
  "full_refinement_job=$full_refinement" \
  "full_confidence_job=$full_confidence" \
  "report_job=$report" \
  "submitted_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$output_root/.submission.pending"
sed -i 's/^status=submitting$/status=submitted/' "$output_root/.submission.pending"
mv "$output_root/.submission.pending" "$output_root/.submission"
committed=1
trap - EXIT
printf 'output_root=%s\nsmoke_sampling=%s\nfull_sampling=%s\nfull_refinement=%s\nfull_confidence=%s\nreport=%s\n' \
  "$output_root" "$smoke_sampling" "$full_sampling" "$full_refinement" "$full_confidence" "$report"
