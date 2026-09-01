#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "${1:-.}" && pwd -P)
cd "$repo_dir"
verify_sha() {
  local path=$1 expected=$2 actual
  [[ -f "$path" && ! -L "$path" ]] || { echo "missing regular file: $path" >&2; exit 2; }
  read -r actual _ < <(sha256sum "$path")
  [[ "$actual" == "$expected" ]] || { echo "SHA mismatch: $path" >&2; exit 2; }
}

verify_sha outputs/eff-dock/s50-raw-refined-confidence-runs/309bb1a7645f8d09a796e45559d018c969d077a937ecb225ea53314a7143c804/training-100k-runs/8641cbe7b5bc99896c6513073ca7d81d4c00db42015a3739685043e5e4fa162f/full/best.pt ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638
verify_sha outputs/eff-dock/s50-raw-refined-confidence-runs/309bb1a7645f8d09a796e45559d018c969d077a937ecb225ea53314a7143c804/training-100k-runs/8641cbe7b5bc99896c6513073ca7d81d4c00db42015a3739685043e5e4fa162f/full/latest.pt 2ea1aca4f1c326cd0841e76c3597e3749231854a523d1ba8bd923c6fb5a9bff8
verify_sha outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step50000_ema_common_init.pt 65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6
verify_sha outputs/benchmarks/guidance_sigma2_eta2_refinement_runs/20260819T060555Z/manifest.json 9e8be4d47dba8e346a6900b6bf02f5b853a93141f571f5c65b4c719de632d695
verify_sha configs/train.yaml 39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec
verify_sha docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json 99f15f557644cc51c3dd1f559b0dd97dd4259c1de3e1403fb761b7c7e079f668

protocol_sha=$(sha256sum docs/S50_RAW_REFINED_CONFIDENCE_EXTERNAL_PROTOCOL.md | cut -d' ' -f1)
runner_sha=$(sha256sum scripts/run_s50_raw_refined_confidence_external_shard.py | cut -d' ' -f1)
helper_runner_sha=$(sha256sum scripts/run_s50_symmetry_confidence_refined_external_shard.py | cut -d' ' -f1)
reporter_sha=$(sha256sum scripts/report_s50_raw_refined_confidence_external.py | cut -d' ' -f1)
helper_reporter_sha=$(sha256sum scripts/report_s50_symmetry_confidence_refined_external.py | cut -d' ' -f1)
scorer_sha=$(sha256sum scripts/score_guidance_sdf_post_refinement_confidence.py | cut -d' ' -f1)
score_batch_sha=$(sha256sum scripts/slurm/s50_raw_refined_confidence_external.sbatch | cut -d' ' -f1)
report_batch_sha=$(sha256sum scripts/slurm/s50_raw_refined_confidence_external_report.sbatch | cut -d' ' -f1)
src_sha=$(find src/effdock -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
refinement_count=$(find outputs/benchmarks/guidance_sdf_post_refinement_runs/sigma2-eta2-adaptive-20260819T062833Z/full/refinement -mindepth 3 -maxdepth 3 -type f -name summary.json | wc -l)
official_count=$(find outputs/benchmarks/guidance_sdf_post_refinement_runs/sigma2-eta2-adaptive-20260819T062833Z/full/posebusters_step100 -mindepth 2 -maxdepth 2 -type f -name summary.json | wc -l)
[[ "$refinement_count" == 393 ]] || { echo "expected 393 refinement summaries" >&2; exit 2; }
[[ "$official_count" == 32 ]] || { echo "expected 32 official validity shards" >&2; exit 2; }

content_id=$(printf '%s\0' \
  EFFDOCK-S50-RAW-REFINED-CONFIDENCE-EXTERNAL-V1 \
  "$protocol_sha" "$runner_sha" "$helper_runner_sha" "$reporter_sha" \
  "$helper_reporter_sha" "$scorer_sha" "$score_batch_sha" "$report_batch_sha" \
  "$src_sha" \
  9e8be4d47dba8e346a6900b6bf02f5b853a93141f571f5c65b4c719de632d695 \
  39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec \
  99f15f557644cc51c3dd1f559b0dd97dd4259c1de3e1403fb761b7c7e079f668 \
  ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638 \
  2ea1aca4f1c326cd0841e76c3597e3749231854a523d1ba8bd923c6fb5a9bff8 \
  65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6 | sha256sum | cut -d' ' -f1)
output_root="$repo_dir/outputs/benchmarks/s50_raw_refined_confidence_external_runs/$content_id"
execution_root="$repo_dir/.effdock_execution_capsules/EFFDOCK-S50-RAW-REFINED-CONFIDENCE-EXTERNAL-V1/$content_id"
[[ ! -e "$output_root" && ! -e "$execution_root" ]] || {
  echo "refusing to reuse output or capsule" >&2
  exit 2
}

temporary="$execution_root.tmp.$$"
mkdir -p "$temporary" "$output_root" "$repo_dir/outputs/slurm"
trap 'rm -rf "$temporary"' EXIT
cp -a src scripts docs configs "$temporary/"
ln -s "$repo_dir/.venv" "$temporary/.venv"
ln -s "$repo_dir/data" "$temporary/data"
ln -s "$repo_dir/outputs" "$temporary/outputs"
mv "$temporary" "$execution_root"
trap - EXIT

export_args="ALL,EFFDOCK_EXECUTION_ROOT=$execution_root,EFFDOCK_OUTPUT_ROOT=$output_root,EFFDOCK_PROTOCOL_SHA=$protocol_sha,EFFDOCK_RUNNER_SHA=$runner_sha,EFFDOCK_HELPER_RUNNER_SHA=$helper_runner_sha,EFFDOCK_REPORTER_SHA=$reporter_sha,EFFDOCK_HELPER_REPORTER_SHA=$helper_reporter_sha,EFFDOCK_SCORER_SHA=$scorer_sha,EFFDOCK_SRC_SHA=$src_sha"
smoke_job=
full_job=
report_job=
cancel_submitted_jobs() {
  local job
  for job in "$smoke_job" "$full_job" "$report_job"; do
    [[ -z "$job" ]] || scancel "$job"
  done
}
trap cancel_submitted_jobs ERR
smoke_job=$(sbatch --parsable --hold --array=0-1%2 --export="$export_args,EFFDOCK_STAGE=smoke" \
  scripts/slurm/s50_raw_refined_confidence_external.sbatch)
full_job=$(sbatch --parsable --dependency="afterok:$smoke_job" --array=0-63%8 \
  --export="$export_args,EFFDOCK_STAGE=full" \
  scripts/slurm/s50_raw_refined_confidence_external.sbatch)
report_job=$(sbatch --parsable --dependency="afterok:$full_job" --export="$export_args" \
  scripts/slurm/s50_raw_refined_confidence_external_report.sbatch)
printf '%s\n' \
  'status=submitted' \
  'protocol_id=EFFDOCK-S50-RAW-REFINED-CONFIDENCE-EXTERNAL-V1' \
  "content_id=$content_id" \
  "execution_root=$execution_root" \
  "smoke_job=$smoke_job" \
  "full_job=$full_job" \
  "report_job=$report_job" \
  'arms=u070000,u100000' \
  'cohort=astex85,posebusters308' \
  'poses_per_complex=100' \
  'stages=raw_step0,refined_step100' \
  'sigma=2' > "$output_root/.submission"
scontrol release "$smoke_job"
trap - ERR
printf 'output_root=%s\nsmoke=%s full=%s report=%s\n' \
  "$output_root" "$smoke_job" "$full_job" "$report_job"
