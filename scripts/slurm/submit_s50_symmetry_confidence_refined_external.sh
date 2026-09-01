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
protocol_sha=91bb6dd6092731afd501ba07ed38ce219a75ab70e48786e5de73105471412bed
runner_sha=1942152b59000880bdc2ad6bb0c05d6f2e2af36f46aa5e1346209c218a55290a
reporter_sha=dd5c41b6c2919898ab2accdc471440860eb712e444031ab3162b1fbf3ef74fb0
scorer_sha=d586b6bc9205e9c00d2b983f5c2802b03105e96c55a63b908ad543e3d471f808
source_manifest_sha=9e8be4d47dba8e346a6900b6bf02f5b853a93141f571f5c65b4c719de632d695
src_sha=ffb5d1190064fad81ff45a538105d5c20cf50a455aca3d813fadd6e3ca5fd17e
verify_sha docs/S50_SYMMETRY_CONFIDENCE_REFINED_EXTERNAL_PROTOCOL.md "$protocol_sha"
verify_sha scripts/run_s50_symmetry_confidence_refined_external_shard.py "$runner_sha"
verify_sha scripts/report_s50_symmetry_confidence_refined_external.py "$reporter_sha"
verify_sha scripts/score_guidance_sdf_post_refinement_confidence.py "$scorer_sha"
verify_sha outputs/benchmarks/guidance_sigma2_eta2_refinement_runs/20260819T060555Z/manifest.json "$source_manifest_sha"
verify_sha outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step50000_ema_common_init.pt 65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6
verify_sha outputs/eff-dock/s50-confidence-symmetry-training/88509b2189ab619c91feee843e4d90388b85b36b181532ca0a54ebf271f69b97/recovery/anchors/u001500_job56496_latest.pt 2af26bf66bec53676b8344e811911bbf47ee85aa6550610f35c3812b7a7f9d15
verify_sha outputs/eff-dock/s50-confidence-symmetry-training/88509b2189ab619c91feee843e4d90388b85b36b181532ca0a54ebf271f69b97/full/best.pt 1c59034172fb925cc8a70777dcba236be349f1a1de1775d49cc17d492b17c030
verify_sha outputs/eff-dock/s50-confidence-symmetry-training/88509b2189ab619c91feee843e4d90388b85b36b181532ca0a54ebf271f69b97/full/latest.pt fd49fa86f67187bf26d6c1bcf2daf925ba3e3b19dfeae733e57535d183280469
actual_src_sha=$(find src/effdock -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
[[ "$actual_src_sha" == "$src_sha" ]] || { echo "source tree SHA mismatch" >&2; exit 2; }
refinement_count=$(find outputs/benchmarks/guidance_sdf_post_refinement_runs/sigma2-eta2-adaptive-20260819T062833Z/full/refinement -mindepth 3 -maxdepth 3 -type f -name summary.json | wc -l)
[[ "$refinement_count" == 393 ]] || { echo "expected 393 refinement summaries" >&2; exit 2; }
content_id=$(printf '%s\0' \
  EFFDOCK-S50-SYMMETRY-CONFIDENCE-REFINED-EXTERNAL-V1 \
  "$protocol_sha" "$runner_sha" "$reporter_sha" "$scorer_sha" "$source_manifest_sha" "$src_sha" \
  2af26bf66bec53676b8344e811911bbf47ee85aa6550610f35c3812b7a7f9d15 \
  1c59034172fb925cc8a70777dcba236be349f1a1de1775d49cc17d492b17c030 \
  fd49fa86f67187bf26d6c1bcf2daf925ba3e3b19dfeae733e57535d183280469 \
  65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6 | sha256sum | cut -d' ' -f1)
output_root="$repo_dir/outputs/benchmarks/s50_symmetry_confidence_refined_external_runs/$content_id"
execution_root="$repo_dir/.effdock_execution_capsules/EFFDOCK-S50-SYMMETRY-CONFIDENCE-REFINED-EXTERNAL-V1/$content_id"
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

export_args="ALL,EFFDOCK_EXECUTION_ROOT=$execution_root,EFFDOCK_OUTPUT_ROOT=$output_root"
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
smoke_job=$(sbatch --parsable --hold --array=0-2%3 --export="$export_args,EFFDOCK_STAGE=smoke" \
  scripts/slurm/s50_symmetry_confidence_refined_external.sbatch)
full_job=$(sbatch --parsable --dependency="afterok:$smoke_job" --array=0-95%8 \
  --export="$export_args,EFFDOCK_STAGE=full" \
  scripts/slurm/s50_symmetry_confidence_refined_external.sbatch)
report_job=$(sbatch --parsable --dependency="afterok:$full_job" --export="$export_args" \
  scripts/slurm/s50_symmetry_confidence_refined_external_report.sbatch)
printf '%s\n' \
  'status=submitted' \
  'protocol_id=EFFDOCK-S50-SYMMETRY-CONFIDENCE-REFINED-EXTERNAL-V1' \
  "content_id=$content_id" \
  "execution_root=$execution_root" \
  "smoke_job=$smoke_job" \
  "full_job=$full_job" \
  "report_job=$report_job" \
  'arms=u001500,u025000,u050000' \
  'cohort=astex85,posebusters308' \
  'poses_per_complex=100' \
  'sigma=2' > "$output_root/.submission"
scontrol release "$smoke_job"
trap - ERR
printf 'output_root=%s\nsmoke=%s full=%s report=%s\n' "$output_root" "$smoke_job" "$full_job" "$report_job"
