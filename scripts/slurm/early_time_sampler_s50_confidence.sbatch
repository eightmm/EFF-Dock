#!/usr/bin/env bash
#SBATCH --job-name=effdock-s50-conf-score
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err
#SBATCH --partition=6000ada
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%4

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
repo_dir=$(cd "$repo_dir" && pwd -P)
cd "$repo_dir"
mkdir -p outputs/benchmarks/logs

protocol_id=EFFDOCK-S50-CONFIDENCE-SCORE-ONLY-PLINDER-V1
protocol=$repo_dir/docs/EARLY_TIME_SAMPLER_S50_CONFIDENCE_PROTOCOL.md
scorer=$repo_dir/scripts/score_early_time_sampler_plinder_confidence.py
reporter=$repo_dir/scripts/report_early_time_sampler_plinder_confidence.py
config=$repo_dir/configs/train.yaml
confidence_checkpoint=$repo_dir/weights/effdock_confidence_extmatch_n80_s25_step42500.pt
s50_checkpoint=$repo_dir/outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step50000_ema_common_init.pt
matched_checkpoint=$repo_dir/weights/effdock_geometry_ft_100k_best.pt

# These placeholders are replaced only after the scorer, reporter, protocol,
# and runtime dependency set pass their final static checks. Any placeholder
# is fatal.
protocol_sha256=bab703f0de5c10baff5506afbe88354f3109b2f483dff8425fbbad512fd9eb1b
scorer_sha256=193f35972076d87d413ee09d50053e55a44cde53e55c5abeac28ca686098d6e6
reporter_sha256=98cf21c15633cd9f9d46d1b99fd62b899199a9ebd6ad8448a7d70da461cd0add
runtime_code_sha256=47bddf89a08dcfd95a095a690f41f21a03c2b22a5ccd617e4c7e535f5062fae3

config_sha256=39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec
eligibility_manifest_sha256=6ebeb2d165e1def6ebf7b5bba301f82d4a9c3ff9d6c5cd43616dcf09edbd38ac
source_sampler_protocol_sha256=0250853ae0793db288be2a6a8dc775db391d25aae32835b65b061782f34ab518
source_sampler_report_sha256=d4814796a9d274f836888dd614e5b6a4a5fba6b86001da83bea6720fabf02316
source_coordinate_audit_sha256=3b6daa4a3d4c74ae384e7c3d2199d3d26f9360fe4b64a33e1c6ab16f4b83eabc
confidence_checkpoint_sha256=e31fde6f351284205c78f7a1510002779c43312e94d9f82003d47a14d72bc78f
s50_checkpoint_sha256=65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6
matched_checkpoint_sha256=6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db

smoke_ids=(
  '1a5s__1__1.A__1.C__1.C'
  '1a69__1__1.A_1.C__1.D_1.E__1.E'
  '1bjo__1__1.A_1.B__1.C_1.D__1.D'
  '1c29__1__1.A__1.C__1.C'
  '1c39__1__1.B__1.F_1.H__1.F'
  '1c9d__1__1.A__1.C__1.C'
  '1cw2__1__2.A__2.C__2.C'
  '1cx9__1__1.A__1.C__1.C'
)

die() {
  echo "$*" >&2
  exit 2
}

verify_sha256() {
  local label=$1
  local path=$2
  local expected=$3
  local actual
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] \
    || die "$label SHA-256 is not a frozen lowercase digest"
  [[ -f "$path" ]] || die "missing $label: $path"
  read -r actual _ < <(sha256sum "$path")
  [[ "$actual" == "$expected" ]] \
    || die "$label SHA-256 mismatch: expected=$expected actual=$actual path=$path"
}

[[ -x .venv/bin/python ]] || die "missing .venv/bin/python; run uv sync before submission"
command -v oms >/dev/null || die "missing oms research-runner"
command -v jq >/dev/null || die "missing jq"

verify_sha256 protocol "$protocol" "$protocol_sha256"
verify_sha256 scorer "$scorer" "$scorer_sha256"
verify_sha256 reporter "$reporter" "$reporter_sha256"
[[ "$runtime_code_sha256" =~ ^[0-9a-f]{64}$ ]] \
  || die "runtime-code aggregate SHA-256 is not a frozen lowercase digest"
verify_sha256 config "$config" "$config_sha256"
verify_sha256 confidence-checkpoint "$confidence_checkpoint" "$confidence_checkpoint_sha256"
verify_sha256 s50-backbone "$s50_checkpoint" "$s50_checkpoint_sha256"
verify_sha256 matched-backbone "$matched_checkpoint" "$matched_checkpoint_sha256"

stage=${EFFDOCK_STAGE:?EFFDOCK_STAGE must be smoke or full}
case "$stage" in
  smoke|full) ;;
  *) die "EFFDOCK_STAGE must be smoke or full; got $stage" ;;
esac

task_id=${SLURM_ARRAY_TASK_ID:?submit as the exact 8-task Slurm array}
task_count=${SLURM_ARRAY_TASK_COUNT:?SLURM_ARRAY_TASK_COUNT is required}
array_job_id=${SLURM_ARRAY_JOB_ID:?SLURM_ARRAY_JOB_ID is required}
[[ "$task_count" -eq 8 && "$task_id" -ge 0 && "$task_id" -lt 8 ]] \
  || die "$stage requires sbatch --array=0-7%4"

[[ ${SLURM_JOB_PARTITION:-} == 6000ada ]] \
  || die "scoring requires partition 6000ada"
gpu_inventory=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
[[ -n "$gpu_inventory" && $(wc -l <<< "$gpu_inventory") -eq 1 ]] \
  || die "scoring requires exactly one visible CUDA GPU"
IFS=, read -r gpu_name gpu_memory_mib <<< "$gpu_inventory"
gpu_memory_mib=${gpu_memory_mib//[[:space:]]/}
[[ "$gpu_name" == *"RTX 6000 Ada"* ]] \
  || die "scoring requires RTX 6000 Ada; got $gpu_name"
[[ "$gpu_memory_mib" =~ ^[0-9]+$ && "$gpu_memory_mib" -ge 45000 ]] \
  || die "scoring requires at least 45000 MiB visible GPU memory"

bank_manifest=${EFFDOCK_BANK_MANIFEST:?EFFDOCK_BANK_MANIFEST is required}
bank_manifest_sha256=${EFFDOCK_BANK_MANIFEST_SHA256:?EFFDOCK_BANK_MANIFEST_SHA256 is required}
[[ "$bank_manifest_sha256" =~ ^[0-9a-f]{64}$ ]] \
  || die "EFFDOCK_BANK_MANIFEST_SHA256 must be a lowercase SHA-256"
[[ "$bank_manifest" == /* ]] \
  || die "EFFDOCK_BANK_MANIFEST must be an absolute lexical-canonical path"
bank_manifest_canonical=$(readlink -f -- "$bank_manifest")
[[ "$bank_manifest_canonical" == "$bank_manifest" ]] \
  || die "EFFDOCK_BANK_MANIFEST must contain no symlink or non-canonical component"
verify_sha256 label-free-bank-manifest "$bank_manifest" "$bank_manifest_sha256"
jq -e \
  --arg protocol_id "$protocol_id" \
  --arg eligibility_sha256 "$eligibility_manifest_sha256" \
  --arg source_protocol_sha256 "$source_sampler_protocol_sha256" \
  --arg source_report_sha256 "$source_sampler_report_sha256" \
  --arg source_audit_sha256 "$source_coordinate_audit_sha256" \
  '.schema_version == "effdock.early_time_sampler_s50_confidence_bank.v1"
   and .protocol_id == $protocol_id
   and .status == "complete_label_free"
   and .information_boundary.outcome_columns_exported == false
   and .information_boundary.score_stage_reads_source_results_csv == false
   and .information_boundary.crystal_reference_exported == false
   and .inputs.eligibility_manifest.sha256 == $eligibility_sha256
   and .inputs.source_sampler_protocol.sha256 == $source_protocol_sha256
   and .inputs.source_sampler_report.sha256 == $source_report_sha256
   and .inputs.source_coordinate_audit.sha256 == $source_audit_sha256' \
  "$bank_manifest" >/dev/null \
  || die "label-free bank manifest schema/protocol mismatch"

output_prefix=$repo_dir/outputs/benchmarks/early_time_sampler_s50_confidence_runs
mkdir -p "$output_prefix"
read -r content_sha256 _ < <(
  printf '%s\0' "$protocol_id" "$protocol_sha256" "$scorer_sha256" \
    "$reporter_sha256" "$runtime_code_sha256" "$bank_manifest_sha256" | sha256sum
)
run_root="$output_prefix/$content_sha256"
if [[ -n ${EFFDOCK_EXPECTED_RUN_ROOT:-} \
  && ${EFFDOCK_EXPECTED_RUN_ROOT} != "$run_root" ]]; then
  die "content-addressed root mismatch: expected=$run_root supplied=$EFFDOCK_EXPECTED_RUN_ROOT"
fi
mkdir -p "$run_root"

# Bind this content root to an exact immutable identity. Parallel tasks may
# race to create the marker but must agree byte-for-byte.
identity_file="$run_root/CONTENT_IDENTITY.sha256"
identity_tmp="$run_root/.CONTENT_IDENTITY.${array_job_id}.${task_id}.tmp"
printf '%s\n' "$content_sha256" > "$identity_tmp"
if ! ln "$identity_tmp" "$identity_file" 2>/dev/null; then
  cmp -s "$identity_tmp" "$identity_file" \
    || die "content identity mismatch at $run_root"
fi
rm -f "$identity_tmp"

if [[ "$stage" == full ]]; then
  smoke_report=${EFFDOCK_SMOKE_REPORT:?EFFDOCK_SMOKE_REPORT is required for full}
  smoke_report_sha256=${EFFDOCK_SMOKE_REPORT_SHA256:?EFFDOCK_SMOKE_REPORT_SHA256 is required for full}
  [[ "$smoke_report" == /* ]] \
    || die "EFFDOCK_SMOKE_REPORT must be an absolute lexical-canonical path"
  smoke_report_canonical=$(readlink -f -- "$smoke_report")
  [[ "$smoke_report_canonical" == "$smoke_report" ]] \
    || die "EFFDOCK_SMOKE_REPORT must contain no symlink or non-canonical component"
  verify_sha256 smoke-integrity-report "$smoke_report" "$smoke_report_sha256"
  jq -e \
    --arg protocol_id "$protocol_id" \
    --arg protocol_sha256 "$protocol_sha256" \
    --arg bank_sha256 "$bank_manifest_sha256" \
    --arg scorer_sha256 "$scorer_sha256" \
    '.schema == "effdock.early_time_sampler_plinder_confidence_report.v1"
     and .protocol_id == $protocol_id
     and .stage == "smoke"
     and .status == "passed"
     and .integrity_only == true
     and .expected_shards == 8
     and .inputs.protocol.sha256 == $protocol_sha256
     and .inputs.label_free_bank_manifest.sha256 == $bank_sha256
     and .inputs.scorer_source_sha256 == $scorer_sha256
     and .configuration.arms == ["s50_backbone", "matched_backbone"]
     and .integrity.score_shard_count == 16
     and .integrity.scored_complexes_per_arm == 8
     and .integrity.scores_per_complex == 100
     and .integrity.smoke_replay_max_abs_tolerance == 0.00001
     and .integrity.efficacy_emitted == false' "$smoke_report" >/dev/null \
    || die "full scoring requires a passed outcome-blind smoke report"
fi

reservation_root="$run_root/.task_reservations"
mkdir -p "$reservation_root"
array_marker="$reservation_root/${stage}.array_job_id"
marker_tmp="$reservation_root/.${stage}.${array_job_id}.${task_id}.tmp"
printf '%s\n' "$array_job_id" > "$marker_tmp"
if ! ln "$marker_tmp" "$array_marker" 2>/dev/null; then
  existing_array_job_id=
  if [[ -f "$array_marker" ]]; then
    read -r existing_array_job_id < "$array_marker" || true
  fi
  [[ "$existing_array_job_id" == "$array_job_id" ]] \
    || die "refusing to mix $stage tasks from different Slurm arrays"
fi
rm -f "$marker_tmp"
task_reservation="$reservation_root/${stage}.task-${task_id}"
mkdir "$task_reservation" \
  || die "refusing to rerun or overwrite reserved task $stage/$task_id"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS

printf 'protocol_id=%s protocol_sha256=%s manifest_sha256=%s\n' \
  "$protocol_id" "$protocol_sha256" "$bank_manifest_sha256"
printf 'stage=%s shard=%03d/008 output_root=%s\n' \
  "$stage" "$task_id" "$run_root"
printf 'gpu=%s memory_mib=%s partition=%s git_commit=%s\n' \
  "$gpu_name" "$gpu_memory_mib" "${SLURM_JOB_PARTITION}" "$(git rev-parse HEAD)"

for arm in s50_backbone matched_backbone; do
  case "$arm" in
    s50_backbone)
      backbone=$s50_checkpoint
      backbone_sha256=$s50_checkpoint_sha256
      arm_role='primary deployment backbone'
      ;;
    matched_backbone)
      backbone=$matched_checkpoint
      backbone_sha256=$matched_checkpoint_sha256
      arm_role='diagnostic confidence-matched backbone'
      ;;
    *) die "unreachable arm: $arm" ;;
  esac

  arm_dir=$(printf '%s/%s/shard-%03d-of-008/arms/%s' \
    "$run_root" "$stage" "$task_id" "$arm")
  mkdir -p "$(dirname "$arm_dir")"
  mkdir "$arm_dir" || die "refusing to overwrite arm output: $arm_dir"
  ledger="$run_root/ledgers/${stage}-shard-$(printf '%03d' "$task_id")-${arm}.jsonl"
  mkdir -p "$(dirname "$ledger")"

  score_command=(
    .venv/bin/python "$scorer" score-shard
    --bank-manifest "$bank_manifest"
    --expected-bank-manifest-sha256 "$bank_manifest_sha256"
    --docking-checkpoint "$backbone"
    --expected-docking-checkpoint-sha256 "$backbone_sha256"
    --confidence-checkpoint "$confidence_checkpoint"
    --expected-confidence-checkpoint-sha256 "$confidence_checkpoint_sha256"
    --expected-protocol-sha256 "$protocol_sha256"
    --expected-scorer-source-sha256 "$scorer_sha256"
    --expected-report-source-sha256 "$reporter_sha256"
    --expected-runtime-code-sha256 "$runtime_code_sha256"
    --scores-root "$run_root"
    --stage "$stage"
    --arm "$arm"
    --shard-index "$task_id"
    --num-shards 8
    --pose-batch-size 20
    --sigma 2.0
    --device cuda
  )
  if [[ "$stage" == smoke ]]; then
    score_command+=(--smoke-replay-id "${smoke_ids[$task_id]}")
  fi

  oms research-runner \
    --question "Can retained confidence rank the immutable S50 PLINDER pose bank under $arm?" \
    --hypothesis "The frozen confidence heads remain numerically usable under the declared score-only deployment shift." \
    --prediction "Smoke replay is exact within 1e-5; full outputs are finite and complete before labels are joined." \
    --baseline "immutable S50 N100 sigma-2 bank; matched-backbone is diagnostic-only" \
    --metric "registered Top1 SR2 after sealed score-to-label join; $stage itself is label-blind" \
    --success "zero hash/count/nonfinite failures; exact replay index and max score delta <=1e-5 in smoke" \
    --change "$arm_role; candidates, confidence checkpoint, sigma, and chunking fixed" \
    --file "$ledger" \
    --no-gate \
    --reason "launcher verifies frozen protocol, code, model, bank, and provenance hashes before each task" \
    -- \
    "${score_command[@]}"
done
