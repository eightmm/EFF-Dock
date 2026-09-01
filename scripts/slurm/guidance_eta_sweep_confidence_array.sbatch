#!/usr/bin/env bash
#SBATCH --job-name=effdock-eta-conf
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err
#SBATCH --partition=6000ada
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-127%8

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"
mkdir -p outputs/benchmarks/logs

code_manifest=${EFFDOCK_CONFIDENCE_CODE_MANIFEST:?EFFDOCK_CONFIDENCE_CODE_MANIFEST is required}
expected_code_manifest_sha256=${EFFDOCK_CONFIDENCE_CODE_MANIFEST_SHA256:?EFFDOCK_CONFIDENCE_CODE_MANIFEST_SHA256 is required}
read -r actual_code_manifest_sha256 _ < <(sha256sum "$code_manifest")
if [[ "$actual_code_manifest_sha256" != "$expected_code_manifest_sha256" ]] \
  || ! sha256sum --check --quiet "$code_manifest"; then
  echo "frozen confidence execution manifest verification failed" >&2
  exit 2
fi

protocol_id=EFFDOCK-UNIFIED-GUIDANCE-ETA-SWEEP-CONFIDENCE-PB-V1
checkpoint=weights/effdock_geometry_ft_100k_best.pt
confidence_checkpoint=weights/effdock_confidence_extmatch_n80_s25_step42500.pt
config=configs/train.yaml
benchmark_input_manifest=docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json
expected_checkpoint_sha256=6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db
expected_confidence_sha256=e31fde6f351284205c78f7a1510002779c43312e94d9f82003d47a14d72bc78f
expected_config_sha256=39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec
expected_input_sha256=99f15f557644cc51c3dd1f559b0dd97dd4259c1de3e1403fb761b7c7e079f668
expected_cohort_audit_sha256=dac7903488ccd36552a9bca134e37e633e3f07166d94f0389837012081ff3048
output_root=${EFFDOCK_OUTPUT_ROOT:?EFFDOCK_OUTPUT_ROOT is required}
cohort_audit=${EFFDOCK_COHORT_MANIFEST:-$output_root/audit/combined.json}
extension_root="$output_root/confidence_selector_replay"

eta_values=(0.0 0.025 0.05 0.1 0.2 0.3 0.4 0.5)
eta_tags=(eta0000 eta0025 eta0050 eta0100 eta0200 eta0300 eta0400 eta0500)
task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
task_count=${SLURM_ARRAY_TASK_COUNT:?SLURM_ARRAY_TASK_COUNT is required}
num_shards=8
only_id=""
raw_root="$extension_root/raw"

if [[ "${EFFDOCK_CONFIDENCE_SMOKE_GRID:-0}" == 1 ]]; then
  if [[ "$task_count" -ne 16 ]]; then
    echo "confidence smoke requires the exact 0-15 array" >&2
    exit 2
  fi
  dataset_index=$((task_id / 8))
  eta_index=$((task_id % 8))
  shard_index=0
  num_shards=1
  raw_root="$extension_root/smoke/raw"
  case "$dataset_index" in
    0) dataset=astex; only_id=1jje ;;
    1) dataset=posebusters; only_id=7b2c_tp7 ;;
    *) echo "invalid confidence-smoke task: $task_id" >&2; exit 2 ;;
  esac
else
  if [[ "$task_count" -ne 128 || "$task_id" -lt 0 || "$task_id" -ge 128 ]]; then
    echo "full confidence replay requires the exact 0-127 array" >&2
    exit 2
  fi
  shard_index=$((task_id % 8))
  eta_index=$(((task_id / 8) % 8))
  dataset_index=$((task_id / 64))
  case "$dataset_index" in
    0) dataset=astex ;;
    1) dataset=posebusters ;;
    *) echo "invalid dataset index: $dataset_index" >&2; exit 2 ;;
  esac
fi

eta=${eta_values[$eta_index]}
eta_tag=${eta_tags[$eta_index]}
case "$dataset" in
  astex)
    expected_count=85
    data_dir=data/external_benchmarks/data/astex_diverse_set
    centers=data/external_test/astex_reference_pocket_centers.json
    ;;
  posebusters)
    expected_count=308
    data_dir=data/external_benchmarks/data/posebusters_benchmark_set
    centers=data/external_test/posebusters_reference_pocket_centers.json
    ;;
  *) echo "dataset must be astex or posebusters, got $dataset" >&2; exit 2 ;;
esac

for required in .venv/bin/eff-dock "$checkpoint" "$confidence_checkpoint" "$config" \
  "$centers" "$benchmark_input_manifest" "$cohort_audit"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 2
  fi
done

verify_sha256() {
  local path=$1
  local expected=$2
  local actual
  read -r actual _ < <(sha256sum "$path")
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch for $path: expected=$expected actual=$actual" >&2
    exit 2
  fi
}
verify_sha256 "$checkpoint" "$expected_checkpoint_sha256"
verify_sha256 "$confidence_checkpoint" "$expected_confidence_sha256"
verify_sha256 "$config" "$expected_config_sha256"
verify_sha256 "$benchmark_input_manifest" "$expected_input_sha256"
verify_sha256 "$cohort_audit" "$expected_cohort_audit_sha256"

only_id_args=()
if [[ -n "$only_id" ]]; then
  only_id_args=(--only-id "$only_id")
fi
run_name="effdock-guidance-direct-drift-eta-sweep-v2-${dataset}-n100-s10-${eta_tag}"
echo "protocol=$protocol_id dataset=$dataset eta=$eta run=$run_name shard=$shard_index/$num_shards only_id=${only_id:-none}"
git rev-parse HEAD
git diff --no-ext-diff | sha256sum
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

.venv/bin/eff-dock evaluate \
  --dataset "$dataset" \
  --data-dir "$data_dir" \
  --external-dir data/external_test \
  --benchmark-input-manifest "$benchmark_input_manifest" \
  --require-full-ligand-atom-mapping \
  --pocket-centers "$centers" \
  --checkpoint "$checkpoint" \
  --confidence-checkpoint "$confidence_checkpoint" \
  --config "$config" \
  --device cuda \
  --output-dir "$raw_root" \
  --protocol-id "$protocol_id" \
  --run-name "$run_name" \
  --num-samples 100 \
  --num-steps 10 \
  --prior-pool-size 100 \
  --sigma 0.5 \
  --time-schedule late \
  --schedule-power 3.0 \
  --pocket-cutoff 10.0 \
  --center-jitter-sigma 0.0 \
  --refine none \
  --unified-guidance-mode normalized_drift \
  --unified-guidance-scale "$eta" \
  --unified-guidance-receptor-policy geometry_only \
  --unified-guidance-start-t 0.5 \
  --unified-guidance-ramp-power 1.0 \
  --unified-guidance-max-force 20.0 \
  --unified-guidance-max-velocity 5.0 \
  --unified-guidance-max-angular-velocity 5.0 \
  --unified-guidance-max-atom-displacement 0.25 \
  --unified-guidance-max-backtracks 8 \
  --unified-guidance-protein-shell 18.0 \
  --seed 42 \
  --num-shards "$num_shards" \
  --shard-index "$shard_index" \
  --expected-discovered-count "$expected_count" \
  --require-complete-success \
  --save-selected-poses \
  --eligibility-manifest "$cohort_audit" \
  "${only_id_args[@]}"
