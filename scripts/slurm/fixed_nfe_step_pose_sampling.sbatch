#!/usr/bin/env bash
#SBATCH --job-name=effdock-nfe-sample
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a5000:1
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err

set -euo pipefail

die() { echo "$*" >&2; exit 2; }
: "${EFFDOCK_OUTPUT_ROOT:?}"
: "${EFFDOCK_RUN_MODE:?}"
: "${EFFDOCK_SAMPLING_CAPSULE:?}"
: "${EFFDOCK_COHORT_AUDIT:?}"

[[ "$EFFDOCK_RUN_MODE" == smoke || "$EFFDOCK_RUN_MODE" == full ]] || die "invalid run mode"
task_id=${SLURM_ARRAY_TASK_ID:?}
task_count=${SLURM_ARRAY_TASK_COUNT:?}
if [[ "$EFFDOCK_RUN_MODE" == smoke ]]; then
  [[ "$task_count" -eq 2 ]] || die "smoke sampling requires two tasks"
  dataset_index=$task_id
  shard_index=0
  num_shards=1
  case "$dataset_index" in
    0) dataset=astex; only_id=1jje ;;
    1) dataset=posebusters; only_id=7b2c_tp7 ;;
    *) die "invalid smoke task $task_id" ;;
  esac
else
  [[ "$task_count" -eq 16 ]] || die "full sampling requires 16 tasks"
  dataset_index=$((task_id / 8))
  shard_index=$((task_id % 8))
  num_shards=8
  only_id=
  case "$dataset_index" in
    0) dataset=astex ;;
    1) dataset=posebusters ;;
    *) die "invalid full task $task_id" ;;
  esac
fi

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
esac

cd "$EFFDOCK_SAMPLING_CAPSULE"
actual_capsule_sha=$(sha256sum execution_capsule.json | cut -d' ' -f1)
[[ "$actual_capsule_sha" == 62c698577a3b4ad407b9926ec922dae201fbce45e189e6aae2b83c4d4fe0cb35 ]] \
  || die "sampling capsule identity mismatch"
for required in \
  .venv/bin/eff-dock \
  weights/effdock_geometry_ft_100k_best.pt \
  weights/effdock_confidence_extmatch_n80_s25_step42500.pt \
  configs/train.yaml \
  docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json \
  "$centers" \
  "$EFFDOCK_COHORT_AUDIT"; do
  [[ -e "$required" ]] || die "missing frozen input: $required"
done

gpu_inventory=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
[[ $(wc -l <<< "$gpu_inventory") -eq 1 ]] || die "exactly one visible GPU is required"
IFS=, read -r gpu_name gpu_memory_mib <<< "$gpu_inventory"
gpu_memory_mib=${gpu_memory_mib//[[:space:]]/}
[[ "$gpu_memory_mib" =~ ^[0-9]+$ && "$gpu_memory_mib" -ge 23000 ]] \
  || die "at least 23,000 MiB GPU memory is required"
[[ "$gpu_name" == *"RTX A5000"* ]] \
  || die "paired sampling requires the preflighted RTX A5000 node family"

sampling_root="$EFFDOCK_OUTPUT_ROOT/$EFFDOCK_RUN_MODE/sampling"
mkdir -p "$sampling_root" "$EFFDOCK_OUTPUT_ROOT/ledgers/$EFFDOCK_RUN_MODE"
reservation="$EFFDOCK_OUTPUT_ROOT/.task_reservations/sampling-${EFFDOCK_RUN_MODE}-${task_id}"
mkdir -p "$EFFDOCK_OUTPUT_ROOT/.task_reservations"
mkdir "$reservation" || die "duplicate sampling task reservation: $reservation"

run_name="effdock-fixed-nfe-step-pose-v1-${dataset}-n40-s25-sigma2000"
only_id_args=()
[[ -z "$only_id" ]] || only_id_args=(--only-id "$only_id")
command=(
  .venv/bin/eff-dock evaluate
  --dataset "$dataset"
  --data-dir "$data_dir"
  --external-dir data/external_test
  --benchmark-input-manifest docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json
  --require-full-ligand-atom-mapping
  --pocket-centers "$centers"
  --checkpoint weights/effdock_geometry_ft_100k_best.pt
  --confidence-checkpoint weights/effdock_confidence_extmatch_n80_s25_step42500.pt
  --selector-profile confidence_cluster_free
  --config configs/train.yaml
  --device cuda
  --output-dir "$sampling_root"
  --protocol-id EFFDOCK-FIXED-NFE-STEP-POSE-V1
  --run-name "$run_name"
  --num-samples 40
  --num-steps 25
  --prior-pool-size 100
  --sigma 2.0
  --time-schedule late
  --schedule-power 3.0
  --pocket-cutoff 10.0
  --center-jitter-sigma 0.0
  --refine none
  --unified-guidance-mode normalized_drift
  --unified-guidance-scale 2.0
  --unified-guidance-receptor-policy geometry_only
  --unified-guidance-start-t 0.5
  --unified-guidance-ramp-power 1.0
  --unified-guidance-max-force 20.0
  --unified-guidance-max-velocity 5.0
  --unified-guidance-max-angular-velocity 5.0
  --unified-guidance-max-atom-displacement 0.25
  --unified-guidance-max-backtracks 8
  --unified-guidance-protein-shell 18.0
  --seed 42
  --num-shards "$num_shards"
  --shard-index "$shard_index"
  --expected-discovered-count "$expected_count"
  --require-complete-success
  --save-selected-poses
  --eligibility-manifest "$EFFDOCK_COHORT_AUDIT"
  "${only_id_args[@]}"
)
export PYTHONPATH="$EFFDOCK_SAMPLING_CAPSULE/src" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
task_cpus=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS="$task_cpus" MKL_NUM_THREADS="$task_cpus" OPENBLAS_NUM_THREADS="$task_cpus"
oms run-ledger \
  --file "$EFFDOCK_OUTPUT_ROOT/ledgers/$EFFDOCK_RUN_MODE/sampling-${task_id}.jsonl" \
  --note "paired S25/N40 sigma2 eta2 sampling" \
  --no-gate --reason "user-authorized repeated-use descriptive fixed-NFE comparison" \
  -- "${command[@]}"
