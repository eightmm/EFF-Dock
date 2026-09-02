#!/usr/bin/env bash
#SBATCH --job-name=effdock-fb558
#SBATCH --partition=6000ada,heavy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err

set -euo pipefail

die() { echo "$*" >&2; exit 2; }
verify_sha() {
  local path=$1 expected=$2 actual
  [[ -f "$path" && ! -L "$path" ]] || die "missing regular file: $path"
  read -r actual _ < <(sha256sum "$path")
  [[ "$actual" == "$expected" ]] || die "SHA mismatch: $path"
}

: "${EFFDOCK_EXECUTION_ROOT:?}"
: "${EFFDOCK_OUTPUT_ROOT:?}"
: "${EFFDOCK_STAGE:?}"
: "${EFFDOCK_PROTOCOL_SHA:?}"
: "${EFFDOCK_RUNNER_SHA:?}"
: "${EFFDOCK_REFINER_SHA:?}"
: "${EFFDOCK_SCORER_SHA:?}"
: "${EFFDOCK_SRC_SHA:?}"
: "${EFFDOCK_MANIFEST_SHA:?}"
: "${EFFDOCK_MAPPING_SHA:?}"
: "${EFFDOCK_CENTERS_SHA:?}"
cd "$EFFDOCK_EXECUTION_ROOT"
mkdir -p outputs/slurm "$EFFDOCK_OUTPUT_ROOT/ledgers"
export PYTHONPATH="$EFFDOCK_EXECUTION_ROOT/src:$EFFDOCK_EXECUTION_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

protocol=docs/FOLDBENCH_FULL_POCKET_PROTOCOL.md
runner=scripts/run_external_temporal_benchmark_shard.py
refiner=scripts/run_guidance_sdf_post_refinement.py
scorer=scripts/score_guidance_sdf_post_refinement_confidence.py
manifest=data/external_benchmarks/foldbench_full/manifests/foldbench.json
mapping=data/external_benchmarks/foldbench_full/external/foldbench_smiles.json
centers=data/external_benchmarks/foldbench_full/external/foldbench_reference_pocket_centers.json
docking=outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step50000_ema_common_init.pt
confidence=outputs/eff-dock/s50-raw-refined-confidence-runs/309bb1a7645f8d09a796e45559d018c969d077a937ecb225ea53314a7143c804/training-100k-runs/8641cbe7b5bc99896c6513073ca7d81d4c00db42015a3739685043e5e4fa162f/full/best.pt

verify_sha "$protocol" "$EFFDOCK_PROTOCOL_SHA"
verify_sha "$runner" "$EFFDOCK_RUNNER_SHA"
verify_sha "$refiner" "$EFFDOCK_REFINER_SHA"
verify_sha "$scorer" "$EFFDOCK_SCORER_SHA"
verify_sha "$manifest" "$EFFDOCK_MANIFEST_SHA"
verify_sha "$mapping" "$EFFDOCK_MAPPING_SHA"
verify_sha "$centers" "$EFFDOCK_CENTERS_SHA"
verify_sha configs/train.yaml 39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec
verify_sha "$docking" 65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6
verify_sha "$confidence" ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638
actual_src_sha=$(find src/effdock -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
[[ "$actual_src_sha" == "$EFFDOCK_SRC_SHA" ]] || die "source tree SHA mismatch"

gpu_inventory=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
[[ $(wc -l <<< "$gpu_inventory") -eq 1 ]] || die "exactly one visible GPU is required"
IFS=, read -r gpu_name gpu_memory_mib <<< "$gpu_inventory"
gpu_memory_mib=${gpu_memory_mib//[[:space:]]/}
[[ "$gpu_memory_mib" =~ ^[0-9]+$ && "$gpu_memory_mib" -ge 48000 ]] \
  || die "at least 48000 MiB GPU memory is required"

task_id=${SLURM_ARRAY_TASK_ID:?}
only_id_args=()
case "$EFFDOCK_STAGE" in
  smoke)
    [[ "${SLURM_ARRAY_TASK_COUNT:?}" -eq 1 && "$task_id" -eq 0 ]] \
      || die "smoke requires one task"
    num_shards=1
    shard_index=0
    only_id_args=(--only-id 8bmv-assembly1__protein-a__ligand-c__ccd-urc)
    ;;
  full)
    if [[ -n "${EFFDOCK_RECOVERY_SHARDS:-}" ]]; then
      IFS=: read -r -a recovery_shards <<< "$EFFDOCK_RECOVERY_SHARDS"
      [[ "${SLURM_ARRAY_TASK_COUNT:?}" -eq "${#recovery_shards[@]}" ]] \
        || die "recovery array size does not match EFFDOCK_RECOVERY_SHARDS"
      [[ "$task_id" -lt "${#recovery_shards[@]}" ]] || die "invalid recovery task"
      shard_index=${recovery_shards[$task_id]}
    else
      [[ "${SLURM_ARRAY_TASK_COUNT:?}" -eq 44 ]] || die "full requires 44 tasks"
      shard_index=$task_id
    fi
    num_shards=44
    ;;
  *) die "invalid stage: $EFFDOCK_STAGE" ;;
esac

command=(
  .venv/bin/python "$runner"
  --dataset foldbench
  --data-dir data/external_benchmarks/foldbench_full/normalized/foldbench
  --external-dir data/external_benchmarks/foldbench_full/external
  --pocket-centers "$centers"
  --dataset-manifest "$manifest"
  --expected-count 558
  --num-shards "$num_shards"
  --shard-index "$shard_index"
  --stage "$EFFDOCK_STAGE"
  --output-root "$EFFDOCK_OUTPUT_ROOT"
  --protocol-file "$protocol"
  --config configs/train.yaml
  --docking-checkpoint "$docking"
  --confidence-checkpoint "$confidence"
  --protocol-id EFFDOCK-FOLDBENCH-POCKET-558-V1
  --run-name-prefix effdock-foldbench-pocket-558-v1
  --sampling-profile unguided
  "${only_id_args[@]}"
)
echo "stage=$EFFDOCK_STAGE shard=$shard_index/$num_shards gpu=$gpu_name"
oms run-ledger \
  --file "$EFFDOCK_OUTPUT_ROOT/ledgers/${EFFDOCK_STAGE}-foldbench-shard${shard_index}.jsonl" \
  --note "FoldBench-Pocket-558 U70k N100/S10/sigma2 raw+refined shard" \
  --no-gate --reason "frozen full-cohort descriptive protocol" \
  -- "${command[@]}"
