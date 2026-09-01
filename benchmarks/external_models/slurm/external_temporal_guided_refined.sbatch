#!/usr/bin/env bash
#SBATCH --job-name=effdock-ext-v1
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

: "${EFFDOCK_EXECUTION_ROOT:?}"
: "${EFFDOCK_OUTPUT_ROOT:?}"
: "${EFFDOCK_STAGE:?}"
cd "$EFFDOCK_EXECUTION_ROOT"
mkdir -p outputs/slurm "$EFFDOCK_OUTPUT_ROOT/ledgers"
export PYTHONPATH="$EFFDOCK_EXECUTION_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

die() { echo "$*" >&2; exit 2; }
verify_sha() {
  local path=$1 expected=$2 actual
  [[ -f "$path" && ! -L "$path" ]] || die "missing regular file: $path"
  read -r actual _ < <(sha256sum "$path")
  [[ "$actual" == "$expected" ]] || die "SHA mismatch: $path"
}

verify_sha configs/train.yaml 39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec
verify_sha docs/EXTERNAL_TEMPORAL_GUIDED_REFINED_PROTOCOL.md "$EFFDOCK_PROTOCOL_SHA256"
verify_sha data/external_benchmarks/manifests/phibench.json 2697ecc14a83646a26aac319193f7ad98c202349836fda3bcac4e533f1a10633
verify_sha data/external_benchmarks/manifests/foldbench.json 7f6a77670d28103afc5eb08509a946b35d2b29cf5b17223e7832ff83fd5cb845
verify_sha data/external_benchmarks/manifests/openbind.json f5f8424698fc30970676c52d4e9d4f1b725e8127e540697d25a2d2822982b81d
verify_sha data/external_test/phibench_smiles.json 2ed3c80c1c0c4736314a9149a3fc8933ef33b6077b45467ac8f03fca4a098e37
verify_sha data/external_test/foldbench_smiles.json fce9550a5739649d0236fd6bc5a95fcdc492470767afd794ab3163425ac90989
verify_sha data/external_test/openbind_smiles.json 0fbda14dcaa25ff2f48ecb7a923d34e9e2dae0b609c31241d8e20829af9ab194
verify_sha data/external_test/phibench_reference_pocket_centers.json 0ad64e2da8fb94cce64dd9e245b215bd30cdddd18ae709da2c1bb423ebda1ecd
verify_sha data/external_test/foldbench_reference_pocket_centers.json bad7b200e75cb41b56945de3d4ee309432136a5f08877a5a0199d63a7c165d77
verify_sha data/external_test/openbind_reference_pocket_centers.json ce0102b6126966a78338abc72502be66704790d235c9be6dd49d029c470d04f4

docking=outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step50000_ema_common_init.pt
confidence=outputs/eff-dock/s50-confidence-symmetry-training/88509b2189ab619c91feee843e4d90388b85b36b181532ca0a54ebf271f69b97/full/best.pt
verify_sha "$docking" 65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6
verify_sha "$confidence" 1c59034172fb925cc8a70777dcba236be349f1a1de1775d49cc17d492b17c030

gpu_inventory=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
[[ $(wc -l <<< "$gpu_inventory") -eq 1 ]] || die "exactly one visible GPU is required"
IFS=, read -r gpu_name gpu_memory_mib <<< "$gpu_inventory"
gpu_memory_mib=${gpu_memory_mib//[[:space:]]/}
[[ "$gpu_memory_mib" =~ ^[0-9]+$ && "$gpu_memory_mib" -ge 48000 ]] \
  || die "at least 48000 MiB GPU memory is required"

task_id=${SLURM_ARRAY_TASK_ID:?}
task_count=${SLURM_ARRAY_TASK_COUNT:?}
only_id_args=()
case "$EFFDOCK_STAGE" in
  smoke)
    [[ "$task_count" -eq 3 ]] || die "smoke requires three tasks"
    num_shards=1
    shard_index=0
    case "$task_id" in
      0) dataset=phibench; expected_count=203; only_id=7hg0_a1a3p_b_1 ;;
      1) dataset=foldbench; expected_count=66; only_id=8bmv-assembly1 ;;
      2) dataset=openbind; expected_count=860; only_id=a71ev2a-x0188a ;;
      *) die "invalid smoke task: $task_id" ;;
    esac
    only_id_args=(--only-id "$only_id")
    ;;
  full)
    [[ "$task_count" -eq 72 ]] || die "full run requires 72 tasks"
    if (( task_id < 13 )); then
      dataset=phibench; expected_count=203; num_shards=13; shard_index=$task_id
    elif (( task_id < 18 )); then
      dataset=foldbench; expected_count=66; num_shards=5; shard_index=$((task_id - 13))
    else
      dataset=openbind; expected_count=860; num_shards=54; shard_index=$((task_id - 18))
    fi
    ;;
  *) die "invalid stage: $EFFDOCK_STAGE" ;;
esac

data_dir="data/external_benchmarks/normalized/$dataset"
external_dir=data/external_test
centers="$external_dir/${dataset}_reference_pocket_centers.json"
dataset_manifest="data/external_benchmarks/manifests/$dataset.json"
command=(
  .venv/bin/python scripts/run_external_temporal_benchmark_shard.py
  --dataset "$dataset"
  --data-dir "$data_dir"
  --external-dir "$external_dir"
  --pocket-centers "$centers"
  --dataset-manifest "$dataset_manifest"
  --expected-count "$expected_count"
  --num-shards "$num_shards"
  --shard-index "$shard_index"
  --stage "$EFFDOCK_STAGE"
  --output-root "$EFFDOCK_OUTPUT_ROOT"
  --protocol-file docs/EXTERNAL_TEMPORAL_GUIDED_REFINED_PROTOCOL.md
  --config configs/train.yaml
  --docking-checkpoint "$docking"
  --confidence-checkpoint "$confidence"
  "${only_id_args[@]}"
)

echo "stage=$EFFDOCK_STAGE dataset=$dataset shard=$shard_index/$num_shards gpu=$gpu_name"
oms run-ledger \
  --file "$EFFDOCK_OUTPUT_ROOT/ledgers/${EFFDOCK_STAGE}-${dataset}-shard${shard_index}.jsonl" \
  --note "external temporal N100/S10 sigma2 eta2 guided and adaptively refined shard" \
  --no-gate --reason "immutable execution capsule passed submission-time focused gates" \
  -- "${command[@]}"
