#!/usr/bin/env bash
#SBATCH --job-name=effdock-guidance-full-b1000
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err
#SBATCH --partition=6000ada
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-47%8

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"
mkdir -p outputs/benchmarks/logs

protocol_id=EFFDOCK-UNIFIED-GUIDANCE-BUDGET1000-FULL-V2
checkpoint=weights/effdock_geometry_ft_100k_best.pt
config=configs/train.yaml
benchmark_input_manifest=docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json
benchmark_input_manifest_sha256=99f15f557644cc51c3dd1f559b0dd97dd4259c1de3e1403fb761b7c7e079f668
output_root=${EFFDOCK_OUTPUT_ROOT:-outputs/benchmarks/guidance_budget1000_full_v2/raw}
combined_audit=${EFFDOCK_COHORT_MANIFEST:-docs/GUIDANCE_BUDGET1000_FULL_COHORT.json}

task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
dataset_index=$((task_id / 24))
within_dataset=$((task_id % 24))
condition_index=$((within_dataset / 8))
shard_index=$((within_dataset % 8))
num_shards=8

steps=(10 20 25)
poses=(100 50 40)
num_steps=${steps[$condition_index]}
num_samples=${poses[$condition_index]}

case "$dataset_index" in
  0)
    dataset=astex
    data_dir=data/external_benchmarks/data/astex_diverse_set
    centers=data/external_test/astex_reference_pocket_centers.json
    dataset_audit=${EFFDOCK_ASTEX_COHORT_MANIFEST:-docs/GUIDANCE_BUDGET1000_FULL_COHORT_ASTEX.json}
    ;;
  1)
    dataset=posebusters
    data_dir=data/external_benchmarks/data/posebusters_benchmark_set
    centers=data/external_test/posebusters_reference_pocket_centers.json
    dataset_audit=${EFFDOCK_POSEBUSTERS_COHORT_MANIFEST:-docs/GUIDANCE_BUDGET1000_FULL_COHORT_POSEBUSTERS.json}
    ;;
  *)
    echo "invalid dataset index: $dataset_index" >&2
    exit 2
    ;;
esac

if [[ ! -x .venv/bin/eff-dock ]]; then
  echo "missing executable .venv/bin/eff-dock; run uv sync first" >&2
  exit 2
fi
if [[ ! -f "$checkpoint" || ! -f "$config" || ! -f "$centers" || ! -f "$benchmark_input_manifest" ]]; then
  echo "missing checkpoint, config, pocket-center, or frozen benchmark-input manifest" >&2
  exit 2
fi
read -r actual_input_manifest_sha256 _ < <(sha256sum "$benchmark_input_manifest")
if [[ "$actual_input_manifest_sha256" != "$benchmark_input_manifest_sha256" ]]; then
  echo "frozen benchmark-input manifest SHA-256 mismatch" >&2
  echo "expected=$benchmark_input_manifest_sha256 actual=$actual_input_manifest_sha256" >&2
  exit 2
fi

cohort_audit=
if [[ -n "${EFFDOCK_ONLY_ID:-}" ]]; then
  : # GPU smoke is intentionally pre-audit and writes to a separate output root.
elif [[ -f "$combined_audit" ]]; then
  cohort_audit=$combined_audit
elif [[ -f "$dataset_audit" ]]; then
  cohort_audit=$dataset_audit
else
  echo "full run requires an audited full-cohort manifest" >&2
  echo "checked: $combined_audit and $dataset_audit" >&2
  exit 2
fi

audit_args=()
if [[ -n "$cohort_audit" ]]; then
  .venv/bin/python -c \
    'import sys; from pathlib import Path; from effdock.workflows.guidance_budget_full_report import validate_full_cohort_audit_for_dataset; validate_full_cohort_audit_for_dataset(Path(sys.argv[1]), sys.argv[2])' \
    "$cohort_audit" "$dataset"
  audit_args=(--eligibility-manifest "$cohort_audit")
fi

only_id_args=()
if [[ -n "${EFFDOCK_ONLY_ID:-}" ]]; then
  if [[ "${SLURM_ARRAY_TASK_COUNT:-0}" -ne 1 ]]; then
    echo "EFFDOCK_ONLY_ID smoke requires an explicit one-task array selection" >&2
    echo "example: EFFDOCK_ONLY_ID=<id> sbatch --array=<task-id> $0" >&2
    exit 2
  fi
  only_id_args=(--only-id "$EFFDOCK_ONLY_ID")
  num_shards=1
  shard_index=0
  output_root=${EFFDOCK_SMOKE_OUTPUT_ROOT:-outputs/benchmarks/guidance_budget1000_full_v2/smoke/$EFFDOCK_ONLY_ID}
fi

echo "protocol=$protocol_id dataset=$dataset cell=N${num_samples}/S${num_steps} shard=$shard_index/$num_shards"
echo "cohort_audit=${cohort_audit:-none-smoke-only} output_root=$output_root"
git rev-parse HEAD
git status --short
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

common_args=(
  --dataset "$dataset"
  --data-dir "$data_dir"
  --external-dir data/external_test
  --benchmark-input-manifest "$benchmark_input_manifest"
  --require-full-ligand-atom-mapping
  --pocket-centers "$centers"
  --checkpoint "$checkpoint"
  --config "$config"
  --no-confidence
  --output-dir "$output_root"
  --protocol-id "$protocol_id"
  --num-samples "$num_samples"
  --num-steps "$num_steps"
  --prior-pool-size 100
  --sigma 0.5
  --time-schedule late
  --schedule-power 3.0
  --pocket-cutoff 10.0
  --center-jitter-sigma 0.0
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
  --save-selected-poses
  "${audit_args[@]}"
)

for arm in unguided guided; do
  if [[ "$arm" == guided ]]; then
    guidance_scale=0.1
  else
    guidance_scale=0.0
  fi
  run_name="effdock-guidance-budget1000-full-v2-${dataset}-n${num_samples}-s${num_steps}-${arm}"
  echo "run=$run_name budget=$((num_samples * num_steps)) scale=$guidance_scale"
  .venv/bin/eff-dock evaluate \
    "${common_args[@]}" \
    --run-name "$run_name" \
    --unified-guidance-scale "$guidance_scale" \
    "${only_id_args[@]}"
done
