#!/usr/bin/env bash
#SBATCH --job-name=effdock-s50c-ext-report
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
: "${EFFDOCK_EXECUTION_ROOT:?}"
: "${EFFDOCK_OUTPUT_ROOT:?}"
cd "$EFFDOCK_EXECUTION_ROOT"
export PYTHONPATH="$EFFDOCK_EXECUTION_ROOT/src:$EFFDOCK_EXECUTION_ROOT"
export PYTHONDONTWRITEBYTECODE=1
read -r actual _ < <(sha256sum scripts/report_s50_symmetry_confidence_refined_external.py)
[[ "$actual" == dd5c41b6c2919898ab2accdc471440860eb712e444031ab3162b1fbf3ef74fb0 ]] || exit 2
.venv/bin/python scripts/report_s50_symmetry_confidence_refined_external.py \
  --scores-root "$EFFDOCK_OUTPUT_ROOT/full" \
  --official-root "$EFFDOCK_EXECUTION_ROOT/outputs/benchmarks/guidance_sdf_post_refinement_runs/sigma2-eta2-adaptive-20260819T062833Z/full/posebusters_step100" \
  --output "$EFFDOCK_OUTPUT_ROOT/report.json"
