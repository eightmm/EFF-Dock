#!/usr/bin/env bash
#SBATCH --job-name=effdock-raw-energy-select
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err

set -euo pipefail
repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
cd "${repo_dir:?EFFDOCK_REPO_DIR or SLURM_SUBMIT_DIR is required}"
: "${EFFDOCK_INPUT_ROOT:?set EFFDOCK_INPUT_ROOT}"
: "${EFFDOCK_REPORT_OUTPUT_DIR:?set EFFDOCK_REPORT_OUTPUT_DIR}"
.venv/bin/python scripts/report_guidance_energy_confidence_raw_selection.py \
  --manifest outputs/benchmarks/guidance_all_pose_pb_eta_runs/20260811T042451Z/manifest.json \
  --input-root "$EFFDOCK_INPUT_ROOT" \
  --before-pb-root outputs/benchmarks/guidance_all_pose_pb_eta_runs/20260811T042451Z/posebusters \
  --before-pb-audit outputs/benchmarks/guidance_all_pose_pb_eta_runs/20260811T042451Z/audit/full.json \
  --after-pb-root "$EFFDOCK_INPUT_ROOT/posebusters_step100" \
  --before-pb-shards 64 --after-pb-shards 32 \
  --protocol-file docs/GUIDANCE_ENERGY_CONFIDENCE_RAW_SELECTION_PROTOCOL.md \
  --output-dir "$EFFDOCK_REPORT_OUTPUT_DIR"
