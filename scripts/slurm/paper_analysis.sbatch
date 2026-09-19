#!/bin/bash
#SBATCH --job-name=effdock-paper-analysis
#SBATCH --partition=cpu_only
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=logs/paper-analysis-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=.:src
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR=/tmp/effdock-paper-mpl
.venv/bin/python scripts/paper_results.py
.venv/bin/python scripts/collect_paper_candidate_metrics.py
.venv/bin/python scripts/paper_candidate_figures.py
.venv/bin/python scripts/collect_paper_runtime.py
.venv/bin/python scripts/paper_runtime_figures.py
.venv/bin/python scripts/collect_paper_overlap.py
.venv/bin/python scripts/paper_overlap_figures.py
