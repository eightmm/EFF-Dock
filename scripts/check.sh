#!/usr/bin/env bash
# Project verification contract. Agents run this before claiming work done.
#   fast      CPU-only, under 60 seconds, safe to run anytime.
#   ml-smoke  ML interface smoke: import/config/data/model/loss one-batch checks.
#   gpu       Short GPU smoke; wrapped in a transient srun on Slurm machines.
# Fill the TODO blocks as the project takes shape. An empty contract fails
# loudly on purpose -- never let "no checks" look like a pass.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE="${1:-fast}"

run_fast() {
  local ran=0
  local dirs=()
  [ -d src ] && dirs+=(src)
  [ -d scripts ] && dirs+=(scripts)

  if [ "${#dirs[@]}" -gt 0 ] && [ -f pyproject.toml ] && command -v uv >/dev/null 2>&1; then
    uv run python -m compileall -q "${dirs[@]}"
    ran=1
  fi
  if [ -f pyproject.toml ] && command -v uv >/dev/null 2>&1 &&
    uv run ruff --version >/dev/null 2>&1; then
    uv run ruff check .
    ran=1
  fi

  uv run python -c "import effdock; from effdock.cli import main"
  ran=1

  if [ "$ran" -eq 0 ]; then
    echo "check fast: no checks ran; configure scripts/check.sh" >&2
    exit 1
  fi
  echo "check fast: ok"
}

run_ml_smoke() {
  local ran=0

  if [ -f scripts/ml_smoke.py ] && [ -f pyproject.toml ] && command -v uv >/dev/null 2>&1; then
    uv run python scripts/ml_smoke.py
    ran=1
  elif [ -f scripts/ml_smoke.py ]; then
    python3 scripts/ml_smoke.py
    ran=1
  fi

  if [ "$ran" -eq 0 ]; then
    echo "check ml-smoke: no ML smoke configured; add scripts/ml_smoke.py or edit scripts/check.sh" >&2
    exit 1
  fi
  echo "check ml-smoke: ok"
}

run_gpu() {
  if command -v srun >/dev/null 2>&1; then
    srun --gres=gpu:1 --time=00:10:00 uv run pytest -q -m gpu
  else
    uv run pytest -q -m gpu
  fi
  echo "check gpu: ok"
}

case "$MODE" in
  fast) run_fast ;;
  ml-smoke) run_ml_smoke ;;
  gpu) run_gpu ;;
  *)
    echo "usage: scripts/check.sh [fast|ml-smoke|gpu]" >&2
    exit 2
    ;;
esac
