#!/usr/bin/env bash
set -euo pipefail

: "${GNINA_REAL_PATH:?Set GNINA_REAL_PATH to the verified GNINA binary}"
exec "$GNINA_REAL_PATH" "$@" --no_gpu
