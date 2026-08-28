#!/usr/bin/env bash
set -euo pipefail

native_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export LD_LIBRARY_PATH="$native_root/obrms-lib:${LD_LIBRARY_PATH:-}"
export BABEL_DATADIR="$native_root/share/openbabel/3.1.0"
exec "$native_root/obrms.real" "$@"
