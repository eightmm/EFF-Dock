#!/usr/bin/env bash
set -euo pipefail

model=${1:-}
if [[ -z "$model" || $# -lt 2 ]]; then
  echo "usage: $0 <model> <command> [args...]" >&2
  exit 2
fi
shift

repo_root=${EFFDOCK_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
model_root="$repo_root/others/$model"
upstream="$model_root/upstream"

[[ -f "$model_root/pyproject.toml" ]] || { echo "unknown model: $model" >&2; exit 2; }
[[ -x "$model_root/.venv/bin/python" ]] || {
  echo "model environment is not synchronized: $model_root/.venv" >&2
  exit 2
}
[[ -d "$upstream" ]] || { echo "missing upstream checkout: $upstream" >&2; exit 2; }

export UV_CACHE_DIR="$model_root/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$model_root/.venv"
export TORCH_HOME="$model_root/.cache/torch"
export HF_HOME="$model_root/.cache/huggingface"
export MPLCONFIGDIR="$model_root/.cache/matplotlib"
export PATH="$model_root/bin:$model_root/.venv/bin:$PATH"
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}

case "$model" in
  sigmadock)
    export PYTHONPATH="$upstream/src:$upstream:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
    sigmadock_nvidia_root=$($model_root/.venv/bin/python -c \
      'import pathlib, sysconfig; print(pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia")')
    sigmadock_cudnn_lib="$sigmadock_nvidia_root/cudnn/lib"
    [[ -f "$sigmadock_cudnn_lib/libcudnn.so.9" ]] || {
      echo "missing SigmaDock cuDNN runtime: $sigmadock_cudnn_lib/libcudnn.so.9" >&2
      exit 2
    }
    sigmadock_nvidia_libs=$($model_root/.venv/bin/python -c \
      'import pathlib, sysconfig; root = pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia"; print(":".join(str(p) for p in sorted(root.glob("*/lib")) if p.is_dir()))')
    export LD_LIBRARY_PATH="$sigmadock_nvidia_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    ;;
  surfdock|diffbindfr)
    export PYTHONPATH="$upstream:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
    ;;
  interformer)
    torch_lib=$($model_root/.venv/bin/python -c \
      'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib")')
    export PYTHONPATH="$upstream/interformer:$upstream/docking:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
    export LD_LIBRARY_PATH="$torch_lib:$model_root/bin/lib:${LD_LIBRARY_PATH:-}"
    export BABEL_DATADIR="$model_root/bin/share/openbabel/3.1.0"
    ;;
  *)
    echo "unsupported model: $model" >&2
    exit 2
    ;;
esac

exec uv run --project "$model_root" --no-sync -- "$@"
