#!/usr/bin/env bash
set -euo pipefail

model=${1:-${MODEL:-}}
if [[ -z "$model" ]]; then
  echo "usage: $0 <sigmadock|surfdock|diffbindfr|interformer>" >&2
  exit 2
fi

repo_root=${EFFDOCK_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
model_root="$repo_root/others/$model"
legacy_source="$repo_root/external_models/src/$model"
upstream="$model_root/upstream"
python="$model_root/.venv/bin/python"
uv_cache="$model_root/.cache/uv"

case "$model" in
  sigmadock|surfdock|diffbindfr|interformer) ;;
  *)
    echo "unsupported model-local uv project: $model" >&2
    exit 2
    ;;
esac

for required in "$model_root/pyproject.toml" "$model_root/.python-version" \
  "$repo_root/configs/external_models.json"; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 2; }
done
command -v uv >/dev/null 2>&1 || { echo "uv is required" >&2; exit 2; }
expected_revision=$(python3 -c \
  'import json, sys; print(json.load(open(sys.argv[1]))["sources"][sys.argv[2]]["commit"])' \
  "$repo_root/configs/external_models.json" "$model")

mkdir -p "$model_root/.cache" "$model_root/logs" "$model_root/state"
if [[ ! -e "$upstream" && ! -L "$upstream" ]]; then
  if [[ -d "$legacy_source/.git" ]]; then
    ln -s "../../external_models/src/$model" "$upstream"
  else
    source_url=$(python3 -c \
      'import json, sys; print(json.load(open(sys.argv[1]))["sources"][sys.argv[2]]["url"])' \
      "$repo_root/configs/external_models.json" "$model")
    git clone "$source_url" "$upstream"
    git -C "$upstream" checkout --detach "$expected_revision"
  fi
fi
[[ -d "$upstream/.git" ]] || { echo "missing upstream checkout: $upstream" >&2; exit 2; }

actual_revision=$(git -C "$upstream" rev-parse HEAD)
if [[ "$actual_revision" != "$expected_revision" ]]; then
  echo "source revision mismatch: model=$model expected=$expected_revision actual=$actual_revision" >&2
  exit 2
fi

case "$model" in
  sigmadock) weight_target=../../external_models/weights/sigmadock ;;
  surfdock) weight_target=upstream/model_weights ;;
  diffbindfr) weight_target=upstream/DiffBindFR/weights ;;
  interformer) weight_target=upstream/checkpoints ;;
esac
if [[ ! -e "$model_root/weights" && ! -L "$model_root/weights" ]]; then
  ln -s "$weight_target" "$model_root/weights"
fi
[[ -d "$model_root/weights" ]] || { echo "missing weights: $model_root/weights" >&2; exit 2; }

if [[ "$model" == sigmadock ]]; then
  checkpoint="$model_root/weights/checkpoint.ckpt"
  gnina_source="$repo_root/external_models/bin/gnina-sigmadock-v1.3.2"
  expected_checkpoint_sha256=db15427ca349e6f1e5f894bff841112c7360384886aa472667d8011307cad382
  expected_gnina_sha256=5d33538324b40050a03aa262d51832837e0ea6cc100945abbd2d7b732589690e
  [[ -f "$checkpoint" ]] || { echo "missing SigmaDock checkpoint: $checkpoint" >&2; exit 2; }
  [[ -x "$gnina_source" ]] || { echo "missing SigmaDock GNINA: $gnina_source" >&2; exit 2; }
  actual_checkpoint_sha256=$(sha256sum "$checkpoint" | cut -d' ' -f1)
  actual_gnina_sha256=$(sha256sum "$gnina_source" | cut -d' ' -f1)
  [[ "$actual_checkpoint_sha256" == "$expected_checkpoint_sha256" ]] || {
    echo "SigmaDock checkpoint digest mismatch" >&2
    exit 2
  }
  [[ "$actual_gnina_sha256" == "$expected_gnina_sha256" ]] || {
    echo "SigmaDock GNINA digest mismatch" >&2
    exit 2
  }
  mkdir -p "$model_root/bin"
  if [[ ! -e "$model_root/bin/gnina" && ! -L "$model_root/bin/gnina" ]]; then
    ln -s ../../../external_models/bin/gnina-sigmadock-v1.3.2 \
      "$model_root/bin/gnina"
  fi
  [[ -x "$model_root/bin/gnina" ]] || { echo "invalid model-local GNINA link" >&2; exit 2; }
fi

export UV_CACHE_DIR="$uv_cache"
export UV_PROJECT_ENVIRONMENT="$model_root/.venv"
uv sync --project "$model_root" --no-install-project
[[ -x "$python" ]] || { echo "uv did not create $python" >&2; exit 2; }

case "$model" in
  sigmadock)
    PYTHONPATH="$upstream/src:$upstream:$repo_root" "$python" -c \
      'import e3nn, hydra, pytorch_lightning, rdkit, sigmadock, torch, torch_geometric; assert torch.__version__.startswith("2.13.0"), torch.__version__; print("sigmadock", torch.__version__)'
    sigmadock_cudnn_lib=$($python -c \
      'import pathlib, sysconfig; print(pathlib.Path(sysconfig.get_paths()["purelib"]) / "nvidia/cudnn/lib")')
    LD_LIBRARY_PATH="$sigmadock_cudnn_lib:${LD_LIBRARY_PATH:-}" \
      "$model_root/bin/gnina" --version
    ;;

  surfdock)
    tools_dir="$upstream/comp_surface/tools"
    if [[ ! -d "$tools_dir/transfer/APBS-3.4.1.Linux" ]]; then
      tar -xzf "$tools_dir/APBS_PDB2PQR.tar.gz" -C "$tools_dir"
    fi
    ln -sfn transfer/APBS-3.4.1.Linux "$tools_dir/APBS-3.4.1.Linux"
    ln -sfn transfer/pdb2pqr-linux-bin64-2.1.1 "$tools_dir/pdb2pqr-linux-bin64-2.1.1"
    PYTHONPATH="$upstream:$repo_root" "$python" -c \
      'import accelerate, esm, openmm, pymesh, rdkit, torch, torch_geometric; from pathlib import Path; from scripts.external_models.prepare_surfdock_runtime import configure_surface_imports; configure_surface_imports(Path(__import__("sys").argv[1])); print("surfdock", torch.__version__)' \
      "$upstream"
    ;;

  diffbindfr)
    find "$upstream/druglib/ops" -type f \
      \( -name mkdssp -o -name msms -o -name smina.static \) -exec chmod +x {} +
    PYTHONPATH="$upstream" "$python" -c \
      'import DiffBindFR, torch, torch_scatter; assert torch.__version__.startswith("1.13.1"), torch.__version__; print("diffbindfr", torch.__version__)'
    PYTHONPATH="$upstream" "$python" \
      "$repo_root/scripts/external_models/run_seeded_upstream.py" \
      --upstream-script "$upstream/DiffBindFR/app/predict.py" \
      --upstream-cwd "$upstream" \
      --seed 0 \
      --stub-diffbindfr-pymol \
      --help >/dev/null
    ;;

  interformer)
    native_root="$model_root/bin"
    native_lib="$native_root/lib"
    native_include="$native_root/include"
    legacy_env="$repo_root/external_models/envs/interformer"
    mkdir -p "$native_lib" "$native_include"
    if [[ ! -d "$legacy_env" ]] && ! \
      "$python" "$repo_root/scripts/others/bootstrap_interformer_native.py" \
        --model-root "$model_root" --verify-only >/dev/null 2>&1; then
      "$python" "$repo_root/scripts/others/bootstrap_interformer_native.py" \
        --model-root "$model_root"
    fi
    if [[ ! -x "$native_root/reduce" ]]; then
      [[ -x "$legacy_env/bin/reduce" ]] || {
        echo "Interformer requires the Reduce executable; no archived copy was found" >&2
        exit 2
      }
      cp -L "$legacy_env/bin/reduce" "$native_root/reduce"
      chmod +x "$native_root/reduce"
      cp -L "$legacy_env/lib/libgcc_s.so.1" "$native_lib/libgcc_s.so.1"
      cp -L "$legacy_env/lib/libstdc++.so.6.0.33" "$native_lib/libstdc++.so.6.0.33"
      ln -sfn libstdc++.so.6.0.33 "$native_lib/libstdc++.so.6"
    fi
    obrms_lib="$native_root/obrms-lib"
    mkdir -p "$obrms_lib" "$native_root/share/openbabel"
    if [[ ! -x "$native_root/obrms.real" ]]; then
      [[ -x "$legacy_env/bin/obrms" ]] || {
        echo "Interformer requires the OpenBabel obrms executable" >&2
        exit 2
      }
      cp -L "$legacy_env/bin/obrms" "$native_root/obrms.real"
      chmod +x "$native_root/obrms.real"
    fi
    if [[ ! -f "$obrms_lib/libopenbabel.so.7.0.0" ]]; then
      [[ -f "$legacy_env/lib/libopenbabel.so.7.0.0" ]] || {
        echo "Interformer requires the native OpenBabel runtime" >&2
        exit 2
      }
      cp -L "$legacy_env/lib/libopenbabel.so.7.0.0" \
        "$obrms_lib/libopenbabel.so.7.0.0"
    fi
    ln -sfn libopenbabel.so.7.0.0 "$obrms_lib/libopenbabel.so.7"
    ln -sfn ../../../scripts/others/interformer_obrms.sh "$native_root/obrms"
    if [[ ! -d "$native_root/share/openbabel/3.1.0" ]]; then
      [[ -d "$legacy_env/share/openbabel/3.1.0" ]] || {
        echo "Interformer requires OpenBabel data files" >&2
        exit 2
      }
      cp -a "$legacy_env/share/openbabel/3.1.0" \
        "$native_root/share/openbabel/3.1.0"
    fi
    rm -f "$native_lib/libopenbabel.so.7" "$native_lib/libopenbabel.so.7.0.0"
    if [[ ! -d "$native_include/boost" ]]; then
      [[ -d "$legacy_env/include/boost" ]] || {
        echo "Interformer requires Boost headers; no archived copy was found" >&2
        exit 2
      }
      cp -a "$legacy_env/include/boost" "$native_include/boost"
      cp -L "$legacy_env/lib/libboost_system.so.1.84.0" \
        "$native_lib/libboost_system.so.1.84.0"
      ln -sfn libboost_system.so.1.84.0 "$native_lib/libboost_system.so"
    fi
    "$python" "$repo_root/scripts/others/bootstrap_interformer_native.py" \
      --model-root "$model_root" --verify-only
    torch_lib=$($python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib")')
    if ! LD_LIBRARY_PATH="$torch_lib:$native_lib:${LD_LIBRARY_PATH:-}" \
      "$python" -c 'import torch, pyvina_core' >/dev/null 2>&1; then
      MAX_JOBS=${MAX_JOBS:-2} \
      CPATH="$native_include:${CPATH:-}" \
      LIBRARY_PATH="$native_lib:${LIBRARY_PATH:-}" \
      LD_LIBRARY_PATH="$native_lib:${LD_LIBRARY_PATH:-}" \
        uv pip install --python "$python" --no-build-isolation --reinstall \
          "$upstream/docking"
    fi
    PYTHONPATH="$upstream/interformer:$upstream/docking" \
    LD_LIBRARY_PATH="$torch_lib:$native_lib:${LD_LIBRARY_PATH:-}" \
      "$python" -c \
      'import openbabel, pyvina_core, rdkit, torch; print("interformer", torch.__version__)'
    BABEL_DATADIR="$native_root/share/openbabel/3.1.0" \
    LD_LIBRARY_PATH="$native_lib:${LD_LIBRARY_PATH:-}" \
      "$native_root/obrms" -h >/dev/null
    ;;
esac

"$python" -c \
  'import json, pathlib, platform, sys; p=pathlib.Path(sys.argv[1]); p.write_text(json.dumps({"schema_version": 1, "python": sys.version.split()[0], "platform": platform.platform()}, indent=2) + "\n")' \
  "$model_root/state/synced.json"
echo "model=$model"
echo "upstream_revision=$actual_revision"
echo "python=$($python --version 2>&1)"
echo "environment=$model_root/.venv"
