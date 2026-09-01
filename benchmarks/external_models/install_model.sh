#!/usr/bin/env bash
set -euo pipefail

model=${1:-${MODEL:-}}
if [[ -z "$model" ]]; then
  echo "usage: $0 <install-target>" >&2
  exit 2
fi

repo_root=${EFFDOCK_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
external_root="$repo_root/external_models"
source_root="$external_root/src"
env_root="$external_root/envs"
lock_root="$external_root/locks"
marker_root="$external_root/installed"
generated_root="$external_root/generated"
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$external_root/cache/micromamba}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$external_root/cache/uv}
mkdir -p \
  "$env_root" "$lock_root" "$marker_root" "$generated_root" \
  "$MAMBA_ROOT_PREFIX" "$UV_CACHE_DIR"

need_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "required command not found: $1" >&2
    exit 2
  }
}

verify_source_revision() {
  local name=$1
  local source="$source_root/$name"
  local expected
  local actual
  need_command python3
  [[ -d "$source/.git" ]] || { echo "missing source: $source" >&2; exit 2; }
  expected=$(python3 -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["sources"][sys.argv[2]]["commit"])' \
    "$repo_root/configs/external_models.json" "$name")
  actual=$(git -C "$source" rev-parse HEAD)
  if [[ "$actual" != "$expected" ]]; then
    echo "source revision mismatch for $name: expected=$expected actual=$actual" >&2
    exit 2
  fi
  printf 'source_revision[%s]=%s\n' "$name" "$actual"
}

create_mamba_env() {
  local prefix=$1
  local yaml=$2
  need_command micromamba
  if [[ -x "$prefix/bin/python" ]]; then
    micromamba env update --yes --prefix "$prefix" --file "$yaml"
  else
    micromamba env create --yes --prefix "$prefix" --file "$yaml"
  fi
}

mamba_python() {
  local prefix=$1
  shift
  micromamba run --prefix "$prefix" python "$@"
}

capture_mamba_lock() {
  local name=$1
  local prefix=$2
  micromamba list --prefix "$prefix" --explicit > "$lock_root/$name-conda-explicit.txt"
  micromamba run --prefix "$prefix" python -m pip freeze --all > "$lock_root/$name-pip-freeze.txt"
}

capture_uv_lock() {
  local name=$1
  local prefix=$2
  "$prefix/bin/python" -m pip freeze --all > "$lock_root/$name-pip-freeze.txt"
}

case "$model" in
  sigmadock|surfdock|diffbindfr|interformer)
    verify_source_revision "$model"
    ;;
  rldiff)
    verify_source_revision rldiff
    verify_source_revision diffdock-pocket
    ;;
  posebench-*)
    verify_source_revision posebench
    ;;
esac

case "$model" in
  sigmadock)
    need_command uv
    src="$source_root/sigmadock"
    prefix="$env_root/sigmadock"
    [[ -d "$src/.git" ]] || { echo "missing source: $src" >&2; exit 2; }
    if [[ ! -x "$prefix/bin/python" ]]; then
      uv venv --python 3.12 --seed "$prefix"
    fi
    if ! "$prefix/bin/python" -c "import sigmadock, torch" >/dev/null 2>&1; then
      (
        cd "$src"
        PATH="$prefix/bin:$PATH" bash install.sh cu126 train,test
      )
    fi
    "$prefix/bin/python" -c "import sigmadock, torch; print('sigmadock', torch.__version__)"
    capture_uv_lock "$model" "$prefix"
    ;;

  surfdock)
    src="$source_root/surfdock"
    prefix="$env_root/surfdock"
    [[ -f "$src/environment.yaml" ]] || { echo "missing source: $src" >&2; exit 2; }
    generated_yaml="$generated_root/surfdock-environment.yaml"
    python3 -c \
      'import pathlib, sys; source=pathlib.Path(sys.argv[1]).read_text(); marker="  - pip:\n"; assert source.count(marker) == 1; conda_part, _ = source.split(marker, 1); pathlib.Path(sys.argv[2]).write_text(conda_part)' \
      "$src/environment.yaml" "$generated_yaml"
    create_mamba_env "$prefix" "$generated_yaml"
    # environment.yaml is a machine export whose pip block contains conda
    # packages and wheel builds no longer published on PyPI. Reinstall only
    # the upstream README's runtime pip set; the full conda portion remains
    # unchanged above.
    mamba_python "$prefix" -m pip install \
      spyrmsd==0.7.0 \
      scikit-learn==1.3.2 \
      accelerate==0.15.0 \
      biopython==1.79 \
      e3nn==0.5.1 \
      huggingface-hub==0.17.3 \
      mdanalysis==2.4.0 \
      posebusters==0.2.13 \
      rdkit==2023.3.1 \
      tokenizers==0.13.3 \
      transformers==4.29.2 \
      wandb==0.16.1 \
      prefetch-generator==1.0.3 \
      pymesh==1.0.2
    # The upstream environment pins PyG CUDA wheels that are hosted only on
    # data.pyg.org. Micromamba's embedded pip phase searches PyPI alone, so
    # install the unchanged upstream versions explicitly from the official
    # torch-2.2/CUDA-12.1 wheel index after the conda transaction.
    mamba_python "$prefix" -m pip install --no-index \
      --find-links https://data.pyg.org/whl/torch-2.2.0+cu121.html \
      pyg-lib==0.4.0+pt22cu121 \
      torch-cluster==1.6.3+pt22cu121 \
      torch-scatter==2.1.2+pt22cu121 \
      torch-sparse==0.6.18+pt22cu121 \
      torch-spline-conv==1.2.2+pt22cu121
    mamba_python "$prefix" -m pip install --no-deps \
      https://github.com/nuvolos-cloud/PyMesh/releases/download/v0.3.1/pymesh2-0.3.1-cp310-cp310-linux_x86_64.whl
    # PyPI no longer serves upstream's pinned 1.3.2 wheel. Conda-forge still
    # publishes the exact noarch release, so keep the version while changing
    # only its distribution channel.
    micromamba install --yes --prefix "$prefix" --channel conda-forge dimorphite-dl=1.3.2
    # Meta tagged the repository code as 2.0.1 but published only 2.0.0 to
    # PyPI. Install the exact official archived-repository commit carrying
    # version 2.0.1.
    mamba_python "$prefix" -m pip install --no-deps \
      "fair-esm @ git+https://github.com/facebookresearch/esm.git@2b369911bb5b4b0dda914521b9475cad1656b2ac"
    tools_dir="$src/comp_surface/tools"
    if [[ ! -d "$tools_dir/transfer/APBS-3.4.1.Linux" ]]; then
      tar -xzf "$tools_dir/APBS_PDB2PQR.tar.gz" -C "$tools_dir"
    fi
    ln -sfn transfer/APBS-3.4.1.Linux "$tools_dir/APBS-3.4.1.Linux"
    ln -sfn transfer/pdb2pqr-linux-bin64-2.1.1 "$tools_dir/pdb2pqr-linux-bin64-2.1.1"
    (
      cd "$src"
      PYTHONPATH="$src" mamba_python "$prefix" -c "import dimorphite_dl, esm, rdkit, torch, torch_geometric; print('surfdock', torch.__version__)"
    )
    capture_mamba_lock "$model" "$prefix"
    ;;

  rldiff)
    src="$source_root/rldiff"
    dd_src="$source_root/diffdock-pocket"
    prefix="$env_root/rldiff"
    [[ -f "$src/inference_env.yml" && -d "$dd_src/.git" ]] || {
      echo "missing RLDiff or DiffDock-Pocket source" >&2
      exit 2
    }
    create_mamba_env "$prefix" "$src/inference_env.yml"
    if [[ -e "$dd_src/RLDiff" && ! -L "$dd_src/RLDiff" ]]; then
      echo "$dd_src/RLDiff exists and is not a symlink" >&2
      exit 2
    fi
    ln -sfn "$src" "$dd_src/RLDiff"
    # RLDiff derives the DiffDock-Pocket package path from its own location and
    # expects the upstream directory's exact capitalization.
    ln -sfn "$dd_src" "$source_root/DiffDock-Pocket"
    (
      cd "$dd_src"
      PYTHONPATH="$dd_src" mamba_python "$prefix" -c "import torch, torch_geometric, rdkit; print('rldiff', torch.__version__)"
    )
    capture_mamba_lock "$model" "$prefix"
    ;;

  diffbindfr)
    src="$source_root/diffbindfr"
    prefix="$env_root/diffbindfr"
    [[ -f "$src/env.yaml" ]] || { echo "missing source: $src" >&2; exit 2; }
    generated_yaml="$generated_root/diffbindfr-environment.yaml"
    python3 -c \
      'import functools, sys; source=open(sys.argv[1]).read(); replacements=(("    - scikit-learn==1.4.1\n", "    - scikit-learn==1.4.1.post1\n"), ("    - triton==2.0.0\n", "")); assert all(source.count(old) == 1 for old, _ in replacements); output=functools.reduce(lambda text, pair: text.replace(*pair), replacements, source); open(sys.argv[2], "w").write(output)' \
      "$src/env.yaml" "$generated_yaml"
    # PyPI no longer serves the exact 1.4.1 artifact selected by the upstream
    # Python 3.9 environment. 1.4.1.post1 is the packaging-only rebuild of the
    # same scikit-learn release and remains compatible with Python 3.9.
    create_mamba_env "$prefix" "$generated_yaml"
    # Installing Triton through pip resolves torch>=2 and silently replaces
    # the conda-pinned torch 1.13.1, making the pt113 PyG wheels unloadable.
    # Triton is not imported by inference setup; retain the upstream version
    # without allowing its dependency resolver to replace torch.
    mamba_python "$prefix" -m pip uninstall --yes torch
    micromamba install --yes --force-reinstall --prefix "$prefix" \
      --channel pytorch pytorch=1.13.1
    mamba_python "$prefix" -m pip install --no-deps triton==2.0.0
    # wandb 0.13.3 imports np.float_, which NumPy 2 removed. Keep the newest
    # Python-3.9-compatible NumPy 1.x release before importing DiffBindFR.
    mamba_python "$prefix" -m pip install --no-deps numpy==1.26.4
    find "$src/druglib/ops" -type f \( -name mkdssp -o -name msms -o -name smina.static \) -exec chmod +x {} +
    (
      cd "$src"
      mamba_python "$prefix" -m pip install --no-deps -e .
      mamba_python "$prefix" -c "import DiffBindFR, druglib, torch, torch_scatter; assert torch.__version__.startswith('1.13.1'), torch.__version__; print('diffbindfr', torch.__version__)"
    )
    capture_mamba_lock "$model" "$prefix"
    ;;

  interformer)
    src="$source_root/interformer"
    prefix="$env_root/interformer"
    [[ -f "$src/environment.yml" ]] || { echo "missing source: $src" >&2; exit 2; }
    create_mamba_env "$prefix" "$src/environment.yml"
    # The upstream environment pins Boost runtime libraries but PyVina also
    # compiles against Boost headers. Keep the same Boost release and add its
    # development package explicitly.
    micromamba install --yes --prefix "$prefix" --channel conda-forge \
      libboost-devel=1.84.0 ninja
    mamba_python "$prefix" -m pip install --no-deps plip
    (
      cd "$src/docking"
      MAX_JOBS=${MAX_JOBS:-2} mamba_python "$prefix" setup.py install
    )
    torch_lib="$prefix/lib/python3.12/site-packages/torch/lib"
    test -f "$torch_lib/libc10.so"
    LD_LIBRARY_PATH="$torch_lib:${LD_LIBRARY_PATH:-}" \
      mamba_python "$prefix" -c "import pyvina_core, torch; print('interformer', torch.__version__)"
    capture_mamba_lock "$model" "$prefix"
    ;;

  posebench-core)
    src="$source_root/posebench"
    prefix="$env_root/posebench"
    [[ -f "$src/environments/posebench_environment.yaml" ]] || { echo "missing source: $src" >&2; exit 2; }
    create_mamba_env "$prefix" "$src/environments/posebench_environment.yaml"
    (
      cd "$src"
      mamba_python "$prefix" -m pip install --no-deps -e .
      mamba_python "$prefix" -m pip install --no-deps numpy==1.26.4 prody==2.4.1
      mamba_python "$prefix" -c "import posebench, torch; print('posebench', torch.__version__)"
    )
    capture_mamba_lock "$model" "$prefix"
    ;;

  posebench-diffdock)
    src="$source_root/posebench"
    prefix="$src/forks/DiffDock/DiffDock"
    upstream_yaml="$src/environments/diffdock_environment.yaml"
    generated_yaml="$generated_root/posebench-diffdock-conda.yaml"
    generated_requirements="$generated_root/posebench-diffdock-requirements.txt"
    # The upstream lock mixes conda and pip dependencies. One pip dependency,
    # OpenFold, compiles a CUDA extension while the environment is being
    # created. CUDA 11.8 rejects the host's GCC 13, so split the phases and pin
    # an environment-local GCC/G++ 11 toolchain before invoking pip.
    python3 -c \
      'import pathlib, sys
source = pathlib.Path(sys.argv[1]).read_text()
marker = "  - pip:\n"
assert source.count(marker) == 1
conda_part, pip_and_prefix = source.split(marker, 1)
pip_part, separator, _ = pip_and_prefix.rpartition("\nprefix:")
assert separator and pip_part
requirements = []
for line in pip_part.splitlines():
    assert line.startswith("      - "), line
    requirements.append(line.removeprefix("      - "))
pathlib.Path(sys.argv[2]).write_text(conda_part)
pathlib.Path(sys.argv[3]).write_text("\n".join(requirements) + "\n")' \
      "$upstream_yaml" "$generated_yaml" "$generated_requirements"
    create_mamba_env "$prefix" "$generated_yaml"
    micromamba install --yes --prefix "$prefix" --channel conda-forge \
      gcc_linux-64=11 gxx_linux-64=11 ninja
    cc="$prefix/bin/x86_64-conda-linux-gnu-cc"
    cxx="$prefix/bin/x86_64-conda-linux-gnu-c++"
    [[ -x "$cc" && -x "$cxx" ]] || {
      echo "PoseBench DiffDock GCC/G++ 11 wrappers not found" >&2
      exit 2
    }
    CC="$cc" CXX="$cxx" CUDAHOSTCXX="$cxx" CUDA_HOME="$prefix" \
      mamba_python "$prefix" -m pip install -r "$generated_requirements"
    mamba_python "$prefix" -m pip install pyg-lib -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
    (
      cd "$src/forks/DiffDock"
      PYTHONPATH="$PWD" mamba_python "$prefix" -c "import torch, torch_geometric; print('posebench-diffdock', torch.__version__)"
    )
    capture_mamba_lock "$model" "$prefix"
    ;;

  posebench-fabind)
    src="$source_root/posebench"
    prefix="$src/forks/FABind/FABind"
    create_mamba_env "$prefix" "$src/environments/fabind_environment.yaml"
    (
      cd "$src/forks/FABind"
      PYTHONPATH="$PWD" mamba_python "$prefix" -c "import torch, torch_geometric; print('posebench-fabind', torch.__version__)"
    )
    capture_mamba_lock "$model" "$prefix"
    ;;

  posebench-dynamicbind)
    src="$source_root/posebench"
    prefix="$src/forks/DynamicBind/DynamicBind"
    create_mamba_env "$prefix" "$src/environments/dynamicbind_environment.yaml"
    mamba_python "$prefix" -m pip install pyg-lib -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
    (
      cd "$src/forks/DynamicBind"
      PYTHONPATH="$PWD" mamba_python "$prefix" -c "import torch, torch_geometric; print('posebench-dynamicbind', torch.__version__)"
    )
    capture_mamba_lock "$model" "$prefix"
    ;;

  posebench-flowdock)
    src="$source_root/posebench"
    prefix="$src/forks/FlowDock/FlowDock"
    create_mamba_env "$prefix" "$src/environments/flowdock_environment.yaml"
    (
      cd "$src/forks/FlowDock"
      mamba_python "$prefix" -m pip install --no-deps -e .
      mamba_python "$prefix" -c "import flowdock, torch; print('posebench-flowdock', torch.__version__)"
    )
    capture_mamba_lock "$model" "$prefix"
    ;;

  posebench-vina)
    src="$source_root/posebench"
    prefix="$env_root/posebench-adfr-clean"
    need_command micromamba
    # PoseBench's exported ADFR environment combines an ADFR Python 2.7.3
    # (UCS2) runtime with newer Python 2.7.18/NumPy (UCS4) packages.  MolKit
    # then fails to import with an unresolved _PyUnicodeUCS4_* symbol.  Install
    # the self-consistent hcc ADFR package without updating its bundled Python
    # or NumPy runtime.
    if [[ ! -x "$prefix/bin/python2.7" ]]; then
      micromamba create --yes --prefix "$prefix" --override-channels \
        --channel hcc --channel conda-forge adfr-suite=1.0=0
    fi
    prepare_receptor="$prefix/CCSBpckgs/AutoDockTools/Utilities24/prepare_receptor4.py"
    [[ -f "$prepare_receptor" ]] || {
      echo "ADFR prepare_receptor4.py not found: $prepare_receptor" >&2
      exit 2
    }
    "$prefix/bin/python2.7" -c \
      'import sys, numpy; from MolKit import Read; assert sys.maxunicode == 65535; print("posebench-vina ADFR", sys.version.split()[0], numpy.__version__)'
    micromamba list --prefix "$prefix" --explicit > \
      "$lock_root/$model-conda-explicit.txt"
    ;;

  *)
    echo "unknown install target: $model" >&2
    exit 2
    ;;
esac

date -u +%Y-%m-%dT%H:%M:%SZ > "$marker_root/$model.complete"
echo "installation complete: $model"
