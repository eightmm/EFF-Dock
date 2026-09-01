#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source_root="$repo_root/external_models/src"
mkdir -p "$source_root"

ensure_source() {
  local name=$1
  local url=$2
  local commit=$3
  local target="$source_root/$name"

  if [[ ! -d "$target/.git" ]]; then
    git clone --filter=blob:none "$url" "$target"
  fi

  local actual_url
  actual_url=$(git -C "$target" remote get-url origin)
  if [[ "$actual_url" != "$url" && "$actual_url" != "${url%.git}" ]]; then
    echo "source remote mismatch for $name: $actual_url" >&2
    exit 2
  fi

  if ! git -C "$target" cat-file -e "$commit^{commit}" 2>/dev/null; then
    git -C "$target" fetch --depth=1 origin "$commit"
  fi
  git -C "$target" checkout --detach "$commit"

  local actual_commit
  actual_commit=$(git -C "$target" rev-parse HEAD)
  if [[ "$actual_commit" != "$commit" ]]; then
    echo "source revision mismatch for $name: $actual_commit" >&2
    exit 2
  fi
  printf '%-18s %s\n' "$name" "$actual_commit"
}

ensure_source sigmadock https://github.com/alvaroprat97/sigmadock.git 64c7e84608a46eb6bcd82353465c72bf4f919e2d
ensure_source rldiff https://github.com/oxpig/RLDiff.git b41f4c2ba608106388351974ab6c27bde914c1e9
ensure_source diffdock-pocket https://github.com/plainerman/DiffDock-Pocket.git 3902bdd4d42ee5254d37aa694d005a992c92ad93
ensure_source diffbindfr https://github.com/HBioquant/DiffBindFR.git b8bb82027fab5e74fc83ce2a44c0f920a9012ad3
ensure_source interformer https://github.com/tencent-ailab/Interformer.git 0914273a684fd53164d3561ab9253565aaa12a0b
ensure_source posebench https://github.com/BioinfoMachineLearning/PoseBench.git c5d728d2a31ddb0a27512be75ea2d44e391e6529
ensure_source fabind https://github.com/QizhiPei/FABind.git 0698bd4f39f74169aaefc5b9350d6968a8076065
ensure_source flowdock https://github.com/BioinfoMachineLearning/FlowDock.git 9473b8e5600981950c88f3963cf3871ce8a8d1f2
ensure_source surfdock https://github.com/Intelligent-Drug-Discovery-Lab/SurfDock.git 6420335c79c7dc83157ffc2ea8e15f84f4bf5e5c
