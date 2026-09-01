#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
external_root="$repo_root/external_models"
source_root="$external_root/src"
failures=0
incomplete=0
strict=0
if [[ ${1:-} == --strict ]]; then
  strict=1
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--strict]" >&2
  exit 2
fi

check_source() {
  local name=$1
  local expected=$2
  local actual=missing
  if [[ -d "$source_root/$name/.git" ]]; then
    actual=$(git -C "$source_root/$name" rev-parse HEAD)
  fi
  if [[ "$actual" == "$expected" ]]; then
    printf 'PASS source %-20s %s\n' "$name" "$actual"
  else
    printf 'FAIL source %-20s expected=%s actual=%s\n' "$name" "$expected" "$actual"
    failures=$((failures + 1))
  fi
}

check_env() {
  local name=$1
  local prefix=$2
  local legacy_marker="$external_root/installed/$name.complete"
  local uv_marker="$(dirname "$prefix")/state/synced.json"
  if [[ -x "$prefix/bin/python" && ( -f "$legacy_marker" || -f "$uv_marker" ) ]]; then
    local version
    version=$($prefix/bin/python --version 2>&1)
    printf 'PASS env    %-20s %s\n' "$name" "$version"
  else
    printf 'WAIT env    %-20s %s\n' "$name" "$prefix"
    incomplete=$((incomplete + 1))
  fi
}

check_optional_env() {
  local name=$1
  local prefix=$2
  if [[ -x "$prefix/bin/python" && -f "$external_root/installed/$name.complete" ]]; then
    local version
    version=$($prefix/bin/python --version 2>&1)
    printf 'PASS optional-env %-11s %s\n' "$name" "$version"
  else
    printf 'SKIP optional-env %-11s excluded from primary comparison\n' "$name"
  fi
}

check_adfr_env() {
  local name=$1
  local prefix=$2
  local prepare_receptor="$prefix/CCSBpckgs/AutoDockTools/Utilities24/prepare_receptor4.py"
  if [[ -x "$prefix/bin/python2.7" && -f "$prepare_receptor" && -f "$external_root/installed/$name.complete" ]]; then
    if "$prefix/bin/python2.7" -c \
      'import sys, numpy; from MolKit import Read; assert sys.maxunicode == 65535' \
      >/dev/null 2>&1; then
      printf 'PASS env    %-20s Python 2.7.3 UCS2 / MolKit\n' "$name"
    else
      printf 'FAIL env    %-20s incompatible ADFR Python/NumPy ABI\n' "$name"
      failures=$((failures + 1))
    fi
  else
    printf 'WAIT env    %-20s %s\n' "$name" "$prefix"
    incomplete=$((incomplete + 1))
  fi
}

check_artifact() {
  local name=$1
  local sentinel
  case "$name" in
    surfdock) sentinel="$source_root/surfdock/model_weights/docking/best_ema_inference_epoch_model.pt" ;;
    rldiff) sentinel="$external_root/weights/rldiff/DD_Pocket_RL_score_model/DD_Pocket_RL_score_model.pt" ;;
    diffbindfr) sentinel="$source_root/diffbindfr/DiffBindFR/weights" ;;
    interformer) sentinel="$source_root/interformer/checkpoints/v0.2_energy_model" ;;
    posebench-diffdock) sentinel="$source_root/posebench/forks/DiffDock/workdir/v1.1/score_model" ;;
    posebench-fabind) sentinel="$source_root/posebench/forks/FABind/ckpt/best_model.bin" ;;
    posebench-dynamicbind) sentinel="$source_root/posebench/forks/DynamicBind/workdir" ;;
    posebench-p2rank) sentinel="$source_root/posebench/forks/P2Rank/p2rank_2.4.2/prank" ;;
    sigmadock) sentinel="$external_root/weights/sigmadock/checkpoint.ckpt" ;;
    *) echo "unknown artifact check: $name" >&2; exit 2 ;;
  esac

  if [[ -f "$external_root/artifacts/$name.complete" && -e "$sentinel" ]]; then
    if [[ "$name" == posebench-fabind ]]; then
      local actual_size
      local actual_sha256
      actual_size=$(stat -Lc %s "$sentinel")
      actual_sha256=$(sha256sum "$sentinel" | cut -d' ' -f1)
      if [[ "$actual_size" -ne 145251173 || "$actual_sha256" != 549d6f1cef6f8fcbc0c068afa572fa99df58886440f67a124c3bb0fbebe09622 ]]; then
        printf 'FAIL weights %-20s invalid Git-LFS object size/hash\n' "$name"
        failures=$((failures + 1))
        return
      fi
    fi
    local count=0
    if [[ -f "$external_root/artifacts/$name.sha256" ]]; then
      count=$(wc -l < "$external_root/artifacts/$name.sha256")
    fi
    printf 'PASS weights %-20s sha256_entries=%s\n' "$name" "$count"
  elif [[ "$name" == sigmadock && -f "$external_root/artifacts/sigmadock.status" ]]; then
    printf 'BLOCKED weights %-17s upstream checkpoint release is missing\n' "$name"
    incomplete=$((incomplete + 1))
  else
    printf 'WAIT weights %-20s\n' "$name"
    incomplete=$((incomplete + 1))
  fi
}

check_source sigmadock 64c7e84608a46eb6bcd82353465c72bf4f919e2d
check_source rldiff b41f4c2ba608106388351974ab6c27bde914c1e9
check_source diffdock-pocket 3902bdd4d42ee5254d37aa694d005a992c92ad93
check_source diffbindfr b8bb82027fab5e74fc83ce2a44c0f920a9012ad3
check_source interformer 0914273a684fd53164d3561ab9253565aaa12a0b
check_source posebench c5d728d2a31ddb0a27512be75ea2d44e391e6529
check_source fabind 0698bd4f39f74169aaefc5b9350d6968a8076065
check_source flowdock 9473b8e5600981950c88f3963cf3871ce8a8d1f2
check_source surfdock 6420335c79c7dc83157ffc2ea8e15f84f4bf5e5c

check_env sigmadock "$repo_root/others/sigmadock/.venv"
check_env surfdock "$repo_root/others/surfdock/.venv"
check_env rldiff "$external_root/envs/rldiff"
check_env diffbindfr "$repo_root/others/diffbindfr/.venv"
check_env interformer "$repo_root/others/interformer/.venv"
check_env posebench-core "$external_root/envs/posebench"
check_env posebench-diffdock "$source_root/posebench/forks/DiffDock/DiffDock"
check_env posebench-fabind "$source_root/posebench/forks/FABind/FABind"
check_env posebench-dynamicbind "$source_root/posebench/forks/DynamicBind/DynamicBind"
check_optional_env posebench-flowdock "$source_root/posebench/forks/FlowDock/FlowDock"
check_adfr_env posebench-vina "$external_root/envs/posebench-adfr-clean"

check_artifact sigmadock
check_artifact surfdock
check_artifact rldiff
check_artifact diffbindfr
check_artifact interformer
check_artifact posebench-diffdock
check_artifact posebench-fabind
check_artifact posebench-dynamicbind
check_artifact posebench-p2rank

if (( failures > 0 )); then
  echo "$failures installation verification check(s) failed" >&2
  exit 1
fi
if (( strict == 1 && incomplete > 0 )); then
  echo "$incomplete installation component(s) are incomplete or blocked" >&2
  exit 1
fi
