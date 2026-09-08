#!/usr/bin/env bash

effdock_resolve_pocket_prior_condition() {
  local condition_index=${1:?condition index required}
  [[ "$condition_index" =~ ^[0-9]+$ && "$condition_index" -lt 27 ]] || {
    echo "invalid condition index: $condition_index" >&2
    return 2
  }
  if (( condition_index < 9 )); then
    EFFDOCK_CONDITION_CUTOFF=14
    EFFDOCK_CONDITION_SIGMA=2
    EFFDOCK_CONDITION_JITTER=$((condition_index / 3))
    EFFDOCK_CONDITION_REPEAT=$((condition_index % 3))
  else
    local offset=$((condition_index - 9))
    local sigma_index=$((offset / 9))
    EFFDOCK_CONDITION_SIGMA=$((sigma_index == 0 ? 1 : 4))
    EFFDOCK_CONDITION_CUTOFF=10
    local within_sigma=$((offset % 9))
    EFFDOCK_CONDITION_JITTER=$((within_sigma / 3))
    EFFDOCK_CONDITION_REPEAT=$((within_sigma % 3))
  fi
  export EFFDOCK_CONDITION_CUTOFF EFFDOCK_CONDITION_SIGMA
  export EFFDOCK_CONDITION_JITTER EFFDOCK_CONDITION_REPEAT
}

effdock_pocket_prior_condition_root() {
  local root=${1:?output root required}
  printf '%s/cutoff_%02d/sigma_%02d/jitter_%02d/repeat_%d' \
    "$root" "$EFFDOCK_CONDITION_CUTOFF" "$EFFDOCK_CONDITION_SIGMA" \
    "$EFFDOCK_CONDITION_JITTER" "$EFFDOCK_CONDITION_REPEAT"
}
