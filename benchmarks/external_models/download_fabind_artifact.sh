#!/usr/bin/env bash
set -euo pipefail

repo_root=${EFFDOCK_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
external_root="$repo_root/external_models"
standalone="$external_root/src/fabind"
checkpoint_repo="$standalone/FABind/ckpt"
posebench_target="$external_root/src/posebench/forks/FABind/ckpt"
weight_root="$external_root/weights/posebench-fabind"
materialized="$external_root/materialized/posebench-fabind/ckpt"
artifact_root="$external_root/artifacts"
parallel_downloader=${EFFDOCK_PARALLEL_DOWNLOADER:-$repo_root/scripts/external_models/parallel_download.py}
expected_revision=88f403994738a6424df977ea1e8531f42c063be3
expected_size=145251173
expected_sha256=549d6f1cef6f8fcbc0c068afa572fa99df58886440f67a124c3bb0fbebe09622
weight_url="https://huggingface.co/QizhiPei/FABind_model/resolve/$expected_revision/best_model.bin?download=true"

[[ -d "$standalone/.git" ]] || { echo "missing FABind source: $standalone" >&2; exit 2; }

git -C "$standalone" submodule update --init --depth 1 FABind/ckpt
actual_revision=$(git -C "$checkpoint_repo" rev-parse HEAD)
if [[ "$actual_revision" != "$expected_revision" ]]; then
  echo "FABind checkpoint revision mismatch: $actual_revision" >&2
  exit 2
fi

pointer="$checkpoint_repo/best_model.bin"
grep -qx "oid sha256:$expected_sha256" "$pointer" || {
  echo "FABind Git-LFS pointer does not declare the expected object" >&2
  exit 2
}
grep -qx "size $expected_size" "$pointer" || {
  echo "FABind Git-LFS pointer does not declare the expected size" >&2
  exit 2
}

mkdir -p "$weight_root"
weight="$weight_root/best_model.bin"
python3 "$parallel_downloader" "$weight_url" "$weight" "$expected_size" --workers 8
actual_size=$(stat -c %s "$weight")
actual_sha256=$(sha256sum "$weight" | cut -d' ' -f1)
if [[ "$actual_size" -ne "$expected_size" || "$actual_sha256" != "$expected_sha256" ]]; then
  echo "FABind checkpoint validation failed: size=$actual_size sha256=$actual_sha256" >&2
  exit 2
fi

mkdir -p "$materialized"
ln -sfn "$weight" "$materialized/best_model.bin"
if [[ -e "$posebench_target" && ! -L "$posebench_target" ]]; then
  echo "$posebench_target exists and is not a symlink" >&2
  exit 2
fi
ln -sfn "$materialized" "$posebench_target"

mkdir -p "$artifact_root"
printf '%s  %s\n' "$actual_sha256" "$weight" > "$artifact_root/posebench-fabind.sha256"
date -u +%Y-%m-%dT%H:%M:%SZ > "$artifact_root/posebench-fabind.complete"
echo "artifacts complete: posebench-fabind"
