#!/usr/bin/env bash
set -euo pipefail

repo_root=${EFFDOCK_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
external_root="$repo_root/external_models"
source_root="$external_root/src/posebench/forks/DiffDock"
archive="$external_root/weights/posebench-diffdock/diffdock_models.zip"
destination="$source_root/workdir/v1.1"
artifact_root="$external_root/artifacts"
quarantine="$artifact_root/quarantine/posebench-diffdock-wrong-root-59841"
expected_size=129825226
expected_sha256=5a95b6a1555be47ab1d6f0a8ffd25152f7fe32f5956005bb821e13e7a37d4a3d

[[ -f "$archive" ]] || { echo "missing DiffDock archive: $archive" >&2; exit 2; }
actual_size=$(stat -c %s "$archive")
actual_sha256=$(sha256sum "$archive" | cut -d' ' -f1)
if [[ "$actual_size" -ne "$expected_size" || "$actual_sha256" != "$expected_sha256" ]]; then
  echo "DiffDock archive validation failed: size=$actual_size sha256=$actual_sha256" >&2
  exit 2
fi

# Job 59841 extracted the valid archive one directory too high. Preserve that
# failed layout for diagnosis instead of deleting it.
for name in score_model confidence_model; do
  if [[ -d "$source_root/$name" ]]; then
    mkdir -p "$quarantine"
    if [[ -e "$quarantine/$name" ]]; then
      echo "quarantine destination already exists: $quarantine/$name" >&2
      exit 2
    fi
    mv "$source_root/$name" "$quarantine/$name"
  fi
done

mkdir -p "$destination"
python3 -m zipfile -e "$archive" "$destination"
[[ -f "$destination/score_model/best_ema_inference_epoch_model.pt" ]] || {
  echo "DiffDock score checkpoint extraction failed" >&2
  exit 2
}
[[ -f "$destination/confidence_model/best_model_epoch75.pt" ]] || {
  echo "DiffDock confidence checkpoint extraction failed" >&2
  exit 2
}

mkdir -p "$artifact_root"
printf '%s  %s\n' "$actual_sha256" "$archive" > "$artifact_root/posebench-diffdock.sha256"
date -u +%Y-%m-%dT%H:%M:%SZ > "$artifact_root/posebench-diffdock.complete"
echo "artifacts complete: posebench-diffdock"
