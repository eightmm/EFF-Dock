#!/usr/bin/env bash
set -euo pipefail

repo_root=${EFFDOCK_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
external_root="$repo_root/external_models"
source_root="$external_root/src/diffbindfr"
archive="$external_root/weights/diffbindfr/weights.tar.gz"
destination="$source_root/DiffBindFR/weights"
wrong_root="$source_root/weights"
artifact_root="$external_root/artifacts"
expected_size=364487597
expected_md5=f64bc5e495e24477d5cd63ca86eb4dcb
expected_sha256=544627773717816a35135be388d52670740e043c6c00dff5d8c1f57682f6276c

[[ -f "$archive" ]] || { echo "missing DiffBindFR archive: $archive" >&2; exit 2; }
actual_size=$(stat -c %s "$archive")
actual_md5=$(md5sum "$archive" | cut -d' ' -f1)
actual_sha256=$(sha256sum "$archive" | cut -d' ' -f1)
if [[ "$actual_size" -ne "$expected_size" || "$actual_md5" != "$expected_md5" || "$actual_sha256" != "$expected_sha256" ]]; then
  echo "DiffBindFR archive validation failed: size=$actual_size md5=$actual_md5 sha256=$actual_sha256" >&2
  exit 2
fi

if [[ ! -d "$destination" ]]; then
  if [[ -d "$wrong_root" ]]; then
    mv "$wrong_root" "$destination"
  else
    python3 -c \
      'import sys, tarfile; tarfile.open(sys.argv[1], "r:gz").extractall(sys.argv[2], filter="data")' \
      "$archive" "$source_root/DiffBindFR"
  fi
fi
[[ -f "$destination/diffbindfr_paper.pth" && -f "$destination/mdn_paper.pt" ]] || {
  echo "DiffBindFR checkpoint materialization failed" >&2
  exit 2
}

mkdir -p "$artifact_root"
printf '%s  %s\n' "$actual_sha256" "$archive" > "$artifact_root/diffbindfr.sha256"
date -u +%Y-%m-%dT%H:%M:%SZ > "$artifact_root/diffbindfr.complete"
echo "artifacts complete: diffbindfr"
