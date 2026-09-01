#!/usr/bin/env bash
set -euo pipefail

model=${1:-${MODEL:-}}
if [[ -z "$model" ]]; then
  echo "usage: $0 <artifact-target>" >&2
  exit 2
fi

repo_root=${EFFDOCK_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
external_root="$repo_root/external_models"
source_root="$external_root/src"
weight_root="$external_root/weights"
artifact_root="$external_root/artifacts"
bin_root="$external_root/bin"
parallel_downloader=${EFFDOCK_PARALLEL_DOWNLOADER:-$repo_root/scripts/external_models/parallel_download.py}
fabind_downloader=${EFFDOCK_FABIND_DOWNLOADER:-$repo_root/scripts/external_models/download_fabind_artifact.sh}
mkdir -p "$weight_root" "$artifact_root" "$bin_root"

extract_zip() {
  local archive=$1
  local destination=$2
  mkdir -p "$destination"
  python3 -m zipfile -e "$archive" "$destination"
}

extract_tar_gz() {
  local archive=$1
  local destination=$2
  mkdir -p "$destination"
  python3 -c \
    'import sys, tarfile; tarfile.open(sys.argv[1], "r:gz").extractall(sys.argv[2], filter="data")' \
    "$archive" "$destination"
}

download() {
  local url=$1
  local target=$2
  local size=$3
  local algorithm=${4:-}
  local digest=${5:-}
  mkdir -p "$(dirname "$target")"
  if [[ ! -f "$target" || $(stat -c %s "$target") -ne "$size" ]]; then
    if command -v aria2c >/dev/null 2>&1; then
      aria2c \
        --continue=true \
        --max-connection-per-server=8 \
        --split=8 \
        --min-split-size=8M \
        --file-allocation=none \
        --allow-overwrite=true \
        --auto-file-renaming=false \
        --summary-interval=30 \
        --dir="$(dirname "$target")" \
        --out="$(basename "$target")" \
        "$url"
    elif command -v curl >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
      python3 "$parallel_downloader" \
        "$url" "$target" "$size" --workers 8
    else
      wget --continue --output-document "$target" "$url"
    fi
  fi
  local actual_size
  actual_size=$(stat -c %s "$target")
  if [[ "$actual_size" -ne "$size" ]]; then
    echo "artifact size mismatch: $target expected=$size actual=$actual_size" >&2
    exit 2
  fi
  if [[ -n "$algorithm" && -n "$digest" ]]; then
    local actual_digest
    case "$algorithm" in
      sha256) actual_digest=$(sha256sum "$target" | cut -d' ' -f1) ;;
      md5) actual_digest=$(md5sum "$target" | cut -d' ' -f1) ;;
      *) echo "unsupported digest algorithm: $algorithm" >&2; exit 2 ;;
    esac
    if [[ "$actual_digest" != "$digest" ]]; then
      echo "artifact digest mismatch: $target" >&2
      exit 2
    fi
  fi
  sha256sum "$target" >> "$artifact_root/$model.sha256"
}

: > "$artifact_root/$model.sha256"

case "$model" in
  sigmadock)
    checkpoint="$weight_root/sigmadock/checkpoint.ckpt"
    download \
      https://github.com/alvaroprat97/sigmadock/releases/download/v0.1.0-beta/sample_checkpoint_0.ckpt \
      "$checkpoint" 242097065 sha256 \
      db15427ca349e6f1e5f894bff841112c7360384886aa472667d8011307cad382
    target="$bin_root/gnina-sigmadock-v1.3.2"
    download \
      https://github.com/gnina/gnina/releases/download/v1.3.2/gnina.1.3.2 \
      "$target" 1426790536 sha256 \
      5d33538324b40050a03aa262d51832837e0ea6cc100945abbd2d7b732589690e
    chmod +x "$target"
    echo "checkpoint_status=available_official_v0.1.0-beta" > "$artifact_root/sigmadock.status"
    ;;

  surfdock)
    src="$source_root/surfdock"
    checks=(
      "6705a8386cd7d8965a845ed38ae59edad17a4e8283fdfbcc5d25caf0f981ce5b model_weights/docking/best_ema_inference_epoch_model.pt"
      "7e057f220bac164ce6f6ab8ac6be5b647e6ce82c826f85926aebdf6c22cc02d9 model_weights/posepredict/best_model.pt"
      "b680af053e870b489ab01572d316eaa541116c92c6f94eb49f5bf150f148f369 model_weights/screen/best_model.pt"
    )
    for entry in "${checks[@]}"; do
      expected=${entry%% *}
      path=${entry#* }
      actual=$(sha256sum "$src/$path" | cut -d' ' -f1)
      [[ "$actual" == "$expected" ]] || { echo "SurfDock embedded weight mismatch: $path" >&2; exit 2; }
      sha256sum "$src/$path" >> "$artifact_root/$model.sha256"
    done
    ;;

  rldiff)
    cache="$weight_root/rldiff"
    download \
      https://github.com/plainerman/DiffDock-Pocket/releases/download/v1.0.0/score_model.zip \
      "$cache/score_model.zip" 281617940 sha256 \
      455d9a35c66f79d81b7887bd883d46d5207439a72e3410d7c0f69c4d3299f5ae
    download \
      https://github.com/plainerman/DiffDock-Pocket/releases/download/v1.0.0/confidence_model.zip \
      "$cache/confidence_model.zip" 17091458 sha256 \
      3cc3aa1ad6e1df3eda85eb81019246bb84f3217f4c679447453bc38a332d6f77
    download \
      https://github.com/oxpig/RLDiff/releases/download/v1.0.0/DD_Pocket_RL_score_model.pt \
      "$cache/DD_Pocket_RL_score_model/DD_Pocket_RL_score_model.pt" 303718544 sha256 \
      6cf7028c30735679814ca5c90aed28816b4e4a673280a5433db0609f6619f802
    extract_zip "$cache/score_model.zip" "$cache/v1.0.0"
    extract_zip "$cache/confidence_model.zip" "$cache/v1.0.0"
    [[ -f "$cache/v1.0.0/score_model/model_parameters.yml" ]] || { echo "RLDiff score model extraction failed" >&2; exit 2; }
    [[ -f "$cache/v1.0.0/confidence_model/model_parameters.yml" ]] || { echo "RLDiff confidence model extraction failed" >&2; exit 2; }
    gnina="$bin_root/gnina-rldiff-v1.0"
    download \
      https://github.com/gnina/gnina/releases/download/v1.0/gnina \
      "$gnina" 560059008 sha256 \
      9d0b8f22a07ee8132c8085f0356ebc3e28878b7d4a2dfa108e21b79def988c8b
    chmod +x "$gnina"
    ;;

  diffbindfr)
    src="$source_root/diffbindfr"
    archive="$weight_root/diffbindfr/weights.tar.gz"
    download \
      https://zenodo.org/api/records/10843568/files/weights.tar.gz/content \
      "$archive" 364487597 md5 f64bc5e495e24477d5cd63ca86eb4dcb
    extract_tar_gz "$archive" "$src/DiffBindFR"
    [[ -d "$src/DiffBindFR/weights" ]] || { echo "DiffBindFR weight extraction failed" >&2; exit 2; }
    ;;

  interformer)
    src="$source_root/interformer"
    archive="$weight_root/interformer/checkpoints.zip"
    download \
      https://zenodo.org/api/records/15694429/files/checkpoints.zip/content \
      "$archive" 1155810314 md5 3cf643713fa3148bed2d636bb224a81c
    extract_zip "$archive" "$src"
    [[ -d "$src/checkpoints/v0.2_energy_model" ]] || { echo "Interformer checkpoint extraction failed" >&2; exit 2; }
    ;;

  posebench-diffdock)
    src="$source_root/posebench/forks/DiffDock"
    archive="$weight_root/posebench-diffdock/diffdock_models.zip"
    download \
      https://github.com/gcorso/DiffDock/releases/download/v1.1/diffdock_models.zip \
      "$archive" 129825226 sha256 \
      5a95b6a1555be47ab1d6f0a8ffd25152f7fe32f5956005bb821e13e7a37d4a3d
    extract_zip "$archive" "$src/workdir/v1.1"
    [[ -d "$src/workdir/v1.1/score_model" && -d "$src/workdir/v1.1/confidence_model" ]] || {
      echo "PoseBench DiffDock model extraction failed" >&2
      exit 2
    }
    ;;

  posebench-fabind)
    EFFDOCK_REPO_ROOT="$repo_root" \
    EFFDOCK_PARALLEL_DOWNLOADER="$parallel_downloader" \
      bash "$fabind_downloader"
    ;;

  posebench-dynamicbind)
    src="$source_root/posebench/forks/DynamicBind"
    archive="$weight_root/posebench-dynamicbind/workdir.zip"
    download \
      https://zenodo.org/api/records/10137507/files/workdir.zip/content \
      "$archive" 236235944 md5 37889c86e162d6fe306d6972335e3607
    extract_zip "$archive" "$src"
    [[ -d "$src/workdir" ]] || { echo "DynamicBind model extraction failed" >&2; exit 2; }
    ;;

  posebench-p2rank)
    src="$source_root/posebench/forks/P2Rank"
    archive="$weight_root/posebench-p2rank/p2rank_2.4.2.tar.gz"
    download \
      https://github.com/rdk/p2rank/releases/download/2.4.2/p2rank_2.4.2.tar.gz \
      "$archive" 351679906 sha256 \
      2b836137a826a3d8a2ae5ab317923c90c525df4bed8eedefb93902364f5ce824
    extract_tar_gz "$archive" "$src"
    [[ -x "$src/p2rank_2.4.2/prank" ]] || { echo "P2Rank extraction failed" >&2; exit 2; }
    ;;

  *)
    echo "unknown artifact target: $model" >&2
    exit 2
    ;;
esac

date -u +%Y-%m-%dT%H:%M:%SZ > "$artifact_root/$model.complete"
echo "artifacts complete: $model"
