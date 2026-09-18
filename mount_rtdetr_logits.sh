#!/bin/bash
# Minimal Drive -> VM copy for RT-DETR logits extraction (VOC or BDD).
#
# Copies:
#   model/rtdetr/{voc,bdd}_vanilla.pt
#   data/id/...                    (train + val image tars)
#   data/ood/near_ood_voc-000000.tar
#   data/ood/near_ood_bdd-000000.tar
#   data/ood/far_ood-000000.tar
#
# Usage (Colab VM, after Drive mount):
#   bash /content/mount_rtdetr_logits.sh              # DATASET=voc (default)
#   DATASET=bdd bash /content/mount_rtdetr_logits.sh
#
# From laptop:
#   colab upload -s roi mount_rtdetr_logits.sh /content/mount_rtdetr_logits.sh
#   colab upload -s roi rtdetr_logits_extraction.py /content/spk/rtdetr_logits_extraction.py
set -euo pipefail

DATASET="${DATASET:-voc}"
ASSETS=/content/drive/MyDrive/assets
SHARED="$ASSETS/shared"
DEST=/content/spk
MODEL_DIR="$DEST/model/rtdetr"
ID_DIR="$DEST/data/id"
OOD_DIR="$DEST/data/ood"

case "$DATASET" in
  voc|bdd) ;;
  *)
    echo "unknown DATASET=$DATASET (use voc or bdd)"
    exit 2
    ;;
esac

LOGITS_DIR="$DEST/data/rtdetr/$DATASET/logits"
mkdir -p "$MODEL_DIR" "$ID_DIR" "$OOD_DIR" "$LOGITS_DIR"

missing=0
copied=0

human_size() {
  du -h "$1" | awk '{print $1}'
}

copy_one() {
  local src=$1 dest_dir=$2 dest_name=${3:-}
  local size dest landed
  dest_dir="${dest_dir%/}"
  mkdir -p "$dest_dir"
  if [ -d "$src" ]; then
    local inner wanted
    wanted=${dest_name:-$(basename "$src")}
    inner=$(find "$src" -type f \( -name "$wanted" -o -name '*.pt' -o -name '*.pth' -o -name '*.tar' \) | head -n 1)
    if [ -z "$inner" ]; then
      echo "MISSING $src (directory with no file inside)"
      missing=$((missing + 1))
      return
    fi
    echo "note: $src is a directory; using $inner"
    src=$inner
  fi
  if [ ! -f "$src" ]; then
    echo "MISSING $src (not a file)"
    missing=$((missing + 1))
    return
  fi
  if [ -z "$dest_name" ]; then
    dest_name=$(basename "$src")
  fi
  dest="$dest_dir/$dest_name"
  size=$(human_size "$src")
  echo
  echo "==> $dest_name  ($size) -> $dest"
  if [ -d "$dest" ]; then
    echo "note: replacing directory $dest with a file"
    rm -rf "$dest"
  fi
  if command -v rsync >/dev/null 2>&1; then
    if ! stdbuf -oL rsync -ah --info=progress2 -- "$src" "$dest_dir/"; then
      echo "FAILED $dest"
      missing=$((missing + 1))
      return
    fi
  else
    cp -f -- "$src" "$dest"
  fi
  landed="$dest_dir/$(basename "$src")"
  if [ "$landed" != "$dest" ]; then
    mv -f "$landed" "$dest"
  fi
  if [ ! -f "$dest" ]; then
    echo "FAILED $dest (not a file after copy)"
    missing=$((missing + 1))
    return
  fi
  echo "ok $dest_name"
  copied=$((copied + 1))
}

copy() {
  local matches=( $1 )
  if [ ! -e "${matches[0]}" ]; then
    echo "MISSING $1"
    missing=$((missing + 1))
    return
  fi
  local src
  for src in "${matches[@]}"; do
    copy_one "$src" "$2"
  done
}

copy_as() {
  copy_one "$1" "$(dirname "$2")" "$(basename "$2")"
}

copy_near_ood_tars() {
  # Distinct local names so VOC + BDD mounts can coexist under data/ood/.
  copy_as "$SHARED/datasets/ood/near-ood-voc/near_ood-000000.tar" "$OOD_DIR/near_ood_voc-000000.tar"
  copy_as "$SHARED/datasets/ood/near-ood-bdd/near_ood-000000.tar" "$OOD_DIR/near_ood_bdd-000000.tar"
  copy "$SHARED/datasets/ood/far-ood/far_ood-000000.tar"         "$OOD_DIR/"
}

copy_voc_images() {
  copy "$SHARED/datasets/id/voc/voc_yolo_train-*.tar"    "$ID_DIR/"
  copy "$SHARED/datasets/id/voc/voc_yolo_val-000000.tar" "$ID_DIR/"
  copy_near_ood_tars
}

copy_bdd_images() {
  BDD_TRAIN_TAR="${BDD_TRAIN_TAR:-bdd_train_10k-*.tar}"
  copy "$SHARED/datasets/id/bdd/$BDD_TRAIN_TAR"     "$ID_DIR/"
  copy "$SHARED/datasets/id/bdd/bdd_val-000000.tar" "$ID_DIR/"
  copy_near_ood_tars
}

echo "mount_rtdetr_logits: DATASET=$DATASET  DEST=$DEST"
echo "  model -> $MODEL_DIR/${DATASET}_vanilla.pt"
echo "  out   -> $LOGITS_DIR/"

if [ "$DATASET" = voc ]; then
  copy "$SHARED/models/rtdetr/voc_vanilla.pt" "$MODEL_DIR/"
  copy_voc_images
else
  copy "$SHARED/models/rtdetr/bdd_vanilla.pt" "$MODEL_DIR/"
  copy_bdd_images
fi

echo
du -sh "$DEST"
echo "copied $copied file(s), missing=$missing"
if [ "$missing" -eq 0 ]; then
  echo "MOUNT_RTDETR_LOGITS_OK"
else
  echo "MOUNT_RTDETR_LOGITS_FAILED missing=$missing"
  exit 1
fi
