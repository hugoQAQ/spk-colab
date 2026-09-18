#!/bin/bash
# Minimal Drive -> VM copy for FRCNN logits extraction (VOC or BDD).
#
# Copies:
#   model/frcnn/{voc,bdd}_vanilla.pth
#   model/frcnn/frcnn_fx/          (Detectron2 FX yaml + utils; required to load the model)
#   data/id/...                    (train + val image tars)
#   data/ood/near_ood-000000.tar
#   data/ood/far_ood-000000.tar
#
# Usage (Colab VM, after Drive mount):
#   bash /content/mount_frcnn_logits.sh              # DATASET=voc (default)
#   DATASET=bdd bash /content/mount_frcnn_logits.sh
#
# From laptop:
#   colab upload -s roi mount_frcnn_logits.sh /content/mount_frcnn_logits.sh
#   colab upload -s roi frcnn_logits_extraction.py /content/spk/frcnn_logits_extraction.py
set -euo pipefail

DATASET="${DATASET:-voc}"
ASSETS=/content/drive/MyDrive/assets
SHARED="$ASSETS/shared"
DEST=/content/spk
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
MODEL_DIR="$DEST/model/frcnn"
ID_DIR="$DEST/data/id"
OOD_DIR="$DEST/data/ood"

case "$DATASET" in
  voc|bdd) ;;
  *)
    echo "unknown DATASET=$DATASET (use voc or bdd)"
    exit 2
    ;;
esac

LOGITS_DIR="$DEST/data/frcnn/$DATASET/logits"
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

copy_tree() {
  local src=$1 dest=$2
  if [ ! -d "$src" ]; then
    echo "MISSING $src (directory)"
    missing=$((missing + 1))
    return
  fi
  mkdir -p "$dest"
  echo
  echo "==> $(basename "$src")/ -> $dest"
  if command -v rsync >/dev/null 2>&1; then
    if ! stdbuf -oL rsync -ah --info=progress2 -- "$src"/ "$dest"/; then
      echo "FAILED $dest"
      missing=$((missing + 1))
      return
    fi
  else
    cp -a -- "$src"/. "$dest"/
  fi
  echo "ok $(basename "$src")/"
  copied=$((copied + 1))
}

copy_voc_images() {
  copy "$SHARED/datasets/id/voc/voc_yolo_train-*.tar"          "$ID_DIR/"
  copy "$SHARED/datasets/id/voc/voc_yolo_val-000000.tar"       "$ID_DIR/"
  copy "$SHARED/datasets/ood/near-ood-voc/near_ood-000000.tar" "$OOD_DIR/"
  copy "$SHARED/datasets/ood/far-ood/far_ood-000000.tar"       "$OOD_DIR/"
}

copy_bdd_images() {
  BDD_TRAIN_TAR="${BDD_TRAIN_TAR:-bdd_train_10k-*.tar}"
  copy "$SHARED/datasets/id/bdd/$BDD_TRAIN_TAR"                "$ID_DIR/"
  copy "$SHARED/datasets/id/bdd/bdd_val-000000.tar"            "$ID_DIR/"
  copy "$SHARED/datasets/ood/near-ood-bdd/near_ood-000000.tar" "$OOD_DIR/"
  copy "$SHARED/datasets/ood/far-ood/far_ood-000000.tar"       "$OOD_DIR/"
}

echo "mount_frcnn_logits: DATASET=$DATASET  DEST=$DEST"
echo "  model -> $MODEL_DIR/${DATASET}_vanilla.pth + frcnn_fx/"
echo "  out   -> $LOGITS_DIR/"

if [ "$DATASET" = voc ]; then
  copy "$SHARED/models/faster_rcnn/voc_vanilla.pth" "$MODEL_DIR/"
  copy_voc_images
else
  copy "$SHARED/models/faster_rcnn/bdd_vanilla.pth" "$MODEL_DIR/"
  copy_bdd_images
fi

fx_src=""
if [ -d "$SHARED/models/faster_rcnn/fx" ]; then
  fx_src="$SHARED/models/faster_rcnn/fx"
elif [ -d "/content/frcnn_fx" ]; then
  fx_src="/content/frcnn_fx"
elif [ -d "$HERE/frcnn_fx" ]; then
  fx_src="$HERE/frcnn_fx"
fi
if [ -z "$fx_src" ]; then
  echo "MISSING frcnn_fx (need Drive shared/models/faster_rcnn/fx, /content/frcnn_fx, or $HERE/frcnn_fx)"
  missing=$((missing + 1))
else
  copy_tree "$fx_src" "$MODEL_DIR/frcnn_fx"
fi

echo
du -sh "$DEST"
echo "copied $copied file(s), missing=$missing"
if [ "$missing" -eq 0 ]; then
  echo "MOUNT_FRCNN_LOGITS_OK"
else
  echo "MOUNT_FRCNN_LOGITS_FAILED missing=$missing"
  exit 1
fi
