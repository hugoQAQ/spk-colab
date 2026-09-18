#!/bin/bash
# Runs ON the Colab VM. Upload it, then execute it as a file:
#   colab upload -s spk mount_data.sh /content/mount_data.sh
# (piping into `colab console` does not work: tmux echoes the script instead
#  of running it)
# Copies the assets run.py needs from Drive to local disk, because reading the
# tars and training_data.pt straight off Drive is far slower than one bulk copy.
#
# Layout under DEST=/content/spk:
#   model/{yolo,frcnn}/{voc,bdd}_vanilla.{pt,pth}
#   data/id , data/ood                          (shared image tars; gt.json is built by run.py)
#   data/{detector}/{dataset}/training_data.pt
#   data/{detector}/{dataset}/{roi,native_knn,concept_head_ood}/
#     copied from Drive experiments/{detector}-{dataset}/ when present so
#     run.py can skip extract / head training
#
# Usage:
#   DETECTOR=yolo DATASET=voc bash mount_data.sh    # default
#   DETECTOR=yolo DATASET=bdd bash mount_data.sh
#   DETECTOR=frcnn DATASET=voc bash mount_data.sh
#
# Expects on Drive (under MyDrive/assets/):
#   shared/models/yolo/{voc,bdd}_vanilla.pt
#   shared/models/faster_rcnn/voc_vanilla.pth
#   shared/datasets/id/{voc,bdd}/...  (bdd train default: bdd_train_10k-*.tar)
#   shared/datasets/ood/{near-ood-voc,near-ood-bdd,far-ood}/...
#   semantic_training_data/yolo-voc.pt | yolo-bdd.pt | frcnn_voc.pt
# Optional on Drive (under MyDrive/experiments/{yolo-voc,yolo-bdd,frcnn-voc}/):
#   roi/  native_knn/  concept_head_ood/
# FRCNN also needs Detectron2 FX yaml+utils at one of:
#   shared/models/faster_rcnn/fx/
#   /content/frcnn_fx/  (upload with run.py)
#   $(dirname mount_data.sh)/frcnn_fx/
#
# Progress: rsync --info=progress2 (percent, MB/s, ETA). For a live bar through
# `colab exec`, do NOT capture stdout — print as you go. A driver that uses
# subprocess.run(..., capture_output=True) only shows the bar after each file.
DETECTOR="${DETECTOR:-yolo}"
DATASET="${DATASET:-voc}"
ASSETS=/content/drive/MyDrive/assets
SHARED="$ASSETS/shared"
SEMANTIC="$ASSETS/semantic_training_data"
EXPERIMENTS=/content/drive/MyDrive/experiments
DEST=/content/spk
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

case "$DETECTOR" in
  yolo|frcnn) ;;
  *)
    echo "unknown DETECTOR=$DETECTOR (use yolo or frcnn)"
    exit 2
    ;;
esac
case "$DATASET" in
  voc|bdd) ;;
  *)
    echo "unknown DATASET=$DATASET (use voc or bdd)"
    exit 2
    ;;
esac
if [ "$DETECTOR" = frcnn ] && [ "$DATASET" != voc ]; then
  echo "frcnn currently only supports DATASET=voc"
  exit 2
fi

EXP_DIR="$EXPERIMENTS/${DETECTOR}-${DATASET}"

echo "DETECTOR=$DETECTOR  DATASET=$DATASET  DEST=$DEST"
echo "  model -> $DEST/model/$DETECTOR/"
echo "  arch  -> $DEST/data/$DETECTOR/$DATASET/"
echo "  exp   -> $EXP_DIR/{roi,native_knn,concept_head_ood}/  (optional reuse)"

MODEL_DIR="$DEST/model/$DETECTOR"
ARCH_DIR="$DEST/data/$DETECTOR/$DATASET"
mkdir -p "$MODEL_DIR" "$ARCH_DIR" "$DEST/data/id" "$DEST/data/ood"

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
  # Drive/rsync can land a folder named *.pt; flatten to the real file.
  if [ -d "$src" ]; then
    local inner wanted
    wanted=${dest_name:-$(basename "$src")}
    inner=$(find "$src" -type f \( -name "$wanted" -o -name 'training_data.pt' -o -name '*.pt' -o -name '*.pth' \) | head -n 1)
    if [ -z "$inner" ]; then
      echo "MISSING $src (directory with no .pt inside)"
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
  # A leftover directory with this name makes rsync try to chdir into it.
  if [ -d "$dest" ]; then
    echo "note: replacing directory $dest with a file"
    rm -rf "$dest"
  fi
  # Always rsync into an existing directory (trailing slash). Copying onto a
  # not-yet-created dest file makes rsync 3.2 treat dest as a directory.
  if command -v rsync >/dev/null 2>&1; then
    if ! stdbuf -oL rsync -ah --info=progress2 -- "$src" "$dest_dir/"; then
      echo "FAILED $dest"
      missing=$((missing + 1))
      return
    fi
  elif command -v pv >/dev/null 2>&1; then
    pv -pterb "$src" > "$dest"
  else
    echo "  (no rsync/pv; silent cp)"
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

copy() {  # copy <glob-or-path> <dest-dir>
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

copy_as() {  # copy_as <src> <dest-file>
  copy_one "$1" "$(dirname "$2")" "$(basename "$2")"
}

copy_first() {  # copy_first <dest-file> <src> [src...]
  local dest=$1 src
  shift
  for src in "$@"; do
    if [ -e "$src" ]; then
      copy_as "$src" "$dest"
      return
    fi
  done
  echo "MISSING (tried $*)"
  missing=$((missing + 1))
}

copy_tree() {  # copy_tree <src-dir> <dest-dir>
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
  copy "$SHARED/datasets/id/voc/voc_yolo_train-*.tar"          "$DEST/data/id/"
  copy "$SHARED/datasets/id/voc/voc_yolo_val-000000.tar"       "$DEST/data/id/"
  copy "$SHARED/datasets/ood/near-ood-voc/near_ood-000000.tar" "$DEST/data/ood/"
  copy "$SHARED/datasets/ood/far-ood/far_ood-000000.tar"       "$DEST/data/ood/"
}

if [ "$DETECTOR" = yolo ] && [ "$DATASET" = voc ]; then
  copy "$SHARED/models/yolo/voc_vanilla.pt"                    "$MODEL_DIR/"
  copy_voc_images
  copy_as "$SEMANTIC/yolo-voc.pt"                              "$ARCH_DIR/training_data.pt"
elif [ "$DETECTOR" = yolo ] && [ "$DATASET" = bdd ]; then
  copy "$SHARED/models/yolo/bdd_vanilla.pt"                    "$MODEL_DIR/"
  # Default BDD id-train: 10k subset tar (override with BDD_TRAIN_TAR=bdd_train-*.tar for full 30k)
  BDD_TRAIN_TAR="${BDD_TRAIN_TAR:-bdd_train_10k-*.tar}"
  copy "$SHARED/datasets/id/bdd/$BDD_TRAIN_TAR"                "$DEST/data/id/"
  copy "$SHARED/datasets/id/bdd/bdd_val-000000.tar"            "$DEST/data/id/"
  copy "$SHARED/datasets/ood/near-ood-bdd/near_ood-000000.tar" "$DEST/data/ood/"
  copy "$SHARED/datasets/ood/far-ood/far_ood-000000.tar"       "$DEST/data/ood/"
  copy_as "$SEMANTIC/yolo-bdd.pt"                              "$ARCH_DIR/training_data.pt"
else
  copy "$SHARED/models/faster_rcnn/voc_vanilla.pth"            "$MODEL_DIR/"
  copy_voc_images
  copy_first "$ARCH_DIR/training_data.pt" \
    "$SEMANTIC/frcnn_voc.pt" "$SEMANTIC/frcnn-voc.pt"
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
fi

echo
reuse_stages=""
if [ -d "$EXP_DIR" ]; then
  echo "reusing experiment caches from $EXP_DIR"
  for stage in roi native_knn concept_head_ood; do
    src="$EXP_DIR/$stage"
    if [ -d "$src" ]; then
      copy_tree "$src" "$ARCH_DIR/$stage"
      reuse_stages="${reuse_stages:+$reuse_stages, }$stage"
    else
      echo "skip $stage (not in $EXP_DIR; run.py will build it)"
    fi
  done
else
  echo "skip experiment reuse (no $EXP_DIR); run.py will build roi/native_knn/concept_head_ood"
fi

echo
echo "gt.json is not copied; run.py stage A builds $DEST/data/id/gt.json from train tars"
if [ ! -f "$ARCH_DIR/training_data.pt" ]; then
  echo "MISSING $ARCH_DIR/training_data.pt (need a file, not a directory)"
  missing=$((missing + 1))
fi
echo "checkpoint -> $MODEL_DIR/"
if [ -n "$reuse_stages" ]; then
  echo "reused     -> $ARCH_DIR/{$reuse_stages}/"
else
  echo "outputs    -> $ARCH_DIR/{roi,native_knn,concept_head_ood}/  (created by run.py)"
fi
echo
du -sh "$DEST"
echo "copied $copied file(s), missing=$missing"
if [ "$missing" -eq 0 ]; then
  echo "MOUNT_DATA_OK"
else
  echo "MOUNT_DATA_FAILED missing=$missing (SHARED=$SHARED)"
fi
