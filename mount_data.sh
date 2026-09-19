#!/bin/bash
# Runs ON the Colab VM. Upload it, then execute it as a file:
#   colab upload -s spk mount_data.sh /content/mount_data.sh
# (piping into `colab console` does not work: tmux echoes the script instead
#  of running it)
# Copies the assets run.py needs from Drive to local disk, because reading the
# tars and training_data.pt straight off Drive is far slower than one bulk copy.
#
# Layout under DEST=/content/spk:
#   model/{yolo,frcnn,rtdetr}/{voc,bdd}_vanilla.{pt,pth}
#   data/id , data/ood                          (shared image tars + gt_{dataset}.json)
#   data/{detector}/{dataset}/training_data.pt
#   data/{detector}/{dataset}/{roi,native_knn,concept_head_ood}/
#     copied from Drive experiments/{detector}-{dataset}/ when present so
#     run.py can skip extract / head training
#
# Usage (env or flags; flags win):
#   DETECTOR=yolo DATASET=bdd bash mount_data.sh
#   bash mount_data.sh --detector yolo --dataset bdd
#   bash mount_data.sh --detector yolo --dataset bdd --eval-only
#     copies concept_head_ood (+ optional native_knn, gt json); skip roi/images/weights
# Then:
#   python run.py --detector yolo --dataset bdd --stage D
#
# Expects on Drive (under MyDrive/assets/):
#   shared/models/yolo/{voc,bdd}_vanilla.pt
#   shared/models/rtdetr/{voc,bdd}_vanilla.pt
#   shared/models/faster_rcnn/{voc,bdd}_vanilla.pth
#   shared/datasets/id/{voc,bdd}/...  (bdd train default: bdd_train_10k-*.tar)
#   shared/datasets/id/voc/gt_voc.json  (for eval-only SPK baselines; also gt_bdd.json)
#   shared/datasets/ood/{near-ood-voc,near-ood-bdd,far-ood}/...
#   semantic_training_data/{detector}-{dataset}.pt  (flat files only; also {detector}_{dataset}.pt)
# Optional on Drive (under MyDrive/experiments/{detector}-{dataset}/):
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
EVAL_ONLY=0
ASSETS=/content/drive/MyDrive/assets
SHARED="$ASSETS/shared"
SEMANTIC="$ASSETS/semantic_training_data"
EXPERIMENTS=/content/drive/MyDrive/experiments
DEST=/content/spk
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

while [ $# -gt 0 ]; do
  case "$1" in
    --detector) DETECTOR="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    --eval-only) EVAL_ONLY=1; shift ;;
    -h|--help)
      echo "Usage: $0 [--detector yolo|frcnn|rtdetr] [--dataset voc|bdd] [--eval-only] [--dest /content/spk]"
      echo "  or: DETECTOR=yolo DATASET=bdd $0"
      echo "SPK eval after mount: python run.py --detector \$DETECTOR --dataset \$DATASET --stage D"
      exit 0
      ;;
    *)
      echo "unknown argument: $1 (use --detector --dataset --eval-only --dest)"
      exit 2
      ;;
  esac
done

case "$DETECTOR" in
  yolo|frcnn|rtdetr) ;;
  *)
    echo "unknown DETECTOR=$DETECTOR (use yolo, frcnn, or rtdetr)"
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

EXP_DIR="$EXPERIMENTS/${DETECTOR}-${DATASET}"

echo "DETECTOR=$DETECTOR  DATASET=$DATASET  DEST=$DEST  EVAL_ONLY=$EVAL_ONLY"
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
  if [ -d "$src" ]; then
    echo "MISSING $src (expected a flat .pt file under semantic_training_data/, not a folder)"
    missing=$((missing + 1))
    return
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

# Drive sometimes stores a nested copy (experiments/frcnn-voc/roi/roi/*.pt).
# run.py expects files at data/{detector}/{dataset}/roi/{id_train,id_val,...}.pt.
unwrap_stage_src() {
  local src=$1 stage=$2 nested="$1/$2"
  if [ -d "$nested" ]; then
    local top_files nested_files
    top_files=$(find "$src" -maxdepth 1 -type f | head -n 1)
    nested_files=$(find "$nested" -maxdepth 2 -type f | head -n 1)
    if [ -z "$top_files" ] && [ -n "$nested_files" ]; then
      echo "note: unwrapping nested $nested/"
      src=$nested
    fi
  fi
  printf '%s' "$src"
}

copy_voc_images() {
  copy "$SHARED/datasets/id/voc/voc_yolo_train-*.tar"          "$DEST/data/id/"
  copy "$SHARED/datasets/id/voc/voc_yolo_val-000000.tar"       "$DEST/data/id/"
  copy_as "$SHARED/datasets/ood/near-ood-voc/near_ood-000000.tar" "$DEST/data/ood/near_ood_voc-000000.tar"
  copy "$SHARED/datasets/ood/far-ood/far_ood-000000.tar"       "$DEST/data/ood/"
}

copy_bdd_images() {
  BDD_TRAIN_TAR="${BDD_TRAIN_TAR:-bdd_train_10k-*.tar}"
  copy "$SHARED/datasets/id/bdd/$BDD_TRAIN_TAR"                "$DEST/data/id/"
  copy "$SHARED/datasets/id/bdd/bdd_val-000000.tar"            "$DEST/data/id/"
  copy_as "$SHARED/datasets/ood/near-ood-bdd/near_ood-000000.tar" "$DEST/data/ood/near_ood_bdd-000000.tar"
  copy "$SHARED/datasets/ood/far-ood/far_ood-000000.tar"       "$DEST/data/ood/"
}

copy_frcnn_fx() {
  local fx_src=""
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
  elif [ "$fx_src" = "$HERE/frcnn_fx" ] && [ ! -e "$MODEL_DIR/frcnn_fx" ]; then
    echo
    echo "==> frcnn_fx/ (bundled) -> symlink $MODEL_DIR/frcnn_fx"
    ln -sfn "$fx_src" "$MODEL_DIR/frcnn_fx"
    echo "ok frcnn_fx/ (symlink)"
    copied=$((copied + 1))
  elif [ "$fx_src" = "$HERE/frcnn_fx" ] && [ -L "$MODEL_DIR/frcnn_fx" ]; then
    echo "ok frcnn_fx/ (existing symlink)"
    copied=$((copied + 1))
  else
    copy_tree "$fx_src" "$MODEL_DIR/frcnn_fx"
  fi
}

copy_gt_index() {
  local dest="$DEST/data/id/gt_${DATASET}.json"
  if [ -f "$dest" ]; then
    echo "ok gt_${DATASET}.json (already at $dest)"
    copied=$((copied + 1))
    return
  fi
  local src
  for src in \
    "$EXP_DIR/gt_${DATASET}.json" \
    "$EXP_DIR/gt.json" \
    "$ARCH_DIR/gt_${DATASET}.json" \
    "$SHARED/datasets/id/${DATASET}/gt_${DATASET}.json" \
    "$SHARED/datasets/id/${DATASET}/gt.json"
  do
    if [ -f "$src" ]; then
      copy_as "$src" "$dest"
      return
    fi
  done
  echo "no gt_${DATASET}.json on Drive; need local $dest (run.py stage A) for TP-filtered SPK eval"
  if [ "$EVAL_ONLY" -eq 1 ]; then
    missing=$((missing + 1))
  fi
}

if [ "$EVAL_ONLY" -eq 0 ]; then
  if [ "$DATASET" = voc ]; then
    copy_voc_images
  else
    copy_bdd_images
  fi

  if [ "$DETECTOR" = yolo ]; then
    copy "$SHARED/models/yolo/${DATASET}_vanilla.pt"             "$MODEL_DIR/"
  elif [ "$DETECTOR" = rtdetr ]; then
    copy "$SHARED/models/rtdetr/${DATASET}_vanilla.pt"           "$MODEL_DIR/"
  else
    copy "$SHARED/models/faster_rcnn/${DATASET}_vanilla.pth"     "$MODEL_DIR/"
    copy_frcnn_fx
  fi

  copy_first "$ARCH_DIR/training_data.pt" \
    "$SEMANTIC/${DETECTOR}-${DATASET}.pt" \
    "$SEMANTIC/${DETECTOR}_${DATASET}.pt"
else
  echo "eval-only: skip images, detector weights, and training_data.pt"
fi

echo
reuse_stages=""
if [ "$EVAL_ONLY" -eq 1 ]; then
  echo "eval-only: concept_head_ood required; native_knn optional (skip roi — not needed for stage D or rescore-from-csv)"
  for stage in native_knn concept_head_ood; do
    src="$EXP_DIR/$stage"
    if [ -d "$src" ]; then
      src=$(unwrap_stage_src "$src" "$stage")
      copy_tree "$src" "$ARCH_DIR/$stage"
      reuse_stages="${reuse_stages:+$reuse_stages, }$stage"
    elif [ -d "$ARCH_DIR/$stage" ] && [ -n "$(find "$ARCH_DIR/$stage" -mindepth 1 -print -quit 2>/dev/null)" ]; then
      echo "ok $stage (already at $ARCH_DIR/$stage)"
      reuse_stages="${reuse_stages:+$reuse_stages, }$stage"
      copied=$((copied + 1))
    elif [ "$stage" = concept_head_ood ]; then
      echo "MISSING $EXP_DIR/concept_head_ood (and no local $ARCH_DIR/concept_head_ood)"
      missing=$((missing + 1))
    else
      echo "skip $stage (not on Drive; ok if activations.csv already has native_knn)"
    fi
  done
elif [ -d "$EXP_DIR" ]; then
  echo "reusing experiment caches from $EXP_DIR"
  for stage in roi native_knn concept_head_ood; do
    src="$EXP_DIR/$stage"
    if [ -d "$src" ]; then
      src=$(unwrap_stage_src "$src" "$stage")
      copy_tree "$src" "$ARCH_DIR/$stage"
      reuse_stages="${reuse_stages:+$reuse_stages, }$stage"
    else
      echo "skip $stage (not in $EXP_DIR; run.py will build it)"
    fi
  done
else
  echo "skip experiment reuse (no $EXP_DIR); run.py will build roi/native_knn/concept_head_ood"
fi

copy_gt_index

echo
if [ "$EVAL_ONLY" -eq 0 ] && [ ! -f "$ARCH_DIR/training_data.pt" ]; then
  echo "MISSING $ARCH_DIR/training_data.pt (need a file, not a directory)"
  missing=$((missing + 1))
fi
echo "checkpoint -> $MODEL_DIR/"
if [ -n "$reuse_stages" ]; then
  echo "reused     -> $ARCH_DIR/{$reuse_stages}/"
else
  echo "outputs    -> $ARCH_DIR/{roi,native_knn,concept_head_ood}/  (created by run.py)"
fi
echo "gt         -> $DEST/data/id/gt_${DATASET}.json"
echo
echo "SPK baselines eval (after MOUNT_DATA_OK):"
echo "  python run.py --detector $DETECTOR --dataset $DATASET --stage D"
echo
du -sh "$DEST"
echo "copied $copied file(s), missing=$missing"
if [ "$missing" -eq 0 ]; then
  echo "MOUNT_DATA_OK"
else
  echo "MOUNT_DATA_FAILED missing=$missing (SHARED=$SHARED)"
fi
