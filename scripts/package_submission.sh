#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TAG="${1:-$(date +%Y%m%d)}"
MODE="${2:-default}"
PKG_NAME="GraduationProject_submission_${TAG}"
if [[ "$MODE" == "with_dataset" ]]; then
  PKG_NAME="${PKG_NAME}_with_dataset"
fi
OUT_DIR="$REPO_ROOT/deliverables"
ARCHIVE_PATH="$OUT_DIR/${PKG_NAME}.tar.gz"
LIST_PATH="$OUT_DIR/${PKG_NAME}_filelist.txt"
SHA256_PATH="$OUT_DIR/${PKG_NAME}.sha256"

mkdir -p "$OUT_DIR"

if [[ -e "$ARCHIVE_PATH" || -e "$LIST_PATH" || -e "$SHA256_PATH" ]]; then
  echo "target already exists under deliverables/: ${PKG_NAME}" >&2
  exit 2
fi

ITEMS=(
  README.md
  DEMO.md
  app_demo.py
  config.toml
  requirements.txt
  environment.yml
  src
  experiments
  scripts
  assets/watermarks
  assets/fonts
  inputs/chaos_batch
  outputs/experiment_reports
  outputs/thesis_figures
  outputs/edge_wm_compare/compare_r1_20260411/compare_summary.json
  outputs/edge_wm_compare/compare_r1_20260411/compare_summary.md
  outputs/edge_wm_ablation/ablation_r1_20260411/ablation_summary.json
  outputs/edge_wm_ablation/ablation_r1_20260411/ablation_summary.md
  outputs/edge_wm_noise/y_only_fresh_20260402/ckpt_last.pt
  outputs/edge_wm_noise/y_only_fresh_20260402/RECOVERY_NOTE.md
  outputs/edge_wm_noise/y_only_fresh_20260402/metrics.jsonl
  outputs/edge_wm_noise/y_only_fresh_20260402/train_summary.json
  outputs/edge_wm_noise/y_only_fresh_20260402/val_metrics.jsonl
)

if [[ "$MODE" == "with_dataset" ]]; then
  ITEMS+=(
    inputs/div2k/DIV2K_train_HR
  )
elif [[ "$MODE" != "default" ]]; then
  echo "unknown mode: $MODE" >&2
  echo "allowed: default | with_dataset" >&2
  exit 2
fi

for item in "${ITEMS[@]}"; do
  if [[ ! -e "$item" ]]; then
    echo "missing required path: $item" >&2
    exit 2
  fi
done

tar \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  -czf "$ARCHIVE_PATH" \
  --transform "s,^,${PKG_NAME}/," \
  "${ITEMS[@]}"

tar -tzf "$ARCHIVE_PATH" > "$LIST_PATH"

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$ARCHIVE_PATH" > "$SHA256_PATH"
fi

echo "$ARCHIVE_PATH"
echo "$LIST_PATH"
if [[ -f "$SHA256_PATH" ]]; then
  echo "$SHA256_PATH"
fi
