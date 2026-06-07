#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TAG="${1:-compare_$(date +%Y%m%d_%H%M%S)}"
STEPS="${2:-800}"
DEVICE="${3:-cpu}"
BATCH_SIZE="${4:-2}"
NUM_WORKERS="${5:-0}"
INPUT_DIR="${6:-inputs/div2k/DIV2K_train_HR}"
TEST_DIR="${7:-inputs/chaos_batch}"
WM_TEMPLATE="${8:-assets/watermarks/luke.png}"

PLAIN_RUN="outputs/edge_wm_noise/${TAG}_plain"
ROBUST_RUN="outputs/edge_wm_noise/${TAG}_robust"

PLAIN_CLEAN="outputs/edge_wm_noise_test/${TAG}_plain_clean"
PLAIN_ATTACK="outputs/edge_wm_noise_test/${TAG}_plain_attack_all"
ROBUST_CLEAN="outputs/edge_wm_noise_test/${TAG}_robust_clean"
ROBUST_ATTACK="outputs/edge_wm_noise_test/${TAG}_robust_attack_all"

COMPARE_DIR="outputs/edge_wm_compare/${TAG}"
mkdir -p "$COMPARE_DIR"

echo "[1/6] Train plain model (no robust attacks)"
bash scripts/with_conda_gp.sh python experiments/edge_wm_noise_train.py \
  --input-dir "$INPUT_DIR" \
  --wm-template "$WM_TEMPLATE" \
  --outdir "$PLAIN_RUN" \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  --save-every 200 \
  --log-every 100 \
  --val-dir "$TEST_DIR" \
  --val-every 200 \
  --val-save-samples 4 \
  --no-robust \
  --no-progress

echo "[2/6] Train robust model (attack-aware)"
bash scripts/with_conda_gp.sh python experiments/edge_wm_noise_train.py \
  --input-dir "$INPUT_DIR" \
  --wm-template "$WM_TEMPLATE" \
  --outdir "$ROBUST_RUN" \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  --save-every 200 \
  --log-every 100 \
  --val-dir "$TEST_DIR" \
  --val-every 200 \
  --val-save-samples 4 \
  --robust \
  --lambda-attack 1.0 \
  --attack-prob 0.7 \
  --attack-crop-keep 0.7 \
  --attack-crop-prob 0.5 \
  --attack-mosaic-prob 0.3 \
  --attack-mosaic-min-block 4 \
  --attack-mosaic-max-block 16 \
  --attack-jpeg-prob 0.3 \
  --attack-jpeg-quality-min 40 \
  --attack-jpeg-quality-max 80 \
  --attack-resize-prob 0.3 \
  --attack-resize-min-scale 0.7 \
  --attack-resize-max-scale 1.3 \
  --attack-blur-prob 0.3 \
  --attack-blur-kernel 5 \
  --attack-blur-sigma-max 1.2 \
  --no-progress

echo "[3/6] Test plain model (clean)"
bash scripts/with_conda_gp.sh python experiments/edge_wm_noise_test.py \
  --input-dir "$TEST_DIR" \
  --wm-template "$WM_TEMPLATE" \
  --ckpt "${PLAIN_RUN}/ckpt_last.pt" \
  --outdir "$PLAIN_CLEAN" \
  --device "$DEVICE" \
  --save-samples 8 \
  --no-attack

echo "[4/6] Test plain model (attack all)"
bash scripts/with_conda_gp.sh python experiments/edge_wm_noise_test.py \
  --input-dir "$TEST_DIR" \
  --wm-template "$WM_TEMPLATE" \
  --ckpt "${PLAIN_RUN}/ckpt_last.pt" \
  --outdir "$PLAIN_ATTACK" \
  --device "$DEVICE" \
  --save-samples 8 \
  --attack --attack-mode all

echo "[5/6] Test robust model (clean + attack all)"
bash scripts/with_conda_gp.sh python experiments/edge_wm_noise_test.py \
  --input-dir "$TEST_DIR" \
  --wm-template "$WM_TEMPLATE" \
  --ckpt "${ROBUST_RUN}/ckpt_last.pt" \
  --outdir "$ROBUST_CLEAN" \
  --device "$DEVICE" \
  --save-samples 8 \
  --no-attack

bash scripts/with_conda_gp.sh python experiments/edge_wm_noise_test.py \
  --input-dir "$TEST_DIR" \
  --wm-template "$WM_TEMPLATE" \
  --ckpt "${ROBUST_RUN}/ckpt_last.pt" \
  --outdir "$ROBUST_ATTACK" \
  --device "$DEVICE" \
  --save-samples 8 \
  --attack --attack-mode all

echo "[6/6] Build compare report"
bash scripts/with_conda_gp.sh python - "$PLAIN_CLEAN/summary.json" "$PLAIN_ATTACK/summary.json" "$ROBUST_CLEAN/summary.json" "$ROBUST_ATTACK/summary.json" "$COMPARE_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

plain_clean = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
plain_attack = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
robust_clean = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
robust_attack = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))
out_dir = Path(sys.argv[5])
out_dir.mkdir(parents=True, exist_ok=True)

def metric(d, name):
    return float(d.get(name, {}).get("mean", float("nan")))

report = {
    "plain": {
        "clean": {
            "host_psnr": metric(plain_clean, "host_psnr"),
            "host_ssim": metric(plain_clean, "host_ssim"),
            "wm_psnr": metric(plain_clean, "wm_psnr"),
            "wm_ssim": metric(plain_clean, "wm_ssim"),
        },
        "attack_all": {
            "wm_psnr_attack": metric(plain_attack, "wm_psnr_attack"),
            "wm_ssim_attack": metric(plain_attack, "wm_ssim_attack"),
        },
    },
    "robust": {
        "clean": {
            "host_psnr": metric(robust_clean, "host_psnr"),
            "host_ssim": metric(robust_clean, "host_ssim"),
            "wm_psnr": metric(robust_clean, "wm_psnr"),
            "wm_ssim": metric(robust_clean, "wm_ssim"),
        },
        "attack_all": {
            "wm_psnr_attack": metric(robust_attack, "wm_psnr_attack"),
            "wm_ssim_attack": metric(robust_attack, "wm_ssim_attack"),
        },
    },
    "delta_robust_minus_plain": {
        "clean_host_psnr": metric(robust_clean, "host_psnr") - metric(plain_clean, "host_psnr"),
        "clean_host_ssim": metric(robust_clean, "host_ssim") - metric(plain_clean, "host_ssim"),
        "clean_wm_psnr": metric(robust_clean, "wm_psnr") - metric(plain_clean, "wm_psnr"),
        "clean_wm_ssim": metric(robust_clean, "wm_ssim") - metric(plain_clean, "wm_ssim"),
        "attack_wm_psnr": metric(robust_attack, "wm_psnr_attack") - metric(plain_attack, "wm_psnr_attack"),
        "attack_wm_ssim": metric(robust_attack, "wm_ssim_attack") - metric(plain_attack, "wm_ssim_attack"),
    },
}

(out_dir / "compare_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def f(x):
    if x != x or math.isinf(x):
        return "NaN"
    return f"{x:.4f}"

md = []
md.append("# Robust vs Plain 对比结果")
md.append("")
md.append("| 指标 | Plain | Robust | Delta (Robust-Plain) |")
md.append("|---|---:|---:|---:|")
rows = [
    ("clean host_psnr", report["plain"]["clean"]["host_psnr"], report["robust"]["clean"]["host_psnr"], report["delta_robust_minus_plain"]["clean_host_psnr"]),
    ("clean host_ssim", report["plain"]["clean"]["host_ssim"], report["robust"]["clean"]["host_ssim"], report["delta_robust_minus_plain"]["clean_host_ssim"]),
    ("clean wm_psnr", report["plain"]["clean"]["wm_psnr"], report["robust"]["clean"]["wm_psnr"], report["delta_robust_minus_plain"]["clean_wm_psnr"]),
    ("clean wm_ssim", report["plain"]["clean"]["wm_ssim"], report["robust"]["clean"]["wm_ssim"], report["delta_robust_minus_plain"]["clean_wm_ssim"]),
    ("attack wm_psnr", report["plain"]["attack_all"]["wm_psnr_attack"], report["robust"]["attack_all"]["wm_psnr_attack"], report["delta_robust_minus_plain"]["attack_wm_psnr"]),
    ("attack wm_ssim", report["plain"]["attack_all"]["wm_ssim_attack"], report["robust"]["attack_all"]["wm_ssim_attack"], report["delta_robust_minus_plain"]["attack_wm_ssim"]),
]
for name, p, r, d in rows:
    md.append(f"| {name} | {f(p)} | {f(r)} | {f(d)} |")
md.append("")
md.append("说明：attack 指 `--attack --attack-mode all` 全攻击链测试。")

(out_dir / "compare_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(str(out_dir / "compare_summary.json"))
print(str(out_dir / "compare_summary.md"))
PY

echo "Done."
echo "plain run:   $PLAIN_RUN"
echo "robust run:  $ROBUST_RUN"
echo "compare dir: $COMPARE_DIR"
