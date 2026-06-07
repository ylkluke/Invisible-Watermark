#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TAG="${1:-ablation_$(date +%Y%m%d_%H%M%S)}"
STEPS="${2:-800}"
DEVICE="${3:-cpu}"
BATCH_SIZE="${4:-2}"
NUM_WORKERS="${5:-0}"
INPUT_DIR="${6:-inputs/div2k/DIV2K_train_HR}"
TEST_DIR="${7:-inputs/chaos_batch}"
WM_TEMPLATE="${8:-assets/watermarks/luke.png}"
# comma-separated subset is supported for quick smoke runs.
# Add A6_CBAM_Full explicitly to compare the full robust model with CBAM attention.
VARIANTS_CSV="${9:-A0_Full,A1_NoRobust,A2_NoNoiseReg,A3_NoHVS,A4_NoPayload,A5_NoSSIMinWM}"

IFS=',' read -r -a VARIANTS <<< "$VARIANTS_CSV"

AB_OUT="outputs/edge_wm_ablation/${TAG}"
mkdir -p "$AB_OUT"

common_attack_args=(
  --lambda-attack 1.0
  --attack-prob 0.7
  --attack-crop-keep 0.7
  --attack-crop-prob 0.5
  --attack-mosaic-prob 0.3
  --attack-mosaic-min-block 4
  --attack-mosaic-max-block 16
  --attack-jpeg-prob 0.3
  --attack-jpeg-quality-min 40
  --attack-jpeg-quality-max 80
  --attack-resize-prob 0.3
  --attack-resize-min-scale 0.7
  --attack-resize-max-scale 1.3
  --attack-blur-prob 0.3
  --attack-blur-kernel 5
  --attack-blur-sigma-max 1.2
)

for v in "${VARIANTS[@]}"; do
  RUN_DIR="outputs/edge_wm_noise/${TAG}_${v}"
  TEST_CLEAN="outputs/edge_wm_noise_test/${TAG}_${v}_clean"
  TEST_ATTACK="outputs/edge_wm_noise_test/${TAG}_${v}_attack_all"

  extra_train_args=()
  case "$v" in
    A0_Full)
      extra_train_args=(--robust "${common_attack_args[@]}")
      ;;
    A1_NoRobust)
      extra_train_args=(--no-robust)
      ;;
    A2_NoNoiseReg)
      extra_train_args=(--robust "${common_attack_args[@]}" --lambda-noise 0.0)
      ;;
    A3_NoHVS)
      extra_train_args=(--robust "${common_attack_args[@]}" --lambda-hvs 0.0)
      ;;
    A4_NoPayload)
      extra_train_args=(--robust "${common_attack_args[@]}" --lambda-payload 0.0)
      ;;
    A5_NoSSIMinWM)
      extra_train_args=(--robust "${common_attack_args[@]}" --alpha-ssim 0.0)
      ;;
    A6_CBAM_Full)
      extra_train_args=(--robust "${common_attack_args[@]}" --attention cbam)
      ;;
    *)
      echo "unknown ablation variant: $v" >&2
      echo "allowed: A0_Full,A1_NoRobust,A2_NoNoiseReg,A3_NoHVS,A4_NoPayload,A5_NoSSIMinWM,A6_CBAM_Full" >&2
      exit 2
      ;;
  esac

  echo "[train] ${v}"
  bash scripts/with_conda_gp.sh python experiments/edge_wm_noise_train.py \
    --input-dir "$INPUT_DIR" \
    --wm-template "$WM_TEMPLATE" \
    --outdir "$RUN_DIR" \
    --steps "$STEPS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --seed 0 \
    --save-every 200 \
    --log-every 100 \
    --val-dir "$TEST_DIR" \
    --val-every 200 \
    --val-save-samples 4 \
    --no-progress \
    "${extra_train_args[@]}"

  echo "[test clean] ${v}"
  bash scripts/with_conda_gp.sh python experiments/edge_wm_noise_test.py \
    --input-dir "$TEST_DIR" \
    --wm-template "$WM_TEMPLATE" \
    --ckpt "${RUN_DIR}/ckpt_last.pt" \
    --outdir "$TEST_CLEAN" \
    --device "$DEVICE" \
    --save-samples 8 \
    --no-attack

  echo "[test attack-all] ${v}"
  bash scripts/with_conda_gp.sh python experiments/edge_wm_noise_test.py \
    --input-dir "$TEST_DIR" \
    --wm-template "$WM_TEMPLATE" \
    --ckpt "${RUN_DIR}/ckpt_last.pt" \
    --outdir "$TEST_ATTACK" \
    --device "$DEVICE" \
    --save-samples 8 \
    --attack --attack-mode all
done

echo "[report] build summary"
bash scripts/with_conda_gp.sh python - "$AB_OUT" "$TAG" "$VARIANTS_CSV" <<'PY'
import json
import math
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
tag = sys.argv[2]
variants = [x for x in sys.argv[3].split(",") if x]

def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def mean_metric(d: dict, key: str) -> float:
    return float(d.get(key, {}).get("mean", float("nan")))

rows: list[dict[str, float | str]] = []
for v in variants:
    clean = read(Path(f"outputs/edge_wm_noise_test/{tag}_{v}_clean/summary.json"))
    attack = read(Path(f"outputs/edge_wm_noise_test/{tag}_{v}_attack_all/summary.json"))
    rows.append(
        {
            "variant": v,
            "host_psnr_clean": mean_metric(clean, "host_psnr"),
            "host_ssim_clean": mean_metric(clean, "host_ssim"),
            "wm_psnr_clean": mean_metric(clean, "wm_psnr"),
            "wm_ssim_clean": mean_metric(clean, "wm_ssim"),
            "wm_psnr_attack_all": mean_metric(attack, "wm_psnr_attack"),
            "wm_ssim_attack_all": mean_metric(attack, "wm_ssim_attack"),
        }
    )

summary = {"tag": tag, "variants": variants, "rows": rows}
(out_dir / "ablation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def f(x: float) -> str:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "NaN"
    return f"{x:.4f}"

md = []
md.append("# Ablation Summary")
md.append("")
md.append("| Variant | host_psnr(clean) | host_ssim(clean) | wm_psnr(clean) | wm_ssim(clean) | wm_psnr(attack_all) | wm_ssim(attack_all) |")
md.append("|---|---:|---:|---:|---:|---:|---:|")
for r in rows:
    md.append(
        f"| {r['variant']} | {f(float(r['host_psnr_clean']))} | {f(float(r['host_ssim_clean']))} | "
        f"{f(float(r['wm_psnr_clean']))} | {f(float(r['wm_ssim_clean']))} | "
        f"{f(float(r['wm_psnr_attack_all']))} | {f(float(r['wm_ssim_attack_all']))} |"
    )
(out_dir / "ablation_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

print(str(out_dir / "ablation_summary.json"))
print(str(out_dir / "ablation_summary.md"))
PY

echo "Done."
echo "ablation dir: $AB_OUT"
