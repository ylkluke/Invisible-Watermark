#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path


DEFAULT_STEMS = [
    "pexels-qui-nguyen-7862521-29537255",
    "pexels-steve-28447920",
    "pexels-steve-28901845",
    "pexels-steve-29101884",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, default="outputs/edge_wm_noise/y_only_fresh_20260402")
    p.add_argument("--input-dir", type=str, default="inputs/chaos_batch")
    p.add_argument("--wm-template", type=str, default="")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--size", type=int, default=0)
    p.add_argument("--steps", type=str, default="")
    p.add_argument("--image-stems", type=str, default=",".join(DEFAULT_STEMS))
    p.add_argument("--print-only", action="store_true")
    return p.parse_args()


def _percentiles(xs: list[float], ps: list[float]) -> dict[str, float]:
    if not xs:
        return {f"p{int(p)}": float("nan") for p in ps}
    ys = sorted(xs)
    n = len(ys)
    out: dict[str, float] = {}
    for p in ps:
        if p <= 0:
            out[f"p{int(p)}"] = float(ys[0])
            continue
        if p >= 100:
            out[f"p{int(p)}"] = float(ys[-1])
            continue
        k = (p / 100.0) * (n - 1)
        i = int(k)
        j = min(n - 1, i + 1)
        t = k - i
        out[f"p{int(p)}"] = float((1.0 - t) * ys[i] + t * ys[j])
    return out


def resolve_image_paths(input_dir: Path, stems_csv: str) -> list[Path]:
    paths: list[Path] = []
    for stem in [s.strip() for s in stems_csv.split(",") if s.strip()]:
        matches = sorted(input_dir.glob(f"{stem}.*"))
        if not matches:
            raise SystemExit(f"missing image for stem: {stem}")
        paths.append(matches[0])
    return paths


def main() -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    import torch
    from PIL import Image

    from src.nn.edge_wm.attacks import AttackConfig, apply_random_attacks
    from src.nn.edge_wm.losses import adjacent_corr_penalty, psnr, ssim
    from src.nn.edge_wm.noise_wm_model import NoiseWatermarkConfig, NoiseWatermarkNet

    run_dir = Path(args.run_dir)
    input_dir = Path(args.input_dir)
    summary_path = run_dir / "train_summary.json"
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    train_args = summary.get("args", {}) if isinstance(summary, dict) else {}

    wm_template = Path(args.wm_template or train_args.get("wm_template") or "assets/watermarks/luke.png")
    size = int(args.size or train_args.get("size") or 256)
    seed = int(train_args.get("seed") or 0)
    robust = bool(train_args.get("robust", True))
    image_paths = resolve_image_paths(input_dir, args.image_stems)

    ckpt_paths = sorted(run_dir.glob("ckpt_step_*.pt"))
    if args.steps.strip():
        wanted = {int(s.strip()) for s in args.steps.split(",") if s.strip()}
        ckpt_paths = [p for p in ckpt_paths if int(p.stem.split("_")[-1]) in wanted]
    if not ckpt_paths:
        raise SystemExit("no checkpoint steps selected")

    device = torch.device(str(args.device))
    resample_lanczos = getattr(getattr(Image, "Resampling", None), "LANCZOS", Image.LANCZOS)

    def resize_square(img, edge: int):
        return img.resize((edge, edge), resample=resample_lanczos)

    def pil_to_tensor_rgb(img) -> "torch.Tensor":
        img = img.convert("RGB")
        b = bytearray(img.tobytes())
        w, h = img.size
        x = torch.frombuffer(b, dtype=torch.uint8).to(torch.float32)
        return x.view(h, w, 3).permute(2, 0, 1).contiguous() / 255.0

    def pil_to_tensor_ycbcr(img) -> "torch.Tensor":
        img = img.convert("YCbCr")
        b = bytearray(img.tobytes())
        w, h = img.size
        x = torch.frombuffer(b, dtype=torch.uint8).to(torch.float32)
        return x.view(h, w, 3).permute(2, 0, 1).contiguous() / 255.0

    watermark_img = resize_square(Image.open(wm_template), size)
    wm0_ycbcr = pil_to_tensor_ycbcr(watermark_img).to(device)
    wm0 = wm0_ycbcr[:1]

    attack_cfg = AttackConfig(
        prob=float(train_args.get("attack_prob", 0.7)),
        crop_keep=float(train_args.get("attack_crop_keep", 0.7)),
        crop_prob=float(train_args.get("attack_crop_prob", 0.5)),
        mosaic_prob=float(train_args.get("attack_mosaic_prob", 0.3)),
        mosaic_min_block=int(train_args.get("attack_mosaic_min_block", 4)),
        mosaic_max_block=int(train_args.get("attack_mosaic_max_block", 16)),
        jpeg_prob=float(train_args.get("attack_jpeg_prob", 0.3)),
        jpeg_quality_min=int(train_args.get("attack_jpeg_quality_min", 30)),
        jpeg_quality_max=int(train_args.get("attack_jpeg_quality_max", 95)),
        jpeg_mode=str(train_args.get("attack_jpeg_mode", "approx")),
        resize_prob=float(train_args.get("attack_resize_prob", 0.3)),
        resize_min_scale=float(train_args.get("attack_resize_min_scale", 0.7)),
        resize_max_scale=float(train_args.get("attack_resize_max_scale", 1.3)),
        blur_prob=float(train_args.get("attack_blur_prob", 0.3)),
        blur_kernel=int(train_args.get("attack_blur_kernel", 5)),
        blur_sigma_max=float(train_args.get("attack_blur_sigma_max", 1.2)),
    )

    rows: list[dict[str, object]] = []
    allow_pil_jpeg = str(train_args.get("attack_jpeg_mode", "approx")).strip().lower() == "pil"
    base_seed = seed if seed else 12345

    for ckpt_path in ckpt_paths:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        cfg_kwargs = dict(ckpt.get("model_config", {}))
        cfg = NoiseWatermarkConfig(**cfg_kwargs)
        model = NoiseWatermarkNet(cfg).to(device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.eval()

        step = int(ckpt.get("step") or int(ckpt_path.stem.split("_")[-1]))
        vrng = random.Random(base_seed + step)

        host_psnrs: list[float] = []
        host_ssims: list[float] = []
        wm_psnrs: list[float] = []
        wm_ssims: list[float] = []
        wm_psnrs_attack: list[float] = []
        wm_ssims_attack: list[float] = []
        payload_stds: list[float] = []
        payload_corrs: list[float] = []

        with torch.no_grad():
            for image_path in image_paths:
                cover_img = resize_square(Image.open(image_path).convert("RGB"), size)
                cover = pil_to_tensor_rgb(cover_img).unsqueeze(0).to(device)
                wm = wm0.unsqueeze(0).expand(cover.shape[0], -1, -1, -1)

                payload = model.wm_to_noise(wm)
                container, _ = model.embedder(cover, payload)
                payload_hat = model.reveal_payload(container)
                wm_hat = model.noise_to_wm(payload_hat)

                host_psnrs.append(float(psnr(cover, container).item()))
                host_ssims.append(float(ssim(cover, container).item()))
                wm_psnrs.append(float(psnr(wm, wm_hat).item()))
                wm_ssims.append(float(ssim(wm, wm_hat).item()))
                payload_stds.append(float(payload.std(unbiased=False).item()))
                payload_corrs.append(float(adjacent_corr_penalty(payload).sqrt().item()))

                if robust:
                    attacked = apply_random_attacks(container, cfg=attack_cfg, rng=vrng, allow_pil_jpeg=allow_pil_jpeg)
                    payload_hat_a = model.reveal_payload(attacked)
                    wm_hat_a = model.noise_to_wm(payload_hat_a)
                    wm_psnrs_attack.append(float(psnr(wm, wm_hat_a).item()))
                    wm_ssims_attack.append(float(ssim(wm, wm_hat_a).item()))

        row: dict[str, object] = {
            "step": step,
            "n": len(image_paths),
            "host_psnr": {"mean": float(sum(host_psnrs) / max(1, len(host_psnrs))), **_percentiles(host_psnrs, [10, 50, 90])},
            "host_ssim": {"mean": float(sum(host_ssims) / max(1, len(host_ssims))), **_percentiles(host_ssims, [10, 50, 90])},
            "wm_psnr_clean": {"mean": float(sum(wm_psnrs) / max(1, len(wm_psnrs))), **_percentiles(wm_psnrs, [10, 50, 90])},
            "wm_ssim_clean": {"mean": float(sum(wm_ssims) / max(1, len(wm_ssims))), **_percentiles(wm_ssims, [10, 50, 90])},
            "wm_psnr_attack": {"mean": float(sum(wm_psnrs_attack) / max(1, len(wm_psnrs_attack))), **_percentiles(wm_psnrs_attack, [10, 50, 90])},
            "wm_ssim_attack": {"mean": float(sum(wm_ssims_attack) / max(1, len(wm_ssims_attack))), **_percentiles(wm_ssims_attack, [10, 50, 90])},
            "payload_std": {"mean": float(sum(payload_stds) / max(1, len(payload_stds))), **_percentiles(payload_stds, [10, 50, 90])},
            "payload_adjcorr": {"mean": float(sum(payload_corrs) / max(1, len(payload_corrs))), **_percentiles(payload_corrs, [10, 50, 90])},
            "robust": robust,
        }
        rows.append(row)

    rows.sort(key=lambda row: int(row["step"]))

    if args.print_only:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0

    for row in rows:
        step = int(row["step"])
        out = run_dir / f"val_summary_step_{step:07d}.json"
        out.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (run_dir / "val_metrics.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
