from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path


def _require_deps() -> tuple[object, object]:
    try:
        import torch  # type: ignore
    except Exception as e:
        print("torch is required for inference but is not installed:", e, file=sys.stderr)
        print("Install it in your environment, then re-run.", file=sys.stderr)
        raise SystemExit(2)

    try:
        from PIL import Image  # type: ignore
    except Exception as e:
        print("Pillow is required for image IO but is not installed:", e, file=sys.stderr)
        print("Install it in your environment, then re-run.", file=sys.stderr)
        raise SystemExit(2)

    return torch, Image


def main() -> int:
    requested_cuda = False
    for i, a in enumerate(sys.argv):
        if a == "--device" and i + 1 < len(sys.argv):
            requested_cuda = str(sys.argv[i + 1]).strip().lower().startswith("cuda")
            break
    if not requested_cuda and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    torch, Image = _require_deps()

    REPO_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO_ROOT))

    from src.utils.toml_min import load_toml  # noqa: E402
    from src.utils.outputs_clean import maybe_clean_outputs  # noqa: E402
    from src.nn.edge_wm.noise_wm_model import NoiseWatermarkConfig, NoiseWatermarkNet  # noqa: E402
    from src.nn.edge_wm.losses import psnr, ssim  # noqa: E402
    from src.nn.edge_wm.attacks import AttackConfig, apply_random_attacks  # noqa: E402

    DEFAULT_CONFIG = REPO_ROOT / "config.toml"
    cfg_all = load_toml(DEFAULT_CONFIG) if DEFAULT_CONFIG.exists() else {}
    cfg_paths = cfg_all.get("paths", {}) if isinstance(cfg_all.get("paths", {}), dict) else {}
    cfg = cfg_all.get("edge_wm_noise_infer", {}) if isinstance(cfg_all.get("edge_wm_noise_infer", {}), dict) else {}
    outputs_dir = str(cfg_paths.get("outputs_dir", "outputs"))
    clean_outputs_default = bool(cfg_paths.get("clean_outputs", True))

    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default=str(cfg.get("input", "")))
    p.add_argument("--input-dir", type=str, default=str(cfg.get("input_dir", "")))
    p.add_argument("--wm-template", type=str, default=str(cfg.get("wm_template", "assets/watermarks/luke.png")))
    p.add_argument("--ckpt", type=str, default=str(cfg.get("ckpt", "")))
    p.add_argument("--outdir", type=str, default=str(cfg.get("outdir", "")))
    p.add_argument("--size", type=int, default=int(cfg.get("size", 256)))
    p.add_argument("--device", type=str, default=str(cfg.get("device", "cpu")))
    p.add_argument("--clean-outputs", action=argparse.BooleanOptionalAction, default=bool(cfg.get("clean_outputs", False if clean_outputs_default else False)))

    # Optional attacks for robustness visualization/eval
    p.add_argument("--attack", action=argparse.BooleanOptionalAction, default=bool(cfg.get("attack", False)))
    p.add_argument("--attack-crop-keep", type=float, default=float(cfg.get("attack_crop_keep", 0.7)))
    p.add_argument("--attack-mosaic-block", type=int, default=int(cfg.get("attack_mosaic_block", 8)))
    p.add_argument("--attack-jpeg-quality", type=int, default=int(cfg.get("attack_jpeg_quality", 50)))
    p.add_argument("--attack-jpeg-mode", type=str, default=str(cfg.get("attack_jpeg_mode", "pil")), choices=["approx", "pil"])
    p.add_argument("--attack-resize-scale", type=float, default=float(cfg.get("attack_resize_scale", 0.8)))
    p.add_argument("--attack-blur-sigma", type=float, default=float(cfg.get("attack_blur_sigma", 1.0)))
    args = p.parse_args()

    if str(args.device).strip().lower().startswith("cpu"):
        warnings.filterwarnings(
            "ignore",
            message=r"CUDA initialization: Unexpected error from cudaGetDeviceCount.*",
            category=UserWarning,
        )

    if not args.input and not args.input_dir:
        raise SystemExit("provide --input or --input-dir")

    def _pick_latest_ckpt(run_dir: Path) -> Path:
        if not run_dir.exists() or not run_dir.is_dir():
            raise SystemExit(f"checkpoint run dir not found: {run_dir}")
        candidates = sorted(run_dir.glob("ckpt_step_*.pt"))
        if (run_dir / "ckpt_last.pt").exists():
            return run_dir / "ckpt_last.pt"
        if not candidates:
            raise SystemExit(f"no checkpoints found in: {run_dir}")
        return candidates[-1]

    def _resolve_ckpt(ckpt_arg: str) -> Path:
        s = str(ckpt_arg or "").strip()
        if not s:
            raise SystemExit("provide --ckpt (or --ckpt latest)")
        if s.lower() == "latest":
            base = Path(outputs_dir) / "edge_wm_noise"
            run_dirs = sorted([p2 for p2 in base.iterdir() if p2.is_dir() and not p2.name.startswith("_")], reverse=True) if base.exists() else []
            if not run_dirs:
                raise SystemExit(f"no runs found in: {base}")
            return _pick_latest_ckpt(run_dirs[0])

        pth = Path(s)
        if pth.is_dir():
            return _pick_latest_ckpt(pth)
        if not pth.exists():
            raise SystemExit(f"checkpoint not found: {pth}")
        return pth

    ckpt_path = _resolve_ckpt(str(args.ckpt))
    wm_path = Path(args.wm_template)
    if not wm_path.exists():
        raise SystemExit(f"watermark template not found: {wm_path}")

    maybe_clean_outputs(repo_root=REPO_ROOT, outputs_dir=outputs_dir, enabled=bool(args.clean_outputs))
    outdir = Path(args.outdir) if args.outdir else Path(outputs_dir) / "edge_wm_noise_infer" / time.strftime("%Y%m%d_%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(str(args.device))
    resample_lanczos = getattr(getattr(Image, "Resampling", None), "LANCZOS", Image.LANCZOS)

    def resize_square(img, size: int):
        return img.resize((size, size), resample=resample_lanczos)

    def pil_to_tensor_rgb(img) -> "torch.Tensor":
        img = img.convert("RGB")
        img = resize_square(img, int(args.size))
        b = bytearray(img.tobytes())
        w, h = img.size
        x = torch.frombuffer(b, dtype=torch.uint8).to(torch.float32)
        x = x.view(h, w, 3).permute(2, 0, 1).contiguous() / 255.0
        return x

    def pil_to_tensor_ycbcr(img) -> "torch.Tensor":
        img = img.convert("YCbCr")
        img = resize_square(img, int(args.size))
        b = bytearray(img.tobytes())
        w, h = img.size
        x = torch.frombuffer(b, dtype=torch.uint8).to(torch.float32)
        x = x.view(h, w, 3).permute(2, 0, 1).contiguous() / 255.0
        return x

    def tensor_to_pil_rgb(x: "torch.Tensor"):
        x = x.detach().clamp(0.0, 1.0).mul(255.0).to(torch.uint8)
        x = x.permute(1, 2, 0).contiguous().cpu()
        h, w, c = x.shape
        if c != 3:
            raise ValueError("expected 3-channel tensor for RGB")
        try:
            import numpy as np  # type: ignore

            arr = np.asarray(x)
            return Image.fromarray(arr, mode="RGB")
        except Exception:
            b = bytes(x.reshape(-1).tolist())
            return Image.frombytes("RGB", (w, h), b)

    def tensor_ycbcr_to_pil_rgb(x: "torch.Tensor"):
        x = x.detach().clamp(0.0, 1.0).mul(255.0).to(torch.uint8)
        arr = x.permute(1, 2, 0).contiguous().cpu()
        h, w, c = arr.shape
        if c != 3:
            raise ValueError("expected 3-channel tensor for watermark output")
        try:
            img = Image.fromarray(arr.numpy(), mode="YCbCr")
        except Exception:
            img = Image.frombytes("YCbCr", (w, h), bytes(arr.reshape(-1).tolist()))
        return img.convert("RGB")

    def compose_y_with_cbcr_to_pil_rgb(y01: "torch.Tensor", cbcr01: "torch.Tensor"):
        if y01.ndim != 3 or y01.shape[0] != 1:
            raise ValueError("expected y01 with shape (1,H,W)")
        if cbcr01.ndim != 3 or cbcr01.shape[0] != 2:
            raise ValueError("expected cbcr01 with shape (2,H,W)")
        if y01.shape[-2:] != cbcr01.shape[-2:]:
            raise ValueError("Y and CbCr spatial sizes must match")
        ycbcr = torch.cat([y01, cbcr01.to(y01.device)], dim=0)
        return tensor_ycbcr_to_pil_rgb(ycbcr)

    def tensor_to_pil_l(x01: "torch.Tensor"):
        x01 = x01.detach().clamp(0.0, 1.0).mul(255.0).to(torch.uint8).contiguous().cpu()
        h, w = int(x01.shape[-2]), int(x01.shape[-1])
        b = bytes(x01.reshape(-1).tolist())
        return Image.frombytes("L", (w, h), b)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise SystemExit("invalid checkpoint")
    try:
        ckpt_step = int(ckpt.get("step", 0) or 0)
        if ckpt_step and ckpt_step < 50:
            print(f"warning: checkpoint step={ckpt_step} is very small; watermark recovery will likely be poor", file=sys.stderr)
    except Exception:
        pass
    cfg_d = ckpt.get("model_config") if isinstance(ckpt.get("model_config"), dict) else {}
    cfg = NoiseWatermarkConfig(**{k: cfg_d[k] for k in cfg_d if k in NoiseWatermarkConfig().__dict__})  # type: ignore[arg-type]

    model = NoiseWatermarkNet(cfg)
    model.load_state_dict(ckpt["model_state"], strict=True)  # type: ignore[arg-type]
    model.to(device).eval()

    watermark_img = Image.open(wm_path)
    wm_ycbcr = pil_to_tensor_ycbcr(watermark_img).to(device)
    wm = wm_ycbcr[:1].unsqueeze(0)  # only watermark Y is encoded/recovered
    wm_cbcr = wm_ycbcr[1:]  # reuse template chroma for RGB visualization

    def safe_stem(stem: str) -> str:
        s = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in stem)
        return (s[:80] or "image").strip("_") or "image"

    input_paths: list[Path] = []
    if str(args.input).strip():
        input_paths = [Path(args.input)]
    elif args.input_dir:
        in_dir = Path(args.input_dir)
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        input_paths = sorted([p2 for p2 in in_dir.rglob("*") if p2.is_file() and p2.suffix.lower() in exts])
        if not input_paths:
            raise SystemExit(f"no images found in --input-dir: {in_dir}")

    manifest: dict[str, object] = {"checkpoint": str(ckpt_path), "wm_template": str(wm_path), "size": int(args.size), "images": []}

    attack_cfg = AttackConfig(
        prob=1.0,
        crop_keep=float(args.attack_crop_keep),
        crop_prob=1.0,
        mosaic_prob=1.0,
        mosaic_min_block=int(args.attack_mosaic_block),
        mosaic_max_block=int(args.attack_mosaic_block),
        jpeg_prob=1.0,
        jpeg_quality_min=int(args.attack_jpeg_quality),
        jpeg_quality_max=int(args.attack_jpeg_quality),
        jpeg_mode=str(args.attack_jpeg_mode),
        resize_prob=1.0,
        resize_min_scale=float(args.attack_resize_scale),
        resize_max_scale=float(args.attack_resize_scale),
        blur_prob=1.0,
        blur_kernel=5,
        blur_sigma_max=float(args.attack_blur_sigma),
    )

    with torch.no_grad():
        for path in input_paths:
            cover_img = Image.open(path).convert("RGB")
            cover = pil_to_tensor_rgb(cover_img).unsqueeze(0).to(device)
            wmb = wm.expand(cover.shape[0], -1, -1, -1)
            # Compute explicitly so we can optionally decode from attacked container.
            payload = model.wm_to_noise(wmb)
            container, _resid = model.embedder(cover, payload)
            payload_hat = model.reveal_payload(container)
            wm_hat = model.noise_to_wm(payload_hat)

            container_attacked = None
            payload_hat_attacked = None
            wm_hat_attacked = None
            if bool(args.attack):
                # Deterministic full-attack chain for visualization.
                rng = __import__("random").Random(12345)
                container_attacked = apply_random_attacks(container, cfg=attack_cfg, rng=rng, allow_pil_jpeg=(str(args.attack_jpeg_mode) == "pil"))
                payload_hat_attacked = model.reveal_payload(container_attacked)
                wm_hat_attacked = model.noise_to_wm(payload_hat_attacked)

            host_psnr = float(psnr(cover, container).item())
            host_ssim = float(ssim(cover, container).item())
            wm_psnr = float(psnr(wmb, wm_hat).item())
            wm_ssim = float(ssim(wmb, wm_hat).item())
            wm_psnr_attack = float(psnr(wmb, wm_hat_attacked).item()) if wm_hat_attacked is not None else float("nan")
            wm_ssim_attack = float(ssim(wmb, wm_hat_attacked).item()) if wm_hat_attacked is not None else float("nan")

            stem = safe_stem(path.stem)
            sub = outdir / stem if len(input_paths) > 1 else outdir
            sub.mkdir(parents=True, exist_ok=True)

            tensor_to_pil_rgb(cover[0]).save(sub / f"{stem}_cover.png")
            tensor_to_pil_rgb(container[0]).save(sub / f"{stem}_container.png")
            resid = (container[0] - cover[0]).abs()
            resid_vis = resid / (resid.max() + 1e-12)
            tensor_to_pil_rgb(resid_vis).save(sub / f"{stem}_residual_abs_norm.png")

            compose_y_with_cbcr_to_pil_rgb(wmb[0], wm_cbcr).save(sub / f"{stem}_wm.png")
            compose_y_with_cbcr_to_pil_rgb(wm_hat[0], wm_cbcr).save(sub / f"{stem}_wm_hat.png")

            p0 = (payload[0] + 1.0) * 0.5
            ph0 = (payload_hat[0] + 1.0) * 0.5
            tensor_to_pil_l(p0).save(sub / f"{stem}_payload.png")
            tensor_to_pil_l(ph0).save(sub / f"{stem}_payload_hat.png")

            if container_attacked is not None and wm_hat_attacked is not None and payload_hat_attacked is not None:
                tensor_to_pil_rgb(container_attacked[0]).save(sub / f"{stem}_container_attacked.png")
                compose_y_with_cbcr_to_pil_rgb(wm_hat_attacked[0], wm_cbcr).save(sub / f"{stem}_wm_hat_attacked.png")
                ph1 = (payload_hat_attacked[0] + 1.0) * 0.5
                tensor_to_pil_l(ph1).save(sub / f"{stem}_payload_hat_attacked.png")

            manifest["images"].append(
                {
                    "input": str(path),
                    "outputs": {
                        "cover": str(sub / f"{stem}_cover.png"),
                        "container": str(sub / f"{stem}_container.png"),
                        "residual_abs_norm": str(sub / f"{stem}_residual_abs_norm.png"),
                        "wm": str(sub / f"{stem}_wm.png"),
                        "wm_hat": str(sub / f"{stem}_wm_hat.png"),
                        "payload": str(sub / f"{stem}_payload.png"),
                        "payload_hat": str(sub / f"{stem}_payload_hat.png"),
                        "container_attacked": str(sub / f"{stem}_container_attacked.png") if container_attacked is not None else "",
                        "wm_hat_attacked": str(sub / f"{stem}_wm_hat_attacked.png") if wm_hat_attacked is not None else "",
                        "payload_hat_attacked": str(sub / f"{stem}_payload_hat_attacked.png") if payload_hat_attacked is not None else "",
                    },
                    "metrics": {
                        "host_psnr": host_psnr,
                        "host_ssim": host_ssim,
                        "wm_psnr": wm_psnr,
                        "wm_ssim": wm_ssim,
                        "wm_psnr_attack": wm_psnr_attack,
                        "wm_ssim_attack": wm_ssim_attack,
                    },
                }
            )

    (outdir / "run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
