from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path


def _require_deps() -> tuple[object, object]:
    try:
        import torch  # type: ignore
    except Exception as e:
        print("torch is required for testing but is not installed:", e, file=sys.stderr)
        print("Install it in your environment, then re-run.", file=sys.stderr)
        raise SystemExit(2)

    try:
        from PIL import Image  # type: ignore
    except Exception as e:
        print("Pillow is required for image IO but is not installed:", e, file=sys.stderr)
        print("Install it in your environment, then re-run.", file=sys.stderr)
        raise SystemExit(2)

    return torch, Image


def _format_eta(seconds: float) -> str:
    if seconds != seconds or seconds < 0:
        return "?"
    s = int(seconds + 0.5)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _format_progress(*, step: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return f"{step}"
    step = max(0, min(step, total))
    done = int(round((step / total) * width))
    bar = "#" * done + "-" * (width - done)
    return f"[{bar}] {step:>5d}/{total:<5d}"


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
    from src.nn.edge_wm.losses import adjacent_corr_penalty, noise_stats_loss, psnr, ssim  # noqa: E402
    from src.nn.edge_wm.attacks import AttackConfig, apply_random_attacks  # noqa: E402

    DEFAULT_CONFIG = REPO_ROOT / "config.toml"
    cfg_all = load_toml(DEFAULT_CONFIG) if DEFAULT_CONFIG.exists() else {}
    cfg_paths = cfg_all.get("paths", {}) if isinstance(cfg_all.get("paths", {}), dict) else {}
    cfg = cfg_all.get("edge_wm_noise_test", {}) if isinstance(cfg_all.get("edge_wm_noise_test", {}), dict) else {}
    outputs_dir = str(cfg_paths.get("outputs_dir", "outputs"))
    clean_outputs_default = bool(cfg_paths.get("clean_outputs", True))

    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default=str(cfg.get("input", "")))
    p.add_argument("--input-dir", type=str, default=str(cfg.get("input_dir", "")))
    p.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=bool(cfg.get("recursive", True)))
    p.add_argument("--limit", type=int, default=int(cfg.get("limit", 0)), help="0 means no limit")
    p.add_argument("--seed", type=int, default=int(cfg.get("seed", 0)))

    p.add_argument("--wm-template", type=str, default=str(cfg.get("wm_template", "assets/watermarks/luke.png")))
    p.add_argument("--ckpt", type=str, default=str(cfg.get("ckpt", "")))
    p.add_argument("--outdir", type=str, default=str(cfg.get("outdir", "")))
    p.add_argument("--size", type=int, default=int(cfg.get("size", 256)))
    p.add_argument("--device", type=str, default=str(cfg.get("device", "cpu")))

    p.add_argument("--save-samples", type=int, default=int(cfg.get("save_samples", 8)))
    p.add_argument("--save-amplify", type=float, default=float(cfg.get("save_amplify", 20.0)))
    p.add_argument("--save-all", action=argparse.BooleanOptionalAction, default=bool(cfg.get("save_all", False)), help="save per-image outputs for all inputs (can be large)")

    # Optional robustness evaluation (apply attacks to container before decoding)
    p.add_argument("--attack", action=argparse.BooleanOptionalAction, default=bool(cfg.get("attack", False)))
    p.add_argument("--attack-mode", type=str, default=str(cfg.get("attack_mode", "random")), choices=["random", "all"])
    p.add_argument("--attack-prob", type=float, default=float(cfg.get("attack_prob", 0.7)))
    p.add_argument("--attack-crop-keep", type=float, default=float(cfg.get("attack_crop_keep", 0.7)))
    p.add_argument("--attack-crop-prob", type=float, default=float(cfg.get("attack_crop_prob", 0.5)))
    p.add_argument("--attack-mosaic-prob", type=float, default=float(cfg.get("attack_mosaic_prob", 0.3)))
    p.add_argument("--attack-mosaic-min-block", type=int, default=int(cfg.get("attack_mosaic_min_block", 4)))
    p.add_argument("--attack-mosaic-max-block", type=int, default=int(cfg.get("attack_mosaic_max_block", 16)))
    p.add_argument("--attack-jpeg-prob", type=float, default=float(cfg.get("attack_jpeg_prob", 0.3)))
    p.add_argument("--attack-jpeg-quality-min", type=int, default=int(cfg.get("attack_jpeg_quality_min", 30)))
    p.add_argument("--attack-jpeg-quality-max", type=int, default=int(cfg.get("attack_jpeg_quality_max", 95)))
    p.add_argument("--attack-jpeg-mode", type=str, default=str(cfg.get("attack_jpeg_mode", "pil")), choices=["approx", "pil"])
    p.add_argument("--attack-resize-prob", type=float, default=float(cfg.get("attack_resize_prob", 0.3)))
    p.add_argument("--attack-resize-min-scale", type=float, default=float(cfg.get("attack_resize_min_scale", 0.7)))
    p.add_argument("--attack-resize-max-scale", type=float, default=float(cfg.get("attack_resize_max_scale", 1.3)))
    p.add_argument("--attack-blur-prob", type=float, default=float(cfg.get("attack_blur_prob", 0.3)))
    p.add_argument("--attack-blur-kernel", type=int, default=int(cfg.get("attack_blur_kernel", 5)))
    p.add_argument("--attack-blur-sigma-max", type=float, default=float(cfg.get("attack_blur_sigma_max", 1.2)))

    p.add_argument("--clean-outputs", action=argparse.BooleanOptionalAction, default=bool(cfg.get("clean_outputs", False if clean_outputs_default else False)))
    args = p.parse_args()

    if str(args.device).strip().lower().startswith("cpu"):
        warnings.filterwarnings(
            "ignore",
            message=r"CUDA initialization: Unexpected error from cudaGetDeviceCount.*",
            category=UserWarning,
        )

    if args.seed:
        random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))

    if not args.input and not args.input_dir:
        raise SystemExit("provide --input or --input-dir")

    def _pick_latest_ckpt(run_dir: Path) -> Path:
        if not run_dir.exists() or not run_dir.is_dir():
            raise SystemExit(f"checkpoint run dir not found: {run_dir}")
        if (run_dir / "ckpt_last.pt").exists():
            return run_dir / "ckpt_last.pt"
        candidates = sorted(run_dir.glob("ckpt_step_*.pt"))
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
    outdir = Path(args.outdir) if args.outdir else Path(outputs_dir) / "edge_wm_noise_test" / time.strftime("%Y%m%d_%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    samples_dir = outdir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

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

    def pil_to_tensor_l(img) -> "torch.Tensor":
        img = img.convert("L")
        img = resize_square(img, int(args.size))
        b = bytearray(img.tobytes())
        w, h = img.size
        x = torch.frombuffer(b, dtype=torch.uint8).to(torch.float32)
        x = x.view(h, w).unsqueeze(0).contiguous() / 255.0
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

    def pil_to_tensor_ycbcr(img) -> "torch.Tensor":
        img = img.convert("YCbCr")
        img = resize_square(img, int(args.size))
        b = bytearray(img.tobytes())
        w, h = img.size
        x = torch.frombuffer(b, dtype=torch.uint8).to(torch.float32)
        x = x.view(h, w, 3).permute(2, 0, 1).contiguous() / 255.0
        return x

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
        raise SystemExit("invalid checkpoint (missing model_state)")
    mcfg_d = ckpt.get("model_config") if isinstance(ckpt.get("model_config"), dict) else {}
    mcfg = NoiseWatermarkConfig(**{k: mcfg_d[k] for k in mcfg_d if k in NoiseWatermarkConfig().__dict__})  # type: ignore[arg-type]
    model = NoiseWatermarkNet(mcfg)
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
        it = in_dir.rglob("*") if args.recursive else in_dir.iterdir()
        input_paths = sorted([p2 for p2 in it if p2.is_file() and p2.suffix.lower() in exts])
        if not input_paths:
            raise SystemExit(f"no images found in --input-dir: {in_dir}")

    if args.limit and int(args.limit) > 0:
        input_paths = input_paths[: int(args.limit)]

    total = len(input_paths)
    metrics_path = outdir / "metrics.jsonl"
    mf = open(metrics_path, "w", encoding="utf-8")

    if bool(args.save_all) or int(args.save_samples) < 0:
        save_paths = set(input_paths)
    else:
        save_n = max(0, int(args.save_samples))
        save_paths = set(random.sample(input_paths, k=min(save_n, len(input_paths)))) if save_n and total else set()

    host_psnrs: list[float] = []
    host_ssims: list[float] = []
    wm_psnrs: list[float] = []
    wm_ssims: list[float] = []
    wm_psnrs_attack: list[float] = []
    wm_ssims_attack: list[float] = []
    payload_stds: list[float] = []
    payload_corrs: list[float] = []

    attack_cfg = AttackConfig(
        prob=float(args.attack_prob),
        crop_keep=float(args.attack_crop_keep),
        crop_prob=float(args.attack_crop_prob),
        mosaic_prob=float(args.attack_mosaic_prob),
        mosaic_min_block=int(args.attack_mosaic_min_block),
        mosaic_max_block=int(args.attack_mosaic_max_block),
        jpeg_prob=float(args.attack_jpeg_prob),
        jpeg_quality_min=int(args.attack_jpeg_quality_min),
        jpeg_quality_max=int(args.attack_jpeg_quality_max),
        jpeg_mode=str(args.attack_jpeg_mode),
        resize_prob=float(args.attack_resize_prob),
        resize_min_scale=float(args.attack_resize_min_scale),
        resize_max_scale=float(args.attack_resize_max_scale),
        blur_prob=float(args.attack_blur_prob),
        blur_kernel=int(args.attack_blur_kernel),
        blur_sigma_max=float(args.attack_blur_sigma_max),
    )
    if str(args.attack_mode).strip().lower() == "all":
        attack_cfg = AttackConfig(
            prob=1.0,
            crop_keep=float(args.attack_crop_keep),
            crop_prob=1.0,
            mosaic_prob=1.0,
            mosaic_min_block=int(args.attack_mosaic_min_block),
            mosaic_max_block=int(args.attack_mosaic_max_block),
            jpeg_prob=1.0,
            jpeg_quality_min=int(args.attack_jpeg_quality_min),
            jpeg_quality_max=int(args.attack_jpeg_quality_max),
            jpeg_mode=str(args.attack_jpeg_mode),
            resize_prob=1.0,
            resize_min_scale=float(args.attack_resize_min_scale),
            resize_max_scale=float(args.attack_resize_max_scale),
            blur_prob=1.0,
            blur_kernel=int(args.attack_blur_kernel),
            blur_sigma_max=float(args.attack_blur_sigma_max),
        )

    avg_step_s = 0.0
    t_start = time.monotonic()

    with torch.no_grad():
        for i, path in enumerate(input_paths, start=1):
            t0 = time.monotonic()

            cover_img = Image.open(path).convert("RGB")
            cover = pil_to_tensor_rgb(cover_img).unsqueeze(0).to(device)
            wmb = wm.expand(cover.shape[0], -1, -1, -1)
            payload = model.wm_to_noise(wmb)
            container, _resid = model.embedder(cover, payload)
            payload_hat = model.reveal_payload(container)
            wm_hat = model.noise_to_wm(payload_hat)

            container_attacked = None
            wm_hat_attacked = None
            payload_hat_attacked = None
            if bool(args.attack):
                rng = random.Random((int(args.seed) if args.seed else 12345) + i)
                container_attacked = apply_random_attacks(container, cfg=attack_cfg, rng=rng, allow_pil_jpeg=(str(args.attack_jpeg_mode) == "pil"))
                payload_hat_attacked = model.reveal_payload(container_attacked)
                wm_hat_attacked = model.noise_to_wm(payload_hat_attacked)

            host_psnr = float(psnr(cover, container).item())
            host_ssim = float(ssim(cover, container).item())
            wm_psnr = float(psnr(wmb, wm_hat).item())
            wm_ssim = float(ssim(wmb, wm_hat).item())
            wm_psnr_attack = float(psnr(wmb, wm_hat_attacked).item()) if wm_hat_attacked is not None else float("nan")
            wm_ssim_attack = float(ssim(wmb, wm_hat_attacked).item()) if wm_hat_attacked is not None else float("nan")

            payload_std = float(payload.std(unbiased=False).item())
            payload_corr = float(adjacent_corr_penalty(payload).sqrt().item())

            host_psnrs.append(host_psnr)
            host_ssims.append(host_ssim)
            wm_psnrs.append(wm_psnr)
            wm_ssims.append(wm_ssim)
            if bool(args.attack):
                wm_psnrs_attack.append(wm_psnr_attack)
                wm_ssims_attack.append(wm_ssim_attack)
            payload_stds.append(payload_std)
            payload_corrs.append(payload_corr)

            row = {
                "i": i,
                "path": str(path),
                "host_psnr": host_psnr,
                "host_ssim": host_ssim,
                "wm_psnr": wm_psnr,
                "wm_ssim": wm_ssim,
                "wm_psnr_attack": wm_psnr_attack,
                "wm_ssim_attack": wm_ssim_attack,
                "payload_std": payload_std,
                "payload_adjcorr": payload_corr,
                "payload_stats_loss": float(noise_stats_loss(payload).item()),
                "attack": bool(args.attack),
                "attack_mode": str(args.attack_mode),
            }
            mf.write(json.dumps(row, ensure_ascii=False) + "\n")

            dt = time.monotonic() - t0
            avg_step_s = dt if i == 1 else (0.98 * avg_step_s + 0.02 * dt)
            eta = (total - i) * avg_step_s
            msg = f"{_format_progress(step=i, total=total)} eta={_format_eta(eta)} host_psnr={host_psnr:.2f} wm_ssim={wm_ssim:.3f}"
            sys.stdout.write("\r" + msg[:160].ljust(160))
            sys.stdout.flush()

            if path in save_paths:
                stem = safe_stem(path.stem)
                sub = samples_dir / stem
                sub.mkdir(parents=True, exist_ok=True)

                tensor_to_pil_rgb(cover[0]).save(sub / "cover.png")
                tensor_to_pil_rgb(container[0]).save(sub / "container.png")
                if container_attacked is not None:
                    tensor_to_pil_rgb(container_attacked[0]).save(sub / "container_attacked.png")

                resid = (container[0] - cover[0])
                resid_abs = resid.abs()
                resid_amp = (resid_abs * float(args.save_amplify)).clamp(0.0, 1.0)
                tensor_to_pil_rgb(resid_amp).save(sub / "residual_abs_amplified.png")

                compose_y_with_cbcr_to_pil_rgb(wmb[0], wm_cbcr).save(sub / "wm.png")
                compose_y_with_cbcr_to_pil_rgb(wm_hat[0], wm_cbcr).save(sub / "wm_hat.png")
                if wm_hat_attacked is not None:
                    compose_y_with_cbcr_to_pil_rgb(wm_hat_attacked[0], wm_cbcr).save(sub / "wm_hat_attacked.png")

                p0 = (payload[0] + 1.0) * 0.5
                ph0 = (payload_hat[0] + 1.0) * 0.5
                tensor_to_pil_l(p0).save(sub / "payload.png")
                tensor_to_pil_l(ph0).save(sub / "payload_hat.png")
                if payload_hat_attacked is not None:
                    ph1 = (payload_hat_attacked[0] + 1.0) * 0.5
                    tensor_to_pil_l(ph1).save(sub / "payload_hat_attacked.png")

        sys.stdout.write("\n")
        sys.stdout.flush()

    mf.close()
    t_total = time.monotonic() - t_start

    summary = {
        "checkpoint": str(ckpt_path),
        "wm_template": str(wm_path),
        "size": int(args.size),
        "n": total,
        "seconds": float(t_total),
        "host_psnr": {
            "mean": float(sum(host_psnrs) / max(1, len(host_psnrs))),
            **_percentiles(host_psnrs, [0, 10, 50, 90, 100]),
        },
        "host_ssim": {
            "mean": float(sum(host_ssims) / max(1, len(host_ssims))),
            **_percentiles(host_ssims, [0, 10, 50, 90, 100]),
        },
        "wm_psnr": {
            "mean": float(sum(wm_psnrs) / max(1, len(wm_psnrs))),
            **_percentiles(wm_psnrs, [0, 10, 50, 90, 100]),
        },
        "wm_ssim": {
            "mean": float(sum(wm_ssims) / max(1, len(wm_ssims))),
            **_percentiles(wm_ssims, [0, 10, 50, 90, 100]),
        },
        "wm_psnr_attack": {
            "mean": float(sum(wm_psnrs_attack) / max(1, len(wm_psnrs_attack))) if wm_psnrs_attack else float("nan"),
            **_percentiles(wm_psnrs_attack, [0, 10, 50, 90, 100]),
        },
        "wm_ssim_attack": {
            "mean": float(sum(wm_ssims_attack) / max(1, len(wm_ssims_attack))) if wm_ssims_attack else float("nan"),
            **_percentiles(wm_ssims_attack, [0, 10, 50, 90, 100]),
        },
        "payload_std": {
            "mean": float(sum(payload_stds) / max(1, len(payload_stds))),
            **_percentiles(payload_stds, [0, 10, 50, 90, 100]),
        },
        "payload_adjcorr": {
            "mean": float(sum(payload_corrs) / max(1, len(payload_corrs))),
            **_percentiles(payload_corrs, [0, 10, 50, 90, 100]),
        },
        "attack": bool(args.attack),
        "attack_mode": str(args.attack_mode),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote: {metrics_path}", file=sys.stderr)
    print(f"wrote: {outdir / 'summary.json'}", file=sys.stderr)
    if save_paths:
        print(f"saved samples to: {samples_dir} ({len(save_paths)}/{total})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
