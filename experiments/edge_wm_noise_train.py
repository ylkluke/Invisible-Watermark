from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path


def _require_deps() -> tuple[object, object]:
    try:
        import torch  # type: ignore
    except Exception as e:
        print("torch is required for training but is not installed:", e, file=sys.stderr)
        print("Install it in your environment, then re-run.", file=sys.stderr)
        raise SystemExit(2)

    try:
        from PIL import Image  # type: ignore
    except Exception as e:
        print("Pillow is required for image IO but is not installed:", e, file=sys.stderr)
        print("Install it in your environment, then re-run.", file=sys.stderr)
        raise SystemExit(2)

    return torch, Image


def _safe_extract_zip(zip_path: Path, *, extract_to: Path) -> None:
    import shutil
    import zipfile

    extract_to.mkdir(parents=True, exist_ok=True)
    base = extract_to.resolve()

    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if not name:
                continue
            parts = [p for p in name.split("/") if p not in ("", ".")]
            if any(p == ".." for p in parts):
                raise ValueError(f"refuse to extract path traversal entry: {name}")
            out_path = (base / Path(*parts)).resolve()
            if out_path != base and base not in out_path.parents:
                raise ValueError(f"refuse to extract outside destination: {name}")
            if name.endswith("/"):
                out_path.mkdir(parents=True, exist_ok=True)
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)


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
    from src.nn.edge_wm.losses import (  # noqa: E402
        adaptive_embedding_loss,
        adjacent_corr_penalty,
        noise_stats_loss,
        psnr,
        reveal_loss,
        ssim,
    )
    from src.nn.edge_wm.noise_wm_model import NoiseWatermarkConfig, NoiseWatermarkNet  # noqa: E402
    from src.nn.edge_wm.attacks import AttackConfig, apply_random_attacks  # noqa: E402

    DEFAULT_CONFIG = REPO_ROOT / "config.toml"
    cfg_all = load_toml(DEFAULT_CONFIG) if DEFAULT_CONFIG.exists() else {}
    cfg_paths = cfg_all.get("paths", {}) if isinstance(cfg_all.get("paths", {}), dict) else {}
    cfg = cfg_all.get("edge_wm_noise_train", {}) if isinstance(cfg_all.get("edge_wm_noise_train", {}), dict) else {}
    outputs_dir = str(cfg_paths.get("outputs_dir", "outputs"))
    clean_outputs_default = bool(cfg_paths.get("clean_outputs", True))

    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=str, default=str(cfg.get("input_dir", "inputs/div2k/DIV2K_train_HR")))
    p.add_argument("--auto-extract", action=argparse.BooleanOptionalAction, default=bool(cfg.get("auto_extract", True)))
    p.add_argument("--extract-dir", type=str, default=str(cfg.get("extract_dir", "")))
    p.add_argument("--wm-template", type=str, default=str(cfg.get("wm_template", "assets/watermarks/luke.png")))
    p.add_argument("--outdir", type=str, default=str(cfg.get("outdir", "")))
    p.add_argument("--resume", type=str, default=str(cfg.get("resume", "")))
    p.add_argument("--size", type=int, default=int(cfg.get("size", 256)))
    p.add_argument("--steps", type=int, default=int(cfg.get("steps", 2000)))
    p.add_argument("--batch-size", type=int, default=int(cfg.get("batch_size", 4)))
    p.add_argument("--num-workers", type=int, default=int(cfg.get("num_workers", 2)))
    p.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=bool(cfg.get("recursive", True)))
    p.add_argument("--max-images", type=int, default=int(cfg.get("max_images", 0)))

    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=bool(cfg.get("augment", True)))
    p.add_argument("--hflip", action=argparse.BooleanOptionalAction, default=bool(cfg.get("hflip", True)))
    p.add_argument("--crop-scale-min", type=float, default=float(cfg.get("crop_scale_min", 0.70)))
    p.add_argument("--crop-scale-max", type=float, default=float(cfg.get("crop_scale_max", 1.00)))
    p.add_argument("--crop-ratio-min", type=float, default=float(cfg.get("crop_ratio_min", 0.85)))
    p.add_argument("--crop-ratio-max", type=float, default=float(cfg.get("crop_ratio_max", 1.15)))

    p.add_argument("--lr", type=float, default=float(cfg.get("lr", 0.0001)))
    p.add_argument("--device", type=str, default=str(cfg.get("device", "cpu")))
    p.add_argument("--seed", type=int, default=int(cfg.get("seed", 0)))

    p.add_argument("--lambda-embed", type=float, default=float(cfg.get("lambda_embed", 20.0)))
    p.add_argument("--lambda-hvs", type=float, default=float(cfg.get("lambda_hvs", 20.0)))
    p.add_argument("--lambda-payload", type=float, default=float(cfg.get("lambda_payload", 1.0)))
    p.add_argument("--lambda-wm", type=float, default=float(cfg.get("lambda_wm", 1.0)))
    p.add_argument("--lambda-noise", type=float, default=float(cfg.get("lambda_noise", 0.5)))
    p.add_argument("--alpha-ssim", type=float, default=float(cfg.get("alpha_ssim", 0.5)))
    p.add_argument("--payload-target-std", type=float, default=float(cfg.get("payload_target_std", 0.35)))

    # Robust attacks (train decoder on attacked container)
    p.add_argument("--robust", action=argparse.BooleanOptionalAction, default=bool(cfg.get("robust", False)))
    p.add_argument("--lambda-attack", type=float, default=float(cfg.get("lambda_attack", 1.0)))
    p.add_argument("--attack-prob", type=float, default=float(cfg.get("attack_prob", 0.7)))
    p.add_argument("--attack-crop-keep", type=float, default=float(cfg.get("attack_crop_keep", 0.7)))
    p.add_argument("--attack-crop-prob", type=float, default=float(cfg.get("attack_crop_prob", 0.5)))
    p.add_argument("--attack-mosaic-prob", type=float, default=float(cfg.get("attack_mosaic_prob", 0.3)))
    p.add_argument("--attack-mosaic-min-block", type=int, default=int(cfg.get("attack_mosaic_min_block", 4)))
    p.add_argument("--attack-mosaic-max-block", type=int, default=int(cfg.get("attack_mosaic_max_block", 16)))
    p.add_argument("--attack-jpeg-prob", type=float, default=float(cfg.get("attack_jpeg_prob", 0.3)))
    p.add_argument("--attack-jpeg-quality-min", type=int, default=int(cfg.get("attack_jpeg_quality_min", 30)))
    p.add_argument("--attack-jpeg-quality-max", type=int, default=int(cfg.get("attack_jpeg_quality_max", 95)))
    p.add_argument("--attack-jpeg-mode", type=str, default=str(cfg.get("attack_jpeg_mode", "approx")), choices=["approx", "pil"])
    p.add_argument("--attack-resize-prob", type=float, default=float(cfg.get("attack_resize_prob", 0.3)))
    p.add_argument("--attack-resize-min-scale", type=float, default=float(cfg.get("attack_resize_min_scale", 0.7)))
    p.add_argument("--attack-resize-max-scale", type=float, default=float(cfg.get("attack_resize_max_scale", 1.3)))
    p.add_argument("--attack-blur-prob", type=float, default=float(cfg.get("attack_blur_prob", 0.3)))
    p.add_argument("--attack-blur-kernel", type=int, default=int(cfg.get("attack_blur_kernel", 5)))
    p.add_argument("--attack-blur-sigma-max", type=float, default=float(cfg.get("attack_blur_sigma_max", 1.2)))

    p.add_argument("--base-channels", type=int, default=int(cfg.get("base_channels", 32)))
    p.add_argument("--unet-depth", type=int, default=int(cfg.get("unet_depth", 4)))
    p.add_argument("--residual-strength", type=float, default=float(cfg.get("residual_strength", 0.02)))
    p.add_argument("--wm-enc-blocks", type=int, default=int(cfg.get("wm_enc_blocks", 4)))
    p.add_argument("--wm-dec-blocks", type=int, default=int(cfg.get("wm_dec_blocks", 4)))
    p.add_argument("--payload-dec-blocks", type=int, default=int(cfg.get("payload_dec_blocks", 6)))
    p.add_argument("--norm", type=str, default=str(cfg.get("norm", "bn")), choices=["bn", "in", "none"])
    p.add_argument("--attention", type=str, default=str(cfg.get("attention", "none")), choices=["none", "cbam"])
    p.add_argument("--attention-reduction", type=int, default=int(cfg.get("attention_reduction", 16)))
    p.add_argument("--attention-spatial-kernel", type=int, default=int(cfg.get("attention_spatial_kernel", 7)), choices=[3, 7])

    p.add_argument("--progress", action=argparse.BooleanOptionalAction, default=bool(cfg.get("progress", True)))
    p.add_argument("--progress-every", type=int, default=int(cfg.get("progress_every", 10)))
    p.add_argument("--log-every", type=int, default=int(cfg.get("log_every", 50)))
    p.add_argument("--save-every", type=int, default=int(cfg.get("save_every", 200)))
    p.add_argument("--print-json", action=argparse.BooleanOptionalAction, default=bool(cfg.get("print_json", False)))
    p.add_argument("--metrics-jsonl", type=str, default=str(cfg.get("metrics_jsonl", "metrics.jsonl")))
    p.add_argument("--clean-outputs", action=argparse.BooleanOptionalAction, default=bool(cfg.get("clean_outputs", False if clean_outputs_default else False)))

    # Validation (automatic test set during training)
    p.add_argument("--val-dir", type=str, default=str(cfg.get("val_dir", "")))
    p.add_argument("--val-recursive", action=argparse.BooleanOptionalAction, default=bool(cfg.get("val_recursive", True)))
    p.add_argument("--val-limit", type=int, default=int(cfg.get("val_limit", 0)), help="0 means no limit")
    p.add_argument("--val-every", type=int, default=int(cfg.get("val_every", 0)), help="0 disables validation")
    p.add_argument("--val-save-samples", type=int, default=int(cfg.get("val_save_samples", 4)))
    p.add_argument("--val-save-amplify", type=float, default=float(cfg.get("val_save_amplify", 20.0)))
    p.add_argument("--val-jsonl", type=str, default=str(cfg.get("val_jsonl", "val_metrics.jsonl")))
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

    in_dir = Path(args.input_dir)
    if args.auto_extract:
        if in_dir.suffix.lower() == ".zip" and in_dir.exists():
            zip_path = in_dir
            extract_base = Path(args.extract_dir) if args.extract_dir else zip_path.parent
            expected = extract_base / zip_path.stem
            if not expected.exists():
                print(f"extracting {zip_path} -> {extract_base}", file=sys.stderr)
                _safe_extract_zip(zip_path, extract_to=extract_base)
            in_dir = expected
        elif not in_dir.exists():
            zip_candidate = in_dir.with_suffix(".zip")
            if zip_candidate.exists():
                extract_base = Path(args.extract_dir) if args.extract_dir else zip_candidate.parent
                expected = extract_base / zip_candidate.stem
                if not expected.exists():
                    print(f"extracting {zip_candidate} -> {extract_base}", file=sys.stderr)
                    _safe_extract_zip(zip_candidate, extract_to=extract_base)
                in_dir = expected

    if not in_dir.exists() or not in_dir.is_dir():
        raise SystemExit(f"input dir not found: {in_dir}")

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    it = in_dir.rglob("*") if args.recursive else in_dir.iterdir()
    paths = sorted([p2 for p2 in it if p2.is_file() and p2.suffix.lower() in exts])
    if not paths:
        raise SystemExit(f"no images found in input dir: {in_dir}")
    if args.max_images and int(args.max_images) > 0 and len(paths) > int(args.max_images):
        paths = paths[: int(args.max_images)]

    wm_path = Path(args.wm_template)
    if not wm_path.exists():
        raise SystemExit(f"watermark template not found: {wm_path}")

    maybe_clean_outputs(repo_root=REPO_ROOT, outputs_dir=outputs_dir, enabled=bool(args.clean_outputs))
    outdir = Path(args.outdir) if args.outdir else Path(outputs_dir) / "edge_wm_noise" / time.strftime("%Y%m%d_%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(str(args.device))
    resample_lanczos = getattr(getattr(Image, "Resampling", None), "LANCZOS", Image.LANCZOS)

    def resize_square(img, size: int):
        return img.resize((size, size), resample=resample_lanczos)

    def random_resized_crop(img, *, size: int):
        w, h = img.size
        if w < 2 or h < 2:
            return resize_square(img, size)
        area = w * h
        for _ in range(10):
            target = random.uniform(float(args.crop_scale_min), float(args.crop_scale_max)) * area
            ratio = random.uniform(float(args.crop_ratio_min), float(args.crop_ratio_max))
            nw = int(round((target * ratio) ** 0.5))
            nh = int(round((target / ratio) ** 0.5))
            if 1 <= nw <= w and 1 <= nh <= h:
                x0 = 0 if w == nw else random.randint(0, w - nw)
                y0 = 0 if h == nh else random.randint(0, h - nh)
                crop = img.crop((x0, y0, x0 + nw, y0 + nh))
                return resize_square(crop, size)
        side = min(w, h)
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        crop = img.crop((x0, y0, x0 + side, y0 + side))
        return resize_square(crop, size)

    def pil_to_tensor_rgb(img) -> "torch.Tensor":
        img = img.convert("RGB")
        b = bytearray(img.tobytes())
        w, h = img.size
        x = torch.frombuffer(b, dtype=torch.uint8).to(torch.float32)
        x = x.view(h, w, 3).permute(2, 0, 1).contiguous() / 255.0
        return x

    def pil_to_tensor_l(img) -> "torch.Tensor":
        img = img.convert("L")
        b = bytearray(img.tobytes())
        w, h = img.size
        x = torch.frombuffer(b, dtype=torch.uint8).to(torch.float32)
        x = x.view(h, w).unsqueeze(0).contiguous() / 255.0
        return x

    def pil_to_tensor_ycbcr(img) -> "torch.Tensor":
        img = img.convert("YCbCr")
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
            raise ValueError("expected 3-channel tensor for YCbCr output")
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

    def tensor_to_pil_l(x01: "torch.Tensor"):
        x01 = x01.detach().clamp(0.0, 1.0).mul(255.0).to(torch.uint8).contiguous().cpu()
        h, w = int(x01.shape[-2]), int(x01.shape[-1])
        b = bytes(x01.reshape(-1).tolist())
        return Image.frombytes("L", (w, h), b)

    def safe_stem(stem: str) -> str:
        s = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in stem)
        return (s[:80] or "image").strip("_") or "image"

    watermark_img = Image.open(wm_path)
    watermark_img = resize_square(watermark_img, int(args.size))
    wm0_ycbcr = pil_to_tensor_ycbcr(watermark_img).to(device)  # (3,H,W)
    wm0 = wm0_ycbcr[:1]  # only watermark Y is encoded/recovered
    wm0_cbcr = wm0_ycbcr[1:]  # template CbCr is reused for RGB visualization

    cfg = NoiseWatermarkConfig(
        cover_channels=3,
        wm_channels=1,
        payload_channels=1,
        base_channels=int(args.base_channels),
        unet_depth=int(args.unet_depth),
        residual_strength=float(args.residual_strength),
        wm_enc_blocks=int(args.wm_enc_blocks),
        wm_dec_blocks=int(args.wm_dec_blocks),
        payload_dec_blocks=int(args.payload_dec_blocks),
        norm=str(args.norm),
        attention=str(args.attention),
        attention_reduction=int(args.attention_reduction),
        attention_spatial_kernel=int(args.attention_spatial_kernel),
    )
    model = NoiseWatermarkNet(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    start_step = 1

    if str(args.resume).strip():
        resume_path = Path(str(args.resume).strip())
        if resume_path.is_dir():
            candidates = sorted(resume_path.glob("ckpt_step_*.pt"))
            if not candidates:
                raise SystemExit(f"no checkpoints found in resume dir: {resume_path}")
            resume_path = candidates[-1]
        if not resume_path.exists():
            raise SystemExit(f"resume checkpoint not found: {resume_path}")
        ckpt = torch.load(str(resume_path), map_location="cpu")
        if not isinstance(ckpt, dict) or "model_state" not in ckpt:
            raise SystemExit("invalid resume checkpoint")
        try:
            model.load_state_dict(ckpt["model_state"], strict=True)  # type: ignore[arg-type]
        except Exception as e:
            raise SystemExit(f"failed to load resume model_state: {e}")
        if "optimizer_state" in ckpt:
            try:
                opt.load_state_dict(ckpt["optimizer_state"])  # type: ignore[arg-type]
            except Exception:
                pass
        prev_step = int(ckpt.get("step", 0) or 0)
        start_step = max(1, prev_step + 1)
        print(f"resumed from {resume_path} (step={prev_step})", file=sys.stderr)

    class CoverDataset(torch.utils.data.Dataset):
        def __init__(self, items: list[Path]) -> None:
            self.items = items

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int):
            pth = self.items[idx]
            img = Image.open(pth).convert("RGB")
            if args.augment:
                img = random_resized_crop(img, size=int(args.size))
                if args.hflip and random.random() < 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                img = resize_square(img, int(args.size))
            return img, str(pth)

    def worker_init_fn(worker_id: int) -> None:
        base = int(args.seed) if args.seed else 12345
        seed = base + worker_id
        random.seed(seed)
        torch.manual_seed(seed)

    ds = CoverDataset(paths)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type != "cpu"),
        worker_init_fn=worker_init_fn,
        collate_fn=lambda batch: batch,
    )
    it_loader = iter(loader)

    total_steps = int(args.steps)
    log_every = max(1, int(args.log_every))
    save_every = max(1, int(args.save_every))
    progress_every = max(1, int(args.progress_every))
    avg_step_s = 0.0

    metrics_path = outdir / str(args.metrics_jsonl or "metrics.jsonl")
    metrics_f = open(metrics_path, "a", encoding="utf-8")
    history: list[dict[str, float]] = []

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

    # Validation set (optional)
    val_paths: list[Path] = []
    if str(args.val_dir).strip():
        vdir = Path(str(args.val_dir).strip())
        if not vdir.exists() or not vdir.is_dir():
            raise SystemExit(f"val dir not found: {vdir}")
        vit = vdir.rglob("*") if args.val_recursive else vdir.iterdir()
        val_paths = sorted([p2 for p2 in vit if p2.is_file() and p2.suffix.lower() in exts])
        if not val_paths:
            raise SystemExit(f"no images found in val dir: {vdir}")
        if args.val_limit and int(args.val_limit) > 0 and len(val_paths) > int(args.val_limit):
            val_paths = val_paths[: int(args.val_limit)]

    val_every = max(0, int(args.val_every))
    val_metrics_f = None
    val_history: list[dict[str, object]] = []
    val_sample_paths: list[Path] = []
    if val_paths and val_every:
        val_path = outdir / str(args.val_jsonl or "val_metrics.jsonl")
        val_metrics_f = open(val_path, "a", encoding="utf-8")
        k = max(0, int(args.val_save_samples))
        if k:
            rng = random.Random(int(args.seed) if args.seed else 12345)
            val_sample_paths = list(val_paths)
            rng.shuffle(val_sample_paths)
            val_sample_paths = val_sample_paths[: min(k, len(val_sample_paths))]

    def _percentiles(xs: list[float], ps: list[float]) -> dict[str, float]:
        if not xs:
            return {f"p{int(p2)}": float("nan") for p2 in ps}
        ys = sorted(xs)
        n = len(ys)
        out: dict[str, float] = {}
        for p2 in ps:
            if p2 <= 0:
                out[f"p{int(p2)}"] = float(ys[0])
                continue
            if p2 >= 100:
                out[f"p{int(p2)}"] = float(ys[-1])
                continue
            k2 = (p2 / 100.0) * (n - 1)
            i2 = int(k2)
            j2 = min(n - 1, i2 + 1)
            t2 = k2 - i2
            out[f"p{int(p2)}"] = float((1.0 - t2) * ys[i2] + t2 * ys[j2])
        return out

    def run_validation(step: int) -> None:
        if not val_paths or not val_every or val_metrics_f is None:
            return

        vrng = random.Random((int(args.seed) if args.seed else 12345) + int(step))
        host_psnrs: list[float] = []
        host_ssims: list[float] = []
        wm_psnrs: list[float] = []
        wm_ssims: list[float] = []
        wm_psnrs_attack: list[float] = []
        wm_ssims_attack: list[float] = []
        payload_stds: list[float] = []
        payload_corrs: list[float] = []

        was_training = bool(getattr(model, "training", False))
        model.eval()
        with torch.no_grad():
            for pth in val_paths:
                cover_img = Image.open(pth).convert("RGB")
                cover_img = resize_square(cover_img, int(args.size))
                cover = pil_to_tensor_rgb(cover_img).unsqueeze(0).to(device)
                wm = wm0.unsqueeze(0).expand(cover.shape[0], -1, -1, -1)

                payload = model.wm_to_noise(wm)
                container, _resid = model.embedder(cover, payload)
                payload_hat = model.reveal_payload(container)
                wm_hat = model.noise_to_wm(payload_hat)

                host_psnrs.append(float(psnr(cover, container).item()))
                host_ssims.append(float(ssim(cover, container).item()))
                wm_psnrs.append(float(psnr(wm, wm_hat).item()))
                wm_ssims.append(float(ssim(wm, wm_hat).item()))
                payload_stds.append(float(payload.std(unbiased=False).item()))
                payload_corrs.append(float(adjacent_corr_penalty(payload).sqrt().item()))

                if bool(args.robust):
                    attacked = apply_random_attacks(container, cfg=attack_cfg, rng=vrng, allow_pil_jpeg=(str(args.attack_jpeg_mode) == "pil"))
                    payload_hat_a = model.reveal_payload(attacked)
                    wm_hat_a = model.noise_to_wm(payload_hat_a)
                    wm_psnrs_attack.append(float(psnr(wm, wm_hat_a).item()))
                    wm_ssims_attack.append(float(ssim(wm, wm_hat_a).item()))

            if val_sample_paths:
                vout = outdir / "val_samples" / f"step_{step:07d}"
                vout.mkdir(parents=True, exist_ok=True)
                for sp in val_sample_paths:
                    stem = safe_stem(sp.stem)
                    sub = vout / stem
                    sub.mkdir(parents=True, exist_ok=True)

                    cover_img = Image.open(sp).convert("RGB")
                    cover_img = resize_square(cover_img, int(args.size))
                    cover = pil_to_tensor_rgb(cover_img).unsqueeze(0).to(device)
                    wm = wm0.unsqueeze(0).expand(cover.shape[0], -1, -1, -1)
                    payload = model.wm_to_noise(wm)
                    container, _resid = model.embedder(cover, payload)
                    payload_hat = model.reveal_payload(container)
                    wm_hat = model.noise_to_wm(payload_hat)

                    tensor_to_pil_rgb(cover[0]).save(sub / "cover.png")
                    tensor_to_pil_rgb(container[0]).save(sub / "container.png")
                    resid_abs = (container[0] - cover[0]).abs()
                    resid_amp = (resid_abs * float(args.val_save_amplify)).clamp(0.0, 1.0)
                    tensor_to_pil_rgb(resid_amp).save(sub / "residual_abs_amplified.png")

                    compose_y_with_cbcr_to_pil_rgb(wm[0], wm0_cbcr).save(sub / "wm.png")
                    compose_y_with_cbcr_to_pil_rgb(wm_hat[0], wm0_cbcr).save(sub / "wm_hat.png")

                    p0 = (payload[0] + 1.0) * 0.5
                    ph0 = (payload_hat[0] + 1.0) * 0.5
                    tensor_to_pil_l(p0).save(sub / "payload.png")
                    tensor_to_pil_l(ph0).save(sub / "payload_hat.png")

                    if bool(args.robust):
                        attacked = apply_random_attacks(container, cfg=attack_cfg, rng=vrng, allow_pil_jpeg=(str(args.attack_jpeg_mode) == "pil"))
                        payload_hat_a = model.reveal_payload(attacked)
                        wm_hat_a = model.noise_to_wm(payload_hat_a)
                        tensor_to_pil_rgb(attacked[0]).save(sub / "container_attacked.png")
                        compose_y_with_cbcr_to_pil_rgb(wm_hat_a[0], wm0_cbcr).save(sub / "wm_hat_attacked.png")

        if was_training:
            model.train()

        row: dict[str, object] = {
            "step": int(step),
            "n": int(len(val_paths)),
            "host_psnr": {"mean": float(sum(host_psnrs) / max(1, len(host_psnrs))), **_percentiles(host_psnrs, [10, 50, 90])},
            "host_ssim": {"mean": float(sum(host_ssims) / max(1, len(host_ssims))), **_percentiles(host_ssims, [10, 50, 90])},
            "wm_psnr_clean": {"mean": float(sum(wm_psnrs) / max(1, len(wm_psnrs))), **_percentiles(wm_psnrs, [10, 50, 90])},
            "wm_ssim_clean": {"mean": float(sum(wm_ssims) / max(1, len(wm_ssims))), **_percentiles(wm_ssims, [10, 50, 90])},
            "wm_psnr_attack": {"mean": float(sum(wm_psnrs_attack) / max(1, len(wm_psnrs_attack))), **_percentiles(wm_psnrs_attack, [10, 50, 90])},
            "wm_ssim_attack": {"mean": float(sum(wm_ssims_attack) / max(1, len(wm_ssims_attack))), **_percentiles(wm_ssims_attack, [10, 50, 90])},
            "payload_std": {"mean": float(sum(payload_stds) / max(1, len(payload_stds))), **_percentiles(payload_stds, [10, 50, 90])},
            "payload_adjcorr": {"mean": float(sum(payload_corrs) / max(1, len(payload_corrs))), **_percentiles(payload_corrs, [10, 50, 90])},
            "robust": bool(args.robust),
        }
        val_history.append(row)
        val_metrics_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        val_metrics_f.flush()
        (outdir / f"val_summary_step_{step:07d}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if start_step > total_steps:
        print(f"nothing to do: resume step {start_step} > --steps {total_steps}", file=sys.stderr)
        return 0

    for step in range(start_step, total_steps + 1):
        t0 = time.monotonic()
        try:
            batch = next(it_loader)
        except StopIteration:
            it_loader = iter(loader)
            batch = next(it_loader)

        covers = []
        for cover_pil, _src in batch:
            covers.append(pil_to_tensor_rgb(cover_pil))
        cover = torch.stack(covers, dim=0).to(device, non_blocking=True)
        wm = wm0.unsqueeze(0).expand(cover.shape[0], -1, -1, -1)

        model.train()
        # Build path explicitly so we can optionally decode from attacked container.
        payload = model.wm_to_noise(wm)
        container, _residual = model.embedder(cover, payload)
        payload_hat_clean = model.reveal_payload(container)
        wm_hat_clean = model.noise_to_wm(payload_hat_clean)

        payload_hat = payload_hat_clean
        wm_hat = wm_hat_clean
        container_for_decode = container
        attacked = None
        if bool(args.robust):
            # Attacks may include non-differentiable components; use STE to pass gradients.
            attacked = apply_random_attacks(container, cfg=attack_cfg, rng=random, allow_pil_jpeg=(str(args.attack_jpeg_mode) == "pil"))
            container_for_decode = container + (attacked - container).detach()
            payload_hat = model.reveal_payload(container_for_decode)
            wm_hat = model.noise_to_wm(payload_hat)

        l_embed = torch.nn.functional.mse_loss(container, cover)
        l_hvs = adaptive_embedding_loss(cover, container)
        l_payload_clean = torch.nn.functional.mse_loss(payload_hat_clean, payload.detach())
        l_wm_clean, wm_parts_clean = reveal_loss(wm, wm_hat_clean, alpha_ssim=float(args.alpha_ssim))
        l_payload = l_payload_clean
        l_wm = l_wm_clean
        wm_parts = wm_parts_clean
        l_payload_attack = None
        l_wm_attack = None
        wm_parts_attack = None
        if bool(args.robust):
            l_payload_attack = torch.nn.functional.mse_loss(payload_hat, payload.detach())
            l_wm_attack, wm_parts_attack = reveal_loss(wm, wm_hat, alpha_ssim=float(args.alpha_ssim))
            l_payload = l_payload_clean + float(args.lambda_attack) * l_payload_attack
            l_wm = l_wm_clean + float(args.lambda_attack) * l_wm_attack
            wm_parts = wm_parts_attack if wm_parts_attack is not None else wm_parts_clean
        l_noise = noise_stats_loss(payload, target_std=float(args.payload_target_std)) + adjacent_corr_penalty(payload)

        loss = (
            float(args.lambda_embed) * l_embed
            + float(args.lambda_hvs) * l_hvs
            + float(args.lambda_payload) * l_payload
            + float(args.lambda_wm) * l_wm
            + float(args.lambda_noise) * l_noise
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        dt = time.monotonic() - t0
        avg_step_s = dt if step == 1 else (0.98 * avg_step_s + 0.02 * dt)

        should_log = (step % log_every == 0) or (step == 1) or (step == total_steps)
        if args.progress and (step % progress_every == 0 or should_log):
            eta = (total_steps - step) * avg_step_s
            msg = f"{_format_progress(step=step, total=total_steps)} eta={_format_eta(eta)} loss={float(loss.detach().item()):.4f}"
            sys.stdout.write("\r" + msg[:160].ljust(160))
            sys.stdout.flush()

        if should_log:
            with torch.no_grad():
                host_psnr = float(psnr(cover, container).item())
                host_ssim = float(ssim(cover, container).item())
                wm_psnr = float(psnr(wm, wm_hat).item())
                wm_ssim = float(ssim(wm, wm_hat).item())
                wm_psnr_clean = float(psnr(wm, wm_hat_clean).item())
                wm_ssim_clean = float(ssim(wm, wm_hat_clean).item())

            row = {
                "step": float(step),
                "loss": float(loss.detach().item()),
                "embed_mse": float(l_embed.detach().item()),
                "embed_hvs": float(l_hvs.detach().item()),
                "payload_mse": float(l_payload.detach().item()),
                "payload_mse_clean": float(l_payload_clean.detach().item()),
                "payload_mse_attack": float(l_payload_attack.detach().item()) if l_payload_attack is not None else float("nan"),
                "wm_loss": float(l_wm.detach().item()),
                "wm_loss_clean": float(l_wm_clean.detach().item()),
                "wm_loss_attack": float(l_wm_attack.detach().item()) if l_wm_attack is not None else float("nan"),
                "wm_mse": float(wm_parts["mse"]),
                "wm_ssim": float(wm_parts["ssim"]),
                "noise_reg": float(l_noise.detach().item()),
                "host_psnr": host_psnr,
                "host_ssim": host_ssim,
                "wm_psnr": wm_psnr,
                "wm_ssim": wm_ssim,
                "wm_psnr_clean": wm_psnr_clean,
                "wm_ssim_clean": wm_ssim_clean,
                "robust": bool(args.robust),
            }
            history.append(row)
            metrics_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            metrics_f.flush()

            if args.progress:
                sys.stdout.write("\n")

            if args.print_json:
                print(json.dumps(row, ensure_ascii=False))
            else:
                eta = (total_steps - step) * avg_step_s
                print(
                    f"{_format_progress(step=step, total=total_steps)} eta={_format_eta(eta)} "
                    f"host_psnr={host_psnr:.2f} host_ssim={host_ssim:.4f} "
                    f"wm_psnr={wm_psnr:.2f} wm_ssim={wm_ssim:.4f} "
                    f"noise_reg={row['noise_reg']:.4f}"
                )

        if step % save_every == 0 or step == total_steps:
            outdir.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                i0 = 0
                tensor_to_pil_rgb(cover[i0]).save(outdir / f"step_{step:07d}_cover.png")
                tensor_to_pil_rgb(container[i0]).save(outdir / f"step_{step:07d}_container.png")
                resid = (container[i0] - cover[i0]).abs()
                resid_vis = resid / (resid.max() + 1e-12)
                tensor_to_pil_rgb(resid_vis).save(outdir / f"step_{step:07d}_residual_abs_norm.png")

                compose_y_with_cbcr_to_pil_rgb(wm[i0], wm0_cbcr).save(outdir / f"step_{step:07d}_wm.png")
                compose_y_with_cbcr_to_pil_rgb(wm_hat[i0], wm0_cbcr).save(outdir / f"step_{step:07d}_wm_hat.png")

                p0 = (payload[i0] + 1.0) * 0.5
                ph0 = (payload_hat[i0] + 1.0) * 0.5
                tensor_to_pil_l(p0).save(outdir / f"step_{step:07d}_payload.png")
                tensor_to_pil_l(ph0).save(outdir / f"step_{step:07d}_payload_hat.png")

                if bool(args.robust) and attacked is not None:
                    tensor_to_pil_rgb(attacked[i0]).save(outdir / f"step_{step:07d}_container_attacked.png")
                    compose_y_with_cbcr_to_pil_rgb(wm_hat[i0], wm0_cbcr).save(outdir / f"step_{step:07d}_wm_hat_attacked.png")

            ckpt = {
                "step": step,
                "model_config": asdict(cfg),
                "wm_template": str(wm_path),
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
            }
            torch.save(ckpt, outdir / f"ckpt_step_{step:07d}.pt")
            torch.save(ckpt, outdir / "ckpt_last.pt")
            if val_paths and val_every and (step % val_every == 0 or step == total_steps):
                run_validation(step)

        if val_paths and val_every and (step % val_every == 0) and (step % save_every != 0) and (step != total_steps):
            run_validation(step)

    if args.progress:
        sys.stdout.write("\n")
    metrics_f.close()
    if val_metrics_f is not None:
        val_metrics_f.close()

    (outdir / "train_summary.json").write_text(
        json.dumps(
            {"args": vars(args), "outdir": str(outdir), "model_config": asdict(cfg), "wm_template": str(wm_path), "history": history, "val_history": val_history},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
