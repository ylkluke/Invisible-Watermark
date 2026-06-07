from __future__ import annotations

import io
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def ste_round(x: torch.Tensor) -> torch.Tensor:
    """
    Straight-through estimator for round(): forward uses round, backward uses identity.
    """
    return x + (torch.round(x) - x).detach()


@dataclass(frozen=True)
class AttackConfig:
    prob: float = 0.7

    crop_keep: float = 0.7
    crop_prob: float = 0.5

    mosaic_prob: float = 0.3
    mosaic_min_block: int = 4
    mosaic_max_block: int = 16

    jpeg_prob: float = 0.3
    jpeg_quality_min: int = 30
    jpeg_quality_max: int = 95
    jpeg_mode: str = "approx"  # approx | pil

    resize_prob: float = 0.3
    resize_min_scale: float = 0.7
    resize_max_scale: float = 1.3

    blur_prob: float = 0.3
    blur_kernel: int = 5
    blur_sigma_max: float = 1.2


def _rand_bool(rng: random.Random, p: float) -> bool:
    if p <= 0:
        return False
    if p >= 1:
        return True
    return rng.random() < p


def random_crop_keep(x: torch.Tensor, *, keep: float, rng: random.Random) -> torch.Tensor:
    """
    Random crop (keep area fraction) then resize back to original size.
    x: (B,C,H,W) in [0,1]
    """
    if x.ndim != 4:
        raise ValueError("expected BCHW")
    b, c, h, w = x.shape
    keep = float(keep)
    keep = max(0.05, min(1.0, keep))
    if keep >= 0.999 or h < 4 or w < 4:
        return x

    side = keep**0.5
    ch = max(2, int(round(h * side)))
    cw = max(2, int(round(w * side)))
    ch = min(ch, h)
    cw = min(cw, w)

    y0 = 0 if h == ch else rng.randint(0, h - ch)
    x0 = 0 if w == cw else rng.randint(0, w - cw)
    cropped = x[:, :, y0 : y0 + ch, x0 : x0 + cw]
    return F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)


def mosaic(x: torch.Tensor, *, block: int) -> torch.Tensor:
    """
    Pixelation: downsample then upsample with nearest.
    """
    if x.ndim != 4:
        raise ValueError("expected BCHW")
    b, c, h, w = x.shape
    block = int(block)
    block = max(2, min(block, min(h, w)))
    dh = max(1, h // block)
    dw = max(1, w // block)
    small = F.interpolate(x, size=(dh, dw), mode="bilinear", align_corners=False)
    return F.interpolate(small, size=(h, w), mode="nearest")


def random_resize_back(x: torch.Tensor, *, min_scale: float, max_scale: float, rng: random.Random) -> torch.Tensor:
    """
    Resize to random scale then resize back.
    """
    if x.ndim != 4:
        raise ValueError("expected BCHW")
    b, c, h, w = x.shape
    s0 = float(min_scale)
    s1 = float(max_scale)
    if s0 > s1:
        s0, s1 = s1, s0
    s0 = max(0.2, s0)
    s1 = min(3.0, s1)
    if abs(s1 - s0) < 1e-6:
        return x
    s = rng.uniform(s0, s1)
    nh = max(2, int(round(h * s)))
    nw = max(2, int(round(w * s)))
    tmp = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
    return F.interpolate(tmp, size=(h, w), mode="bilinear", align_corners=False)


def gaussian_blur(x: torch.Tensor, *, k: int, sigma: float) -> torch.Tensor:
    """
    Depthwise Gaussian blur. Differentiable.
    """
    if x.ndim != 4:
        raise ValueError("expected BCHW")
    b, c, h, w = x.shape
    k = int(k)
    if k < 3:
        return x
    if k % 2 == 0:
        k += 1
    sigma = float(sigma)
    if sigma <= 0:
        return x

    device = x.device
    dtype = x.dtype
    r = k // 2
    xs = torch.arange(-r, r + 1, device=device, dtype=dtype)
    g = torch.exp(-(xs * xs) / (2.0 * sigma * sigma + 1e-12))
    g = g / (g.sum() + 1e-12)
    k2 = (g[:, None] * g[None, :]).contiguous()
    k2 = k2.view(1, 1, k, k).repeat(c, 1, 1, 1)
    xpad = F.pad(x, (r, r, r, r), mode="reflect")
    return F.conv2d(xpad, k2, groups=c)


def jpeg_approx(x: torch.Tensor, *, quality: int) -> torch.Tensor:
    """
    Rough JPEG-like artifact simulation (differentiable with STE quantization).
    It does NOT implement true JPEG DCT; it mainly adds quantization artifacts.
    """
    if x.ndim != 4:
        raise ValueError("expected BCHW")
    q = int(quality)
    q = max(1, min(100, q))
    # map quality to quantization step (lower quality => larger step)
    step = 1.0 + (100.0 - float(q)) / 100.0 * 15.0  # ~[1..16]
    y = x.clamp(0.0, 1.0) * 255.0
    yq = ste_round(y / step) * step
    return (yq / 255.0).clamp(0.0, 1.0)


def jpeg_pil(x: torch.Tensor, *, quality: int) -> torch.Tensor:
    """
    True JPEG encode/decode using Pillow, per-image on CPU.
    Non-differentiable; use STE wrapper at call site if training needs gradients.
    """
    try:
        from PIL import Image  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Pillow is required for jpeg_pil: {e}")

    if x.ndim != 4:
        raise ValueError("expected BCHW")
    b, c, h, w = x.shape
    if c != 3:
        raise ValueError("jpeg_pil expects 3-channel RGB")
    q = int(quality)
    q = max(1, min(100, q))

    out = []
    x_cpu = x.detach().clamp(0.0, 1.0).mul(255.0).to(torch.uint8).cpu()
    for i in range(b):
        arr = x_cpu[i].permute(1, 2, 0).contiguous()
        # Avoid hard dependency on numpy: build from raw bytes.
        raw = bytes(arr.view(-1).tolist())
        img = Image.frombytes("RGB", (w, h), raw)
        bio = io.BytesIO()
        img.save(bio, format="JPEG", quality=q, optimize=False)
        bio.seek(0)
        img2 = Image.open(bio).convert("RGB")
        b2 = bytearray(img2.tobytes())
        t = torch.frombuffer(b2, dtype=torch.uint8).to(torch.float32)
        t = t.view(h, w, 3).permute(2, 0, 1).contiguous() / 255.0
        out.append(t)
    return torch.stack(out, dim=0).to(device=x.device, dtype=x.dtype)


def apply_random_attacks(x: torch.Tensor, *, cfg: AttackConfig, rng: random.Random, allow_pil_jpeg: bool = False) -> torch.Tensor:
    """
    Apply a random combination of attacks. Output in [0,1].
    """
    if x.ndim != 4:
        raise ValueError("expected BCHW")

    if not _rand_bool(rng, float(cfg.prob)):
        return x

    y = x
    if _rand_bool(rng, float(cfg.crop_prob)):
        y = random_crop_keep(y, keep=float(cfg.crop_keep), rng=rng)

    if _rand_bool(rng, float(cfg.resize_prob)):
        y = random_resize_back(y, min_scale=float(cfg.resize_min_scale), max_scale=float(cfg.resize_max_scale), rng=rng)

    if _rand_bool(rng, float(cfg.mosaic_prob)):
        block = rng.randint(int(cfg.mosaic_min_block), int(cfg.mosaic_max_block))
        y = mosaic(y, block=block)

    if _rand_bool(rng, float(cfg.blur_prob)):
        sigma = rng.uniform(0.2, float(cfg.blur_sigma_max))
        y = gaussian_blur(y, k=int(cfg.blur_kernel), sigma=sigma)

    if _rand_bool(rng, float(cfg.jpeg_prob)):
        q = rng.randint(int(cfg.jpeg_quality_min), int(cfg.jpeg_quality_max))
        mode = str(cfg.jpeg_mode).strip().lower()
        if mode == "pil":
            if not allow_pil_jpeg:
                y = jpeg_approx(y, quality=q)
            else:
                y = jpeg_pil(y, quality=q)
        else:
            y = jpeg_approx(y, quality=q)

    return y.clamp(0.0, 1.0)
