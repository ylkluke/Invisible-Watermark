from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, C, H, W) in [0,1]
    returns: (B, 1, H, W) gradient magnitude (approx), detached? (no, keep differentiable)
    """
    if x.ndim != 4:
        raise ValueError("expected BCHW")
    b, c, h, w = x.shape
    dtype = x.dtype
    device = x.device
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype, device=device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=dtype, device=device).view(1, 1, 3, 3)
    kx = kx.repeat(c, 1, 1, 1)
    ky = ky.repeat(c, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=c)
    gy = F.conv2d(x, ky, padding=1, groups=c)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)  # (B, C, H, W)
    return mag.mean(dim=1, keepdim=True)


def adaptive_embedding_loss(cover: torch.Tensor, container: torch.Tensor, *, eps: float = 1e-6, normalize: bool = True) -> torch.Tensor:
    """
    Penalize container-cover more in smooth areas and less near edges.
    """
    if cover.shape != container.shape:
        raise ValueError("cover/container shape mismatch")
    g = sobel_magnitude(cover)
    w = 1.0 / (g + float(eps))
    if normalize:
        w = w / (w.mean() + 1e-12)
    diff2 = (cover - container) ** 2
    diff2 = diff2.mean(dim=1, keepdim=True)  # (B,1,H,W)
    return (w * diff2).mean()


def _gaussian_kernel2d(window_size: int, sigma: float, *, device, dtype) -> torch.Tensor:
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    radius = window_size // 2
    xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g = torch.exp(-(xs * xs) / (2.0 * (sigma * sigma) + 1e-12))
    g = g / (g.sum() + 1e-12)
    k = (g[:, None] * g[None, :]).contiguous()
    return k


def ssim(x: torch.Tensor, y: torch.Tensor, *, data_range: float = 1.0, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """
    Differentiable SSIM (mean over batch+channels+spatial).
    x,y: (B, C, H, W) in [0,1]
    returns: scalar tensor in ~[0,1]
    """
    if x.shape != y.shape:
        raise ValueError("ssim shape mismatch")
    if x.ndim != 4:
        raise ValueError("expected BCHW")
    b, c, _, _ = x.shape
    device = x.device
    dtype = x.dtype

    k = _gaussian_kernel2d(window_size, sigma, device=device, dtype=dtype).view(1, 1, window_size, window_size)
    k = k.repeat(c, 1, 1, 1)

    mu_x = F.conv2d(x, k, padding=window_size // 2, groups=c)
    mu_y = F.conv2d(y, k, padding=window_size // 2, groups=c)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, k, padding=window_size // 2, groups=c) - mu_x2
    sigma_y2 = F.conv2d(y * y, k, padding=window_size // 2, groups=c) - mu_y2
    sigma_xy = F.conv2d(x * y, k, padding=window_size // 2, groups=c) - mu_xy

    k1 = 0.01
    k2 = 0.03
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = num / (den + 1e-12)
    return ssim_map.mean()


def reveal_loss(watermark: torch.Tensor, revealed: torch.Tensor, *, alpha_ssim: float = 0.5) -> tuple[torch.Tensor, dict[str, float]]:
    """
    L = MSE + alpha*(1-SSIM)
    """
    mse = F.mse_loss(revealed, watermark)
    s = ssim(revealed, watermark)
    loss = mse + float(alpha_ssim) * (1.0 - s)
    return loss, {"mse": float(mse.detach().item()), "ssim": float(s.detach().item())}


def psnr(x: torch.Tensor, y: torch.Tensor, *, data_range: float = 1.0) -> torch.Tensor:
    mse = F.mse_loss(x, y)
    if float(mse.detach().item()) == 0.0:
        return torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
    return 10.0 * torch.log10((data_range * data_range) / mse)


def adjacent_corr_penalty(x: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """
    Penalize adjacent-pixel correlation to encourage noise-like textures.
    x: (B, C, H, W)
    """
    if x.ndim != 4:
        raise ValueError("expected BCHW")
    b, c, h, w = x.shape
    if h < 2 or w < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)

    def corr(a: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
        a0 = a - a.mean()
        b0 = b2 - b2.mean()
        num = (a0 * b0).mean()
        den = a0.std(unbiased=False) * b0.std(unbiased=False) + eps
        return num / den

    xh_a = x[:, :, :, :-1]
    xh_b = x[:, :, :, 1:]
    xv_a = x[:, :, :-1, :]
    xv_b = x[:, :, 1:, :]
    xd_a = x[:, :, :-1, :-1]
    xd_b = x[:, :, 1:, 1:]

    ch = corr(xh_a, xh_b)
    cv = corr(xv_a, xv_b)
    cd = corr(xd_a, xd_b)
    return ch * ch + cv * cv + cd * cd


def noise_stats_loss(x: torch.Tensor, *, target_std: float = 0.35) -> torch.Tensor:
    """
    Encourage ~zero-mean, fixed-std payload. Useful for 'noise-like' payload regularization.
    x: (B,C,H,W)
    """
    mean = x.mean()
    std = x.std(unbiased=False)
    return mean * mean + (std - float(target_std)) ** 2
