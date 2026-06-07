from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .blocks import DoubleConv, Down, ResidualBlock, Up


@dataclass(frozen=True)
class NoiseWatermarkConfig:
    cover_channels: int = 3
    wm_channels: int = 1
    payload_channels: int = 1

    norm: str = "bn"
    base_channels: int = 32
    unet_depth: int = 4
    residual_strength: float = 0.02  # tanh(residual) * strength

    wm_enc_blocks: int = 4
    wm_dec_blocks: int = 4
    payload_dec_blocks: int = 6
    attention: str = "none"
    attention_reduction: int = 16
    attention_spatial_kernel: int = 7


class WatermarkToNoise(nn.Module):
    """
    watermark W -> noise-like payload P in [-1, 1]
    """

    def __init__(self, cfg: NoiseWatermarkConfig) -> None:
        super().__init__()
        ch = int(cfg.base_channels)
        self.stem = DoubleConv(
            int(cfg.wm_channels),
            ch,
            norm=cfg.norm,
            attention=cfg.attention,
            attention_reduction=int(cfg.attention_reduction),
            attention_spatial_kernel=int(cfg.attention_spatial_kernel),
        )
        self.body = nn.Sequential(
            *[
                ResidualBlock(
                    ch,
                    norm=cfg.norm,
                    attention=cfg.attention,
                    attention_reduction=int(cfg.attention_reduction),
                    attention_spatial_kernel=int(cfg.attention_spatial_kernel),
                )
                for _ in range(int(cfg.wm_enc_blocks))
            ]
        )
        self.head = nn.Sequential(
            nn.Conv2d(ch, int(cfg.payload_channels), kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
        )

    def forward(self, wm: torch.Tensor) -> torch.Tensor:
        h = self.body(self.stem(wm))
        return self.head(h)


class NoiseToWatermark(nn.Module):
    """
    payload P_hat in [-1,1] -> watermark W_hat in [0,1]
    """

    def __init__(self, cfg: NoiseWatermarkConfig) -> None:
        super().__init__()
        ch = int(cfg.base_channels)
        self.stem = DoubleConv(
            int(cfg.payload_channels),
            ch,
            norm=cfg.norm,
            attention=cfg.attention,
            attention_reduction=int(cfg.attention_reduction),
            attention_spatial_kernel=int(cfg.attention_spatial_kernel),
        )
        self.body = nn.Sequential(
            *[
                ResidualBlock(
                    ch,
                    norm=cfg.norm,
                    attention=cfg.attention,
                    attention_reduction=int(cfg.attention_reduction),
                    attention_spatial_kernel=int(cfg.attention_spatial_kernel),
                )
                for _ in range(int(cfg.wm_dec_blocks))
            ]
        )
        self.head = nn.Sequential(
            nn.Conv2d(ch, int(cfg.wm_channels), kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, payload: torch.Tensor) -> torch.Tensor:
        h = self.body(self.stem(payload))
        return self.head(h)


class EmbedUNet(nn.Module):
    """
    cover + payload -> container (residual learning)
    """

    def __init__(self, cfg: NoiseWatermarkConfig) -> None:
        super().__init__()
        in_ch = int(cfg.cover_channels) + int(cfg.payload_channels)
        ch = int(cfg.base_channels)

        self.inc = DoubleConv(
            in_ch,
            ch,
            norm=cfg.norm,
            attention=cfg.attention,
            attention_reduction=int(cfg.attention_reduction),
            attention_spatial_kernel=int(cfg.attention_spatial_kernel),
        )
        downs = []
        feats: list[int] = [ch]
        for i in range(int(cfg.unet_depth)):
            in_c = ch * (2**i)
            out_c = ch * (2 ** (i + 1))
            downs.append(
                Down(
                    in_c,
                    out_c,
                    norm=cfg.norm,
                    attention=cfg.attention,
                    attention_reduction=int(cfg.attention_reduction),
                    attention_spatial_kernel=int(cfg.attention_spatial_kernel),
                )
            )
            feats.append(out_c)
        self.downs = nn.ModuleList(downs)

        bottleneck_ch = ch * (2 ** int(cfg.unet_depth))
        self.bottleneck = nn.Sequential(
            DoubleConv(
                bottleneck_ch,
                bottleneck_ch,
                norm=cfg.norm,
                attention=cfg.attention,
                attention_reduction=int(cfg.attention_reduction),
                attention_spatial_kernel=int(cfg.attention_spatial_kernel),
            ),
            ResidualBlock(
                bottleneck_ch,
                norm=cfg.norm,
                attention=cfg.attention,
                attention_reduction=int(cfg.attention_reduction),
                attention_spatial_kernel=int(cfg.attention_spatial_kernel),
            ),
            ResidualBlock(
                bottleneck_ch,
                norm=cfg.norm,
                attention=cfg.attention,
                attention_reduction=int(cfg.attention_reduction),
                attention_spatial_kernel=int(cfg.attention_spatial_kernel),
            ),
        )

        ups = []
        for i in reversed(range(int(cfg.unet_depth))):
            in_c = ch * (2 ** (i + 1))
            skip_c = ch * (2**i)
            out_c = ch * (2**i)
            ups.append(
                Up(
                    in_c,
                    skip_c,
                    out_c,
                    norm=cfg.norm,
                    attention=cfg.attention,
                    attention_reduction=int(cfg.attention_reduction),
                    attention_spatial_kernel=int(cfg.attention_spatial_kernel),
                )
            )
        self.ups = nn.ModuleList(ups)

        self.to_residual = nn.Conv2d(ch, int(cfg.cover_channels), kernel_size=1, stride=1, padding=0)
        self.strength = float(cfg.residual_strength)

    def forward(self, cover: torch.Tensor, payload: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([cover, payload], dim=1)
        feats: list[torch.Tensor] = []
        h = self.inc(x)
        feats.append(h)
        for down in self.downs:
            h = down(h)
            feats.append(h)
        h = self.bottleneck(h)
        for up, skip in zip(self.ups, reversed(feats[:-1]), strict=True):
            h = up(h, skip)

        residual = torch.tanh(self.to_residual(h)) * self.strength
        container = (cover + residual).clamp(0.0, 1.0)
        return container, residual


class RevealPayload(nn.Module):
    """
    container -> payload_hat
    """

    def __init__(self, cfg: NoiseWatermarkConfig) -> None:
        super().__init__()
        ch = int(cfg.base_channels) * 2
        norm = str(cfg.norm)
        self.stem = nn.Sequential(
            nn.Conv2d(int(cfg.cover_channels), ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch) if norm == "bn" else (nn.InstanceNorm2d(ch, affine=True) if norm == "in" else nn.Identity()),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            *[
                ResidualBlock(
                    ch,
                    norm=norm,
                    attention=cfg.attention,
                    attention_reduction=int(cfg.attention_reduction),
                    attention_spatial_kernel=int(cfg.attention_spatial_kernel),
                )
                for _ in range(int(cfg.payload_dec_blocks))
            ]
        )
        self.head = nn.Sequential(
            nn.Conv2d(ch, int(cfg.payload_channels), kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
        )

    def forward(self, container: torch.Tensor) -> torch.Tensor:
        h = self.body(self.stem(container))
        return self.head(h)


class NoiseWatermarkNet(nn.Module):
    """
    End-to-end invisible watermark:
      W -> P (noise-like)
      (H, P) -> H' (container)
      H' -> P_hat -> W_hat
    All images in [0,1], payload in [-1,1].
    """

    def __init__(self, cfg: NoiseWatermarkConfig = NoiseWatermarkConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.wm_to_noise = WatermarkToNoise(cfg)
        self.embedder = EmbedUNet(cfg)
        self.reveal_payload = RevealPayload(cfg)
        self.noise_to_wm = NoiseToWatermark(cfg)

    def forward(self, cover: torch.Tensor, watermark: torch.Tensor) -> dict[str, torch.Tensor]:
        payload = self.wm_to_noise(watermark)
        container, residual = self.embedder(cover, payload)
        payload_hat = self.reveal_payload(container)
        watermark_hat = self.noise_to_wm(payload_hat)
        return {
            "cover": cover,
            "watermark": watermark,
            "payload": payload,
            "container": container,
            "residual": residual,
            "payload_hat": payload_hat,
            "watermark_hat": watermark_hat,
        }
