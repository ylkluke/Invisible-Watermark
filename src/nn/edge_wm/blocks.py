from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, k: int = 3, s: int = 1, norm: str = "bn") -> None:
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=pad, bias=False)
        if norm == "bn":
            self.norm = nn.BatchNorm2d(out_ch)
        elif norm == "in":
            self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        elif norm == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"unsupported norm: {norm}")
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ChannelAttention(nn.Module):
    def __init__(self, ch: int, *, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, int(ch) // max(1, int(reduction)))
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, ch, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        mx = F.adaptive_max_pool2d(x, output_size=1)
        return self.act(self.mlp(avg) + self.mlp(mx))


class SpatialAttention(nn.Module):
    def __init__(self, *, kernel_size: int = 7) -> None:
        super().__init__()
        k = int(kernel_size)
        if k not in (3, 7):
            raise ValueError("CBAM spatial kernel_size must be 3 or 7")
        self.conv = nn.Conv2d(2, 1, kernel_size=k, stride=1, padding=k // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        return self.act(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, ch: int, *, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        self.channel = ChannelAttention(ch, reduction=reduction)
        self.spatial = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel(x)
        return x * self.spatial(x)


def make_attention(
    ch: int,
    *,
    attention: str = "none",
    reduction: int = 16,
    spatial_kernel: int = 7,
) -> nn.Module:
    kind = str(attention).strip().lower()
    if kind in ("", "none"):
        return nn.Identity()
    if kind == "cbam":
        return CBAM(ch, reduction=reduction, spatial_kernel=spatial_kernel)
    raise ValueError(f"unsupported attention: {attention}")


class ResidualBlock(nn.Module):
    def __init__(
        self,
        ch: int,
        *,
        norm: str = "bn",
        attention: str = "none",
        attention_reduction: int = 16,
        attention_spatial_kernel: int = 7,
    ) -> None:
        super().__init__()
        self.c1 = ConvNormAct(ch, ch, k=3, s=1, norm=norm)
        self.c2 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False)
        if norm == "bn":
            self.n2 = nn.BatchNorm2d(ch)
        elif norm == "in":
            self.n2 = nn.InstanceNorm2d(ch, affine=True)
        elif norm == "none":
            self.n2 = nn.Identity()
        else:
            raise ValueError(f"unsupported norm: {norm}")
        self.attn = make_attention(
            ch,
            attention=attention,
            reduction=attention_reduction,
            spatial_kernel=attention_spatial_kernel,
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.c1(x)
        y = self.n2(self.c2(y))
        y = self.attn(y)
        return self.act(x + y)


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        norm: str = "bn",
        attention: str = "none",
        attention_reduction: int = 16,
        attention_spatial_kernel: int = 7,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvNormAct(in_ch, out_ch, k=3, s=1, norm=norm),
            ConvNormAct(out_ch, out_ch, k=3, s=1, norm=norm),
            make_attention(
                out_ch,
                attention=attention,
                reduction=attention_reduction,
                spatial_kernel=attention_spatial_kernel,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        norm: str = "bn",
        attention: str = "none",
        attention_reduction: int = 16,
        attention_spatial_kernel: int = 7,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            DoubleConv(
                out_ch,
                out_ch,
                norm=norm,
                attention=attention,
                attention_reduction=attention_reduction,
                attention_spatial_kernel=attention_spatial_kernel,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        *,
        norm: str = "bn",
        attention: str = "none",
        attention_reduction: int = 16,
        attention_spatial_kernel: int = 7,
    ) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv = DoubleConv(
            out_ch + skip_ch,
            out_ch,
            norm=norm,
            attention=attention,
            attention_reduction=attention_reduction,
            attention_spatial_kernel=attention_spatial_kernel,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
