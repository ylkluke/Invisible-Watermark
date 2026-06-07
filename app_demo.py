from __future__ import annotations

import io
import random
import sys
from pathlib import Path
from typing import Any

try:
    import streamlit as st
except Exception as e:  # pragma: no cover - shown only when launched without deps
    raise SystemExit(
        "Streamlit is required for the demo. Run with:\n"
        "  bash scripts/with_conda_gp.sh streamlit run app_demo.py\n"
        f"Import error: {e}"
    )

try:
    import torch
    from PIL import Image
except Exception as e:  # pragma: no cover - shown only when launched without deps
    st.error("Torch and Pillow are required. Please run in the project conda environment.")
    st.code("bash scripts/with_conda_gp.sh streamlit run app_demo.py", language="bash")
    st.stop()


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.nn.edge_wm.attacks import (  # noqa: E402
    AttackConfig,
    gaussian_blur,
    jpeg_approx,
    jpeg_pil,
    mosaic,
    random_crop_keep,
    random_resize_back,
    apply_random_attacks,
)
from src.nn.edge_wm.losses import psnr, ssim  # noqa: E402
from src.nn.edge_wm.noise_wm_model import NoiseWatermarkConfig, NoiseWatermarkNet  # noqa: E402


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PREFERRED_CKPT = ROOT / "outputs" / "edge_wm_noise" / "y_only_fresh_20260402" / "ckpt_last.pt"
DEFAULT_WATERMARK = ROOT / "assets" / "watermarks" / "luke.png"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _scan_checkpoints() -> list[Path]:
    base = ROOT / "outputs" / "edge_wm_noise"
    if not base.exists():
        return []
    items = sorted(base.glob("*/ckpt_last.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if PREFERRED_CKPT.exists():
        items = [PREFERRED_CKPT] + [p for p in items if p != PREFERRED_CKPT]
    return items


def _scan_watermarks() -> list[Path]:
    base = ROOT / "assets" / "watermarks"
    if not base.exists():
        return []
    items = sorted([p for p in base.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if DEFAULT_WATERMARK.exists():
        items = [DEFAULT_WATERMARK] + [p for p in items if p != DEFAULT_WATERMARK]
    return items


def _resolve_latest_ckpt() -> Path:
    ckpts = _scan_checkpoints()
    if not ckpts:
        raise FileNotFoundError("No checkpoints found under outputs/edge_wm_noise/*/ckpt_last.pt")
    return ckpts[0]


def _pil_resize_square(img: Image.Image, size: int) -> Image.Image:
    resampling = getattr(getattr(Image, "Resampling", None), "LANCZOS", Image.LANCZOS)
    return img.resize((size, size), resample=resampling)


def _pil_to_tensor_rgb(img: Image.Image, size: int) -> torch.Tensor:
    img = _pil_resize_square(img.convert("RGB"), size)
    b = bytearray(img.tobytes())
    w, h = img.size
    x = torch.frombuffer(b, dtype=torch.uint8).to(torch.float32)
    return x.view(h, w, 3).permute(2, 0, 1).contiguous() / 255.0


def _pil_to_tensor_ycbcr(img: Image.Image, size: int) -> torch.Tensor:
    img = _pil_resize_square(img.convert("YCbCr"), size)
    b = bytearray(img.tobytes())
    w, h = img.size
    x = torch.frombuffer(b, dtype=torch.uint8).to(torch.float32)
    return x.view(h, w, 3).permute(2, 0, 1).contiguous() / 255.0


def _tensor_to_pil_rgb(x: torch.Tensor) -> Image.Image:
    x = x.detach().clamp(0.0, 1.0).mul(255.0).to(torch.uint8)
    arr = x.permute(1, 2, 0).contiguous().cpu()
    h, w, c = arr.shape
    if c != 3:
        raise ValueError("expected CHW RGB tensor")
    return Image.frombytes("RGB", (w, h), bytes(arr.reshape(-1).tolist()))


def _tensor_y_to_pil_l(x: torch.Tensor) -> Image.Image:
    x = x.detach().clamp(0.0, 1.0).mul(255.0).to(torch.uint8).contiguous().cpu()
    h, w = int(x.shape[-2]), int(x.shape[-1])
    return Image.frombytes("L", (w, h), bytes(x.reshape(-1).tolist()))


def _compose_y_with_cbcr(y01: torch.Tensor, cbcr01: torch.Tensor) -> Image.Image:
    if y01.ndim != 3 or y01.shape[0] != 1:
        raise ValueError("expected y01 with shape (1,H,W)")
    if cbcr01.ndim != 3 or cbcr01.shape[0] != 2:
        raise ValueError("expected cbcr01 with shape (2,H,W)")
    ycbcr = torch.cat([y01.detach().cpu(), cbcr01.detach().cpu()], dim=0)
    x = ycbcr.clamp(0.0, 1.0).mul(255.0).to(torch.uint8)
    arr = x.permute(1, 2, 0).contiguous()
    h, w, _ = arr.shape
    return Image.frombytes("YCbCr", (w, h), bytes(arr.reshape(-1).tolist())).convert("RGB")


def _pil_bytes(img: Image.Image, *, fmt: str = "PNG") -> bytes:
    bio = io.BytesIO()
    img.save(bio, format=fmt)
    return bio.getvalue()


@st.cache_resource(show_spinner=False)
def _load_model_cached(ckpt_str: str, device_str: str) -> tuple[NoiseWatermarkNet, dict[str, Any]]:
    ckpt_path = _resolve_latest_ckpt() if ckpt_str == "latest" else Path(ckpt_str)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    device = torch.device(device_str)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise ValueError("invalid checkpoint: missing model_state")

    cfg_d = ckpt.get("model_config") if isinstance(ckpt.get("model_config"), dict) else {}
    cfg = NoiseWatermarkConfig(**{k: cfg_d[k] for k in cfg_d if k in NoiseWatermarkConfig().__dict__})  # type: ignore[arg-type]
    model = NoiseWatermarkNet(cfg)
    model.load_state_dict(ckpt["model_state"], strict=True)  # type: ignore[arg-type]
    model.to(device).eval()
    meta = {
        "path": str(ckpt_path),
        "step": int(ckpt.get("step", 0) or 0),
        "config": cfg_d,
    }
    return model, meta


def _apply_demo_attack(
    container: torch.Tensor,
    *,
    mode: str,
    seed: int,
    crop_keep: float,
    mosaic_block: int,
    jpeg_quality: int,
    jpeg_mode: str,
    resize_scale: float,
    blur_sigma: float,
) -> torch.Tensor | None:
    if mode == "无攻击":
        return None

    rng = random.Random(int(seed))
    if mode == "裁剪":
        return random_crop_keep(container, keep=float(crop_keep), rng=rng).clamp(0.0, 1.0)
    if mode == "马赛克":
        return mosaic(container, block=int(mosaic_block)).clamp(0.0, 1.0)
    if mode == "JPEG 压缩":
        if jpeg_mode == "pil":
            return jpeg_pil(container, quality=int(jpeg_quality)).clamp(0.0, 1.0)
        return jpeg_approx(container, quality=int(jpeg_quality)).clamp(0.0, 1.0)
    if mode == "缩放":
        return random_resize_back(container, min_scale=float(resize_scale), max_scale=float(resize_scale), rng=rng).clamp(0.0, 1.0)
    if mode == "模糊":
        return gaussian_blur(container, k=5, sigma=float(blur_sigma)).clamp(0.0, 1.0)
    if mode == "组合攻击":
        cfg = AttackConfig(
            prob=1.0,
            crop_keep=float(crop_keep),
            crop_prob=1.0,
            mosaic_prob=1.0,
            mosaic_min_block=int(mosaic_block),
            mosaic_max_block=int(mosaic_block),
            jpeg_prob=1.0,
            jpeg_quality_min=int(jpeg_quality),
            jpeg_quality_max=int(jpeg_quality),
            jpeg_mode=str(jpeg_mode),
            resize_prob=1.0,
            resize_min_scale=float(resize_scale),
            resize_max_scale=float(resize_scale),
            blur_prob=1.0,
            blur_kernel=5,
            blur_sigma_max=float(blur_sigma),
        )
        return apply_random_attacks(container, cfg=cfg, rng=rng, allow_pil_jpeg=(jpeg_mode == "pil")).clamp(0.0, 1.0)

    raise ValueError(f"unknown attack mode: {mode}")


def _run_pipeline(
    *,
    cover_img: Image.Image,
    watermark_img: Image.Image,
    ckpt: str,
    device_str: str,
    size: int,
    attack_mode: str,
    attack_seed: int,
    crop_keep: float,
    mosaic_block: int,
    jpeg_quality: int,
    jpeg_mode: str,
    resize_scale: float,
    blur_sigma: float,
) -> dict[str, Any]:
    model, meta = _load_model_cached(ckpt, device_str)
    device = torch.device(device_str)

    cover = _pil_to_tensor_rgb(cover_img, size).unsqueeze(0).to(device)
    wm_ycbcr = _pil_to_tensor_ycbcr(watermark_img, size).to(device)
    wm = wm_ycbcr[:1].unsqueeze(0)
    wm_cbcr = wm_ycbcr[1:].detach().cpu()

    with torch.no_grad():
        payload = model.wm_to_noise(wm)
        container, _resid = model.embedder(cover, payload)
        payload_hat = model.reveal_payload(container)
        wm_hat = model.noise_to_wm(payload_hat)

        attacked = _apply_demo_attack(
            container,
            mode=attack_mode,
            seed=attack_seed,
            crop_keep=crop_keep,
            mosaic_block=mosaic_block,
            jpeg_quality=jpeg_quality,
            jpeg_mode=jpeg_mode,
            resize_scale=resize_scale,
            blur_sigma=blur_sigma,
        )
        wm_hat_attacked = None
        payload_hat_attacked = None
        if attacked is not None:
            payload_hat_attacked = model.reveal_payload(attacked)
            wm_hat_attacked = model.noise_to_wm(payload_hat_attacked)

        residual_abs = (container[0] - cover[0]).abs()
        residual_norm = residual_abs / (residual_abs.max() + 1e-12)
        residual_amp = (residual_abs * 20.0).clamp(0.0, 1.0)

        metrics = {
            "Host PSNR": float(psnr(cover, container).item()),
            "Host SSIM": float(ssim(cover, container).item()),
            "Watermark PSNR": float(psnr(wm, wm_hat).item()),
            "Watermark SSIM": float(ssim(wm, wm_hat).item()),
            "Attack Watermark PSNR": float(psnr(wm, wm_hat_attacked).item()) if wm_hat_attacked is not None else None,
            "Attack Watermark SSIM": float(ssim(wm, wm_hat_attacked).item()) if wm_hat_attacked is not None else None,
        }

    return {
        "meta": meta,
        "metrics": metrics,
        "cover": _tensor_to_pil_rgb(cover[0]),
        "container": _tensor_to_pil_rgb(container[0]),
        "residual_norm": _tensor_to_pil_rgb(residual_norm),
        "residual_amp": _tensor_to_pil_rgb(residual_amp),
        "attacked": _tensor_to_pil_rgb(attacked[0]) if attacked is not None else None,
        "wm": _compose_y_with_cbcr(wm[0], wm_cbcr),
        "wm_y": _tensor_y_to_pil_l(wm[0]),
        "wm_hat": _compose_y_with_cbcr(wm_hat[0], wm_cbcr),
        "wm_hat_y": _tensor_y_to_pil_l(wm_hat[0]),
        "wm_hat_attacked": _compose_y_with_cbcr(wm_hat_attacked[0], wm_cbcr) if wm_hat_attacked is not None else None,
        "wm_hat_attacked_y": _tensor_y_to_pil_l(wm_hat_attacked[0]) if wm_hat_attacked is not None else None,
    }


def _metric_text(value: float | None, *, digits: int) -> str:
    if value is None:
        return "N/A"
    if value == float("inf"):
        return "inf"
    return f"{value:.{digits}f}"


def _render_download(label: str, img: Image.Image, filename: str) -> None:
    st.download_button(label, data=_pil_bytes(img), file_name=filename, mime="image/png", use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="不可见水印答辩演示", layout="wide")
    st.title("不可见水印嵌入、攻击与恢复演示")
    st.caption("基于现有 NoiseWatermarkNet checkpoint，仅做现场推理展示，不重新训练。")

    ckpts = _scan_checkpoints()
    watermarks = _scan_watermarks()

    with st.sidebar:
        st.header("输入")
        uploaded = st.file_uploader("上传载体图片", type=sorted({e.lstrip('.') for e in IMAGE_EXTS}))

        st.header("模型")
        ckpt_labels = [_rel(p) for p in ckpts]
        ckpt_choice = st.selectbox("Checkpoint", ckpt_labels, index=0 if ckpt_labels else None, placeholder="未找到 checkpoint")
        custom_ckpt = st.text_input("自定义 checkpoint 路径", value="")
        ckpt_value = custom_ckpt.strip() or ckpt_choice or ""

        wm_labels = [_rel(p) for p in watermarks]
        wm_choice = st.selectbox("水印模板", wm_labels, index=0 if wm_labels else None, placeholder="未找到水印模板")
        custom_wm = st.text_input("自定义水印路径", value="")
        wm_value = custom_wm.strip() or wm_choice or ""

        size = st.select_slider("推理尺寸", options=[128, 192, 256, 320, 384, 512], value=256)
        cuda_ok = bool(torch.cuda.is_available())
        device_options = ["cuda", "cpu"] if cuda_ok else ["cpu"]
        device = st.radio("设备", device_options, horizontal=True)

        st.header("攻击")
        attack_mode = st.selectbox("攻击模式", ["无攻击", "裁剪", "JPEG 压缩", "缩放", "模糊", "马赛克", "组合攻击"])
        attack_seed = st.number_input("随机种子", min_value=0, max_value=999999, value=12345, step=1)
        crop_keep = st.slider("裁剪保留比例", 0.30, 1.00, 0.70, 0.05)
        jpeg_quality = st.slider("JPEG 质量", 1, 100, 50, 1)
        jpeg_mode = st.radio("JPEG 模式", ["pil", "approx"], horizontal=True)
        resize_scale = st.slider("缩放比例", 0.30, 1.50, 0.80, 0.05)
        blur_sigma = st.slider("模糊 sigma", 0.10, 3.00, 1.00, 0.10)
        mosaic_block = st.slider("马赛克块大小", 2, 32, 8, 1)

        run = st.button("开始演示", type="primary", use_container_width=True)

    if uploaded is None:
        st.info("请先在左侧上传一张载体图片。")
        return
    if not ckpt_value:
        st.error("未找到 checkpoint。请填写 `outputs/edge_wm_noise/.../ckpt_last.pt`。")
        return
    if not wm_value:
        st.error("未找到水印模板。请填写 `assets/watermarks/luke.png` 或其他模板路径。")
        return

    cover_img = Image.open(uploaded).convert("RGB")
    wm_path = Path(wm_value)
    if not wm_path.is_absolute():
        wm_path = ROOT / wm_path
    if not wm_path.exists():
        st.error(f"水印模板不存在：{wm_path}")
        return
    watermark_img = Image.open(wm_path).convert("RGB")

    with st.expander("当前演示流程", expanded=False):
        st.write("上传图片 -> 水印 Y 通道编码为噪声载荷 -> 嵌入 RGB 载体图 -> 可选攻击 -> 恢复水印 Y 通道 -> 合成彩色可视化")
        st.write("这里的隐藏/恢复是模型水印流程，不是传统密码学文件加密。")

    if not run:
        cols = st.columns(2)
        cols[0].image(cover_img, caption="待演示载体图", use_container_width=True)
        cols[1].image(watermark_img, caption="水印模板", use_container_width=True)
        return

    try:
        with st.spinner("正在运行现有模型推理..."):
            result = _run_pipeline(
                cover_img=cover_img,
                watermark_img=watermark_img,
                ckpt=ckpt_value,
                device_str=device,
                size=int(size),
                attack_mode=str(attack_mode),
                attack_seed=int(attack_seed),
                crop_keep=float(crop_keep),
                mosaic_block=int(mosaic_block),
                jpeg_quality=int(jpeg_quality),
                jpeg_mode=str(jpeg_mode),
                resize_scale=float(resize_scale),
                blur_sigma=float(blur_sigma),
            )
    except Exception as e:
        st.exception(e)
        return

    meta = result["meta"]
    st.success(f"推理完成：{_rel(Path(meta['path']))}，step={meta.get('step', 0)}")

    metrics = result["metrics"]
    metric_cols = st.columns(6)
    metric_cols[0].metric("Host PSNR", _metric_text(metrics["Host PSNR"], digits=2))
    metric_cols[1].metric("Host SSIM", _metric_text(metrics["Host SSIM"], digits=4))
    metric_cols[2].metric("WM PSNR", _metric_text(metrics["Watermark PSNR"], digits=2))
    metric_cols[3].metric("WM SSIM", _metric_text(metrics["Watermark SSIM"], digits=4))
    metric_cols[4].metric("Attack WM PSNR", _metric_text(metrics["Attack Watermark PSNR"], digits=2))
    metric_cols[5].metric("Attack WM SSIM", _metric_text(metrics["Attack Watermark SSIM"], digits=4))

    st.subheader("载体图像链路")
    c1, c2, c3, c4 = st.columns(4)
    c1.image(result["cover"], caption="原始图片", use_container_width=True)
    c2.image(result["container"], caption="含水印图 / 嵌入输出", use_container_width=True)
    c3.image(result["residual_amp"], caption="残差放大图 x20", use_container_width=True)
    c4.image(result["attacked"] or result["container"], caption="攻击后图片" if result["attacked"] else "未启用攻击", use_container_width=True)

    st.subheader("水印恢复")
    w1, w2, w3, w4 = st.columns(4)
    w1.image(result["wm"], caption="原始水印", use_container_width=True)
    w2.image(result["wm_hat"], caption="直接恢复水印", use_container_width=True)
    w3.image(result["wm_hat_attacked"] or result["wm_hat"], caption="攻击后恢复水印" if result["wm_hat_attacked"] else "未启用攻击", use_container_width=True)
    w4.image(result["residual_norm"], caption="残差归一化图", use_container_width=True)

    with st.expander("Y 通道灰度结果"):
        gy1, gy2, gy3 = st.columns(3)
        gy1.image(result["wm_y"], caption="原始水印 Y", use_container_width=True)
        gy2.image(result["wm_hat_y"], caption="直接恢复 Y", use_container_width=True)
        if result["wm_hat_attacked_y"] is not None:
            gy3.image(result["wm_hat_attacked_y"], caption="攻击后恢复 Y", use_container_width=True)
        else:
            gy3.info("未启用攻击。")

    with st.expander("下载结果图"):
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            _render_download("下载含水印图", result["container"], "container.png")
        with d2:
            _render_download("下载残差放大图", result["residual_amp"], "residual_x20.png")
        with d3:
            _render_download("下载恢复水印", result["wm_hat"], "watermark_recovered.png")
        with d4:
            if result["attacked"] is not None:
                _render_download("下载攻击后图", result["attacked"], "container_attacked.png")
            else:
                st.button("未启用攻击", disabled=True, use_container_width=True)


if __name__ == "__main__":
    main()
