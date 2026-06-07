# Streamlit 答辩演示

本演示只复用现有模型和 checkpoint 做推理，不重新训练。

## 启动

```bash
bash scripts/with_conda_gp.sh streamlit run app_demo.py
```

启动后浏览器会打开本地页面。如果没有自动打开，按终端中显示的 Local URL 访问。

## 演示流程

1. 左侧上传一张载体图片。
2. 选择 checkpoint，默认优先使用 `outputs/edge_wm_noise/y_only_fresh_20260402/ckpt_last.pt`。
3. 选择水印模板，默认使用 `assets/watermarks/luke.png`。
4. 选择攻击模式：
   - 无攻击
   - 裁剪
   - JPEG 压缩
   - 缩放
   - 模糊
   - 马赛克
   - 组合攻击
5. 调整攻击参数。
6. 点击“开始演示”。

## 页面展示

页面会展示：

- 原始图片
- 含水印图 / 嵌入输出
- 残差放大图
- 攻击后图片
- 原始水印
- 直接恢复水印
- 攻击后恢复水印
- Host PSNR、Host SSIM、Watermark PSNR、Watermark SSIM

## 术语说明

页面中的“隐藏/恢复”指当前模型的不可见水印流程：

```text
水印 Y 通道 -> 噪声载荷 payload -> 嵌入 RGB 载体图像 -> 恢复水印 Y 通道
```

这不是传统密码学文件加密。恢复水印的彩色图是将模型恢复出的 Y 通道与原水印模板的 Cb/Cr 色度通道合成得到的可视化结果。
