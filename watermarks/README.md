# Watermark Templates

把水印模板图放在本目录，推荐 `PNG`。

默认模板：
- `assets/watermarks/luke.png`

## 当前项目如何使用模板

- 训练、推理和测试时，模板会先转换为 YCbCr。
- 模型只编码和恢复模板的亮度通道 Y。
- 恢复输出 `wm_hat` 是单通道 Y；保存彩色可视化图时，会把预测的 `Y_hat` 与模板原始 Cb/Cr 合成为 RGB 图。
- 这里的“隐藏/恢复”指模型水印流程，不是传统密码学文件加密。

## 常用命令

训练（鲁棒模式）：

```bash
bash scripts/with_conda_gp.sh python experiments/edge_wm_noise_train.py \
  --input-dir inputs/div2k/DIV2K_train_HR \
  --wm-template assets/watermarks/luke.png \
  --robust
```

单图推理：

```bash
bash scripts/with_conda_gp.sh python experiments/edge_wm_noise_infer.py \
  --input inputs/chaos_batch/xxx.jpg \
  --wm-template assets/watermarks/luke.png \
  --ckpt outputs/edge_wm_noise/<run>/ckpt_last.pt
```

批量测试：

```bash
bash scripts/with_conda_gp.sh python experiments/edge_wm_noise_test.py \
  --input-dir inputs/chaos_batch \
  --wm-template assets/watermarks/luke.png \
  --ckpt outputs/edge_wm_noise/<run>/ckpt_last.pt
```
