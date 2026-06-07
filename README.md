# GraduationProject

本项目实现的是一个基于 PyTorch 的端到端数字图像不可见水印系统，包含训练、单图推理、批量测试、鲁棒性对比、消融实验和 Streamlit 演示程序。

当前代码以 RGB 载体图像为输入，将水印模板转换到 YCbCr 后只编码其 Y 通道；恢复阶段输出水印 Y 通道，并在可视化时复用模板的 Cb/Cr 色度通道。

## 本次提交包含的内容

- `src/`：模型、损失函数、攻击模块
- `experiments/`：训练、推理、测试脚本
- `scripts/`：环境启动、对比实验、消融实验、答辩材料生成脚本
- `app_demo.py`：Streamlit 演示程序
- `assets/watermarks/luke.png`：默认水印模板
- `inputs/chaos_batch/`：演示和测试样例图像
- `outputs/edge_wm_noise/y_only_fresh_20260402/ckpt_last.pt`：已训练好的演示模型
- `outputs/experiment_reports/`、`outputs/thesis_figures/`：论文和实验结果摘要
- `requirements.txt`、`environment.yml`：复现环境文件

## 未随默认代码包附带的大文件

- `inputs/div2k/DIV2K_train_HR/`：训练集，约 3.4 GB
- 其余完整 `outputs/` 实验输出：约 21 GB

为了便于提交老师，本次打包默认只保留源码、脚本、样例输入、一个可直接演示的 checkpoint，以及关键结果摘要。

如果用于本科论文抽检，建议额外提供包含训练集的“抽检版”压缩包，确保检查人员可以直接核对训练数据目录。

## 环境要求

已在如下环境验证通过：

- Python `3.11.14`
- torch `2.10.0`
- Pillow `12.0.0`
- numpy `2.3.5`
- streamlit `1.57.0`
- matplotlib `3.10.9`
- python-pptx `1.0.2`

建议在 Linux 或 WSL 环境中使用 Conda。

### 方式 1：Conda

```bash
conda env create -f environment.yml
conda activate gp
```

### 方式 2：pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

说明：

- `requirements.txt` 中默认写的是 `torch==2.10.0`。
- 如果使用 NVIDIA GPU，可按本机 CUDA 版本改装对应的 PyTorch 轮子。
- 如果不使用 Conda，`scripts/with_conda_gp.sh` 不再适用，请直接执行 `python ...` 命令。

## 快速复现：直接使用已训练模型

### 1. 单图推理

```bash
bash scripts/with_conda_gp.sh python experiments/edge_wm_noise_infer.py \
  --input inputs/chaos_batch/pexels-connorscottmcmanus-35528106.jpg \
  --wm-template assets/watermarks/luke.png \
  --ckpt outputs/edge_wm_noise/y_only_fresh_20260402/ckpt_last.pt \
  --outdir outputs/edge_wm_noise_infer/demo_run \
  --device cpu
```

输出目录中会保存：

- 含水印图
- 残差可视化
- 恢复水印图
- 运行清单 `run.json`

### 2. 批量测试

```bash
bash scripts/with_conda_gp.sh python experiments/edge_wm_noise_test.py \
  --input-dir inputs/chaos_batch \
  --wm-template assets/watermarks/luke.png \
  --ckpt outputs/edge_wm_noise/y_only_fresh_20260402/ckpt_last.pt \
  --outdir outputs/edge_wm_noise_test/demo_batch \
  --device cpu \
  --save-samples 8 \
  --attack --attack-mode all
```

测试结果会输出到：

- `metrics.jsonl`：逐图指标
- `summary.json`：汇总指标
- `samples/`：样例结果图

### 3. 启动答辩演示页面

```bash
bash scripts/with_conda_gp.sh streamlit run app_demo.py
```

页面默认优先读取：

- `outputs/edge_wm_noise/y_only_fresh_20260402/ckpt_last.pt`
- `assets/watermarks/luke.png`

更具体的页面操作说明见 [DEMO.md](/home/luke/Workspace/GraduationProject/DEMO.md)。

## 完整训练复现

### 1. 准备训练数据

将 DIV2K 训练集高分辨率图像放到：

```text
inputs/div2k/DIV2K_train_HR/
```

训练脚本也支持直接给 zip 文件；如果路径对应的 `.zip` 存在，脚本会自动解压。

### 2. 运行鲁棒训练

```bash
bash scripts/with_conda_gp.sh python experiments/edge_wm_noise_train.py \
  --input-dir inputs/div2k/DIV2K_train_HR \
  --wm-template assets/watermarks/luke.png \
  --outdir outputs/edge_wm_noise/reproduce_run \
  --steps 2000 \
  --batch-size 4 \
  --num-workers 2 \
  --device cpu \
  --robust
```

训练输出默认包含：

- `ckpt_last.pt` 和 `ckpt_step_*.pt`
- `metrics.jsonl`
- `train_summary.json`
- 训练过程样例图

### 3. 用新模型做测试

```bash
bash scripts/with_conda_gp.sh python experiments/edge_wm_noise_test.py \
  --input-dir inputs/chaos_batch \
  --wm-template assets/watermarks/luke.png \
  --ckpt outputs/edge_wm_noise/reproduce_run/ckpt_last.pt \
  --outdir outputs/edge_wm_noise_test/reproduce_run \
  --device cpu \
  --attack --attack-mode all
```

## 对比实验与消融实验

### Plain vs Robust 对比

```bash
bash scripts/run_compare_robust_vs_plain.sh compare_r1 800 cpu 2 0 \
  inputs/div2k/DIV2K_train_HR inputs/chaos_batch assets/watermarks/luke.png
```

输出摘要目录：

- `outputs/edge_wm_compare/<tag>/compare_summary.json`
- `outputs/edge_wm_compare/<tag>/compare_summary.md`

### A0-A5 消融实验

```bash
bash scripts/run_ablation_suite.sh ablation_r1 800 cpu 2 0 \
  inputs/div2k/DIV2K_train_HR inputs/chaos_batch assets/watermarks/luke.png
```

输出摘要目录：

- `outputs/edge_wm_ablation/<tag>/ablation_summary.json`
- `outputs/edge_wm_ablation/<tag>/ablation_summary.md`

## 关键目录说明

- `config.toml`：默认配置，命令行参数优先级更高
- `src/nn/edge_wm/`：核心模型与攻击实现
- `experiments/edge_wm_noise_train.py`：训练入口
- `experiments/edge_wm_noise_infer.py`：单图/批量推理入口
- `experiments/edge_wm_noise_test.py`：批量测试入口
- `outputs/edge_wm_noise/y_only_fresh_20260402/`：当前随包提供的演示 checkpoint

## 打包脚本

已提供提交脚本：

```bash
bash scripts/package_submission.sh
```

该脚本会在 `deliverables/` 下生成老师可直接提交的压缩包，并附带文件清单和校验文件。

如果用于论文抽检，生成包含训练集的版本：

```bash
bash scripts/package_submission.sh 20260603 with_dataset
```

该版本会额外打入：

- `inputs/div2k/DIV2K_train_HR/`：800 张训练图像，当前目录大小约 3.3 GB

建议提交方式：

- 普通老师查阅：提交默认精简包
- 论文抽检：提交 `with_dataset` 版本，或把训练集目录单独打包附上
