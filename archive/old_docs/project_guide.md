# Protein JEPA Frame — 项目文档

> 蛋白质动力学 V-JEPA 世界模型，带有 MSM state 分类辅助损失和 VAMP-2 动力学最大化损失。

---

## 1. 环境准备

### 1.1 Python 环境

当前使用的 Python 环境是项目根目录下的虚拟环境：

```bash
# 虚拟环境路径
/liuyutian/rhodes_figs/.venv/

# 激活方式
source /liuyutian/rhodes_figs/.venv/bin/activate

# 或直接用绝对路径的 python3
/liuyutian/rhodes_figs/.venv/bin/python3 -c "import torch; print(torch.__version__)"
```

**关键依赖版本**：

| 包 | 版本 |
|---|---|
| Python | 3.13.5 |
| PyTorch | 2.8.0+cu128 |
| CUDA | 12.8 (NVIDIA A800-SXM4-80GB) |
| deeptime | 0.4.5 |
| numpy | 2.5.1 |
| h5py | 3.16.0 |
| PyYAML | 6.0.3 |

### 1.2 项目安装

`protein-jepa-frame` 以 editable 模式安装在虚拟环境中：

```bash
cd /liuyutian/protein_jepa_frame
pip install -e .
```

安装后可通过 `protein-jepa-frame` 命令调用 CLI：

```bash
protein-jepa-frame train --config configs/mdcath_full_dynamics.yaml
```

---

## 2. 代码结构

```
/liuyutian/protein_jepa_frame/
├── pyproject.toml                        # 项目配置
├── configs/                              # 配置文件 (YAML)
│   ├── mdcath_full.yaml                  # 纯 JEPA baseline (无动力学损失)
│   └── mdcath_full_dynamics.yaml         # 启用 MSM + VAMP-2 动力学损失
│
├── protein_jepa_frame/                   # 主代码包
│   ├── cli.py                            # CLI 入口
│   ├── config.py                         # 配置 dataclass 定义
│   │
│   ├── data/
│   │   ├── manifest.py                   # 加载 manifest.jsonl
│   │   ├── dataset.py                    # TrajectoryClipDataset (含 MSM label 加载)
│   │   ├── collate.py                    # 批处理 collate 函数
│   │   └── masks.py                      # SpatiotemporalBlockMasker
│   │
│   ├── models/
│   │   ├── encoder.py                    # ProteinGeometryEncoder (SE(3)-invariant)
│   │   ├── predictor.py                  # SpatiotemporalPredictor
│   │   └── jepa.py                       # ProteinJEPA + 损失函数 (核心)
│   │       ├── forward()                 # V-JEPA 前向传播
│   │       ├── compute_msm_state_loss()  # per-protein MSM 分类损失
│   │       ├── compute_vamp2_loss()      # VAMP-2 动力学最大化损失
│   │       └── update_target_encoder()   # EMA 更新 target encoder
│   │
│   ├── training/
│   │   └── train.py                      # 训练循环
│   │
│   └── eval/
│       ├── evaluate.py                   # 线性探测评估
│       └── dynamics_eval.py              # VAMP-2 / TICA / 状态分离评估
│
├── artifacts/                            # 训练产物
│   ├── mdcath_full/                      # 纯 JEPA baseline (无动力学损失)
│   ├── mdcath_full_dynamics/             # 当前训练 (含动力学损失)
│   ├── mdcath_full_dynamics_bugged/      # 有 bug 的旧版本
│   └── mdcath_full_run/                  # 之前的一次完整训练 + 评估结果
│
├── tests/                                # 单元测试
│
└── docs/                                 # 文档
    └── project_guide.md                  # 本文件
```

### 2.1 核心代码说明

#### `jepa.py` — 模型 + 损失函数

- **`ProteinJEPA.forward()`**: 标准 V-JEPA 前向。context encoder 看 masked 输入，target encoder 看完整输入（no_grad），predictor 预测 masked 区域的 latent。
- **`ProteinJEPA.compute_msm_state_loss()`**: per-protein 线性分类器对 context latents 做 MSM state 分类，CrossEntropyLoss。每个蛋白有独立的 6-way 分类头（`nn.ModuleDict`），在训练开始时预注册所有 603 个蛋白的分类器。
- **`ProteinJEPA.compute_vamp2_loss()`**: 用 context encoder 对完整结构做第二次前向，per-frame mean pooling 后计算 per-clip VAMP-2 score。使用固定随机投影 128D→8D 避免小样本协方差估计问题。eigh 在 `no_grad` 内计算，梯度只通过原始数据传播。

#### `train.py` — 训练循环

训练流程：
```
batch → forward → jepa_loss + msm_state_weight * msm_loss + vamp2_weight * vamp2_loss → backward → step
```

三个 loss 加权求和后反向传播：
- `jepa_loss`: predictor 预测 vs target latent 的 smooth L1
- `msm_state_loss`: per-protein MSM state 分类 CrossEntropy
- `vamp2_loss`: 1.0 - normalized VAMP-2 score (最小化 = 最大化 VAMP-2)

---

## 3. 配置文件说明

### 3.1 完整配置 (`mdcath_full_dynamics.yaml`)

```yaml
seed: 0
device: auto                       # 自动选择 GPU
output_dir: artifacts/mdcath_full_dynamics
log_every: 20                      # 每 20 步记录一次日志
save_every: 1                      # 每 1 个 epoch 保存 checkpoint

data:
  manifest_path: /liuyutian/protein_jepa/artifacts/manifests/mdcath.deeptime_lcc_pcca_v2.full_lag1.train.filtered.jsonl
  batch_size: 4
  num_workers: 4
  num_frames: 8                    # 每个 clip 8 帧
  frame_stride: 2
  window_stride: 8
  max_residues: 384
  msm_labels_path: /liuyutian/protein_jepa/artifacts/msm_state_labels/mdcath.deeptime_lcc_pcca_v2.full_lag1.train.filtered_labels.json

mask:
  num_blocks: 2
  temporal_scale: [0.25, 0.5]
  residue_scale: [0.15, 0.4]
  min_context_ratio: 0.2

model:
  latent_dim: 128                  # 潜空间维度
  encoder_depth: 4                 # encoder transformer 层数
  predictor_depth: 4               # predictor transformer 层数
  num_heads: 4
  num_rbf: 16
  dropout: 0.1
  max_frames: 64
  max_residues: 4096
  feature_norm: true               # target latent 做 layer norm

optimization:
  epochs: 10
  lr: 0.0003
  encoder_lr_scale: 0.2            # encoder 学习率 = 0.2 * 0.0003
  weight_decay: 0.05
  warmup_steps: 1000
  ema_start: 0.996
  ema_end: 1.0
  grad_clip: 1.0
  variance_reg: 0.0                # variance 正则化 (未使用)
  msm_state_weight: 0.01           # MSM 分类损失权重
  vamp2_weight: 0.001              # VAMP-2 损失权重
  vamp2_lag: 1                     # VAMP-2 lag time
```

### 3.2 配置对比

| 配置 | 文件 | MSM 损失 | VAMP-2 损失 | 用途 |
|------|------|---------|------------|------|
| baseline | `mdcath_full.yaml` | ❌ | ❌ | 纯 JEPA 对照 |
| dynamics | `mdcath_full_dynamics.yaml` | ✅ (0.01) | ✅ (0.001) | 主实验 |

---

## 4. 启动训练

### 4.1 单次训练

```bash
cd /liuyutian/protein_jepa_frame

# 方式 1: 通过 CLI
python3 -m protein_jepa_frame.cli train \
    --config configs/mdcath_full_dynamics.yaml

# 方式 2: 通过已安装的命令
protein-jepa-frame train \
    --config configs/mdcath_full_dynamics.yaml
```

### 4.2 后台训练（推荐）

```bash
nohup bash -c 'cd /liuyutian/protein_jepa_frame && \
  python3 -c "from protein_jepa_frame.cli import main; \
  main([\"train\", \"--config\", \"configs/mdcath_full_dynamics.yaml\"])" \
  > artifacts/mdcath_full_dynamics/training.log 2>&1' &
```

或使用快捷脚本：

```bash
bash /liuyutian/run_training_fixed.sh
```

### 4.3 训练参数

| 参数 | 值 |
|------|------|
| 训练集轨迹数 | 2917 条 |
| 训练集蛋白数 | 603 个 |
| 验证集轨迹数 | 354 条 |
| Batch size | 4 |
| clip 长度 | 8 帧 |
| 每 epoch 步数 | ~2272 步 (2917 / 4 * 8 / 8 * 8 / 8) |
| 总步数 (10 epochs) | ~22720 步 |
| 训练时间 | ~5-6 小时 (10 epochs) |

### 4.4 监控训练

```bash
# 查看训练日志
tail -f /liuyutian/protein_jepa_frame/artifacts/mdcath_full_dynamics/training.log

# 查看结构化日志
tail -f /liuyutian/protein_jepa_frame/artifacts/mdcath_full_dynamics/train.jsonl

# 查看进程
ps aux | grep protein_jepa | grep -v grep
```

---

## 5. 评估

### 5.1 动力学评估 (VAMP-2 / TICA)

```bash
cd /liuyutian/protein_jepa_frame
python3 -m protein_jepa_frame.eval.run_eval \
    --checkpoint artifacts/mdcath_full_dynamics/checkpoint_latest.pt \
    --device cuda --no-msm
```

### 5.2 MSM State 线性探测评估

**重要：** 训练集和验证集是蛋白留出（protein holdout），不重叠。因此评估使用**验证集内部 70/30 拆分**，每个蛋白训练独立的分类器，再平均准确率。

```bash
cd /liuyutian/protein_jepa_frame
python3 -m protein_jepa_frame.eval.run_eval \
    --checkpoint artifacts/mdcath_full_dynamics/checkpoint_latest.pt \
    --device cuda --no-vamp2
```

### 5.3 完整评估（两者都跑）

```bash
cd /liuyutian/protein_jepa_frame
python3 -m protein_jepa_frame.eval.run_eval \
    --checkpoint artifacts/mdcath_full_dynamics/checkpoint_latest.pt \
    --device cuda
```

---

## 6. 训练结果

### 6.1 产物目录结构

```
artifacts/mdcath_full_dynamics/
├── config.json                    # 训练配置 (副本)
├── training.log                   # 训练日志 (stdout/stderr)
├── train.jsonl                    # 结构化训练日志 (JSONL, 每步一条)
│   ├── step                       # 全局步数
│   ├── epoch                      # 当前 epoch
│   ├── loss                       # 总损失
│   ├── jepa_loss                  # V-JEPA 预测损失
│   ├── variance_reg               # variance 正则化项
│   ├── msm_state_loss             # MSM 分类损失
│   ├── vamp2_loss                 # VAMP-2 损失 (归一化到 [0,1])
│   ├── lr                         # 当前学习率
│   ├── ema                        # 当前 EMA momentum
│   ├── target_tokens              # 被遮蔽的 token 数
│   └── context_tokens             # 可见的 token 数
├── checkpoint_latest.pt           # 最新 checkpoint (含 model + optimizer)
└── eval/                          # 评估结果 (需手动创建)
    ├── dynamics_summary.json      # VAMP-2 / TICA 结果
    ├── state_separation.png       # 状态分离图
    ├── confusion_matrix.png       # 混淆矩阵
    ├── pca_variance.png           # PCA 方差解释
    ├── pca_by_trajectory.png      # PCA 投影
    └── temporal_correlation.png   # 时间相关性
```

### 6.2 当前评估结果 (step 30,000, epoch 0 ~14%)

**VAMP-2 动力学评分** (50 条验证轨迹，单帧 stride=5 提取序列):

| Lag | Baseline (纯 JEPA) | 当前模型 | 提升倍数 |
|-----|-------------------|---------|:-------:|
| 1 | 1.17±0.36 | **62.9±10.4** | **53.7×** |
| 2 | 1.14±0.29 | **62.8±10.2** | **55.0×** |
| 5 | 1.12±0.26 | **62.8±9.9** | **56.3×** |
| 10 | 1.09±0.24 | **62.8±8.8** | **57.6×** |
| 20 | 1.09±0.21 | **61.3±6.3** | **56.2×** |

> 128D 潜空间理论最大 VAMP-2 = 128，当前达到 ~63。

**MSM State 分类准确率** (per-protein 70/30 拆分，30 个验证蛋白):

| 指标 | 值 |
|------|-------|
| 平均准确率 | **75.6% ± 7.5%** |
| 平均 baseline (多数类) | 39.8% |
| 平均提升 | **+35.8%** |
| 超过 baseline 的蛋白 | **30/30 (100%)** |

**状态分离**: 分离比 1.53，不同 MSM state 在 latent 空间中明显分离。

**结论**: 两个辅助损失都在正常工作。训练仅完成 ~14% 已有显著效果。

---

## 7. 损失设计说明

### 7.1 V-JEPA 损失 (主损失)

```
loss = smooth_l1(predictor(context_latents[masked]), target_latents[masked])
```

- **作用**: 学习蛋白质结构的静态几何表示
- **问题**: 不包含动力学信息，不同 MSM state 的 latent 可能坍缩到同一区域

### 7.2 MSM State 分类损失 (辅助损失)

```
msm_loss = cross_entropy(per_protein_classifier(pooled_context_latents), msm_state_labels)
```

- **权重**: 0.01
- **作用**: 强制 encoder 提取 MSM state 可区分的信息
- **注意**: 每个蛋白有独立的 6-way 线性分类头，在 optimizer 中预注册。state 标签是 per-protein 的（蛋白 A 的 state 0 ≠ 蛋白 B 的 state 0）。

### 7.3 VAMP-2 最大化损失 (辅助损失)

```
vamp2_loss = 1.0 - (vamp2_score - 1.0) / (proj_dim - 1.0)    # 归一化到 [0, 1]
```

- **权重**: 0.001
- **作用**: 最大化 latent 序列的动力学信息（VAMP-2 score）
- **实现细节**:
  - 128D latent → 固定随机投影到 8D → 每 clip 8 帧估计 8×8 协方差矩阵
  - eigh 在 `no_grad` 内计算，避免 NaN 梯度
  - 梯度只通过原始数据 X0, X1 传播，不通过白化矩阵

### 7.4 总损失

```
total_loss = jepa_loss + 0.01 * msm_loss + 0.001 * vamp2_loss
```

---

## 8. 常见问题

### Q: 训练时 GPU 显存不足？
- 减少 `batch_size` 或 `max_residues`
- 确认使用 GPU 0（当前 GPU 0 使用 26GB/80GB，GPU 1 空闲）

### Q: VAMP-2 loss 恒定不变？
- 检查 `_vamp2_projector` 是否正确初始化
- 检查是否有 `len(seq) < D // 4` 之类的 guard 过滤了所有 clip
- 确认投影后的维度 `d` 小于 T（T=8, d 应 ≤ 8）

### Q: MSM state loss 为 0？
- 确认 `msm_labels_path` 配置正确
- 确认 `dataset.records` 不为空且包含 `protein_id`
- 确认分类器在 optimizer 的 param_group 中

### Q: 如何添加新的配置文件？
参考 `configs/mdcath_full_dynamics.yaml` 的格式，复制并修改相应参数。

---

## 9. 数据依赖

所有训练数据存储在 `/liuyutian/protein_jepa/`：

| 数据 | 路径 |
|------|------|
| 训练集 manifest | `/liuyutian/protein_jepa/artifacts/manifests/mdcath.deeptime_lcc_pcca_v2.full_lag1.train.filtered.jsonl` |
| 验证集 manifest | `/liuyutian/protein_jepa/artifacts/manifests/mdcath.deeptime_lcc_pcca_v2.full_lag1.validation.filtered.jsonl` |
| MSM 标签 | `/liuyutian/protein_jepa/artifacts/msm_state_labels/mdcath.deeptime_lcc_pcca_v2.full_lag1.train.filtered_labels.json` |
| 原始轨迹 | `/liuyutian/protein_jepa/datasets/mdCATH/repo/data/mdcath_dataset_*.h5` |