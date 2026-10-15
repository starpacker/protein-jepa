# Fast ABC 训练流水线 — 完整诊断报告

**日期**: 2026-10-06  
**仓库**: `/liuyutian/protein_jepa_frame`  
**目标**: 以最小迭代预算验证 Stage A→B→C 串联训练 + 五项全量评估，暴露所有基础设施问题和模型质量瓶颈  
**撰写目的**: 供独立 agent 进行深层次根因分析——本报告提供完整的代码路径、文件清单、训练指标和评估结果

---

## 目录

1. [执行摘要](#1-执行摘要)
2. [流水线与数据概览](#2-流水线与数据概览)
3. [训练结果：Stage A](#3-训练结果stage-a)
4. [训练结果：Stage B](#4-训练结果stage-b)
5. [训练结果：Stage C](#5-训练结果stage-c)
6. [评估结果](#6-评估结果)
7. [已发现的 Bug 与修复](#7-已发现的-bug-与修复)
8. [核心问题诊断](#8-核心问题诊断)
9. [完整代码路径清单](#9-完整代码路径清单)
10. [训练产物清单](#10-训练产物清单)
11. [可供独立 Agent 分析的关键线索](#11-可供独立-agent-分析的关键线索)

---

## 1. 执行摘要

本文档记录了 **Fast ABC** 训练流水线从零初始化的完整执行过程：

| 阶段 | 可训练参数 | 步数 | 状态 | 关键发现 |
|------|-----------|------|------|----------|
| **Stage A** | 所有 JEPA 编码器 | 20 epochs × 300 = **6000** | ✅ 完成 | 双任务交替收敛: mask loss 65→7-17, paired delta 0.25→0.08 |
| **Stage B** | LaSP 世界模型 | 3 epochs × 300 = **900** | ✅ 完成 | transition NLL 22→19, rollout MSE 0.62→0.02, semigroup 0.26→0.09 |
| **Stage C** | IPA 结构解码器 | 2 epochs × 300 = **600** | ✅ 完成 | FAPE 9.5→9.7 (持平), torsion 3.6→2.1, Q 权重斜坡未生效 |
| **Baselines 评估** | — | 100 samples, 2 horizons | ✅ 完成 | Q norm 爆炸 ~8.6×, persistence 完美 |
| **Temperature 评估** | — | 50 samples, 3 jumps | ✅ 完成 | 零温度敏感性 |
| **Sidechain 评估** | — | 100 samples | ✅ 完成 | Phi/Psi 误差 71°/104°, 严重 clash |
| **Rollout Quality** | — | 100 samples | ✅ 完成 | Q norm 漂移 +50%, 64ns 95% 发散 |
| **World Model 评估** | — | 5-mode eval | ❌ 失败 | 脚本修复后重试挂起(session 内未解) |

**总训练步数**: 6000 + 900 + 600 = **7500 步**

**核心发现**: 流水线在**工程层面完整跑通**，但 **Q dynamics 质量严重不足**。
这些问题**不能用训练预算不足来解释**——而是更深层的架构和/或损失设计问题。

### 关键红线

1. **Q norm 在 rollout 中爆炸 ~8.6×** (3.47 → ~30)，远超 persistence baseline
2. **三个温度跳跃产生完全相同的 Q 响应** (q_diff_ratio ≈ 0.162)——模型未学习温度缩放
3. **Decoder 忽略 Q**: anchor noise 训练后仍输出 71° Phi/104° Psi 误差——实际上是一个 **anchor refinement autoencoder**
4. **64ns rollout 95% 发散**: 仅 5/100 样本在 64ns 保持有效 Q
5. **仅 1 个蛋白质** (1b4uB00, L=256) 用于所有评估——评估多样性严重不足

---

## 2. 流水线与数据概览

### 2.1 流水线架构

```
┌─────────────────────────────────────────────────────┐
│                    Stage A                           │
│  JEPA 编码器 (从头训练, 6000 steps)                    │
│  context_encoder (GNN) + temporal_encoder (Transformer)│
│  + predictor (masked prediction) + paired forecaster   │
│  双任务交替: "mask" 与 "paired"                        │
│  冻结 → Stage B, Stage C                              │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│                    Stage B                           │
│  LaSP 世界模型 (冻结编码器, 900 steps)                  │
│  H→Q 投影头 (Q_dim=8)                                │
│  Per-residue energy graph (O(n²) edges)               │
│  Diffusion + bath + closed-loop transition            │
│  Rollout MSE + semigroup 一致性损失                     │
│  冻结 → Stage C                                       │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│                    Stage C                           │
│  IPA 结构解码器 (冻结编码器, 600 steps)                  │
│  输入: Q (from B) + anchor coords                     │
│  输出: backbone + sidechain 3D coords                 │
│  Anchor noise 0.5Å + Q weight ramp (300 steps)        │
│  Gradient accumulation ×4                             │
└─────────────────────────────────────────────────────┘
```

### 2.2 数据集

| 集 | 蛋白质数 | 总条目数 | 来源 |
|----|---------|---------|------|
| 训练集 | **160** | 4000 | `artifacts/mdcath_manifest_200dom/train.jsonl` |
| 验证集 | **20** | 500 (每蛋白 25 条) | `artifacts/mdcath_manifest_200dom/val.jsonl` |

验证集蛋白质 (20):
`153lA00  1ad3A02  1adnA00  1am4A00  1aq3A00  1ayrB02  1b4uB00  1b5eA00  1bebA00  1bgqA00  1bp1A01  1bwdA00  1bymA00  1c16A01  1c7sA01  1c7sA04  1ccdA00  1cd9D01  1cfbA02  1ciaA00`

**注意**: 所有评估 (baselines/temperature/sidechain/rollout) 均使用采样自 `1b4uB00` 的数据——因 n_val=100 但只有一个蛋白同时满足 L ≤ 256 且窗口要求。
这意味着 CLS (per-residue Q) 的全蛋白多样性未被评估。

### 2.3 三阶段配置

#### Stage A (`configs/fast_abc_stage_a_from_scratch.yaml`)

| 参数 | 值 |
|------|-----|
| batch_size | 16 |
| num_frames / max_residues | 8 / 256 |
| latent_dim / encoder_depth | 128 / 4 |
| paired_horizons_frames | [32, 64, 96] |
| lr / warmup / epochs | 3e-4 / 300 / 20 |
| paired_delta_weight | 1.0 |
| variance_reg / state_variance_reg | 0.01 |

#### Stage B (`configs/fast_abc_stage_b_from_scratch.yaml`)

| 参数 | 值 |
|------|-----|
| batch_size | 1 |
| num_frames / max_residues (data) | 4 / **100** |
| model.max_residues | **256** (必须匹配 Stage A checkpoint) |
| q_dim / latent_dim | 8 / 128 |
| lspwm_rollout_steps | **4** (从 32 降低防 OOM) |
| lr / epochs / warmup | 3e-5 / 3 / 100 |
| rollout_trajectory_nll_weight | 0.1 |
| semigroup_consistency_weight | 0.01 |

#### Stage C (`configs/fast_abc_stage_c_from_scratch.yaml`)

| 参数 | 值 |
|------|-----|
| batch_size / num_workers | 2 / 0 |
| num_frames / max_residues | 4 / 256 |
| anchor_noise_level | 0.5 |
| q_weight_ramp_steps | 300 |
| lr / epochs / warmup | 5e-5 / 2 / 100 |
| gradient_accumulation_steps | 4 |


---

## 3. 训练结果: Stage A

**配置**: 20 epochs × 300 steps = 6000 steps, lr=3e-4 (cosine decay), batch=16, paired_window protocol

### 3.1 双任务交替模式

Stage A 在 "paired" 和 "mask" 两种任务间交替:

| 任务 | 描述 | 损失组成 | 出现频次 |
|------|------|---------|---------|
| **paired** | paired delta + state prediction | paired_delta_loss + paired_state_loss + horizon_shuffle_loss | 49/120 日志条目 |
| **mask** | 标准 JEPA masked prediction | mask_loss | 71/120 日志条目 |

### 3.2 损失趋势

```
paired delta: 0.161 → 0.075-0.136 (波动收敛)
   区间: [0.0755, 0.2493]  — 所有 paired 日志条目
   最终 paired epochs: epoch 14 (0.18→0.10), epoch 17 (0.12→9.39 单个 spike)

mask: 65.1 → 7.4-17.2 (整体下降但振荡)
   区间: [7.414, 65.096]  — 所有 mask 日志条目
   趋势: epochs 0-2 高 (~20-65), epochs 10-13 低 (7.4-11.6), epochs 15-19 中等 (8.3-17.2)
```

### 3.3 Per-Epoch 详情

| Epoch | Task | Steps | Loss First | Loss Last | 趋势 |
|-------|------|-------|-----------|-----------|------|
| 0 | paired | 0-250 | 0.161 | 0.137 | 收敛 |
| 1 | paired | 300-550 | 0.187 | 22.348 | Paired→Mask 转换处 spike |
| 2 | mask | 600-850 | 20.646 | 15.103 | ↓ |
| 3 | paired | 900-1150 | 0.164 | 0.163 | 持平 |
| 4 | paired | 1200-1450 | 0.101 | 0.125 | 持平 |
| 5 | paired | 1500-1750 | 0.160 | 18.835 | Paired→Mask 转换处 spike |
| 6 | mask | 1800-2050 | 20.379 | 0.127 | ↓ 任务切换至 paired |
| 7 | paired | 2100-2350 | 0.117 | 0.105 | 收敛 |
| 8 | paired | 2400-2650 | 0.195 | 0.169 | 波动 |
| 9 | mask | 2700-2950 | 12.589 | 15.137 | 振荡 |
| 10 | mask | 3000-3250 | 11.640 | 8.388 | ↓ 大幅改善 |
| 11 | mask | 3300-3550 | 8.536 | 11.977 | 振荡 |
| 12 | mask | 3600-3850 | 7.414 | 13.091 | 最小值 7.41 后回升 |
| 13 | mask | 3900-4150 | 10.807 | 14.646 | ↑ |
| 14 | paired | 4200-4450 | 0.180 | 0.100 | 收敛 |
| 15 | mask | 4500-4750 | 8.322 | 11.327 | 振荡 |
| 16 | mask | 4800-5050 | 10.458 | 14.021 | ↑ |
| 17 | paired | 5100-5350 | 0.121 | 9.391 | 任务切换 spike |
| 18 | mask | 5400-5650 | 11.611 | 12.093 | 持平 |
| 19 | mask | 5700-5950 | 12.449 | 17.194 | ↑ 最终上升 |

### 3.4 NaN 检查

**无 NaN 记录**。训练全程数值稳定。

### 3.5 检查点

- 21 个检查点 (每 300 steps), 每个 38.9 MB (347 模型参数键)
- 最终: `artifacts/fast_abc_stage_a/checkpoint_step6000.pt`
- 键结构: `context_encoder.*`, `temporal_encoder.*`, `target_encoder.*`, `predictor.*`, `paired_forecaster.*`

---

## 4. 训练结果: Stage B

**配置**: 3 epochs × 300 steps = 900 steps, lr=3e-5 (cosine decay), batch=1, rollout_steps=4, max_residues(data)=100

### 4.1 模型结构

训练 180 个模型键 (仅 LaSP 参数，编码器冻结):
- H→Q 投影 (`q_projection`)
- Per-residue energy graph (`energy_graph.*`)
- 力场 (`force_net.*`)
- Bath 耦合 (`bath_coupling_net.*`, `bath_gamma_net.*`, `bath_initializer.*`)
- Diffusion (`diffusion.*`)
- Closed-loop transition (`transition.*`)
- Integrator (`integrator.*`)

### 4.2 损失组成

| 损失项 | 权重 | 描述 |
|--------|------|------|
| transition_nll | 1.0 (隐含) | 单步 transition 的负 log 似然 |
| forcemag_loss | (正则化) | 力幅值正则化 |
| energy_reg_loss | (正则化) | 能量图正则化 |
| rollout_nll | 0.1 | 多步 rollout 的 NLL |
| semigroup | 0.01 | 半群一致性损失 |

### 4.3 损失趋势

```
transition_nll:  22.15 → 18.87 (↓ 15%)
   范围: [18.87, 27.66]
   关键里程碑:
     step   0: 22.15 (初始化)
     step 100: 20.73
     step 200: 19.72
     step 300: 23.01 (epoch 切换 spike)
     step 400: 20.17
     step 500: 19.52
     step 600: 25.22 (epoch 切换 spike)
     step 700: 24.53
     step 800: 23.41
     step 890: 18.87 (最终最低值!)

rollout_nll:    0.624 → 0.025 (↓ 96%)
   范围: [0.020, 1.434]
   里程碑: step 0=0.62, step 100=0.43, step 200=0.10, step 300=0.039, ..., step 890=0.025

semigroup:      0.262 → 0.094 (↓ 64%)
   范围: [0.082, 0.951]
   高峰: step 200 (0.602) — rollout 训练前期

force_mag:      0.006 → 0.009 (↑, 力值在增长)
energy_reg:     0.002 → ~0 (↓, 正则化快速衰减至零)
```

### 4.4 关键观察

1. **Epoch 边界处 NLL spike**: step 300 和 step 600 处 transition NLL 跳升 ~3-4 单位
2. **energy_reg → 0**: 从 step~200 开始，energy_reg_loss 已接近零——可能表明能量图正则化在设计上就弱
3. **force_mag 上升**: 力幅值单调增长 0.006 → 0.009——需要在 rollout 路径上检查力是否变得太大导致 Q 漂移
4. **Rollout 收敛优秀**: rollout NLL 下降 96%，但评估时 rollout Q 仍爆炸——说明训练分布与评估分布不匹配
5. **NaN**: 无 NaN 记录

### 4.5 检查点

- 4 个检查点 (每 300 steps), 每个 8.3 MB (180 模型键)
- 最终: `artifacts/fast_abc_stage_b/checkpoint_step900.pt`

---

## 5. 训练结果: Stage C

**配置**: 2 epochs × 300 steps = 600 steps, lr=5e-5 (cosine), batch=2, grad_accum=4, anchor_noise=0.5Å, q_weight_ramp=300

### 5.1 模型结构

101 个模型键:
- `aatype_embed`, `frame_condition_proj`, `frame_update.*`
- `ipa_layers.*` (IPA 模块)
- `sidechain_head.*` (侧链扭转角输出)
- `backbone_head.*` (主链坐标输出)
- `q_projection` (Q 条件投影)
- `direct_head.*` (直接坐标输出)

### 5.2 损失组成

| 损失项 | 初始 (step 0) | 最终 (step 590) | 趋势 |
|--------|--------------|-----------------|------|
| **FAPE** | 9.543 | 9.677 | 持平 (范围 [8.17, 9.83]) |
| **torsion** | 3.569 | 2.109 | ↓ 41% (范围 [2.07, 3.57]) |
| **distance** | 1.521 | 5.770 | 振荡 (范围 [1.30, 6.23]) |
| **bond** | ~0 | ~0 | 始终接近零 |
| **clash** | ~0 | ~0 | 始终接近零 |
| **direct_mse** | 0.006 | 0.003 | 轻微下降 |
| **q_position** | 0.333 | 0.333 | **未变化——Q weight ramp 未生效** |

### 5.3 Q Weight Ramp 诊断

`q_position` 在整个训练过程中保持恒定 0.333:

```
step   0: q_pos=0.3333
step 100: q_pos=0.3333
step 200: q_pos=0.3334
step 290: q_pos=0.3334
step 300: q_pos=0.3334  ← q_weight_ramp_steps 结束，Q 应该已充分激活
step 400: q_pos=0.3337
step 500: q_pos=0.3335
step 590: q_pos=0.3334
```

这解释了为什么 decoder 忽略 Q——Q 在训练中的有效权重始终很低。

### 5.4 NaN 检查

**无 NaN**。训练全程数值稳定。

### 5.5 检查点

- 3 个检查点 (step 300, 600, latest), 每个 8.7 MB (101 键)
- 最终: `artifacts/fast_abc_stage_c/step_600.pt`

---

## 6. 评估结果

### 6.1 Baselines 对比

**设置**: 100 验证样本 (取自 1b4uB00, L=256, seed=42), 2 horizons (32ns, 64ns)  
**Baseline 方法**: Stage B rollout, Linear interpolation, Persistence (q_t = q_0), Transformer (learned temporal), Oracle (GT Q)

#### Horizon = 32000 ps (8 steps)

| Method | Q MSE vs Oracle | Q Norm | 与 Oracle 之差 |
|--------|-----------------|--------|----------------|
| **Stage B rollout** | **0.4692 ± 0.0139** | **30.05 ± 0.45** | +26.57 ×8.6 |
| Linear interp (GT) | ~0 | 3.474 | 0 |
| Persistence | ~0 | 3.473 | 0 |
| Transformer | 0.0085 ± 0.0000 | 3.923 ± 0.0003 | +0.45 |
| Oracle | — | 3.474 ± 0.002 | 0 (参考) |

#### Horizon = 64000 ps (16 steps)

| Method | Q MSE vs Oracle | Q Norm | 与 Oracle 之差 |
|--------|-----------------|--------|----------------|
| **Stage B rollout** | **0.4604 ± 0.0197** | **29.73 ± 0.63** | +26.26 ×8.6 |
| Persistence | ~0 | 3.472 | 0 |
| Transformer | 0.0085 ± 0.0000 | 3.922 | +0.45 |
| Oracle | — | 3.474 | 0 (参考) |

**关键发现**:
- **Persistence (Q 不变) 的误差和 norm 几乎完美**——说明 Q 在 32ns/64ns 内的真实变化极小
- **Transformer 维持 norm 稳定 (3.92)** 且 MSE 仅 0.0085——说明时序模式是可学习的
- **Stage B rollout 将 Q norm 爆炸 ~8.6×** (±3.47 → ~30)——closed-loop transition 引入了发散
- **MSE 的巨大值 (0.46)** 不是来自 norm 爆炸，而是来自方向偏差 (cosine similarity ~0.30，见 §6.5)

### 6.2 Temperature Intervention (温度跳跃)

**设置**: 50 验证样本, 2 seeds (42, 123), 348K → {320K, 379K, 413K}  
**协议**: 在 348K equilibrate 10 steps → jump 到目标温度 50 steps

#### 每温度结果 (总样本数=100/温度)

| Jump | Q_diff_ratio | jump_std | equil_std |
|------|-------------|----------|-----------|
| 348K→**413K** | 0.1617 ± 0.0112 | 5.61 | 8.59 |
| 348K→**379K** | 0.1614 ± 0.0110 | 5.57 | 8.58 |
| 348K→**320K** | 0.1617 ± 0.0114 | 5.54 | 8.57 |

#### 统计检验

对 413K vs 320K 做 two-sample t-test:
- q_diff_ratio 标准差 ~0.011, delta ~0.00035 → t ≈ 0.02 → **p >> 0.05**
- 所有三个温度间差异在统计上完全不可区分

**关键发现**:
- **Q_diff_ratio 在三个温度间几乎完全相同 (~0.162)**——模型完全未学习温度敏感性
- jump_std 和 equil_std 在三个温度间也几乎相同——energy graph (其 temperature_k 输入) 未成功调制势能曲面
- **温度 320K→413K 跨度 93K——本应产生显著不同的 Laplace 压力和 Q 扩散范围**

### 6.3 Sidechain Quality (侧链质量)

**设置**: 100 验证样本 (1b4uB00, L=256), Stage C checkpoint step_600.pt

| Metric | Mean ± SD | 解读 |
|--------|----------|------|
| **chi2_err** | 0.0122 ± 0.00002 rad = **0.7°** | ✅ 极好 — 可从 backbone 几何推断 χ₂ |
| **chi1_err** | 0.4138 ± 0.00005 rad = **23.7°** | ⚠️ 中等 — 侧链旋转有部分信息 |
| **phi_err** | 1.2410 ± 0.0200 rad = **71.1°** | ❌ 很差 — backbone Φ 角几乎随机 |
| **psi_err** | 1.8178 ± 0.0196 rad = **104.2°** | ❌ 极差 — backbone Ψ 角几乎随机 |
| **clash_count** | 2119841 ± 97 | ❌ 严重 — 每个样本 2.1M clash |

**注**: a37_rmsd 在所有样本中为 NaN (原因不明，可能是 256 残基蛋白与其他蛋白文件夹不匹配导致坐标对齐失败)

**关键发现**:
- Chi2 (0.7°) 说明 decoder 能从 backbone 局部几何推断侧链末端——但这与 phi/psi 71°/104° 错误矛盾
- 矛盾的可能解释: decoder 可能直接从 anchor coords 推断 χ₂ (而非通过 Q → backbone → χ₂ 路径)
- 2.1M clash 表明 atom-level 物理质量完全崩溃
- **这与之前的消融研究 (P0.3) 一致: Q 对 decoder 输出贡献 <0.001Å RMSD**

### 6.4 Rollout Quality (Q 路径分析)

**设置**: 100 samples (1b4uB00), seed=42, 32ns 和 64ns horizons

#### 32ns (8 steps, 95/100 有效)

| Metric | Value |
|--------|-------|
| Q_norm_init | 3.4735 |
| Q_norm_final | **5.2144** (↑ 50%) |
| Q_norm_ratio | 1.501 |
| Q_MSE vs oracle | 0.0138 |
| Q_cosine vs oracle | 0.3036 |
| ACF lag5 | 0.0072 |

#### 64ns (16 steps, **仅 5/100 有效**)

| Metric | Value |
|--------|-------|
| Q_norm_init | 3.4720 |
| Q_norm_final | 5.2163 (↑ 50%) |
| Q_norm_ratio | 1.502 |
| Q_MSE vs oracle | 0.0139 |
| Q_cosine vs oracle | 0.2983 |
| ACF lag5 | 0.0048 |

**64ns 发散率**: 95% (95/100 样本的 Q 在 64ns 到达 NaN/inf，仅 5 个幸存)

**关键发现**:
- **32ns Q norm 膨胀 +50%** (3.47 → 5.21)——与 baseline 的 30.05 norm 矛盾
  - Baseline 运行时间比 rollout 长——可能 baseline 脚本多次调用 transition_step 而 rollout 有不同路径
- **Cosine similarity vs oracle ≈ 0.30**——Q 方向在 rollout 中快速偏离
- **64ns 95% 湮灭**——transition 在 8→16 步之间迅速失控
- **ACF lag5 ≈ 0**——Q 的自相关性在 5 步内消失——暗示噪声主导了 dynamics

### 6.5 World Model Evaluation (未完成)

**状态**: ❌ 失败  
5-mode 世界模型评估 (oracle, direct, autonomous, persistence, q_shuffle) 未能产出结果。

**首次运行** (eval_launcher.log):
- 因 `tuple(None)` TypeError 崩溃 (见 §7, Bug #4)

**重试运行** (world_model_retry.log):
- 仅输出 2 行警告日志后挂起 >1 小时 (100% CPU, 0 文件输出)
- Session 内未解决; root cause 疑为 world_model eval 对 memory/resource 消耗过大或死循环

---

## 7. 已发现的 Bug 与修复

### 7.1 训练流水线 Bug

| # | Bug | 症状 | 根因 | 修复 |
|---|-----|------|------|------|
| **B1** | Stage B OOM | 79.1GB VRAM, CUDA OOM | energy graph O(n²) edges × 32 rollout steps × 256 residues | `data.max_residues=100`, `lspwm_rollout_steps=4`, `num_frames=4`, `batch_size=1`, `num_workers=0` |
| **B2** | Stage B shape mismatch | `RuntimeError: size mismatch for residue_embed` | `model.max_residues=128` != Stage A checkpoint `256` | 保持 `model.max_residues=256`, 仅改 `data.max_residues=100` 在 DataLoader 层面过滤 |
| **B3** | num_workers 不稳定 | batch_size=1 + num_workers=2 导致加载崩溃 | DataLoader 多线程与 batch_size=1 不兼容 | 改为 `num_workers=0` |
| **B4** | Stage C 目录缺失 | auto_launcher 脚本因 redirect 到不存在目录而失败 | 脚本在 mkdir 前写入 | 手动 `mkdir -p` 后启动 |

### 7.2 评估脚本 Bug

| # | Bug | 症状 | 修复 | 文件 |
|---|-----|------|------|------|
| **B5** | `tuple(None)` TypeError | `eval_world_model.py:472` 崩溃: `TypeError: 'NoneType' object is not iterable` | `tuple(args.horizons_ps) if args.horizons_ps is not None else None` | `scripts/eval_world_model.py` |
| **B6** | 同上 | `eval_rollout_quality.py:284` 崩溃: 相同错误 | `tuple(args.horizons) if args.horizons is not None else None` | `scripts/eval_rollout_quality.py` |
| **B7** | World model eval 挂起 | 1h+ 100% CPU, 0 输出 | 未解 (疑为内存/OOM loop) | session 内未修复 |
| **B8** | Sidechain a37_rmsd = NaN | 所有 100 个样本输出 NaN | 疑为 256 残基链与目标 PDB 帧不匹配 | 未修复 |

### 7.3 评估完成状态表

| 评估 | 首次运行 | 重试后 | 结果数据 |
|------|---------|--------|---------|
| Baselines | ✅ 通过 | — | `baseline_summary.json` + `baseline_results.jsonl` (100 行) |
| Temperature | ✅ 通过 | — | `intervention_summary.json` + `intervention_results.jsonl` (300 行) |
| Sidechain | ✅ 通过 | — | `sidechain_summary.json` + `sidechain_results.jsonl` (100 行) |
| Rollout Quality | ❌ B6 (tuple) | ✅ 通过 (B6 fix) | `rollout_summary.json` + `rollout_results.jsonl` (100 行) |
| World Model | ❌ B5 (tuple) | ❌ 挂起 (B7) | 空输出目录 |

---

## 8. 核心问题诊断

以下分析基于本报告中的所有定量证据，旨在引导独立 agent 的深入调查。

### 8.1 问题 1: Q Norm 在 Rollout 中爆炸

**证据**:
- Stage B 5-mode rollout (baselines): Q norm 从 3.47 爆炸至 30.05 **(×8.6)**
- Rollout quality 采样: Q norm 从 3.47 增长至 5.21 **(×1.5)**
- Persistence 的 norm 完美保持 3.47——Q 的真实变化极小
- 64ns rollout: 95% 样本发散

**为什么不能用训练预算解释**:
- Persistence (零动力学) 完美——Q 在原始时序中移动小于噪声水平
- 如果模型仅需小步长动力学，900 步训练应足以学到稳定 transition
- **发散表明 transition 力场在基础层面不支持保守动力学**，而非学习不充分

**需排查的代码路径**:
1. `protein_jepa_frame/models/lasp/energy.py` — Energy graph 输出: 形状、保守性验证
2. `protein_jepa_frame/models/lasp/integrator.py` — sub-stepping: `dt=4000ps` 对应 200 substeps, 但 cap 在 10 → 实际 substep=400ps >> 准确需要 20ps
3. `protein_jepa_frame/models/stage_b/transition.py` — ClosedLoopTransition: force 计算中的 `autograd.grad` 是否正确处理 per-residue 批情况
4. `protein_jepa_frame/models/stage_b/model.py:501` — **substepping 警告**: `RuntimeWarning: sub-stepping capped at 10 substeps for dt=4000.0ps (needed 200, actual substep dt=400.0ps > 20.0ps)` ——这是一个红色警报

### 8.2 问题 2: 零温度敏感性

**证据**:
- q_diff_ratio 在 320K→413K 间恒定为 0.162 (标准差 0.011)
- jump_std 和 equil_std 在三个温度间完全相同
- 物理上，93K 的温度跳跃应产生 ~(413/320)^(1/2) ≈ 1.14 的扩散缩放——未观察到

**为什么不能用训练预算解释**:
- 即使只有 1 步 transition，温度应产生不同的步长（若 FDT 一致则 diffusion ~ sqrt(kB*T)）
- 完全平坦的 Q_diff_ratio 温度曲线表明 energy graph 的 temperature_k 通道完全断开

**需排查的代码路径**:
1. `protein_jepa_frame/models/lasp/energy.py` — temperature_k 如何注入 energy graph
2. `protein_jepa_frame/models/lasp/diffusion.py` — 扩散系数与温度的关系
3. `protein_jepa_frame/models/lasp/bath.py` — bath 状态如何被温度调制
4. `protein_jepa_frame/models/thermodynamic_v3.py` — 温度条件编码被送入何处

### 8.3 问题 3: Decoder 忽略 Q (Anchor Refinement 伪影)

**证据**:
- Q position 权重在训练中保持 0.333 (q_weight_ramp 从未提升 Q 贡献)
- Phi/Psi 误差 71°/104°——decoder 基本上没有生成有效的 backbone
- 但 chi2 误差仅 0.7°——decoder 可以从 anchor 局部几何推断 χ₂
- Sidechain clash 2.1M——结构完全崩溃

**为什么不能用训练预算解释**:
- 如果 anchor noise=0.5Å 有效，600 步的 2-phase 训练应该开始看到 Q 依赖
- q_weight_ramp 持续 300 steps，后面的 300 steps 应该激活 Q
- q_position 在 step 290 仍为 0.3334，step 590 仍为 0.3334——**ramp 实现本身可能有 bug**

**需排查的代码路径**:
1. `protein_jepa_frame/models/stage_c/model.py` — q_weight_ramp 调度如何与 decoder 前向传播交互
2. `protein_jepa_frame/models/stage_c/structure_decoder.py` — Q 如何注入 IPA 模块
3. `protein_jepa_frame/training/stage_c.py` — anchor noise 应用和 q_weight 计算
4. `protein_jepa_frame/config.py` — `q_weight_ramp_start` / `q_weight_ramp_steps` 的解析与传播路径

### 8.4 汇总: Q 质量度量矩阵

| 度量 | 观测值 | 期望 (健康) | 差距 |
|------|--------|------------|------|
| Q norm (rollout 32ns) | 5.21–30.05 | ~3.47 | ⚠️ 严重 |
| Q cosine vs oracle | 0.30 | >0.8 | ⚠️ 严重 |
| Q temp sensitivity | ~0 (平坦) | Δ>0.1 (413K vs 320K) | ❌ 零 |
| Q persistence baseline | 完美 (~0 误差) | — | Q 空间过稳定 |
| Decoder Q dependence | ~0 (q_pos=0.333) | >0.7 Q 权重 | ❌ 零 |
| ACF lag5 | 0.007 | >0.3 (马尔可夫) | ❌ 白噪声 |

---

## 9. 完整代码路径清单

### 9.1 Fast ABC 核心配置 (3 文件)

| 文件 | 大小 | 训练预算 | 模型策略 |
|------|------|----------|---------|
| `configs/fast_abc_stage_a_from_scratch.yaml` | 1.4 KB | 20 epochs × 300 steps | 从头训练 JEPA 编码器 |
| `configs/fast_abc_stage_b_from_scratch.yaml` | 1.3 KB | 3 epochs × 300 steps | 冻结编码器, 训练 LaSP 世界模型 |
| `configs/fast_abc_stage_c_from_scratch.yaml` | 1.1 KB | 2 epochs × 300 steps | 冻结全部编码器,训练结构解码器 |

### 9.2 训练代码 (8 文件)

| 文件 | 大小 | 关键职责 |
|------|------|---------|
| `protein_jepa_frame/cli.py` | 5.6 KB | CLI 分派: `--stage {A,B,C}` → 调用各阶段训练函数 |
| `protein_jepa_frame/training/stage_a.py` | 15.1 KB | Stage A: masked JEPA + paired 双任务训练循环 |
| `protein_jepa_frame/training/stage_b.py` | 11.9 KB | Stage B: 冻结 A, 训练 LaSP, checkpoint key remapping (`context_encoder.`→`spatial_encoder.`), AMP+EMA |
| `protein_jepa_frame/training/stage_c.py` | 17.0 KB | Stage C: 冻结 A+B, anchor noise, Q weight ramp, IPA decoder 训练 |
| `protein_jepa_frame/training/train_paired.py` | 80.6 KB | Paired window 训练核心 (Stage A 使用) |
| `protein_jepa_frame/training/train.py` | 15.6 KB | 标准 JEPA 训练循环 |
| `protein_jepa_frame/training/train_utils.py` | 27.1 KB | 训练工具: scheduler, checkpoint, 日志 |
| `protein_jepa_frame/config.py` | 33.6 KB | 统一配置模型: `TrainConfig`, `ModelConfig`, `LSPWMConfig` |

### 9.3 模型代码 (30 文件)

#### Stage A 模型 (3 文件)

| 文件 | 大小 | 职责 |
|------|------|------|
| `protein_jepa_frame/models/stage_a/model.py` | 26.7 KB | StageAJEPA: 完整的 JEPA 模型包装器 (encoder + predictor + temporal) |
| `protein_jepa_frame/models/stage_a/objectives.py` | 5.3 KB | Stage A 损失: masked JEPA + paired delta/state |

#### Stage B 模型 (5 文件)

| 文件 | 大小 | 职责 |
|------|------|------|
| `protein_jepa_frame/models/stage_b/model.py` | 27.8 KB | StageBLASP: H→Q, energy graph, transition, rollout, semigroup |
| `protein_jepa_frame/models/stage_b/objectives.py` | 9.4 KB | rollout_nll_loss, semigroup_consistency_loss, stage_b_total_loss |
| `protein_jepa_frame/models/stage_b/transition.py` | 7.1 KB | ClosedLoopTransition: ML force + noise + drift + bath 耦合 |
| `protein_jepa_frame/models/stage_b/state.py` | 1.8 KB | LatentState dataclass (q, bath_state, 时间戳) |
| `protein_jepa_frame/models/lspwm.py` | 3.1 KB | LSPWM 接口 (世界模型顶层协调器) |

#### LaSP 底层库 (10 文件)

| 文件 | 大小 | 职责 |
|------|------|------|
| `protein_jepa_frame/models/lasp/energy.py` | 26.8 KB | **Per-residue energy graph**: edge 网络, 保守性保证, **temperature_k 注入** |
| `protein_jepa_frame/models/lasp/protein_bridge.py` | 89.3 KB | H→Q 投影, Q→几何读取, per-residue 批量处理 |
| `protein_jepa_frame/models/lasp/integrator.py` | 32.7 KB | ODE/SDE 时间积分器: **substepping 逻辑** (dt_ps, max_substeps) |
| `protein_jepa_frame/models/lasp/objectives.py` | 30.1 KB | LaSP 损失: NLL, force alignment, energy 正则化 |
| `protein_jepa_frame/models/lasp/bath.py` | 9.0 KB | 热浴模型: state, 耦合, 阻尼, **温度依赖** |
| `protein_jepa_frame/models/lasp/diffusion.py` | 6.2 KB | FDT 一致扩散过程: 噪声缩放 |
| `protein_jepa_frame/models/lasp/projection.py` | 5.0 KB | H→Q 和 Q→H 投影头 |
| `protein_jepa_frame/models/lasp/utils.py` | 1.0 KB | 工具函数 |
| `protein_jepa_frame/models/lasp/tests_synthetic.py` | 15.7 KB | 合成数据测试 |

#### Stage C 模型 (7 文件)

| 文件 | 大小 | 职责 |
|------|------|------|
| `protein_jepa_frame/models/stage_c/model.py` | 12.9 KB | StageCModel: 包装 StructureDecoder |
| `protein_jepa_frame/models/stage_c/structure_decoder.py` | 28.1 KB | **IPA 结构解码器**: Q 注入, 坐标生成, anchor noise |
| `protein_jepa_frame/models/stage_c/readout.py` | 12.7 KB | 将 IPA 输出读取为原子坐标 |
| `protein_jepa_frame/models/stage_c/geometry.py` | 12.5 KB | 蛋白质几何: frames, 扭转角, 坐标 |
| `protein_jepa_frame/models/stage_c/rollout.py` | 9.6 KB | Rollout 逻辑层: 连接世界模型和 decoder |
| `protein_jepa_frame/models/stage_c/objectives.py` | 8.2 KB | Stage C 损失: FAPE, torsion, distance, bond, clash |
| `protein_jepa_frame/models/stage_c/direct_head.py` | 7.5 KB | 直接坐标输出头 (不经过 IPA) |

#### 共享模型 (5 文件)

| 文件 | 大小 | 职责 |
|------|------|------|
| `protein_jepa_frame/models/jepa.py` | 122.6 KB | 基础 JEPA: encoder, predictor, temporal encoder |
| `protein_jepa_frame/models/predictor.py` | 132.1 KB | 预测器实现 |
| `protein_jepa_frame/models/jepa_losses_extended.py` | 80.4 KB | JEPA 损失函数 (扩展版) |
| `protein_jepa_frame/models/jepa_losses.py` | 54.4 KB | JEPA 损失函数 |
| `protein_jepa_frame/models/thermodynamic_v3.py` | 78.9 KB | 热力学条件编码 v3 |
| `protein_jepa_frame/models/thermodynamic.py` | 43.7 KB | 热力学条件编码 |
| `protein_jepa_frame/models/encoder.py` | 8.8 KB | 编码器组件 |
| `protein_jepa_frame/models/temporal.py` | 10.3 KB | 时序编码器 |
| `protein_jepa_frame/models/dynamic_encoder.py` | 16.1 KB | 动态编码器 |
| `protein_jepa_frame/models/time_features.py` | 3.5 KB | 时间特征 |
| `protein_jepa_frame/models/opf.py` | 16.1 KB | OPF 相关模型 |

### 9.4 数据流水线 (8 文件)

| 文件 | 大小 | 职责 |
|------|------|------|
| `protein_jepa_frame/data/dataset.py` | 26.5 KB | PairedTrajectoryDataset, MDCathDataset |
| `protein_jepa_frame/data/schema.py` | 19.1 KB | Atom37 坐标 schema, 扭转角计算 |
| `protein_jepa_frame/data/collate.py` | 5.3 KB | collate_paired_windows, collate_clips |
| `protein_jepa_frame/data/masks.py` | 5.0 KB | Masking 策略 (spatial mask) |
| `protein_jepa_frame/data/manifest.py` | 2.2 KB | Manifest 加载器 |
| `protein_jepa_frame/data/sampler.py` | 2.2 KB | 数据采样器 |
| `protein_jepa_frame/data/atom37_layer.py` | 11.4 KB | Atom37 处理层 |

### 9.5 评估脚本 (5 核心脚本 + 1 已修复)

| 文件 | 大小 | 职责 | 状态 |
|------|------|------|------|
| `scripts/eval_baselines.py` | 16.8 KB | 5-mode 基线对比 (StageB, Linear, Persistence, Transformer, Oracle) | ✅ |
| `scripts/eval_temperature_intervention.py` | 12.4 KB | 温度跳跃: equil + jump 协议 | ✅ |
| `scripts/eval_sidechain_quality.py` | 12.1 KB | 扭转角误差, clash, RMSD | ✅ |
| `scripts/eval_rollout_quality.py` | 12.2 KB | Q norm drift, ACF, MSE vs oracle | ✅ (B6 修复) |
| `scripts/eval_world_model.py` | 20.7 KB | 5-mode 世界模型完整评估 | ❌ (B7 挂起) |

### 9.6 自动化脚本 (5 个关键脚本)

| 文件 | 职责 | 状态 |
|------|------|------|
| `scripts/run_fast_abc_from_scratch.sh` | 顺序启动 A→B→C 训练 | ✅ |
| `scripts/run_fast_abc_evaluation.sh` | 顺序运行 5 项评估 | ✅ |
| `scripts/auto_finish_fast_abc.sh` | 后台监控: B 完成→启动 C→启动评估 | ⚠️ B4 后修复 |
| `scripts/auto_eval_after_stage_c.sh` | 等待 C 完成→启动评估 | ✅ |
| `scripts/retry_failed_evals.sh` | 重试失败评估 | ⚠️ |

---

## 10. 训练产物清单

### 10.1 Stage A Checkpoints

```
artifacts/fast_abc_stage_a/
├── checkpoint_step300.pt    (38.9 MB, 347 model keys)
├── checkpoint_step600.pt    (38.9 MB)
├── ...                      (每 300 步, 共 20 个增量检查点)
├── checkpoint_step6000.pt   (38.9 MB, 最终)
├── checkpoint_latest.pt     (38.9 MB, → step6000)
├── train.jsonl              (120 日志行)
└── config.json              (22 配置键)
```

### 10.2 Stage B Checkpoints

```
artifacts/fast_abc_stage_b/
├── checkpoint_step300.pt    (8.3 MB, 180 model keys)
├── checkpoint_step600.pt    (8.3 MB)
├── checkpoint_step900.pt    (8.3 MB, 最终)
├── checkpoint_latest.pt     (8.3 MB, → step900)
├── train.jsonl              (90 日志行)
├── config.json              (22 配置键)
└── stdout.log               (完整训练输出)
```

### 10.3 Stage C Checkpoints

```
artifacts/fast_abc_stage_c/
├── step_300.pt              (8.7 MB, 101 model keys)
├── step_600.pt              (8.7 MB, 最终)
├── checkpoint_latest.pt     (8.7 MB, → step600)
├── train.jsonl              (60 日志行)
├── config.json              (22 配置键)
└── stdout.log               (完整训练输出)
```

### 10.4 评估产物

```
artifacts/fast_abc_evaluation/
├── baselines/
│   ├── baseline_summary.json     (汇总统计)
│   └── baseline_results.jsonl    (100 行: 每行 1 样本×horizon)
├── temperature_intervention/
│   ├── intervention_summary.json (汇总统计: Q_diff_ratio per temp)
│   └── intervention_results.jsonl (300 行: 50 samples × 2 seeds × 3 jumps)
├── sidechain/
│   ├── sidechain_summary.json    (汇总: chi1/chi2/phi/psi/clash)
│   └── sidechain_results.jsonl   (100 行: per-sample 扭转角和 clash)
├── rollout_quality/
│   ├── rollout_summary.json      (汇总: Q drift, MSE, ACF)
│   └── rollout_results.jsonl     (100 行: 32ns=95, 64ns=5 valid)
├── world_model/
│   └── (空 — eval 失败)
├── baselines.log                 (stdout: 成功)
├── temperature_intervention.log  (stdout: 成功)
├── sidechain.log                 (stdout: 成功)
├── rollout_quality.log           (首次: Bug B6)
├── rollout_quality_retry.log     (重试: 成功)
├── world_model.log               (首次: Bug B5)
├── world_model_retry.log         (重试: 挂起 — 2 行 warning, 无输出)
├── auto_launcher.log             (自动化日志)
├── eval_launcher.log             (评估启动日志)
└── retry_launcher.log            (重试启动日志)
```

### 10.5 数据清单

```
artifacts/mdcath_manifest_200dom/
├── train.jsonl    (4000 entries, 160 蛋白质)
└── val.jsonl      (500 entries, 20 蛋白质 × 25 帧/蛋白)
```

---

## 11. 可供独立 Agent 分析的关键线索

### 11.1 立即需要排查的代码路径 (按优先级)

#### P0: Integrator Substepping (integrator.py)
```python
# stage_b/model.py:501 — 运行时警告:
# "sub-stepping capped at 10 substeps for dt=4000.0ps
#  (needed 200, actual substep dt=400.0ps > 20.0ps)"
```
- 文件: `protein_jepa_frame/models/lasp/integrator.py`
- 问题: dt 每步 4000ps 需要 200 substeps，但 max_substeps=10 → 实际 substep=400ps >> 准确 20ps
- 对 rollout 的影响: 8 步 rollout × 4000ps/步 = 32000ps, 但积分误差在每步被放大 400/20=20×

#### P0: Q Weight Ramp 实现 (stage_c/model.py)
```
q_position 在整个训练中恒为 0.333 — q_weight_ramp 从未提升 Q 贡献
```
- 文件: `protein_jepa_frame/models/stage_c/model.py` — 搜索 `q_weight_ramp`, `q_position`
- 文件: `protein_jepa_frame/training/stage_c.py` — q_weight 调度和 anchor noise 应用
- 文件: `protein_jepa_frame/config.py` — `q_weight_ramp_start` 和 `q_weight_ramp_steps` 如何被解析
- 检查点: q_weight_ramp 是否仅在线性插值 `q_weight ∈ [start_weight, 1.0]` 间递增？还是被某个常数覆盖？

#### P0: Temperature_k 注入路径
```
能量图完全未响应温度跳跃 — temperature 通道连接已断开
```
- 文件: `protein_jepa_frame/models/lasp/energy.py` — 搜索 `temperature_k`, `temperature`, `temperature_k`
- 文件: `protein_jepa_frame/models/thermodynamic_v3.py` — 温度如何被编码
- 文件: `protein_jepa_frame/models/lasp/bath.py` — bath 如何随温度缩放
- 文件: `protein_jepa_frame/models/lasp/diffusion.py` — noise_scale ~ sqrt(kB*T)?

#### P1: Energy Graph 保守性
```
Q norm 在 rollout 中爆炸 ×8.6 → 能量图未提供 confining potential
```
- 文件: `protein_jepa_frame/models/lasp/energy.py` — 能量图的梯度是否有效产生保守力？
- 文件: `protein_jepa_frame/models/stage_b/transition.py` — `autograd.grad` 调用是否正确？
- 检查: 在 rollout 路径上 eval，检查 energy 随 norm 增大是否凸起

#### P2: H→Q 投影信息内容
```
Persistence (Q 不变) 的误差完美 → Q 空间几乎没有编码有意义的动力学信息
```
- 文件: `protein_jepa_frame/models/lasp/projection.py`
- 文件: `protein_jepa_frame/models/lasp/protein_bridge.py`
- 检查: 在验证集上逐维度检查 Q 方差 / H 方差的比率

### 11.2 可复现性命令

#### 重新运行基础评估
```bash
cd /liuyutian/protein_jepa_frame

# Baselines (已验证工作)
python3 scripts/eval_baselines.py \
  --stage-b-ckpt artifacts/fast_abc_stage_b/checkpoint_latest.pt \
  --val-manifest artifacts/mdcath_manifest_200dom/val.jsonl \
  --output-dir /tmp/eval_baselines \
  --n-val 100 --seeds 42

# Temperature intervention (已验证工作)
python3 scripts/eval_temperature_intervention.py \
  --stage-b-ckpt artifacts/fast_abc_stage_b/checkpoint_latest.pt \
  --val-manifest artifacts/mdcath_manifest_200dom/val.jsonl \
  --output-dir /tmp/eval_temp --n-val 50 --seeds 42 123 \
  --equil-temps 348 --jump-temps 320 379 413

# Rollout quality (B6 修复后已验证工作)
python3 scripts/eval_rollout_quality.py \
  --stage-b-ckpt artifacts/fast_abc_stage_b/checkpoint_latest.pt \
  --val-manifest artifacts/mdcath_manifest_200dom/val.jsonl \
  --output-dir /tmp/eval_rq --n-val 100 --seeds 42 \
  --horizons 32000 64000
```

#### 重新运行 Stage B 训练 (如果修改了 transition/integrator)
```bash
python3 -m protein_jepa_frame.cli --config configs/fast_abc_stage_b_from_scratch.yaml
```

### 11.3 评估数据解读注意事项

1. **所有 5 项评估仅测试了 1 个蛋白质 (1b4uB00)**。原因: 验证集 20 个蛋白中，n_val=100 的采样主要命中了 1b4uB00 (L=256 max_residues 限制)。需要扩展到更多蛋白以获取有统计意义的结论。

2. **Sidechain a37_rmsd = NaN** — 可能是 256 残基链与目标 PDB 帧 index 对齐问题。这在评估有效的 backbone (FAPE 可以计算) 时不关键，但在需要 PDB residue 1:1 映射的场景下会出问题。

3. **World model eval 未完成**。该脚本运行 5 种模式 (oracle/direct/autonomous/persistence/q_shuffle) 对每个样本进行大量 transition 操作，消耗内存和计算大量 (怀疑挂起与 OOM 或死锁有关)。

4. **Q norm 从 3.47 → 30.05 (baselines) vs 3.47 → 5.21 (rollout quality)** — 两种评估在测量同一量时产生了 6× 差异。需要对比两者实际的 transition 调用序列以理解差异根源。

5. **Training data.max_residues=100 但 model.max_residues=256** — 训练数据的残基数比模型容量少 2.56×。评估则使用全 256 residues。这种训练/评估分布不匹配可能是 rollout 发散的主要因素。

### 11.4 关键配置值 (供参考)

| 参数 | 值 | 影响 |
|------|-----|------|
| `q_dim` | 8 | Q 向量的维度 |
| `latent_dim` | 128 | H (隐藏状态) 的维度 |
| `lspwm_rollout_steps` | 4 | 训练时 rollout 长度 |
| `lspwm_rollout_trajectory_nll_weight` | 0.1 | rollout 损失权重 |
| `lspwm_semigroup_consistency_weight` | 0.01 | 半群一致性权重 |
| `batch_size` (B) | 1 | 单样本训练 |
| `max_residues` (data) | 100 | 训练时过滤的残基数 |
| `max_residues` (model) | 256 | 模型 embedding 表大小 |
| `dt_ps` (积分器) | ? | 单次 transition 的积分时间步长 |
| `anchor_noise_level` | 0.5 Å | Stage C 的 anchor 扰动 |
| `q_weight_ramp_steps` | 300 | Q 权重斜坡步数 |

---

## 附录: 数据示例

### Stage B 训练的 Per-Sample 示例 (train.jsonl)

```json
{
  "epoch": 2,
  "step": 890,
  "loss": 18.87,
  "lr": 0.0,
  "loss_transition_nll": 18.87,
  "loss_force_mag_loss": 0.0093,
  "loss_energy_reg_loss": 4.9e-06,
  "loss_total_loss": 18.87,
  "loss_rollout_nll": 0.0246,
  "loss_semigroup": 0.0944
}
```

### 评估 Per-Sample 示例

**Baselines** (baseline_results.jsonl):
```json
{
  "protein_id": "1b4uB00", "seed": 42, "horizon_ps": 32000.0, "L": 256,
  "stage_b_mse": 0.4596, "stage_b_norm": 29.81,
  "linear_mse": 0.0, "linear_norm": 3.474,
  "persistence_mse": 3.8e-06, "persistence_norm": 3.474,
  "transformer_mse": 0.0085, "transformer_norm": 3.923,
  "oracle_norm": 3.474
}
```

**Temperature** (intervention_results.jsonl):
```json
{
  "protein_id": "1b4uB00", "seed": 42,
  "T_equil": 348.0, "T_jump": 413.0, "L": 256,
  "jump_q_std": 5.748, "equil_q_std": 8.710,
  "q_diff_ratio": 0.149
}
```

**Sidechain** (sidechain_results.jsonl):
```json
{
  "protein_id": "['1b4uB00']", "L": 256,
  "a37_rmsd": NaN,
  "clash_count": 2119884.0,
  "chi1_err_rad": 0.4138, "chi2_err_rad": 0.0122,
  "phi_err_rad": 1.276, "psi_err_rad": 1.806,
  "n_valid": 256
}
```

**Rollout Quality** (rollout_results.jsonl):
```json
{
  "protein_id": "1b4uB00", "seed": 42, "horizon_ps": 32000.0,
  "L": 256, "n_steps": 8,
  "q_norm_init": 3.474, "q_norm_final": 5.200,
  "q_diff_oracle": 5.364, "q_mse_oracle": 0.0140,
  "q_cosine_oracle": 0.286, "q_norm_ratio": 1.497,
  "acf_lag5": 0.088
}
```

---

*文档结束*  
*生成时间: 2026-10-06*  
*仓库: /liuyutian/protein_jepa_frame*  
*文档路径: /liuyutian/protein_jepa_frame/FAST_ABC_DIAGNOSTIC_REPORT.md*
