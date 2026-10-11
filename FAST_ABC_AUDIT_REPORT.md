# Fast ABC 训练流水线 — 完整审计报告

**日期**: 2026-10-06
**项目**: protein_jepa_frame — Stage A→B→C 串联训练 + 全量评估
**目标**: 以最小迭代预算 (900+900+600=2400 steps) 验证完整流水线，暴露所有基础设施问题和模型瓶颈

---

## 执行摘要

本报告记录了 Fast ABC 训练流水线从零初始化的完整执行过程，包括：

1. **Stage A** (20 epochs × 300 steps = 6000 steps): 联合 masked JEPA + multi-horizon endpoint forecasting。
2. **Stage B** (3 epochs × 300 steps = 900 steps): 冻结 Stage A 编码器，训练 LaSP 世界模型（投影 H→Q, 能量图, diffusion, bath, 闭环 transition）。
3. **Stage C** (2 epochs × 300 steps = 600 steps): 冻结 Stage A/B 编码器，训练 IPA 结构解码器（加 anchor noise 强制 Q 依赖）。
4. **5 项评估**: world model, rollout quality, baselines, temperature intervention, sidechain quality。

**核心发现**:
- 流水线在技术上完整跑通，但 **Q dynamics 质量严重不足**。
- Q norm 在 rollout 中爆炸至 ~30（oracle ~3.5），温度跳跃无区分度（三个温度 Q_diff_ratio 几乎相等），decoder 基本忽略 Q（扭转角误差 Φ=71°, Ψ=104°，clash=2×10⁶）。
- 这些问题**不是训练预算的问题**，而是更深层的架构和/或损失设计问题。

---

## 1. 流水线架构

```
Stage A (encoder)
  │  masked JEPA + paired delta/state prediction
  │  outputs: context_encoder, temporal_encoder, predictor
  ▼
Stage B (world model)
  │  freeze Stage A encoders → train LaSP world model
  │  H→Q projection, energy graph, diffusion, bath, closed-loop transition
  │  losses: transition NLL + rollout MSE + semigroup consistency
  ▼
Stage C (decoder)
  │  freeze Stage A/B → train IPA structure decoder
  │  input: Q (from B) + anchor coords → output: backbone + sidechain coords
  │  anchor noise to force Q dependence
  ▼
Evaluation (5 scripts)
```

---

## 2. 训练日志

### 2.1 Stage A

| 指标 | 初始 (step 0) | 最终 (step 5950) | 趋势 |
|------|---------------|------------------|------|
| total_loss | 0.161 | 17.194 (mask phase) | mask_loss 65→5-17, paired_loss ~0.05-0.20 |
| lr | 1.0×10⁻⁶ | 5.7×10⁻⁸ | cosine decay |
| NaN | — | **无** | — |

**训练步数**: 6000/6000 ✅
**检查点**: `artifacts/fast_abc_stage_a/checkpoint_step6000.pt` (37 MB, 347 keys)

### 2.2 Stage B

| 指标 | 初始 (step 0) | 最终 (step 890) | 趋势 |
|------|---------------|------------------|------|
| transition_nll | 22.146 | 18.870 | ↓ 收敛中 |
| rollout_mse | 0.624 | 0.025 | ↓ 显著下降 |
| semigroup | 0.262 | 0.094 | ↓ 收敛中 |
| lr | 3.0×10⁻⁷ | 1.2×10⁻⁸ | cosine decay |
| NaN | — | **无** | — |

**训练步数**: 900/900 ✅
**检查点**: `artifacts/fast_abc_stage_b/checkpoint_step900.pt` (7.9 MB)

### 2.3 Stage C

| 指标 | 初始 (step 0) | 最终 (step 590) | 趋势 |
|------|---------------|------------------|------|
| total_loss | 2.949 | 3.142 | ~3.0 震荡 |
| fape_loss | 9.543 | 9.677 | 无明显改善 |
| torsion_loss | 3.569 | 2.109 | ↓ 部分改善 |
| distance_loss | 1.521 | 5.770 | 波动大 |
| lr | 5.0×10⁻⁷ | 2.95×10⁻⁵ | cosine warmup+decay |
| NaN | — | **无** | — |

**训练步数**: 600/600 ✅
**检查点**: `artifacts/fast_abc_stage_c/step_600.pt` (8.3 MB)

---

## 3. 评估结果

### 3.1 Baseline 对比 (Q latent 空间)

**设置**: 100 验证样本, seed=42, 2 个 horizon (32ns, 64ns)

| Horizon | 方法 | MSE | Q Norm | Oracle Norm |
|---------|------|-----|--------|-------------|
| **32ns** | Stage B rollout | **0.469** | **30.05** | 3.47 |
| 32ns | Linear interp (GT) | ≈0 | 3.47 | 3.47 |
| 32ns | Persistence | ≈0 | 3.47 | 3.47 |
| 32ns | Transformer | 0.009 | 3.92 | 3.47 |
| **64ns** | Stage B rollout | **0.460** | **29.73** | 3.47 |
| 64ns | Linear interp (GT) | ≈0 | 3.47 | 3.47 |
| 64ns | Persistence | ≈0 | 3.47 | 3.47 |
| 64ns | Transformer | 0.009 | 3.92 | 3.47 |

**关键发现**: 
- Stage B 在 rollout 过程中 Q norm 爆炸 ~8.6×（3.47 → ~30），远超所有 baseline。
- Transformer (learned temporal baseline) 保持 norm 稳定 (3.92)，但 MSE 也较小 (0.009)。
- **Persistence (q_t = q_0) 的 MSE 和 norm 几乎完美**，暗示 Q 本身的变化量很小，模型引入的 "dynamics" 反而破坏了状态。

### 3.2 Temperature Intervention (温度跳跃)

**设置**: 50 验证样本, 2 seeds (42, 123), 348K → {320K, 379K, 413K} 跳跃, equil=10 steps, jump=50 steps

| 跳跃方向 | Q_diff_ratio | jump_std | equil_std |
|----------|-------------|----------|-----------|
| 348K → 413K | 0.162 | 5.61 | 8.59 |
| 348K → 379K | 0.161 | 5.57 | 8.58 |
| 348K → 320K | 0.162 | 5.54 | 8.57 |

**关键发现**:
- **三个跳跃温度产生几乎完全相同的 Q_diff_ratio (~0.16)** — 模型对温度不敏感。
- jump_std 和 equil_std 在三个温度间几乎一致，温度信号未能通过 energy graph 传播到 Q dynamics。
- 这表明 energy graph 的温度缩放机制可能在快速迭代预算下未能有效学习。

### 3.3 Sidechain Quality (侧链质量)

**设置**: 100 验证样本, decoder 从 Stage C checkpoint 加载

| 指标 | 均值 | 标准差 | 解读 |
|------|------|--------|------|
| chi2_err | 0.012 rad (1°) | 2.2×10⁻⁵ | ✅ 极好 |
| chi1_err | 0.414 rad (24°) | 5.1×10⁻⁵ | ⚠️ 中等 |
| phi_err | 1.24 rad (71°) | 0.020 | ❌ 很差 |
| psi_err | 1.82 rad (104°) | 0.020 | ❌ 极差 |
| clash_count | 2,119,841 | 97 | ❌ 严重 clash |

**关键发现**:
- Chi2 角度误差仅 1° — 表明 decoder 能从 backbone 局部几何推断 χ₂。
- Phi/Psi 主链扭转角误差 71°/104° — **decoder 基本未能从 Q 中提取有意义的构象信息**，仍在依赖 anchor coords 做 refinement。
- 2M+ 的 clash 计数表明 atom-level 结构质量极差。
- 这与之前的 P0.3 消融研究一致：Q 对 decoder 输出的贡献 <0.001Å RMSD。

---

## 4. 基础设施 Bug 与修复

在训练过程中发现并修复了以下问题：

| # | Bug | 症状 | 修复 |
|---|-----|------|------|
| 1 | **Stage B OOM** | max_residues=256 + rollout_steps=32 → 79GB VRAM, CUDA OOM | 降低 data.max_residues=100, rollout_steps=4, num_frames=4 |
| 2 | **模型 shape mismatch** | model.max_residues=128 vs Stage A checkpoint 256 → RuntimeError | 保持 model.max_residues=256, 仅改 data.max_residues=100 |
| 3 | **num_workers 不稳定** | batch_size=1 + num_workers=2 → 加载错误 | 改为 0 |
| 4 | **eval_world_model.py** | `tuple(None)` → TypeError | 改为 `tuple(args.horizons_ps) if args.horizons_ps is not None else None` |
| 5 | **eval_rollout_quality.py** | 同上 `tuple(None)` 错误 | 同上修复 |
| 6 | **Stage C 目录缺失** | auto-launcher 脚本未 mkdir Stage C 输出目录 | mkdir -p |
| 7 | **auto_finish_fast_abc.sh** | 硬编码 stdout.log 路径，Stage C dir 未创建时失败 | 修复后手动启动 |

---

## 5. 完整代码路径清单

### 5.1 配置文件

| 文件 | 大小 | 用途 |
|------|------|------|
| `configs/fast_abc_stage_a_from_scratch.yaml` | 1415 B | Stage A: 20 epochs, lr=3e-4, batch=16, max_res=256 |
| `configs/fast_abc_stage_b_from_scratch.yaml` | 1342 B | Stage B: 3 epochs, lr=3e-5, batch=1, rollout_steps=4, max_res=100 |
| `configs/fast_abc_stage_c_from_scratch.yaml` | 1141 B | Stage C: 2 epochs, lr=5e-5, batch=2, anchor_noise=0.5, q_weight_ramp=300 |

### 5.2 训练入口

| 文件 | 大小 | 用途 |
|------|------|------|
| `protein_jepa_frame/cli.py` | 5.7 KB | CLI 入口: `--stage A/B/C` 路由 |
| `protein_jepa_frame/training/stage_a.py` | 15.4 KB | Stage A: masked JEPA + paired delta loss |
| `protein_jepa_frame/training/stage_b.py` | 12.2 KB | Stage B: LaSP world model, rollout/semigroup losses, AMP+grad_accum |
| `protein_jepa_frame/training/stage_c.py` | 17.4 KB | Stage C: IPA decoder, anchor noise, Q-weight ramp |
| `protein_jepa_frame/config.py` | 34.4 KB | 统一配置模型 (TrainConfig, ModelConfig, etc.) |

### 5.3 核心模型

| 文件 | 大小 | 用途 |
|------|------|------|
| `protein_jepa_frame/models/jepa.py` | 125 KB | 基础 JEPA 模型 (encoder, predictor, temporal encoder) |
| `protein_jepa_frame/models/stage_a/model.py` | 27 KB | StageAJEPA: 专用模型封装 |
| `protein_jepa_frame/models/stage_a/objectives.py` | 5.5 KB | Stage A 损失函数 |
| `protein_jepa_frame/models/stage_b/model.py` | 28 KB | StageBLASP: H→Q projection, energy graph, transition, step/rollout |
| `protein_jepa_frame/models/stage_b/objectives.py` | 9.7 KB | rollout_mse_loss, semigroup_consistency_loss, stage_b_total_loss |
| `protein_jepa_frame/models/stage_b/transition.py` | 7.3 KB | ClosedLoopTransition: force + noise + learned residual |
| `protein_jepa_frame/models/stage_b/state.py` | 1.8 KB | LatentState dataclass |
| `protein_jepa_frame/models/stage_c/model.py` | 13 KB | StageCModel: wraps StructureDecoder |
| `protein_jepa_frame/models/stage_c/structure_decoder.py` | 29 KB | IPA-based 3D structure decoder |
| `protein_jepa_frame/models/stage_c/objectives.py` | 8.4 KB | FAPE, torsion, distance, bond, clash losses |
| `protein_jepa_frame/models/lasp/` | 多个文件 | 底层 LaSP 实现 (energy, bath, diffusion, integrator) |
| `protein_jepa_frame/models/lspwm.py` | 3.1 KB | Latent State Projected World Model 接口 |

### 5.4 数据流水线

| 文件 | 大小 | 用途 |
|------|------|------|
| `protein_jepa_frame/data/dataset.py` | 27 KB | PairedTrajectoryDataset, build_dataset |
| `protein_jepa_frame/data/collate.py` | 5.4 KB | collate_paired_windows, collate_clips |
| `protein_jepa_frame/data/schema.py` | 19.6 KB | Atom37 坐标 schema |
| `protein_jepa_frame/data/manifest.py` | 2.3 KB | 数据清单加载 |

### 5.5 评估脚本

| 文件 | 大小 | 用途 |
|------|------|------|
| `scripts/eval_world_model.py` | 21 KB | 5-mode 世界模型评估 (oracle/direct/autonomous/persistence/q_shuffle) |
| `scripts/eval_rollout_quality.py` | 12.5 KB | Rollout Q MSE, norm drift, ACF, MMD |
| `scripts/eval_baselines.py` | 17 KB | 3 baseline: Linear, Persistence, Transformer |
| `scripts/eval_temperature_intervention.py` | 12.7 KB | 温度跳跃协议 |
| `scripts/eval_sidechain_quality.py` | 12.4 KB | 扭转角误差, clash 计数, 残基 RMSD |
| `scripts/eval_ablations.py` | 14 KB | Q/H 消融 (q_zero, h_zero, etc.) |

### 5.6 自动化脚本

| 文件 | 大小 | 用途 |
|------|------|------|
| `scripts/run_fast_abc_from_scratch.sh` | 487 B | 顺序启动 Stage A→B→C |
| `scripts/run_fast_abc_evaluation.sh` | 1.9 KB | 全部 5 项评估 |
| `scripts/auto_finish_fast_abc.sh` | 3.3 KB | 后台监控: 等 Stage B→启动 Stage C→评估 |
| `scripts/auto_eval_after_stage_c.sh` | 2.6 KB | 等 Stage C 完成→评估 |
| `scripts/retry_failed_evals.sh` | 1.8 KB | 补跑失败的评估 |

### 5.7 训练产物

| 目录 | 内容 |
|------|------|
| `artifacts/fast_abc_stage_a/` | 20 检查点 (37 MB each), train.jsonl (120 entries), config.json |
| `artifacts/fast_abc_stage_b/` | 3 检查点 (7.9 MB each), train.jsonl (90 entries), config.json, stdout.log |
| `artifacts/fast_abc_stage_c/` | 2 检查点 (8.3 MB each), train.jsonl (60 entries), config.json, stdout.log |
| `artifacts/fast_abc_evaluation/` | 评估日志 + 结果 JSON |

---

## 6. 问题诊断

### 6.1 Q Dynamics 质量不足（核心问题）

以下观察**不能仅用训练预算不足来解释**：

1. **Q norm 在 rollout 中爆炸 ~8.6×**: 即使只有 4 步 rollout 训练，transition NLL 从 22→19 正在收敛，但 rollout 期间 Q 迅速偏离 oracle manifold。这表明 closed-loop transition 的力场 (energy graph) 在远离初始化点的区域没有有效的 confining potential。

2. **温度不敏感**: 三个不同温度跳跃产生相同的 Q_diff_ratio (0.16)，jump_std 和 equil_std 也几乎相同。energy graph 的 temperature_k 输入可能未有效调制势能曲面。

3. **decoder 忽略 Q**: anchor noise=0.5Å 在 Stage C 训练中使用，但 decoder 仍产生 71°/104° 的 Phi/Psi 误差和严重 clash。`q_weight_ramp_steps=300` 可能不够，或 Q 的信息结构与 decoder 的架构不兼容。

### 6.2 需要排查的方向

1. **Energy graph 架构**: 
   - 检查能量图在 rollout 路径上是否产生合理的保守力
   - 验证 FDT 一致性：diffusion scale 与 temperature 的缩放是否正确
   - 检查 force magnitude 和 direction 与 oracle Q 变化量的对齐程度

2. **Q 空间的意义**:
   - Q 投影头是否仅学到 trivial 映射？(persistence 的完美表现暗示了这一点)
   - 验证 H→Q 投影在训练前后是否有显著变化
   - 检查 Q 各维度的方差分布

3. **Transition 实现**:
   - sub-stepping warning (dt=4000ps, capped at 10 substeps) 可能引入大积分误差
   - force 计算中的 `torch.autograd.grad` 是否正确处理了 batched per-residue 情况
   - bath 状态是否被正确传播

4. **Decoder 架构**:
   - Q 的 8 维是否能有效注入 IPA 模块
   - anchor noise 的幅度 (0.5Å) 是否足够大
   - Q-weight ramp 的等差数列是否需要更慢的 schedule

---

## 7. 附录: 评估结果汇总

### 7.1 Training Metrics Summary

| Stage | Steps | Initial Loss | Final Loss | NaN | Checkpoints |
|-------|-------|-------------|------------|-----|-------------|
| A | 6000 | 0.161 (total) | 17.19 (mask) | None | 20 |
| B | 900 | 22.15 (NLL) | 18.87 (NLL) | None | 3 |
| C | 600 | 2.95 (total) | 3.14 (total) | None | 2 |

### 7.2 Baseline Comparison Summary

```
        32ns MSE     32ns Norm    64ns MSE     64ns Norm
StageB    0.469         30.05        0.460         29.73
Linear    ≈0            3.47         ≈0            3.47
Persist   ≈0            3.47         ≈0            3.47
Transf.   0.009         3.92         0.009         3.92
Oracle    —             3.47         —             3.47
```

### 7.3 Sidechain Quality Summary

```
chi2_err: 0.012 rad (1°)     ← acceptable
chi1_err: 0.414 rad (24°)    ← borderline
phi_err:  1.241 rad (71°)    ← poor
psi_err:  1.818 rad (104°)   ← very poor
clash:    2,119,841           ← severe
```

---

## 8. 结论

Fast ABC 流水线的**工程基础设施**已完整验证：
- Stage A→B→C 三级训练从随机初始化成功跑通
- 全部 5 项评估脚本可正常运行
- 7 个基础设施 bug 已被修复

但**模型质量**存在根本性问题：
- Q dynamics 在 rollout 中发散
- 温度调节无效
- decoder 忽略 Q

这些问题需要深入排查架构设计和损失函数，而非简单增加训练步数。

---

*报告生成时间: 2026-10-06*
*仓库根目录: /liuyutian/protein_jepa_frame*
