# Stage A 通过报告 — `stage_a_fix4`（多时间尺度时序编码）

**日期:** 2026-10-15
**对应规范:** `STAGE_AB_REDESIGN_PLAN.md` §1.7 Stage A 验收 gate
**结论:** ✅ **Stage A §1.7 gate 5/5 通过 — 可推进 Stage B1a**
**冻结 checkpoint:** `artifacts/abc_v3/stage_a_fix4_frozen/checkpoint_step6450.pt`
（内部 `step` 字段 = 6000，即训练到 launch-log step 6450 时对应的最近 500 步保存点）

> **给后续 agent：** 本文档是 `STAGE_A_ABS_BASELINE_REPORT.md` 的继任。`stage_a_abs` 虽也被标为 §1.7 通过，但独立 probe 证明其 H **只编码 fold 身份、不编码构象变化**（见 §4 对比）。`stage_a_fix4` 修复了这个问题，是进入 B1a 的推荐 encoder。请使用本文档末尾 §6 的命令启动 B1a。

---

## 1. 为什么需要 fix4：abs 的隐性失败

`stage_a_abs` 形式上通过了 §1.7 gate，但那套 gate 的读出器用的是 **随机划分（same-protein leakage）**，会把"H 认得这是哪个蛋白"误判为"H 携带动力学"。用**蛋白级 held-out** 的独立 probe 重测后，真相是：

- abs 的 H 是一个 **fold 身份码**：从 H 预测 held-out 蛋白的成对 CA 距离 R² = **+0.71**（16 ns）。
- abs 的 ΔH **不携带构象变化**：ΔH → ΔCA 的蛋白级 held-out ridge R² ≈ **0**（各 lag 在 −0.06 ~ +0.04）；内部距离变化 `delta_dij` R² 在 16/64/192 ns 为 **−0.18 / +0.08 / −0.07**。

**根因（用户诊断，已验证）：** abs 训练时 `frame_stride=1, num_frames=8`，每个样本只覆盖 **8 ns** 轨迹。mdCATH 有 440 ns 数据，但 encoder 永远看不到 100–400 ns 的构象运动，因此 H 在同一条轨迹内近似时间不变。**这是数据采样问题，不是 loss 问题** —— 没有任何 loss 能教会模型它从未见过的时间尺度。

---

## 2. fix4 的修复：多时间尺度采样（`paired_history_mode=multiscale`）

核心改动是让**每个 context 窗口同时包含细粒度与粗粒度的帧间隔**，而不是增加新 loss：

```
recent = 4 帧 @ 1 ns 间隔   (快动力学)
slow   = 4 帧 @ 32 ns 间隔  (慢动力学)
=> context 偏移 = [0, 32, 64, 96, 97, 98, 99, 100] ns
=> 连续帧 lag   = [32, 32, 32, 1, 1, 1, 1] ns
```

每个样本都同时覆盖 0–100 ns，temporal-contrast loss 因此同时看到 1 ns 和 32 ns 的构象变化对；paired future target 在 101/150/200 ns，预测任务本身就要求 H 编码多尺度动力学。

**从头训练（无 warm-start）：** abs 已被 8 ns 短 lag 塌缩污染，fix4 从随机初始化开始。

### 训练配置（`run_stage_a_fix4.sh`）

| 参数 | 值 |
|:---|:---|
| steps / warmup | 15000 / 500 |
| lr / encoder_lr_scale | 3e-4 / 0.3（cosine decay） |
| batch_size × grad_accum | 4 × 4 |
| latent_dim / factors / depth | 128 / 4 / 4 |
| history_frames | 8 |
| **frame_stride** | **1** |
| **paired_history_mode** | **multiscale** |
| **paired_recent_frames** | **4**（@ 1 ns） |
| **paired_slow_stride** | **32**（@ 32 ns） |
| paired_horizons (frames) | 101, 150, 200 |
| temporal_lags (frames) | 1, 2, 4, 7 |
| **temporal_contrast_weight** | **2.0** |
| **temporal_contrast_rmsd_scale** | **5.0**（适配 32–100 ns、2–6 Å RMSD） |
| temporal_variance_weight / min_std | 0.10 / 0.40 |
| struct_recon_weight / pairs / dim | 0.3 / 512 / 384 |
| delta_target | False |
| EMA start → end / warmup | 0.998 → 0.9999 / 5000 |

**temporal-contrast loss（regression + ranking）：** 对每个连续帧对，把归一化 H-距离回归到 `tanh(CA_RMSD / scale)`，并加三元组 ranking（RMSD 更大的帧对，H-距离必须更大 ≥ margin）。ranking 项防止"把所有 H-距离缩到 0"的平凡解（fix3 的失败模式）。

---

## 3. §1.7 Gate 结果总览（冻结 `stage_a_fix4_frozen/checkpoint_step6450.pt`）

```
=== Stage A §1.7 gate: 5/5 passed ===
>>> GATE PASSED — proceed to Stage B1a <<<
```

| # | Gate 项 | 阈值 | fix4@6450 | abs（对比） | 结果 |
|:---:|:---|:---|:---|:---|:---:|
| 1 | latent persistence skill (lag 1,2, vs time-shuffle) | > 0.0 | **+0.166** | +0.140 | ✅ |
| 2 | H 有效秩 | ≥ 4 | **53.85** | 66.42 | ✅ |
| 2 | 同蛋白同残基时间 std | ≥ 1e-3 | **0.14617** | 0.12825 | ✅ |
| 2 | 死通道 (sd<1e-3) | = 0 | **0/128** | 0/128 | ✅ |
| 2b | 跨蛋白 cos（去均值） | ≤ 0.95 | **0.6017** | 0.4292 | ✅ |
| 3 | 读出器 R² (CA 距离, held-out) | ≥ 0.05 | **+0.4472** | +0.4567 | ✅ |
| 4 | full history 不差于对照 | ≥ −0.05 | **−0.039** | +0.0001 | ✅ |

§1.7.1 细目（vs time-shuffle null）：lag1 = +0.237，lag2 = +0.094，lag4 = −0.048；真实坐标天花板 lag1/2/4 = +0.37 / +0.21 / −0.06。fix4 的 lag1 skill 高于 abs（+0.237 vs +0.188）。

---

## 4. 关键证据：fix4 修复了构象变化编码（蛋白级 held-out）

这是 fix4 相对 abs 的**实质进步**，也是 B1a oracle 解码器真正需要的信号。
probe: `tmp_claude/probe_h_internal_dist.py`（60 蛋白，20% 蛋白 held-out，SE(3)-不变的成对 CA 距离目标，小 MLP 读出）。

| lag | 指标 | abs | **fix4@6450** |
|:---:|:---|:---:|:---:|
| 16 ns | `abs_dij`（绝对距离，fold 身份） | +0.71 | **+0.94** |
| 64 ns | `abs_dij` | +0.66 | **+0.89** |
| 192 ns | `abs_dij` | +0.45 | **+0.88** |
| 16 ns | `delta_dij`（距离**变化**，构象动力学） | −0.18 | **+0.23** |
| 64 ns | `delta_dij` | +0.08 | **+0.36** |
| 192 ns | `delta_dij` | −0.07 | **+0.38** |

**解读：**
- `abs_dij` 从 0.45–0.71 升到 0.88–0.94 —— fold 身份编码不但没退化，反而更强（多尺度数据提供了更干净的几何监督）。
- `delta_dij` 从 **≈0 / 负** 升到 **+0.23 ~ +0.38** —— 这是从无到有：fix4 的 ΔH 现在能在 **未见过的蛋白**上线性解出构象变化，且在长 lag（64/192 ns）效果更强，正是多时间尺度采样的目标。

> 注：该进步在 §1.7 官方 gate 脚本里看不出来（它的 §1.7.3 读出器 shuffled≡full，说明只用了末帧 H、未用历史顺序）。fix4 的真实增益体现在上面这个**蛋白级 held-out 的构象-变化** probe 上。

---

## 5. 当前训练状态与产物

- **训练仍在继续**（PID 2900679，launch-log step ≈ 7150 / 15000，35 s/step）。gate 用的是 step 6450 的冻结快照。lr 为 cosine decay，后 1/3 训练（step 10000–15000）预计 `delta_dij` 还会继续上升。若需更强 encoder，可等训练跑完后用 `checkpoint_latest.pt` 重跑 §3/§4 评估再决定是否替换冻结点。
- **冻结快照：** `artifacts/abc_v3/stage_a_fix4_frozen/checkpoint_step6450.pt`（本报告所有数字的来源）。
- **后台监控：** `tmp_claude/monitor_fix4.sh`（PID 2952221）会在 step 8000/12000/15000 自动跑 `delta_dij` probe，结果写入 `tmp_claude/monitor_fix4_probes.log`。
- **encoder 规范（写入 B1a 时必须一致）：** `StageAV2Config` latent_dim=128, encoder_depth=4, num_heads=4, num_rbf=16, orientation_features=True, num_factors=4, delta_target=False。

---

## 6. 给后续 agent：如何开始 Stage B1a

按 §2.0，B1a 分两小步：先 same-frame oracle 冒烟（确认 decoder 可用），再冻结/受控微调正式校准。**把所有 B1a 脚本的 `--stage-a-ckpt` 指向 fix4 冻结点。**

### 6.1 第一步 — same-frame oracle 冒烟（确认 decoder 架构可用）

目标：anchor = target 帧时 `oracle_ca_mse` 应在 ~500 步内降到 < 5 Å²。否则是 decoder 架构问题，与 H 质量无关。

```bash
cd /liuyutian/protein_jepa_frame
CUDA_VISIBLE_DEVICES=1 python -u -m protein_jepa_frame.training.train_stage_b1a \
  --manifest artifacts/mdcath_manifest_200dom/train.jsonl \
  --stage-a-ckpt artifacts/abc_v3/stage_a_fix4_frozen/checkpoint_step6450.pt \
  --output artifacts/abc_v3/stage_b1a_fix4_sameframe \
  --steps 1000 --warmup-steps 100 --batch-size 4 --num-workers 4 \
  --history-frames 4 --unfreeze-top-n-blocks 0 \
  --lr 3e-4 --weight-decay 0.01 --grad-clip 1.0 \
  --fape-weight 1.0 --ca-mse-weight 1.0 --torsion-weight 0.2 --distance-weight 0.1 \
  --phys-future-weight 0.0 --anchor-weight 0.0 \
  --same-frame-oracle --fit-normalizer-batches 20 \
  --log-every 25 --save-every 250
```

（模板见 `run_b1a_sameframe.sh`，但其 `--stage-a-ckpt` 仍指向 abs，需改。）

### 6.2 第二步 — 冻结基线正式校准（§2.0 第 1 步）

`oracle_ca_mse < 5 Å²` 后，冻结 encoder 训练 oracle decoder + 物理未来 probe：

```bash
CUDA_VISIBLE_DEVICES=1 python -u -m protein_jepa_frame.training.train_stage_b1a \
  --manifest artifacts/mdcath_manifest_200dom/train.jsonl \
  --stage-a-ckpt artifacts/abc_v3/stage_a_fix4_frozen/checkpoint_step6450.pt \
  --output artifacts/abc_v3/stage_b1a_fix4_frozen \
  --steps 5000 --warmup-steps 200 --batch-size 4 --num-workers 4 \
  --history-frames 4 --unfreeze-top-n-blocks 0 \
  --lr 3e-4 --weight-decay 0.01 --grad-clip 1.0 \
  --fape-weight 1.0 --ca-mse-weight 10.0 --torsion-weight 0.2 --distance-weight 0.1 \
  --phys-future-weight 0.5 --anchor-weight 0.0 \
  --fit-normalizer-batches 20 --log-every 25 --save-every 500
```

### 6.3 第三步（可选）— 受控微调对照（§2.0 第 2 步）

从 same-frame checkpoint 续训，解冻顶部 1–2 个 block，encoder lr ≪ decoder lr，加 anchor 蒸馏：

```bash
  --unfreeze-top-n-blocks 2 --encoder-lr-scale 0.1 --lr 1e-4 \
  --ca-mse-weight 1.0 --phys-future-weight 0.5 --anchor-weight 0.1 \
  --resume artifacts/abc_v3/stage_b1a_fix4_sameframe/checkpoint_latest.pt
```

**选择规则（§2.0 第 4 步）：** 在相同数据和读出器预算下比较冻结 vs 微调，以固定物理目标上的 oracle 重构、动态构象区分、多 lag 预测选定；微调无收益则用冻结版本。

### 6.4 B1a → B1b 交接前必须通过的检查（§2.0 第 5 步 / §3.5）

1. **Oracle 重构 gate（§3.5-1）：** 给定真实 H_t，backbone RMSD **< 1.5 Å**（中位数，held-out 帧），非局部 CA 距离误差 < 1 Å（90th pct）。**这是硬 gate，用纯 `D_χ(H_t, c)` 评分。**
2. **构象替换检查（§3.5-2）：** 同蛋白不同时刻的 H 替换后，解码结构相应改变（证明 decoder 用了 H 的构象信息而非记忆平均结构）。fix4 的 `delta_dij` R² > 0 是通过这项的前提条件。
3. 选定 encoder 后，拟合并**冻结** train-only H 的逐通道中心/尺度（有下限），重生成 H 缓存，固定 decoder/probe，记录 encoder/normalization/schema hash。

> ⚠️ §3.5-1 的 1.5 Å 是 **B2 oracle gate**，在 B1a 校准完成、decoder 训练后检查，**不是 Stage A gate**。Stage A 已按 §1.7 通过。若 B1a 冻结 decoder 无法达到 1.5 Å，再考虑 §2.0 的受控微调。

---

## 7. 复现命令

```bash
# §1.7 官方 gate（5/5）
GATE_CKPT=artifacts/abc_v3/stage_a_fix4_frozen/checkpoint_step6450.pt \
CUDA_VISIBLE_DEVICES=1 python -u tmp_claude/eval_stage_a_gate_full.py

# 蛋白级 held-out 构象-变化 probe（§4 的核心证据）
CUDA_VISIBLE_DEVICES=1 python tmp_claude/probe_h_internal_dist.py \
  artifacts/abc_v3/stage_a_fix4_frozen/checkpoint_step6450.pt \
  --label fix4_s6450 --n-proteins 60 --steps 1500

# ΔH→ΔCA 蛋白级 ridge（确认 abs 的 +0.33 是 leakage）
CUDA_VISIBLE_DEVICES=1 python tmp_claude/probe_dh_dca_holdout.py \
  artifacts/abc_v3/stage_a_fix4_frozen/checkpoint_step6450.pt --label fix4 --n-proteins 60
```

## 8. 新增 / 修改的文件

- `run_stage_a_fix4.sh` — 多时间尺度训练脚本
- `protein_jepa_frame/data/dataset.py::_paired_context_offsets` — multiscale 偏移构造
- `protein_jepa_frame/training/train_stage_a_v2.py` — 新增 `--paired-history-mode/--paired-recent-frames/--paired-slow-stride/--frame-stride` CLI
- `protein_jepa_frame/models/stage_a_v2/objectives.py::compute_temporal_contrast_loss` — regression + ranking 混合
- `tmp_claude/probe_h_internal_dist.py` — 蛋白级 held-out 内部距离 / 距离变化 probe（§4）
- `tmp_claude/probe_dh_dca_holdout.py` — 蛋白级 ΔH→ΔCA ridge probe
- `tmp_claude/eval_stage_a_gate_full.py` — 新增 `GATE_CKPT` 环境变量支持
- `tmp_claude/monitor_fix4.sh` — 训练里程碑自动评估
