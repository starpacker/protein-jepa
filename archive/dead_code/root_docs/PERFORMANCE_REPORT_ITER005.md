# Protein-JEPA-Frame 性能报告与 Gate 未通过原因分析

**模型**：iter_005，`artifacts/iter_005_basin_decoder/step_400.pt`，`decoder_basin_only_ca=True`
**Stage-B**：`artifacts/full_abc_stage_b_v3_final/checkpoint_latest.pt`（external，frozen）
**评估协议**：canonical eval（`scripts/eval_world_model.py`），held-out `val` split（protein-disjoint），5 种 mode：oracle / direct / autonomous / persistence / qshuffle
**Skill 定义**（严格 V2 gate 口径）：

$$
\text{RMSD\_skill} = 1 - \frac{\text{RMSD}_{model}}{\text{RMSD}_{p}} \qquad
\text{Contact\_skill} = \frac{\text{CJ}_{model} - \text{CJ}_{p}}{1 - \text{CJ}_{p}}
$$

**Gate 通过条件**：direct **与** autonomous 必须在 horizon 32 / 64 / 96 ns 全部同时满足
`RMSD_skill ≥ 0.10 且 ΔRMSD ≥ 0.15 Å`、`ΔCJ ≥ 0.03 且 Contact_skill ≥ 0.10`（Weak / Transient 同理），并额外通过若干统计与 Q-dependence gate。

---

## 0. 评估协议说明

| 项目 | 值 |
|---|---|
| 帧间隔 | mdCATH `frame_interval_ps = 1000` → **1 raw frame = 1 ns** |
| 长 horizon 上下文 | `num_frames=8, frame_stride=4` → context 跨度 28 raw frames |
| 短 horizon 上下文 | `num_frames=4, frame_stride=1` → context 跨度 4 raw frames（保证与 4/8/16 ns target **不重叠**） |
| 长 horizon 样本 | 60 个 held-out val 样本，seed 42 |
| 短 horizon 样本 | 180 个 held-out val 样本，seed 42 |

> **短 horizon 的重要说明**：标准 eval 的 8 帧 / stride-4 上下文跨度（28 ns）与任何 < 32 ns 的 horizon 都会重叠，违反 context/target 不共享帧的协议。因此短 horizon 评估改用 **4 帧 / stride-1** 的上下文（约 4 ns 的历史），目标 horizon（4/8/16 ns）与之严格不重叠。这意味着短 horizon 的上下文比长 horizon 更短——这一点在解读数字时必须记住（见 §3 的对照实验）。

---

## 1. 全 horizon 汇总表（direct mode）

`Δ` 的定义：RMSD 用 `persist − model`（正值=更好），contact 类用 `model − persist`（正值=更好）。加粗=通过该 horizon 对应 gate。

| Horizon | RMSD model | RMSD pers | ΔRMSD (Å) | RMSD_skill | ΔContactJ | Contact_skill | ΔWeakJ | Weak_skill | ΔTransJ | Trans_skill |
|---|---|---|---|---|---|---|---|---|---|---|
| **4 ns**  | 3.230 | 3.471 | **+0.241** | +0.069 | +0.027 | +0.082 | +0.021 | +0.083 | +0.017 | +0.074 |
| **8 ns**  | 3.772 | 4.066 | **+0.293** | +0.072 | +0.018 | +0.052 | +0.017 | +0.062 | +0.015 | +0.058 |
| **16 ns** | 5.452 | 5.896 | **+0.444** | +0.075 | +0.025 | +0.057 | +0.019 | +0.055 | +0.019 | +0.057 |
| **32 ns** | 4.884 | 5.485 | **+0.601** | **+0.110** | **+0.032** | +0.074 | **+0.029** | +0.084 | **+0.027** | +0.083 |
| **64 ns** | 5.752 | 6.305 | **+0.553** | +0.088 | −0.002 | −0.004 | −0.007 | −0.018 | +0.006 | +0.015 |
| **96 ns** | 5.803 | 6.327 | **+0.524** | +0.083 | +0.013 | +0.028 | +0.006 | +0.015 | +0.007 | +0.020 |

**Gate 判定**（RMSD_skill ≥ 0.10；各类 Δ ≥ 0.03 且 skill ≥ 0.10）：

- RMSD_skill：**仅 32 ns 通过**（+0.110）；4/8/16 ns 在 +0.07 附近，64/96 ns 在 +0.08~0.09，全部不通过。
- ΔRMSD ≥ 0.15 Å：**6 个 horizon 全部通过**。
- Contact / Weak / Transient skill：**6 个 horizon 全部不通过**（最高 +0.084）。
- ΔContact 类 ≥ 0.03：仅 32 ns 的 ContactJ（+0.032）勉强越过，Weak/Transient 最高的 32 ns 也仅为 +0.029 / +0.027。

autonomous mode 与 direct 在本报告中数值一致到小数点后 4 位（见 §5 的原因），故不重复列表。

---

## 2. 逐 horizon 明细

### 2.1 短 horizon（4 / 8 / 16 ns，180 samples，ctx=4 frames stride1，不重叠）

| Horizon | Metric | model | persistence | Δ | skill | n |
|---|---|---|---|---|---|---|
| 4 ns | RMSD (Å) | 3.230 | 3.471 | +0.241 | +0.069 | 70 |
| 4 ns | ContactJ | 0.7013 | 0.6747 | +0.027 | +0.082 | 70 |
| 4 ns | WeakJ | 0.7700 | 0.7493 | +0.021 | +0.083 | 70 |
| 4 ns | TransientJ | 0.7822 | 0.7649 | +0.017 | +0.074 | 70 |
| 8 ns | RMSD (Å) | 3.772 | 4.066 | +0.293 | +0.072 | 57 |
| 8 ns | ContactJ | 0.6678 | 0.6495 | +0.018 | +0.052 | 57 |
| 8 ns | WeakJ | 0.7499 | 0.7332 | +0.017 | +0.062 | 57 |
| 8 ns | TransientJ | 0.7518 | 0.7366 | +0.015 | +0.058 | 57 |
| 16 ns | RMSD (Å) | 5.452 | 5.896 | +0.444 | +0.075 | 53 |
| 16 ns | ContactJ | 0.5857 | 0.5606 | +0.025 | +0.057 | 53 |
| 16 ns | WeakJ | 0.6692 | 0.6498 | +0.019 | +0.055 | 53 |
| 16 ns | TransientJ | 0.6867 | 0.6678 | +0.019 | +0.057 | 53 |

### 2.2 长 horizon（32 / 64 / 96 ns，60 samples，ctx=8 frames stride4）

| Horizon | Metric | model | persistence | Δ | skill | n |
|---|---|---|---|---|---|---|
| 32 ns | RMSD (Å) | 4.884 | 5.485 | +0.601 | **+0.110** | 21 |
| 32 ns | ContactJ | 0.6008 | 0.5687 | +0.032 | +0.074 | 21 |
| 32 ns | WeakJ | 0.6834 | 0.6542 | +0.029 | +0.084 | 21 |
| 32 ns | TransientJ | 0.6989 | 0.6716 | +0.027 | +0.083 | 21 |
| 64 ns | RMSD (Å) | 5.752 | 6.305 | +0.553 | +0.088 | 19 |
| 64 ns | ContactJ | 0.5055 | 0.5073 | −0.002 | −0.004 | 19 |
| 64 ns | WeakJ | 0.6095 | 0.6165 | −0.007 | −0.018 | 19 |
| 64 ns | TransientJ | 0.6356 | 0.6300 | +0.006 | +0.015 | 19 |
| 96 ns | RMSD (Å) | 5.803 | 6.327 | +0.524 | +0.083 | 20 |
| 96 ns | ContactJ | 0.5453 | 0.5320 | +0.013 | +0.028 | 20 |
| 96 ns | WeakJ | 0.6301 | 0.6243 | +0.006 | +0.015 | 20 |
| 96 ns | TransientJ | 0.6498 | 0.6427 | +0.007 | +0.020 | 20 |

> 每 horizon 的 n 略有不同，因为不同目标 horizon 对应的窗口在同一 val manifest 上的可用样本数不同（分层抽样后按 horizon 拆分）。

### 2.3 Q-dependence 门槛

| 比较 | oracle / direct / autonomous / qshuffle |
|---|---|
| RMSD | 完全相同（如 32 ns 全部 5.465 Å） |
| ContactJ | 完全相同（0.5521） |
| **oracle − qshuffle RMSD 差** | **0.0000 Å**（gate 要求 ≥ 0.10 Å） |

四种 mode 的所有 metric 一致到小数点后 4 位，说明当前 decoder 输出与 Q 无关。**Q-dependence gate ❌。**

### 2.4 统计稳定性 gate

| 指标 | 当前值 | 门槛 | 状态 |
|---|---|---|---|
| RMSD improvement 95% CI 下界 | +0.409 Å | > 0 | ✅ |
| ContactJ improvement 95% CI 下界 | +0.0007 | > 0 | ✅（勉强） |
| protein-level Contact win rate | 16/20 = 80% | ≥ 60% | ✅ |
| protein-level RMSD win rate | 19/20 = 95% | ≥ 60% | ✅ |
| failed samples | 0 | = 0 | ✅ |

### 2.5 总体通过情况
16 条 V2 STRICTER gate **未全部满足**。统计类全过；RMSD Δ 全过、RMSD_skill 仅 32 ns 过；Contact / Weak / Transient skill **0/6 通过**；Q-dependence 完全不通过。

**状态：当前仍未达标，继续迭代。**

---

## 3. 短 horizon 的对照实验（上下文长度 vs horizon 长度）

为区分"horizon 变短"与"上下文变短"各自的影响，补充了第三个配置：**8 帧 / stride-4 上下文（overlapping）测 4/8/16 ns**，90 samples。

| Horizon | 配置 | RMSD_skill | Contact_skill | Weak_skill | Trans_skill |
|---|---|---|---|---|---|
| 4 ns | ctx4·s1（不重叠，协议合法） | +0.069 | +0.082 | +0.083 | +0.074 |
| 4 ns | ctx8·s4（重叠，仅对照） | **−0.139** | −0.047 | −0.064 | −0.077 |
| 8 ns | ctx4·s1 | +0.072 | +0.052 | +0.062 | +0.058 |
| 8 ns | ctx8·s4 | +0.108 | −0.013 | +0.004 | +0.026 |
| 16 ns | ctx4·s1 | +0.075 | +0.057 | +0.055 | +0.057 |
| 16 ns | ctx8·s4 | +0.090 | +0.036 | +0.045 | +0.046 |

**解读**：在极短 horizon（4 ns），用 28 ns 跨度的大上下文反而严重受损（RMSD_skill −0.139）——锚点距 target 太"远"的上下文平均位移（basin prior）在这种尺度上变成了负资产。用短上下文（4 帧）则恢复为正值。这个对照说明**短 horizon 数值受上下文配置显著影响**，报告中的短 horizon 数字应标注其上下文口径。

---

## 4. 为什么通不过——根因分析

以下结论均有 held-out val 上的 canonical 探针数据支撑（探针位于 `tmp_claude/` 与 `artifacts/iter_006_qca/analysis/`）。

### 4.1 是 Stage B 的 Q 没信息，还是 Stage C 把它压掉了？

**两者都部分成立，但 binding 原因是更深的第三点：Q 不含 per-residue 位移信息。**

**(a) Stage B 的 Q 没有坍缩**（`probe_q_perresidue.py` / `probe_qoracle_canon.py`，canonical encode_state 路径）

| 量 | 值 | 含义 |
|---|---|---|
| cross-residue spread / \|q\| | 0.30 | 非全局常数 |
| PC1 fraction | 0.44 | 56% 方差是真实 per-residue 内容 |
| corr(Δ‖q‖, internal conf. RMSD) | **0.797** | Q 变化幅度强跟踪全局构象变化 |
| oracle-Q 与 context-Q 余弦 | 0.9998 | oracle 不提供额外结构信息 |
| **corr(\|q_i\|, \|disp_i\|)** | **0.024 ≈ 0** | **Q 的逐残基内容与逐残基位移无关** |

**(b) Stage C 训练后忽略了 Q**（`probe_decoder_qpath.py`，v5b decoder）

| 分量 | 幅度 |
|---|---|
| 总 CA 平移更新 | 1.54 Å |
| frame_update（H/IPA 驱动，Q 无关） | 1.51 Å |
| q_translation_update（Q→位移路径） | 0.071 Å |
| oracle-Q 替换 context-Q 后 CA 位移变化 | **0.0001 Å** |

训练动态导致 H/IPA 路径主导，Q 路径权重趋零；这也是 §2.3 中 oracle = direct = qshuffle 的微观解释。

**(c) 即使修好 Stage C，Q 也无法过关**（`probe_qdisp_ceiling.py`）
用 oracle-Q 训练 MLP 预测 per-residue 位移，val 结果比纯 basin 更差（contact skill ≈ −0.90，灾难性过拟合）。说明 oracle-Q 里没有可解码的 per-residue 位移方向。

### 4.2 合法 context-only 预测器的天花板（`probe_ceiling_final.py`，250 val samples）

| 预测器 | hz4 cj_sk | hz8 | hz16 | hz32 | hz64 | hz96 |
|---|---|---|---|---|---|---|
| basin (β=1) | +0.076 | +0.050 | +0.060 | +0.039 | +0.026 | +0.046 |
| basin (β=0.8) | +0.075 | +0.051 | +0.059 | +0.041 | +0.030 | +0.045 |
| consensus_dist（无坐标） | +0.111 | +0.079 | +0.075 | +0.092 | +0.079 | +0.083 |
| **Gate** | **0.10** | **0.10** | **0.10** | **0.10** | **0.10** | **0.10** |

| 预测器 | hz4 rmsd_sk | hz8 | hz16 | hz32 | hz64 | hz96 |
|---|---|---|---|---|---|---|
| basin (β=1) | +0.078 | +0.059 | +0.075 | +0.105 | +0.088 | +0.088 |
| basin (β=0.8) | +0.082 | +0.065 | +0.074 | +0.113 | +0.093 | +0.093 |

- 最佳坐标预测器 basin：contact skill 最高 +0.076（远低于 0.10），RMSD skill 仅在 hz32 越过 0.10。
- 最佳 contact 预测器 consensus 距离矩阵：+0.075~+0.111（hz4 刚好越线），**不输出坐标**，无法同时满足 RMSD gate。

### 4.3 Contact 与 RMSD gate 在坐标域内不可同时达到（`probe_hybrid_stress.py`，200 val samples）

从 basin 坐标出发梯度下降匹配 consensus 距离矩阵：

| steps | hz32 cj_sk | hz32 rmsd_sk |
|---|---|---|
| 0 | +0.039 | +0.106 |
| 30 | +0.055 | −0.007 |
| 60 | +0.060 | −0.007 |

contact skill 上升时 RMSD skill 转负；两者单调 trade-off，无同时满足两个 0.10 gate 的区域。

### 4.4 Basin 窗口变体无法突破 RMSD gate（`probe_recency_basin.py`，200 val samples）
uniform 均匀平均已是最优 RMSD 配置；recency 加权反而降低 RMSD skill（hz64 从 +0.087 降到 +0.063~+0.079），无改善空间。

---

## 5. 根本瓶颈总结

| 瓶颈 | 类型 | 说明 |
|---|---|---|
| per-residue 位移不可预测 | **信息论瓶颈** | 当前 context 编码不含足够的逐残基未来位移方向信息；最佳合法预测器 contact skill 停在 +0.09 以下，低于 gate 0.10 |
| Stage C 不使用 Q | 架构/训练问题 | Q→位移路径仅 0.071 Å vs 总量 1.54 Å；oracle 替换零效果 |
| Q 缺乏 per-residue 结构编码 | Stage B 问题 | Q 跟踪全局构象幅度（0.797），但逐残基位移方向 corr 仅 0.024 |
| Contact/RMSD 目标互斥 | 结构性矛盾 | 推向 contact-optimal 会破坏 RMSD，无单一坐标预测器同时满足两 gate |
| 短 horizon 上下文敏感 | 协议因素 | 4 ns 尺度下大上下文反而有害（§3） |

---

## 6. 下一步方向

要突破当前天花板，必须改变**信息来源**，而非继续在现有 Stage C 上调参：

1. **Stage B 加 per-residue 结构自编码约束**：增加辅助损失，强制 `Q_i` 能重建第 i 个残基从 anchor 到 future 的位移方向，把 Q 从"全局构象状态码"升级为"逐残基局部位移码"。
2. **Stage A 表征改进**：当前冻结的 H 不含逐残基位移方向。给 Stage A 增加未来帧预测目标或更长序列上下文，可能使 H 携带更多可预测局部动力学信息。
3. **先探针后全量训练**（纪律）：在重训 Stage B 前，先小规模验证 aux loss 后 `‖q_fut − q_ctx‖` 增大且 `R²(disp from Q)` 显著提升，再投入 full 训练。

**当前状态：当前仍未达标，继续迭代。**

---

## 附录：产物路径

| 内容 | 路径 |
|---|---|
| 长 horizon canonical eval | `tmp_claude/eval_iter005_s100/full_report.json` |
| 短 horizon eval（180 samples，ctx4·s1） | `tmp_claude/eval_iter005_short_hz_v2/full_report.json` |
| 短 horizon 对照（90 samples，ctx8·s4） | `tmp_claude/eval_iter005_short_hz_ctx8/full_report.json` |
| 逐 horizon 汇总 | `tmp_claude/short_hz_final.json`、`tmp_claude/combined_hz_table.json` |
| ceiling 探针（全 horizon） | `tmp_claude/probe_ceiling_allhz.log` |
| Q 诊断探针链 | `artifacts/iter_006_qca/analysis/iter_006_diagnosis.md` |
