# 诊断：为什么 Stage A / true-H 读出门槛总是失败

日期：2026-10-18。方法：直接加载真实 Stage A checkpoint（`artifacts/repaired_v2/comparison_20261018/stage_a/checkpoint_best.pt`，latent32 / 2 层 encoder）与真实 mdCATH 数据（`npz/train`，8 个蛋白，320K，1ns/frame），
用一个**独立于项目 decoder 的探针**测量 H 到底携带多少可解码构象信息。所有数字可在 `.tmpwork/probe*.py` 复现。

---

## 0. 结论（一句话）

**问题不在"decoder 拟合的蛋白太少"，也不在"数据不够"。**
核心事实是：**Stage A 的 H 本身就没有可线性/浅层解码的构象信息** —— 一个用真实 H 训练的探针，
在**同一蛋白内**（train/test 按时间切分）预测未来构象，**误差比"什么都不做"（persistence）还大 15–20%**。
跨蛋白时更差（2.2–2.3 倍）。因此任何 decoder、任何蛋白数量、任何验证集都无法把它读出来。
**当前的瓶颈是表示（encoder/Stage A 目标），不是 decoder，也不是数据规模。**

---

## 1. 决定性证据：探针 H → 内部 CA 位移

探针设置：对每个 (anchor t, horizon h)，标签 = 目标帧 CA 云在 anchor 帧内做 Kabsch 对齐后的**内部位移** δ
（与 B1a 的 `target_delta_local` 同定义）。指标 = 解码后结构的 shape RMSD（再次 Kabsch 对齐）。
对照 persistence = δ=0（即直接输出 anchor 构象）。

| horizon | 探针 H（MLP，跨蛋白 held-out） | persistence | 判定 |
|---:|---:|---:|---|
| 1 ns | 2.311 Å | 2.096 Å | **输** |
| 2 ns | 2.671 Å | 2.370 Å | **输** |
| 4 ns | 3.264 Å | 2.747 Å | **输** |
| 8 ns | 3.485 Å | 3.032 Å | **输** |

| horizon | 探针 H（线性 Ridge，**同蛋白**时间切分） | persistence | ratio |
|---:|---:|---:|---:|
| 1 ns | 2.552 Å | 2.132 Å | 1.197 |
| 2 ns | 2.965 Å | 2.491 Å | 1.190 |
| 4 ns | 3.385 Å | 2.844 Å | 1.190 |
| 8 ns | 3.843 Å | 3.356 Å | 1.145 |

| horizon | 探针 H（线性，**跨蛋白** LOO） | persistence | ratio |
|---:|---:|---:|---:|
| 1 ns | 5.190 Å | 2.319 Å | 2.24 |
| 8 ns | 8.222 Å | 3.648 Å | 2.25 |

**关键判读**：
- 连**同蛋白**、用**线性探针**（最容易的读出假设）都打不过 persistence → 不是"decoder 太弱"或"没泛化"，
  而是 H 里**根本没有足够的、关于未来构象的线性可读信号**。
- 跨蛋白时误差翻倍 → 额外一层泛化问题，但那是次要的。
- 探针远好于项目 decoder（探针 2.5 Å vs h_gate true-H 14–20 Å）→ **decoder 确实还要单独修**，
  但**先修 decoder 也不可能让 true-H 通过 gate**，因为信息上限已被探针钉死在 ~persistence 附近。

---

## 2. H 不是"塌缩"了，而是"变了但读不出"

常见误解是"H 塌缩成常数"。实测**不是**：

| 蛋白 | cos(H_t, H_{t+1ns}) | \|ΔH\|/\|H\| | 结构 RMSD |
|---|---:|---:|---:|
| 1a02F00 | 0.840 | 0.46 | 3.93 Å |
| 1avyB00 | 0.838 | 0.48 | 3.43 Å |
| 1b33N01 | 0.883 | 0.40 | 1.65 Å |

H 随时间**明显变化**（|ΔH|/|H| ≈ 0.4–0.6）。所以问题不是"表示不变"，而是
**H 的变化方向与真实构象变化方向不对齐**：它编码的是"几何局部的、身份/拓扑主导的"分量，
缺失了决定长程构象（尤其大尺度内部位移）的信息。

---

## 3. 根因分析（为什么 H 缺构象信息）

结合代码，主要有四条，按影响排序：

### 3.1 encoder 是**逐帧、局部雷达**，没有构象的全局低维坐标
`protein_jepa_frame/models/encoder.py`：输入只有 (a) 每残基 6 个内部距离 + (b) 可选 backbone torsion sin/cos，
再叠 aatype/residue_index/chain 身份 embedding，过 2 层 distance-biased attention。
- 它**没有任何 global pooling / 全局坐标 / 质心 / 回转半径 / 长程接触**这类"整体构象模式"的显式通道；
  蛋白大尺度构象变化 = 非局部的、collective 的坐标变化，而 encoder 的感受野被局部距离主导。
- latent_dim 只有 **32**（且仅 2 层、CPU、400 step）。跑道本身也窄。

### 3.2 Stage A 的目标奖励"锚点标定"，不奖励"预测真实位移"
`train_stage_a_v2.py` + Stage A 模型：temporal 目标写成
`H_future = compose(decompose(H_anchor) + pred_factors)`（见 `_predict_future_h` 注释与 Stage A 模型），
即**预测的是 factor 空间的增量**。配合 `temporal_min_std` floor、contrast、variance 等正则，
最优解很容易是"让 H 对时间有可测量的变化（满足 variance/contrast），但不要求该变化与真实构象位移同向"。
这正是探针观察到的：H 变了，但变化不可解码成结构。

### 3.3 空间 masked query 无法区分残基（报告 §3.7 已指出，未修）
masked residue 的 H 置零后直接进 predictor，predictor 没有残基身份 query → 同一蛋白所有 masked query 相同。
这削弱了 Stage A 学"哪个残基要动多少"的能力。

### 3.4 decoder 侧另有独立的、更严重的失败（报告 §3.5、本次实测）
`h_gate` 的 true-H（14–20 Å）比探针（2.5 Å）差一个数量级，说明 `StructureDecoder`(anchor_free 路径) 本身有问题：
- `_coil_prior` 用固定随机种子生成"自回避线圈"作为 H 的初始基底，`coord_scale=40` 让 H 只做有界修正；
  这相当于把 H 信息**乘了一个极小的有效增益**，H 的梯度信号被先验淹没。
- 训练时只训了 200–2000 step、batch 2、CPU，且 `fape_weight/torsion_weight/relative_disp_weight` 在最新配置里**全为 0**
  （只剩 `ca_mse_weight=1` + `distance_weight=0.1`）。

---

## 4. 关于"用更多蛋白拟合 decoder"的结论

**方向对，但不能解决当前问题，会浪费时间。** 理由：

1. 探针实验说明**同蛋白内**都读不出 → 增加蛋白数量只优化"泛化"，而这里连**训练分布内**都失败。
2. 单蛋白 overfit 实验（`pilot_summary.json`）也显示：单个构象可拟合到 0.121 Å，但 8 窗口只能到 ~4 Å →
   不是容量问题，是**输入 H 里信息不足**。
3. 因此把 5398 个 mdCATH 蛋白拿来训 decoder，只会得到一个**在新蛋白上同样打不过 persistence 的 decoder**，
   却付出巨大算力。

**正确顺序**：先修表示（让 H 携带可解码构象信息）→ 再用少量蛋白验证读出能显著优于 persistence →
最后才用大量蛋白扩大规模。

---

## 5. 建议的修改（按优先级）

### P0-1 给 encoder 加"全局构象"通道（最可能直接见效）
在 `ProteinGeometryEncoder` 中显式引入对**构象整体模式**敏感的特征，例如：
- 对全部 CA 做**提升/对齐后**的全局池化（masked mean / max）并广播回每残基；
- 加入全局几何量：回转半径、端到端距离、惯量张量特征值、若干长程 CA-CA 距离（|i-j|>8）；
- 给 attention 增加"全局 token"（类似 [CLS]）。
目标：让 H 的每一维都能反映 collective 的构象变化，而非仅局部几何。

### P0-2 改 Stage A 目标，使"时间变化方向"对齐真实结构位移
- 增加/优先使用一个**结构监督**的 temporal 项：用真实未来帧的 **Kabsch 对齐后内部 CA 位移**（或 distance-map 变化）
  直接监督 predictor 的 factor 增量（即 OPF 后的 ΔH 要能线性恢复 δX），而不只是 variance/contrast。
- 即先做一个"可解码性"目标：jointly 训练 (H, linear head→δX)，确保 H 里**存在**可读的位移信号，再谈 JEPA。
- 检查 `temporal-min-std` floor：它允许"有变化即可"，应替换为"变化需与真实位移相关"。

### P0-3 立刻验证一个小目标：**训练分布内** true-H 必须显著优于 persistence
- 用现有 8 蛋白，训练一个 latent_dim≥128、depth≥4 的 encoder + **线性/浅层** decoder，
  要求同蛋白 held-out 窗口 true-H error ≤ 0.5× persistence，否则不要进 B1b。
- 这个"最小充分性"测试比继续加 decoder 步数更有信息量。

### P0-4 修 decoder 的两个具体 bug（在表示修好后再评估）
- 去掉/弱化 `_coil_prior` 的固定基底，让 H 的输出不被先验淹没（或把 prior 做成可学习、小范数）；
- 当前 config 里 `fape/torsion/relative_disp` 权重为 0 时，decoder 实际只学绝对 CA MSE，
  这与"内部位移才是 H 的职责"相矛盾，需打开 `relative_disp_weight`。

### P1 数据规模（**在上述通过后**才做）
- 现有 manifest 只用 200 个 domain；磁盘上有 **5398** 个。
- 建议：Stage A 用几百–上千蛋白 + 多温度多 replica；B1a 用几十–上百蛋白；
  **留出**独立蛋白集做 gate（当前 8 训练 / 4 验证太少，无法区分表示问题与泛化问题）。

---

## 6. 一句话给决策

> true-H 读不出，**不是因为蛋白太少、decoder 太弱或训练不够**，而是因为 **Stage A 的 H 里没有可解码的构象信息**
> （探针在同蛋白内都打不过 persistence）。先修 encoder 的全局通道 + Stage A 的结构对齐目标，
> 用"同蛋白 held-out true-H ≤ 0.5× persistence"作为**进入 B1b 的硬门槛**；通过后再用 5398 蛋白扩规模，
> 否则只是把失败放大。

