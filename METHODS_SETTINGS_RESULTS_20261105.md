# Protein JEPA Frame — 方法、实验设置与结果（Methods · Settings · Results）

> **生成日期：** 2026-11-05 · **仓库：** `/liuyutian/protein_jepa_frame`
> **范围：** 本文档整合 LASP「correct-framework」主线及其 2026-11 后续试验的模型架构、训练配方、评测方法学与全部定量结果。内容来自仓库自身的结果报告、模型代码、训练脚本与冻结的 JSON 产物。
>
> **权威来源（按优先级）：**
> 1. `RESULTS_LASP_CORRECT_FRAMEWORK_20261031.md` — 9-arm S₀ 消融 + 预注册判定。
> 2. `EVALUATION_METHODS_AND_COMMANDS_20261021.md` — 规范指标定义 + 复现命令。
> 3. `PROJECT_REPORT_20261021.md` §13/§14 — 五方法同口径对比 + Lasp v3 400 窗逐格明细。
> 4. `RESULTS_HS_JOINT_20261101.md` / `RESULTS_BACKBONE_AND_PATHS_20261101.md` — 2026-11 试验。
> 5. `protein_jepa_frame/models/stage_b1b/model.py` — 自洽 GLE 内核（唯一事实来源）。
> 6. `protein_jepa_frame/training/train_lasp_ca_dynamics.py` — 训练目标与循环。

---

## 1. 问题定义与数据

系统学习蛋白质构象运动的**潜变量随机动力学模型**，并以「给定已观测锚点帧（及可选的历史）预测未来构象」作为评测任务。

### 1.1 数据集

| 项目 | 设置 |
|---|---|
| 数据集 | **mdCATH**（CATHe 结构域的分子动力学轨迹） |
| 帧间隔 | **1 ns / 帧**（`step_ps = 1000`） |
| 温度 | **5 档**：{320, 348, 379, 413, 450} K |
| 规范划分 | **1000dom**：800 训练结构域 / 100 验证结构域 / 0 重叠（`protocol.validate_manifest_split`） |
| 输入张量 | `X [B, K, L, 4, 3]` — 帧 × 残基（padding 到 `max_residues`）× 4 骨架原子 × 3 坐标 |
| 温度留出 | `train_not450`（4T，15999）/ `val_not450`（4T，2000）/ `val_t450`（500）/ `train_t450`（已见蛋白 × 450 K） |
| 保留测试集 | `test.jsonl` **全程保持密封** |

锚点是一个固定已观测帧；预测窗口为 `anchor → future`，时标 `h ∈ {1, 4, 16, 64, 192} ns`（即 1–192 帧，1 ns/帧）。

---

## 2. 模型架构（Stage B1b — 自洽 GLE）

核心模型是在学习到的潜表示 `H` 上的**自洽 H/S 广义朗之万（GLE）动力学**，附带一个「浴场」记忆变量 `S`。实现于 `protein_jepa_frame/models/stage_b1b/model.py`（`StageB1b`）。

### 2.1 潜表示 `H`

- **编码器被冻结**，且**不由 `StageB1b` 持有**。训练脚本先施加冻结的 Stage-A V2 编码器，再施加冻结的 B1a 归一化器，然后对得到的 `H` 做 `stop_grad`。
- `H` 形状为 `[B, L, dH]`，`dH = 128`（`latent_dim`）。
- `H_t = stop_grad(normalize_H(f_θ(X_t)))` — 编码器与归一化器均冻结。

### 2.2 能量函数（漂移项）

- `HSpaceEnergy` 是定义在逐残基 H token 上的**标量势** `U(H; T, c)`，带**全（非因果）残基自注意力**（置换等变、序列定义的边、不依赖坐标或初始 CA 距离图）。
  - `energy_depth = 3` 层 TransformerEncoder，`energy_heads = 4`，`energy_hidden = 128`，`norm_first=True`，`dropout=0`。
  - 温度以**对数相对特征** `log(T / 348.0)` 输入。
  - 每通道的**正定二次尾部** `0.5·softplus(log_tail)·H²`（init `tail_init=1e-3` legacy；`1e-2` from-none）提供恢复趋势。
- 力：`f = −∇_H U`，经 autograd（`compute_forces`，带 `create_graph` 以支持可微 rollout）。

### 2.3 浴场参数（静态、锚点条件化）

`StaticBathParams` 由**已观测锚点特征** `H₀` 与温度产生逐残基的 `D`、`C`、`Γ`（显式条件化于起始构象，而非仅序列）：

| 量 | 形状 | 作用 |
|---|---|---|
| `D` | `[B,L,dH]` | 正对角**迁移率**；`D = diffusion_scale · softplus(d_net(cond)+1)` |
| `Γ`（Gamma） | `[B,L,dS]` | 正对角**浴场弛豫率** |
| `C` | `[B,L,dH,dS]` | 逐残基**耦合**矩阵 |

- 浴场维度 `dS = 8`。
- **非零耦合初始化至关重要**：浴场对革新协方差的贡献与 `C` **二次**相关（`B_h = h·D·C`），故 `C=0` 将永远零梯度。`c_net[-1].weight` 初始化为 `~ N(0, c_init_std²)`，`c_init_std = 1e-1`。
- 尺度：`diffusion_scale = 1e-4`（→ `R = 2hD ~ 0.2`），`coupling_scale = 1e-3`（legacy）/ `0.1`（active），`gamma_scale = 2e-4`，`gamma_floor = 2e-5`。

### 2.4 采样器（单步 Euler–Maruyama，训练/部署一致）

`sample_step(collective_mobility=True)` 是部署时的转移。令 `M = √D·K·√D`（K 为对角模型的单位阵），`E = U + |S|²/2`：

```
p     = f − C·S
H'    = H + h·√D·(√D·p) + √(2hD)·ε_H          [ε_H ~ N(0,I)]
S'    = S + Cᵀ·(H'−H) − h·Γ·S + √(2hΓ)·ε_S    [ε_S ~ N(0,I)]
```

- `R = 2hD` 是 H 的观测噪声（由 D 导出，**不是**独立参数）。
- 联合迁移率 `[[M, MC],[CᵀM, Γ+CᵀMC]]` **由构造半正定**（分解 `[I; Cᵀ] M [I, C] + diag(0, Γ)`），即采样器涨落–耗散自洽。
- Legacy `bath_coupling="force"`（2026-10-22 之前）仅为复现旧 checkpoint 保留；它结构性发散（64 步内 `|H| 11 → 3969`），不是可用模型。

### 2.5 S₀ 的历史后验

初始浴场状态 `S₀` 可由四种模式（`s0_mode`）产生：

| 模式 | 定义 |
|---|---|
| `gru` | 摊销因果 GRU `r_ψ(H_{≤0}) → (m₀, chol(P₀))`（历史默认；**在冻结 SOTA 中其实从未真正训练**——见 §4） |
| `exact` | 采样器动力学上的闭式**卡尔曼后验** `p(S₀ | H_hist)`（块对角、逐残基） |
| `exact_spatial` | 空间相关集体噪声的跨残基全浴场后验 |
| `prior` | 单位方差先验，忽略历史 |

**精确卡尔曼后验**（`exact_bath_posterior`）：把 `H_{k+1}` 读作 `S_k` 的线性高斯观测，观测矩阵 `C_obs = −h·D·C`、噪声 `R = 2hD`，一次标准卡尔曼更新（Joseph 形式）后接采样器自身的时间更新，即得精确滤波后验。无需摊销网络、无需训练信号。

### 2.6 诊断卡尔曼 NLL

`kalman_nll` 是**仅诊断用**（永不优化）的似然。它对 `H_{1:N} | H_{−K+1:0}` 打分，并采用**正确**的 `B_h = h·D·C`（旧 `h·C` 形式漏掉了迁移率 `D`，偏 ~1/D ≈ 5e4）。

---

## 3. 训练（Stage B1b — CA 动力学）

实现：`protein_jepa_frame/training/train_lasp_ca_dynamics.py`（`CADynamics`）。

### 3.1 训练目标

对每个时标 `h`，模型从已观测锚点 rollout `n_samples` 条轨迹，并用**能量得分**（proper scoring rule）对真实未来目标打分：

```
loss = latent_weight·ES(Ĥ, H_target)
     + ES(CÂ, CA_target)                 # 坐标能量得分
     + contact_weight·contact_loss
     + rmsf_weight·RMSF_loss
     + distance_weight·ES(dist̂, dist)   # 非局部 CA 距离
     + clash_weight·clash_loss
     + mode_weight·slow_mode_scale_loss
     + mode_distribution_weight·slow_mode_distribution_loss
```

其中 `ES(samples, target) = mean(cross) − ½·mean(pair)` 是无偏样本能量得分（欧氏、逐维归一）。接触损失为 `|σ((8−d)/0.5) − σ((8−d_true)/0.5)|`（MAE 或 Brier）。

**关键默认权重**（来自 `argparse`）：`latent_weight=0.1`、`contact_weight=10.0`、`rmsf_weight=1.0`、`distance_weight=0.0`、`clash_weight=0.0`、`mode_weight=0.0`、`mode_distribution_weight=0.0`。`n_samples=4`（训练）、`grad_steps=8`（截断 BPTT）。

### 3.2 优化设置

| 项目 | 设置 |
|---|---|
| 优化器 | **AdamW**，`weight_decay = 0.01` |
| 学习率 | `lr = 1e-4`（内核）；开启 `--train-encoder` 时编码器 `lr·encoder_lr_scale`，`encoder_lr_scale=0.1` |
| 调度 | 线性 warmup（`warmup`）后可选 **cosine** 衰减（`_lr_scale`） |
| 梯度裁剪 | `clip_grad_norm_(parameters, 1.0)` |
| Batch size | 每步 1 个 clip（序列）；时标轮转 |
| 步数 | 7000（S₀ arms，对齐 SOTA 的 val 最优点）/ 500（HS-joint pilots）/ 30000（legacy SOTA） |
| 验证 | 每 `val_every=50`（arms 500）步；val 选出的 `checkpoint_best.pt` + `checkpoint_latest.pt` |
| 精度/后端 | `sdpa_kernel(SDPBackend.MATH)`（注意力 double-backward 用于力梯度） |

训练读入最后 4 帧已观测帧作历史，及 `horizon` 帧真实未来帧（1 ns 间隔）。开启 `--train-encoder` 时，用**冻结的目标编码器**（起始编码器的 deepcopy）作教师，`representation_retention` 加入 `teacher_weight·teacher + teacher_delta_weight·teacher_delta`。

---

## 4. 冻结 SOTA 及其缺陷（动机）

旧「SOTA」是 3 个 seed 的冻结系综（`results/sota_frozen_20261024_backup/`，checkpoint SHA256 前缀 `138909db… / 93e90975… / 99d51a4a…`）。调查发现记忆变量 `S` **数学上失效**：

| 证据 | 数值 |
|---|---|
| 浴场方差 / R（3 seed） | 4.5e-8 / 2.9e-8 / 6.0e-8（≈1e-8） |
| `history_posterior.head` 权重 | **全零**（`history_conditioning=False`，从未训练） |
| 30k 步后有效耦合 vs 初始 | 9.999e-05 vs 1.000e-04（保留 4 位有效数字不变） |
| S₀ 杠杆（3 来源） | 0.0000 Å |

修复了三处缺陷：
1. **卡尔曼增益符号** — `K = −P·B_hᵀ·V⁻¹`（均值更新符号错了；Joseph 协方差不受影响，这正是单步 NLL「看起来对」的原因）。
2. **替换 S₀** — GRU 摊销推断 → 精确卡尔曼后验（`s0_mode=exact`）。
3. **激活浴场** — 以 `coupling_scale=0.1, gamma_scale=2e-5, gamma_floor=2e-6`（耦合 ×100、Γ×0.1）训练，使 `S` 真正推动轨迹。

---

## 5. 评测方法学

规范指标定义与复现命令见 `EVALUATION_METHODS_AND_COMMANDS_20261021.md`；冻结的对照基准是 **BioEmu2**（Microsoft 全原子扩散系综模型，zero-shot）加上 `persistence` 与 `oracle` 参考基线。

### 5.1 七项系综指标（standard-200 协议）

标准面板：**200 窗口 / 50 蛋白 / 12 样本 / 时标 {1,4,16,64} ns / 5 温度 × 各 40 窗口**，`eval seed=0`。窗口键与冻结 BioEmu2 参考**逐位一致**；采样器噪声由窗口键派生，保证配对比较精确。

| 指标 | 定义 | 方向 |
|---|---|---|
| RMSF MAE | 对齐逐残基 RMSF 谱 L1 vs 真值 | ↓ |
| RMSF 相关 | 逐残基 RMSF 谱 Pearson r | ↑ |
| Root-Mean W2 | 各残基 CA 位置分布 1-D Wasserstein-2 的 RMS | ↓ |
| Variance W2 | 逐残基涨落幅度（σ²）分布的 W2 | ↓ |
| MD-PCA W2 | top-k=3 慢模投影分布的 W2 之 RMS | ↓ |
| Weak-contact J | `P(d<8Å) ∈ [0.05,0.5)` 位点的 Jaccard | ↑ |
| Transient-contact J | `P(d<8Å) ∈ (0.01,0.99)` 位点的 Jaccard | ↑ |

所有 W2 均为精确 1-D Wasserstein-2（排序分位函数，无分箱）：`W2² = ∫₀¹ (F_a⁻¹(q) − F_b⁻¹(q))² dq`。

### 5.2 统计推断（预注册）

- 在**相同窗口键**与**相同采样器噪声**上配对比较（噪声 seed 由窗口身份 `protein/trajectory/anchor/horizon` 派生，与模式无关）。
- **蛋白聚类 bootstrap**，10000 次抽样，seed `20261024`，单侧 98.33% 上/下界。
- **WIN**：3-seed 均值改进 > 0 **且** ≥2/3 seed 的单侧界 > 0。**LOSS**：对称。否则 **INCONCLUSIVE**。
- **Seed 噪声地板**：先对冻结 SOTA 自身 3 个 seed 两两配对。在 h=64 有 10/21 指标×配对格显著分歧，因此任何 S₀ 效应都必须在 3-seed 均值上超过此地板。

### 5.3 附加指标族

- **几何（CA）**：CA 键长均值/非法占比（合法带 3.3–4.3 Å）、非局域碰撞（截断 2.5 Å）、Rg、内部距离 RMSE/接触误差、伪扭转角 MAE。
- **骨架（T1）**：N/CA/C/O RMSD、键长/键角合法性、Ramachandran、ω trans、肽 C–N 合法性、骨架碰撞 — 经 `backbone_metrics_v2.py`。
- **静态系综（T2A，BioEmu 风格）**：参考态覆盖、自由能 MAE、布居 MAE/TV/JSD。
- **动力学/MSM（T2B）**：滞后态精度、转移矩阵误差、隐含时间尺度、MFPT、转移检测 F1。
- **序感知轨迹**：逐步 RMSD、路径长度、端到端、迂回度、Rg 曲线、lag-1 ACF、walk-RMSF、轨迹能量得分。
- **温度泛化（T3）**：5×{已见,未见}×{正确,错误,无} 温度矩阵。

---

## 6. 结果 — LASP correct-framework（2026-10-31）

### 6.1 机制已修复，S 真正进入动力学

在 active 尺度重训后，耦合**保持 ×100 active 尺度**，S₀ 杠杆变为非零：

| Arm | 有效耦合 | (C·m₀)/(f) | S₀ 杠杆 h=1/4/16/64（Å） |
|---|---|---|---|
| SOTA（惰性浴场） | 9.999e-05 | 3.2e-05 | 0.0000 / 0.0000 / 0.0000 / 0.0000 |
| exact | 1.003e-02 | 0.311 | 0.090 / 0.102 / 0.069 / 0.092 |
| prior | 1.003e-02 | 0.230 | 0.044 / 0.078 / 0.093 / 0.077 |
| gru | 1.003e-02 | 0.189 | 0.333 / 0.728 / 1.281 / 0.891 |

训练/验证损失在 step 7000 前已平台化（exact 4.28–4.35、prior 4.28–4.38、gru 4.23–4.35），无欠训练。

### 6.2 主问题 — exact vs prior（仅 S₀ 来源不同）

3-seed 聚合（`results/lasp_s0_arms_20261031/decision_3seed.json`）：

| h | md_pca_w2 (↓) | rmsf_corr (↑) | variance_w2 (↓) |
|---|---|---|---|
| 1 | −0.013 INCONCLUSIVE | −0.002 INCONCLUSIVE | +0.030 INCONCLUSIVE |
| 4 | **−0.140 LOSS** | +0.001 INCONCLUSIVE | **+0.617 WIN** |
| 16 | −0.106 INCONCLUSIVE | −0.004 INCONCLUSIVE | +0.307 INCONCLUSIVE |
| 64 | −0.232 INCONCLUSIVE | +0.023 INCONCLUSIVE | −0.914 INCONCLUSIVE |

**三个主指标无一致 WIN/LOSS。** exact 在 h=4 的 md_pca_w2 上显著*更差*、同 horizon 的 variance_w2 上显著*更好*；其余均在噪声地板内。次要指标给 exact 小胜（rmsf_mae h=1/4、root_mean_w2 h=4），但 h=16/64 无一致。

### 6.3 对 SOTA — **未击败**

| h | rmsf_corr (↑) | rmsf_mae (↓) | root_mean_w2 (↓) | md_pca_w2 (↓) |
|---|---|---|---|---|
| 1 | −0.037 LOSS | −0.221 LOSS | −0.197 LOSS | **+0.396 WIN** |
| 4 | −0.021 INCONCL | −0.344 LOSS | −0.190 LOSS | **+0.613 WIN** |
| 16 | −0.057 LOSS | −0.256 INCONCL | −0.132 LOSS | +0.066 INCONCL |
| 64 | −0.063 LOSS | −0.114 INCONCL | −0.050 INCONCL | −0.422 INCONCL |

各 arm 在 **md_pca_w2**（系综分布 PCA-W2）上一致更好，但在 rmsf_corr、rmsf_mae、root_mean_w2 与接触指标上更差。**correct framework 未击败 SOTA。**

### 6.4 尺度控制 — 差距来自重训，而非激活浴场

把冻结 SOTA 权重放到 active 尺度（不重训）指标变化 <2%，排除了「浴场尺度毁掉 SOTA」的解释。重训 arm 与 SOTA 的差距来自重训本身。

### 6.5 判定（预注册）

1. **击败 SOTA：否。** rmsf_corr 在 h=1/16/64 LOSS；rmsf_mae/root_mean_w2 在短时标 LOSS；仅 md_pca_w2 稳定 WIN。
2. **正确的历史条件 S₀ 有帮助：方向性但非显著/不一致。**
3. **「机制未连接」被排除** — 耦合保持 active 尺度，S₀ 杠杆 0.07–0.13 Å。
4. **框架仍正确**，但「correct」≠ 在此数据上「better」。

> **2026-11-01 审计修正：** 早前「2–4% 线性记忆上界」结论**已撤回**。该探针的 anchor-only 控制用了截距（非 H₀ 预测器）、其 λ 在被打分蛋白上选取，且线性预测器无法界定非线性可预测信息。exact-vs-prior 的 INCONCLUSIVE 是「未检测到增益」的证据，而非「无记忆」的证据。

---

## 7. 结果 — 2026-11 后续试验（均未晋升）

以下均为开发集上的 500 步（或诊断）实验。**无一晋升为新配方，无一在完整面板上击败 BioEmu2。**

### 7.1 H+S joint（frozen vs train-encoder）— `pilot_v1`

两 arm（frozen / joint 编码器更新），seed 42，500 步，collective mobility + exact S₀ + `noise_mix=0`。joint arm 最佳 vs frozen 最佳（BioEmu2 列作参考）：

| 指标 | frozen | joint | BioEmu2 |
|---|---:|---:|---:|
| RMSF MAE ↓ | 1.0394 | 1.0579 | 1.9917 |
| RMSF corr ↑ | 0.4981 | 0.4983 | 0.5797 |
| Root-Mean W2 ↓ | 1.4985 | 1.5130 | 2.2904 |
| Variance W2 ↓ | 7.9542 | 8.0337 | 18.5754 |
| MD-PCA W2 ↓ | 7.3730 | 7.3259 | 6.1143 |
| Weak-contact J ↑ | 0.1032 | 0.1016 | 0.1316 |
| Transient J ↑ | 0.2519 | 0.2479 | 0.2653 |

更新 H 略微改善 MD-PCA W2，但位置分布、涨落幅度与接触指标变差；RMSF 相关无可靠增益。**未通过预注册的扩训规则。**

### 7.2 Readout-retention 四臂（每臂 500 步）

观测帧 readout 监督改善了若干历史系综得分并降低了 oracle readout 退化，但 readout 仍差于冻结 H，且坐标/距离端点能量得分相对冻结变差。四臂均未通过 16/64 ns oracle-ratio 预算。**未晋升。**

### 7.3 骨架（T1）— 全骨架 readout 仍是明确失败

- 即便真实未来 CA 坐标在现行理想 lift 下也产生 **~2.61 Å** 肽 C–N 距离；`legacy_valid_frac = 0`（所有时标）。
- 预测 N/C/O RMSD ≈ **4.63 Å**（差于 persistence ≈ 3.49 Å）；oracle N/C/O ≈ 1.66 Å。
- 局部帧传输（固定 500 步）把预测 N/C/O RMSD 从 4.6326 改善到 4.5756 Å、C–N 软损失 0.6006→0.2599 Å、oracle 1.6620→1.4423 Å — 是测量到的权衡，非 T1 修复。
- Ramachandran 离群：预测 33.04%（传输）vs 目标 MD 1.69%。

### 7.4 序感知轨迹 — 动力学过慢/过粘

64 ns：路径长度 **67.91 Å vs MD 130.23 Å**（≈44% MD），lag-1 ACF **0.836 vs 0.521**。连续预测改善了整条路径能量得分（优于 persistence），但逐帧位置误差仍更高、lag-1 相关超过 MD。

### 7.5 其它已完成的试验（均阴性或 inconclusive）

| 实验 | 主要结果 | 判定 |
|---|---|---|
| H-Markov 训练控制（coupling=0） | C0−H+S 长时标 U 能量差 **−0.0101 Å [−0.0510, +0.0205]** | 不晋升配方 |
| Endpoint vs path-mixture 目标 | 16/64 ns U-score **2.0027 / 2.0277 Å**，差 −0.0250 Å | 不支持 endpoint |
| Joint-path vs time-marginal | 长时标 U-score 差 **−0.0324 Å [−0.0846, +0.0025]** | 不支持 |
| Contact 估计器（population-U） | path U-score 差 **−0.0159 Å [−0.0597, +0.0092]** | 保留 plugin |
| 物理距离目标（增量 vs 绝对） | I−A **+0.0278 Å [+0.0016, +0.0726]** | 均未击败 BioEmu2 |
| CA 局部残差解码分支 | oracle RMSD 差 **−0.0010 Å [−0.0036, +0.0014]** | 不采用分支 |
| Stage-A relpos 适应（3 arm） | true-future-H log-MSE 0.0096/0.206/0.141；predicted-H 每个时标都输给 persistence | 不采用 |

---

## 8. 基线

来自 `EVALUATION_METHODS_AND_COMMANDS_20261021.md` §3 与 `PROJECT_REPORT_20261021.md` §13/§14。

### 8.1 BioEmu2 zero-shot（冻结对照基准，200 窗口，n_samples=12，3 次重复 mean±std）

| horizon | RMSF-r ↑ | RMSF ↓ | pca_W2 ↓ | weak_J ↑ | trans_J ↑ |
|---|---|---|---|---|---|
| 1 ns | 0.444±0.003 | 2.164±0.011 | 0.0807±0.0008 | 0.000 | 0.173±0.001 |
| 4 ns | 0.595±0.009 | 1.846±0.049 | 0.0819±0.0021 | 0.177±0.002 | 0.298±0.003 |
| 16 ns | 0.597±0.011 | 1.548±0.042 | 0.0771±0.0032 | 0.243±0.001 | 0.361±0.002 |
| 64 ns | 0.613±0.018 | 1.622±0.041 | 0.0876±0.0014 | 0.212±0.001 | 0.382±0.003 |

### 8.2 三/五方法同口径判定

- **系综（分布）指标：BioEmu2 领先我们 19:4。** 我们在锚定**位置精度（Kabsch）**四个时标领先，且 64 ns rollout 更稳定。MDGen 对 BioEmu2 输 0:20（zero-shot 域外：10 ps vs 1000 ps/帧）。
- **persistence**：RMSF-r ≡ 0（delta 系综退化）、contact J ≡ 0；Kabsch ≈ 3.3–4.0 Å（短时标上它反而强，因为 1 ns 内锚点几乎不动）。

---

## 9. 温度泛化（T3）摘要

- **Model 1**（温度条件头，`cond_dim=2`）：喂**错误温度显著退化**输出（如 T320 RMSF 0.32 → 0.96），证明模型*真的用了*温度。未见 T=450 差距最大（RMSF-r 0.34 vs 已见 T 约 0.58–0.62）。`temperature_monotone=True`。
- **Model 2**（from-none GLE v4g）：`correct ≈ wrong ≈ none`（<1% 差）— 此路径上温度输入**近乎失效**（架构支持 `temperature_proj`，但训练后通路贡献极小）。温度留出代价温和（Δskill −0.01 至 −0.05），主要由隐式锚点上下文驱动，而非显式温度输入。

---

## 10. 当前状态与开放问题

**底线：** 三处机制 bug 全部修复、`S` 真正进入动力学，但修复后的模型在完整评测上**未击败冻结 SOTA（BioEmu2）**。正确的历史后验未在预注册标准下转化为可测量的预测增益。

对 SOTA 的持久差距：

1. **rmsf_corr / rmsf_mae / root_mean_w2 / 接触指标** — 重训 arm 在这些上回退。
2. **骨架 readout（T1）** — 跨残基 C–N 连续性即便 oracle CA 也失败（~2.61 Å）；预测 N/C/O RMSD ≈ 4.6 Å。
3. **长时标动力学** — 路径长度 ~44% MD、lag-1 ACF 0.84 vs 0.52（过粘）。
4. **温度**在 from-none GLE 路径上失效（Model 2）。

**最高价值后续**（来自 2026-10-31 报告，按性价比排序）：
- 先把 `gru` 的先验方差降到 exact/prior 水平再复比 exact vs gru（当前差异是方差，非后验质量）。
- 优先攻 rmsf/contact 指标（对 SOTA 的主要差距），而非继续 S₀ 工作。
- 任何新记忆测试，登记一个恰当的 H₀ 条件对照并独立选取 λ（2026-11-01 审计修正）。

---

## 11. 复现

```bash
export TMPDIR=/liuyutian/protein_jepa_frame/.tmpwork
export PYTHONPATH=/tmp/lasp_stability_deps:/liuyutian/protein_jepa_frame

# 9-arm S₀ 训练网格 + standard-200 评测 + 判定
bash run_scripts/drive_lasp_s0_arms.sh        # 训练（slot 驱动，6 路并发）
bash run_scripts/run_lasp_s0_arms_eval.sh     # standard-200 系综指标
python -m scripts.report_s0_arms ...          # 3-seed 聚合 + WIN/LOSS/INCONCLUSIVE

# H+S joint pilot
bash run_scripts/run_hs_joint_pilot.sh

# 骨架诊断 + 有序路径（2026-11）
python -m scripts.eval_ca_backbone_diagnostic --checkpoint ... --out ... --device cuda:0 --max-windows 200
python -m scripts.eval_ca_ordered_paths       --checkpoint ... --out ... --device cuda:0

# 回归
python -m pytest -q                            # 视阶段约 332–756 tests
```

---

## 12. 四模型对比表（我们最好的模型 vs BioEmu2 vs persistence vs MDGen）

> **本节诚实声明（务必先读）：** 我们目前**尚未在核心系综指标上全面击败 BioEmu2**。
> 真相分两层，必须分开表述，不可混为一谈：
>
> 1. **同协议、逐格统计判定层面**（§14.4，400 窗 / 100 蛋白，蛋白聚类 bootstrap 95% CI）：
>    我们的最佳模型 **v3 对 BioEmu2 显著 14 胜 2 负**（12 格无显著差异）。这是一个真实的、
>    可复现的领先——但集中在**分布形态/接触/W2** 类指标；唯一系统性败点是
>    **RMSF-r（16 ns、64 ns 两个显著败格）**，即「涨落谱的相关性」我们仍输。
> 2. **绝对数字层面**：BioEmu2 在长时标（16–64 ns）的 RMSF-r（~0.60–0.61）仍显著高于我们
>    （v3 ≈ 0.56/0.53），这也是为什么「RMSF-r 全时标败于旧 Lasp」会在 §14.1 被点名。
>
> 因此：**「我们在 W2/接触/位置精度上达到并局部超过 SOTA」是成立的；「我们全面达到 SOTA」不成立。**
> 下表用两个口径分别呈现，绝不把 400 窗 v3 数字与 140 窗 Lasp 数字混排。

### 12.1 表 A — 我们最佳模型（v3）vs BioEmu2（同协议 400 窗 / 100 蛋白，逐格统计判定）

`Δ = v3 − bioemu2`；**粗体** = 配对 bootstrap 95% CI 不含 0 且方向利于 v3。数据源：§14.4。

| 时标 | RMSF↓ | RMSF-r↑ | root_W2↓ | var_W2↓ | pca_W2↓ | weak_J↑ | trans_J↑ |
|---|---|---|---|---|---|---|---|
| 1 ns | 0.570 vs 0.614 | 0.473 vs 0.466 | 6.327 vs 6.346 | 2.661 vs 2.580 | 0.058 vs 0.056 | 0.000 vs 0.000 | **0.164 vs 0.125** ✅ |
| 4 ns | **0.555 vs 0.609** ✅ | 0.576 vs 0.586 | **3.588 vs 3.634** ✅ | **4.162 vs 4.725** ✅ | 0.059 vs 0.058 | **0.154 vs 0.121** ✅ | **0.284 vs 0.230** ✅ |
| 16 ns | 1.033 vs 1.090 | 0.560 vs 0.602 ❌ | **2.229 vs 2.263** ✅ | 11.337 vs 11.793 | 0.104 vs 0.102 | **0.223 vs 0.166** ✅ | **0.360 vs 0.301** ✅ |
| 64 ns | **1.693 vs 1.846** ✅ | 0.528 vs 0.583 ❌ | **3.212 vs 3.251** ✅ | **21.559 vs 22.197** ✅ | 0.145 vs 0.148 | **0.196 vs 0.156** ✅ | **0.353 vs 0.300** ✅ |

> ✅ = v3 显著赢，❌ = bioemu2 显著赢，无标记 = 无显著差异。总账 **14 ✅ : 2 ❌ : 12 无差异**。

### 12.2 表 B — 五方法同口径（140 窗 / 40 蛋白，unseen 蛋白，系综指标均值）

数据源：`results/five_method/all.jsonl`（unseen 子集）。**本表为绝对数字均值，非配对统计判定**；用于横向看五方法的相对量级。

| 时标 | 指标 | 我们（lasp） | bioemu2 | persistence | MDGen | Model2 |
|---|---|---|---|---|---|---|
| 1 ns | RMSF-r ↑ | 0.510 | 0.581 | ≡0 | 0.466 | 0.442 |
| 1 ns | RMSF MAE ↓ | 0.593 | 1.534 | 1.84 | 2.039 | 1.278 |
| 4 ns | RMSF-r ↑ | 0.625 | 0.686 | ≡0 | 0.571 | 0.692 |
| 4 ns | RMSF MAE ↓ | 0.781 | 1.430 | 1.84 | 2.015 | 0.818 |
| 16 ns | RMSF-r ↑ | 0.604 | 0.718 | ≡0 | 0.583 | 0.644 |
| 16 ns | RMSF MAE ↓ | 1.342 | 1.285 | — | 2.116 | 1.433 |
| 64 ns | RMSF-r ↑ | 0.474 | 0.725 | ≡0 | 0.558 | 0.407 |
| 64 ns | RMSF MAE ↓ | 1.942 | 1.678 | — | 2.462 | 2.273 |
| 64 ns | Kabsch ↓ | 7.50 Å | 7.78 Å | ~3.3 Å | 15.30 Å | 6.59 Å |

> 口径说明：
> - **persistence 的 RMSF-r / contact J ≡ 0**（delta 系综的谱是常数，Pearson 按定义退化）。其 Kabsch（~3.3 Å）短时标占优，因为 1 ns 内锚点几乎不动——这是「轨迹推进」任务的内在性质，不代表系综质量。
> - **MDGen zero-shot** 在 mdCATH 上劣于 persistence（步长 OOD：10 ps vs 1000 ps/帧），**不代表其正常水平**。
> - 本表 Lasp 列为 §13 旧口径（`plan_a_unfrozen_dec_1000dom`，force 耦合）；我们的**最佳模型 v3** 是 velocity 耦合，其逐格统计判定见 **表 A**，两者窗口集不同、不可混排。

### 12.3 表 C — 位置精度（Kabsch CA RMSD，Å）

| 时标 | 我们 | bioemu2 | persistence | MDGen |
|---|---|---|---|---|
| 1 ns | 3.61 | 4.52 | ~3.3 | 12.90 |
| 4 ns | 4.23 | 5.40 | — | 13.43 |
| 16 ns | 5.51 | 6.43 | — | 14.29 |
| 64 ns | 7.50 | 7.78 | ~3.3 | 15.30 |

> **我们（lasp/v3）在四个时标的锚定位置精度上均优于 BioEmu2**——这是 GLE「轨迹推进」相对扩散「系综采样」的结构性优势，也是我们目前**最确定、最可辩护的 SOTA 领先**。Model 1（Stage-A 系综）rollout 8.39 Å 的自回归漂移在此被 GLE 修复（path_length 7.6 Å vs 33.4 Å，walk RMSF-r 0.579 vs 0.267，均显著）。

### 12.4 一句话总结

| 维度 | 我们的位置 | 相对 BioEmu2 |
|---|---|---|
| 锚定位置精度（Kabsch） | **领先**（4 时标全胜） | ✅ 稳定赢 |
| 分布形态（root/var W2） | 领先（4 ns 起显著） | ✅ 多数赢 |
| 接触谱（weak/trans J） | 领先（4–64 ns 显著） | ✅ 稳定赢 |
| 涨落谱相关性（RMSF-r） | **落后**（16/64 ns 显著败） | ❌ 稳定输 |
| 长时标 RMSF 幅度（MAE） | 64 ns 赢、短时标混 | ⚖️ 互有胜负 |
| 慢模 PCA-W2 | 相当 | ⚖️ 无显著差异 |

**结论：** 我们的模型在**位置精度、分布形态与接触谱**上确确实实达到并局部超过了 SOTA（BioEmu2）；但在**涨落谱相关性（RMSF-r）**这一项核心系综指标上仍落后，因此**「全面 SOTA」尚未达成**。差距的根因已定位为「速度耦合把系综展布压窄、涨落谱被抹平——方向对了、展布窄了」，是下一步的主攻点，而非机制性缺陷。

---

## 附录 A — 关键超参默认值（`StageB1bConfig`）

| 参数 | 默认 | 含义 |
|---|---|---|
| `latent_dim` | 128 | H 状态维度 |
| `bath_dim` | 8 | S 状态维度（0 关闭浴场） |
| `energy_depth / heads / hidden` | 3 / 4 / 128 | 能量 transformer |
| `history_hidden` | 128 | GRU 历史后验宽度 |
| `step_ps` | 1000.0 | 观测间隔（ps） |
| `diffusion_scale` | 1e-4 | `R = 2hD ~ 0.2` |
| `coupling_scale` | 1e-3（legacy）/ 0.1（active） | C 尺度 |
| `gamma_scale / gamma_floor` | 2e-4 / 2e-5 | 浴场速率（active：2e-5 / 2e-6） |
| `c_init_std` | 1e-1 | C 初始化标准差（非零以获得梯度） |
| `reference_temperature_k` | 348.0 | log-T 参考 |
| `tail_init` | 1e-3（legacy）/ 1e-2（from-none） | 二次尾部 |
| `bath_coupling` | "velocity" | FDT 正确；"force" 是发散的 legacy |
| `jitter` | 1e-6 | 协方差 Cholesky jitter |
