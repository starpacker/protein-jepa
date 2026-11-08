# v3 SOTA 蛋白质 GLE 动力学：根因审计、代码修复、全量重新训练与主链重建

**Version:** v3.0 · 2026-11-07 · 交付对象：项目 Coding Agent / Codex / Copilot

> **文档目的**：这是必须实际执行的 v3 SOTA 代码审计与重训练任务书，不是建议性总结。要求 Agent **在 v3 训练/推理代码基线上确定根因 → 修改训练目标/表示/解码/动力学等相应代码 → 从头或完整端到端重训练整个 v3 核心模型 → 生成新的测试集 rollout → 几何/统计物理/全主链验收**。不能只修 PDB exporter、只在后处理强行投影或仅做局部微调来宣称完成。
>
> **输入证据**：原文件 `1ka1A01_1ka1A01_T320_R1_h64_s0 (1).pdb` 和本次新上传 `1ka1A01_1ka1A01_T320_R1_h64_s0.pdb`；**逐字节一致**，没有新的轨迹改善。PDB `REMARK` 为 `ORIGIN v3 HSEnsembleProducer single-H-space rollout n_frames=64 temp=320.0K`。
>
> **分析基准**：65 个 `MODEL`（第一个为 Anchor，后 64 个为预测），每帧 96 个 Cα，残基编号 A:2–97；坐标单位按 PDB 规范解释为 Å。当前只有 **单条** 320 K、64 步轨迹，不能据此证明任何动力学系综规律。
>
> **审计状态**：**BLOCKER / P0**。现有坐标几何不合格，应暂停使用此输出进行真实蛋白动力学结论或作为下游精细原子模型的可靠输入。

---

## 本次复核：新上传 PDB 未发生任何变化（重要）

对两份 PDB 进行逐字节比较，内容完全一致（`cmp` 结果相同）；两者 SHA-256 为：

`acfe320cfb7f0c0e7d9e6ce5e870edefcdb0c58c76a0b7d6afbaf7bcd78c5eb2`

因此新文件**不是修复后的新轨迹**，不能将本次上传解释为模型改进。两者均为 65 MODEL / 96 CA，Anchor、t=1 和 t=64 的结构缺陷数据完全相同。新文件 `MODEL 2` 的 95 个相邻 CA 对中有 74 对 `<3Å`，`MODEL 65` 有 70 对 `<3Å`。任何声称“新版本几何更好”的论断都必须提交**不同 hash 的新生成轨迹**，并附训练 run ID、v3 commit、checkpoint hash、训练配置、rollout 配置。

**必须保留两份输入文件作为 immutable baseline**；接收新轨迹后先检查文件哈希、模型权重及生成日志，防止误上传/旧缓存覆盖。

---

## 本轮最重要的新增指令：先做 GT latent 解码与 encoder 根因定位

**优先级 P0-A / P0-B（必须先于盲目改 loss）：** 取得真实 MD 多帧 GT 构象 → `z_GT=E_v3(X_GT)` → **不跑 GLE** 用 v3 decoder 解码 → 比较真实几何/变化；再审计 v3 的 Encoder、H-space 表示和真实 OpenFold 几何模块是否被采纳。若 GT latent 解码也塌缩，应优先修 encoder/decoder/表示及相应训练损失；若 GT latent 解码正常而 rollout 坍塌，优先修 GLE 与训练–推理分布失配。**根因一旦得到代码和实验支持，就及时修复对应源码，并从可靠初始化重新训练整个 v3 核心模型**；不可停留在分析报告或只修改 inference/postprocess。

## 必须完成的 v3 全流程任务（优先级最高）

```text
Freeze v3 baseline + data/splits/checkpoint metadata
        │
        ▼
Reproduce t=0 / first-step geometric failures on v3
        │
        ▼
GT source QC → GT latent encode/decode (no GLE) → encoder/OpenFold audit
        │
        ▼
Trace data → representation → H-space → GLE → decoder → exporter
        │
        ▼
Prove root cause with isolated tests, no speculative patching
        │
        ▼
Fix faulty code + correct missing geometry-aware training objective
        │
        ▼
Train entire v3 core model END-TO-END (fresh v3 experiment)
        │
        ▼
Re-generate 65-frame and longer open-loop rollouts (held-out)
        │
        ▼
Geometry + MD-matched statistical/kinetic QC + no-regression checks
        │
        ▼
For each frame reconstruct N, CA, C, O with OpenFold-informed pathway
        │
        ▼
Publish diffs, full training evidence, artifacts, failures, comparison
```

**必须亲自执行代码和训练，而不是交回一份规划；在需要训练 GPU/原始 MD 数据/仓库访问但实际缺失时，必须精确报告阻塞项、验证已做部分，绝不能伪造 retrain、checkpoint 或结果。**

## 数据驱动的核心假设：整体指标好，不代表生成构象合法

用户目前观察到 RMSD 和 Rg 表现可以接受，但几何上 CA 链已经严重塌缩。可能的真正根因包括：

- 训练优化的是全局 `RMSD/Rg/mode-space` 统计量，未明确惩罚逐帧、逐残基 **3.8Å 邻接几何 / 主链拓扑**；
- 损失定义了几何项但**仅施加于训练真值、encoder latent 或 one-step teacher-forcing 输出**，没有正确反传到自由 rollout 的 decoded coordinates；
- 高维 Cα 位置用低维或低频 H-space 表示，decoder 不保留局部高频结构；
- 蛋白被错误编码为平滑网络输出（图卷积/邻域平均/模态截断），导致看似合理的 Rg、RMSD，却把 alpha-helix 压扁；
- 数据预处理、输入 Anchor 已损坏，或 exporter/单位/残基重排有 bug；
- GLE 积分单步输出、方差尺度、热噪声或 `dt` 错误，加速了局部结构塌缩。

**这些都是待证伪的候选机制。Agent 必须查阅 v3 实际代码、训练日志及数据定义来确定真正根因。** 禁止预设“只要加 bond loss 就全部解决”。

---

## 0. 交付任务：直接给 Coding Agent 的任务定义

### 总目标

修复 `HSEnsembleProducer` / GLE rollout / latent-to-coordinate decoder / PDB exporter 链路，使得系统：

1. 在 **当前 v3 SOTA 训练工程代码**上定位、修复并证明根因；**重新训练整个 v3 模型**，使之能够生成连续、数值稳定、拓扑与局部几何合理的 **Cα 轨迹**，而不仅仅是用后处理遮掩；
2. 能够在每个有效的 rollout 帧上生成 **完整蛋白主链坐标 `N, CA, C, O`**，优先研究、集成 **OpenFold** 的原子坐标表示或结构模块，并用真正受当前帧约束的主链重建方法，而不是使用一次静态序列折叠预测冒充多帧动力学；
3. 具有固定的自检脚本、回归测试、质量报告和基准实验；训练过程加入可验证的局部几何目标、开放循环 rollout 评价与训练监控，不能再仅靠 RMSD、Rg 作为主要成功信号；
4. 明确区分“结构几何通过”、“动力学连续性通过”、“统计力学/GLE 验证通过”三个等级，不得混为一谈。

### 工作纪律

- **不要先调参掩盖根因**。首先做第一步 rollout 的确定性消融与坐标映射 round-trip 测试。
- **不要把 OpenFold 当作 Cα 错误的补救工具**。必须先修好上游不合理的 Cα，再做主链重建。
- **不要只输出建议**。必须定位到仓库实际代码文件、函数、调用栈和数据流；提交最小可验证 patch，执行测试，并报告修改前后指标。
- **不要假设当前项目已有 OpenFold 的 Cα-conditioned 接口**。先审计官方 API/当前安装版本；没有就显式新增约束模块，或对比可复现的 Cα→backbone 基线。
- **确保输出时间语义**。`MODEL 1` 对应 `t=0`/Anchor，`MODEL 2` 对应第一次 rollout；新增文件元数据注明 `dt`、单位、温度、seed、模型版本及重建方法。

---

## 1. 已确认的实测缺陷

### 1.1 核心 QC（直接从上传 PDB 重新解析）

| 指标 | Anchor (t=0 / MODEL 1) | t=1 (MODEL 2) | t=32 | t=64 (MODEL 65) |
|---|---:|---:|---:|---:|
| Cα–Cα 相邻距离均值 (Å) | 3.694 | 2.371 | 2.332 | 2.655 |
| 相邻距离 `<3.0 Å` | 2/95 | 74/95 | 74/95 | 70/95 |
| 相邻距离 `>4.1 Å` | 0/95 | 4/95 | 4/95 | 10/95 |
| 相邻距离在 `3.6–4.0 Å` 的比例 | 85.3% | 8.4% | 7.4% | 8.4% |
| 回转半径 Rg (Å) | 14.257 | 13.919 | 14.008 | 13.941 |
| 刚体对齐到 Anchor 的 Cα RMSD (Å) | 0 | 2.263 | 2.153 | 2.754 |
| Anchor 的 `|i-j|≥4, d<8 Å` 接触保留率 | 100% | 61.2% | 59.7% | 56.8% |

**严重异常举例：**

- Anchor 的残基 **2–3** 相邻 Cα 仅 **1.306 Å**，表明起始帧已经有局部错误；需核对该位置是否经历裁剪、编号映射、缺失残基填充、坐标混合、尺度转换或原始数据损坏。
- t=1：74/95 个相邻 Cα 间距 `<3 Å`。**一步即失真**，排查重点应优先是表示/解码/更新公式，而不是长时累积误差。
- t=32：残基 **28–29** 相邻 Cα 距离仅 **0.062 Å**，原子几乎重合。
- t=64：残基 **63–64** 仅 **0.739 Å**；残基 **45–46** 达 **5.335 Å**。
- t=64：对 `|i-j|≥4` 的非局部残基对，有 2 对 Cα 距离 `<3 Å`（例如 39–43 为约 2.27 Å），提示潜在排斥体积问题。

**结论 A（高置信度）**：局部几何在第一步已严重失败，轨迹现阶段不是物理合理的蛋白质 Cα 主链。

**结论 B（高置信度）**：Rg 和 RMSD 单独不能判断轨迹合理性；它们稳定不等于局部键长、主链手性、二级结构或非键排斥正确。

### 1.2 有力但尚未确认的机制线索：沿序列方向过度平滑

定义 Anchor 的离散空间 Laplacian：

\[
L_i = \frac{\mathbf r_{i-1}^{(0)}+\mathbf r_{i+1}^{(0)}}{2}-\mathbf r_i^{(0)},\quad 2\le i\le N-1.
\]

将每帧刚体对齐后相对 Anchor 的位移投影到 `L`：

\[
\Delta \mathbf r_i^{(t)}\simeq \alpha_t L_i.
\]

实际可得到：

- t=1：`α≈0.833`，最小二乘残差解释后的 `R²≈0.778`；
- t=64：`α≈1.009`，`R²≈0.767`。

这表明观察到的位移有大约 **77% 的平方范数变化可由 Anchor 的单一空间平滑方向解释**；请注意该 `R²` 是这里定义的投影拟合度，**不是模型预测准确率**，也不能作为根因证据。

**优先排查以下假设（待代码验证）：**

1. 构象在 H-space 中只保留低频模态，逆变换/decoder 缺失局部高频几何；
2. 使用邻域平均、GraphConv/message passing、残差平滑或显式 Laplacian 操作，使螺旋波动被滤除；
3. Latent normalization / unnormalization、坐标均值恢复、尺度、模态 basis、质量权重或残差索引错误；
4. 模型更新的是“绝对坐标”还是“相对 Anchor 的增量”混淆；
5. PDB 输出时错误使用不同时间、残基或链的坐标数组，造成结构扭曲。

**结论 C（中等置信度）**：空间平滑高度可疑，但必须利用 ablation / round-trip / 中间张量输出确认，不能提前认定为唯一原因。

### 1.3 动力学时间相关性疑点（尚不能下物理结论）

去掉每帧整体平移，且从 t=1 后开始计算：

- lag=1 的平均平方位移 MSD≈**1.607 Å²**；
- lag=2 ≈**1.646 Å²**；lag=8 ≈**2.007 Å²**；lag=16 ≈**2.079 Å²**；
- 连续位移增量的一阶归一化相关约 **−0.489**。

负相关与早期快速饱和可能提示回摆、高频振荡、独立重采样或错误时间积分；但也可能受到输出帧间隔与强约束模态影响。**在没有 `dt`、积分器、变量定义及 MD 基准前，不应凭它断定 GLE 有误。**

### 1.4 元数据与序列一致性风险

- 文件只有链 A 的 **96 个残基，编号 2–97**。必须核查这是独立 96 aa 蛋白，还是从更大结构裁出的片段；如果是后者，缺失环境/端点边界会改变动力学。
- 核对 `1ka1A01` 标识对应的原始源结构、真实序列、链 ID、残基编号、插入码和 PDB 导出时 AA3 映射。**不要仅凭 PDB ID 推断结构身份。**
- 生成 `{source_file, source_chain, residue_id, aa, index, missing_mask}` 显式映射；要求每帧序列、残基顺序一致。

---

## NEW P0-A：必须用 **Ground-Truth latent → decoder** 识别表示/解码崩塌（最高优先级）

> 用户本轮新增的第一条**强制任务**。即刻在 **v3 真实运行的模型源码与 checkpoint** 上执行，**不要运行 GLE** 就先完成这里的实验。与只有 Cα 的生成 PDB 不同，此实验还需要来自训练/验证/测试集的真实 MD 结构及其 latent/encoder 接口；没有 GT 数据时必须报告 `BLOCKED: missing matched GT MD/latents`，绝不将生成 PDB 冒充 GT。

### A1. 不要把 “GT latent” 不加区分地称为 ground truth

建立下列对象的**精确定义**，防止将来源不同的向量混用：

- `X_GT[t]`：原始 MD 或可信构象数据中的真实坐标，包含序列、mask、单位、链映射和时间索引；源文件应经过几何 QC。
- `z_GT[t] = E_v3(X_GT[t], context)`：**把真实构象送入 v3 现有 encoder 得到的 latent**，这是 *encoded GT*，并不是信息完备的真值；encoder 可能已经把细节丢掉。
- `z_projected[t]`：若原项目的 H-space 是可分析的模态投影，可选记录在同一数学定义下的**无学习投影结果**及其逆变换，作为可控基线；不要默认它可逆。
- `z_roll[t]`：从 `z_GT[0]` 经 v3 GLE 递推得到的预测 latent，不可称之为 GT。
- `X_rec_GT[t] = D_v3(z_GT[t], context)`；`X_roll[t] = D_v3(z_roll[t], context)`。如果 decoder 输出的是 `ΔX` 而非绝对 `X`，要使用与训练一致的 Anchor/reference；若还需噪声、时间、温度或序列条件，要逐项传入。

必须核对训练与推理所用的 `encoder/decoder` 是**同一个 checkpoint、归一化参数、投影基底、参考坐标、mask 和 residue order**；不能混用缓存的 latent 或不同版本的 scaler。

### A2. 强制 5 组有判别力的实验（每组必须独立生成报告）

| ID | 输入 → 输出 | 检验目的 | 不可遗漏的结果 |
|---|---|---|---|
| **GT-0** | `X_GT → dataset preprocessing → X_seen_by_encoder` | 是否在进入 encoder 以前已错位、损坏、被平滑/截断 | 原始与预处理键长直方图、residue mapping、max difference、mask、单位 |
| **GT-1** | `X_GT → E_v3 → z_GT → D_v3 → X_rec_GT`，**完全绕过 GLE** | **最重要**：确定真实 latent 本身是否能够还原合法 Cα / 局部细节 | 重构 RMSD、Rg、局部 3.8Å 键长、伪角/伪扭转、clash、每残基误差、frequency spectrum、结构可视化 |
| **GT-2** | `X_GT → reference analytic projection/basis → inverse`（如果存在） | 判断理论 H-space 降维本身的可逆性上限，与可学习 decoder 区分 | 理论投影重建误差随 latent 维数/保留频率变化的曲线 |
| **GT-3** | `z_GT[0] → GLE → z_roll[t] → D_v3`，同时对照 `D_v3(z_GT[t])` | 在同样的 decoder 下严格分离 **latent dynamics** 误差与 **decode** 误差 | `z_roll-z_GT`（在可比定义与时间对齐下）、分布偏差、解码几何、first-invalid-step |
| **GT-4** | 直接构造真实数据的邻近 `z_GT` 插值、数据主成分扰动、温度/不同构象的条件化输入后 `D_v3` | 判断 decoder 是否对不同真实构象的变化不敏感 / 平均化 / OOD 失真 | decoded variance ratio、CA per-residue RMSF、响应曲线、有限差分敏感度、合法性 |

**GT-1 的每个时间帧必须 independently 编码真实 `X_GT[t]`**。不可把单个 `z_GT[0]` 当作整个时间序列的“GT latent”；若已有 GT latent 缓存，先选 20 个样本现场 re-encode 并逐位核对缓存。

建议至少覆盖 **train/val/held-out test，多蛋白、多真实 MD 帧、多不同构象**；同一蛋白时间相邻帧作为补充。严禁仅用单个 Anchor 的重建成功来证明不存在 collapse。报告数据量、相关性、采样方法与分组。

### A3. 将四种常被混淆的“坍塌”独立检验

1. **Representation collapse（encoder 坍塌）**：许多不同 `X_GT` 被映射为相同/近似 `z_GT`，或者 latent 协方差仅极少数方向有方差。输出每维方差、按蛋白/时间分组的 latent covariance spectrum、effective rank，以及构象差异与 latent 距离相关性。注意低秩**不自动**证明病理 collapse，必须和 GT 实际可变性、设计 latent 维数与数据复杂度对照。
2. **Decoder / posterior-mean collapse（解码平均化）**：真实 `z_GT` 足以区分构象，但 `D(z_GT)` 输出趋于平均结构并丢失局部二级结构/扭转自由度。输出 `Var_t(X_rec_GT)/Var_t(X_GT)`、逐残基 RMSF ratio、接触/角度分布、局部距离谱，以及不同 latent 扰动的 decoder 敏感度。
3. **Dynamical collapse（GLE 状态空间坍塌）**：`D(z_GT)` 良好，`D(z_roll)` 失败；检查 `z_roll` 是否发生均值回归、方差萎缩、非真实方向漂移、过强 damping / noise 或数值积分错。与真实 latent 的时间协方差、auto-correlation、转移分布对比。
4. **Spatial over-smoothing / projection bias（局部高频丢失）**：即使 latent 和全局 RMSD/Rg 看起来正常，沿蛋白序列的高频结构被编码器、basis truncation、message passing 或 decoder 滤掉；比较 `X_GT`、`X_rec_GT`、`X_roll` 的 Laplacian norm、螺旋局部半径、局部 CA tangent/curvature 以及空间频谱。不得仅凭一张 PCA 图断言 mode collapse。

推荐指标（定义清楚统计样本维度）：

\[
\text{rank}_{\rm eff}(Z)=\exp\!\left[-\sum_k p_k\log p_k\right],
\qquad p_k=\frac{\lambda_k}{\sum_j\lambda_j},
\]

\[
R_{\rm fluct}(i)=\frac{\mathrm{tr}\,\mathrm{Cov}_t(\widehat{\mathbf r}_i(t))}{\mathrm{tr}\,\mathrm{Cov}_t(\mathbf r_i^{GT}(t)) + \varepsilon}.
\]

报告 latent 谱、`GT→decode` 后的原子级局部几何、局部位移统计；样本要按统一参考框架做刚体对齐，防止平移/旋转污染协方差。**只画 latent t-SNE/UMAP 不构成 collapse 证据。**

### A4. 额外的识别力与可逆性测试（必须有定量输出）

- **Encoder sensitivity**：同一样本施加小幅物理合理构象扰动、残基局部旋转（或从真实 MD 中选邻近真实帧），检查 `E(X)` 的变化是否连续、是否完全不敏感；对输入整体刚体旋转/平移检查期望的不变/等变性质。
- **Decoder sensitivity**：沿真实数据的 latent 主成分和局部差分 `δz`，计算 `||D(z+εδz)-D(z)||/ε`、局部键长响应与重构精度；必要时使用 JVP/有限差分，避免构造巨大 Jacobian。
- **Information ceiling**：对于 PCA/ENM/H-space basis 截断，扫 latent 维度、空间频率与残差通道；绘制**重构可达上限**而非只增加神经网络容量。若不可逆，要明确是否应该把稳定几何部分放在 deterministic backbone base，再只预测动力学残差。
- **No-change baseline**：`D(E(X_GT))` 与复制 Anchor、低通/邻域平滑输入、线性逆投影和简单几何感知 decoder 同时比较，防止模型只学会“平均结构”。
- **Training/eval discrepancy**：评估 `train()` vs `eval()`、BatchNorm/LayerNorm、dropout、normalization、teacher-forcing、clamping、finite precision、PDB exporter 不一致。保存中间张量，定位第一个开始失真的步骤。

### A5. 将 GT-1 结果变成强制决策门（先审计，后精准修改代码）

| 观察到的结果 | 主要指向 | 必须优先采取的动作 |
|---|---|---|
| 原始 `X_GT` 几何就坏 | 数据/结构源/解析 | 修数据与 mapping，重建训练数据/缓存，重新训练 |
| `X_GT` 正常，但 `X_seen_by_encoder` 已坏 | 预处理 / 对齐 / 归一化 | 最小定位预处理 bug，修复并更新数据 pipeline，重新训练 |
| `E(X_GT)` 丧失关键变化，解析投影也失真 | **H-space / encoder 表示瓶颈** | 改 latent 表示、维数/基底或增加结构高频 residual，重新训练 |
| `E(X_GT)` 仍有差异，但 `D(E(X_GT))` 结构/变化坍塌 | **decoder / reconstruction objective** | 修 decoder、归一化、几何监督/训练目标，重新训练 |
| `D(E(X_GT))` 良好，但 `D(z_roll)` 很坏 | **GLE rollout / train-test latent distribution mismatch** | 查 drift/noise/memory/integrator、OOD latents、多步训练；修改后重新训练 |
| Model tensor 正常，写出 PDB 后错误 | exporter | 修 residue/atom mapping 与多 MODEL 输出；通过 exporter 测试，并视受影响训练数据范围决定重训 |
| 同时存在多处错误 | 复合根因 | 按第一个失真层顺序修复，各 patch 有单独单测和训练 ablation |

**必须提交 `GT_LATENT_DECODE_AUDIT.md`、机器可读 `gt_latent_audit.json/csv`、`X_GT vs X_rec_GT vs X_roll` 可视化，并将此决策表填成**有实际路径、函数、checkpoint、实验证据**的结论，而非猜测。

---

## NEW P0-B：审计并重设计 **v3 Encoder / H-space representation**，正式评估 OpenFold 几何建模（最高优先级）

> 用户本轮新增的第二条**强制任务**。不能因为“之前建议过 OpenFold”就假定当前 v3 已使用它；也不能仅仅安装 OpenFold 或从中导出一个静态 fold，就称已升级建模。Agent 必须从**当前真实代码**识别现有 encoder 的数学表示、输入张量、目标与训练梯度，并比较可信的几何表达方案。

### B1. 要求首先回答的 12 个源码级问题

1. v3 的 Encoder 实际是什么：PCA / normal modes / Laplacian eigenbasis / learned MLP、GNN、SE(3)-equivariant encoder、VAE / VQ / transformer，还是混合？给出真实类、函数、源码路径、commit，而不是凭名字猜测。
2. 输入是 `Cα xyz`、全部 `N/CA/C/O`、atom37、距离图、局部 torsions、时间窗、速度/动量，还是某种已处理的 H-state？逐项写出形状、单位、mask。
3. `z` 的维度、物理语义、与 GLE 方程中 `q`, `v`, `K`, `η` 的关系是什么？哪些表示变换是线性的、可逆的，哪些不可逆？
4. Encoder 是否对整体 `SE(3)` 变换具有合理的不变/等变性？是通过 Kabsch 对齐、相对距离、局部 frames，还是网络自动学习？残基重新排序或断链如何处理？
5. 使用什么坐标基准：每蛋白 Anchor、全局平均结构、每帧自对齐、独立中心化？解码时能否唯一恢复正确构象？是否错误地将每帧归一化/中心化后丢失局部信息？
6. 如果 `z` 是低频/低秩投影，真实二级结构的局部高频成分保留在哪里？有没有显式 `X_ref + ΔX`、rigid residue frames、torsion channel 或高频 residual branch？
7. 空间图的边如何定义：序列邻居、三维邻居、cutoff、动态图、接触图？是否重复邻域平均导致过平滑？
8. 训练 loss 的重构目标是 latent MSE、全局 Cartesian MSE、pair distance、torsion、FAPE，还是物理/统计损失？会不会促使网络以“平均构象”最小化损失？
9. 训练用全真值 latent 时 `D(E(X))` 的指标是多少？**和 GT-1 接口必须共用**，不能重造一套看似通过的单独指标。
10. 如果使用不同蛋白、不同温度，encoder 是否提供充分的 residue/sequence、chain topology 与 thermodynamic context？温度如何进入 dynamical model 与 embedding？
11. 训练时是否 `detach` encoder / 冻结 encoder / 先训后冻结，使 decoder/GLE 几何目标无法修正表示缺陷？提供真实 `requires_grad`、梯度 norm 与 optimizer parameter groups。
12. 是否已经依赖 OpenFold、AlphaFold 模块、`Rigid`、FAPE、atom14/37、局部 torsion 或 torsion-to-frame？如未采用，说明原方案与 OpenFold-based 表示的客观技术差异与引入成本。

### B2. 正确使用 OpenFold：借鉴 **结构表示与几何原语**，不是把静态预测误认为动力学编码

对当前安装或计划使用的版本，核验官方库中相关实现：

- `openfold/data/data_transforms.py`：`atom37_to_frames`, `atom37_to_torsion_angles` 等，把**真实全原子 GT** 转成残基局部刚体坐标系与二面角监督目标；这些函数需要所要求的原子与 mask，**不能假定只提供 Cα 即可无损恢复 GT torsion**。
- `openfold/model/structure_module.py`：残基 `Rigid` frames、torsion-to-frame / atom14 positional reconstruction 的思路，可作为**几何感知 decoder** 的先验或模块设计参考；不能盲目套用其主网络结构或假设 pretrained structure-module 能直接接受任意 H-space latent。
- `openfold/utils/feats.py` / `openfold/utils/rigid_utils.py` / `openfold/np/residue_constants.py`：atom14/atom37 mask 映射、刚体操作、氨基酸特异原子几何；按照实际 checkout 核验 API 与单位，使用版本固定的 import，避免魔法下标。
- `openfold/utils/loss.py`：FAPE、torsion loss 等理念可用于合理的**局部坐标系几何监督**；注意正确处理有效 mask、手性、断链、global SE(3)、数据噪声。

**尤其重要：`OpenFold trained sequence → folded structure` 不等于 `OpenFold features → frame-conditioned dynamics`。** 在 v3 上更可能有价值的是**借鉴 OpenFold 的结构几何表征与重建模块**，保留物理动力学模型；并非直接用 OpenFold 的静态输出替换 GLE 轨迹。集成必须有源码、版本、公式、测试、真实输入输出和科学目的解释。

### B3. 最低要求：公平比较四个 Encoder / Decoder 方案

| 方案 | 设计 | 主要假说 | 必须检验的点 |
|---|---|---|---|
| **R0 原始 v3** | 原生 Encoder、H-space、Decoder | 根因基线 | GT-1 重构、多样性、结构几何、计算成本 |
| **R1 v3 + bugfix / reconstruction loss** | 不改 H-space 物理主干，修归一化/映射，加入正确 decoder 几何监督 | 是否 loss/实现错误 | GT-1 改善，rollout 不退化；最小代码改动 |
| **R2 v3 + OpenFold-inspired geometry-aware encoder/decoder** | 真实 backbone 监督下引入 residue-local rigid frames / sin-cos torsion / atom-masks / pair geometry；保留 GLE 核心动力学目标 | 是否局部几何先验解决投影/解码低频化 | Cα、N/CA/C/O GT 重建；有效帧率；latent dynamics；多蛋白泛化 |
| **R3 v3 + invertible geometry-preserving structural residual** | 低频 H-state + 显式高频几何残差、内坐标或约束 decoder | 是否在小幅改动下恢复 H-space 的信息上限 | 解码误差对 latent 维数、结构细节、动力学指标的影响 |

`R2`、`R3` 是**候选研究方案**，不是事先认定正确。必须先做 GT-1、GT-2 的无 GLE 真实数据重构对比，选择技术上实际必要的重设计，再在 v3 上进行可对照的端到端重训练。OpenFold-inspired 结构先验不得把所有构象强制收敛到静态预测构象，也不得损害运动多样性。

### B4. 训练目标中应检验的表示质量，而非单靠 RMSD/Rg

无 GLE 的 AE/encoder–decoder 阶段至少记录：

\[
\mathcal L_{\rm recon}=\lambda_{CA}\mathcal L_{CA-pos}
+\lambda_{dist}\mathcal L_{local-dist}
+\lambda_{angle}\mathcal L_{pseudo-angle}
+\lambda_{tor}\mathcal L_{torsion}
+\lambda_{clash}\mathcal L_{exclusion},
\]

若有合法 `N/CA/C/O` GT，另考虑 **frame-aware / FAPE-like backbone positional loss、主链 torsion（`sin φ, cos φ, sin ψ, cos ψ, sin ω, cos ω`）、真实 atom-mask** 和化学几何约束；若仅 Cα 真值，不得编造 N/C/O 的 GT labels。

建议额外测量：

- `L_geom(D(E(X_GT)))` 的绝对值和梯度能否反向传到 Encoder / Decoder；单独对输入 GT latent 做 `L_geom(D(z_GT))` 并确认梯度作用位置。
- GT 多帧局部坐标分布、二面角的多模态性是否保持；不能通过输出单一理想角度使指标“漂亮”。
- GT-1 的主链 N/CA/C/O reconstruction 与 GT-3 的 free rollout 必须分别通过。**编码重构好≠动力学正确**。
- `SE(3)` 一致性、不同蛋白长度、residue mask/断链与真实空间近邻，作为可自动化单元测试。
- 若 H-space 是非线性的流形坐标，G​​LE 的随机噪声/摩擦核/热力学权重在新坐标下是否仍正确；不能只保留旧的 `K` 和 `noise` 而忽略坐标变换。

### B5. 本轮新增强制交付件

1. `GT_LATENT_DECODE_AUDIT.md`：GT-0..GT-4、训练/测试重构、四类 collapse 证据、同图 `GT/AE-recon/GLE-rollout`、最早出错层和根因判决。
2. `ENCODER_HSPACE_OPENFOLD_REVIEW.md`：真实 v3 representation/decoder 数学定义、现有源码模块、信息上限、OpenFold 接入与非接入情况、R0–R3 的取舍和实验依据。
3. `latent_diagnostics/`：可复现的执行脚本、config、编码后 GT latent、预测 latent、实际 decoded 结果、per-residue CSV、rank/variance/spectral/sensitivity 报告与图；如 GT/版权/数据体积受限，只提供合法的轻量产物与重新生成命令。
4. `root_cause_to_patch_table.md`：每个已证实根因→文件/函数/行→最小复现→patch commit→回归测试→重训练 experiment ID；不得只说“可能存在坍塌”。
5. 新增 **GT-decode checkpoint gate**：除非 `X_GT→E→D` 的局部几何与可变性经独立验证合格，否则**禁止把后续 GLE 的改善仅归因于动力学方程**，并把真实序列/多态构象重构纳入所有后续 v3 checkpoint 的 CI。

---

## 2. 根因排查顺序：必须按优先级执行

| 级别 | 模块 | 假设与待检验项 | 可复现的隔离实验 | 合格判据 |
|---|---|---|---|---|
| **P0-A** | **GT latent 无 GLE 解码 / collapse 四分类** | 独立验证 `D(E(X_GT))` 与真实 MD 的差异、局部结构和构象多样性 | GT-0..GT-4；latent covariance/rank，decoded bond/torsion，variance response | 真实 GT 的重构没有低通塌缩；明确编码、解码或 GLE 的首个错误层 |
| **P0-B** | **v3 Encoder/H-space + OpenFold-aware representation** | 表示是否丢失主链局部自由度；OpenFold 几何模块是否真实使用 | 12 项源码审计、GT-1/GT-2，R0-R3 结构表示对照 | 证明选择的表示保留所需几何并可被 GLE 一致地建模 |
| **P0** | Anchor/数据预处理 | 残基顺序、2–3 异常、链截断、坐标单位、mask、填充 | 原始结构→预处理→Anchor→PDB 分阶段逐项比较；比对原始 PDB/mmCIF | 序列、拓扑映射正确；与源结构等价或明确解释偏差 |
| **P0** | Decoder/逆变换 | H-space→Cartesian 抹除了高频几何 | **不运行 GLE**：encode(Anchor)→decode→与 Anchor 对比；对所有训练样本执行重建 | 无 GLE 时恢复链几何；误差分布透明 |
| **P0** | 第一步 GLE | 单步更新符号/系数/Δt/单位/积分错误 | 零更新、`dt=0`、关噪声、关 memory、关 drift、固定 seed、极小 `dt` 各自独立检查 | 零步严格保形；小步连续；找到损坏首先发生的位置 |
| **P0** | 空间平滑 | 消掉局部高频、局部螺旋塌陷 | 每个中间层测 `Cα–Cα` 分布、Laplacian 投影、谱分量；单独开/关平滑 | 指明哪一算子导致 3.8 Å 几何失效 |
| **P1** | 结构/物理约束 | 未对 CA 键长、角度、排斥体积建模 | 约束项和几何投影模块的消融；对比轨迹变化 | 不再出现塌缩/断链，且不抹平真实动力学 |
| **P1** | 时间积分 / GLE | FDT / memory kernel / noise 尺度不匹配 | 简化解析 GLE、OU 或谐振子 toy test；固定噪声序列逐步比对 | 实现与选定理论离散化一致；统计基准通过 |
| **P1** | PDB exporter | 坐标索引和单位出错 | tensor→PDB→tensor 的 round-trip，含多 MODEL 与首尾帧 | 误差仅来自 PDB 小数位截断；原子/帧映射正确 |
| **P2** | 多轨迹统计验证 | 单 seed 无法代表系综 | 多 seed、不同温度/间隔、参考 MD 评估 | 时间尺度、分布、统计量有可解释的一致性 |

**强制要求：每个假设均给出 `evidence / experiment / result / patch / regression_test`；如果尚无证据，标注 `UNVERIFIED`。**

---

## 3. 针对代码的具体修改建议

### 3.1 建立显式的中间状态观测（最优先，不改物理模型）

在现有代码中定位：

`source PDB/mmCIF → crop/map → Cα / full-atom anchor → encoder → H latent → GLE integrator → decoder → coordinates → postprocess → PDB writer`

建议每个关键节点可选输出 `.npz` / `.pt`，至少保存：

- `residue_index`, `chain_id`, `aatype`, `atom_mask`, `original_residue_id`；
- `anchor_ca`, `encoded_h`, `decoded_anchor_ca`, `h_t`, `decoded_ca_t`；
- `force/drift`, `noise`, `memory_integral`, `dt`, `temperature`, `seed`（若可用）；
- `ca_before_projection`, `ca_after_projection`, `backbone_before_relax`, `backbone_after_relax`。

要求统一 shape 与单位约定，并在代码入口断言。不能静默挤掉/广播 residue 维度，不能把 `Å` 与 `nm`、`K` 与无量纲温度、每帧步长和物理时间混用。

### 3.2 将几何检验变为必需的回归测试

在独立的 `geometry_qc` 模块实现以下质量统计，并按每帧/每残基输出 CSV 或 JSON：

1. 相邻 Cα 距离及分位数、越界比例；
2. Cα pseudo-bond angle、pseudo-dihedral，虚拟手性与二级结构保真度；
3. 非相邻 Cα pair clash（应排除正常邻近序列原子）；
4. 每帧刚体对齐的 RMSD、Rg、distance-map drift、contact retention；
5. 轨迹跨帧增量分布、极端跳变、非有限数；
6. 若有 MD 参考，保留 `reference` 与 `predicted` 的严格一致采样时间/单位/序列映射。

**核心工程门槛（可根据真实基准校准，并非普适热力学定理）：**

- 每个有效帧必须无 NaN/Inf、无错误链编号；
- 相邻 Cα `<3.0 Å` 或 `>4.2 Å` 计数必须为 **0**（除明确标注的结构缺失/非连续链边界）；
- 相邻 Cα 在 `3.6–4.0 Å` 比例建议 **≥95%**，且必须与高质量参考的实际分布校准；
- Cα 过近的非局部残基碰撞需要为 0 或给出有化学依据的例外；
- `t=0` 通过初始输入/导出 round-trip，一阶变化随 `dt→0` 收敛为 0；
- 不得因为硬投影导致动力学分布严重失真；要测量修复前后的位移、MSD、VACF 和位置分布。

> 重要：残基对是否为链上相邻要由真实链拓扑判断，**不能只用数组中相邻**。PDB 片段边界、缺失残基和断链必须特殊处理。

### 3.3 如需加约束，要区分“局部物理几何”与“真正的动力学力场”

可以实验性添加柔性约束：

\[
E_{CA\text{-}bond} = \frac{k_b}{2}\sum_{i \in \mathcal B}(\lVert r_{i+1}-r_i\rVert-d_{i,i+1}^{ref})^2,
\]

配合 pseudo-angle、torsion、steric/excluded-volume。这里 `𝔅` 是真实连续链连接集合，`d_ref` 应校准到真实几何（一般接近 3.8 Å），不应任意跨缺失片段连键。

**但如果 GLE 在 latent H-space 演化，直接把 Cartesian 修正力塞进原 GLE 方程可能破坏其物理一致性。** 需要说明：

- 几何由表示/decoder 保证？还是由显式势能保证？
- 修复算子是否可微？是否需要把其 Jacobian 映射回 H-space？
- 做几何投影是否改变平衡分布、噪声-耗散关系和动力学轨迹？

推荐先定位和修复表示/解码缺陷，再比较：**无投影、软约束、每步约束投影、内坐标或局部 SE(3) 表示** 四种方案。优先考虑从结构表示层确保有效拓扑，而不是在错误 Cartesian 输出上强行事后拉伸。

---

## 4. **新增核心产品需求：不仅 Cα，还必须通过 OpenFold 技术路线得到主链 N–Cα–C–O**

### 4.1 预期输出格式（明确验收）

对于每一帧 `t=0..64` 和每个存在的残基 `i`，至少提供以下 **4 个主链重原子**：

\[
X^{(t)}_{i,\mathrm{backbone}}=\{\mathbf r_{i,N},\mathbf r_{i,CA},\mathbf r_{i,C},\mathbf r_{i,O}\}.
\]

必选输出：

- `trajectory_backbone.pdb` 或 `.cif`：65 帧 `MODEL`，`N, CA, C, O` 的 atom name、残基名、残基编号和链编号完整正确；
- `trajectory_backbone.npz`：建议 shape `[T=65, N_res=96, 4, 3]`，附 mask、sequence、frame index、物理时长、单位、reconstruction mode；
- `backbone_qc.csv/json`：每帧主链共价几何、CA 一致性、冲突率和手性；
- 采用 OpenFold 的路径应有 `openfold_version/checkpoint/config`, 采用辅助程序亦须说明来源与参数；
- 如暂时只得 `N, CA, C`，则**不视为完成**，因为要求包含 **O**。

### 4.2 关键技术边界：OpenFold ≠ 默认的 Cα-conditioned 动力学重建器

OpenFold 官方模型可输出 `final_atom_positions`（atom37 表示，含 N、CA、C、O 等原子），但它的标准推理任务是根据氨基酸序列、MSA/模板等**预测蛋白结构**，并没有一个保证“以任意用户提供的 65 帧 Cα 坐标为硬约束，逐帧补全 N/C/O”的现成接口。

所以必须验证以下三个不同任务，不能偷换概念：

- **Sequence → structure prediction**：输入序列，仅获取静态 3D 预测；可作为结构先验/Anchor 候选，**不能自动代表 GLE 的逐帧真实构象**。
- **Backbone reconstruction (CA → N, C, O)**：要求每帧 **严格/高精度保留该帧 Cα**，补全剩余主链坐标。
- **OpenFold-informed conditional reconstruction**：借助 OpenFold 的结构模块、残基刚体 frames、原子几何先验或经验证的条件输入/约束优化，在各个有效帧条件下重建，而非独立运行 65 次无约束折叠。

### 4.3 推荐分阶段实施（必须逐阶段测试）

**A. 恢复并保存真实的 Anchor 全主链（最快、最可信的起点）**

- 如果原始源 PDB/mmCIF 本来含 `N, CA, C, O`，直接从相同残基 ID、相同 `altloc`、相同坐标系导出 Anchor 主链；不要对 t=0 再做无意义预测。
- 检查 Anchor 当前 2–3 问题是否由预处理引入；**在这一步修复之后再做后续重建**。

**B. 建立独立、可比较的 Cα→全主链几何重建基线**

- 建议实现 **PULCHRA** 或等价 Cα backmapping 作为可复现 baseline；它专门用于从 Cα trace 重建主链/原子结构，但对于畸形 Cα 坐标也无法保证产生合理真实构象。
- 在真实高质量结构上人工只保留 Cα，再重建全主链；比较 backbone RMSD、N–CA / CA–C / C–N / C=O 的几何误差、CA 漂移、计算时间。
- **不要仅凭输出了 PDB 文件就视为成功**。

**C. 集成 OpenFold 作为结构先验/原子坐标生成器（核心拓展）**

至少验证两种实现路线，实测后择优：

1. **OpenFold structural prior + constrained refinement**：使用 OpenFold 预测的 Anchor / 局部结构先验，构造每帧主链几何变量，并用 CA 坐标约束、肽键几何、扭转角和立体约束进行优化。若使用 N/C/O 参数化，在优化时严格保持目标 CA；可采用残基局部刚体框架/内坐标，而不是直接自由拟合全部笛卡尔坐标。
2. **OpenFold representation / structure-module adaptation**：若工程上可行，研究让 OpenFold 结构模块显式条件化于每帧 Cα 和序列的方式（包括输入特征、CA-only 模板 mask 的可行性和分布外问题），在已有预训练权重基础上做约束推理或微调。**不能假设官方 `run_pretrained_openfold.py` 已支持任意 CA-only PDB 无损补全**。需要代码级证据和实际 benchmark。

在两路线中，将目标明确写为：

\[
\underset{X_{BB}}{\mathrm{argmin}}\quad E_{\mathrm{peptide\ geometry}}(X_{BB})
+\lambda_{CA} E_{CA\text{-}restraint}(X_{BB},X_{CA}^{(t)})
+\lambda_{prior} E_{OpenFold\text{-}prior}(X_{BB})
+\lambda_{temp} E_{temporal}(X_{BB}^{(t)},X_{BB}^{(t-1)}).
\]

上式是**研究性设计目标，不是现成 OpenFold API 或已证明的物理能量形式**。如果采用硬 CA 约束而非软惩罚，`E_CA-restraint` 应被替换为等式约束；如引入时序正则，必须验证不抹去真实动态。

**D. 输出逐帧主链而非重复的静态结构**

- 检查生成的每一帧 `CA` 与 GLE 输出该帧 `CA` 一一对应，除非明确记录了受约束投影的移动量；
- 检查主链原子的时序连续性与各步局部几何；
- 如果模型将不同 Cα 帧重建成几乎完全相同的静态主链，**判定失败**，即使 OpenFold 预测单帧高质量也不算实现动力学。

### 4.4 主链几何验收（每一帧）

- `N–CA`、`CA–C`、相邻残基 `C–N`、`C=O` 键长处于合理肽链几何范围；常见参考量级约 `1.46 / 1.53 / 1.33 / 1.23 Å`，应对照可靠结构库和具体化学类型设定容差。
- 验证主链键角和 `ω` 二面角，检查 cis/trans（尤其 Pro）、手性、链的连续性、原子碰撞；使用可靠的结构质量工具，不要只用键长平均数。
- 在几何合格的输入帧上，建议目标 `CA preservation RMSD ≤ 0.1 Å`，并报告每残基最大偏移；如算法需要改变 CA，应记录修正前后位置和分布影响，不能偷偷覆盖原有 rollout。
- 随时间的 N/C/O 构象必须响应对应的 CA 变化，不能将 t=0 主链直接复制到所有帧；也不能凭几何合格就声称动力学正确。

### 4.5 推荐代码模块边界

```text
protein_dynamics/
  geometry/
    residue_mapping.py            # residue / chain / gap map
    ca_metrics.py                 # per-frame & per-residue CA QC
    backbone_metrics.py           # N,CA,C,O geometry QC
    constraints.py                # geometry energy / projections
  rollout/
    gle_integrator.py             # GLE time stepping
    latent_decoder.py             # H -> Cartesian and round-trip audit
  reconstruction/
    base.py                       # interface: reconstruct_frame(sequence, ca, metadata)
    pulchra_baseline.py           # CA backmapping baseline
    openfold_prior.py             # OpenFold model loading/inference, validated API
    conditional_backbone.py       # CA-constrained mainline reconstruction
    temporal_consistency.py       # optional temporal-aware refinement
  io/
    trajectory_writer.py         # correct 65-model backbone PDB/mmCIF
  tests/
    test_anchor_roundtrip.py
    test_decoder_identity.py
    test_gle_zero_step.py
    test_ca_geometry.py
    test_backbone_geometry.py
    test_backbone_preserves_ca.py
    test_trajectory_multimodel_io.py
    test_temperature_and_seed_reproducibility.py
```

目录仅为**模块职责建议**。Agent 必须先看仓库已有组织，尽量小范围改动，避免为了“整理目录”进行高风险重构。

---

## 5. OpenFold / 主链恢复的额外审计要求

具体到版本和调用方式，Agent 需要回答并提供证据：

1. 当前仓库是否安装 OpenFold？版本、commit、checkpoint 来源、GPU/依赖能否复现？
2. 现有系统是否已经有全主链 Anchor，而只是 PDB writer 输出 Cα？如果是，应先恢复 exporter，避免额外引入估计误差。
3. OpenFold 模型 `final_atom_positions` 与 `final_atom_mask` 如何提取？atom37 原子索引 **从当前安装版本的 residue constants 获取，禁止魔法数字硬编码**。
4. 如果采用 OpenFold 作为 frame-specific 先验，**Cα 条件是如何强制施加的**？未加约束独立预测不可称 CA→backbone 重建。
5. 65 帧重建耗时、峰值显存、是否重复昂贵地运行 OpenFold trunk、是否可以缓存序列特征和结构先验？
6. 重建方法是否保留 GLE 生成的构象变化，还是强行把它们拉回某个平均/最稳定结构？请给跨帧 RMSD、CA retention、主链分布的量化结果。

**方案选择约束**：即使暂时发现 OpenFold 不适合做任意 Cα 条件补全，也必须实现可靠的 `N/CA/C/O` 基线，并在最终报告中清楚说明 OpenFold 的已集成部分、尚待开发的条件化部分，**不能假装完成**。

---

## 6. GLE 统计物理正确性：放在结构修复之后验证

对适用的**欠阻尼** GLE，可用下式作为核对物理语义的参考：

\[
m\ddot q(t) = -\nabla F(q(t)) - \int_0^t K(t-s)\dot q(s)\,ds + \eta(t),\]

\[
\langle\eta(t)\eta(s)^T\rangle=k_B T K(|t-s|),
\]

其中第二式仅在相应平衡、噪声定义及核对称约定下成立；如果项目采用过阻尼 GLE、H-space 投影、非平衡驱动或带辅助变量的近似离散化，应写出**项目真正实现的方程和离散化**，而非机械套用上述公式。

待核对的具体项目：

- `dt`、时间单位、质量/摩擦/噪声幅度、`k_B T` 的单位与数值转换；
- memory kernel 的正定性/数值稳定性、离散卷积积分、核与噪声协方差关系；
- Markovian embedding / 辅助变量的采样与更新顺序；
- 显式/隐式积分器、稳定区域和步长收敛；
- 目标稳态分布、温度对构象统计的影响、短时/长时 MSD、VACF（速度存在且定义可靠时）、模态相关函数和 drift；
- 与参考 MD 在匹配的物理时间分辨率和体系条件下对照；
- 多 seed 统计及 bootstrap 置信区间，不能仅展示一条轨迹的视觉效果。

**时间相关性测试的层级**：

1. 最小解析 toy（例如广义 Langevin 谐振子）
2. 高质量初始构象上的短轨迹
3. 匹配条件 MD 的结构/时间统计
4. 多温度泛化（例如不仅 320 K）

---

## 7. 必须执行的 ablation & failure localization

| 实验 ID | 实验 | 目的 | 预期输出 |
|---|---|---|---|
| **E0** | 源结构→Anchor→PDB round-trip | 区分源数据与导出错误；隔离 2–3 异常 | 逐原子最大差值、映射检查 |
| **E1** | `decode(encode(anchor))`，无 rollout | 判断信息是否在 latent 表示/解码阶段丢失 | per-residue error、CA bond dist、Laplacian fit |
| **E2** | `dt=0`, drift=0, noise=0, memory=0 | 排除非物理的“恒等更新仍改变结构” | 坐标差、最大差值、失败层 |
| **E3** | 单独启用 drift / noise / memory，固定 seed | 定位第一步骤变来自哪项 | t=1 每一项增量/几何指标 |
| **E4** | 极小 `dt` 扫描，步长收敛 | 检查单位和积分器 | log-log 或表格收敛数据 |
| **E5** | 去掉/替换各层空间邻域操作 | 识别过平滑源头 | projection `α`,`R²`、谱衰减 |
| **E6** | 同一合理 Cα→PULCHRA vs OpenFold-conditioned | 选择主链恢复路线 | CA 保留、backbone 质量、耗时 |
| **E7** | 65 帧主链重建 + 逐帧 QC | 检验多帧真正可用 | 完整 PDB、报告和直方图 |
| **E8** | 多 seed、多温度、MD 参照 | 检验结构质量外的动力学/统计物理 | 分布、时间相关性、区间估计 |

**顺序改为：GT-0→GT-1/GT-2→GT-4→Encoder 源码审计→E0/E1→GT-3/E2→E3/E4→E5→结构/训练修复→v3 全量重训→E6/E7→E8。** 不允许通过后期 OpenFold/PULCHRA 后处理掩盖 E0–E5 未通过的事实。


---

## 7.5 v3 训练代码的专项审计：为什么全局 RMSD / Rg 好、局部几何却失败？

### 7.5.1 审计 **实际参与反向传播** 的损失，而不是只看配置名称

逐行定位 v3 中的 `train_step`, `forward`, `loss`, `decoder`, `H-space basis`, `noise/memory`, `rollout`, `optimizer.step`。输出一个 `TRAINING_OBJECTIVE_AUDIT.md`，至少包含：

| 审计对象 | 必须回答的证据问题 |
|---|---|
| 所有 loss 项 | 公式、代码位置、计算空间（Cartesian/H-space/距离图/隐变量）、单位和 reduction；具体权重和日志 |
| 梯度有效性 | 几何 loss 是否真的计算，是否在 `detach()` / `no_grad()` / `.item()` 后断梯度；从 CA 局部 loss 到 encoder/decoder/GLE 参数的梯度是否非零 |
| loss 适用的时间点 | 仅用于静态 frame、one-step、teacher forcing？还是覆盖训练 rollout 的每一个预测帧？ |
| 表示可逆性 | `decode(encode(x))` 在训练集与测试集上相邻 CA 和 torsion 是否保真；是否只拟合 low-pass 结构 |
| 几何与宏观指标冲突 | 训练日志中总 loss、RMSD、Rg 改善时，短键率是否恶化；是否出现优化目标与物理有效性不一致 |
| 数据标签与边界 | MD/结构数据有无缺失残基、链断裂、序列错配、PDB vs nm 单位、mask 与排序错位 |
| 训练/推理不一致 | train mode、dropout、noise、teacher forcing、解码器 normalization、采样间隔、temperature 是否与 inference 不一致 |

要求保留至少 20 个关键 batch 的单步梯度 norm 与 decoded CA 几何质量，形成**从 loss 到坏 PDB 的因果链**。

### 7.5.2 几何约束应在 **decoder 输出的 Cα / backbone 空间** 真正生效

建议在不损害 v3 核心物理建模的前提下比较以下可微辅助损失。这里给的是可检验设计，**不是已验证的最优权重**：

\[
\mathcal L_{\rm geom}=\lambda_b\mathcal L_{\rm bond}+
\lambda_a\mathcal L_{\rm angle}+
\lambda_\tau\mathcal L_{\rm torsion}+
\lambda_{\rm ev}\mathcal L_{\rm excl}+
\lambda_\chi\mathcal L_{\rm chiral}.
\]

- `bond`: 对真实连续链边上 `||r_i-r_{i+1}||-d_i^{ref}` 加稳健 penalty；参考值一般约 3.8 Å，但需根据数据的实际残基/坐标分布校准，不要对跨断链 pair 施加约束。
- `angle/torsion`: 约束 Cα 虚拟键角、虚拟二面角及其条件分布；**不要把所有残基强迫到同一参考角/扭转角**，更不能把所有 MD 帧固定到 Anchor 构象。
- `excluded volume`: 排除局部链上合法邻近关系，惩罚非键重叠；采用可微的截断或 softplus 势并检查梯度尺度。
- `chirality`: 比对二级结构/四体 Cα 手性符号及其合理分布，防止 mirror inversion。
- 若有高质量 full backbone 标签，还可训练或监督 backbone-aware decoder；但不能在只有 Cα 数据时凭空编造 N/C/O 真值。

**关键原则**：loss 直接作用于 **经过 H-space → Cartesian decoder 还原的生成坐标**，且确认非零梯度传到 v3 需要更新的参数。不要只在输出到 PDB 之前做不可微 geometry-fix，也不要把几何能量和真实物理势能混为一谈。

### 7.5.3 加入多步 rollout 训练，解决 train–inference mismatch

除了 one-step loss，加入**短到长、多步自由 rollout 的训练或验证机制**：

\[
\mathcal L_{\rm train}=\mathcal L_{\rm v3\_original}
+\lambda_g\,\mathbb E_{t\in\mathcal T}\big[\mathcal L_{\rm geom}(\widehat X_t)\big]
+\lambda_m\mathcal L_{\rm multistep}
+\lambda_s\mathcal L_{\rm statistics}.
\]

训练实现需处理：

1. **Teacher-forced one-step 与 free-running rollout 两条支路分别计分**，同时记录训练和测试指标；
2. 首先在 `t=0,1,2,4,8` 训练/评估局部几何，再渐进增加到 `16,32,64`；如果反向传播跨 64 步过昂贵，可使用截断 BPTT、分段 rollout 或随机训练 horizon，并说明梯度覆盖范围，不能假装 full BPTT；
3. 在每个被监督的 rollout 帧，监控 `bond violation`, `clash`, `chirality`, `secondary-structure`, `RMSD`, `Rg`, `contacts`，权重应通过验证集选定；
4. 对动力学模型，**不要直接以逐帧 Cartesian MSE 作为唯一监督**：不同噪声下真实随机动力学轨迹不必逐点匹配，优先使用有定义的转移统计、条件分布、时间相关、结构合法性和适当物理一致性；
5. 不能用极强几何投影使所有输出变成静态 Anchor 或因强行降低运动幅度而“通过”几何检查；需要同步监督跨帧非平凡运动。

### 7.5.4 可考虑 **结构天然有效的表示**，避免后验惩罚救火

必须对比：A 原 v3 H-space + loss；B 几何约束的 decoder；C 局部刚体/内坐标参数化；D 温和几何投影（仅作 ablation），报告同等 compute / 数据量下的差异。合理的主链表示可通过固定骨架键长和键角、动态二面角构象实现几何有效性，但仍需处理顺反肽键、局部环与多残基冲突。若改变 H-space 引入新的质量矩阵或坐标变换，核对 GLE 的 Jacobian、噪声协方差与统计权重是否仍有理论依据。

**保持科学目标**：改进后应更正确地建模真实动力学，而不仅仅把 Ca–Ca 距离钉在 3.8 Å。

---

## 7.6 必须执行：在 v3 SOTA 工程上修改代码并完整重新训练

### 阶段 T0：冻结基线、完整复现

- 定位 v3 **实际运行**的代码分支、commit SHA、checkpoint hash、模型配置与训练入口；标记它为 `baseline_v3_immutable`。不预设 v3 训练路径名称，必须用源码和训练日志确认。
- 固定原始数据、train/val/test split（按蛋白/序列同源性及轨迹来源合理防泄漏）、时间采样和 seed；保存依赖、GPU、CUDA、PyTorch、MD 数据处理版本。
- 用 **当前原始 checkpoint** 复现实验结果；若原始训练不可复现，报告原因并先修复复现问题。
- 基线不仅要报告 RMSD、Rg、原论文/项目 SOTA 指标，还要增加 `bad CA fraction`, `chain breaks`, `clash`, `pseudo-angle`, `secondary-structure`, `rollout collapse time`，按多个种子/蛋白统计。

### 阶段 T1：根因测试 + minimal patch（在 v3 原代码上）

- 在 `GT-0→GT-4` 与 `E0→E5` 得到最小反例、失败最早一层与因果证据后，优先最小改动修复表示、单位、索引、训练目标、梯度流或 GLE 实现；能独立复现的 bug 各自单独 commit。
- **针对已确认的原因**修改 v3 训练代码；如 loss 不足，实际加入可微几何约束并正确 mask 断链；如 decoder 不可逆，修表示或 decoder；如单位/解码错误，优先 fix bug。不要通过任意加大 `lambda_b` 掩盖训练数据的根因。
- 为每一个 patch 增添自动化单测（forward/backward、gradcheck/gradient norm、identity、one-step、multi-step、export round-trip）。
- 先对小数据/小模型 overfit 检查：可否重建合法结构？随机初始化时、无噪声时、完全无 GLE 时是否仍会塌缩？

### 阶段 T1.5：明确 GT latent / Encoder / OpenFold 结论并立即提交对应修复

- **GT latent 无 GLE 解码试验是进入全量训练前的硬性关卡**：先报告 `D(E(X_GT))` 是否可还原有效且多样的结构；如没有 GT 数据，标 `BLOCKED` 并提出需要的实际源数据路径，不能用同一条生成 PDB 假装 GT。
- 按 NEW P0-A/B 的决策表落地 patch；若是 Encoder/H-space 信息上限问题，允许重设计几何感知编码或残差表示，并对照 R0-R3；若是 GLE 问题，不应无理由更换 encoder。
- 在小规模 GT 构象上先证明重构几何、构象变化、latent 敏感性和梯度路径正常。**一旦根因确定，及时修改训练代码并启动完整 v3 重训**，不允许“只做建议不执行”。

### 阶段 T2：**完整重训练整个 v3 核心模型（硬性任务）**

- 新实验使用**v3 架构及修补后的完整数据和训练程序**；对所有参与 v3 生成轨迹的 trainable modules 端到端更新（按真实工程模块清单列明）。
- 正式主试验必须从可记录的 fresh v3 初始化重新训练，训练到事先规定的停止准则/与 baseline 可比的预算。可另外跑热启动以快速筛选，但**不得用旧 checkpoint 的小范围微调代替最终全模型重训**。
- 保留清晰的 checkpoint family：`v3_original`, `v3_bugfix_only_retrained`, `v3_geom_retrained`，若资源允许增添 `v3_geom_multistep`、`v3_representation_variant`，避免同时改了多项却无法归因。
- 在同一 split / comparable GPU-hours / objective 下比较 baseline 和新模型；记录 model selection 是按 val 而不是 test 做的，避免 cherry-picking seeds 或单条轨迹。
- 必需实验日志：配置 YAML、Git commit、训练 seed、样本量、optimizer/LR schedule、batch、grad scaler、epoch/steps、GPU-hours、显存峰值、完整训练曲线和权重 hash；确认 checkpoint 可以独立 reload 并重复生成相同 seeded rollout。
- **不能中途只跑若干步就声称已重训练**。若无法获取资源，实际结果标注 `NOT RUN / BLOCKED`，不得生成假日志/表格。

### 阶段 T3：新 checkpoint 的 held-out 轨迹物理验收

- 同序列、同 seed、同 320K、同 64 步、同 `dt` 生成**新的** 65-frame rollout；附原 v3 和 v3 retrained 的文件 SHA-256，确保不同训练 run 确实被调用且没有缓存旧 PDB。
- 将结构 QC 从单条扩展到测试集多个蛋白（至少报告覆盖数）、多 seed、温度。测试 horizon 包括 1, 8, 16, 32, 64，并可扩展长 rollout。
- **Must-not-regress**：RMSD/Rg 原优势不能因无运动/强弹簧而人为维持；要对比 MD 统计分布、各 residue RMSF、contact map、角度/扭转分布、MSD/autocorrelation，具备时间单位时再检验 kinetic metrics。
- **新增核心进步指标**：`valid frames %`, `short CA neighbor pairs %`, `max CA bond deviation`, `nonlocal clash count`, `helix/strand integrity`, `first invalid step`, `backbone valid frame %`，这些和旧指标平等且为 release blocker。
- 统计按每条轨迹/每个蛋白聚合，报告均值、分位数、置信区间和失败案例，不得只展示最好的一条。

### 阶段 T4：OpenFold-informed 完整主链

- 在 **经过几何 Gate 的新训练模型轨迹**上做 frame-conditioned `N/CA/C/O` 补全。原始 backbone Anchor 若存在，应直接保留；OpenFold 模型可作为预训练 prior，不强求从头重训 OpenFold 本身（除非自行证明必要）。
- 明确 OpenFold 的作用和条件重建方法：无约束 `sequence→structure` **不等于** `Cα(t)→backbone(t)`；逐帧 `CA` 保留与 backbone stereochemistry 必须独立验收。
- 如选择训练一个与 v3 联合的 backbone decoder，则把它纳入 T2 全模型训练、损失和 checkpoint 记录；如选择可验证的 post-hoc backmapping，则与 v3 物理轨迹训练明确分开。

### 训练对照矩阵（最小必须交付）

| 实验 | v3 实现 | retrain | 几何 loss | multi-step | 意义 |
|---|---|---|---|---|---|
| B0 | 原始 v3 | 原始 checkpoint | 原有设定 | 原有设定 | 原始 SOTA baseline |
| B1 | 修 bug 后的 v3 | **全模型** | 原有设定 | 原有设定 | 分离纯 bugfix 的收益 |
| B2 | B1 + 几何/表示修正 | **全模型** | 是 | 至少监控 | 明确检验局部几何约束是否带来增益 |
| B3 | B2 + 多步一致性训练 | **全模型** | 是 | 是 | 检验 rollout 稳定性及训练–推理失配 |
| **B4（若 GT-1/GT-2 证实有表示瓶颈）** | v3 + OpenFold-inspired geometry-aware encoder/decoder 或显式 structural residual | **全模型** | 含 backbone/几何监督 | 按根因选择 | 验证几何表示创新是否真实优于 B2，且不损伤运动多样性 |

如果根因确认并非 loss 缺失，必须在实验结论中说明哪些分支没有合理适用性；仍需完成原 v3 与修复后 v3 **公平的全模型重训练对比**。

---

## 8. 定义完成的硬性验收 Gate

### Gate A0：**GT latent 解码 + Encoder 建模有效性（NEW P0，必须通过）**

- [ ] 从真实 MD 多帧 GT 得到 `z_GT=E_v3(X_GT)`，绕开 GLE 的 `D(z_GT)` 实验完成；报告 raw/encoded/decoded 的 Cα 距离、torsion、clash、RMSD/Rg、RMSF 与多样性。
- [ ] Representation / decoder / dynamical / over-smoothing 四类 collapse 已分别用定量证据区分；失真最早层可定位到源码函数/行/张量。
- [ ] 原始 v3 Encoder/H-space 的数学定义、数据流、损失/梯度、截断/平滑、SE(3) 性质和 OpenFold 使用状态已审计。
- [ ] 已按根因实现源码修复与最小回归测试；如发现编码信息瓶颈，完成几何感知方案（含 OpenFold-inspired 对照）与 GT 数据重建验证。
- [ ] 新 checkpoint 不仅在 GLE rollout 上表现合理，还必须在 GT latent 解码门中表现合格；绝不将 GT loss / 几何合格与正确动力学混为一谈。

### Gate A：结构输入与第一步恢复（P0，必须通过）

- [ ] 原始序列、原子顺序、链拓扑、裁剪边界、温度、`dt` 和单位检查通过；异常的 Anchor 2–3 问题已定位或修复。
- [ ] Encoder/decoder identity 和 PDB round-trip 测试通过，或明确证明不满足的原因并修复。
- [ ] `dt=0` 与全零更新时输出严格等于输入（容差按内部浮点精度）；对于真实生成帧，主链 Cα 不再大规模失真。
- [ ] 逐帧的 CA 键长、角度、非局部碰撞 QC 报告生成；严重几何损坏为 0。

### Gate B：全主链输出（P1，必须通过）

- [ ] 每帧每个有效残基均有 `N`, `CA`, `C`, `O` 坐标与有效 mask。
- [ ] 主链键长、键角、二面角、手性、肽键连接、clash 通过已定义的结构 QC。
- [ ] OpenFold 使用方式明确：**结构先验、主链重建、还是经验证的条件推理**；不存在伪造“OpenFold 直接把 65 帧 Cα 转全原子”的描述。
- [ ] Frame-specific CA 保留误差、主链跨帧变化、计算成本均已报告；未把各帧压回同一个静态 fold。
- [ ] 输出真实可读的 65-model `N/CA/C/O` PDB/mmCIF 与机器可读 NPZ。

### Gate C：科学动力学验证（P2，单独标注）

- [ ] toy GLE/离散化验证通过；热浴、memory/noise 条件与假设一致。
- [ ] 多 seed 与 MD 参考下的静态分布和时间相关性质合理，并报告失败案例。
- [ ] 无法检验的物理条件标为 `NOT EVALUATED`，而不是 `PASSED`。


### Gate D：v3 源代码修复与 **full retraining**（P0，不可跳过）

- [ ] 提交了当前 **v3** 的真实代码路径、commit、训练入口、数据版控、original baseline checkpoint checksum。
- [ ] 有 `ROOT_CAUSE.md`：说明故障的最早来源、可独立复现的消融，以及排除其他备选解释的证据；不把猜测当根因。
- [ ] 有 **训练 loss / gradient audit**；修补后的几何约束（若使用）确实施加在 decoded rollout 输出并可以反传。
- [ ] 在 v3 原工程修改代码并提交 patch + regression tests；有 bugfix-only 与 geometry-aware 的对照（不可适用项须解释）。
- [ ] **完整重训练 v3 核心模型**：提供 fresh initialization、完整 run log、训练曲线、checkpoint hash、compute 和可复现实验命令；仅局部微调或修 exporter **FAIL**。
- [ ] 新 v3 checkpoint 在多蛋白、多 seed 的 held-out 轨迹上几何 QC 通过；对 RMSD、Rg 等原指标报告 no-regression，对时间动力学做合条件的独立检验。
- [ ] 新 PDB SHA-256 **与本次重复上传的旧轨迹不同**；对应最新 commit/checkpoint/seed，无读取缓存老轨迹的情况。

**Release policy:** Gate A、B、D 为必要交付；Gate C 中尚缺数据的项目应标为 `NOT EVALUATED`，不能声称动力学物理正确。下游 protein design/科学结论不得基于未通过 Gate A 的轨迹。

---

## 9. Agent 最终必须返回的内容

### A. Root-Cause Analysis（**必须包含 NEW P0-A/P0-B**）

逐项写出：`代码路径 → 函数/行 → 错误机制 → 最小证据 → 因果排除实验 → 置信度`。如果未找到根因，不得提交“已修复”。

### B. Patch Set

提供每个 commit/diff 的修改目的、受影响 API、向后兼容性以及潜在性能/科学副作用。尽量让 anchor、decoder、GLE、reconstruction 分别可以独立测试。

### C. Reproducible Results

- 修改前/后同一序列、Anchor、seed、温度、rollout 步数、物理 `dt` 的对照；
- 关键表格：Cα 邻接距离异常比例、clash、RMSD、Rg、contact、谱过平滑指标；
- 主链回填：CA preservation、`N–CA/C–N/CA–C/C–O` 键长/角度质量、帧连续性；
- 如无 MD 对照，明确标记“仅通过几何检验，未完成动力学验证”。

### D. Deliverables

```text
reports/
  ROOT_CAUSE.md
  CODE_CHANGES.md
  GLE_ROLLOUT_QC_BEFORE_AFTER.md
  V3_TRAINING_OBJECTIVE_AUDIT.md
  V3_RETRAIN_EXPERIMENTS.md
  V3_TRAINING_LOGS_AND_CONFIGS.md
  OPENFOLD_BACKBONE_INTEGRATION.md
  LIMITATIONS.md
artifacts/
  trajectory_ca_fixed.pdb
  v3_original_rollout.pdb
  v3_retrained_rollout.pdb
  new_v3_checkpoint_sha256.txt
  trajectory_backbone.pdb
  trajectory_backbone.npz
  ca_qc.csv
  backbone_qc.csv
  ablations.csv
  train_curves.csv
  full_retrain_comparison.csv
  qc_figures/
tests/
  <相关单测和集成测试>
```

上述文件名称是建议；若仓库已有约定，应沿用该约定。无法完成的交付项必须列出原因、实验阻碍和最短下一步，**不能用空文件冒充**。

---

## 10. 核心工程判断与优先决策

**当前最需要排查的不是 GLE 的 64-step 长时稳定性，而是第一个 step 为什么把 3.8 Å 左右的局部几何一下变成约 2.37 Å。** `t=1` 出现 74/95 对相邻 Cα 过短，暗示表示/逆变换/单步更新/坐标导出的确定性问题，应优先解决。

**当前最有信息量的测试是 `decode(encode(Anchor))`。** 如果在零动力学时已经出现约同样的 Laplacian 空间平滑，则问题很可能位于 H-space 表示的可逆性、模态截断或 decoder；如果纯重建保真但第一步失真，则进一步检查 integrator、dt、噪声和 memory。若 latent tensor 合理而导出 PDB 才出现错误，则优先修 exporter。

**OpenFold 主链输出是独立但强相关的新增需求**：要从 Cα-only 升级为逐帧完整 backbone，但不能让静态 folding 代替动力学，也不能让 backmapping 掩盖 upstream Cα 错误。最终希望得到经过验证的、几何合理的多帧 `N/CA/C/O` 轨迹；更高层次的真实 GLE 动力学须另外用参考 MD 与统计力学检验。

---

## 本轮新查阅的 OpenFold 官方实现入口（Agent 必须固定 checkout 并自行核验函数签名）

- https://github.com/aqlaboratory/openfold/blob/main/openfold/data/data_transforms.py — atom37 → residue frames / torsion angles；必须有有效 backbone atoms/mask。
- https://github.com/aqlaboratory/openfold/blob/main/openfold/model/structure_module.py — structure module、local frames 和原子坐标生成逻辑。
- https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/rigid_utils.py — `Rigid` / rotation / frame primitives。
- https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/feats.py — atom14/atom37 mapping 与相关特征。
- https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/loss.py — FAPE / torsion 等几何损失实现。

以上是可选的库内工具/理论参考，不是已证明与当前 v3 兼容的即插即用解决方案。原训练环境中没有 OpenFold 的话，先运行现有 v3 的可复现基线；比较新旧编码器时固定数据、seed、算力预算与 held-out 测试。

## 参考实现/官方材料

- OpenFold repository / model output (`final_atom_positions`, atom37): <https://github.com/aqlaboratory/openfold/blob/main/openfold/model/model.py>
- OpenFold official inference documentation（序列、MSA、模板、保存模型输出）: <https://openfold.readthedocs.io/en/latest/Inference.html>
- OpenFold backbone-violation loss implementation（肽键几何参考）: <https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/loss.py>
- PULCHRA Cα-to-full-atom reconstruction baseline: <https://github.com/euplotes/pulchra>

> **请 Agent 在执行时校验这些接口的实际版本、license 与当前项目环境。不能因为技术名词匹配就认为 OpenFold 原生支持任意 Cα 条件主链补全。**

---

## 可直接复制给 Coding Agent 的执行指令（强制执行版）

> **新增最高优先级，必须先做**：读取当前项目真实 `v3 SOTA` 训练仓库与 checkpoint；从可信真实 MD 帧重新 `encode` 得到 `GT latent`，关闭 GLE，原样 `decode(z_GT)` 检查局部几何、结构多样性和空间频率，四分类排除 encoder/decoder/dynamics/spatial collapse；完整审计现有 encoder/H-space 表示，以及此前是否实际采用 OpenFold 的 residue frames、torsion/atom37/FAPE 相关模块。如未采纳，结合 GT 重构证据比较 OpenFold-inspired geometry-aware encoder/decoder 与保留局部高频的 structural residual。**确定根因后必须及时修改 v3 代码、执行回归测试、从可记录初始化重新训练 entire v3 model，而非只微调 decoder 或修 PDB 后处理**，随后交付 held-out 65 帧 Cα 和逐帧 `N,CA,C,O` 的结构/统计物理 QC。不得声称已运行未实际完成的训练。


> 你是项目 v3 SOTA 模型的研发/训练 Agent。请以当前仓库中 v3 的**实际训练、推理及 GLE rollout 代码**作为唯一主线，审计该模型的 geometry collapse。首先冻结 v3 baseline 的源码/权重/配置，复现 65 帧轨迹。用户本次上传的 PDB 与上次**逐字节相同**，不代表任何改善；第一步 74/95 个 Cα 邻接距离 `<3 Å`，最后一步 70/95，RMSD 和 Rg 虽不崩溃仍然明显不合格。
>
> **你必须定位训练及推理链路的具体根因，而不是停留在猜想。** E0–E5 排查原始结构与 Anchor、encoder/decoder 往返、H-space 表示/空间平滑、GLE 单步、noise/memory/dt、PDB exporter；重点审计 v3 训练损失与其梯度路径是否遗漏了局部几何、角度、手性、clash、multi-step open-loop 约束，是否仅优化了 RMSD/Rg 或 teacher forcing 指标。对每一项给代码路径、证据、排除性实验、回归测试。
>
> **找到根因后立即在 v3 代码上实施 patch，而不只是交方案。修改训练机制与相关模块，将经过验证且合理的可微几何约束作用于 decoder/rollout 输出，保证多步真实生成质量；不要盲加 loss，也不能用不可微后处理掩盖模型缺陷。** 保留原版 v3 作为对照。
>
> **随后必须完整重新训练整个 v3 核心模型**，从记录清楚的 fresh 初始化启动正式 training run；允许短程 warm-start 只作为 debug，不得用旧 checkpoint 局部微调替代。至少做 original v3、bugfix-only full retrain、geometry-aware full retrain 的对照；检验 multi-step 训练收益。保存训练配置、完整日志、训练曲线、GPU-hours、代码 commit、checkpoint 哈希，并实测新权重的 65 帧/多 seed/held-out 测试集，逐帧比较 Cα 几何、RMSD、Rg、接触、二级结构、动力学统计，确认没有用“失去运动”换来几何合法性。验证新轨迹 hash 与旧版不同且没有读到缓存。
>
> 最终输出**不止 Cα**，还要逐帧得到完整主链 `N, CA, C, O`：优先恢复原始真实 Anchor 主链，调研并集成 OpenFold-informed 的结构/原子位置先验，结合严格的 frame-specific Cα 约束完成 backbone reconstruction，并与独立 backmapping baseline 比较。OpenFold 的普通静态序列折叠**不可以**冒充逐帧 Cα 条件重建。报告完整的主链几何、CA 保真和跨帧时序验证。
>
> **最终交付**：根因报告、v3 代码 diff、单测/消融、全量训练日志和新 checkpoint、baseline vs retrain 数值表、重新生成的 PDB/NPZ 与逐帧主链 QC。若缺 GPU/原数据/完整仓库导致无法完成，不得声称训练或修复已完成，必须给出准确阻塞原因和已经实际执行的命令/证据。
