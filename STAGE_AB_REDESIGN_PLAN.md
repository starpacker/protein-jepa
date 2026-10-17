# Protein Dynamics World Model：两阶段训练计划（Stage A + Stage B）

版本：2026-10-13-v3。性质：经设计审查修订的实施计划，尚未完成新方案训练验证。目标：在 4 ns～96 ns 上同时交付 direct 终点预测与 autonomous 轨迹生成，通过前置表示校准、小规模端到端验收和固定数据协议，降低正式训练返工风险；不承诺未经实验即可保证一次训练成功。

**本文件的执行范围：** 本轮按本文件 v3 实施。`STAGE_ABC_INCREMENTAL_PLAN.md` 是另一条保留旧架构的路线，不与本文件混用模块命名、冻结规则或验收标准。

**本版相对上一版的核心变更：**
1. Stage A 对齐 JEPA-Anything 的 OPF 核心；蛋白几何适配、mask、网络和超参数明确为本项目设计。
2. Stage C 合并进 Stage B。B1 内部先做 B1a 表示校准，再做 B1b 动力学学习；B2 完成共享解码器与双模式交付。
3. 仍使用单帧 `H_t ∈ [B,L,dH]` 作为状态；B1a 允许受控解冻 encoder，并前置真实 H 的结构解码验证，之后冻结表示坐标再训练 NLL。
4. direct 预测器与 autonomous 核分别训练、分别评分，共用 H 规范和结构解码器；direct 不能仅保留未训练的 eval-only 头。
5. 修正噪声协方差、标量能量梯度、滤波时间索引、完整耦合稳定性和结构对齐协议；保留无 force floor、无 softplus-NLL、无逐步 detach 的主线。

---

## 0. 总体架构一览

```
Stage A（JEPA-Anything 核心 + 蛋白适配）
  输入：多帧蛋白质几何特征序列  X_{1..K}  和未来目标帧  X_{t+τ}
  输出：每残基 token 历史序列   H_{1..K} ∈ [B, K, L, dH]
  目标：通过 JEPA/OPF 自监督，使 H 保留几何信息与可预测的时间演化

Stage B（动力学 + 结构解码，Stage C 已并入）
  B1a：表示校准
    先冻结 A，训练 oracle 解码器；再小学习率解冻 encoder 顶层/adapter
    用当前几何与固定物理未来目标校准 H，通过后冻结 encoder 和 H 归一化
  B1b：动力学学习
    输入：H_{≤t}（因果历史），H_t 直接作为当前状态（替代 Q_t）
    模型：有限记忆浴 (H_t, S_t) → (H_{t+1}, S_{t+1})，S 为辅助记忆变量
    目标：Kalman 创新 NLL warm-up，随后加入自主多步分布评分
    输出：训练好的动力学核（drift/diffusion 参数、浴后验网络）

  B2：共享结构解码 + direct 预测
    解码器：D_χ(Q:=H_t, anchor, c) → X_t（backbone 坐标），由 B1a oracle 训练初始化
      分解：X̂ = anchor_frame ⊗ Δ_local(Q:=H_t, c)，anchor 只作等变刚体基
      对比线（诊断）：D_χ'(Q:=H_t, S_t, anchor, c)，仅多一个 S 入口
    direct：G(H_{≤t}, τ, T, c, z) → Ĥ_{t+τ}（单帧）→ 共享解码器
    autonomous：B1b 从 (H_t,S_t) 逐步生成 H → 共享解码器（递归 rollout 出轨迹）

部署：mode=direct 返回所请求 horizon 的终点样本；mode=autonomous 返回完整轨迹
```

---

## 1. Stage A：JEPA-Anything 核心与蛋白适配

### 1.1 总体设计原则

对齐 JEPA-Anything（`Gen-Verse/JEPA-Anything@c6e6c88`）的 teacher、投影梯度、OPF 块正交和方差正则语义。公开 core 接受领域 encoder，不提供本项目完整的蛋白动力学训练方案；下述几何特征、图网络、时序 predictor、mask 比例、归一化及权重均为本项目适配，需实验验收。依据见 `analysis_notes/jepa_anything_audit.md`；不得把“对齐核心”写成已有端到端效果保证。

### 1.2 输入特征构造（SE(3) 不变几何）

每帧蛋白质结构提取以下逐残基特征，合并为 token `x_i ∈ R^{d_in}`：

**局部几何（每残基）：**
- backbone 二面角 φ,ψ,ω 的 sin/cos，共 6 维
- Cβ 距离分箱嵌入（近邻 k_nn=16 个残基），32 维；仅 backbone 数据时由 N/CA/C 构造 virtual Cβ，并标明为输入特征，Gly 采用同一虚拟点约定，不冒充真实侧链标签

**成对几何（每残基 × 近邻）：**
- CA-CA 距离（除以 20 Å 缩放，不将超过 20 Å 的距离截断）
- 相对局部框架方向：将邻居 CA 位置投影到当前残基 N/CA/C 定义的局部坐标系，取 3 维向量
- 带符号的 Cα-Cα 键向量相对 torsion（保留手性）

以上特征经过逐通道 z-score 归一化（统计量在训练集计算，推理时固定）。
不使用绝对坐标、全局旋转或平移信息（SE(3) 不变性）。
镜像区分依赖带符号 torsion 和相对框架方向，不能只用距离。

### 1.3 Encoder（在线 + EMA teacher）

**结构：** 图 Transformer（GVP-Transformer 变体），输入逐残基 token，输出每残基隐向量 `h_i ∈ R^{dH}`，dH=128（起始值）。

**两个 encoder：**
- 在线 encoder `f_θ`：正常训练，参数由梯度更新。
- EMA teacher encoder `f_θ̄`：参数为在线 encoder 的指数移动平均，`θ̄ ← m·θ̄ + (1-m)·θ`，m 从 0.996 线性增大到 0.9999（本项目起始 schedule）。teacher 始终处于 `eval()` 模式，`no_grad()`；每次 optimizer.step() 后执行一次 EMA 更新，包括浮点 buffers，非浮点 buffers 复制。

**关键合约（必须通过的单元测试）：**
- teacher 在相同输入下连续两次调用输出完全相同（确认 eval/no-grad 状态）。
- teacher 只在显式 EMA update 时改变；不能因前向调用更新 batch statistics。
- teacher 不参与 autograd 图。

### 1.4 OPF Predictor（按 JEPA-Anything 实现）

**Orthogonal Predictive Factorization（OPF）：** 学习 J 组 analysis rows `P_j ∈ R^{r×dH}`，要求 `J·r=dH`。各块低维，拼接后是完整坐标变换；软正交惩罚不保证每一步严格正交，也不自动产生独立物理模态。

设 `h̄ = f_θ̄(X_{t+τ})`（目标帧的 teacher 编码），OPF target 为：

```
y_j = P_j · stop_grad(h̄),    y_j ∈ R^{r}
```

注意：stop_grad 放在 teacher encoder 输出处，**不对** `P_j` 的输出做整体 detach——投影有自己的 target-side 梯度路径（与 JEPA-Anything 官方设计一致）。

Predictor `g_ω`（因果 Transformer）接受上下文 H_{≤t} 和时间间隔 τ、温度 T、序列 c，输出预测：

```
ŷ_j = g_ω(H_{≤t}, τ, T, c)_j,    ŷ_j ∈ R^{L×r}
```

预测损失（仅对有效 target 位置计算）：

```
L_pred = Mean_{valid j,k} (ŷ_{j,k} - y_{j,k})²
```

正交性损失（按 JEPA-Anything 块投影计数）：

```
L_orth = Σ_j ||P_j P_j^T - I_r||_F² + Σ_{i<j} ||P_i P_j^T||_F²
```

**起始配置：** J=4，r=32，历史 K=8，预测 lag τ ∈ {1,2,4,8} 个观测间隔，稳定后加入 {16,32}。

### 1.5 Mask 策略（同帧空间 + 因果未来，各占一半采样）

**空间掩码（同帧）：**
- mask 比例 40%，连续残基片段与空间邻域各半
- 被 mask 的残基：其坐标、边特征、由其坐标产生的可见侧特征**同步屏蔽**；邻居选择/边存在性也不能暴露隐藏几何，须重建可见图或使用与隐藏坐标无关的拓扑
- 编码器在 mask 输入上运行，预测 teacher 对完整帧的编码

**时间掩码（因果未来）：**
- context 帧 H_{1..K} 可见，预测目标帧 H_{t+τ} 的 teacher 表示
- 目标是**目标单帧**的 teacher 编码 `h̄_{t+τ}`，不是未来窗口均值，不是整段历史的摘要
- predictor 接受相对时间间隔和真实 dt_ps，不以轨迹绝对帧号作为预测线索

### 1.6 总损失与训练细节

```
L_A = L_future + 0.5·L_masked + λ_o·L_orth/dH² + λ_f·V(y) + λ_h·V(H)
```

其中方差惩罚 `V(z) = mean_d max(0, γ - sqrt(Var_valid(z_d) + ε))`，γ=0.1，λ_o=λ_f=λ_h=0.01（起始值）。有效样本的 population variance，不把 padding 或 mask token 算进方差。**不对每个 H 做单位 L2 归一化。**

各项按实际有效标量数归一化。记录每项对 encoder 的梯度量级；若相差两个数量级，先检查尺度与 mask，再调权重。

**优化器：** AdamW，lr=3e-4，grad clip=1，余弦退火。

**配置字段消费要求：** 所有配置字段必须有消费记录。未使用的 loss 权重应报错，不允许日志记录了某项 loss 而总目标中没有加它。

### 1.7 Stage A 验收 gate

冻结 encoder，在固定 held-out split 上评估：

1. 每蛋白、每温度、每 lag 的 latent persistence skill（有效帧对才计算比值）。
2. H 的有效秩与同蛋白同残基的时间变化统计（防止只编码身份标签）。
3. 用小型固定读出器（在 train 集拟合，held-out 集评分）预测内部距离、带符号 torsion、接触变化。
4. 对照：打乱过去帧顺序、移除历史、只给末帧；历史是否有用由数据决定。

**对照 baseline：** 普通 JEPA（无 OPF）同等容量配置，用于验证 OPF 的增量贡献。

**导出产物：** `f_θ`（在线 encoder checkpoint）、几何特征规范（含归一化统计量）、mask 规范、时间单位（ps）、训练数据版本与 hash。Stage A 的小 loss 只说明对 teacher 表示的预测准确度，不能直接证明完整动力学充分性。

---

## 2. Stage B1：动力学学习（H_t 直接作为状态）

### 2.0.0 统一几何分解（本节适用于 B1a/B1b/B2 全部解码通路）

所有"把隐状态兑现成结构"的通路，统一按**不变量 × 等变基**分解：

```
X̂ = anchor_frame ⊗ Δ_local(Q_t , c , [S_t])
    └── 等变刚体基 ──┘   └────── SE(3) 不变量 ──────┘

- Δ_local：每残基在其 anchor 局部坐标系中的内部位移，SE(3) 不变。
  H 只对这一项负责 —— 这是 H 唯一的构象任务。
- anchor_frame：由最后观测帧的 N/CA/C 三原子定义的刚体基（R_anchor [L,3,3],
  t_anchor [L,3]），SE(3) 等变。它只提供**绝对位置**，不提供任何内部构象信息。
- c：序列不变量（aatype / residue_index / chain_id）。
```

**分工契约（强约束）：**
- 内部构象信息**只能**来自不变量 `(Q, c, [S])`；anchor 只当旋转/平移基，绝不允许被用于恢复内部构象。
- anchor 是**已知量**（当前帧观测），不是需要额外求解的未知量；因此模块内部**不做 Kabsch 对齐**去寻找 anchor——anchor 已经给定。
- 训练时 loss 直接监督 `Δ_local`（anchor-local 内部位移）；Kabsch 全局对齐只作为**诊断读数**，不作主损失。

**与旧表述的区别：** 旧文档 §3.1/§11 写"解码器不接 anchor"。该表述有误：绝对坐标在定义上就必须有一个刚体基，否则无从摆放。精确表述是"**anchor 不得提供内部构象，只可作为等变刚体基**"（见 §11 红线修订）。

### 2.0 B1a：先校准 H，再固定动力学坐标

**决定：允许并预设一个受控微调阶段，但不让 encoder 与 latent NLL 一起自由优化。** JEPA 的可预测表示不保证完整可解码；“H 比 Q 信息多”也不能证明 H 已保留任务所需信息。微调只能帮助 encoder 保留输入已有的几何/预测信息，无法恢复输入中不存在的速度、侧链或溶剂信息。

1. **冻结基线：** 固定 A，先训练 §3 的 oracle 解码器 `D_χ(Q:=H_t, anchor, c)`，并训练轻量历史预测头预测固定物理目标 `O(X_{t+τ})`。O 包含距离、带符号 torsion 和接触；当前重构与未来预测分别记录。
2. **受控解冻：** 从同一 A checkpoint 出发，优先只解冻顶层 1–2 个 block 或残差 adapter，底层先冻结。encoder lr 从 `1e-5` 起步，读出器/解码器 lr 从 `1e-4` 起步。此阶段不训练动力学 NLL。
3. **目标：** `L_cal = L_oracle + λ_pred L_phys_future + λ_anchor L_anchor`，其中 **`L_oracle` 的主项是 anchor-local 内部位移损失** `‖Δ̂_local − Δ_local‖`（`Δ_local` 由 `_kabsch_target_delta_local` 生成：target CA 云先对齐到 anchor CA 云，再在 anchor 局部帧中表达，只保留内部变化、不含全局刚体位移）。Kabsch 绝对坐标 MSE 降级为诊断项，权重远低于 `L_disp`。`L_phys_future` 对真实物理 observable 监督：连续目标用标准化 MSE、接触用 BCE，按 lag 均衡；它只用于表示校准。`L_anchor` 对固定 A checkpoint 的 H 做弱蒸馏，尺度由训练集固定统计量定义；不要求微调 H 完全复制旧 H。起始 `λ_pred=1, λ_anchor=0.1`，以实际梯度和验证指标选定。
4. **H 信息量的干净测量（必报）：** 对同一 held-out 帧，分别用 **真实 H_t / 零 H / 错时刻 H** 喂 decoder，报告同一 `Δ_local` 指标。只有当 **真实 H 显著优于零 H** 时，才算"H 被用上"；否则该 decoder 通路对 H 不敏感，任何后续结论无效。禁止用 same-frame/target-anchor 等退化设置（零初始化残差在 step 0 即精确复现目标、与 H 无关）冒充 H 的可解码性。
5. **选择：** 在相同数据和读出器预算下比较冻结与微调版本，以固定物理目标上的 oracle 重构、动态构象区分、多 lag 预测和有效秩选择。微调无收益则采用冻结版本；不因增加训练步骤而默认微调获胜。
6. **冻结交接：** 在 B1b 前必须通过 §3.5 的 oracle 与构象替换检查。选定 encoder 后，拟合并冻结 train-only H 的逐通道中心/尺度（尺度有下限），统一用于历史、未来标签、direct、sampler 和 decoder。重新生成 H 缓存，固定 decoder/probe 作为动力学评分工具，记录 encoder/normalization/schema hash。

**NLL 的边界：** 若自由缩放 `H'=sH` 并相应重参数化密度，每步 NLL 可变化 `d·log(s)`，并不代表物理预测改善。EMA、stop-grad target 或自由训练的 decoder 单独都不能消除这一问题。B1b 中 encoder 和归一化必须冻结；不同 encoder 版本的 raw latent NLL 不直接排名，统一比较固定物理目标。后续若再解冻，须重新做本阶段并重训/校准全部依赖模块。

### 2.1 设计动机：为什么用 H_t 直接替代 Q_t

原方案中 `Q_t = g_φ(h_t)` 是一个额外训练的瞬时坐标读出器，目的是将 H 压缩为更低维的构象坐标并赋予几何语义。这一步骤带来的好处（维度压缩、几何解释性）同时也带来了额外的训练阶段、几何监督标签构造和 schema 版本管理负担。

用 `H_t ∈ R^{L×dH}` 直接替代 `Q_t` 的理由：
- H_t 是否保留所需几何和时间变化，由 A probe 与 B1a oracle 解码共同检查。
- 对确定性 Q=g(H)，H 不会比该 Q 少信息；但高维、冗余 H 不一定更容易拟合动力学。
- 省去独立 Q 状态模块，仍保留 B1a 校准与结构读出验收。
- 动力学损失（Kalman 创新 NLL）在 H 空间仍然有明确的数学定义。

**符号与历史遗留（重要）：** 新框架**不再有独立的 Q 读出器 `Q_t=g_φ(h_t)`**。`Q` 这个符号一律指"被特化为当前帧隐表示的动力学状态"，即 **`Q_t := H_t` 是同一个张量**，只是沿用了旧 `StructureDecoder` 签名里 `q_dynamics` 插槽的位置。因此：
- decoder 的旧「context 通路(`h_frozen`)」与「dynamics 通路(`q_dynamics`=Q)」**合并为单一 H 通路**（H 只喂一次，避免同一信息喂两遍造成容量不对称）。
- 旧框架的 `h_frozen` 独立 context 旁路**取消**：历史/context 信息改由 `S` 承载（见 §2.3 与 §3.7），不再用"整段历史 H"旁路恢复构象。

**代价：** H_t 维度（L×dH）远高于原 Q_t（L×8），Kalman 滤波每步成本更高，需要逐残基独立浴假设（块对角协方差）以保持计算可行。若后续发现跨残基创新有强相关，可升级为全局浴或降维 H。

128 通道的确定性几何表示可能集中在低维流形附近，正对角扩散会向该流形外加噪声。NLL 是指定环境空间高斯模型的拟合目标，不保证真实 H 有满维密度。pilot 必须检查小噪声扰动及多步 H 的可解码性、动态协方差谱和创新相关；若失败，先考虑固定压缩坐标/结构化扩散，并更新推导，不能只增大 dH 或让 decoder 掩盖漂移。对 `V=R+B P Bᵀ` 使用对角加低秩求解（Woodbury/行列式引理，基于 Cholesky），避免逐步构造全蛋白稠密协方差；与小矩阵稠密参考数值核对。

### 2.2 状态空间定义

**当前状态：** `H_t ∈ R^{L×dH}`，由 B1a 选定 encoder 与归一化规范对观测帧编码得到：

```
H_t = stop_grad(normalize_H(f_θ*(X_t)))
```

**记忆辅助变量：** `S_t ∈ R^{L×dS}`，每残基独立的辅助浴变量，初始值由历史后验网络给出：

```
(m_0, chol_P_0) = r_ψ(H_{≤0}, T, c)
S_0 ~ N(m_0, P_0),    P_0 = chol_P_0 · chol_P_0^T + ε·I
```

`dS` 起始取 8，与 {0, 4, 16} 做消融对照。

**离散转移核（首版：静态条件，无 H 依赖）：**

为避免 Itô 散度项，首版 D、C、Γ 仅依赖静态条件（序列 c、温度 T），不随 H_t 变化：

```
f(H;T,c) = -∇_H U_η(H;T,c)
H_{t+1} = H_t + D·f(H_t; T,c) · h + C·S_t · h + √(2D·h) · ε_H
S_{t+1} = S_t + (C^T·f(H_t; T,c) - Γ·S_t) · h + √(2Γ·h) · ε_S
```

其中 h = Δobs（观测间隔），`U_η` 为全蛋白标量无量纲能量，f 通过 autograd 取其负梯度；不得将任意向量网络输出称为能量梯度。D、Γ 为正对角矩阵，C 为每残基矩形矩阵（`dH × dS`），两组标准高斯噪声相互独立。此符号下 `∇U·CS + SᵀCᵀf = 0`。

**能量网络 U_η(H_t; T,c)：** 输入当前 H 与静态条件，输出标量；H 虽来自冻结 encoder，计算能量力时仍需对 H 求导，并保留训练所需二阶导数。首版使用序列决定的边与全残基 attention，不以初始构象决定静态能量图。若采用初始 CA 图，必须显式加入条件 `G_0`，并将模型解释为起点图条件过程；不能继续声称只依赖 `(T,c)` 的共同势能。温度 `1/T` 只是输入特征，不自动建立跨温度热力学一致性。连续模型的候选 Gibbs 结构、真实 MD 分布及 Euler 离散稳态分别验证。

**稳定性条件：** `h·λ_max(Γ) < 2` 仅约束孤立浴，不能保证完整 H/S Euler 系统稳定。对局部线性模型检查完整漂移 Jacobian J 的 `ρ(I+hJ)<1`；非线性模型还必须做自主长程测试、能量/状态范数与离开训练支持的诊断。能量可用正二次尾势加有界平滑残差提供恢复趋势，但仍需检查离散步长稳定性，不用 force floor。

反例：`U=H²/2, D=Γ=1, C=10` 时 `J=[[-1,10],[-10,-1]]`，Euler 需 `h<2/101`，远强于 `h<2`。如果实际 Δobs 下核不合格，应在 pilot 阶段调整模型/参数约束，或明确升级到含潜在子步的训练方法；不得只在部署偷偷减小 h。非线性且中间 H 未观测时，普通 Kalman 不能给出这些子步的精确边缘似然。

### 2.3 历史后验网络

`r_ψ(H_{≤0}, dt_{≤0}, T, c) → (m_0, chol_P_0)`

结构：因果 Transformer 或 GRU，输入历史 H token 序列和时间间隔，逐残基输出后验参数。chol_P_0 对角元素经 softplus 保证正值；初始均值置零，P 初值 I，C 初始小非零。

**重要：** 推理时从后验 N(m_0, P_0) 采样 S_0，不能仅用 m_0 替代全部不确定性。

### 2.4 Kalman 创新 NLL（主训练损失）

设观测序列 H_{0:N} 由真实帧编码得到（已停梯度），初始浴后验 (m_0, P_0) 由历史网络给出。对每步创新：

```
μ_{n+1} = H_n + D·f(H_n;T,c)·h + C·m_n·h
e_n     = H_{n+1} - μ_{n+1}
B_h     = h·C
V_n     = R + B_h·P_n·B_h^T
```

其中 **`R=R_h=2hD`**，是 H 转移噪声协方差，与 sample_step 完全相同，由 D 和 h 导出，不另设独立可学习 R。这里真实帧编码 H 视为被精确观测的状态；若另加测量噪声，H 本身成为潜变量且非线性力的处理会改变，不能直接把额外噪声塞进 R 后继续称本滤波精确。

主损失：

```
L_B1 = (1/2) · Σ_{n=0}^{N-1} [ e_n^T · V_n^{-1} · e_n + log det V_n + d·log(2π) ]
```

d 为有效 H 总维数（L×dH，不含 padding 残基）。连续密度 NLL 可以为负，**不使用 softplus 对 NLL 做截断**。

每步更新浴后验（Joseph 协方差更新保持正定性）：

```
K_n     = P_n · B_h^T · V_n^{-1}
m_n^+   = m_n + K_n · e_n
P_n^+   = (I - K_n·B_h)·P_n·(I - K_n·B_h)^T + K_n·R·K_n^T
m_{n+1} = A·m_n^+ + u_n
P_{n+1} = A·P_n^+·A^T + W
```

`A=I-hΓ`，`u_n=hCᵀf(H_n;T,c)`，`W=2hΓ`。条件于已观测 H 序列，S 的递推是线性高斯，因此上述边缘化对这个指定离散核精确，不代表它等于真实 MD 的转移律。使用 Cholesky solve/logdet，不显式求逆。记录 jitter 触发次数；若频繁触发，检查步长和协方差路径。

**梯度路径：** 梯度经整段滤波递推回到能量网络 U、参数 D、C、Γ 和历史后验网络 r_ψ；R 通过 D 求导。B1b 的 encoder 与归一化冻结。

**时间索引：** 历史为 `H_{-K+1:0}`，r_ψ 输出 `S_0` 后验；似然只评分 `H_{1:N}|H_{-K+1:0}`。实现中分开传 history 与 `[H_0,H_1,...,H_N]`。禁止用序列前 K 帧初始化 S_0 后又从同一序列第一帧开始评分，这会把被评分的未来用于初始化。

**单元测试（必须通过）：**
- 小型已知线性 Q/S 系统：创新似然等于直接联合 Gaussian 边缘化结果，误差 < 1e-5。
- sample_step 的经验均值/协方差与该核的解析公式相符（M=10000 个样本，10% 相对误差内）。
- 训练路径和部署路径共用完全相同的核参数、时间步长和噪声初始化。

### 2.5 默认课程：NLL warm-up 后加入多步 Energy Score

B1b 首先单独训练 Kalman NLL 直至数值与创新统计稳定；随后默认加入小权重的自主多步条件 energy score，并保留 NLL-only 消融。自主生成仅用起始历史初始化一次，评分过程中不注入真实 future。这样在长程正式训练前就暴露累积误差：

```
ES_hat = (1/M) Σ_m ||v^(m) - v*||_2 - (1/(2M(M-1))) Σ_{m≠m'} ||v^(m) - v^(m')||_2
```

`v = Φ(H_{1:N})` 为固定标准化的多时刻路径特征向量，包含距离、torsion、contact 的连续读出及变化，由 B1a 冻结的 decoder/probe 提取。M=4 独立样本起步，验证集用更多。目标特征由真实 H 经过同一个冻结 Φ 得到，另以真实坐标物理量评估 decoder 偏差；Φ 不随 B1b 更新，但保留对输入 H 的梯度。ES 只约束 Φ 保留的路径分布，不证明所有未观测自由度正确。

```
L_B1_full = mean_valid_NLL + λ_ES·ES
```

`λ_ES` 从 0 逐步增加，0.1 仅作归一化后的候选值，以两项梯度和 held-out 分布校准选择。路径 horizon 用真实物理时间定义，短程起步并覆盖 4/16/32/64/96 ns；最长 horizon 必须有验证，训练样本不足时明确该处为外推。不得用“所有随机样本各自贴合唯一真实未来”的 MSE 替代此分布评分。

### 2.6 优化设置

- AdamW，lr=1e-4，grad clip=1。
- 序列长度 N：从 8 步开始，稳定后加到 16、32。
- 历史 K=8 帧。
- 批次构造：连续轨迹片段，时间块之间留隔离区（覆盖 max context + future）。
- 每个验证周期都执行自主 rollout；先要求有限值/结构有效性与长程统计不过度退化，再按固定物理路径评分选择 checkpoint，NLL 作为密度拟合诊断。早停耐心值可从 5 次起步，保存 best/last 与完整选择依据。

### 2.7 Stage B1 验收 gate

与 dS=0 的 H-only 模型、无历史浴模型和容量匹配条件 Gaussian 预测器比较：

1. held-out 序列 NLL（按有效维数归一化）相比 baseline 有统计显著改善（bootstrap 95% 区间不含零）。
2. 白化创新的时间相关、跨残基相关及尾部覆盖率处于 pilot 预定的可接受误差范围；报告效应量/置信区间和多重检验规则，不能由“不显著”宣称假设已被证明。
3. 浴 S 的增量价值：在相同 H 和能量容量下，有历史后验的模型显著优于 S_0~N(0,I) 初始化。
4. 多 lag 路径统计（ACF、MSD、torsion 分布、contact 变化）、分布评分和失败率在 4/16/32/64/96 ns 全部报告，按 §7.4 判定；不能仅挑每个 horizon 最好的一项。

若 S 无显著收益，保留 H-only 动力学继续 B2，不强制引入浴。

---

## 3. Stage B2：共享结构解码与 direct / autonomous 双模式

### 3.1 设计原则

B2 复用 B1a 前置训练并冻结的 oracle 结构解码器；默认不等 B1b 结束才检查可解码性。B2 训练独立 direct 预测器，完成两种模式的结构交付：

```
主线：   X̂_t = D_χ(Q:=H_t, anchor, c)
对比线： X̂_t = D_χ'(Q:=H_t, S_t, anchor, c)        # 见 §3.7，仅诊断
```

按 §2.0.0 的分解：`D_χ` 的输入 = 不变量 `(Q:=H_t, c)` + **等变刚体基 anchor**。

**anchor 的角色（修正旧表述）：** anchor（最后观测帧的 N/CA/C，即 `R_anchor, t_anchor`）只提供**绝对位置**——它是把预测出的内部位移 `Δ_local` 摆到绝对坐标系所必需的等变基。它**不提供内部构象信息**；内部构象必须全部来自 `H_t`。因此：

- 旧文档"解码器不接 anchor"的表述**有误**，绝对坐标在定义上就需要一个刚体基。精确契约是"anchor 不得提供内部构象，只可作为等变刚体基"。
- decoder 内部**不做 Kabsch 对齐**去找 anchor（anchor 已知），Kabsch 只是评估诊断。

**取消独立 context 旁路：** 旧 decoder 的 `h_frozen`（来自 H 的 context 通路）被移除；`Q:=H_t` 是 H 唯一入口。历史/context 信息不再通过"整段历史 H_{≤t} 旁路"注入主解码器——若需要承载历史，由对比线 `D_χ'` 的 `S_t` 提供（S 携带历史摘要），主线 `D_χ` 保持"仅当前帧 H_t + c + anchor"。

`H_t` 对全局 SE(3) 不变，因此 decoder 只能把构象重建到一个整体刚体变换（由 anchor 固定）。训练/结构评分采用一致的刚体不变损失；可视化时用初始帧的参考残基 N/CA/C 定义坐标架。该放置不能恢复真实整体平移/转动，本模型不交付整体扩散。内禀构象 RMSD 与固定 context 坐标下误差必须分开，见 §7.3。

### 3.2 模型结构

复用现有 OpenFold atom14/atom37、rigid-frame、torsion 与原子映射代码，不重写几何计算层。

**输入：** 不变量 `Q_t := H_t ∈ R^{L×dH}` + sequence embedding（氨基酸类型嵌入），**外加等变刚体基** `anchor = (R_anchor [L,3,3], t_anchor [L,3])`。

**骨干：** 图/Attention 模块，产生每残基 frame（旋转+平移，在内部规范坐标系中）和有效 torsion 角（sin/cos 表示）。初始 frame 使用固定规范/理想模板，不从当前真实结构复制残差。

**输出范围：** 首版只解码 backbone `[N, CA, C, O]`，交付物称为 "backbone 动力学"。若源数据有完整 atom14/37 标签，再扩展侧链 χ 角——没有 χ 标签时忽略该项，绝不填零监督。

### 3.3 损失函数

```
L_B2 = L_FAPE + 0.5·L_distance + 0.2·L_torsion + 0.01·L_geometry
```

其中位置项 `L_FAPE` 运行在 §2.0.0 的分解上：`X̂ = anchor_frame ⊗ Δ_local(Q, c)`，即 loss 监督的是 anchor 局部系内的位移（等价于 anchor-local 位移项），anchor 只提供刚体基。绝对坐标 MSE / Kabsch 只作诊断。

- **L_FAPE：** 按有效 frame-point 对归一化，使用一致的 Å 尺度（参照 AlphaFold2 FAPE 定义）。
- **L_distance：** 非局部 CA pair 距离误差，Huber loss，距离除以 10 Å 归一化；覆盖所有序列距离 > 6 的残基对，不 clamp 远距离（否则失去展开构象的监督信号）。
- **L_torsion：** 有效 backbone 二面角的 sin/cos 平方误差，按 atom 存在性、链边界和有效角 mask 评分，不对缺失角施加零监督。
- **L_geometry：** 键长、键角、手性和 clash 的轻度约束，容许 MD 参考轨迹的合理涨落（不把所有键刚性投影到零误差）。

上述权重为归一化后的起始值，不混入 `||H_pred - H_oracle||²`、force floor 或未来位移方向 loss。

### 3.4 训练顺序

1. 在 B1a 中提前完成真实 H 的 oracle 解码训练、encoder 受控微调对照与验收。主损失为 anchor-local 内部位移（§2.0 第 3 条），anchor 只作等变刚体基；必报 真实 H / 零 H / 错时刻 H 三列对照。选定 H 归一化后用该规范重新校准 decoder/probe（或显式封装等价的逆归一化输入层），再冻结；B1b 期间它们只向输入 H 传梯度。
2. B1b 学习 autonomous 核；B2 的 direct 预测器只学习未来 H 的条件分布，使用同一冻结 H 坐标和 decoder。两条路径不共用未来预测头、不用 direct 输出逐步替换 autonomous 状态。
3. decoder 的主监督来自真实 H 对应的真实当前结构，不把任意随机 rollout H 与唯一真实未来强行配对重构。可按独立消融验证训练支持附近的小扰动鲁棒性；严重离开支持时优先修复生成器。
4. 如 B2 确需继续 oracle 微调 decoder，固定 encoder 与两个生成器，升级 decoder 版本，并重新验证两种模式和所有依赖旧 decoder 的物理评分；不得无记录更换读出标准。

**冻结边界：** 默认只有 B1a 允许 encoder 与 oracle decoder 联合校准；B1b/B2 不再同时更新 A、动力学和 decoder。

### 3.5 Stage B2 验收 gate

1. Oracle 重构：给定真实 H_t，backbone RMSD < 1.5 Å（中位数，held-out 帧），非局部 CA 距离误差 < 1 Å（90th percentile）。
2. 同蛋白不同时刻的 H 替换导致解码结构相应改变（验证解码器确实使用了 H 的构象信息，而非仅靠序列记忆平均结构）。
3. 用同蛋白错时刻 H / 恒定 H 替代正确 H，比较对真实动态 observable 的误差；确定性 decoder 的恒定输入自然产生恒定输出，该事实本身不算动态信息证据。
4. Direct 与 autonomous 在 4/16/32/64/96 ns 上分别报告结构质量、分布评分、失败率及与 persistence 的比较；真实 H 的 oracle 重构只用于隔离表示/decoder 误差，不冒充预测成绩。

### 3.6 两种推理模式都保留，职责分开

| 模式 | 计算路径 | 交付与限制 |
|---|---|---|
| direct | `G_ω(H_{≤t},τ,T,c,z) → Ĥ_{t+τ} → D_χ(Q:=Ĥ, anchor, c)` | 直接请求指定时刻的全残基状态/结构，适合终点预测；**单帧输出**，独立查询多个 τ 不自动构成连续轨迹 |
| autonomous | `r_ψ → S_t`，重复执行同一 `(H,S)` 核，再用 D_χ 解码 | 交付连续生成轨迹，用于 ACF、转变、路径分布；误差可能随步数积累 |
| oracle（仅诊断） | `E(X_{t+τ}^{true}) → D_χ` | 检查可解码性，使用了未来真实帧，不可作为部署模式 |
| autonomous+S（可选对比，见 §3.7） | `r_ψ → S_t`，`(H,S)` 核逐步生成，再用 `D_χ'(Q:=H_t, S_t, anchor, c)` 解码 | 仅用于诊断"S_t 是否补上 H_t 缺失的定位信息"；不替换主线 `D_χ`，oracle gate 1 仍只用纯 H |

**接口约定（predictor 只出单帧）：** 因果 predictor `g_ω`（Stage A `CausalTransformerPredictor` 及 §3.6 的 `G_ω`）**一次只输出一帧** `Ĥ_{t+τ} ∈ R^{L×dH}`，其内部张量在时间轴上只到"最后一帧摘要"，**不输出多帧**。理由与后果：

- 时间轴上的 T 维（如 `H_ctx [B,K,L,dH]`）是**把多帧打包进 batch 轴**用于 parallel 编码，帧间无交互；这不是"一次预测一段"的能力。
- 要得到一段连续未来轨迹，**必须递归 rollout**：把预测出的 `Ĥ` 作为下一步的输入再预测下一帧（B1b 的 kernel.rollout / B2 autonomous 就是这样做的）。
- 因此评估"未来轨迹"（ACF/MSD/RMSF/分布）只能通过 autonomous 递归产生，不能指望单次 predictor 调用给出整段；direct 模式只对**单个请求时刻 τ** 负责，多次独立查询 τ 不构成连续轨迹（§3.6 表已注明）。

**direct 首版实现：** 因果历史编码 + 真实 τ/T/序列条件，输出 `H_t+μ_ω` 及正方差；使用逐残基对角噪声加共享低秩高斯噪声产生相关终点样本。若单峰不足，再升级混合分布/flow；普通确定性 MLP 不得标为分布模型。direct 不预测不可观测的 S 真值，也不要求内部模拟 N 步。

**direct 训练：** encoder、H 统计量、decoder 与 B1b 核全部冻结，只训练 G_ω。按 horizon 均衡采样，M=4 独立终点样本，用固定归一化 H 的 energy score 与冻结 decoder/probe 物理读出的 energy score 联合监督；物理读出目标采用同一读出器作用于真实 H，另报相对真实结构的误差。两项按固定训练集尺度归一化，权重通过验证确定，不能让 decoder 随生成器一起改变评分尺度。若需要确定性点预测，单独训练结构/observable 的点预测头并标为点预测基线；`D(E[H])` 不等于 `E[D(H)]`，不能直接声称解码 latent 均值就是最优结构预测。

**比较规则：** direct 与 autonomous 使用同一 context、split、horizon、encoder/decoder 版本和采样数；分别比较终点分布与点预测，报告耗时。direct 的 RMSD 更好不证明其路径动力学更好；autonomous 的单条随机轨迹也不必最接近唯一真实未来。首版不对两种模式的单个随机样本施加彼此相等的 MSE；需要一致性约束时比较条件分布，并作为后续独立实验。

### 3.7 对比解码路径（可选）：带浴状态的解码器 D_χ'(H_t, S_t, c)

**性质：这是一条额外的、与 §3.1–§3.6 主线并列的对比解码路径，不替换、不修改主线 `D_χ(Q:=H_t, anchor, c)`。** 新增它的唯一目的，是回答一个独立的诊断问题：当仅靠 `H_t` 的 oracle 重构（§3.5 gate 1）未达标，或 S 消融（§2.7 gate 3）显示浴状态 `S_t` 携带了未被当前 decoder 利用的信息时，把 `S_t` 一并喂给解码器能否补上缺失的构象定位信息。主线结论、冻结合约和所有主成绩仍以 `D_χ(Q:=H_t, anchor, c)` 为准。

**动机与边界：** `H_t` 来自冻结的 JEPA encoder，是慢变构象表示；`S_t` 是 §2.2 的有限浴辅助变量，沿轨迹积累了**历史与快变量摘要（即 context 信息）**。这就是为什么主线不再保留独立的"整段历史 H 旁路"（§2.1/§3.1）：历史信息由 `S_t` 承载。若"同一 `H_t` 对应多个构象、区别在近期历史/快自由度"，`S_t` 原则上可补上这部分。但 `S_t` 不是可观测量，没有真实标签——因此这条路径**只适用于 autonomous 模式**，其中 `S_t` 由 §2.2 的核沿轨迹自洽产生；direct 模式没有 `S_t` 真值，不使用这条路径（不以任何形式为 direct 伪造 S 标签）。

**解码形式：**

```
X̂_t, H_t, S_t = rollout(X_{0:K}, H_{t-1}, S_{t-1})
                └─ 逐步：(H_{t-1}, S_{t-1}) --B1b 核--> (H_t, S_t)，再 X̂_t = D_χ'(Q:=H_t, S_t, anchor, c)
```

即在 §2.2 的 `(H, S)` 核逐步生成 `(H_t, S_t)` 之后，用一个**独立的**解码器 `D_χ'` 把 `(Q:=H_t, S_t, c)` 连同等变刚体基 `anchor` 映射到结构。`S_t` 的来源严格限定为核的递推输出，不额外引入历史 H 或 context-mean basin 旁路——`D_χ'` 相对主线**只多接一个 `S_t` 入口**（其余通路：`Q:=H_t`、`anchor`、`c` 与主线**容量完全一致**），几何生成层（OpenFold frame/torsion/atom37）与 §3.2 完全一致。

**训练：** encoder、H 统计量、B1b 核全部冻结，与主线一致；`D_χ'` 单独训练，不与 `D_χ` 共享参数。监督沿用 §3.3 的 `L_FAPE + 0.5·L_distance + 0.2·L_torsion + 0.01·L_geometry`，目标为真实当前结构；`S_t` 取真实观测序列经冻结核逐步推进得到的后验样本/均值，训练时明确记录用的是样本还是均值，并在评估时与部署保持一致。不混入 `||H_pred - H_oracle||²`、force floor 或位移方向 loss。

**评估与红线对齐：**

- `D_χ'` 的任何重构**不替代** §3.5 gate 1 对 `H_t` 信息量的干净测量——gate 1 永远只用纯主线 `D_χ(Q:=H_t, anchor, c)` 评分。`D_χ'` 的 oracle 重构单独报告，并显式标注"使用了 `S_t`"，不与纯 H 的 oracle 成绩混排。
- **公平对比约束：** `D_χ` 与 `D_χ'` 的 `Q:=H_t` 通路、`anchor` 通路、`c` 通路必须**同容量**（同样的投影维度）；唯一差别只能是 `D_χ'` 多一个 `S_t` 入口。禁止用"主线把 H 喂两遍 / 对比线把 H 喂一遍"这类容量不对称来制造伪差异。
- 对比口径：在同一 encoder/核版本、同一 split 与 horizon 下，用相同的 autonomous 轨迹分别经 `D_χ(Q:=H_t, anchor, c)` 和 `D_χ'(Q:=H_t, S_t, anchor, c)` 解码，报告 §7.2 的路径层与结构层指标差。差异的归因必须同时附带 §2.7 gate 3 的 S 消融结果：若 dS=0 与有浴模型在纯 H decode 下已无差，则 `D_χ'` 的任何增益都应被怀疑为对 `S_t` 的过拟合/旁路，而非真实信息增益。
- 这条路径不解除 §11 的任何红线：它不是让主解码器吃 S，而是**另建一条可开关的对比解码器**；主线 `D_χ` 的"仅 `Q:=H_t` + `c` + 等变 `anchor`、无 S/history 旁路"约束保持不变。若 `D_χ'` 被证明有稳健增益，按 §9 升级路径记录为一次显式 schema 变更，而不是静默替换主解码器。

**与 §9 的关系：** 本节是 §9 第 2 条"浴容量/跨残基相关"诊断的解码侧对照实验，不是默认交付路径。只有在 oracle gate 未过且 S 消融显示富余信息时才启用；若 §3.5 gate 1 本身已达标，优先不引入 `D_χ'`，避免评估被浴旁路污染。

---

## 4. 数据规范与预处理

### 4.1 时间轴核实（P0 必须完成）

**首先核实数据时间轴，这是 P0 级别的前置任务。**

区分三个量：

| 量 | 定义 | 要求 |
|---|---|---|
| Δobs | 原始数据实际保存间隔（ps） | 从 HDF5/来源元数据核实，写入 manifest；不信脚本默认值 |
| h | 模型一步时间 | 首版 = Δobs；训练与采样使用同一 h |
| τ | 用户要求的预测时长 | = n_steps × h；不直接用 τ 代入单步 Euler NLL |

Checkpoint 必须保存 Δobs 到 ps 的变换。若数据存储间隔是 1000 ps（1 ns），则 K=8、stride=4 对应 28 ns 历史，horizon 32/64/96 对应真实 32/64/96 ns——这个换算必须在 manifest 中明确记录。

### 4.2 Split 策略

两种泛化分别建立：
- **已知蛋白的动力学泛化：** 同一蛋白 held-out replica 或按时间块 split（块之间留隔离区）。
- **未见蛋白的泛化：** 按结构/同源簇划分蛋白（仅按 domain ID 不保证无同源泄漏，应使用序列相似度阈值）。

轨迹切分后再生成窗口，隔离区覆盖实际最大历史跨度加 max_τ。规则采样历史跨度为 `(K-1)·stride·Δobs`，不能漏掉 stride；不规则采样按真实时间戳核对。所有归一化统计量、PCA 基、basin 定义和读出器权重仅在 train split 拟合。

丢帧、跨 replica、跨温度段、时间重置不能当作普通相邻转移；在数据加载时标记并排除这些不连续边界。

### 4.3 Mask 与有效性

- 有效残基：存在 N/CA/C/O 原子且无 insertion code 的标准残基。
- 有效角：由实际原子坐标可计算的二面角（不对缺失原子填零）。
- Padding token 的坐标、边特征和 loss 贡献全部为零，不进入任何统计量。

### 4.4 全原子范围

当前数据路径仅提取 backbone `[N,CA,C,O]`。首版交付物称 backbone 动力学。若源 HDF5 有 atom14/37 标签，B2 阶段可扩展侧链，但必须先验证 χ 角 mask 的正确性（atom 存在性 mask，对称原子交换）。OpenFold 的理想原子放置只解决几何生成，不提供缺失的动力学信息——不用理想化坐标替代真实标签。

---

## 5. 完整训练顺序

| 步骤 | 工作内容 | 冻结 / 可训练 | 完成标志 |
|---|---|---|---|
| **P0：数据审计** | 核实时间单位、原子/角 mask、split 边界；建立数据 schema v2 和 hash | 无模型训练 | manifest、schema hash、审计报告 |
| **P1：Stage A pilot** | 普通 JEPA 与 OPF-JEPA 各一个可比配置；同等容量对照 | 仅 A encoder/predictor/projector | A checkpoint、几何/时间 probe 报告，通过 §1.7 gate |
| **P2a：B1a 校准** | 提前训练 oracle decoder；比较冻结与受控微调 encoder；固定 H 规范 | 顶层/adapter 按 §2.0 解冻；训练 decoder/probe | oracle 和动态区分 gate 通过；冻结 encoder、H 统计量、decoder/probe |
| **P2b：B1b 动力学** | NLL warm-up → 多步 ES；dS 最小消融 | 训练能量网络/浴参数/历史后验 | 通过 §2.7 gate；每次验证自主生成至 96 ns |
| **P3：B2 双模式** | 训练 direct 终点生成器；共享 decoder；oracle 仅诊断 | 固定 encoder/decoder/B1b，仅训练 direct | 两种模式分别通过 §7.4 的适用 gate |
| **P4：端到端评估** | direct 与 autonomous 同协议比较 | 无新训练 | 4/16/32/64/96 ns 全部有报告、置信区间和失败记录 |
| **P5：扩展实验** | 更多蛋白、多温度、更多种子 | 保持已验收合同 | held-out 结果及失败边界报告 |

**Pilot 规模建议：** 8-16 个构象多样、轨迹足够长的蛋白；保留独立验证蛋白和 replica；不只选始终不动的易例。每个方案先排除实现失败，再对关键候选运行至少 3 个随机种子。

**一次正式训练的前置条件：** 小 pilot 必须走完 P0–P4，而不只是 A loss 收敛。优先跑冻结/微调 encoder、H-only/浴、direct/autonomous 这三组最小对照；不全组合穷举。确认数据支持 96 ns、oracle 可解码、核一致、长程不发散并锁定验收阈值后，再扩规模。若某模式未达标，不能以另一模式成绩替代；如实保留已通过部分和未通过边界。

---

## 6. 超参数起始配置

| 模块 | 参数 | 起始值 | 调整依据 |
|---|---|---|---|
| Stage A encoder | dH | 128 | held-out probe 饱和度 |
| Stage A OPF | J, r | 4, 32 | 消融 OPF vs 普通 JEPA |
| Stage A | 历史 K，lag set | 8，{1,2,4,8} | 稳定后加 {16,32} |
| Stage A | lr, EMA m | 3e-4，0.996→0.9999 | held-out predictor loss |
| B1a | encoder / decoder lr | 1e-5 / 1e-4 | 固定物理目标与 oracle 重构，比较冻结基线 |
| Stage B1 | dS | 8 | 消融 {0,4,16} |
| Stage B1 | 序列长度 N | 8→16→32 起步 | 多步训练/验证按真实时间覆盖至 96 ns，不能把 32 步当固定终点 |
| Stage B1 | lr | 1e-4 | 创新校准与预测评分 |
| Stage B1 | Γ 初值弛豫时间 | {2,4,8,16}×Δobs | 拟合后记录实际时间常数 |
| Stage B2 direct | lr / 样本数 M | 1e-4 / 4 | 固定尺度的终点分布评分与校准 |
| Stage B2 | FAPE / dist / torsion 权重 | 1 / 0.5 / 0.2 | held-out backbone 指标 |

---

## 7. 评估矩阵

### 7.1 必需 baseline

| 组别 | 对照 | 要区分的问题 |
|---|---|---|
| 表示 | 普通 JEPA（无 OPF）；冻结/微调 encoder；固定物理特征；末帧 persistence | OPF、表示校准、H 和记忆分别贡献什么 |
| 动力学 | dS=0（H-only）；无历史浴（S_0~N(0,I)）；线性 Gaussian/OU；容量匹配条件 Gaussian | 浴和历史记忆的真实增益 |
| 结构解码 | Oracle / direct / autonomous → 同一 decoder；persistence；错时刻 H | 表示、终点预测、动力学、decoder 各自的瓶颈 |

不做全组合穷举。先完成最小对照，发现问题后针对性增加消融。

### 7.2 三层指标

**条件分布层（B1 评估）：**
- 序列 NLL（按有效维数和时间步归一化均值）
- 白化创新的时间和跨残基自相关
- 尾部覆盖率（创新的分位数校准）
- 以上均按温度、蛋白、lag 分层报告
- direct/autonomous 终点 energy score、距离/接触分布校准；raw latent NLL 仅比较同一固定 encoder/尺度下的模型

**路径层（B1+B2 联合评估）：**
- CA RMSD vs 真实轨迹（多 horizon：4/16/32/64/96 ns）
- contact-J（Jaccard 相似度，接触定义：CA 距离 < 8 Å 且同链序列间隔 > 6；有效残基对 mask 固定）
- backbone torsion 分布的 Jensen-Shannon 散度
- CA-CA 距离的 MSD（均方位移）曲线
- RMSF（每残基均方根涨落）

**结构 / ensemble 层（B2 单独评估）：**
- Oracle backbone FAPE/RMSD（给定真实 H 的重构误差）
- 非局部 CA 距离误差
- 手性 / 键几何有效性（clash 率、peptide 键平面性）
- Autonomous 轨迹的 RMSF 与真实轨迹的相关系数

每次实验报告所有失败、发散和几何无效样本（不只汇总存活轨迹）。bootstrap 95% 区间基于蛋白/轨迹 cluster，不用单次 loss 大小作为 gate 判据。

### 7.3 关键评分约定

- 主结构指标使用逐帧最优刚体对齐的 CA RMSD，仅衡量内禀构象；FAPE、内部距离/角提供补充。固定 context 对齐下的坐标误差另报为含规范选择的诊断，不用于声称重现整体扩散。RMSF 同样固定并记录去除刚体运动的规则。
- Contact-J 使用同一阈值和非局部有效 pair mask，预测与真实接触集合二值化后计算 Jaccard；并集为空时记 1，并单列该类样本比例，避免空接触轨迹抬高均值。
- 所有指标的"真实值"来自 held-out split 的真实轨迹，不用训练集轨迹作为参考。
- 随机采样的 rollout 与单条真实未来的 RMSD 不作为随机动力学的唯一排名标准（参照 Part 11 §11.13 的分析）。
- 不用从 M 个样本中按真实答案挑出的 best-of-M 作为主成绩；点预测与样本平均评分分列。置信区间按蛋白/独立轨迹 cluster，相关窗口不当作独立重复。

### 7.4 正式训练前锁定的验收规则

- **数值与结构有效性：** pilot 每次验证覆盖最长 horizon；观察到 NaN/Inf 立即判当前配置不合格。有限样本零失败不等于总体零风险，报告失败率及上置信界。clash/键几何阈值用 train/validation MD 参考涨落定义，测试集不调阈值。
- **Oracle：** §3.5 的 RMSD/距离门槛与同蛋白错时刻替换检查必须先通过；仅静态折叠结构准确不够。
- **Direct：** 主终点分布评分相对 persistence 与容量匹配条件基线有改善，按蛋白/轨迹配对 bootstrap 给出区间；若交付确定性点预测，再单独要求预定 horizon 的 RMSD/contact-J 改善。
- **Autonomous：** 除终点分布评分外，ACF、MSD、RMSF、torsion/contact 动态分别预先指定允许误差，以独立参考轨迹之间的变化和任务需要校准；每个目标 horizon 都必须检查，不能只凭 one-step NLL 选模型。
- 将上述具体阈值、主指标、容许退化幅度和样本数在 pilot 结束时写入 `acceptance.yaml` 并冻结，正式训练前校验无空项。95% 改善区间跨零记为证据不足，不改称成功；长 horizon 接近不可预测区间时诚实报告统计预测能力，不能承诺逐样本复现真实未来。

---

## 8. 代码实现清单

### 8.1 新增 / 重写模块

| 模块路径 | 内容 |
|---|---|
| `data/dataset.py`, `collate.py`, `manifest_builder.py` | 连续序列采样；严格时间元数据；有效残基/角 mask；轨迹边界标记；时间单位存储 |
| `models/stage_a/model.py` | JEPA-Anything 核心与蛋白适配：单帧 teacher；OPF；EMA eval/update；mask 几何/拓扑隔离 |
| `models/stage_a/objectives.py` | L_pred + L_orth + L_masked + 方差惩罚；配置字段消费检查 |
| `training/stage_a.py` | 记录实际启用的 loss 项及其梯度量级；导出 encoder/schema/hash |
| `models/stage_b/bath_posterior.py` | 因果 Transformer/GRU → (m_0, chol_P_0)；仅接历史 H，不接当前帧坐标 |
| `training/representation_calibration.py` | B1a 冻结/微调对照；前置 oracle；选定 H 规范、hash 并重建缓存 |
| `models/stage_b/kernel.py` | 唯一定义 sample_step 和滤波系数 D/C/Γ，派生 R=2hD、A/u/W；时间/噪声单位固定 |
| `models/stage_b/kalman_likelihood.py` | 创新 NLL；Joseph 协方差更新；Cholesky logdet；无 softplus 截断；jitter 监控 |
| `models/stage_b/energy.py` | 标量 U(H;T,c)，f=-∇U；序列静态边/全局 attention；完整求导与稳定性诊断 |
| `training/stage_b.py` | B1 / B2 明确训练模式；删除 force floor / softplus-NLL / 逐步 detach；legacy baseline 单独保留 |
| `models/stage_b/structure_decoder.py` | H_t + seq → backbone frames + torsion；复用 OpenFold 几何；无 H history / anchor 旁路 |
| `training/stage_b_decode.py` | B1a oracle 训练及 B2 共享解码验收；有效角 mask；oracle/direct/autonomous 分列 |
| `models/stage_b/direct_predictor.py`, `training/stage_b_direct.py` | 真实 τ 条件终点分布；冻结 H/decoder；多 horizon proper score |
| `rollout/sampler.py` | 统一 sampler；S 初始化来自历史后验采样；随机模式 / 步长 / 状态初始化一致 |
| `checkpoint_loader.py` | 精确匹配 schema hash；missing / unexpected keys 直接失败；输出已加载/未加载参数清单 |

### 8.2 删除 / 降级的旧组件

| 组件 | 处理方式 |
|---|---|
| Force floor loss | 删除，不作为默认；若需要对照，保留为独立 legacy baseline |
| softplus-NLL | 删除，改为原始创新 NLL |
| 旧 C 的 direct future-Q head | 替换为 §3.6 的独立、实际训练的 direct-H 预测器，不与旧 Q schema 混用 |
| H、逐残基 anchor、context mean basin 在解码器中的旁路 | 删除；解码器主通路仅接 H_t 和序列 |
| rollout 中间状态的逐步 detach | 删除；滤波递推及 ES rollout 都保留所声明窗口内的梯度 |
| 目标帧的窗口均值代替单帧 teacher 编码 | 修正为目标单帧 teacher 编码 |

### 8.3 必须通过的关键测试

```
1. A encoder SE(3) 不变性：
   随机 SE(3) 变换后的输入，H 在数值误差内不变（< 1e-5 相对误差）。

2. Mask 几何隔离：
   mask token 的坐标、相关边特征和可见侧特征全部为零，不泄漏被遮挡残基信息。

3. EMA teacher 确定性：
   teacher 连续两次在同一输入上调用，输出完全相同；
   只有 online optimizer.step() 后，teacher 参数才改变。

4. 配置字段消费：
   每个 loss 权重字段在配置中出现，则必须在总损失中有对应项；否则训练开始时报错。

5. Kalman 精确性（对照线性系统）：
   在已知闭合解的线性 Q/S 系统上，Kalman NLL = 联合 Gaussian NLL，绝对误差 < 1e-5。

6. sample_step 统计正确性：
   M=10000 样本的经验均值和协方差与解析值相符（10% 相对误差内）。

7. 训练/部署核一致性：
   training kernel 和 deployment sampler 使用完全相同的 D, C, Γ, h, R。

8. 有效 mask 完整性：
   缺失残基/缺失角不产生任何监督梯度；
   chain 断点处的角 mask 正确；
   χ 角在无 atom14 标签时全部 mask。

9. 解码器不使用旁路：
   将 H_t 替换为全零输入，解码器输出不应与正常输出相同（验证 H 被实际使用）；
   将序列替换为随机序列，解码器输出发生显著变化（验证序列被实际使用）。
```

### 8.4 伪代码速览

```python
# ── Stage A 训练 ──────────────────────────────────────────────────────────────
H_ctx = online_encoder.encode_frames(context_frames)   # [B, K, L, dH]
with torch.no_grad():
    teacher_encoder.eval()
    h_target = teacher_encoder.encode_frame(target_frame)   # [B, L, dH]
y_target = [projector_j(h_target) for j in range(J)]        # J × [B, L, r]
y_pred   = predictor(H_ctx, lag_tau, temp, seq)              # J × [B, L, r]
loss_A   = jepa_opf_loss(y_pred, y_target, H_ctx)
optimizer_A.step()
update_ema(online_encoder, teacher_encoder, ema_momentum)

# ── Stage B1 训练 ─────────────────────────────────────────────────────────────
with torch.no_grad():
    H_obs = frozen_A.encode_frames(observed_sequence)   # [B, N+1, L, dH]
m0, chol_P0 = bath_posterior(H_obs[:, :K], dt_history, temp, seq)
loss_B1 = kalman_nll(H_obs, m0, chol_P0, energy_net, D, C, Gamma, R, h)
# kalman_nll 内部执行完整 Kalman filter，梯度经整段递推回到所有参数
optimizer_B1.step()

# ── Stage B2 训练 ─────────────────────────────────────────────────────────────
with torch.no_grad():
    H_true = frozen_A.encode_frame(frame)   # [B, L, dH]
loss_B2 = structure_loss(decoder(H_true, seq), frame)   # FAPE + dist + torsion + geom
optimizer_B2.step()

# ── 部署：一次初始化，自主 rollout ────────────────────────────────────────────
H0 = frozen_A.encode_frame(context_last)
m0, P0 = bath_posterior(context_H, context_dt, temp, seq)
S0 = sample_gaussian(m0, P0)   # 从后验采样，不用均值代替
trajectory = []
H, S = H0, S0
for step in range(n_steps):
    H, S = kernel.sample_step(H, S, energy_net, D, C, Gamma, temp, h)
    X = decoder(H, seq)
    trajectory.append(X)
```

---

## 9. 升级路径（诊断驱动，按需升级）

按诊断结果选择升级，不默认全部实施：

1. **H 信息不足（gate 1 未过）：** 回 Stage A，改几何特征/OPF 配置/容量，不通过浴旁路绕过。
2. **浴容量不足或跨残基创新有强相关：** 增大 dS，或引入逐残基之间有耦合的全局浴（需重新处理协方差块结构）。
3. **振荡记忆明显：** 在浴漂移加入反对称 Ω，`dS = [C^T·f - (Γ+Ω)·S]dt + √(2Γ)dW`，仍保持线性高斯滤波。
4. **非线性/多峰转移无法拟合：** 引入粒子滤波/SMC 或序列 ELBO；此时不能再称为精确 Kalman。
5. **D/C 需要随 H 变化：** 必须补齐 Itô 散度项 `(div_H D)` 和 `(div_H C)`，不用 detach 代替。
6. **需要端到端联合训练：** 仅在消融确认有收益且梯度稳定后引入；联合训练后 A/B schema 版本号递增，所有依赖模块重新校准。

---

## 10. 理论依据摘要

| 理论依据 | 本方案如何使用 | 不推出什么 |
|---|---|---|
| Part 05（表示充分性） | H 的可预测性 gate；动态塌缩诊断 | 低 JEPA loss 自动等于物理充分 |
| Part 06（势能可识别性） | 静态条件能量；温度接口；不用历史依赖能量 | 任意 history-conditioned energy = 真实 PMF |
| Part 07（有限浴热力学） | Q/S 耦合结构；耗散/噪声关系；散度修正条件 | 固定小浴能表达任意蛋白；Euler 自动保 Gibbs |
| Part 08-09（数值核/统计识别） | 规范化 NLL；训练/采样同核；力与增量区分 | softplus-NLL 等价；大 lag 单步 NLL = 多步边缘 |
| Part 10（误差界） | 表示/记忆/拟合/数值/解码误差分开报告 | 有限时间界 = 固定模型无限时间保证 |
| Part 11-12（评估协议/可证伪假设） | Proper score；随机 MSE 界；受控消融 | Oracle 必然排名第一；单一对照能唯一识别旁路 |
| JEPA-Anything 论文 | Stage A 全部架构与损失 | OPF 自动产生物理慢模态；OPF = 统计独立 |

---

## 11. 不能越过的红线（实现禁止清单）

以下行为会静默破坏模型的数学一致性，任何情况下都不允许：

- 用窗口均值代替目标单帧的 teacher 编码。
- 对 teacher encoder 整体做 stop_grad（允许，且必须），但同时对 OPF 投影 `P_j` 的输出也整体 detach（禁止）。
- 对 Kalman NLL 使用 softplus 截断（连续密度 NLL 可以为负，负值是合法的）。
- 在 D、C、Γ 依赖 H_t 的情况下忽略 Itô 散度项（首版通过"静态条件"绕过，不是通过 detach）。
- 训练时 Kalman filter 中途 detach S 的梯度（截断反传必须明确设计和记录）。
- **在解码器中用 anchor 恢复内部构象。** 精确边界：anchor（`R_anchor, t_anchor`）**只可作为等变刚体基提供绝对位置**，**不得**提供任何内部构象信息；不得用逐残基 anchor 位移、context-mean basin、或"整段历史 H 旁路"来恢复内部构象（破坏动力学主通路评估）。绝对坐标需要一个刚体基才能定义，因此"完全不接 anchor"不是本红线的要求——红线是"anchor 不得越过刚体基的角色"。
- 用容量不对称的 `D_χ` / `D_χ'` 对比（如主线把 H 喂两遍、对比线只喂一遍）冒充 S 的信息增益。
- 对缺失 χ 角的残基用零监督（必须 mask 掉）。
- 用不同的时间步长、噪声尺度或状态初始化在训练和部署之间制造不一致。
- 把 padding / mask token 的坐标纳入任何统计量（方差、均值、NLL 等）。
- force floor 或确定性 rollout MSE 作为主损失。
