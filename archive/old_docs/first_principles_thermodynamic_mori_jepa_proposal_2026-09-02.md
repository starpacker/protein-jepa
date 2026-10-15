# 从第一性原理重构 Protein-JEPA 动力学

**日期：** 2026-09-02  
**建议名称：** Thermodynamic Mori-JEPA（TM-JEPA）  
**定位：** 对 V5.1/V5.3 的物理审计，以及下一代框架的可证伪设计

## 0. 核心结论

当前 V5.1 的强项是条件均值预测，不是已验证的有限温度动力学。V5.3 加入 physical anchor 后，状态更容易被人解释，但它仍未回答四个决定性的物理问题：

1. 什么分布是平衡分布；
2. 自由能梯度如何产生系统漂移；
3. 耗散和热涨落为何具有当前强度；
4. 丢掉侧链、溶剂和动量后，历史依赖与随机噪声应满足什么关系。

因此，下一步不应继续给 deterministic rollout 叠加更多 ranking loss、observable head 或 mode regularizer。建议把模型的中心从

$$
\dot z=f_\theta(z,h,c)
$$

改成由自由能、耗散矩阵、热噪声和有限记忆共同定义的随机粗粒化动力学。最小可行形式是一个 Markovian-embedded generalized Langevin equation（GLE）：

$$
\begin{aligned}
dq &= \left[-D\nabla_q \mathcal U(q;c,T)+Ca\right]dt+\sqrt{2D}\,dW_q,\\
da &= \left[-C^\top\nabla_q \mathcal U(q;c,T)-\Lambda a\right]dt
      +\sqrt{2\Lambda}\,dW_a,\\
\mathcal U(q;c,T)&=\beta F_\theta(q;c,T),\qquad
\beta=(k_BT)^{-1}.
\end{aligned}
$$

其中 `q` 是可解释的慢 collective variables，`a` 是由历史推断的未解析浴变量，`F` 是条件自由能，`D` 是 mobility/diffusion，`Lambda^{-1}` 是记忆弛豫时间，`C` 是慢变量与记忆的耦合。噪声强度不再由一个自由的 `log_sigma` head 决定，而是和耗散共享参数，因而满足涨落耗散关系。

这不是给现有模型增加一个“physics loss”，而是改变模型可表达的动力学类别：物理定律由参数化方式保证，数据只估计自由能面、扩散率和记忆核。

## 1. 从完整体系开始，而不是从 latent rollout 开始

### 1.1 全原子有限温度动力学

对包含蛋白、溶剂和离子的完整体系，一个常见微观起点是 underdamped Langevin dynamics：

$$
\begin{aligned}
dR &= M^{-1}P\,dt,\\
dP &= -\nabla_R U(R)\,dt-\Gamma M^{-1}P\,dt
      +\sqrt{2k_BT\Gamma}\,dW.
\end{aligned}
$$

它给出三个不能任意拆开的对象：

- 保守力 `-grad U`；
- 耗散 `Gamma`；
- 热噪声 `sqrt(2 k_B T Gamma)`。

在平衡条件下，三者共同保证 canonical distribution：

$$
\pi(R,P\mid c,T)\propto
\exp\left[-\beta\left(U(R;c)+\tfrac12P^\top M^{-1}P\right)\right].
$$

只拟合平均位移而不定义耗散和噪声，无法推出平衡占据、跃迁率、温度响应或时间反演性质。

### 1.2 当前观测不是完整微观状态

当前数据只向 V5 暴露 backbone `N/CA/C/O`，而真实未来还受以下隐藏变量影响：

- 原子动量；
- 侧链 rotamer；
- 溶剂和离子；
- thermostat 随机力；
- 未保留的局部高频振动。

mdCATH manifest 表明存储帧间隔为 `1000 ps`，当前 context 又以 stride 2 取样，即相邻输入相隔 `2 ns`。在这个尺度上，代码中的 frame difference 不能解释为微观速度或动量。它至多是 coarse displacement / flux 的统计特征。

所以目标不应是唯一的未来构象，而应是条件分布：

$$
p(q_{t+\tau}\mid q_{\le t},c,T,\tau).
$$

### 1.3 投影必然产生记忆与有色噪声

令 `q=xi(R_backbone)`。Mori-Zwanzig 投影告诉我们，消去隐藏自由度后，一般不会得到任意 deterministic ODE，而会得到 GLE：

$$
M_q\ddot q(t)
=-\nabla F(q(t))
-\int_0^tK(t-s)\dot q(s)ds
+\eta(t),
$$

并且在平衡附近：

$$
\langle \eta(t)\eta(s)^\top\rangle
=k_BT K(|t-s|).
$$

这给出历史的正确物理含义：历史不是为了提高一个 ranking metric，而是为了估计未解析自由度对当前的残余作用。记忆核和随机力必须成对出现。

### 1.4 为什么不能强迫 history advantage 为正

若 `q_t` 已经足够 Markov，理论上有

$$
p(q_{t+\tau}\mid q_t,q_{<t})=p(q_{t+\tau}\mid q_t).
$$

此时历史不提供增量信息是成功，不是失败。对平衡可逆体系，仅位置轨迹本身还具有时间反演对称性。因而“reversed history 必须更差”不是第一性原理要求，尤其是在 `1-2 ns` 采样间隔下。

应该测量的是：

1. `I(q_future; history | q_t)` 是否显著非零；
2. 加入辅助状态 `a_t` 后，innovation 是否不再有时间自相关；
3. augmented state `(q,a)` 是否比 `q` 更满足 Chapman-Kolmogorov；
4. 若 memory 不被数据支持，模型是否自动退化到 `C=0` 的 Markov diffusion。

## 2. 对当前 V5.1/V5.3 的物理审计

### 2.1 做对了什么

现有框架已有几个值得保留的基础：

- 使用真实 `horizon_time_ps`，而不只用 frame index；
- 使用 causal history，避免未来信息直接进入在线状态；
- 开始区分 instantaneous conformation 与 history memory；
- 使用 SE(3)-invariant 几何量；
- 建立 persistence、wrong horizon、history intervention 等反捷径控制；
- 尝试连续积分和多 horizon 一致性；
- 已认识到单一确定性未来并不完整，并输出方差估计。

这些是良好的工程基础，但还不足以构成物理模型。

### 2.2 当前 rollout 是确定性回归，不是随机动力学

`ContextualSparseMotionDynamics._rollout` 在
`protein_jepa_frame/models/predictor.py:1066` 只积分 mean drift。`variance_rate`
仅累加为 endpoint `log_sigma`，没有 Wiener increment，也没有生成 trajectory ensemble。

因此当前模型学习的是近似条件均值：

$$
\hat z_{t+\tau}\approx E[z_{t+\tau}\mid history].
$$

对多峰跃迁，均值往往落在没有物理概率的 basin 之间。更重要的是，随机过程的条件均值通常不构成 flow：

$$
E[X_{t+s}\mid X_0=x]
\ne
\Phi_s(E[X_t\mid X_0=x]).
$$

所以把确定性均值强行做 semigroup，不等价于约束真实 transition kernel。

### 2.3 不确定性不满足涨落耗散关系

当前 `motion_expert` 同时自由输出 `drift` 和 `log_sigma`
（`predictor.py:1045-1053`），二者没有共享 free energy 或 mobility。结果可能拟合 NLL，却不能解释为何某温度下噪声是该强度，也不能保证长期分布稳定。

物理要求不是“有 sigma”，而是：

$$
\text{noise covariance}=2\times\text{dissipation}
$$

（在无量纲化后），并显式带有 `k_B T` 或 `beta`。

### 2.4 V5.1/V5.3 动力学没有显式接收温度

数据包含 `320, 348, 379, 413, 450 K`，每个 protein-temperature 有 5 条 replica。虽然其他旧分支有 `ThermodynamicConditionEmbedding`，V5.1 调用
`v51_contextual_dynamics` 时没有传入 `temperature_k`
（`protein_jepa_frame/models/jepa.py:989-1009`）。

这意味着同一几何和同一 horizon 在五种温度下被要求共享一个隐式动力学。模型最多通过构象分布间接猜温度，不能分离：

- 温度导致的 equilibrium occupancy 改变；
- 温度导致的 diffusion/rate 改变；
- protein sequence/topology 导致的能量面改变。

这是第一优先级错误。

### 2.5 所谓 velocity 没有物理单位

`DynamicSequenceStateEncoder.velocity` 在 `predictor.py:721-729` 将 frame difference 除以 `dt / mean(dt)`。均匀采样时该分母为 1，所以输出仍是“每帧位移”，不是“每 ps 位移”。

这会造成两个问题：

- 变量没有固定单位，无法解释 mobility 或 relaxation time；
- 一旦混入 ATLAS 等不同采样频率数据，同一数值代表不同物理速度。

建议当前代码里先把它重命名为 `frame_displacement`；真正的 coarse flux 必须除以物理 `dt`，再用训练集尺度无量纲化。

### 2.6 physical anchor 是描述符，不是物理状态方程

V5.3 的 48D anchor 来自四个固定 sequence segments，每段统计 contact density、平均 torsion、Rg、局部距离等（`jepa.py:2158-2235`）。它有 SE(3) 不变性，但存在信息瓶颈：

- 固定四等分不对应真实 domain、hinge 或 allosteric network；
- 段内平均会抵消局部 torsion 和 loop transition；
- 主要统计段内 pair，弱化跨 segment 的关键 tertiary contact；
- hard contact cutoff 对几何变化不光滑；
- 这些量没有共轭力、质量、mobility 或 equilibrium measure。

因此它是“具有物理名称的 feature vector”，不是从状态方程闭合出来的物理状态。它可以作为 observable 或初始 CV，但不应直接被称为完整 physical state。

### 2.7 sparse mode 目前没有可守恒的物理含义

16 个 mode 是 MLP drift 的离散专家，top-k 边界会让 vector field 随 state 不连续。load balance 和 orthogonality 只能防止计算容量塌缩，不能让 mode 对应 normal mode、reaction channel、diffusion eigenvector 或 relaxation process。

物理上更自然的 mode 是 mobility 或 memory kernel 的低秩分解：

$$
D(q,c,T)=\epsilon I+
\sum_{m=1}^r \lambda_m(q,c,T)b_m b_m^\top,
\qquad \lambda_m\ge0.
$$

此时：

- `b_m` 是 collective diffusion direction；
- `lambda_m` 有 `q^2/time` 的意义；
- `Lambda_m^{-1}` 是 memory relaxation time；
- PSD、热噪声和模式贡献都由构造保证。

不需要用 hard top-k 人为声明每个样本必须激活 4 个 mode。若数据只支持 2 个主要通道，强迫 4 个反而是不物理的。

### 2.8 当前 semigroup 约束不是 transition-kernel semigroup

真实 Markov 过程满足的是算子关系：

$$
P_{t+s}=P_tP_s,
$$

其中 `P` 作用在概率分布或 observable 上。当前实现比较的是两个 deterministic endpoints。

此外：

- `horizon_calibration` 在积分结束后显式读取总 horizon
  （`predictor.py:1110-1117`），从结构上破坏 autonomous semigroup；
- composition 从较短 horizon 的 state 继续积分时，复用了原始 `memory`
  （`predictor.py:1331-1335`），而不是第一段 rollout 后的 memory；
- 对 projected non-Markov `q`，只有 augmented `(q,a)` 的 kernel 才应满足 semigroup。

所以当前 semigroup loss 最多是数值自洽正则项，不能作为已学到物理时间演化的证据。

### 2.9 12D observable loss 是诊断，不是定律

contact、torsion、Rg 和 RMSD 对解释结果有用，但预测它们的 MSE 不约束：

- equilibrium distribution；
- detailed balance；
- barrier crossing rate；
- fluctuation-dissipation；
- path probability；
- temperature response。

observable decoder 还可能与 latent generator 共适应，产生报告中已经观察到的 decoder/generator collusion。应保留这些量作为 held-out diagnostics，不再让它们承担“使 latent 变物理”的主要责任。

## 3. 建议框架：Thermodynamic Mori-JEPA

### 3.1 状态分解

定义：

$$
x_t \longrightarrow (c,q_t),
\qquad q_t=[q_t^{phys},q_t^{slow}],
\qquad h_{\le t}\longrightarrow a_t.
$$

- `c`：sequence、topology、fold context 和 simulation domain；
- `q_phys`：固定、可微、定义参考测度的 internal-coordinate CV；
- `q_slow`：由 JEPA/VAC 学到的少量慢变量；
- `a`：历史滤波得到的辅助浴变量，不是另一个任意 semantic latent。

JEPA 的角色被限制为表征学习和 amortized filtering。它不再自行定义物理 law。

### 3.2 可解释 collective variables

第一版不要直接在任意 48D latent 上声明 free energy。自由能依赖坐标参考测度；任意非线性 reparameterization 会引入 Jacobian。

建议分两阶段：

**阶段 A：固定物理 CV，先验证动力学。**

- backbone torsion 的 sin/cos；
- smooth contacts，而不是 hard cutoff；
- domain/segment 间相对 orientation 和 distance；
- Rg、native-contact fraction、secondary-structure occupancy；
- 基于 residue graph 的少量 differentiable pooling。

domain 可以由只使用 train proteins 的结构图谱算法确定，也可以使用固定多尺度 contiguous partitions；不能只有四等分一种尺度。

**阶段 B：学习额外 slow CV。**

使用 orientation-aware SE(3)-equivariant encoder，并通过 VAC/VAMP、time-lagged reconstruction 和 committor-relevant task 学习。若要把 learned coordinate 上的 `F` 解释为自由能，需使用以下之一：

- 可逆 normalizing flow 并显式计入 log-Jacobian；
- 固定 gauge/whitening 和明确的 reference measure；
- 只解释 `F + D` 所定义的动力学，不单独把 latent energy 当成 atomistic energy。

### 3.3 热力学势

对每个 protein/context 和温度定义 PMF：

$$
F_\theta(q\mid c,T)=-k_BT\log \pi(q\mid c,T)+const.
$$

不要把不同 protein 或温度的样本汇总成一个 stationary distribution。绝对能量常数不可辨识，只有 basin 间差值和梯度有意义。

一个进一步可证伪的温度参数化是：

$$
F_\theta(q;c,T)=U_\theta(q;c)-T S_\theta(q;c),
$$

它利用五个温度分离 effective enthalpic 和 entropic contribution。若线性温度形式在 held-out temperature 上失败，再加入 heat-capacity term，而不是让任意 temperature embedding 吞掉所有物理意义。

### 3.4 满足涨落耗散的扩展随机动力学

令

$$
X=(q,a),\qquad
\mathcal U(X)=\beta F_\theta(q;c,T)+\tfrac12a^\top a.
$$

取 symmetric PSD matrix `S` 与 antisymmetric matrix `A`：

$$
S=
\begin{bmatrix}D&0\\0&\Lambda\end{bmatrix},
\qquad
A=
\begin{bmatrix}0&-C\\C^\top&0\end{bmatrix}.
$$

一般 Itô 形式为：

$$
dX=
\left[-(S+A)\nabla_X\mathcal U+\nabla_X\cdot(S+A)\right]dt
+\sqrt{2S}\,dW.
$$

若第一版令 `D,C,Lambda` 只依赖 `(c,T)` 而不依赖 `q`，divergence correction 为零，训练和数值实现都会简单很多。该构造具有：

- stationary marginal `pi(q) proportional exp(-beta F(q))`；
- PSD diffusion；
- noise-dissipation parameter tying；
- auxiliary variable 消元后的 exponential memory kernel；
- 对 `a` 赋予 odd time-reversal parity 后的 generalized detailed balance。

消去 `a` 后，memory kernel 具有形式：

$$
K(t)=C\exp(-\Lambda t)C^\top.
$$

若多个弛豫尺度必要，令 `Lambda=diag(1/tau_1,...,1/tau_r)`。这正好把现有“16 个 motion modes”改造成可测的 memory modes。

### 3.5 温度与单位

统一使用：

- 时间：`ps`；
- 长度：`Angstrom` 或明确无量纲化后的 CV unit；
- 能量：`kcal/mol`；
- `k_B = 0.0019872041 kcal mol^-1 K^-1`。

模型中的 `tau` 只作为 SDE solver 的积分时长，不能进入 endpoint calibration head，也不能进入 local drift 网络。所有 normalization scale 必须保存在 checkpoint 中并可逆转换回物理单位。

初始 base step 应使用数据最小可辨时间 `1 ns`，而不是人为选择 `16 ns`。若 `1 ns` 对 Markov approximation 仍太粗，应通过多步 transition density 或 memory variables 解决，而不是把 base step 调大后继续称为 local dynamics。

### 3.6 transition 是分布，不是一个点

每个 context 采样 `K` 条 SDE trajectory：

$$
\{q_{t+\tau}^{(k)}\}_{k=1}^K
\sim P_\theta^\tau(\cdot\mid q_t,a_t,c,T).
$$

训练和评估使用 proper scoring rule：

- short-lag transition NLL（可用局部 Gaussian 或 learned SDE density）；
- ensemble energy score；
- sliced Wasserstein / MMD 作为补充；
- basin occupancy calibration；
- event probability、MFPT 和 committor calibration。

单条真实 future 仍可用于 proper score；不应把同一 context 的单个 realization 当成可精确回归的唯一答案。

## 4. 训练目标

总目标应尽量由可识别物理量组成：

$$
\mathcal L=
\lambda_{eq}\mathcal L_{eq}
+\lambda_{dyn}\mathcal L_{transition}
+\lambda_{mem}\mathcal L_{innovation}
+\lambda_{slow}\mathcal L_{VAC}
+\lambda_{force}\mathcal L_{mean-force}
+\lambda_{geom}\mathcal L_{geometry}.
$$

### 4.1 Equilibrium/free-energy loss

候选方法：

- conditional score matching；
- energy-based contrastive estimation；
- 每个 protein-temperature 内的 occupancy likelihood；
- replica-held-out free-energy difference。

partition function 不需要跨 protein 比较。必须按 `(protein, temperature)` 分组。

### 4.2 Transition loss

优先拟合最短可用 lag `1 ns`，再由同一个 kernel rollout 到 `2, 4, 8, ... ns`。长 lag 训练可以修正短步误差，但不能拥有独立 horizon head。

### 4.3 Innovation whiteness

从观测 transition 反推标准化 innovation：

$$
\epsilon_t=S^{-1/2}
\left(\Delta X_t-b(X_t)\Delta t\right)/\sqrt{2\Delta t}.
$$

检查：

- mean 接近 0；
- covariance 接近 identity；
- lagged autocorrelation 接近 0；
- 与当前 `q` 不再有系统相关。

这比 history ranking 更直接地检验状态是否闭合。

### 4.4 VAC/VAMP slow-coordinate loss

VAC/VAMP 在这里不只是 benchmark，而是平衡/非平衡转移算子的变分原理。它用于选择 slow CV，但必须：

- blocked trajectory CV；
- protein-held-out reporting；
- 固定维度和相同输入；
- 报告 implied timescale，而不只报告随维度增大的 raw VAMP-2。

### 4.5 Force matching 的正确位置

mdCATH 有 all-atom instantaneous forces，这是高价值信息，但不能把 `1 ns` finite displacement 直接等同于 force。

若 `q=xi(R)`，应使用 established local mean-force estimator，将 atomistic force 投影到 CV，并包含 coordinate metric/Jacobian（Fixman）修正。当前简单比较
`-dF(xi(R))/dR` 与 backbone instantaneous force，只能作为 representation-shaping 辅助项，不能证明 `F(q)` 是 PMF。

第一版 pilot 可以不依赖 force；第二版再加入严格 mean-force matching，并分别报告：

- projected mean-force cosine；
- magnitude error；
- conservative curl residual；
- held-out temperature transfer。

### 4.6 应删除或降级的 loss

从主物理目标中删除：

- 强制 positive reversed-history advantage；
- hard top-k load balance；
- arbitrary mode orthogonality；
- endpoint horizon calibration；
- deterministic endpoint semigroup；
- 用 observable decoder 证明 latent“物理”的逻辑。

这些可以保留为工程诊断，但不应主导优化。

## 5. 分阶段实施方案

### Phase 0：数据和可识别性审计

目标不是训练新大模型，而是先回答数据能支持哪种物理。

1. 使用 contiguous stride-1 mdCATH frames，确认每条轨迹真实 `dt=1000 ps`。
2. 按 protein、temperature、replica 建立严格 split。
3. 检查每条 trajectory 的 burn-in、stationarity 和温度稳定性。
4. 在固定物理 CV 上估计 autocorrelation、implied timescale 和 conditional mutual information。
5. 比较 `q_t` 与 `(q_t, history)` 的 transition likelihood，确定 memory rank。
6. 检查不同温度下 basin occupancy 是否足以估计 `F(q,T)`。

**停止条件：** 若固定 CV 在 `1 ns` 下没有可重复 kinetic signal，先更换数据/lag，不进入大模型训练。

### Phase 1：无记忆热力学基线

设置 `a=0, C=0`，训练 reversible overdamped diffusion：

$$
dq=-D\nabla(\beta F)dt+\sqrt{2D}\,dW.
$$

这必须成为所有复杂模型的物理 baseline。它比 persistence、zero-motion 更有意义，因为它已经包含平衡分布和热涨落。

### Phase 2：低秩 GLE

加入 `r=2,4,8` 个 auxiliary modes，通过 validation evidence 选择最小 rank。history encoder 只输出 `a_t` 的 filtering posterior `q_phi(a_t | q_{<=t})`，并用 KL/whitening 约束其 equilibrium prior。

比较：

- transition proper score；
- innovation autocorrelation；
- augmented CK；
- inferred relaxation times `Lambda_m^{-1}`。

只有这些指标证明 memory 必要，才保留 history branch。

### Phase 3：JEPA learned slow coordinates

在 Phase 1/2 已通过后，再让 JEPA 增加 `q_slow`。冻结 thermodynamic solver 做表示筛选，然后小学习率联合优化。每加入一维 slow CV，都要求带来 held-out variational score 或 MFPT 改善。

### Phase 4：跨蛋白与跨温度泛化

使用 sequence/structure conditioner 预测 `F,D,C,Lambda` 的低秩参数。评估：

- unseen replica；
- unseen trajectory；
- unseen temperature；
- unseen protein；
- 与训练 fold 距离不同的 protein family。

不同层级不能混成一个平均分数。

## 6. 最小激进实验

在全面重写前，建议先做一个能在较小算力下推翻或支持本方案的实验。

### 6.1 数据

- 选择 20 个具有明显多 basin 行为的 mdCATH proteins；
- 每个 protein 使用全部 5 个温度、5 个 replicas；
- train 3 replicas，validation 1 replica，test 1 replica；
- 额外 leave-one-temperature-out；
- 使用 contiguous 1 ns frames，不使用 16 ns base step。

### 6.2 状态

- 8-16D fixed differentiable CV；
- smooth contact、Rg、torsion Fourier feature、domain relative orientation；
- 暂不引入 arbitrary learned residual。

### 6.3 四个模型

1. persistence / empirical transition baseline；
2. V5.1 deterministic rollout；
3. reversible overdamped SDE；
4. rank-4 thermodynamic GLE。

参数量应大致匹配，所有模型使用相同 split 和 horizons。

### 6.4 一票否决门槛

TM-JEPA pilot 只有同时满足以下要求才晋升：

1. long rollout 的 `(protein,T)` occupancy 与 held-out occupancy 一致；
2. transition probability 在每个 horizon 上优于 empirical Markov 和 persistence baseline；
3. augmented state 的 CK error 显著低于 `q-only`；
4. standardized innovation 无显著 lag autocorrelation；
5. noise covariance 与 learned dissipation 使用同一参数，不能单独调 sigma；
6. equilibrium entropy production 与 generalized detailed-balance residual 接近统计噪声底；
7. MFPT、committor 和 implied timescale 在 held-out replica 上有校准置信区间；
8. leave-one-temperature-out 能恢复 occupancy 与 rate 的温度趋势；
9. 10 倍训练 horizon 的 rollout 不发散、不塌缩到单点；
10. 所有结论报告 bootstrap CI，不以单次 seed 的 aggregate MSE 决定。

若 GLE 不优于 overdamped SDE，结论应是“当前 CV/采样下无可识别 memory”，并删除 history branch，而不是继续提高 history ranking 权重。

## 7. 物理解释将如何改变

采用本方案后，每个核心量都有明确含义：

| 模型量 | 物理含义 | 可观测验证 |
|---|---|---|
| `F(q;c,T)` | potential of mean force | occupancy/free-energy difference |
| `grad F` | thermodynamic mean restoring force | projected mean force |
| `D` | collective diffusion/mobility | short-lag variance, Green-Kubo |
| `Lambda_m^-1` | hidden bath memory time | residual autocorrelation |
| `C_m` | slow CV 与 memory mode 的耦合 | history-conditioned transition |
| SDE noise | thermal fluctuation fixed by dissipation | innovation covariance/FDT |
| `P_tau` | transition kernel | CK, detailed balance, likelihood |
| committor | 到达两个 basin 的概率 | shooting/held-out trajectory |
| MFPT/rate | barrier crossing kinetics | replica-held-out event statistics |

这比“mode 7 在某样本被激活”或“12D observable MSE 较低”更接近可检验的蛋白动力学理论。

## 8. 与 V5 的继承关系

不需要丢弃全部现有工程：

**保留：**

- spatial encoder 与 causal data path；
- physical-time dataset；
- EMA encoder 作为 representation stabilization；
- multi-horizon batching；
- held-out control evaluator 的基础设施；
- VAMP/MSM baseline pipeline；
- observable 计算作为 diagnostics。

**替换：**

- `ContextualSparseMotionDynamics` -> thermodynamic GLE solver；
- GRU semantic memory -> auxiliary-state filter；
- drift experts -> PSD mobility / memory modes；
- free `log_sigma` -> FDT-tied diffusion；
- endpoint MSE -> transition proper score；
- deterministic semigroup -> kernel CK；
- forced history advantage -> innovation/Markov closure test。

**暂时禁用：**

- horizon calibration residual；
- hard top-k routing；
- V5.3 learned residual，直到 fixed-CV SDE 通过；
- 将 training-manifest 重采样称为 held-out dynamics evaluation。

## 9. 最关键的科学判断

1. **V5.1 的 91.41% feature skill 不能回答热力学问题。** 它证明了一个预测器能拟合当前 target representation 的平均未来位移。
2. **历史接近无效不一定是坏事。** 在 1-2 ns 位置采样下，这可能意味着动量信息已消失，或当前 CV 已接近 Markov，也可能意味着 hidden slow variables 没被表征。必须用 closure test 区分。
3. **自由能不是 atomistic potential energy。** backbone 投影后，溶剂和侧链熵已经进入 PMF；instantaneous force 不能直接解释 1 ns 位移。
4. **semigroup 属于 transition operator，不属于一般随机过程的 conditional mean。**
5. **物理性应该由架构保证。** PSD、FDT、stationary distribution、time units 和 generalized detailed balance 不应依赖 loss 权重碰巧学出来。
6. **最激进但最可靠的改进，是先缩小模型。** 用 8-16D fixed CV 和 rank-4 GLE 证明自由能、扩散和 memory 可识别，再把 JEPA 表征能力接回来。

## 10. 决策建议

立即停止以 V5.3 physical-anchor + 更多 ranking loss 为主线的扩展。保留 V5.1 作为 deterministic forecasting baseline，同时开一条独立的 TM-JEPA pilot：

1. 先完成 Phase 0 可识别性审计；
2. 实现 fixed-CV reversible overdamped SDE；
3. 再实现 rank-4 Markovian GLE embedding；
4. 通过 thermodynamic/kinetic gates 后才接入 learned JEPA slow coordinates。

真正值得追求的声明不再是“rollout 比 persistence 好 91%”，而是：

> 在 unseen protein/temperature/replica 上，同一个满足涨落耗散与平衡测度的随机生成元，同时解释 stationary occupancy、transition distribution、relaxation times、committor 和 MFPT。

这是一个更难、但有明确物理内容且可以被实验推翻的目标。

## 参考理论

- Mori-Zwanzig projection：粗粒化后记忆核与随机力的来源。
- Kubo fluctuation-dissipation theorem：平衡涨落与线性响应/耗散的联系。
- Generalized Langevin equation：非 Markov coarse dynamics。
- GENERIC / reversible-irreversible decomposition：反对称可逆流与对称耗散流。
- Force matching / multiscale coarse graining：从微观力估计 coarse mean force。
- VAC/VAMP：慢转移算子的变分表征。
- Chapman-Kolmogorov 与 generalized detailed balance：transition kernel 的时间一致性与平衡可逆性。
