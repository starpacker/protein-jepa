有，而且我现在看下来，比 residue 维度更值得优先修的有 5 个。其中第 1 个我觉得尤其关键，可能会直接改变我们对当前 R1 结果的解释。

1. 最大的问题：\(z_t\) 到底是不是“单时刻状态”？

当前报告说 Stage A 用 24-frame windows 做 future prediction，并且 K1 用的是 context_endpoint features。

这就产生一个危险问题。

如果所谓

$$ z_t $$

其实是：

$$ z_t=E(x_{t-23:t}) $$

而不是

$$ z_t=E(x_t), $$

那么 R1 做：

$$ z_t \quad\text{vs}\quad [z_{t-H},...,z_t] $$

发现 history 没帮助，完全不能说明瞬时 state 是 Markov-sufficient。

因为：

$$ \boxed{ z_t\text{ 自己已经包含 history} } $$

了。

那 0/20 的合理解释可能只是：

已经压进 \(z_t\) 的 24 帧历史足够，再给更多历史没有收益。

而当前报告直接从 0/20 得出：

“single-time representation is a dynamical sufficient statistic”

这个结论明显太强。

我建议彻底拆开

定义：

$$ h_t=E_{\rm spatial}(x_t) $$

必须只看单帧。

然后：

$$ m_t=E_{\rm temporal}(h_{t-H:t}) $$

才是 memory/context。

这样才有：

$$ \boxed{ \text{state }q_t \neq \text{memory }m_t } $$

否则你会出现：

JEPA 已经偷偷编码了 memory，然后再在后面加 Mori memory。

这个逻辑会很乱。

2. 当前所谓 GLE，其实还不能严格称作 Mori-Zwanzig GLE

现在报告的动力学近似是：

$$ q_{t+1} = q_t+ M\nabla U(q_t)\Delta t + \sum_\tau K_\tau(q_{t-\tau}-\bar q)\Delta t + \sqrt{2D(q_t)\Delta t}\xi_t. $$

这里有两个问题。

2.1 Memory kernel 为什么作用于 \(q-\bar q\)？

标准 Mori-Zwanzig 意义下，memory term 通常来自 projected unresolved dynamics，形式可能作用于 velocity、force 或其他 resolved observable：

$$ \int_0^t K(t-s)\,\Phi(q_s)\,ds. $$

你现在：

$$ \sum_\tau K_\tau(q_{t-\tau}-\bar q) $$

本质上非常像一个：

$$ \boxed{\text{linear autoregressive memory}} $$

这当然可以很好用。

但目前还没有理论依据说明它就是 Mori-Zwanzig kernel。

因此论文里最好先叫：

finite-memory generalized latent dynamics

或者：

MZ-inspired memory kernel

除非我们真正从 projection formalism 推导这个 parameterization。

2.2 memory 和 noise 没有建立第二类 FDT

这个更严重。

真正 GLE 里：

$$ \text{memory friction} $$

和：

$$ \text{colored random force} $$

不是两个互不相关的东西。

在 equilibrium 下它们通常通过 fluctuation-dissipation relation 联系。

而现在你是：

$$ F_{\rm mem} = \sum K_\tau q_{t-\tau} $$

再独立放：

$$ \sqrt{2D(q)}\,\xi_t $$

的 white noise。

所以：

$$ K_\tau $$

和 noise covariance 没有必然关系。

那严格来说不能直接说：

$$ \boxed{\text{thermodynamically consistent GLE}} $$

你之前 README 的 auxiliary Mori bath 反而在这方面更干净：

$$ ds=-\Gamma s\,dt+\cdots+\sqrt{2\Gamma}\,dW_s $$

因为 memory decay 和 bath noise 是一起参数化的。

这一点我甚至倾向于：回到 auxiliary-state formulation 会比直接学 \(K_\tau\) 更漂亮。

3. \(M\) 和 \(D(q)\) 现在也是两个独立的东西

报告同时定义：

$$ M=\text{mobility} $$

以及：

$$ D(q)=\text{state-dependent diffusion}. $$

但 equilibrium overdamped Langevin 中两者通常不能随便独立：

如果 \(U\) 是物理 energy：

$$ D(q)=k_BT M(q). $$

如果 \(A=U/k_BT\) 是 dimensionless free energy，则可以写成：

$$ dq= [-D(q)\nabla A(q)+\nabla\cdot D(q)]dt + \sqrt{2D(q)}dW. $$

也就是说：

$$ \boxed{ \text{mobility 和 diffusion 应该统一参数化} } $$

而不是一个 \(M\)、一个 \(D\)。

更何况 \(D(q)\) 是 state-dependent 时，还有：

$$ \nabla\cdot D(q) $$

这个 Itô drift。

你前一版 README 其实已经明确写了这个项。

所以我现在会直接把方程改成：

$$ \boxed{ dQ= [-D(Q)\nabla_Q A(Q) +\nabla_Q\cdot D(Q) +F_{\rm mem}]dt + \sqrt{2D(Q)}dW } $$

而不再单独引入 \(M\)。

这样少一个变量，反而更严谨。

4. Learned \(q\) 不能那么轻易称为“physical coordinate”

这一点很隐蔽。

现在：

$$ z \xrightarrow{\operatorname{Linear+Tanh}} q. $$

然后直接在 \(q\) 上：

$$ p(q) \rightarrow A(q)=-\log p(q) \rightarrow \nabla_qA. $$

作为统计 PMF本身没有问题。

但如果进一步把：

$$ \nabla_qA $$

叫 physical force，并直接用 Euclidean Langevin dynamics，就不那么简单了。

因为 \(q\) 是一个 learned nonlinear coordinate。

坐标变换之后一般会涉及：

measure
Jacobian
metric
diffusion tensor
divergence correction

你自己的 README 其实已经指出：

arbitrary JEPA latent 不能在不声明 measure / metric / Jacobian 的情况下直接称作 physical coordinate。

所以我建议一个很重要的术语调整：

不要叫：

$$ \text{physical CV} $$

而叫：

$$ \boxed{\text{learned reduced coordinate}} $$

或者：

latent collective coordinate.

然后我们真正保证的是：

$$ \text{statistically consistent dynamics in the learned coordinate system}. $$

这样稳得多。

5. 我们刚才提出的 \(N\times8\rightarrow G_8\) 也有一个坑

刚才我们说：

$$ Q_t\in\mathbb R^{N\times8} $$

然后 pooling：

$$ G_t=P(Q_t)\in\mathbb R^8 $$

再在 \(G\) 上做 KDE-PMF。

这个想法可以保留，但：

$$ \boxed{ A(G) \text{不能成为整个 }Q\text{ dynamics 的唯一 energy} } $$

为什么？

因为：

$$ A=A(G(Q)). $$

那么 residue force：

$$ -\nabla_Q A = - \left(\frac{\partial G}{\partial Q}\right)^T \nabla_G A. $$

所有 residue force 都只能通过一个 8D bottleneck 传回来。

等于整个：

$$ 8N $$

维 residue state 的 thermodynamics 被压成 8 个 global directions。

这会漏掉大量：

local loop energy
contacts
residue-specific basin
local rearrangement

所以更合理的是：

$$ \boxed{ A(Q) = A_{\rm local}(Q) + A_{\rm pair}(Q) + A_{\rm global}(G(Q)) } $$

例如：

$$ A(Q) = \sum_i A_i(q_i) + \sum_{(i,j)\in E}A_{ij}(q_i,q_j,r_{ij}) + A_g(G). $$

其中只有：

$$ A_g(G) $$

用 8D KDE 来监督。

而整个 \(A(Q)\) 还需要 dynamics / force / equilibrium objectives。

这会比：

$$ Q\to G\to PMF $$

然后让一个 global PMF 控制所有 residues 合理得多。

6. 实验层面也有几个明显的问题
K1 实际只证明 JEPA，不证明 LSP-WM

K1 用 frozen JEPA feature + linear readout。

所以：

$$ 0.465\text{ vs }0.627 $$

证明的是：

$$ \boxed{ \text{JEPA representation captures useful kinetic modes} } $$

不是：

$$ \boxed{ \text{GLE / PMF / memory dynamics works} } $$

这两个 claim 必须拆开。

M1 的结果其实没有“全面胜出”

one-step MSE：

$$ \text{None}=0.00185 $$ $$ \text{Full}=0.00518. $$

也就是说 Full 明显更差。

而 GenericMarkov 的 stationary JSD：

$$ 0.187 $$

也远好于 Full：

$$ 0.447. $$

Full 真正强的是：

$$ \text{innovation whiteness}. $$

所以应该讲：

physical decomposition improves residual temporal closure / selected long-time statistics

而不能讲：

overall dynamics prediction outperforms baselines.

反而这种诚实的 trade-off 我觉得更有意思：

$$ \boxed{ \text{best endpoint predictor} \neq \text{best stochastic dynamics model}. } $$
7. Force 的效果现在其实非常小

Force vs None：

$$ \Delta JSD=-0.00290. $$

统计上当然可以显著，但 effect size 很小。

所以：

“confirms a physically meaningful potential energy landscape”

我觉得太强。

目前更稳的是：

force term produces a small but reproducible improvement in stationary and autocorrelation statistics.

然后想办法做一个更直接的 PMF 验证：

$$ P_{\rm rollout}(G) \overset{?}{\approx} P_{\rm MD}(G) $$

以及：

$$ \Delta A(G) = -k_BT\log \frac{P(G_2)}{P(G_1)}. $$

这样才真正验证 free-energy landscape。

8. 还有两个很像“代码/报告 bug”的地方

一个就是我们刚说的：

报告最后写 potential：

$$ 8\rightarrow128\rightarrow128\rightarrow8 $$

如果是 scalar PMF，应该输出：

$$ 1. $$

另一个很明显：

training table 写：

Det. Skill ↓

从：

$$ 0.070\rightarrow0.011. $$

但之前 README 里的 DetSkill 明明是：

$$ 0.774,\ 0.889 $$

而且 越高越好。

这里实际上大概率记录的是：

$$ \texttt{paired\_delta\_loss} $$

而不是 DetSkill。

报告后面也正好写 final paired delta loss = 0.0108。

这个一定要改，不然 reviewer 很容易困惑。

9. Curriculum 还有一个我很想查的点

第一阶段写：

Representation，0–1200 steps，adapt learned_projection

但同时：

$$ w_{\rm force}=0,\qquad w_{\rm dyn}=0. $$

那问题就是：

$$ \boxed{ 256\rightarrow8 \text{ 的 projection 到底靠什么 loss 学？} } $$

如果 paired_delta_loss 是在 \(z_{256}\) 上计算，那么 projection 根本收不到梯度。

如果 Koopman loss 在 \(q_8\) 上，那才可能学。

所以这里我会直接做一次：

$$ \|\nabla_W L\| $$

检查 learned projection 在 0–1200 是否真的有 gradient。

这是非常值得查的工程点。

如果让我现在重构，我会收敛成这样

我现在觉得最干净的 state decomposition 是：

$$ \boxed{ X_t \rightarrow H_t \rightarrow Q_t } $$

其中：

$$ X_t:N\times3 $$ $$ H_t:N\times256 $$ $$ Q_t:N\times8. $$

\(Q_t\) 必须只由当前帧决定。

历史信息不塞进 \(Q\)。

然后单独：

$$ \boxed{ m_t= M(Q_{t-L:t}) } $$

作为 memory state。

再有：

$$ G_t=P(Q_t)\in\mathbb R^8 $$

只作为 global slow coordinate / PMF diagnostic。

动力学：

$$ \boxed{ dQ_t = [ -D(Q_t)\nabla_QA(Q_t) +\nabla\cdot D(Q_t) +F_{\rm mem}(Q_t,m_t) ]dt + \sqrt{2D(Q_t)}dW_t. } $$

而：

$$ A(Q) = A_{\rm local} + A_{\rm pair} + A_{\rm global}. $$

这样四个概念被彻底拆开：

$$ \boxed{ \begin{aligned} Q_t &: \text{current protein state}\\ m_t &: \text{unresolved history}\\ A(Q) &: \text{equilibrium thermodynamics}\\ D(Q) &: \text{stochastic kinetics} \end{aligned}} $$

我觉得这比当前版本最大的提升不是“模型更复杂”，而是每个变量终于只有一个职责。

