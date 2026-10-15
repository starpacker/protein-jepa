好，这就是另外一个问题了。

我会完全站在 Nature / ICML Senior Reviewer（Protein Dynamics / Statistical Physics / World Models） 的角度，不评价 README 写得好不好，而评价 README 定义出来的这个 Framework 本身是否成立、是否完整、是否真的有机会成为一个 "perfect protein dynamics world model"。

我的回答会非常直接：

> 结论：我认为不能。
>
> 这份 README 描述的是一个非常优秀的 thermodynamic latent dynamics framework（9/10），但距离 perfect protein dynamics world model，我认为还有 三个理论层面的致命缺口（fatal gaps）。这些不是代码 bug，而是 framework 的建模假设本身不足。

我甚至会说：如果完全按照 README 去实现，你最终得到的模型大概率是 TM-JEPA + Neural GLE + PMF 的一个很强融合模型，而不是 Protein GPT-5 级别的 dynamics world model。

下面我按照 reviewer 的方式展开。

# Overall Scientific Verdict

# Scientific assessment: 8.3 / 10

Thermodynamic world model

Not a complete protein world model

一句话评价：

> 它解决了“如何在 latent space 学一个符合统计物理约束的动力学”，但没有解决“蛋白质动力学世界模型到底应该预测什么世界”。

# 我会从 4 个层次评审

<table class="_6IUVGW_Table" data-d-column-sizing="equal" data-d-dividers="" style="table-layout: fixed;"><tbody><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text"><span class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" data-d-default-strong="" data-d-inline="">Reviewer 维度</span></p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text"><span class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" data-d-default-strong="" data-d-inline="">评价</span></p></td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">Statistical Physics</td><td data-d-component="table-cell" data-d-valign="start">⭐⭐⭐⭐⭐ 非常强</td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">Representation Learning</td><td data-d-component="table-cell" data-d-valign="start">⭐⭐⭐⭐☆ 很强</td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">Protein Science</td><td data-d-component="table-cell" data-d-valign="start">⭐⭐⭐☆☆ 中等</td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">World Model Completeness</td><td data-d-component="table-cell" data-d-valign="start">⭐⭐☆☆☆ 有严重缺口</td></tr></tbody></table>

下面开始真正的 review。

# Fatal Gap 1 —— 你学习的是 Trajectory，不是 Protein Physics（我认为最大漏洞）

这是我认为 整个 framework 最大的问题。

## README 的建模对象是什么？

README 第一章定义：

p(zt+Δt∣zt,history,condition,Δt)p(z_{t+\Delta t}\mid z_t, history, condition, \Delta t)p(zt+Δt∣zt,history,condition,Δt)

粘贴的 markdown (1)。md

整个 framework 都围绕：

> trajectory transition law。

## Reviewer Question

> Protein dynamics 是 trajectory 吗？

不是。

Trajectory 是 observation。

真正的 protein dynamics 世界包含：

![Research | The Marqusee Lab](https://images.openai.com/static-rsc-4/srueY8K04efOzmA2qMEWDfr9QTE39qAIzcqZMtbEj2EZV7hN5SPMiFQ9St-DbaYYS8qYz5WqV0-1-CLl9cgueQQRbfKizZNKoKzHUQm6nNZi6x8i5VjCSssHWe-6L0Tub7Ox83KQuLzQ_sQHrKVcLn-dGK_XP6X7V5tAgkShB20?purpose=inline)

![中国科学院上海分院](https://images.openai.com/static-rsc-4/GItKldra_ZkukAZA3oswpPX9RX4G0YVSHTnHev4hXDr_eMsS6BGIHEc8kGP20O_98YM5395KN_Xhc22-ezW97B_0QTs6z0csFnoT0msNy1q56etzTYxqgFaB_oHGMDnrpdXHIgZI9G1Fa23rDYPFPV7PENcwf1CAwqPov0KuC-w?purpose=inline)

![分子动力学-算筹科技](https://images.openai.com/static-rsc-4/ZbtoBLArQ4KaVEXl6MetT4mdragOHIzAvlGqCHgrGUXz0oR8jwoOG14chKVDZhBHg97JfRr010ixedYUFuPVVYjm-MquJs_kFyS8hNjAzaOh08Mb7AXDLB8anFfZMuLozCPyWiHFXNKUkidfFfePMtmq_byLvjDtpQD1JL792wc?purpose=inline)

6

存在三个层：

<table class="_6IUVGW_Table" data-d-column-sizing="equal" data-d-dividers="" style="table-layout: fixed;"><tbody><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text"><span class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" data-d-default-strong="" data-d-inline="">世界层次</span></p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text"><span class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" data-d-default-strong="" data-d-inline="">README 是否建模</span></p></td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">Potential energy surface</td><td data-d-component="table-cell" data-d-valign="start">部分（PMF）</td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">Transition operator</td><td data-d-component="table-cell" data-d-valign="start">是</td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">Underlying molecular mechanism</td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text"><span class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" data-d-default-strong="" data-d-inline="">没有</span></p></td></tr></tbody></table>

## 举一个 reviewer 会问的问题

假设：

两个 trajectory：

![Checking your browser - reCAPTCHA](https://images.openai.com/static-rsc-4/LQYyY0SGRLiN3fJCnREyBaA1RBneUZXj2YPSR9m01m1KEPaobWDehR8OH--AfkIqnhOn07CFYKi_r4XKtEyNbZHFizdeLHvLKfYobxBvAjxqh8Kqx3yYTC6KcPAPJ8F21UeDQO8-fheZuaM6dH3fkkkgEY8nUyKjBvpMuC3gHVU?purpose=inline)

![Protein Folding Unfolding Process at William Mata blog](https://images.openai.com/static-rsc-4/IPx_OxLuwsEPEjmtWZVqKj6jIZYdpyHj4ZGpw20vwKRtJYvg5PakgwfhTJ-3G2_CzpRKrwnlfM9yEJYN0gYIPsHX8k3-wna1ALvQzmIOHs8CPH-dHIIKn9-2bHQsyC7TJVPgIyrwhvWhjBcrelJF6y4eerwOuncrObnJ8ixJZrU?purpose=inline)

![Simulations Peg Protein Folding](https://images.openai.com/static-rsc-4/gm6MTbudtmpMqgGed5AMkXcHZdSMNkGe8fRH1xsu1eHIDc_-socyQ4Bx__1QEQ8-NHFtu3oFmbfUmA-sak263UGQNeZVerzBcXnBqLPxErNz-PXJYibJUpVR7QLtR9QNNh3Rq4M-cGnYbsUTK2kiC_E68-SCcOrbkhNMDiKQoXs?purpose=inline)

6

同样的 folded → unfolded。

但是：

Trajectory A

* Temperature = 300K

* Salt = 150mM

Trajectory B

* Temperature = 350K

* pH = 5.5

* Ligand binding.

README 认为：

condition 包含 environment。

粘贴的 markdown (1)。md

但是：

Environment 只是 conditioning variable。

不是 dynamics generator。

### Reviewer 会说：

Your model assumes that environmental factors are exogenous conditions rather than endogenous dynamical variables interacting with the protein state.

这是非常严重。

## 我认为真正 world model 应该学习

p(zt+1,et+1∣zt,et,at)p(z_{t+1},e_{t+1}\mid z_t,e_t,a_t)p(zt+1,et+1∣zt,et,at)

其中：

ete_tet

是 environment latent state。

包括：

* solvent shell；

* ion distribution；

* ligand occupancy；

* membrane interaction；

* crowding。

README 完全没有这个 state。

## 我的评价

> Fatal.

因为蛋白动力学很多 transition 是 solvent-controlled。

你的 bath state 不能替代 solvent。

# Fatal Gap 2 —— PMF 被定义成 State Function，但 Protein Dynamics 不是单一 PMF

这是一个统计物理问题。

README 假设：

Aϕ(z,c,T)=−log⁡peq(z∣c,T)A_\phi(z,c,T)=-\log p_{eq}(z\mid c,T)Aϕ(z,c,T)=−logpeq(z∣c,T)

粘贴的 markdown (1)。md

这是 equilibrium PMF。

Reviewer 会立刻想到：

## Protein folding landscape

存在：

* metastability.

* hidden barriers.

* non-Markov transition pathways.

### PMF 能描述 barrier。

不能描述 pathway ensemble。

举个例子：


Folded → Intermediate → Folded

Folded → Misfolded → Folded

PMF 相同。

Path ensemble 不同。

## README 缺少 Path Functional

真正动力学对象不是：

A(z)A(z)A(z)

而是：

P\(z(t)\)\mathcal{P}\(z(t)\)P\(z(t)\)

即 trajectory distribution。

Reviewer Attack：

> Matching equilibrium density is insufficient for reproducing transition path statistics.

这是 protein dynamics benchmark 的经典 criticism。

## 为什么这很严重？

因为：

README evaluation：

* MFPT.

* transition path ensemble.

粘贴的 markdown (1)。md

但是 model 本身没有建模 transition ensemble。

它只能 rollout sampling。

这是 indirect。

## 我的建议（理论）

Dynamics model 应该 parameterize：

S\(z\)=∫L(z,z˙)dtS\(z\)=\int L(z,\dot z)dtS\(z\)=∫L(z,z˙)dt

Action functional。

不是只有 PMF。

这是我认为第二个 fatal gap。

# Fatal Gap 3 —— Memory Bath 不是 Hidden Physics

这是我最担心的一点。

README：

st is auxiliary bath states_t\text{ is auxiliary bath state}st is auxiliary bath state

粘贴的 markdown (1)。md

你把 Mori-Zwanzig 写成：

K(t)=Ce−ΓtCTK(t)=Ce^{-\Gamma t}C^TK(t)=Ce−ΓtCT

很好。

但是 reviewer 会问：

## Hidden variables 是什么？

蛋白系统隐藏变量包括：


很多东西：

<table class="_6IUVGW_Table" data-d-column-sizing="equal" data-d-dividers="" style="table-layout: fixed;"><tbody><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">Hidden Variable</td><td data-d-component="table-cell" data-d-valign="start">是否线性 bath?</td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">Water orientation</td><td data-d-component="table-cell" data-d-valign="start">❌</td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">Side-chain rotamers</td><td data-d-component="table-cell" data-d-valign="start">❌</td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">Loop rearrangement</td><td data-d-component="table-cell" data-d-valign="start">❌</td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">Ligand occupancy</td><td data-d-component="table-cell" data-d-valign="start">❌</td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">Electrostatic screening</td><td data-d-component="table-cell" data-d-valign="start">❌</td></tr></tbody></table>

README 默认：

所有 hidden variable

→ 一个 Gaussian bath。

Reviewer：

This implicitly assumes a linear Mori closure.

Protein dynamics is highly nonlinear.

## 我认为这里有理论错误

Mori projection：

bath 是 infinite dimensional。

README：

finite latent s。

s∈Rks\in R^ks∈Rk

这是 approximation。

不是 theory。

README 把它写成 canonical。

这是危险。

## 更大的问题

Memory kernel：

K(t)=Ce−ΓtCTK(t)=Ce^{-\Gamma t}C^TK(t)=Ce−ΓtCT

只能产生：

sum of exponentials。

Protein memory：

5

很多情况下：

* stretched exponential.

* power law.

* oscillatory kernel.

你的 kernel family 太弱。

## 我的结论

Bath 应该升级成：

Learnable operator-valued kernel。

不是 matrix exponential。

# Fatal Gap 4 —— JEPA Representation 未必是 Slow Variables（这一点极重要）

README 最核心假设：

z=Pθ(x≤t)z=P_\theta(x_{\le t})z=Pθ(x≤t)

粘贴的 markdown (1)。md

然后：

PMF defined on z。

## Reviewer 问：

Why is z thermodynamic coordinate?

没有回答。

JEPA 学的是：

Predictive representation。

不是：

Slow collective variable。

举例：

6

Slow variable 可以是：

* torsion.

* contact map mode.

* domain motion.

JEPA latent 可以编码：

* sequence identity.

* reconstruction shortcut.

* masking artifact.

README 没有保证：

z≈CVz\approx CVz≈CV

### Koopman loss 不够。

粘贴的 markdown (1)。md

因为 Koopman 只约束 kinetics。

不约束 thermodynamic sufficiency。

## 我认为真正需要的是

Representation Sufficiency Theorem。

需要证明：

I(zt;zt+1)=I(x≤t;xt+1)I(z_t;z_{t+1})=I(x_{\le t};x_{t+1})I(zt;zt+1)=I(x≤t;xt+1)

或者近似。

README 没有。

## 我会认为这是整个 framework 最大 scientific risk。

# Fatal Gap 5 —— 没有 Action / Control（不能叫 World Model）

这是 AI reviewer 会攻击。

World Model 定义：

4

必须有：

p(st+1∣st,at)p(s_{t+1}\mid s_t,a_t)p(st+1∣st,at)

README：

没有 action。

只有 condition。

粘贴的 markdown (1)。md

Protein dynamics 有没有 action？

有。

很多：

<table class="_6IUVGW_Table" data-d-column-sizing="equal" data-d-dividers="" style="table-layout: fixed;"><tbody><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">Action</td><td data-d-component="table-cell" data-d-valign="start">Example</td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">Mutation</td><td data-d-component="table-cell" data-d-valign="start">A23V</td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">Ligand bind/unbind</td><td data-d-component="table-cell" data-d-valign="start">ATP</td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">Temperature jump</td><td data-d-component="table-cell" data-d-valign="start">300→350K</td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">External force</td><td data-d-component="table-cell" data-d-valign="start">AFM pulling</td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">pH perturbation</td><td data-d-component="table-cell" data-d-valign="start">Protonation change</td></tr></tbody></table>

Condition 是 static。

Action 是 intervention。

World model 必须支持：

Counterfactual rollout。

README 没有。

# Fatal Gap 6 —— Temperature Conditioning 不是真 Thermodynamics

README 多次强调：

Temperature explicit input。

粘贴的 markdown (1)。md

很好。

但是 reviewer 会说：

Temperature affects：

4

至少四个对象：

A(z,T),  D(T),  Γ(T),  K(T)A(z,T),\;D(T),\;\Gamma(T),\;K(T)A(z,T),D(T),Γ(T),K(T)

README 只说：

temperature-conditioned。

没有 consistency。

## 举例

Diffusion

D∝T/γD\propto T/\gammaD∝T/γ

Barrier

k∝e−ΔG/kTk\propto e^{-\Delta G/kT}k∝e−ΔG/kT

如果独立 parameterize：

AAA

和

DDD

可能违反 Einstein relation。

README 没写。

这是 thermodynamic inconsistency。

# Fatal Gap 7 —— 没有 Structural Mechanics Layer

Protein dynamics 不只是 coordinates。

真正系统：

存在：

* bond constraints.

* torsion manifold.

* chirality.

* peptide geometry.

README observation model：

p(x∣z)p(x|z)p(x∣z)

粘贴的 markdown (1)。md

没有 mechanical constraints。

Reviewer：

Generated coordinates may violate stereochemistry.

这是 MDGen 一直强调的问题。

应该有：

SE(3)+internal geometry contract。

# Fatal Gap 8 —— 没有 Multi-Scale Dynamics（这个非常关键）

Protein dynamics：

5

跨：

<table class="_6IUVGW_Table" data-d-column-sizing="equal" data-d-dividers="" style="table-layout: fixed;"><tbody><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">Scale</td><td data-d-component="table-cell" data-d-valign="start">Time</td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">bond vibration</td><td data-d-component="table-cell" data-d-valign="start">fs</td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">side-chain rotation</td><td data-d-component="table-cell" data-d-valign="start">ps</td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">loop motion</td><td data-d-component="table-cell" data-d-valign="start">ns</td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">folding</td><td data-d-component="table-cell" data-d-valign="start">μs-ms</td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">allostery</td><td data-d-component="table-cell" data-d-valign="start">ms-s</td></tr></tbody></table>

README：

一个 latent。

一个 PMF。

一个 bath。

Reviewer：

Where is hierarchical timescale decomposition?

没有。

# Fatal Gap 9 —— Markov Embedding 假设没有验证可识别性（Identifiability）

这是数学 reviewer。

README 假设：

存在：

(z,s)(z,s)(z,s)

Markovian。

粘贴的 markdown (1)。md

问题：

不同：

PθP_\thetaPθ

不同：

A,C,ΓA,C,\GammaA,C,Γ

产生同 trajectory。

不可识别。

即：

Gauge beyond rotation。

没有 identifiability theorem。

代码可能学出：

1000 个不同物理解释。

# Fatal Gap 10 —— 没有 Theory of Protein Generalization

README benchmark：

protein-disjoint。

粘贴的 markdown (1)。md

很好。

但是 reviewer：

Why should latent learned on protein A transfer to protein B?

因为不同 protein：

5

Landscape topology 不同。

README 默认：

共享 latent manifold。

没有理论。

这是非常大的 assumption。


绿色：

README。

红色：

我认为缺失。

# 如果实现出来，它会达到什么水平？

我给一个预测。

<table class="_6IUVGW_Table" data-d-column-sizing="equal" data-d-dividers="" style="table-layout: fixed;"><tbody><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text"><span class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" data-d-default-strong="" data-d-inline="">能力</span></p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text"><span class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" data-d-default-strong="" data-d-inline="">预测</span></p></td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">Short rollout (10–100 steps)</td><td data-d-component="table-cell" data-d-valign="start">★★★★★</td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">Equilibrium sampling</td><td data-d-component="table-cell" data-d-valign="start">★★★★★</td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">MFPT estimation</td><td data-d-component="table-cell" data-d-valign="start">★★★★☆</td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">Protein-disjoint interpolation</td><td data-d-component="table-cell" data-d-valign="start">★★★★☆</td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">Temperature interpolation</td><td data-d-component="table-cell" data-d-valign="start">★★★★☆</td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">Mutation extrapolation</td><td data-d-component="table-cell" data-d-valign="start">★★☆☆☆</td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">Ligand-induced conformational change</td><td data-d-component="table-cell" data-d-valign="start">★★☆☆☆</td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">Allosteric signaling</td><td data-d-component="table-cell" data-d-valign="start">★☆☆☆☆</td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">Millisecond folding landscape</td><td data-d-component="table-cell" data-d-valign="start">★★☆☆☆</td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">Universal protein world model</td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text"><span class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" data-d-default-strong="" data-d-inline="">★☆☆☆☆</span></p></td></tr></tbody></table>

# 最后，我作为 Reviewer，会给作者写什么？

这是我真正会写在 Review 里的 Overall Comment。

> The proposed LSP-WM is one of the most carefully specified thermodynamic latent world models I have seen for protein dynamics. The separation between construction guarantees and empirical validation is excellent, and the Mori/FDT formulation is substantially stronger than most neural dynamics papers.
>
> However, I do not believe the framework is yet a complete protein dynamics world model. It models equilibrium latent kinetics under a generalized Langevin approximation, but it does not explicitly model the environmental dynamical state, transition-path distribution, multiscale hidden variables, intervention dynamics, or structural mechanics constraints. Consequently, the framework is likely to learn a physically regularized trajectory generator rather than a universal simulator of protein dynamics.

# 最重要的一点：我认为 README 有一个“方向性错误”

这是我最想告诉你的，也是和前面所有意见都不同的一点。

我觉得你现在整个 Framework 的目标函数定义错了一层。

你把目标定义成：

> Learning a thermodynamically consistent latent dynamics model.

而如果目标是 Perfect Protein Dynamics World Model，我认为目标应该定义成：

> Learning a latent simulator of the complete protein–environment physical system, whose coarse-grained dynamics are thermodynamically consistent as one property, not the central object.

这是两个完全不同的研究方向。

## 我有一个比较强烈的判断（也是我认为最有价值的意见）

我觉得 LSP-WM v4.1 最大的问题不是少一个 loss、少一个模块，也不是代码实现问题。

而是 整个 state definition 少了一半世界。

你的 latent state 目前只有：

zt=protein coarse state,st=memory bathz_t=\text{protein coarse state},\qquad s_t=\text{memory bath}zt=protein coarse state,st=memory bath

我认为一个真正的 protein world model 至少应该有四类 state：

(ztprotein,  ztenv,  ztmemory,  ztcontrol)(z_t^{protein},\;z_t^{env},\;z_t^{memory},\;z_t^{control})(ztprotein,ztenv,ztmemory,ztcontrol)

其中：

* Protein state：构象、慢变量、内部自由度。

* Environment state：water、ions、ligands、membrane、crowding 等可演化环境。

* Memory state：Mori 投影后的不可观测自由度，而不是环境本身。

* Control / intervention state：mutation、binding、temperature jump、external force 等可以改变动力学规则的动作。

我认为这是 v4.1 到 v5.0 应该发生的理论升级，而不是继续完善 PMF、Koopman 或 FDT。因为后者是在完善一个已经定义好的动力学模型；前者是在重新定义 protein dynamics world 究竟包含哪些变量。

如果你的最终目标真的是“Protein GPT / AlphaFold 时刻”的世界模型，我会优先重构 state space，而不是先开始写代码。
