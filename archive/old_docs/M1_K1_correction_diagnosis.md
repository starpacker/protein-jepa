# M1 / K1 修正诊断与实施记录

日期：2026-09-17
目标：为 ICLR 投稿修正 M1(力场无效)与 K1(次要指标输给 baseline)两个问题。
用户要求：**不改叙事**；M1 要从代码/框架层面让力场真正进入 world model 起作用；K1 立即修 bug 并重跑。

## 证据(全部来自真实产物,非报告转述)

### M1 力场为何是死的
`artifacts/reviewer_experiments/thermodynamic_closure_v2/m1_strict_v2_v6_final.json`:
- M_None → M_Force 全指标逐位相同(差在 1e-6);M_Memory → M_Full 同理。改善全部来自 memory kernel。
- `effective_drift_increment_norm(force) = 0.00056`,而每步真实位移 ~0.72;力场比信号小 ~1000×。

诊断脚本(held-out latent transitions, 827 条, dim=256):
- `||dz|| = 0.716`,`||F|| = 1.9e-6`(力场小 6 个量级)
- `cos(F, dz) = 0.0012`, frac>0 = 0.491 → 单步方向与真实增量无相关(噪声主导,per-step SNR 极低)
- best scalar force_scale = 597 时 `var(dz)` 不变 → 力场解释真实增量方差 ≈ 0
- checkpoint: `diffusion.log_amplitude→exp=0.977`,但 `lspwm_diffusion_scale=5e-5`,`physical_calibration.log_mobility_scale=0`(未启用/未训),故 native drift `D·∇U·dt ~ 1e-8`。

根因链:
1. 力损失是纯 cosine(丢幅度) —— `mean_force_matching_loss` lspwm.py:595-599。
2. 力幅度只靠 dynamic NLL,但 D 太小 → drift<<noise → NLL 被 persistence+噪声协方差最小化,力场无梯度压力去增大。
3. 单步 dt 下动力学是扩散主导(persistence 最优),力场只在稳态分布/多步 rollout 上才起作用,而这些项权重≈0 或被 gate 关掉。

→ 结论:必须重构训练目标,让力场在"它真正有用的地方"(稳态分布、多步漂移、长 lag 条件均值)被有幅度地监督。仅重训或仅改幅度无效。

### K1 ITS 获胜是双重人为假象
`artifacts/reviewer_experiments/protein_kinetics/k1_fastfolding_qnative_v6.json` +
`artifacts/lspwm_v44_stage_b_v6_kernel_fix/eval/fastfolding_predictions.jsonl`:
1. readout 塌成常数:1372 条预测 `prior_blend=1.0` → 输出=验证集类别先验,模型特征被丢弃;仅 4 个 unique 值,balanced_acc=0.333(三类瞎猜)。次要指标全输由此而来。
2. ITS 掩码不对称:JEPA 常数算子秩退化,第二 timescale=0.0;误差只在 `pred_times>0` 上平均(evaluate_k1_fastfolding.py:401,419)→ JEPA 只算 1 个 timescale,baseline 算 2 个。pooled ITS(不受影响)JEPA relerr=2.28 为最差。0.1445 的"胜利"是掩码制造的。

好消息:`--readout-mode fixed` 产物非退化(1371/1372 unique),模型特征本身有区分度。

## 修正计划
- K1(立即):①修 ITS 对称掩码 bug ②用 fixed readout 重生成 v6 预测 ③重跑,如实报告 6 个指标。
- M1(重构):见下方实施进度。

## M1 关键判定实验(offline drift probe, 2026-09-17)

在决定"重构训练目标"之前,必须先证明:latent 动力学本身是否可从状态回归出来(带幅度)?
如果不可回归,再多的目标重构也没用;如果可回归,则训练目标修正必然 work。

脚本:对 held-out latent transitions(train 3495 / test 827, dim=256)拟合"带幅度"的漂移
`E[dz|z]`,再在 test 上评估泛化:

- 线性漂移 `dz ~ A(z-c)`:held-out **R² = 0.129**,cos(pred,dz) = **0.434**
- MLP 漂移(2×128 SiLU):held-out **R² = 0.201**
- 对比 checkpoint 原生力场:cos(F,dz) = **0.0012**,R² ≈ 0(见上文诊断)

**判定**:latent 动力学是**可回归、可泛化**的真实信号(R²≈0.13–0.20,方向 cos≈0.43)。
checkpoint 力场之所以死,不是因为信号不存在,而是因为:
1. 力损失只训方向(cosine,`mean_force_matching_loss` lspwm.py:566-604),主动剥离幅度;
2. 幅度本应来自 dynamic NLL,但 drift = `D·∇U·dt`,D≈5e-5 → drift~1e-8 << ||dz||≈0.72,
   NLL 对力的梯度 ∝ `D·dt` ≈ 1e-8 → 力场几乎收不到"增大幅度"的梯度压力
   (`overdamped_transition_nll` lspwm.py:729-733)。

→ **结论确认**:幅度监督的漂移目标可以注入真实、可泛化的动力学。重构方向正确。

### 第二个判定:信号是否"可由保守场(PMF 梯度)表达"?(2026-09-17)

模型的力被约束为**保守场**(标量 PMF 的梯度,curl-free)。若真实增量主要是**旋转性**的,
再怎么训 PMF 梯度也抓不住,M_Force 仍会失败。必须先证明信号是保守的。

脚本:同一 held-out 集,拟合"漂移 = -∇U(z)"(U 为标量 MLP,漂移取精确梯度,严格无旋),
再与自由 MLP 漂移对比:

- 保守场(∇标量)漂移:held-out **R² = 0.131**,cos = **0.417**
- 自由 MLP 漂移:held-out R² = 0.193
- **保守场捕获了自由 MLP 可预测方差的 68%**,R² 与无约束线性漂移(0.129)几乎相同

**判定**:latent 漂移**以保守成分为主**。模型现有的 PMF 梯度力架构在结构上**足以**承载动力学。
→ 不需要新增旋转/非保守漂移头。**M1 修复完全落在"训练目标 + 标定"层面**,架构无需扩张。
