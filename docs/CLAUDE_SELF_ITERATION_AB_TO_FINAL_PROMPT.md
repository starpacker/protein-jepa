# Claude Code Prompt: 自主迭代 Stage A/B/C，直到绝对性能严格达标

你现在负责继续优化 Protein-JEPA-Frame / LSP-WM protein dynamics world model。

本次任务允许你自主调研、修改代码、设计模型、改变训练目标、重构 Stage A/B/C
之间的接口、训练、诊断并反复迭代，直到最终模型通过严格的绝对性能 gate。

你可以自行决定技术路线。本文不规定你应该使用哪一种 architecture、loss、
encoder、equivariant module、latent factorization、training schedule 或
optimization strategy。

但是，最终目标、evaluation protocol、实验纪律和停止条件是固定的，不能自行
降低、修改或绕过。

---

## 1. 当前已确认的事实

当前 Stage A/B 的 consolidated audit 已完成，记录在：

```text
artifacts/iter_007_phase0_audit/analysis/
docs/STAGE_AB_INVESTIGATION_PLAN_V2.md
```

已经确认：

- raw context velocity 在 protein-disjoint validation 上携带 future local
  displacement directional signal；
- 当前 frozen Stage-A/Stage-B representation 没有保留足够的 residue-level
  dynamic information；
- current Q 有弱的 global conformational information，但 residue-level Q
  基本 collapse；
- oracle-Q、direct-Q 和 qshuffle 在当前 Stage-C decoder 中几乎没有差异；
- Gate D：energy、force、semigroup 等基础物理稳定性目前总体健康；
- Gate E：rollout/temperature response 较弱，主要表现为 Q representation
  不足的下游症状；
- canonical Contact-J gate 对中间精度的逐残基坐标预测非常苛刻；
- 仅靠 current basin prior 或 current Stage-C Q-path 不足以达到最终目标。

因此，本任务不能停留在“让 Q 数值产生变化”或“让某个 probe 的 R² 变高”。
必须让 Q 成为真正有用的 protein dynamical state，并最终转化为 direct 和
autonomous structure prediction 的绝对性能改善。

---

## 2. Q 的目标定义

Q 不能退化为一个只包含 residue displacement 的向量，也不能退化为一个只含
global amplitude 的标量状态。

最终 Q 必须同时保留并支持以下信息：

### 2.1 Residue-local dynamics

- local displacement magnitude；
- local displacement direction；
- local backbone geometry change；
- local fluctuation and mobility；
- contact formation/breaking；
- residue-local uncertainty。

### 2.2 Global conformational state

- radius of gyration 或等价的 global size；
- domain/domain distance；
- domain orientation；
- long-range distance geometry；
- global conformational mode；
- basin identity。

### 2.3 Collective and correlated dynamics

- correlated residue motion；
- domain-scale collective modes；
- long-range coupling；
- contact-preserving coordinated deformation；
- fast/intermediate/slow time-scale separation。

### 2.4 Thermodynamic world-model state

- energy-relevant variables；
- force/drift-relevant information；
- temperature response；
- diffusion/mobility；
- semigroup-consistent evolution；
- metastable or kinetic state information；
- uncertainty and ensemble dispersion。

残基位移只是 Q 应该支持的 observable 之一。任何只提高 displacement probe、
但破坏 global structure、energy landscape、semigroup、temperature response
或 uncertainty 的修改，都不算成功。

---

## 3. 允许修改的范围

你可以修改除 canonical evaluation protocol 之外的任何必要代码，包括但不限于：

- Stage A encoder；
- Stage B latent projection；
- Stage B transition/SDE/semigroup dynamics；
- temporal encoder；
- state factorization；
- invariant/equivariant/vector/collective channels；
- energy model；
- force and diffusion parameterization；
- Stage C decoder；
- training losses；
- auxiliary prediction heads；
- teacher/student or scheduled-sampling strategy；
- checkpoint loading and saving；
- optimizer、training schedule 和 early stopping；
- data preprocessing，只要不改变 canonical split、target 或 evaluation mask；
- Stage A/B/C 之间的 tensor contract。

具体技术方案由你自行研究和决定，不要把本 prompt 当成固定实现方案。

---

## 4. 不得修改的 evaluation contract

以下内容是冻结的，不能为了通过 gate 而修改：

- `scripts/eval_world_model.py` 的 canonical evaluation semantics；
- RMSD 的 Kabsch protocol；
- Contact J、Weak Contact J、Transient Contact J 的定义；
- residue mask；
- chain-aware pair mask；
- residue index handling；
- horizons；
- protein-disjoint train/val/test split；
- persistence baseline；
- failure protocol；
- NaN/Inf handling；
- direct/autonomous/oracle/qshuffle mode semantics；
- target endpoint 和 context window 的定义。

不允许：

- 修改 metric threshold 来制造改善；
- 修改 contact cutoff；
- 删除困难 protein；
- 修改 sample selection 以挑选好看的结果；
- 将失败 sample 静默丢弃；
- 将 NaN/Inf 转成零；
- 使用 future coordinates 作为 direct/autonomous 输入；
- 用 oracle 结果替代 direct/autonomous；
- 用 test split 选择 checkpoint；
- 改变 persistence baseline；
- 把 supplementary metric 当成 canonical metric；
- 只优化或只报告 persistence-relative skill，而不达到预先规定的绝对指标。

可以新增 diagnostic 或 supplementary analysis，但必须与 canonical evaluation
分开保存，不能覆盖 canonical report，也不能改变最终 gate。

---

## 5. 最终绝对性能目标

最终候选模型必须在 protein-disjoint held-out evaluation 上同时评估：

- direct；
- autonomous；
- oracle diagnostic；
- persistence；
- qshuffle；
- qzero/random/noise Q diagnostic。

正式 gate 使用六个 horizon：

```text
4 ns、8 ns、16 ns、32 ns、64 ns、96 ns
```

每个正式 horizon 都必须通过，不能只报告平均 horizon、长 horizon、短 horizon
或最佳 horizon。4/8/16 ns 不是 supplementary horizon，而是与 32/64/96 ns
完全同等的硬性 evaluation horizon。

direct 和 autonomous 必须分别通过，oracle 不能替代它们。

### 5.1 RMSD absolute gate

对于 direct 和 autonomous 的每个正式 horizon，必须满足：

```text
CA Kabsch RMSD < 3.0 Å
```

`3.0 Å` 是硬性上限，不是建议值。不能用 RMSD skill、相对 persistence 改善、
平均 horizon 或某个更容易的子集替代它。

### 5.2 Contact J absolute gate

每个正式 horizon 必须满足：

```text
Contact J >= 0.75
```

### 5.3 Weak Contact J absolute gate

每个正式 horizon 必须满足：

```text
Weak Contact J >= 0.82
```

### 5.4 Transient Contact J absolute gate

每个正式 horizon 必须满足：

```text
Transient Contact J >= 0.84
```

上述绝对值是硬性指标。最终是否达标只由绝对指标决定，不设置 RMSD skill、
Contact skill、Weak skill 或 Transient skill 的最低要求，也不能用 skill 替代
任何一个绝对指标。

Persistence 结果仍然必须在最终报告中列出，用于解释模型相对于 baseline 的行为，
但 persistence-relative skill 不构成成功条件。

最终报告必须列出：

- model absolute metric；
- persistence absolute metric；
- oracle absolute metric；
- qshuffle absolute metric；
- direct/autonomous absolute metric；
- protein-level distribution；
- protein-clustered bootstrap CI；
- failed sample count；
- NaN/Inf count。

如果 canonical held-out 数据显示上述绝对阈值过高，不能自行降低、替换或重新
定义阈值。必须继续寻找合理的模型、表示或训练改进，并在报告中说明当前差距。

---

## 6. Q-information gate

最终模型必须证明 decoder 真正使用 Q，而不是只依赖 H、anchor 或 basin prior。

至少需要满足：

1. correct Q 显著优于 qshuffle；
2. correct Q 显著优于 qzero/random/noise controls；
3. qshuffle 的退化不是浮点噪声；
4. 六个正式 horizon 都有 protein-clustered bootstrap 支持；
5. direct/autonomous 中也能观察到 Q dependence，不能只有 oracle mode；
6. decoder 输出、frame update 或 atom37 geometry 对 Q 有稳定响应；
7. Q dependence 不能以破坏 global contact graph、RMSD 或 physical stability
   为代价。

建议报告但不限定实现的诊断包括：

- qzero；
- random-Q；
- valid-residue permutation；
- q-noise；
- Q scaling；
- `Q_future - Q_context`；
- `Q_rollout - Q_context`；
- output sensitivity；
- frame/coordinate Jacobian；
- local、collective、global 三类 Q ablation。

---

## 7. Stage A/B representation gates

在进入 full Stage-C retraining 前，必须先验证 Stage A/B。

### 7.1 Q 不能 collapse

不能只用 cosine 判断成功或失败。需要报告：

- Q norm；
- residue-wise variance；
- protein-wise variance；
- delta-Q norm；
- delta-Q direction；
- horizon dependence；
- qshuffle/qzero/random sensitivity；
- rollout-to-future latent discrepancy。

Q 变化变大本身不算成功。变化必须对应 future-specific、可复现、可预测的
dynamics information，而不是 noise。

### 7.2 Q 必须保留多尺度信息

必须验证 local、global、collective、thermodynamic 和 uncertainty 信息没有
互相替代或塌缩。

任何只提高 local displacement R²、但让 energy、semigroup、collective mode、
temperature response 或 rollout stability 变坏的模型必须判定为失败。

### 7.3 Direct/autonomous information must exist

future-encoded Q 只能是诊断或 teacher target。最终必须验证：

```text
Q_context / Q_direct / Q_rollout
```

在 held-out 数据上也能支持下游 improvement。

如果只有 Q_future 变好，而 Q_direct/Q_rollout 不变，立即判定为 forbidden
future-Q autoencoder failure。

---

## 8. Gate D/E physical integrity

所有模型修改都必须重新检查：

- energy finite；
- force finite；
- semigroup consistency；
- short-step/composed-step consistency；
- temperature response；
- diffusion/mobility；
- rollout stability；
- uncertainty calibration；
- no NaN/Inf；
- no hidden instability behind clipping。

Gate D 不能因为优化结构指标而退化。

Gate E 必须从 context-only initialization 正常运行，不能依赖 future state。

---

## 9. 自主迭代纪律

你需要持续进行：

```text
hypothesis → implementation → controlled training → diagnostic eval
→ canonical validation → analysis → next hypothesis
```

每一轮必须：

- 使用独立 output directory；
- 保存 config、git commit 和 checkpoint；
- 保存训练曲线；
- 保存 raw per-sample records；
- 保存 Q diagnostics；
- 保存 bootstrap report；
- 记录失败样本；
- 记录本轮 hypothesis；
- 记录为什么继续、停止或回退。

禁止：

- 无限增加 steps 代替模型改进；
- 重复相同配置直到随机达标；
- 只保存最好的偶然 seed；
- 只报告 point estimate；
- 看到小幅改善就宣布完成；
- 用 evaluation bug 或 metric 改动制造成功。

如果需要增加训练预算，必须先提供证据证明当前模型仍在 underfitting，而不是
已经进入 plateau 或结构性失败。

---

## 10. Checkpoint selection 和 early stopping

不能使用 test split 选择 checkpoint。

Validation checkpoint selection 必须同时考虑：

- absolute RMSD；
- absolute Contact J；
- absolute Weak Contact J；
- absolute Transient Contact J；
- persistence-relative differences；
- Q-dependence；
- Gate D/E stability。

一个 checkpoint 不能因为 Contact skill 上升但 absolute Contact J 下降而被选为
最佳模型。skill 不是 checkpoint selection objective。

如果模型在某一个 horizon 改善、另一个 horizon 退化，不能只选择改善的 horizon。

---

## 11. 最终成功定义

只有以下所有条件同时满足，才可以宣布任务完成：

1. Stage A/B Q 通过多尺度 representation gates；
2. Q 不是 displacement-only latent；
3. Q 保留 local、global、collective、thermodynamic 和 uncertainty information；
4. direct/autonomous Q 是 context-predictable 的；
5. decoder 通过 Q-dependence gate；
6. Gate D/E 没有退化；
7. direct 和 autonomous 在 4/8/16/32/64/96 ns 全部通过
   `CA Kabsch RMSD < 3.0 Å`；
8. direct 和 autonomous 在 4/8/16/32/64/96 ns 全部通过
   `Contact J >= 0.75`；
9. direct 和 autonomous 在 4/8/16/32/64/96 ns 全部通过
   `Weak Contact J >= 0.82`；
10. direct 和 autonomous 在 4/8/16/32/64/96 ns 全部通过
    `Transient Contact J >= 0.84`；
11. 六个 horizon 的 protein-clustered bootstrap 和 stability checks 全部通过；
12. failed sample = 0；
13. metric 全部 finite；
14. 没有 future leakage；
15. 结果在独立 seed 和 checkpoint 上稳定；
16. 最终报告完整给出六个 horizon 的 absolute metrics 和所有对照模式；
17. 未达到全部硬性指标前，必须继续反思、设计新假设、修改、训练和验证，
    不得宣布任务完成。

在上述条件全部满足以前，状态必须保持为：

```text
NOT COMPLETE — continue iteration
```

---

## 12. 最终 handoff report

完成时必须提供：

- 最终 checkpoint 路径；
- git commit；
- 完整训练配置；
- Stage A/B/C 结构说明；
- Q 各组成部分的诊断；
- direct/autonomous/oracle/persistence/qshuffle 结果；
- 每个 horizon 的 absolute metrics；
- 每个 horizon 的 persistence-relative metrics；
- RMSD、Contact J、Weak J、Transient J 的 bootstrap CI；
- per-protein win rate；
- failed samples 和 NaN/Inf report；
- energy/force/semigroup/temperature/rollout report；
- 所有失败实验和淘汰理由；
- 说明最终改善来自真实 Q/world-model 信息，而不是 basin、anchor、metric
  或 persistence artifact。

开始工作后，不需要等待用户为每一个小步骤批准。你可以自主完成符合本 prompt
的修改、训练和诊断；但不得开始会改变 canonical evaluation semantics 的修改，
也不得在未满足全部最终 gate 时宣布成功。
