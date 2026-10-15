# Claude Code Prompt: Stage A/B 动力学状态表示重建与自主迭代

你现在负责继续优化 Protein-JEPA-Frame / LSP-WM protein dynamics world model。

当前的主要瓶颈已经不在简单的 Stage-C decoder 调参，而在 Stage A/B
representation：当前 Q 与 context state 的余弦相似度接近 1，decoder 的
oracle-Q、direct-Q、qshuffle 输出几乎相同，说明 Q 尚未成为一个真正可用于
未来结构演化的动力学状态。

本阶段的目标不是把 Q 改造成一个“残基位移预测向量”。Q 必须仍然是一个
能够支持蛋白质动力学演化、能量景观和长期 rollout 的 latent dynamical state。
残基级位移信息只是其中一个必要的可观测结果，不是 Q 的全部定义。

你可以自行调研、修改代码、设计模型、训练和评估；本文只规定目标、语义约束、
实验纪律、质量门槛和停止条件，不规定固定的实现方法。

除非所有规定的 Stage A/B gate 通过，否则本任务没有完成。不能因为某一个
probe 或某一个 horizon 有小幅改善就停止。

---

## 1. 当前问题与基线

当前已观察到的现象：

- `cos(Q_context, Q_future) ≈ 0.9998`；
- `||Q_future - Q_context|| / ||Q_context||` 约为几个百分点；
- Q 的变化幅度可以跟踪全局构象变化大小，但尚未证明它编码了正确的局部
  动力学方向；
- Stage-C decoder 的 oracle-Q 与 qshuffle 几乎相同；
- current basin-only decoder 可以超过 persistence 一部分，但没有通过严格 gate；
- context/basin prior 只能提供有限的结构改善；
- 当前 Stage-C 的 Q-path 已经被证明没有真正支配坐标输出。

相关基线和代码必须先自行确认，不得假设文件路径或 checkpoint 一定仍然是最新的。
优先检查：

```text
artifacts/full_abc_stage_b_v3_final/checkpoint_latest.pt
artifacts/full_abc_stage_c_quse_v5/step_400.pt
artifacts/iter_005_basin_decoder/step_400.pt
scripts/eval_world_model.py
protein_jepa_frame/models/stage_b/model.py
protein_jepa_frame/training/stage_b.py
protein_jepa_frame/training/stage_c.py
```

如果当前 branch 中存在更新的 canonical checkpoint，必须记录实际使用的路径、
git commit、配置、数据 manifest 和 checkpoint hash。

已有的 temporary SOTA 只能作为 baseline，不能直接覆盖或修改：

```text
commit: 9ef07e9
tag: temporary-sota-v5b
branch: protein-dynamics-world-model
```

---

## 2. Q 的明确语义定义

Q 是蛋白质动力学 world model 的 latent state，不是单一任务的 supervised
feature。Q 必须同时保留多个层次的信息。

### 2.1 必须包含的 residue-local dynamical information

Q 必须能够支持或预测以下局部动力学可观测量：

- 每个 residue 的局部位移方向和幅度；
- anchor frame 中的局部相对位移，而不是任意的全局平移；
- 局部 backbone geometry 的变化；
- residue-local fluctuation、mobility 或 uncertainty；
- 局部接触的形成、断裂和距离变化；
- 不同 residue 之间的局部协同运动。

必须检查 vector direction，不能只检查：

```text
corr(||Q_i||, ||displacement_i||)
```

该相关性不能作为唯一的 residue-level information 证据。必须至少考虑
`delta-Q`、向量方向、pairwise interaction 和 local frame 坐标。

### 2.2 必须保留的 global conformational information

Q 不能退化成互相独立的 residue token。它必须保留全局构象信息，例如：

- domain/domain relative orientation；
- global expansion/contraction；
- radius of gyration 或等价的 collective size；
- long-range contact rearrangement；
- inter-domain distance 和 orientation；
- global conformational mode；
- chain-level collective motion。

如果 Q 的每个 residue 都能独立预测局部 displacement，但无法恢复 global
contact graph、domain motion 或 long-range coordination，仍然视为失败。

### 2.3 必须保留的 collective / correlated dynamics

真实蛋白质运动不是 residue-wise independent noise。Q 必须保留：

- correlated residue groups；
- low-frequency collective modes；
- domain-scale motions；
- contact-preserving coordinated deformations；
- local motion 与 global motion 之间的 coupling。

验证不能只使用逐 residue MLP。必须至少有一种能够测试 residue-residue
interaction、pairwise geometry 或 graph/attention relation 的 probe。

### 2.4 必须保留的 thermodynamic / energy-landscape information

Q 还必须服务于当前 framework 的 thermodynamic world-model 目标，包括：

- energy-relevant state information；
- local force 或 drift-relevant information；
- temperature response；
- diffusion / mobility state；
- metastable basin identity或等价的慢变量；
- transition propensity；
- semigroup-consistent evolution；
- long-horizon stability。

不能为了让 Q 更容易重建坐标，而删除 energy landscape、temperature 或
transition dynamics 的语义。

### 2.5 必须包含 temporal / multi-timescale information

Q 必须能够区分至少不同时间尺度上的动力学：

- fast local fluctuations；
- intermediate local rearrangements；
- slow collective/domain transitions；
- long-horizon metastable evolution。

不能只让 Q 记住当前帧的静态结构，也不能只编码一个单一时间尺度的
displacement magnitude。

### 2.6 必须包含 uncertainty 或 ensemble-relevant information

蛋白质未来不是确定性单轨迹。Q 或其配套 state 必须支持：

- future uncertainty；
- multiple plausible transition directions；
- diffusion scale；
- ensemble diversity；
- calibrated rollout dispersion。

如果模型只学习一个 future endpoint 的 deterministic coordinate code，导致
rollout collapse 或 energy dynamics 被破坏，不能接受。

---

## 3. 关键区分：绝对 Q、delta-Q、rollout state 和 observable

必须明确区分以下对象：

```text
Q_context       = encoder(context trajectory)
Q_future        = encoder(future trajectory)     # oracle diagnostic only
delta-Q         = Q_future - Q_context
Q_rollout       = Stage-B transition(Q_context, thermodynamic state)
Q_direct        = learned context-only forecast of future Q
H_context       = Stage-A/context representation
```

`Q_future` 只能用于 representation diagnostic 或 teacher target，不能作为
direct/autonomous inference 的输入。

不能只报告 `cos(Q_context, Q_future)`。至少要报告：

- absolute Q norm；
- per-residue Q variance；
- `delta-Q` norm and distribution；
- per-dimension delta-Q variance；
- Q change versus horizon；
- Q rollout versus future encoded state；
- Q sensitivity to temperature；
- Q sensitivity to perturbation and local motion；
- Q permutation/zero/random controls。

如果 absolute Q 的 cosine 接近 1，但 delta-Q 携带有效的 future-specific
information，不能简单把 Q 判定为失败；必须验证 decoder 和 transition 是否
正确使用了 delta-Q。

---

## 4. Stage A/B 的正确目标

Stage A/B 必须共同实现以下功能：

```text
context trajectory
        ↓
Stage-A representation H
        ↓
Stage-B latent state Q
        ↓
thermodynamic / semigroup transition
        ↓
future-compatible Q_rollout
        ↓
structure, contacts, energy and ensemble observables
```

Stage A/B 目标不是单纯让 future encoder 记住 future coordinates。

真正的目标是：

1. Stage A/H 保留足够的静态结构、局部环境、拓扑和历史信息；
2. Stage B/Q 从 H 中提取有动力学意义的 state；
3. transition 能从 context-only Q 推进到 future-compatible state；
4. Q 同时支持 local、global、collective、thermodynamic 和 uncertainty
   observables；
5. Stage-C decoder 未来可以使用 Q，但 Stage A/B 本身必须先通过独立的
   representation gates。

---

## 5. 必须先完成的 baseline audit

在改代码或开始长训练前，先建立一次完整 baseline：

- Stage A checkpoint、Stage B checkpoint；
- exact config；
- train/val/test manifest hash；
- protein-disjoint split；
- current Q statistics；
- semigroup loss；
- energy landscape diagnostics；
- temperature response；
- rollout stability；
- context/future/rollout Q comparison；
- Stage-C qzero/qshuffle/oracle/direct/autonomous diagnostic。

当前 baseline 的所有问题必须以 raw per-sample records 保存，而不是只保存
aggregate table。

---

## 6. Stage A/B representation quality gates

### Gate A — no collapse, but no arbitrary variance either

Q 不能满足以下任何退化模式：

- `Q_future ≈ Q_context` at all horizons with no useful delta signal；
- all residues have nearly identical Q；
- Q 只有一个 global amplitude；
- Q 只记住 static structure；
- Q variance increases while predictive content does not improve；
- Q 的变化只来自 noise 或 temperature artifact。

同时，不能把“Q 数值变化大”当成成功。必须证明变化与可预测动力学有关。

### Gate B — future-specific local information

在 train split 训练 probe、在 protein-disjoint val split 评估，比较：

- H-only；
- Q-context-only；
- delta-Q-only；
- H + Q-context；
- H + delta-Q；
- oracle Q-future；
- shuffled Q；
- zero Q；
- random Q。

至少测试：

- local displacement vector；
- displacement magnitude；
- local frame geometry change；
- local distance change；
- contact formation/breaking。

成功标准不是某一个相关系数，而是：

- future-aware Q 在至少两个正式 horizon 上显著优于 shuffled/zero controls；
- context-only 或 rollout-predictable Q 能预测一部分 future-specific signal；
- protein-clustered bootstrap CI 支持改善；
- directional vector prediction 不能只依赖 displacement magnitude。

### Gate C — global and collective information

至少评估：

- global radius/size observables；
- domain/domain distances；
- long-range distance matrix；
- contact-map changes；
- correlated residue groups；
- collective mode or graph-level observables。

成功的 Q 不能只在逐 residue displacement probe 上变好，同时在 global
contact graph 和 long-range geometry 上变坏。

### Gate D — thermodynamic and semigroup integrity

重新评估：

- energy finite and stable；
- force finite；
- temperature response nontrivial and physically consistent；
- semigroup composition consistency；
- short-step versus composed long-step consistency；
- Q rollout does not diverge；
- diffusion and uncertainty do not collapse；
- no NaN/Inf；
- no artificial clipping hiding instability。

Q 的 structure probe 变好但 energy/semigroup 明显变坏，不能接受。

### Gate E — rollout predictability

比较：

```text
Q_context as persistence latent
Q_direct forecast
Q_rollout at each horizon
Q_future encoded from held-out future
```

Q_rollout 必须相对于 latent persistence 有真实改善，并且改善不能只存在
oracle future encoder 中。

至少记录：

- Q MSE；
- cosine；
- delta-Q error；
- per-residue error；
- global/collective error；
- energy/temperature observables；
- downstream geometry/contact probe。

---

## 7. 训练和实验纪律

不得：

- 只增加训练步数直到随机通过；
- 只使用 test split 选模型；
- 使用 future coordinates 作为 direct/autonomous input；
- 通过 future-Q autoencoding 单独制造 oracle 成功；
- 修改 metric、mask、horizon、split 或 failure handling；
- 静默删除失败样本；
- 用 NaN-to-zero 隐藏 rollout failure；
- 把 per-residue displacement loss 变成 Q 的唯一目标；
- 为了提高坐标指标而删除 energy/semigroup/temperature 目标；
- 用一个 global Q amplitude 代替完整动力学状态；
- 将单一逐 residue MLP probe 称为 oracle upper bound。

每个实验必须有：

- 独立 output directory；
- config snapshot；
- git commit；
- manifest hash；
- training log；
- checkpoint；
- probe code/version；
- raw per-sample records；
- bootstrap report；
- failure report；
- 明确的 hypothesis、result 和 go/no-go conclusion。

在没有通过 representation probe 前，不得进行昂贵的 full Stage-C training。

---

## 8. 阶段化自主迭代流程

### Phase 0 — audit

复现 baseline，修复明显的 eval/probe provenance 问题，确认所有指标和
sample selection 正确。

### Phase 1 — Q semantics probe

不进行长训练。完成 Q/context/future/delta-Q/rollout 的多任务 probe，确认
信息到底存在于哪里。

### Phase 2 — controlled Stage A/B experiment

只有 Phase 1 找到明确瓶颈后，才选择最小的 Stage A、Stage B 或 joint A/B
修改。训练预算必须受控，并使用 validation early stopping。

### Phase 3 — representation gate evaluation

在 held-out validation 上重新检查 Gate A-E。任何结构改善都必须同时检查
energy、semigroup、temperature、uncertainty 和 rollout stability。

### Phase 4 — Stage C re-entry

只有当 Q 通过 representation gate 后，才重新开放 Stage C。Stage C 必须证明：

- correct Q 比 qshuffle 更好；
- qzero/random 明显退化；
- direct/autonomous 使用的是 context-predictable Q；
- oracle improvement 不能替代 direct/autonomous；
- Q 的使用不破坏 global contact graph 和 RMSD。

### Phase 5 — canonical evaluation

最后才运行完整 evaluation：

- direct；
- autonomous；
- oracle diagnostic；
- persistence；
- qshuffle；
- qzero/random/noise diagnostics；
- RMSD；
- Contact J；
- Weak Contact J；
- Transient Contact J；
- protein-clustered bootstrap；
- per-protein raw records；
- zero failed samples；
- no NaN/Inf。

---

## 9. 最终成功标准

Stage A/B 阶段不能只以最终 Stage-C RMSD 判断成功。至少必须同时满足：

1. Q 不 collapse；
2. Q 包含 residue-local 动力学信息；
3. Q 保留 global/collective conformational information；
4. Q 保留 energy/thermodynamic/semigroup information；
5. Q 支持多个时间尺度；
6. Q 支持 uncertainty/ensemble evolution；
7. context-only / rollout Q 中存在可预测的 future-specific signal；
8. qshuffle、qzero、random controls 显著劣于正确 Q；
9. Q rollout 稳定；
10. 不依赖 future leakage。

Stage C 和最终 evaluation 只有在上述条件通过后才算进入下一阶段。

最终 world-model success 仍然需要在 protein-disjoint held-out evaluation 上
同时改善 RMSD、Contact J、Weak Contact J 和 Transient Contact J，并且
autonomous mode 不能依赖 oracle Q。

---

## 10. 停止与报告规则

不得因为出现以下情况就宣布成功：

- Q variance 变大；
- q_future 与 q_context cosine 变小；
- oracle-Q probe 变好但 rollout-Q 没变；
- 单个 horizon 变好；
- 只有 RMSD 变好；
- 只有 contact-only predictor 变好；
- Stage-C oracle 变好但 direct/autonomous 不变；
- 训练 loss 下降但 semigroup/energy/rollout 崩坏。

每轮必须明确回答：

```text
Q 是否真正学到了动力学，而不是只产生了更多噪声？
Q 是否同时保留 local、global、collective、thermodynamic 和 uncertainty 信息？
context-only 或 rollout Q 是否包含同样的信息？
Stage A/B 是否仍然是一个 world model，而不是 displacement autoencoder？
是否有充分证据进入 Stage C？
```

如果经过受控 probe 证明当前 mdCATH context 对某些 long-horizon future
observable 本身不可预测，必须诚实报告 information ceiling，并给出：

- 哪些 observable 可预测；
- 哪些不可预测；
- 哪些 representation 修改已经被排除；
- 还缺少什么输入或数据；
- 是否需要改变 context protocol、数据规模或目标定义。

不要用无限训练掩盖信息论或数据协议问题。
