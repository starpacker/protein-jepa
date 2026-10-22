# From-None Stage B（B1b → B2）下一步决策：先换 decoder，还是先查 H+S rollout？

**日期：** 2026-10-18  
**范围：** `From-None Stage B (B1b → B2): Pipeline & Decoder Architecture`  
**依据：** `REPAIR_PROGRESS.md` 中截至 2026-10-18 的真实实验、oracle/autonomous 对照和 H/S latent audit。

## 结论先行

当前最合理的判断不是“GLE rollout 已经证明正确”，也不是“应该立刻换 decoder”，而是：

> **在当前可观测指标下，主瓶颈优先位于 decoder/可解码表示的容量与训练信号；尚没有证据表明 H+S rollout 是主要瓶颈。**

理由是：

1. held-out protein 上，from-none 已经在 1、4、16 ns 稳定超过 persistence，且优势随 horizon 增长（约 +0.060、+0.092、+0.140 Å）。这说明 pipeline 已经学到了一点真实的非平凡信息。
2. `oracle-H ≈ autonomous`（4 ns：3.950 vs 3.951；16 ns：4.889 vs 4.889）。在**当前 decoder 和当前联合训练分布**下，把真实未来 H 换给 decoder 几乎没有收益，因此继续优先修 rollout，短期内不太可能改善 CA 指标。
3. 但这并不等于 rollout 无问题：目前缺少“固定 decoder/encoder 后，oracle latent 与 autonomous latent 的逐步误差”和“只替换 H、只替换 S、只替换 decoder”的严格干预。`H_cos` 与 `H_gt_cos` 的差距提示潜在 rollout 误差，但它尚未证明该误差会转化为 CA RMSD。
4. decoder 目前是 per-residue、无 message passing 的小 MLP。对 mdCATH 的跨残基协同运动、接触重排和全局约束，它很可能是表达瓶颈；而 `ΔH` 虽然被显式输入，却没有被证明已经具有足够的可解码结构信息。

因此建议采用 **“先做可证伪的瓶颈分解，再做 decoder 升级”**，不要直接投入更长 rollout 或更大数据规模。

---

## 一、现在不要做什么

### 不要直接把问题归因于 H+S rollout

`H_gt_cos≈0.9997` 只说明 autonomous H 在 cosine 意义上接近 encoder 对未来帧的 H；它不是坐标预测正确性的证明。相反，`H_cos` 从约 0.88 降至约 0.68，说明 latent 轨迹存在可测的误差，但尚未知道该误差是否是 CA loss 的主导项。

### 不要立刻扩大到 5398 个蛋白或继续单纯加 steps

当前收益只有零点几 Å，且 oracle 与 autonomous 已接近。扩大数据主要会改善泛化，不能解决 decoder 无法表达跨残基协同运动的问题；单纯加步数也无法区分“容量不足”和“rollout 不准”。

### 不要先改掉全部 H/S 架构

当前 H 不是常数塌缩，S 也不是 identity；两者都在工作。一次性重写会丢失已有的 persistence 起点、SE(3) equivariance 和稳定性证据。

---

## 二、必须先完成的瓶颈分解（P0，优先级最高）

所有实验使用同一 checkpoint、同一 held-out proteins、同一 anchor/horizon、同一 Kabsch 与 persistence 实现。每个 horizon 输出 CA RMSD、相对 persistence skill、protein bootstrap，以及 latent 误差曲线。

### P0.1 固定 decoder/encoder，测真实 latent 的 rollout 误差

对同一个已训练模型，分别计算：

- `H_gt = E(X_future)`；
- `H_auto = rollout(H_hist, S0)`；
- `S_auto` 与如有需要的 `S_teacher`；
- 逐 lag 的 MSE、cosine、per-residue norm、ΔH 方向余弦。

重点不是只报告总体 cosine，而是验证：

```text
latent_error(τ) = ||H_auto(τ) - H_gt(τ)||
latent_signal(τ) = ||H_gt(τ) - H_anchor||
relative_error(τ) = latent_error / (latent_signal + eps)
```

如果 relative error 很小但 CA 仍只有微弱 skill，瓶颈在 decoder/表示；如果 relative error 随 horizon 快速超过 latent signal，才有理由优先修 rollout。

### P0.2 四路 decoder intervention

在**不重新训练**的前提下，给 decoder 四种输入：

1. `H_auto, S_auto`：现有 autonomous；
2. `H_gt, S_auto`：只替换 H；
3. `H_auto, S_gt`：只替换 S（若无法构造 teacher S，明确标为 unavailable）；
4. `H_gt, S_gt`：oracle 上限。

同时保留：

5. `H_anchor, S_anchor` 或 zero-ΔH：persistence-equivalent；
6. 随机打乱 residue 的 `ΔH`：检查 decoder 是否只依赖幅度/时间，而非残基对应关系。

决策规则：

| 观察 | 解释 | 下一步 |
|---|---|---|
| `oracle-H - autonomous` < 0.02× persistence，且两者都仅略胜 persistence | rollout 不是主瓶颈；decoder/表示容量不足 | P1 decoder ablation |
| oracle 明显优于 autonomous（≥0.05× persistence） | rollout 是实质瓶颈 | P1 rollout diagnosis |
| `H_gt,S_gt` 仍不优于 persistence | H/decoder 没有足够结构信息，不能先怪 rollout | 改 decoder/表示目标 |
| `H_gt,S_gt` 明显优于 persistence，但 `H_auto,S_auto` 不优 | 信息存在，动力学没有传过去 | 修 H+S rollout |
| 打乱 `ΔH` 与原始相近 | decoder 使用了错误的全局/幅度 shortcut | 增加 residue correspondence / message passing 约束 |

### P0.3 训练期 teacher-forced rollout 对照

用完全相同的 encoder、decoder、loss 和预算，增加一个只用于诊断的 teacher-forced 版本：每一步 dynamics 输入真实编码的上一帧 H，但 decoder 仍只看对应预测状态。不要把这个版本当作正式结果。

- teacher-forced 明显好、autonomous 差：rollout/长程误差为主；
- 两者都差：decoder/表示为主；
- 两者都好但 held-out 差：泛化/temperature/protein split 为主。

---

## 三、P1：若 P0 继续显示 decoder/表示是瓶颈，如何换 decoder

不要直接换成很大的 Transformer。按以下递进式 ablation，每次只改一个因素，保留 zero-head 初始化、anchor-frame reconstruction、`max_update` 和 SE(3) 约束。

### D0：当前 SimpleCADecoder（基线）

记录参数量、训练吞吐和每个 horizon 的 skill。当前结果作为零点，不能只比较最优 checkpoint。

### D1：残基间 message passing decoder（首选下一版）

在 per-residue trunk 后增加 2–4 层 masked residue graph/message passing：

- edge feature：anchor CA 距离、当前/anchor 的局部距离差、序列相对位置；
- node feature：`H_tau`, `H_anchor`, `ΔH`, `S_tau`；
- 只使用观测 anchor 与预测状态，不使用未来坐标；
- 输出仍为 local-frame `dx`，保留 `tanh` bounded update。

原因：当前 decoder 对每个残基独立处理，无法显式表达接触网络的协同运动；D1 是最小的结构性补强。

### D2：两阶段 residual decoder

第一层预测局部位移，第二层读取邻域 residual 与 anchor pair geometry，再修正位移。总更新仍限制在 `max_update` 内。该版本用于判断 message passing 是否真的有效，而不是为了堆容量。

### D3：低秩/全局 latent context

加入 masked global pooling 或 attention summary，并将其广播回 residue head。必须做 ablation：

- local-only；
- global-only；
- local + global。

如果 global-only 也有效，说明当前 `H/S` 没有携带足够的 frame-level collective coordinate；这时应回到 encoder/latent objective，而不是继续堆 decoder。

### D4：仅在必要时增加概率 decoder

`predict_log_scale` 暂不作为首要修复。先确认 point estimate 的结构信息已经存在；否则不确定性 head 只会把不可解码误差重新参数化。

---

## 四、P1：若 P0 反而证明 rollout 是瓶颈，如何查 H+S

### R1：逐步 rollout 误差定位

分别记录每一步的：

- force norm、clipped force fraction；
- `||H_{t+1}-H_t||` 与真实编码位移；
- `||S_{t+1}-S_t||`；
- coupling contribution、tail/restoring contribution；
- gradient norm 与 `force_clip` 饱和率。

目标是确定误差来自 H drift、S bath、coupling，还是数值限幅。

### R2：短程训练、长程评估的 horizon curriculum ablation

固定 decoder，比较：

- 只训练 8–16 ns；
- 只训练 32–96 ns；
- 当前 multi-lag cycle；
- 先短后长 curriculum。

若 curriculum 能在相同总预算下显著降低 relative latent error，说明 rollout 优化困难，而不是模型表达不足。

### R3：S 的功能性 ablation

比较：

- learned S；
- `S=0`；
- frozen S；
- no-coupling / reduced coupling。

必须同时看稳定性和 CA skill。`|S_tau-S_0|` 非零本身不是好事，只有移除 S 后性能下降且 oracle/rollout gap 增大，才能称为 S 在贡献动力学信息。

---

## 五、推荐的执行顺序（给服务器 agent 的直接指令）

### Agent 1：做 P0 intervention，不改模型

> 使用当前 held-out-protein 与 held-out-temperature 的 best checkpoint。实现一个只读诊断脚本，输出 autonomous、oracle-H、oracle-S、oracle-HS、persistence、shuffle-ΔH 六种条件；每种条件按 1/4/16/32/64/96 ns 输出 Kabsch CA RMSD、skill、protein bootstrap，并保存 latent relative error。不得重新训练，不得修改正式 evaluation contract。

### Agent 2：做 teacher-forced rollout 诊断

> 在同一 checkpoint 上增加诊断模式：每个 rollout step 使用真实未来编码 H 作为下一步 dynamics 输入，但 decoder/loss/坐标重建保持不变。输出 teacher-forced 与 autonomous 的 H/S/CA 曲线，禁止把 teacher-forced 结果写入正式性能表。

### Agent 3：实现 D1 message-passing decoder ablation

> 从 SimpleCADecoder 派生独立模块，保留 local-frame dx、SE(3) equivariance、zero-dx persistence、mask、max_update。加入 2 层 masked residue message passing；默认只使用 anchor pair geometry 和当前 residue features。补充 equivariance、mask、zero-update、未来坐标不进入预测路径的测试。

### Agent 4：复核 evaluation 与数据契约

> 检查所有 intervention 是否共享相同 manifest、protein split、temperature split、history K、物理 dt、horizon、Kabsch convention 和 persistence 实现。输出一份 machine-readable JSON，避免 oracle 与 autonomous 使用不同样本或不同对齐方式。

---

## 六、建议的硬门槛与 go/no-go

### P0 通过条件

- intervention 的样本、split、Kabsch 和 persistence 完全配对；
- 至少 3 个独立 seed 或 bootstrap 置信区间；
- 报告 `relative_latent_error`，不能只报告 cosine；
- 明确区分 diagnostic-only 与正式 checkpoint。

### 进入 decoder ablation 的条件

满足以下任一项即可：

- `oracle-HS` 与 autonomous 差异小于 persistence 的 2%，但两者只比 persistence 好零点几 Å；
- `oracle-HS` 也没有稳定超过 persistence；
- message passing 的必要性可由 residue shuffle 或跨残基相关性证据支持。

### 进入 rollout 修复的条件

只有在：

- `oracle-HS` 明显优于 autonomous（建议至少为 persistence 误差的 5%）；并且
- 固定 decoder/encoder 后 teacher-forced 显著优于 autonomous；

才优先投入 H/S rollout。否则 rollout 修复是低回报方向。

### B2 资格

B2 不应由“比 persistence 好 0.0 几 Å”单独触发。至少需要：

1. held-out protein 与 held-out temperature 均稳定胜 persistence；
2. 1–16 ns 不只靠单一 horizon；
3. oracle/autonomous gap 已解释；
4. message-passing 或更强 decoder 的增益能在多个 seed 复现；
5. 长程 rollout 没有通过 `force_clip` 隐藏数值发散；
6. 训练路径仍满足“未来坐标只进 target/loss，不进 prediction path”。

---

## 最终建议

**下一步不是二选一地“换 decoder”或“重做 rollout”。** 最优先动作是运行 P0 的四路 oracle/intervention + teacher-forced 对照。根据目前已有的 `oracle-H ≈ autonomous` 证据，我预期结果更可能指向：

> **先做一个小而受控的 message-passing decoder ablation，同时保留现有 rollout；不要先重写 H+S。**

如果 D1 在相同 checkpoint/预算下把 oracle 与 autonomous 都一起抬高，说明 decoder 是瓶颈；如果 D1 的 oracle 明显抬高但 autonomous 不抬高，再转向 rollout。这样可以避免把 decoder 表达能力不足误诊为动力学错误，也避免用一个更大的 decoder 掩盖 H/S 的真实误差。
