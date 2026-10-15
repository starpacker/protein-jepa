# 框架 vs JEPA-Anything 合规性审查报告

**日期:** 2026-10-13
**审查对象:** Stage A (H) → Stage B (Q_t) 的两阶段流水线
**对照基准:** `jepa_anything/JEPA-Anything-main`（design-protocol / config-contract / architecture / losses.py / opf.py）

---

## 0. 结论先行（TL;DR）

你的直觉是对的，但**问题不在“把 H 压缩成 Q_t”这个动作本身**，而在于：

1. **Q_t 被当成了"状态"，而不是"从状态里抽出来的预测因子坐标"**。JEPA-Anything 里 `z`（完整态）和 `y=Az`（因子坐标）是同一层的两个视角，`K*r=d`，是**可逆的无损分解**；而你这里是 `H(d=256) → Q(d=8)` 的**有损降维**，Q 丢失了 H 里除 8 维之外的全部信息，却被当作唯一状态拿去 rollout。
2. **Stage A 的 H 是"时间维度的状态"，Stage B 把整个时间窗塌缩成一个 Q_t（只取 `frame_states[:, -1]`）**。这等于在 Stage B 里**主动丢弃了 JEPA 最重要的时序预测结构**，把 JEPA 退化成了一个静态自编码器。
3. **JEPA 的核心是"预测 latent 未来"，但你的两阶段之间没有 latent 预测契约**：Stage A 训练的是 H 的预测，Stage B 训练的却是 Q 上的能量/力学（NLL、力、semigroup）。**没有任何损失函数要求 `Q_t` 能被预测出 `Q_{t+Δt}`**。JEPA 的 "predict-then-compare-in-latent-space" 在 Stage B 消失了。

换句话说：**Stage A 是 JEPA，Stage B 不是 JEPA，而 Stage B 又覆盖/取代了 Stage A 的 latent**。这是结构性的语义断裂，不是调参能解决的。

---

## 1. JEPA-Anything 给出的"正确范式"是什么

读 `design-protocol.md` / `config-contract.md`，JEPA-Anything 把状态语义钉死为以下几点（原文）：

### 1.1 分析坐标 ≠ 状态合成（最关键的一条）

> "Concatenating the `K` predicted `r`-dimensional blocks creates the analysis-coordinate vector `u`, while a separate Moore-Penrose pseudoinverse synthesis maps `u` ... back to a complete `d`-dimensional latent state. **A design must not call concatenation alone state synthesis.**"

也就是说：
```
z                      # 完整 latent state, 维度 d
y = A z = concat(P_k^T z)   # 因子坐标 (analysis), 维度 K*r = d
z_synth = pinv(A) y         # 用伪逆无损重建回 d 维
```
**`y` 和 `z` 是同维的、可通过 `pinv(A)` 往返的。** 因子化是"换一组正交坐标来看同一个 d 维态"，**不是降维压缩**。而且强制 `rank(A)=d`、有条件数上限、有 synthesis-NMSE 上限 —— 保证无损。

### 1.2 四条损失语义必须分离（config-contract.md）

| 损失 | 语义 |
|---|---|
| `factor_prediction` | 预测 `factor_hat_k` vs **stop-grad** 的 `P_k^T z_target` |
| `projector_orthogonality` | within `‖P_k^T P_k − I‖²` + cross `‖P_i^T P_j‖²`（**几何正交**，非统计） |
| `factor_activity` | 每个因子坐标的 std 下限（防塌缩） |
| `online_encoder_activity` | online encoder 每个维度的 std 下限 |

注意第 1 条：**预测发生在因子坐标空间，target 是 target-encoder 的 online 输出的 stop-grad 版本** —— 这才是 JEPA 的"latent 预测"。第 3、4 条是**防止表征塌缩**的（你框架里恰好缺这两条，见 §3）。

### 1.3 "一个预测边界 + 一个消费者"

`design-protocol.md`：一个系统、一组 identity keys、一个 cutoff、**恰好一个 downstream consumer**（`terminal_readout` / `repeated_transition` / `factor_analysis`）。`repeated_transition` 要求"state 可被反复推进（repeatedly advanced）"，且每个 horizon 都要能被评估。

### 1.4 三种使用模式由消费者决定，不由架构决定

- 一次性读出 → `terminal_readout`
- 反复推进（rollout / MPC）→ `repeated_transition`，**必须评估每一个声明的 horizon**
- 干预/解耦分析 → `factor_analysis`，**光有预测误差不算，必须做 probe/intervention**

---

## 2. 你框架的实际实现（代码事实）

从 `models/stage_a/model.py`、`models/stage_b/model.py`、`models/stage_b/state.py`、`train_iter008/009` 读出的事实链：

### 2.1 Stage A：产生带时间维度的 H（这一层是 JEPA）

```python
# stage_a/model.py
context_latents = encoder(coords)             # [B,T,L,D]  逐帧逐残基 H
context_states  = temporal_encoder(...)        # 时序编码
context_states["residue_states"]              # [B,T,L,D_H]   <-- 时间维度的 H
predictions = predictor(context_latents, target_mask=...)
target_states = target_temporal_encoder(...)   # EMA target, no_grad
state_predictions = state_predictor(context_states["frame_states"], ...)
```
- 上下文/目标双塔 + EMA target + mask prediction + multi-horizon endpoint：**这确实是 JEPA 式的 latent 预测**。
- H 保留 `[B,T,L,D]` 的时间轴。Stage A 输出 `context_residue_states` / `future_residue_states`。

### 2.2 Stage B：把 H 的时间轴压成单个 Q_t

```python
# stage_b/model.py  encode_state()
temporal_out = self.temporal_encoder(context_latents, ...)
frame_state  = temporal_out["frame_states"][:, -1]    # <-- 只取最后一帧！
h_last       = context_latents[:, -1] + frame_state.unsqueeze(1)   # [B,L,D_H]
q = self.project_to_q(h_last)                          # [B,L,8]
```
```python
# stage_b/model.py  project_to_q()
self.projection = nn.Sequential(
    nn.Linear(latent_dim, 256), nn.SiLU(),
    nn.Linear(256, 128), nn.SiLU(),
    nn.Linear(128, state_dim),   # state_dim = 8
)
```
- **Q_t 是一个 256→8 的 MLP 有损压缩**（两次降维）。
- **H 的时间维度被 `[:, -1]` 完全丢掉**，只留下一个静止快照 `h_last`。
- `state.py`: `LatentState(q:[B,L,8], m:[B,L,16], e:[B,8])` —— Q 被**命名为状态本身**，而不是"状态的因子坐标视角"。

### 2.3 Stage B 的损失：全是力学/能量，没有 latent 预测

`stage_b/objectives.py`：`transition_nll`、`force_floor`、`force_direction`、`rollout_mse`(n步自主 rollout 端点 MSE)、`semigroup_consistency`、能量正则。全部是**在 Q 上做 SDE 能量景观拟合**。
- `train_iter008_stage_b.py` 顶部注释自述根因：*"The frozen H carries NO per-residue kinematic signal… Q = proj(H) is therefore an empty per-residue latent: mag R² ~ 0.002"*。他们的"修复"是**额外塞入 `kin_feat` 到 Q**（`Q = proj_base(h_last) + proj_kin([h_last, kin_feat])`），并用辅助回归头（disp_mag / disp_dir / rg / rmsf）去"塑造"Q。
- 也就是说，**Q 里的动力学信息不是由 JEPA 预测目标产生的，而是靠监督回归头硬灌进去的**。

---

## 3. 问题定位：到底哪里违背了 JEPA

### ❌ 问题 1（致命）：把"因子坐标"当成了"状态"，且做成了有损降维

JEPA-Anything 的硬约束是 **`K*r = d`，`rank(A)=d`，`pinv(A)` 无损往返**。
你的实现是 **`d=256 → state_dim=8`**，是纯降维，**不存在返回 256 维的 synthesis 接口**，也没有 rank/NMSE/条件数审计。

- **后果**：JEPA-Anything 里 `z` 和 `y` 是同一信息的两种坐标；你这里 Q 只是 H 的一个**下采样投影**。把它叫 "latent state" 意味着你**宣称**这 8 维足以决定未来，但流水线里**没有任何东西保证或检验这件事**（README §7 自己要求 `I(u_t;a_t;y_{t+Δt}) ≈ I(obs_{≤t};y_{t+Δt})` 的 sufficiency 审计，且明说 "Koopman compatibility alone does not establish thermodynamic sufficiency"）。
- **更糟**：Stage B 的 force-floor / auxiliary heads 会**把 Q 拉向一个只利于力学拟合的子空间**，进一步破坏 H 作为预测表征的结构（README §11.1 甚至提到 Stage B 里让 H→Q projection 联合训练）。

### ❌ 问题 2（结构）：JEPA 的预测契约在 Stage B 断掉了

JEPA 的定义性特征是 **"predict the future in latent space, compare with stop-grad target"**。
- Stage A：有（`state_predictor` + EMA target + mask prediction）。✅
- Stage B：**没有**。Stage B 的 target 是观测坐标经 rollout MSE / NLL 对齐的能量一致性，**不是** "online encoder 对 `t+Δt` 的表示经 stop-grad 得到的 latent target"。`rollout_mse` 比的是 Q 自主演化后的端点，本质还是一个能量/积分器要"复现观测位移"，不是 latent prediction。

于是整个流水线里，**唯一在 latent 空间做 predict-then-compare 的是 Stage A**，而 Stage B 却把 Stage A 的产物 `H(T维)` 塌缩成 `Q(单帧)` 后**换个目标函数重训**。JEPA 的核心机制在 Stage B 被替换成了"物理能量拟合"。这不是 JEPA-Anything 说的"用同一个预测边界贯穿一个消费者"，而是**换了预测边界、换了目标、换了状态**。

### ❌ 问题 3：时间维度在 Stage B 被直接丢弃

Stage A 的 H 是 `[B,T,L,D]`（你说得对，H 有时序）。Stage B 用 `frame_states[:, -1]` 和 `context_latents[:, -1]`。
- 唯一的时序信息通过 `temporal_encoder` 的**最后一帧聚合**进入 `h_last`，**没有把整条 H(t) 序列作为条件或作为要预测的对象**。
- 对一个 `repeated_transition` 消费者（你的 rollout），JEPA-Anything 要求状态能被"反复推进"，且**每个 horizon 都要评估**；同时 `L_dyn` 要求和 integrator 一致的 SDE。你的 Q 是**静态端点**，rollout 靠的是能量+SDE 自己积分，**没有 JEPA 的"多 horizon latent 预测"**。Stage A 好不容易学到的多 horizon 表示，在 Stage B 被浪费。

### ❌ 问题 4：C 与 D 的名字相同、语义不同，全流水线没有可审计的接口

JEPA-Anything 的 `state` 契约（`d,K,r,coordinate_layout:"concatenate"` + `analysis.learned_projectors` + `synthesis.moore_penrose_pseudoinverse`）在你仓库里**完全没实现**（有 `opf_enabled` 开关，但 Stage B 主路径 `project_to_q` 是普通 MLP，不是正交投影 + 伪逆合成）。
- 因此 JEPA-Anything §3.4 的四条损失（factor MSE / Gram / factor activity / encoder activity）**都无法挂载**。
- 你缺 `factor_activity` 与 `online_encoder_activity`（两个 std 下限）——这正是 JEPA 防塌缩的机制。`train_iter008` 自己也观察到 *"frozen H carries NO per-residue kinematic signal"*、Q 近乎常量（mag R² ~ 0.002）——**这就是未施加 activity 约束导致的表征塌缩的典型症状**。

### ❌ 问题 5：优化顺序与 STOP-GRADIENT 原则冲突

- JEPA-Anything：目标因子 `P_k^T z_target` 必须 **stop-grad**；online encoder 只通过预测损失更新。
- 你的 Stage B：加载**冻结**的 Stage A 权重，然后**用辅助回归头 / force-floor 反过来塑造 Q**，`train_iter008` 甚至把 Q 的构造改成 `proj_base + proj_kin(...)` 直接注入 `kin_feat`（context-only kinematic features）。这已经**越过了"预测未来"**，变成"把已知的 context 运动学回归进 Q"——是**表示泄漏/捷径**，JEPA-Anything 的 leakage 审计（`feature_lineage`, `conditioning_lineage`）会直接判 fail。
- 如果 Q 来自 context 帧内的运动学（`kin_feat` 是对 context 求的），而你要它预测 future displacement，那些 future 信息**并非通过 latent prediction 得到**，而是通过一个监督头——这**动摇了 JEPA 叙事**：它不再是自监督 latent 预测，而是**有监督的手工特征注入**。

---

## 4. 与 JEPA-Anything 契约的逐条对照表

| JEPA-Anything 契约 | 你的框架 | 判定 |
|---|---|---|
| `K*r = d`，因子坐标与完整态同维、`pinv(A)` 无损 | `256 → 8` 有损 MLP 降维 | ❌ 违反 |
| 因子 = "换坐标视角"，非压缩 | Q 被当作唯一小状态 | ❌ 违反 |
| `rank(A)=d`、条件数、synthesis-NMSE 审计 | 无任何 synthesis 接口/审计 | ❌ 缺失 |
| `factor_prediction` 用 stop-grad 目标因子 | Stage B 无此损失 | ❌ 缺失 |
| projector Gram（几何正交，非统计） | `opf_enabled` 存在但主路径非正交投影 | ⚠️ 未启用 |
| `factor_activity` / `online_encoder_activity` std 下限 | 无（症状：Q 塌缩到 mag R²≈0.002） | ❌ 缺失 |
| 一个预测边界贯穿一个消费者 | Stage A 边界≠Stage B 边界，目标函数不同 | ❌ 违反 |
| `repeated_transition` 要能被反复推进、逐 horizon 评估 | Q 是静态端点，靠外部 SDE 积分 | ⚠️ 部分 |
| 无损/无泄漏 lineage 审计 | `kin_feat` 注入 Q 属于表示捷径 | ❌ 违反风险 |
| 目标 stop-grad / EMA | Stage A 有；Stage B 靠冻结权重 | ⚠️ 部分 |

---

## 5. 根因归纳（一句话）

> **你把 JEPA-Anything 里"同一个 d 维状态的正交因子视角"错误地实现成了"从 H 到 Q 的有损降维"，并且在 Stage B 用物理能量拟合替换了 JEPA 的 latent 预测契约；同时丢弃了 Stage A 的时序结构，也没有施加防塌缩的 activity 约束。**

三个独立的错误叠加：
1. **几何错误**：把"因子坐标"（=d 维无损）当成"降维状态"（≪d 维有损）。
2. **契约错误**：两阶段预测边界不一致，Stage B 不再是 latent prediction。
3. **正则错误**：缺失 activity/Gram 约束 → 表征塌缩 + 为救塌缩而注入监督特征 → 泄漏。

---

## 6. 修复方向（对齐 JEPA-Anything）

### 方案 A（推荐，最小改动、保留物理叙事）
把 Stage B 从"压缩成 Q_t"改为"**H_t 的预测因子视角**"，并补回 JEPA 契约：

1. **Stage A 产出完整 d 维 latent state `z_t`（= H 的 pooled d 维），并 `K*r=d` 分解**。用 `jepa_anything_core.opf` 的 `OrthogonalFactorProjection(learnable=True, orthogonality_mode="soft_gram")` 替换 `project_to_q` 的 MLP；`Q` 变成 `decompose_state(z)` 的 `[K,r]` 坐标，`compose_state` 用 `pinv(A)` 无损回 `d` 维。
2. **把 A 的时序保留下来**：State 输入用 `H(t)` 序列（或 temporal_encoder 的 `frame_states` 全序列），不要 `[:, -1]`。让因子预测在**每个 horizon** 上运行（对齐 `repeated_transition` 的 `rollout_horizons`）。
3. **补上四条损失**：`factor_prediction`（stop-grad 目标因子）+ `projector_orthogonality`（within+cross）+ `factor_activity` + `online_encoder_activity`。直接复用 `jepa_anything_core.losses` 的 `projector_orthogonality_loss` / `factor_activity_loss` / `online_encoder_activity`。
4. **物理能量项**作为**辅助**（`L_dyn`, `L_eq`），而不是替代 latent 预测。能量应该在**完整 z / 因子坐标**上定义，且保证 `pinv(A)` 可回。
5. **删掉 `kin_feat` 注入 Q 的捷径**，改为让 activity + factor prediction 自然地把运动学信号留在 latent 里；或把 `kin_feat` 作为**合法的 exogenous conditioning**（明确列为 `exogenous input`，声明 availability cutoff），而不是塞进被当作状态的对象。

### 方案 B（如果坚持要 8 维小状态）
那么必须**诚实声明这是 `terminal_readout` / 有损 bottleneck**，并：
- 提供 H→Q 的**信息审计**（predictive sufficiency probe，README §7 自己已要求）；
- 明确 Q 不是完整 state，配套一个 observation model `L_obs`；
- 不要用 JEPA 的名字声称它，因为 JEPA 的因子契约是**无损**的。

### 立即可做的验证实验（证明问题存在）
1. **可逆性测试**：加一个 `H → Q → pinv → H'` 的 synthesis 接口，测 `NMSE(H,H')`。当前结构下必然很差（256→8 无逆），直接证实"这不是 JEPA 因子"。
2. **因子 activity 诊断**：统计 Q 每个坐标的 std（对齐 `coordinate_std_floor`）。`train_iter008` 已显示接近塌缩。
3. **latent prediction probe**：在 Q 上训一个 probe 去预测 `Q_{t+Δt}`（EMA target），看 R²。若接近 0，说明 Stage B 的 Q 不具备 JEPA 要求的可预测性。
4. **leakage audit**：检查 `kin_feat` 的 lineage 是否 only context-frames；对齐 JEPA-Anything 的 `target_leakage` 六项。
5. **逐 horizon 评估**：把 Stage A 的多 horizon 表示能力与 Stage B rollout 对齐评估，确认时序结构是否被浪费。

---

## 7. 附：参考实现位置

- 契约定义：`jepa_anything/JEPA-Anything-main/jepa-anything-skill/references/{design-protocol,config-contract,world-model-conversion}.md`
- 信任边界/失败语义：`.../docs/architecture.md`
- 可复用代码：
  - `jepa-anything-core/src/jepa_anything_core/opf.py`（`OrthogonalFactorProjection`, `decompose_state`, `compose_state`(pinv), `projector` 几何）
  - `jepa-anything-core/src/jepa_anything_core/losses.py`（`projector_orthogonality_loss`, `factor_activity_loss`, `encoder_variance_loss`, `factor_cross_correlation`）
- 你的关键代码：
  - `protein_jepa_frame/models/stage_a/model.py`（H 的时序产出，JEPA 部分正确）
  - `protein_jepa_frame/models/stage_b/model.py:195-210`（`project_to_q` 有损降维）、`:429-441`（`frame_states[:,-1]` 丢时序）
  - `protein_jepa_frame/models/stage_b/state.py`（Q 被命名为 full state）
  - `protein_jepa_frame/models/stage_b/objectives.py`（全是力学/能量，无 latent prediction）
  - `train_iter008_stage_b.py`（`kin_feat` 注入 Q 的捷径 + 辅助回归头）
