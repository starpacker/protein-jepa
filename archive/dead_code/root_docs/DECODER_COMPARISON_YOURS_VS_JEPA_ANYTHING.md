# Decoder 对比:你的 Q_t→原子坐标  vs  JEPA-Anything 的 z→position

**日期:** 2026-10-13
**结论先行:** JEPA-Anything **根本没有从 z 到原子坐标的 decoder**。它刻意把"latent 世界模型"和"观测解码器"划成两个分离的东西。你的框架里有 decoder,这**不是错误**,但你必须理解 JEPA-Anything 为什么不做、以及它把这件事交给谁。

---

## 0. 一句话回答你的问题

> **JEPA-Anything 在具体的分子位置预测任务上,如何从 z decode 得到 position?**

**答案是:它不 decode。** 在 JEPA-Anything 的契约里:

- **预测目标是 latent(predictive coordinates),不是原子坐标。**
- **decoder 不在这个框架里** —— 它的存在被明确"外置"成一个独立的、被约束的观测模型 `p_ψ(x, y | u, c)`,属于"下游/观测层",不属于 JEPA 内核。
- 框架内唯一的"从 coords 回 state"操作是 **`synthesize = pinv(A) y`**,即"因子坐标 → 完整 d 维 latent state",**不是"latent → 原子坐标"**。
- 位置(原子坐标)在 JEPA-Anything 里是**观测空间(x, y)**,它是**输入**,不是被预测的输出。被预测的是它在 latent 里的对应物 `z_{t+Δ}`。

---

## 1. JEPA-Anything 的完整"位置"处理链(证据)

### 1.1 原子坐标是 **observation(x,y)**,是输入侧

`world-model-conversion.md` §1 的原始任务形式:
```
system and instance -> available history -> known action/input -> future target
                    -> intended consumer -> evaluation
```
对分子/结构类任务,`observation` 指的就是原子坐标/结构;它经 **adapter** 变成 tokens:
```
观察像素/坐标/结构 → adapter(逐 token/逐帧) → encoder → z(d维)
```
`config-contract.md` 里 adapter 只有两个 allowlist:`context_input_fields` 和 `target_input_fields`,**全部是观测字段**。没有任何"decode 回这些字段"的接口。

### 1.2 预测发生在 latent 空间,不进观测空间

`design-protocol.md` §4 第 1 条:
```
L_factor = mean_k ||factor_hat_k - stopgrad(P_k^T z_target)||^2
```
预测的是**因子坐标**,target 是 **target-encoder 对 future 观测编码后的 stop-grad 因子**。**从头到尾没有"predict future atom positions"这一项。** 这是 JEPA 的定义性选择——预测在表示空间做,不重建观测。

### 1.3 唯一的"回投"是 latent→latent,不是 latent→坐标

`opf.py::compose_state`:
```
z_tilde = pinv(A) y      # y 是 (K,r) 因子坐标, 输出 (d,) 完整 latent
```
`audit.py::audit_state_round_trip`:测 `compose_state(decompose_state(z))` 的 NMSE。
**这是 z⇄y 的往返,不涉及原子坐标。** 框架里**没有任何一个函数**把 d 维 z 映射回 3D 原子位置。

### 1.4 观测模型被显式外置

README §2 五条边界里,观测模型是**独立一层**:
```
constrained observation model p_psi(x,y | u,c)     # 在架构图最底部
```
`architecture.md` 的信任边界表把 **"Semantic interpretation"** 和 **"Training"** 都划在 Skill 之外。也就是说:
- JEPA-Anything 只负责 **latent 世界状态接口**;
- 从 latent 到**可观测的原子坐标**的映射,是**另一个观测模型 p_ψ** 的责任,**不在这个框架的契约里**,也不受 `d/K/r`/OPF 约束。

### 1.5 下游消费模式里,也没有"decode 到坐标"

三种 mode(`usage.py.tmpl`):
- `terminal_readout`:状态用一次 → readout **任务**(如分类/回归到某个标量性质),不是生成坐标;
- `repeated_transition`:在 latent 空间反复推进 `z_t → z_{t+1}`;
- `factor_analysis`:对坐标做 probe/intervention。

**注意 `repeated_transition` 的 rollout 全程只在 latent 空间**(`usage.py.tmpl` 里 `current = transition(current, ...)`,永远是 d 维状态),**不 decode 回原子坐标**。要评结构指标,是靠**外置的观测模型/评估协议**,不是内核。

---

## 2. 你的框架:Q_t → 原子坐标 decoder

### 2.1 你的实际链路

```
coords ─[Stage A encoder]→ H/z[B,T,L,256]   (JEPA, latent)
                 ↓ 压缩(现在是有损 MLP/[:,-1])
             Q_t[B,L,8]                       (瞬时物理坐标,进朗之万)
                 ↓ 力学 rollout(能量/FDT/Mori浴)
             Q_rollout[B,L,8]
                 ↓ [Stage C structure_decoder]  ← 你在这一层 decode!
             atom37 坐标[B,L,37,3]
```

代码证据:
- `protein_jepa_frame/models/stage_c/structure_decoder.py`(871 行)—— **这就是你的 decoder**,从 latent Q 生成原子坐标。
- `models/stage_b/model.py::step()` —— Q 在物理方程里推进。
- README §11.5 提到 `L_obs`:"constrained structure/environment observation model",对应 Stage C。

### 2.2 关键区别:你的 Q_t 有**物理坐标的最小可辨识要求**

你的 Q_t 不是普通 latent —— 它要进 `−dA/dQ`(力是能量的梯度)、要满足 FDT、要作为朗之万的广义坐标。这意味着:

- **Q_t 天然比纯 pose-latent 更接近"坐标语义"**(它是被物理方程约束出来的广义坐标);
- 所以"从 Q_t decode 到原子坐标"**在物理上是有意义、可解释的**——因为 Q_t 本来就是物理坐标的压缩。
- JEPA-Anything 的 z **不做这个承诺**:它是纯粹的 predictive latent,`architecture.md` 甚至禁止给坐标起语义名(用 `pc_000`)。

---

## 3. 逐条对比

| 维度 | JEPA-Anything | 你的框架 |
|---|---|---|
| 原子坐标的角色 | **观测 x**,adapter 输入 | 既是**输入**(encoder),又是**输出**(Stage C decoder) |
| 被预测的对象 | latent 因子坐标 `y_{t+Δ}` | 先 latent Q,再 **decode 成原子坐标** |
| 从 state 回投 | `pinv(A) y` → **仍是 latent** | `structure_decoder(Q)` → **原子坐标** |
| decoder 归属 | **外置**为 `p_ψ(x,y|u,c)`,不在契约内 | **内置于 Stage C**,是训练图的一部分 |
| 是否重建观测 | **不**。预测在 latent 做,不重建 | **是**。Stage C 显式重建 atom37 |
| 坐标语义 | **禁止**赋予语义(`pc_000`) | Q_t **被赋予物理意义**(广义坐标) |
| 评价位置 | 靠外置观测模型 + 结构指标(框架外) | 直接评 decode 出的坐标 vs ground truth |
| 泄漏防线 | adapter allowlist + lineage 审计 | **没有对 decoder 的等价约束** |

---

## 4. 这意味着什么(对你的设计)

### 4.1 你**没有**违背 JEPA 的内核——只要你满足一个条件

JEPA 内核的要求是:**预测/表示学习在 latent 空间做**。你的 Stage A/B 满足这点(Stage A 有 stop-grad latent 预测;若按上一轮补 `L_Q`,Stage B 也满足)。
你**额外加了一个 Stage C decoder**,这相当于 JEPA-Anything 里那个**外置的观测模型 `p_ψ`**。**这是允许的,而且是生物/分子领域必须的**——因为论文要展示结构指标(atom37 RMSD 等),必须能回到坐标。

**前提**:decoder 必须被当作"观测模型层",遵守三件事:
1. **它的输入只能是被预测的 latent**(Q_rollout),不能偷看未来真值;
2. **它不能反过来污染 latent 的训练目标**(即 decoder 的损失不应该成为"把 future 坐标塞进 Q"的捷径);
3. **它要接受自己的可辨识性审计**(decoder 能不能真的从 Q 恢复坐标?这是 `L_obs` / observation model 的职责)。

### 4.2 你的潜在风险(也是最该查的)

你目前的问题是:**你有一个内嵌 decoder,却缺少 JEPA-Anything 对 latent 的"防塌缩 + 防泄漏"纪律**。导致:

- Q 塌缩(mag R²≈0.002)→ decoder 学不动 → 有人用 `kin_feat` 注入硬修 → **这等于让 decoder 的便捷反馈去污染 latent**。
- 这正是 JEPA-Anything 把观测模型**外置**(且不参与 latent 预测损失)的原因:**防止重建目标把 latent 拉向"容易重建"而不是"容易预测"的子空间**。

### 4.3 建议的定位

| 层 | 你的对应物 | JEPA-Anything 对应 | 纪律 |
|---|---|---|---|
| 表示/预测 | Stage A(+ Stage B 的 `L_Q`) | encoder + predictor + L_factor | **JEPA 契约**:stop-grad 预测 + activity 防塌缩 |
| 物理动力学校准 | Stage B 朗之万/Mori | **无直接对应**(它是通用框架,不假设 SDE) | 物理约束挂在 Q_t 上 |
| 观测解码 | **Stage C structure_decoder** | **外置 `p_ψ(x,y|u,c)`** | 只消费 latent;不污染 latent 目标;自审可辨识性 |

**关键**:你的 Stage C decoder **应该对标 JEPA-Anything 的外置观测模型,而不是内核**。所以:
- ✅ 保留它(分子任务必须);
- ✅ 但**明确它是"观测模型层"**,在论文里这样定位;
- ⚠️ **确保它的存在不被当作"latent 可以随便塌缩,反正 decoder 能救"的借口**;
- ⚠️ **确保没有从 decoder 反传到 context encoder 的"未来坐标捷径"**。

---

## 5. 直接回答:两个问题合起来

**Q1: JEPA-Anything 如何从 z decode 到 position?**
**它不 decode。** 位置是**观测输入**,预测在 latent 空间完成,唯一的"回投"是 `pinv(A)y` 回到 latent。要把 latent 变成原子坐标,得用**外置的观测模型 `p_ψ(x,y|u,c)`**,那不在 JEPA-Anything 的契约内,也不受 OPF 约束。

**Q2: 那你(内嵌 decoder)的差异意味着什么?**
意味着**你的框架 = JEPA-Anything 内核 + 一个你必须自己负责的观测模型层**。这不是缺陷;但你必须:
1. 用 JEPA 的纪律管住 latent(stop-grad 预测 + activity 防塌缩);
2. 把 Stage C decoder 明确定位为"观测模型层",只消费 latent、不污染 latent;
3. 杜绝用 decoder/`kin_feat` 的反馈去"救"塌缩的 latent。

**一句话**:JEPA-Anything 把 decoder 拿出去是**为了不让重建目标扭曲 latent**;你把 decoder 放进来**是分子任务的必要**,但你得**自己承担 JEPA-Anything 用"外置"换来的那份纪律**——否则 decoder 会变成掩盖 latent 塌缩的创可贴,而这正是你现在 `train_iter008` 遇到的症状。
