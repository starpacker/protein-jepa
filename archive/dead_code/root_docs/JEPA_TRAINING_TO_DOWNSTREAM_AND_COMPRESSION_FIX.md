# JEPA-Anything 全链路逻辑 + 你的压缩问题与解法

**日期:** 2026-10-13
**配套文件:** `JEPA_COMPLIANCE_REVIEW.md`(问题定位)

---

# 第一部分:JEPA-Anything 从训练 latent state 到下游任务的完整逻辑与维度变化

## 1.1 核心不变量

```
d = K × r
state (..., d)  ⇄  predictive coordinates (..., K, r)
```

- `d`:完整 latent state 宽度
- `K`:因子组数
- `r`:每组宽度
- **关键:decompose 与 synthesize 是无损的(up to 数值误差),因为基是完备正交基。**

## 1.2 数据与视图(维度变化 #1:obs → tokens)

```
context tokens : (B, T_ctx, N_tok, D_feat)   # 只到 context_end
target  tokens : (B, T_tgt, N_tok, D_feat)   # 严格 future
```
约束 `context.start <= context.end < target.start <= target.end`。两个 allowlist + leakage 审计。

## 1.3 编码(维度变化 #2:tokens → state)

```
context_state = encoder_context(context_tokens) → (B, d)   # online
target_state  = encoder_target(target_tokens)   → (B, d)   # stop-grad / EMA
```
共享 encoder 家族与宽度,points 同空间。

## 1.4 分析(维度变化 #3:state → factor coordinates)

projector `P_k:[d,r]`,`A = stack(P_0^T..P_{K-1}^T)` 形状 `[Kr,d]=[d,d]`。
```
y = concat(P_0^T z, ..., P_{K-1}^T z) = A z   # (B,K,r), 共 Kr=d 个
```
**不是降维**,是换正交坐标看同一个 d 维态。

## 1.5 预测(维度变化 #4:→ predicted factor blocks)

```
pred_factor_k = predictor_k(context_state, target_descriptor, exogenous)  # (B,r)
L_factor = mean_k || pred_factor_k - stopgrad(P_k^T z_target) ||^2
```

## 1.6 状态合成(维度变化 #5:factor coords → complete state)

```
z_tilde = pinv(A) y   # (B,d)  ← Moore-Penrose 伪逆,不是简单 concat
```
审计:`rank(A)=d`、`cond <= max`、`synthesis_nmse <= 1e-6`。**往返无损。**

## 1.7 下游(维度变化 #6:state → 任务输出)

| mode | 逻辑 | 维度流 |
|---|---|---|
| `terminal_readout` | 用一次 | `z(B,d) → readout → output` |
| `repeated_transition` | 反复推进 | `z_t → z_{t+1} → ... → z_H`(逐 horizon 评估) |
| `factor_analysis` | probe/intervene | `(B,K,r) → probe → 结论` |

rollout 实现:
```
current = z
for step in 1..max(horizons):
    current = transition(current, step, exogenous[step])  # 始终 d 维
    if step in horizons: record(current)
```

## 1.8 四条损失(必须分离)

1. `factor_prediction`:`mean_k ||pred_k - sg(P_k^T z_tgt)||^2`
2. `projector_orthogonality`:`Σ_k ||P_k^T P_k - I||² + Σ_{i<j} ||P_i^T P_j||²`(几何正交)
3. `factor_activity`:每个 `(K,r)` 坐标 std 下限
4. `online_encoder_activity`:online encoder `0..d-1` 每维 std 下限

> 3/4 是 hinge `relu(min_std-std)`,完全常数因子是其对称驻点(grad=0),需非退化初始化 + 不对称预测损失打破。

## 1.9 基线

- `standard_jepa`:输出 d 宽
- `unconstrained_multihead`:`K` 个 r 宽头(`heads=K, head_dim=r`)
- `d=K*r` 时参数量严格相等,还需匹配 FLOPs、步数。

## 1.10 全链路维度一句话

```
obs_tokens(B,T,N,Df) → [encoder] → z(B,d) → [analyze A] → y(B,K,r)   # 无损, Kr=d
                                       ↑                                   ↓ [predict]
                                  z_tgt(B,d) ──[sg P_k^T]──→ target coords(B,K,r)
                                                                         ↓
                                z_tilde = pinv(A) y (B,d) ← [synthesize]
                                       ↓
                        downstream: readout / repeated_transition / factor_analysis
```
**贯穿始终:state 永远是 d 维;因子坐标也是 d 个分量;二者是双射。**

---

# 第二部分:你的问题重新表述

## 2.1 你的物理诉求是对的

郎之万方程 / `-dA/dQ` / FDT 噪声 / Mori 浴 作用在**瞬时、有限维、可微的物理坐标 q_t** 上。`H[B,T,L,D]` 是时间窗表示,不能直接进 `-dA/dQ` 的梯度。**所以"时间窗 → 瞬时物理坐标"的映射(压缩)是必要的,不是错误。**

代码事实:
- `StageBLASP.step()`:Euler-Maruyama,`drift = sub_dt·D·(-dA/dQ) + sub_dt·C^T S`,`q_next = q + drift + sqrt(2·sub_dt·D)·ξ`。
- `dA/dQ` = `autograd.grad(energy_total, q)`,**q 必须是 `[B,L,8]` 瞬时坐标**。
- `S_t`(`[B,L,16]`)= `bath_initializer(h_last)`,只调一次。

所以你需要 `Compress: H[B,T,L,D] → (Q[B,L,8], S[B,L,16])`。**这个压缩本身合法。**

## 2.2 真正的问题:一次性有损映射 + 无 JEPA 契约

```
H[B,T,L,256] ──[temporal_encoder 只取 :,-1]──→ h_last[B,L,256]
             ──[MLP 256→256→128→8]──────────→ Q[B,L,8]
```
- 丢时间(`[:,-1]`)、丢维度(256→8 无逆)、无 predict 契约、无 activity(→塌缩:mag R²≈0.002)、靠 kin_feat 注入硬修(→泄漏)。

## 2.3 你识别的核心矛盾会互相放大

1. H 没到位 → 压缩器抓不到动力学。
2. 压缩有损 → H 到位也装不下。
3. 监督头抢救 Q → 变成监督回归,破坏 JEPA + 泄漏。
4. Stage B force-floor/NLL 反向拉扯 Q → 损害 H 的预测结构。

**四者构成恶性循环。**

---

# 第三部分:解法(打破恶性循环)

核心思想:**把"有损压缩"改成"带 JEPA 契约的无损因子化 + 物理子空间",并把 H 训练与物理校准解耦成可审计的两段。**

## 方案 A(推荐):OPF 无损因子化 + 物理子空间

### A.1 用 OPF 替换 256→8 MLP
```python
from jepa_anything_core import OrthogonalFactorProjection
opf = OrthogonalFactorProjection(state_dim=d, num_factors=K, factor_dim=r, learnable=True)
coords = opf(z)            # (B,K,r)
z_rec  = opf.compose(coords) # (B,d) 用 pinv,无损
```
保证 `rank(A)=d`、无损、可审计。

### A.2 物理坐标 = 因子坐标切片
**(A2a,推荐)** 把 K 个因子分为"物理组"(进 SDE)与"记忆/环境组"(Mori 浴 S 来源):
```
q_phys = pinv(A_phys) @ concat(coords[phys_group_ids])  # 维度 = K_phys*r
```
q 从无损因子化中切片,可回投 z,而不是独立黑箱。
**(A2b)** 用有损 bottleneck 但显式建模 + 审计(见方案 B)。

### A.3 补回 JEPA 契约(Stage A 和 B 都加)
```python
L_factor = mean_k || pred_coord_k - sg(analyze(target_z)_k)||^2
L_gram   = projector_orthogonality_loss(opf.analysis_basis())
L_fact   = factor_activity_loss(coords, min_std=...)
L_enc    = encoder_variance_loss(context_tokens, valid_mask=..., min_std=...)
```
直接 import `jepa_anything_core.losses`。

### A.4 时序:不要只取 [:,-1]
- `z_t` 由 temporal_encoder 全序列聚合(可 last+mean+attn pool 拼接)。
- SDE 初始 `(Q_t,S_t)` 从 `z_t` 物理子空间 + 全序列得到。
- 对每个 horizon 都施加 `L_factor`。

### A.5 物理约束作为辅助
```
L_total = L_factor + w_gram·L_gram + w_act·(L_fact+L_enc)
        + w_dyn·L_dyn + w_force·L_force + w_semi·L_semigroup
```
物理项挂在 `q_phys` 上,latent 预测损失始终存在。

### A.6 删捷径
删掉 `Q = proj_base + proj_kin(...)` 和辅助回归头。kin_feat 若有用,声明为合法 exogenous input(带 cutoff),走预测通路。

**效果**:压缩无损、时序不丢、activity 防塌缩、物理约束挂合法坐标、无泄漏。

## 方案 B(若坚持 8 维小状态):诚实声明有损 bottleneck

1. 声明 usage=`terminal_readout` 或有损 bottleneck,记录 `preservation_error`。
2. 加信息审计(README §7 已要求):Q 上的 `Q_{t+Δt}` probe R²;`I(Q;y) vs I(H;y)` sufficiency;Q 对 H 的 explain-variance。
3. **先做 Stage A.5 物理对齐**:冻结 H,只训 H→q 的 encoder,用 L_dyn/L_force 校准,**保留 activity 防 q 塌缩**;测 `I(H;q)`、`I(q;y_future)`。
4. **禁止 kin_feat 注入 Q**(泄漏)。

论文里必须**诚实**说这是有损 bottleneck,而非 JEPA 因子(后者契约是无损的)。

## 方案 C(快速止血):先加诊断让恶性循环显形

### C.1 三个必测诊断
1. **可逆性 NMSE**:加临时 `H→Q→MLP_inv→H'` 头(仅诊断),测 NMSE。256→8 必然很差。
2. **因子活性**:统计 Q 每坐标 std。
3. **latent prediction probe**:Q 上预测 `Q_{t+Δt}`(EMA target)的 R²。≈0 → 不具备 JEPA 可预测性。

### C.2 泄漏审计
对齐 `target_leakage` 六项,查 `kin_feat` lineage(cutoff 是否在 prediction time 前)。

### C.3 H 质量诊断
- frozen H 做 `I(H;y_future)` probe(held-out)。
- 对比 `I(obs;y_future)`,直接回答"H 有没有训练到位"。

---

# 第四部分:推荐落地顺序

```
Step 0 [诊断] 跑 C.1–C.3,定量确认 ①压缩损失 ②H 里的未来信息
Step 1 [决策] H 未来信息低 → 先修 Stage A(补 activity + 多 horizon 因子预测)
              H 不错但 Q 丢信息 → 走方案 A 的 OPF 重构
Step 2 [重构] 用 OrthogonalFactorProjection 替换 256→8 MLP(A2a)
Step 3 [契约] 接上四条损失(import jepa_anything_core)
Step 4 [时序] 去掉 [:,-1],SDE 初始条件来自全序列 + 物理子空间
Step 5 [清捷径] 删 kin_feat 注入 / 辅助头,或改声明为 exogenous
Step 6 [审计] geometry + activity + leakage + capacity 审计
Step 7 [下游] usage=repeated_transition,逐 horizon 评估,对比 standard_jepa + multihead
```

## 关于"H 没训练到位"

这是**最该先解决的**,压缩器只是放大器。
1. 冻结压缩器,单独用 probe 测 `I(H; future_disp/dir/Rg)`,和 `I(obs;...)` 比。差距大 → H 是瓶颈本体。
2. 给 Stage A 补 activity + 多 horizon factor prediction,重训到 H 的 probe 接近 obs 天花板。
3. H 没到位前,不要用 kin_feat 抢救 Q(会掩盖问题、破坏叙事)。
4. 用 `L_factor`(stop-grad)显式驱动 H 的物理子空间,让 H 改善与物理校准由**同一损失**推动。

---

# 附录:关键文件对照

**JEPA-Anything**
- 契约:`jepa-anything-skill/references/{design-protocol,config-contract,world-model-conversion}.md`
- 边界:`docs/architecture.md`
- 代码:`jepa-anything-core/.../opf.py`(`OrthogonalFactorProjection/decompose_state/compose_state`)、`losses.py`、`baselines.py`
- 示例:`recipes/synthetic-linear-dynamics/design.expected.json`(d=4,K=2,r=2)
- 下游 dispatcher:`jepa-anything-skill/assets/scaffold/src/jepa_task/usage.py.tmpl`

**你的框架**
- Stage A(产 H):`protein_jepa_frame/models/stage_a/model.py`
- 压缩点:`protein_jepa_frame/models/stage_b/model.py`(`project_to_q` L102-103,`encode_state` L429-441 `[:,-1]`)
- 物理 SDE:`protein_jepa_frame/models/stage_b/model.py` `step()`、`compute_energy_and_forces`
- 状态容器:`protein_jepa_frame/models/stage_b/state.py`
- 损失:`protein_jepa_frame/models/stage_b/objectives.py`
- 捷径:`train_iter008_stage_b.py`
