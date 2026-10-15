# PLAN: Stage A 对齐 JEPA-Anything (OPF) + Stage B 有损物理 Bottleneck 重设计

**版本:** v1
**日期:** 2026-10-13
**状态:** 待评审
**相关报告:** `JEPA_COMPLIANCE_REVIEW.md`, `JEPA_TRAINING_TO_DOWNSTREAM_AND_COMPRESSION_FIX.md`, `DECODER_COMPARISON_YOURS_VS_JEPA_ANYTHING.md`

---

## 0. 目标与一句话总结

**目标:** 解决两个核心痛点 —— (1) latent state `z` 训练不到位;(2) 从 `z` 到 `Q_T, S_T` 的压缩信息损失过大。

**方案一句话:**
> **Stage A 严格照搬 JEPA-Anything**,对 encoder 输出的 state `z` 做**无损正交因子化(OPF,`K*r=d=128`)**,并打开 anti-collapse 正则;
> **Stage B 用两个独立 MLP** 从 `z[B,T,L,128]` **有损映射**到 `Q_T`(物理坐标,进化简之万)与 `S_T`(记忆态,进 Mori 浴),**不受正交约束**,并加 `L_Q_keep` 保证动力学信息不丢;
> **Stage C 不变。**

**设计原则:** OPF 负责"z 的内部几何(无损)",MLP 负责"物理语义抽取(有损)"。两层职责分离,互不约束。

---

## 1. 现状诊断(为什么需要这个 plan)

### 1.1 现状事实(代码核对)

| 组件 | 现状 | 问题 |
|---|---|---|
| Stage A OPF | `OPFPredictor` 对 **`delta_target`** 做因子化(`jepa.py:508`) | ❌ OPF 作用在**预测目标**上,不是 JEPA-Anything 要求的"对 **latent state** 因子化" |
| Stage A 正则 | `objectives.py` 有 `coordinate_activity_regularizer` / `temporal_variance_regularizer` / `temporal_velocity_loss` / `temporal_covariance_regularizer` | ❌ **`train_iter009_stage_a.py` 未启用**(未传入 `stage_a_total_loss`) |
| Stage A 主力损失 | `paired_delta_loss`(w=0.5)+ `displacement_head`(w=1.0) | ⚠️ displacement 是 3D 坐标回归,把 z 往"易回归坐标"拉,权重过高 |
| Stage B 压缩 | `frame_states[:,-1]` + `projection` MLP(128→256→128→8) | ❌ 丢时间维;有损;无 keep-alive 损失 |
| Stage B S_T | `S` 内容来自 `h_last`(H 空间),但 `Γ/C` 只依赖 `Q` | ❌ 记忆内容与记忆动力学**不同源** |
| Stage B 损失 | NLL / force / semigroup / rollout_mse | ❌ **无任何 latent 预测契约**,Q 可塌缩(mag R²≈0.002) |

### 1.2 恶性循环

```
z 训不好(正则未开 + displacement 主导)
   → 压缩抓不到动力学
   → Q 塌缩
   → 用 kin_feat 注入 / 辅助头硬修(泄漏捷径)
   → Stage B 力学损失反向拉扯 Q
   → 进一步破坏 z 的预测结构
```

---

## 2. 目标架构

```
┌─ Stage A (严格 JEPA-Anything) ──────────────────────────────────────┐
│                                                                      │
│   obs ─[encoder + temporal]→ z[B,T,L,128]                           │
│                                   │                                  │
│                    ┌──────────────┴──────────────┐                  │
│                    │  对 state z 做 OPF 因子化     │                  │
│                    │  P_k:[128, r=32], K=4        │                  │
│                    │  K*r = 128 = d  (无损, 守恒) │                  │
│                    │  y_k = P_k^T z                │                  │
│                    │  z_hat = pinv(P^T)·concat(y) │                  │
│                    └──────────────┬──────────────┘                  │
│                                   │                                  │
│   损失:                                                             │
│     L_pred  : 因子预测 vs stop-grad 目标因子                          │
│     L_orth  : within Gram + cross Gram                              │
│     L_fac   : 因子坐标 activity floor                               │
│     L_enc   : encoder 输出 activity floor                           │
│     L_delta : paired_delta_loss (latent 预测, 主力)                  │
│     L_disp  : displacement head (降权, 辅助)                         │
│   正则(打开): temporal_variance + velocity + covariance             │
└──────────────────────────────────────────────────────────────────────┘
                                   │ z[B,T,L,128]
┌─ Stage B (有损物理 bottleneck, 无正交约束) ─────────────────────────┐
│                                                                      │
│   Q_T = MLP_q( z 的时序特征 )  → [B,L,8]    物理坐标 → 朗之万 SDE    │
│   S_T = MLP_s( z 的时序特征 )  → [B,L,16]   记忆态   → Mori 浴       │
│                                                                      │
│   L_Q_keep = ‖ g_θ(Q_T, Δ) − sg(Q_{T+Δ}) ‖²    ← 必加!              │
│   (S 的 Γ/C 网络允许看 z, 与 Q 同源)                                │
│   (删除 kin_feat 注入 + 辅助回归头)                                  │
└──────────────────────────────────────────────────────────────────────┘
                                   │ rollout (Q,S 按朗之万/Mori 推进)
┌─ Stage C (不变) ────────────────────────────────────────────────────┐
│   structure_decoder(Q_rollout) → atom37                             │
│   (decoder 损失不反传到 Stage A/B 的 latent, 必要时 detach)          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. 维度对照表

| 量 | 符号 | 维度 | 说明 |
|---|---|---|---|
| latent state | `z` | `[B,T,L,128]` | Stage A encoder 输出 |
| 因子数 | `K` | 4 | OPF |
| 因子宽度 | `r` | 32 | `K*r = 128 = d` |
| 因子坐标 | `y_k` | `[B,L,32]` × 4 | OPF 输出,**无损** |
| 物理坐标 | `Q_T` | `[B,L,8]` | Stage B MLP_q,有损 |
| 记忆态 | `S_T` | `[B,L,16]` | Stage B MLP_s,有损 |
| 时间步 | `T` | 8 | context 帧数 |
| 残基 | `L` | 256 | `max_residues` |

> 注:`Q_T`/`S_T` 维度沿用现有 `q_dim=8` / `bath_dim=16`(见 `configs/full_abc_stage_b_v3_final.yaml`)。这两个是**有损物理 bottleneck**,与 OPF 的 `K*r=d` **无关**,可自由选择。

---

## 4. Stage A 改动点

### 4.1 新增/改造:OPF 作用在 encoder 输出的 state 上

**文件:** `protein_jepa_frame/models/opf.py`

- 现有 `OPFPredictor.forward(endpoint_state, horizon_time_ps, ...)` 的 `endpoint_state` 语义是"用于预测的上下文状态",`_synthesize` 综合的是**预测的因子**。
- **改造:** 新增一个模式或在 `StageAJEPA` 中新增 `opf_state_head`,对**encoder 输出的 `z`**(而非 delta_target)做分解:
  ```
  y_k = P_k^T z          # 分析 (analyze)
  z_hat = pinv(P^T)·y    # 合成 (synthesize), 无损
  ```
- **复用已有的几何工具**:`orthonormalize_projectors`、`concatenated_projector`、`projector_condition_number`、`cross_factor_overlap`、`_synthesize` 均已实现,直接复用。
- **约束校验:** `latent_dim % opf_factor_count == 0`(已有,`jepa.py:261`)。

**接口草案:**
```python
class OpfStateHead(nn.Module):
    """对 encoder 输出的 state z 做 K 个正交投影的无损分解。"""
    def __init__(self, latent_dim=128, num_factors=4, ...): ...
    def analyze(self, z):        # z[B,L,d] → y[B,L,K,r]
        ...
    def synthesize(self, y):     # y[B,L,K,r] → z_hat[B,L,d] via pinv
        ...
    def orthonormalize(self):    # QR retraction (复用现有实现)
        ...
```

### 4.2 启用 anti-collapse 正则(最高优先级)

**文件:** `train_iter009_stage_a.py`(以及 `configs/mdcath_stage_a.yaml` / `full_abc_stage_a_from_scratch.yaml`)

现有 `objectives.py` 中已实现但**未启用**:

| 正则函数 | 作用 | 建议权重 |
|---|---|---|
| `coordinate_activity_regularizer(features, mask, target_std=0.02)` | 防空间塌缩(每维 std 下限) | `w=0.1` |
| `temporal_variance_regularizer(frame_states, frame_mask, target_std=1.0)` | 防时间冻结 | `w=0.1` |
| `temporal_velocity_loss(frame_states, frame_mask)` | 强制编码速度 | `w=0.1` |
| `temporal_covariance_regularizer(frame_states, frame_mask)` | 去维度冗余 | `w=0.01` |

**改动:** 在 `stage_a_total_loss` 调用处,把这些正则的权重传入并累加进总损失。当前调用:
```python
paired_loss, paired_log = stage_a_total_loss(
    mask_outputs=None,
    paired_outputs=paired_out,
    paired_weight=PAIRED_WEIGHT,
    mask_weight=0.0,
)   # ← 缺 activity / dynamic_jepa / dynamic_covariance 权重
```

### 4.3 调整损失权重

| 损失 | 现状 | 改为 | 理由 |
|---|---|---|---|
| `paired_delta_loss` | 0.5 | 1.0(主力) | latent 预测是 z 变好的核心信号 |
| `displacement_head` (`DISP_WEIGHT`) | 1.0 | 0.1(或 0) | 3D 坐标回归会把 z 拉向"易回归"而非"易预测" |
| `opf_pred_loss` | 未启用 | 1.0 | JEPA-Anything 因子预测 |
| `opf_orth_loss` | 未启用 | 0.05 | Gram 几何正交 |
| `opf_factor_activity_loss` | 未启用 | 0.1 | 因子防塌缩 |
| `opf_encoder_activity_loss` | 未启用 | 0.1 | encoder 防塌缩 |

### 4.4 OPF 四损失的定义(对齐 JEPA-Anything)

```
L_pred  = mean_k ‖ pred_k − sg(P_k^T z_target) ‖²
L_orth  = Σ_k ‖P_k^T P_k − I‖_F² + Σ_{i<j} ‖P_i^T P_j‖_F²
L_fac   = mean over (K,r) coord of relu(gamma_fac − std(coord))
L_enc   = mean over d dim of relu(gamma_enc − std(encoder_dim))
```
> 现有 `opf.py::opf_terms` 已实现这四项(见 `opf_terms` 的 `L_pred/L_orth/L_fac/L_enc`),可直接复用,只需确保 target 用 stop-grad。

### 4.5 正交策略

- 默认 `soft_gram`(JEPA-Anything 官方默认);
- 可选 `qr_retraction`(你 config 里 `opf_qr_retraction: True`),**注意:QR 必须在 optimizer step 之后**,不在 forward path 内。

---

## 5. Stage B 改动点

### 5.1 Q_T / S_T 映射(去掉正交约束)

**文件:** `protein_jepa_frame/models/stage_b/model.py`

现有(`encode_state`,L429-441):
```python
frame_state = temporal_out["frame_states"][:, -1]     # 丢时间
h_last = context_latents[:, -1] + frame_state.unsqueeze(1)
q = self.project_to_q(h_last)                          # MLP 128→8
```

**改为:**
```python
# (a) 时序特征:显式含差分,不让 MLP 从快照盲猜
z_seq    = context_latents                          # [B,T,L,128]
z_pool   = temporal_pool(z_seq)                     # attention/mean 池化 → [B,L,128]
z_delta  = z_seq[:, 1:] - z_seq[:, :-1]             # 时序差分
z_feat   = concat([z_pool, temporal_pool(z_delta)]) # 显式运动特征

# (b) 两个独立 MLP,无正交约束
q = self.mlp_q(z_feat)          # → [B,L,8]   物理坐标
s0 = self.mlp_s(z_feat)         # → [B,L,16]  记忆态 (替代 bath_initializer)
```

**要点:**
- `MLP_q` 和 `MLP_s` 独立,**不受正交约束**;
- `S` 的初始化(`mlp_s`)与动力学网络(`bath_gamma_net`、`bath_coupling_net`)**同源**:让 `Γ/C` 网络接受 `z_feat` 作为辅助条件,而不是只看 `Q`。

### 5.2 新增 keep-alive 损失 `L_Q_keep`(必加)

**文件:** `protein_jepa_frame/models/stage_b/objectives.py` + `train_iter008_stage_b.py`

```
L_Q_keep = ‖ g_θ(Q_T, Δ) − sg(Q_{T+Δ}) ‖²
```
- `Q_{T+Δ}`:同一批 **future 帧**经**同一 `MLP_q`** 得到,**stop-grad**;
- `g_θ`:一个轻量的 Δ 条件预测器(可用现有 predictor 结构);
- **不是**回归 3D 坐标(target 是 latent 的 Q);
- 直接对齐 JEPA 的 predict-then-compare-in-latent。

**目的:** 防止 Q 退化成"只够拟合力场的静态偏置"(解决 train_iter008 的 mag R²≈0.002)。

### 5.3 清理泄漏捷径

**文件:** `train_iter008_stage_b.py`

- **删除** `Q = proj_base(h_last) + proj_kin([h_last, kin_feat])` 的 kin_feat 注入;
- **删除**辅助回归头(`disp_mag/disp_dir/rg/rmsf` 对 Q 的塑造);
- 若 `kin_feat` 确实有用,**明确声明为合法 exogenous input**(带 `available_through` cutoff),走预测通路,不进被当作 state 的对象。

---

## 6. Stage C(不变)

- `structure_decoder(Q_rollout) → atom37` 保持不变;
- **唯一新增纪律:** 确保 decoder 的损失(FAPE/RMSD/torsion 等)**不反传**到 Stage A/B 的 latent(必要时对输入 `detach`);
- 论文定位:Stage C decoder 对应 JEPA-Anything 的**外置观测模型 `p_ψ(x,y|u,c)`**,属于"观测层",不是 JEPA 内核。

---

## 7. 实施顺序(分步、可回退)

| Step | 内容 | 风险 | 依赖 |
|---|---|---|---|
| **S0** | 写 **z 诊断 probe**(冻结 z,测 `I(z; future_disp/dir/Rg)` R²,对比 obs) | 零 | 无 |
| **S1** | Stage A 启用 activity/velocity/variance/covariance 正则 + 降权 displacement | 低 | 无 |
| **S2** | Stage A OPF 接到 encoder 输出(`K*r=128`),配四损失 | 中 | S1 |
| **S3** | 重训 Stage A,用 S0 的 probe 验证 z 是否改善 | 中 | S1,S2 |
| **S4** | Stage B Q/S 改为双 MLP(显式时序特征 + 同源) | 中 | S3 |
| **S5** | Stage B 加 `L_Q_keep`,删 kin_feat/辅助头 | 中 | S4 |
| **S6** | 重训 Stage B,验证 rollout 稳定性 + probe | 中 | S4,S5 |
| **S7** | Stage C 复用,确认 decoder 损失不反传 | 低 | S6 |

**S0 是决策依据:** 若 z 的 probe R² ≪ obs 的 R²,说明 z 是瓶颈本体,S1/S2 是必做;若接近,则可快速推进 S4/S5。

---

## 8. 验证与审计

### 8.1 Stage A 审计(JEPA-Anything 对齐)

| 审计项 | 判据 | 来源 |
|---|---|---|
| `analysis_full_rank` | `rank(P^T) = d = 128` | `opf.py:projector_min_singular_value` |
| `analysis_condition_number` | `κ₂ ≤ 100` | `opf.py:projector_condition_number` |
| `pseudoinverse_state_synthesis` | `NMSE ≤ 1e-6` | `opf.py:_synthesize` |
| `within_projector_orthonormality` | `‖P_k^T P_k − I‖_F² → 0` | `opf_terms` L_orth |
| `cross_projector_orthogonality` | `‖P_i^T P_j‖_F² → 0` | `opf.py:cross_factor_overlap` |
| `factor_activity` | 每个 `(K,r)` 坐标 std ≥ gamma | `opf_terms` L_fac |
| `online_encoder_activity` | 每个 d 维 std ≥ gamma | `opf_terms` L_enc |

### 8.2 Stage B 审计

| 审计项 | 判据 |
|---|---|
| `L_Q_keep` 收敛 | loss 下降且 Q 的 future-predictability probe R² 上升 |
| rollout 稳定 | 无 Q-norm 增长(对照 `CURRENT_FRAMEWORK_SUPERVISION_REPORT.md` 的 v1/v2 失败模式) |
| 无泄漏 | `kin_feat` 若保留,lineage cutoff 在 prediction time 前 |
| semigroup 非零 | 对照 v1 的"semigroup 恒为 0"失败 |

### 8.3 基线对比(对齐 JEPA-Anything 契约)

- `standard_jepa`:单头,输出 d 宽;
- `unconstrained_multihead`:`K` 个头,`heads=K, head_dim=r`;
- **参数匹配**:`K*r=d` 时两者参数量相等;还需匹配 predictor FLOPs + 训练步数。

---

## 9. 关键纪律(贯穿全程)

> **只有"预测未来 latent"的损失(`L_delta`、`opf_pred_loss`、`L_Q_keep`)可以把梯度流回 encoder → 表示;一切"回归到 3D 坐标/位移"的损失(displacement head、decoder 的 FAPE/RMSD)只能作用在各自的层,不得反传去塑造 z/Q。**

当前违反点:
- displacement head 在塑造 z(→ 降权);
- `kin_feat` 在塑造 Q(→ 删除);
- decoder 重建压力可能塑造 Q(→ 确保 detach)。

---

## 10. 文件改动清单

| 文件 | 改动 |
|---|---|
| `protein_jepa_frame/models/opf.py` | 新增 `OpfStateHead`(对 state 的 analyze/synthesize),复用几何工具 |
| `protein_jepa_frame/models/stage_a/model.py` | 接 `OpfStateHead` 到 encoder 输出;`forward_paired` 输出因子 target |
| `protein_jepa_frame/models/stage_a/objectives.py` | 已含正则,无需改;确保被调用 |
| `train_iter009_stage_a.py` | 传入 activity/velocity/variance/covariance 权重;降权 `DISP_WEIGHT`;接 OPF 四损失 |
| `protein_jepa_frame/models/stage_b/model.py` | `encode_state`:时序特征 + 双 MLP(mlp_q/mlp_s);Γ/C 同源 |
| `protein_jepa_frame/models/stage_b/objectives.py` | 新增 `L_Q_keep` |
| `train_iter008_stage_b.py` | 加 `L_Q_keep`;删 kin_feat 注入 + 辅助头 |
| `configs/mdcath_stage_a.yaml` | 打开正则权重、OPF 参数(`opf_factor_count=4`) |
| `configs/full_abc_stage_b_v3_final.yaml` | `q_dim`/`bath_dim` 沿用;加 keep-alive 权重 |
| `scripts/` | 新增 `diagnose_z_probe.py`(S0) |

---

## 11. 未决问题(评审时确认)

1. **`Q_T`/`S_T` 最终维度**:沿用 `8/16`,还是放大到 `64/64`?前者物理可解释性强(8 维广义坐标),后者信息容量大但物理语义弱。
2. **OPF `r` 的取值**:`K=4, r=32` 还是 `K=8, r=16`?影响因子的"粒度"。
3. **`L_Q_keep` 的 Δ 范围**:与 `paired_horizons_frames` 对齐(`[4,8,16,32,64,96]`)?
4. **`temporal_causal`**:目前 `False`(双向)。若切 `True`(因果),则"最后一帧已含全历史"的论据不再成立,需要显式 attention 池化(现有方案已含)。
5. **synthesize 的语义**:现有 `_synthesize` 综合的是"预测的因子"。对 state 因子化后,`synthesize` 应明确为 `pinv` 无损往返,并加 NMSE 审计。
