# v3 Collapse 修复 — 代码修改 TODO / 方案

**日期:** 2026-11-07 · **目标:** 在 v3 SOTA 基础上修改代码并重训，得到「指标突出 + 几何合法 + 不坍缩」的构象。
**状态:** 调研完成，代码修改前。本文是「改什么 / 为什么」的完整论证，落地前先对齐。

---

## 0. 结论摘要（一句话）

v3 的坍缩是**三个独立、可叠加**的缺陷，根因都在「损失没有在**解码后的 Cartesian 坐标**上监督**局部几何**」：

1. **解码器方差坍缩**（`D(E(X_GT))` 只保留真实运动方差的 1–2%）——`_roundtrip_loss` 已经用 `weighted_ca_rmsd`（**全局** RMSD）监督 `Enc→Dec→X`，但**没有**对解码坐标的**局部键长/伪角/碰撞**施加任何监督，所以 decoder 学会输出「平均结构」即可。
2. **GLE 单步几何坍缩**（一步后 74/95 键 `<3Å`）——学习到的力 `f=-∇U(H)` 无界（`force_clip=None`，`tail_init=1.0` 强二次尾），`h·D·f≈17` 被 `max_step_norm=20` 硬截断，再加噪声 11.3，单步 H 位移 ≈ 2.5×‖H‖，把 H 推出 decoder 流形。
3. **多步 rollout 无几何监督**（train/inference 失配）——`_run_batch` 只解码**终点** H（`H_tau = traj[:,-1]`），中间帧的几何从不进入损失。

三个缺陷的修复**不是后处理**，而是把可微的局部几何损失接到 decoder/rollout 输出的 Cartesian 坐标上，并让力/步长有界。

---

## 1. 已证实的实测证据（grounded in 真实 checkpoint + 源码）

所有数字来自对 `artifacts/lasp_v3_velocity/checkpoint_best.pt` 的直接测量。

### 1.1 GLE 单步位移量级（`/tmp/diag_step_magnitude.py`，protein 1ka1A01, 320K）

| 量 | 值 |
|---|---:|
| ‖H₀‖（latent 范数） | 11.31 |
| ‖f‖ = ‖−∇U(H)‖ | **272.6**（`force_clip=None`） |
| D（mobility） | 4.975e-4 |
| ‖h·D·f‖（drift，h=1000ps） | 17.03（**被 `max_step_norm=20` 截断**） |
| ‖sqrt(2hD)·ε‖（噪声） | 11.01 |
| **单步总位移 / ‖H‖** | **2.48×** |

### 1.2 解码器方差坍缩（GT-1 多帧，`/tmp/export_gt_decode_pdb.py`）

| protein | var(true) | var(encdec) | ratio |
|---|---:|---:|---:|
| 1nhwC00 | 429.2 | 3.3 | 0.01 |
| 1ka1A01 | 41.9 | 0.7 | 0.02 |
| 1kafD00 | 48.0 | 0.4 | 0.01 |

即 `D(E(X_GT))` 只保留 1–2% 的真实运动方差（anchor 帧重构 RMSD 0.2–0.6Å 没问题，但**帧间运动被平均掉了**）。

### 1.3 dt 扫描确认「位移量级是驱动项」（E4）

| step_ps | 一步后 `<3Å` 比例 |
|---|---:|
| 1000 | 0.768 |
| 200 | 0.442 |
| 50 | 0.095 |
| 10 | **0.032** |

位移越小几何越合法 → **坍缩由单步位移量级驱动**，不是 encoder 信息丢失。

### 1.4 v3 实际配置（从 checkpoint 读出）

- `train_encoder=True, train_decoder=True`（lr-scale 0.1，都 unfrozen）
- `roundtrip_weight=0.5, roundtrip_frames=2`（decoder roundtrip 已开）
- `force_clip=None`，`max_step_norm=20.0`，`tail_init=1.0`（强二次尾！legacy 是 1e-3）
- `bath_coupling=velocity`，`coupling_scale=1e-3`（S 惰性，见 memory `lasp-bath-inert-two-scale-bug`）
- 训练入口是 `train_stage_b_hs.py`（不是 `train_lasp_ca_dynamics.py`）

---

## 2. 源码 → 根因 → patch 的完整映射

### 缺陷 1：解码器方差坍缩（decoder / posterior-mean collapse）

**代码路径：**
- `models/stage_c/structure_decoder.py:727-792` anchor-free 路径：
  `prior = _coil_prior(L)`（3.8Å 自回避 coil）→ `delta = coord_scale·tanh(internal_position_head(s))`（`coord_scale=40`）→ `x_int = prior + delta`。
  每个残基的 `delta` 是**独立**的（`[B,L,3]` 逐残基 head 输出），相邻残基互相不知道，所以 `+ delta` 会**破坏 coil 的 3.8Å 键**。
- `training/train_stage_b_hs.py:455-504` `_roundtrip_loss`：唯一几何信号是 `weighted_ca_rmsd(pred_ca, true_ca)`（**全局 Kabsch RMSD**），没有任何局部键长/角度/碰撞项。

**为什么坍缩：** 全局 RMSD 的最小值 = 条件均值（平均结构）。decoder 学会把不同的 H 映射到近似相同的 CA（平均），因为这样全局 RMSD 最小，且没有任何局部几何项惩罚「把二级结构压扁」或「键长坍缩」。`coord_scale·tanh` 的有界性 + 独立逐残基 delta 让 decoder 天然无法表达需要「链上相邻残基协调」的大幅运动。

**patch（缺陷 1）：**
1. **新增** `geometry/ca_geometry_loss.py`（已写好，见 §5）：对**解码后的 CA** 计算可微的 bond（3.8Å，平方距离型，`d→0` 梯度安全）/ pseudo-angle（宽参考 ~110°，弱权重）/ nonlocal clash（`softplus(3.5−d)`）损失，并用 `residue_index/chain_id` 屏蔽断链（**绝不**用数组相邻）。
2. **把几何损失接入 `_roundtrip_loss`**：在 `weighted_ca_rmsd` 之外，对 `pred_ca = decode(encode(frame))` 施加 `ca_geometry_loss`。梯度经 decoder（unfrozen）和 encoder（unfrozen）反传。
3. **（可选，你已选「thorough」）把有界 `coord_scale·tanh` 换成无界/softplus 位置头**，让 H 的位移信号真正能移动坐标，去掉方差上限。做成 config flag（`anchor_free_position_head = 'tanh' | 'softplus'`），默认关闭，A/B 对照。

### 缺陷 2：GLE 单步几何坍缩（force/step 无界）

**代码路径：**
- `models/stage_b1b/model.py:449-466` `compute_forces`：`f = -grad_H U(H)`，`force_clip` 仅在 `self.cfg.force_clip is not None` 时生效（v3 是 None）。
- `model.py:711-720` `sample_step` velocity 分支：`dH = h·D·p`，仅在 `max_step_norm` 处截断**步长**（不是力），再加噪声 `sqrt(2hD)·ε`。
- `model.py:225-239` `per_residue_energy`：`tail = 0.5·softplus(log_tail)·‖H‖²`，`tail_init=1.0` → `log_tail≈0.54` → `softplus≈1.0`，**强二次尾**贡献力 ≈ `1.0·H`。

**为什么坍缩：** 力 `f` 无界（272），drift `h·D·f≈17` 被截断到 20（已经是 1.8×‖H‖），噪声再 +11（总 2.5×‖H‖）。decoder 的 `coil_prior+tanh` 无法吸收 2.5× 范数的 H 位移，输出坍缩坐标。这是**动力学 ↔ decoder 流形失配**，不是 encoder 信息丢失。

**patch（缺陷 2）：**
1. **`tail_init` 回落到物理尺度**（1.0 → 1e-2 或 1e-3）：减少强二次尾的无界梯度源。这是最直接的「力太大」根源。
2. **`--force-clip` 打开**（已在 `train_stage_b_hs.py:153` 暴露，只是没传值）：设一个物理标度的 `force_clip`（≈10–20，即 O(1)×‖H‖），让 `h·D·f` 不再被截断到「1.8×‖H‖」而是稳定在 O(1)×‖H‖ 内。**注意**：单独 `force_clip` 不够（噪声 11 仍在），必须配合缺陷 1 的 decoder 容量 + 缺陷 3 的多步监督一起。
3. **`max_step_norm` 从 20 降到 ~10**：让 `drift+noise` 的总位移 ≈ O(1)×‖H‖，而不是 2.5×。（这是 dt-scan 的直接推论。）

> **关键诚实点（我在调研中修正了早先结论）：** 之前我说「unbounded force 是根因」，但 `force_clip` 单独作用在力上只能把 drift 从 17 降到 ~10，**不足以**让总位移（drift+noise≈28）降到 1×‖H‖。真正的修复是**三件套**：decoder 容量（缺陷1）+ 位移有界（缺陷2）+ 多步几何监督（缺陷3）。不能只 `force_clip`。

### 缺陷 3：多步 rollout 无几何监督（train/inference 失配）

**代码路径：**
- `training/train_stage_b_hs.py:384-414` `_rollout_decode`：`H_tau = traj[:, -1]`（**只解码终点**），loss 只对终点云计分。
- `training/train_stage_b_hs.py:507-569` `_run_batch`：`compute_ensemble_loss(ca, ...)` 只对终点 `ca` 计分；`_decode_path` 存在但只用于 `path_ensemble`/`dense_trajectory` 分支，v3 默认不用。

**为什么坍缩：** 训练只监督终点，GLE 中间步可以任意漂移（只要终点云统计对上），中间帧几何无人看管 → inference 时 64 步 rollout 的中间帧坍缩。

**patch（缺陷 3）：**
1. 在 `_run_batch` 加一个 **multi-step rollout 几何项**：用已有的 `_decode_path`（`train_stage_b_hs.py:588`）解码 `H_traj` 在 `t=1,2,4,8,…` 的 CA，对每个监督帧施加 `ca_geometry_loss`。
2. 用 `rollout_grad_steps`（短 BPTT，如 8–16）控制梯度覆盖，避免 64 步全图 OOM；在报告中**如实声明梯度覆盖范围**（spec §7.5.3.2 不许假装 full BPTT）。
3. 新增 CLI `--rollout-geom-weight`、`--rollout-geom-frames`（默认 0 = 关闭，A/B）。

---

## 3. 关于 OpenFold 的定位（spec P0-B / §4）

**调研结论（grounded）：**
- **encoder 不用 OpenFold**：`models/encoder.py` `ProteinGeometryEncoder` 是自定义 graph-attention，输入是手算的 6 个 intra-residue 距离 + sin/cos(φ,ψ,ω) + 成对 CA RBF + orientation bias。**没有** residue rigid frames / torsion-to-frame / FAPE。
- **decoder 已经用 OpenFold**（vendored `protein_jepa_frame/vendored/openfold`，Apache-2.0，官方 frame algebra）：`structure_decoder.py:852` `reconstruct_atom37_from_frames` → `openfold_atom37.py` `decoder_frames_to_atom37` → 官方 `torsion_angles_to_frames` + `frames_and_literature_positions_to_atom14_pos` + `atom14_to_atom37`。**主链 N/CA/C/O + 侧链 atom37 已经是 OpenFold 生成**。
- **但 CA 的放置不是 OpenFold**：`x_int = coil_prior + coord_scale·tanh(head)` 是自定义的有界校正，这才是几何坍缩的源头。OpenFold 的 residue frame 是「从 N/CA/C 三点构 frame」（`Rigid.make_transform_from_reference`），本身**不能**保证 CA-CA 键长——键长由 CA 放置决定。

**因此 spec 的「encoder 借 OpenFold」「decoder 借 OpenFold」应这样落地（我的判断）：**

- **encoder：不强改**。先修损失（缺陷 1/3）让现有 encoder 在几何监督下被正确训练；如果重训后 GT-1 方差仍坍缩，再评估引入 OpenFold 的 residue-frame / torsion 特征（spec B3 的 R2 方案）。理由：encoder 的 6 距离 + torsion 特征**信息上足以**区分构象（anchor 重构 0.2–0.6Å 证明信息在），坍缩在 decoder 的「平均化」而不是 encoder 的「丢信息」。先最小改动，避免 R2 的引入成本 + 训练风险。
- **decoder：OpenFold 已在 atom37 层正确使用**。要修的是 CA 放置层（有界 tanh → 可表达大幅运动的头 + 几何损失）。**不要**用「OpenFold 静态折叠」冒充动力学（spec §4.2 明确禁止）。
- **主链 N/CA/C/O**：已经由 decoder 的 OpenFold atom37 路径输出（`out["backbone_coords"]`, `out["atom37_coords"]`）。几何验收要在 CA 修复之后做（spec「先修 CA，再主链」）。

---

## 4. 训练对照矩阵（你选 B0 + B2）

| 实验 | 内容 | force_clip | 几何 loss | 多步几何 | 目的 |
|---|---|---|---|---|---|
| **B0** | 原始 v3（frozen checkpoint） | 否 | 否（原 roundtrip 只有全局 RMSD） | 否 | baseline |
| **B2** | v3 + 全部三缺陷修复 | 是（+tail_init 回落 +max_step_norm 降） | 是（bond/angle/clash） | 是（t=1,2,4,8） | 主修复 |

- 按你「B0 + B2 only」的决策：跳过独立的 bugfix-only arm（B1）和 multi-step-only arm（B3），把 force_clip 折进 B2。**代价**：无法单独归因「force_clip 的收益」vs「几何 loss 的收益」。若 B2 有效但需要归因，再加 B1/B3。
- B2 从 **fresh v3 初始化**重训（不是旧 checkpoint 微调），`--train-encoder --train-decoder` 保持与 v3 一致，`--rollout-steps 8`、`--n-samples 6`、`--multi-horizon --paired-horizons-frames 1 4 16 64`、`--per-sample-temperature`、`--roundtrip-weight 0.5`、seed 42。

---

## 5. 已写好的代码（新模块）

`protein_jepa_frame/geometry/ca_geometry_loss.py`（新建，已 smoke-test 通过）：

- `ca_bond_loss(ca, mask, residue_index, chain_id, d_ref=3.8)`：平方距离型 Huber，`d→0` 梯度安全（用 `‖v‖²` 而非 `‖v‖`，避免重合点梯度为零——t=32 的 resid 28-29 间距 0.06Å 正是这种情形）。
- `ca_pseudo_angle_loss(...)`：`1−cos(θ−θ_ref)` 周期光滑，弱权重，不把柔性 loop 钉死在单一角度。
- `ca_clash_loss(...)`：`softplus(3.5−d, beta=10)` 非局部排斥，`|i−j|≥4` 或不同链才计入。
- `ca_geometry_loss(...)`：聚合 + 返回逐项 dict 供 val 选权重。

**验证记录**（已跑）：clean 3.8Å 链 → bond loss ≈ 0；坍缩链 → bond 80.9 / clash 3.5；断链 gap → 跨 gap 的键**不计**；近坍缩点梯度范数 6.1（梯度流动正常）。

---

## 6. 待落地的代码修改（顺序）

1. **[已做]** 新建 `geometry/ca_geometry_loss.py`。
2. **`training/train_stage_b_hs.py`**：
   - `_roundtrip_loss`（455 行）加 `ca_geometry_loss(pred_ca, ...)` 项，权重走新 CLI `--roundtrip-geom-weight`（默认 0，A/B）。
   - `_run_batch` 加 multi-step rollout 几何项：`_decode_path` 解码 `t=1,2,4,8` → `ca_geometry_loss`，权重 `--rollout-geom-weight`（默认 0）。
   - 新 CLI：`--roundtrip-geom-weight`、`--rollout-geom-weight`、`--rollout-geom-frames`、`--anchor-free-position-head`。
3. **`models/stage_c/structure_decoder.py`**：加 config flag 支持无界位置头（softplus）替代 `coord_scale·tanh`，默认关闭（A/B）。
4. **`run_scripts/run_lasp_v3_velocity.sh`**（或新建 B2 脚本）：加 `--force-clip <val> --tail-init 1e-2 --max-step-norm 10 --roundtrip-geom-weight <w> --rollout-geom-weight <w>`。
5. **回归测试**（`tests/test_ca_geometry_loss.py`）：gradcheck / 断链屏蔽 / 重合点梯度安全 / clean chain≈0。
6. **QC 脚本**：rollout 后逐帧输出 `bad CA fraction`、`short bond %`、`clash count`、`first-invalid-step`（spec §7.6 T3 的新指标）。

---

## 7. 重训后验收 Gate（对应 spec §8）

- **Gate A0**：重训后 `D(E(X_GT))` 的方差 ratio 从 0.01–0.02 显著提升（目标 >0.5，至少 >0.2），anchor 重构不退化。
- **Gate A**：一步 rollout `<3Å` 键比例从 67% → ~0；`dt=0`/零更新严格保形；`first-invalid-step` 延后或消失。
- **Gate D**：fresh 重训，新 PDB SHA-256 与旧的不同；RMSD/Rg 不退化（不能靠失去运动换几何合法）；多蛋白/多 seed 统计。
- **Gate B**：主链 N/CA/C/O 几何验收（在 CA 修复后），CA 保留 RMSD ≤0.1Å。

---

## 8. 诚实声明 / 风险

- **`force_clip` 单独不够**（已论证）：必须三件套。若只 force_clip，drift 降到 ~10 但噪声 11 仍在，总位移仍 ~2×‖H‖。
- **`tail_init=1.0` 是 v3 的显式选择**（legacy 1e-3），回落到 1e-2 会改变 H-space 能量的稳态分布，需在重训中重新校准；这是「修 bug」还是「改超参」要如实记录（我倾向：tail_init 是根因之一，但主修复是几何损失 + 位移有界，tail_init 回落作为辅助）。
- **几何损失有「钉死结构」风险**（spec §7.5.3.5）：bond 参考用宽分布（Huber delta=1.0Å），angle 用弱权重，clash 只碰真正碰撞，避免把所有帧逼成静态 anchor。会用「跨帧非平凡运动」指标对冲验证（不能靠失去运动换几何合法）。
- **多步 rollout 几何项的内存**：用短 BPTT（`rollout_grad_steps=8`），如实声明梯度覆盖，不假装 full BPTT。

---

## 9. 下一步（等你确认）

请确认上面 §6 的修改清单和 §4 的 B0+B2 矩阵，我就开始落地代码（先补 §6 的 2/3/4/5，再跑 B2 重训）。
