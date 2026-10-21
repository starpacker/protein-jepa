# Stage A 修复结果追踪表 (2026-10-18)

> 汇总本轮所有实验、诊断与结论。目标：让下游 decoder 的 **Kabsch CA RMSD < 1.2 Å**。
> 所有 within/cross ratio 来自 `scripts/probe_h_readout.py`（同蛋白时间切分 Ridge 探针，
> val 上 16 个蛋白，`--no-cross` 为 within-only）。

---

## 1. 实验对照表

| # | 运行 | 关键改动 | steps | within ratio (h=1/2/4/8) | cross ratio | `struct_recon` | step 结论 |
|---|------|---------|------:|--------------------------|-------------|----------------|-----------|
| 0 | 旧 baseline（修复前） | — | — | ~1.12 / 1.20 | — | — | 输 persistence |
| 1 | `stage_a_disp_v2` | 位移 head 读 **pred_factors/synth**，anchor=teacher(detach) | 3000–14000 | **0.946/0.942/0.939/0.935** | 1.14–1.20 | ~0.3 | 首次 <1.0，但远未过 gate |
| 2 | `stage_a_disp_ablation_nodec` | 去掉位移监督（w=0） | 2000–13000 | 0.946/0.942/0.937/0.935 | 1.18–1.23 | ~0.49 | **与 #1 完全相同** → 位移监督无效 |
| 3 | `stage_a_disp_fix` | 修改1+2：head 只读 ΔH、anchor=**online**(grad)、加方向余弦项 | 2000 | 0.949/0.945/0.937/0.938 | 1.21–1.29 | ~0.75 | **无改善** |
| 4 | `stage_a_disp_relpos` | +修改4：encoder 加 within-frame **PCA canonical-position** 通道 | 2000 | 0.960/0.965/0.943/0.943 | 1.34–1.36 | **0.019** | probe 无改善，但结构保真度大增 |

文件：
- 探针输出：`.tmpwork/probe_stage_a_disp_v2.json`, `.tmpwork/probe_stage_a_disp_ablation_nodec.json`,
  `.tmpwork/probe_stage_a_disp_fix.json`, `.tmpwork/probe_relpos_final.json`
- 训练日志：`.tmpwork/logs/stage_a_disp_v2.log`, `stage_a_ablation_nodec.log`,
  `stage_a_disp_fix.log`, `stage_a_disp_relpos.log`
- Checkpoints：`artifacts/stage_a_disp_v2/`, `artifacts/stage_a_disp_fix/`, `artifacts/stage_a_disp_relpos/`

---

## 2. 决定性诊断：gate 本身不可达

用 **真实 anchor 坐标**（不是 H！）预测 h-步后的内部 CA 位移，同样 within-protein time-split：

| h (frames) | 探针 (H→位移) held-out R² | **真 anchor 坐标→位移** held-out R² | 同任务 in-sample R² | 训练段均值预测留出 R² |
|---:|---:|---:|---:|---:|
| 1 | ~0.05 | **0.022** | 0.88 | — |
| 2 | ~0.06 | **0.029** | 0.88 | — |
| 4 | ~0.05 | **0.026** | 0.88 | — |
| 8 | ~0.05 | **0.051** | 0.88 | −0.08 |

**结论**：
- 给**真实结构**做同样的拟合，留出 R² 也只有 0.02–0.05 → **单帧→未来位移不是稳定函数**，
  被不可预测的热涨落/混沌动力学主导。
- 拟合器能 **in-sample R²=0.88**，但映射**不迁移**；训练段均值甚至反向（−0.08）。
- **H（R²≈0.05–0.09）已达到/超过真坐标上界** → **H 不是瓶颈**。
- "within held-out ≤ 0.5× persistence" 这个 gate **在物理上不可达**。
  前几轮"H 坏了/缺参考帧"的结论**部分错误**（overfit R²=1.0 是帧身份记忆没错，
  但 held-out 低不是 H 的锅）。

---

## 3. H 可解码性上界（overfit vs held-out, MLP）

| checkpoint | overfit R² | held-out R² | 绝对 anchor-relative 位置 R² |
|---|---:|---:|---:|
| `stage_a_disp_fix` (无 position 通道) | 1.000 | 0.093 | 0.039 |
| `stage_a_disp_relpos` (+position 通道) | 1.000 | 0.120 | **0.131** |

→ position 通道**确实提升**了 H 对结构/位置的编码能力（0.04→0.13），
只是**位移预测**这个 probe 测不出（因为位移本身不可预测）。

---

## 4. 各版本训练侧指标（step ~2000）

| 版本 | total | future | struct_recon | displacement |
|---|---:|---:|---:|---:|
| v2 (3000) | 1.91 | 0.44 | 0.31 | 0.97 |
| fix (2000) | 2.92 | 0.43 | 0.75 | 1.85 |
| relpos (2000) | 2.65 | 0.43 | **0.020** | 1.84 |

（displacement 绝对值不同是因为 fix/relpos 加了方向项基线≈1.0；不可直接跨版本比。）

---

## 5. 代码改动清单（本轮）

1. `protein_jepa_frame/models/stage_a_v2/model.py`
   - `forward_temporal_multilag`：anchor 改用 **online encoder**（`context_H[:, -1]`），
     `anchor_factors` / `anchor_H` **不再 detach**（修"死 anchor"）。
   - `displacement_readout_loss`：head **只读 `synth_delta_H`**（去掉 anchor_H/H_mean 旁路）；
     新增**逐残基方向余弦项** `(1 - cos)`，权重 `direction_weight`。
   - `StageAV2Config`：新增 `relative_position_features: bool = True`。
   - `displacement_head` 输入维度 `3*dH → dH`。
2. `protein_jepa_frame/models/encoder.py`
   - 新增 `_within_frame_canonical_positions`（每帧 CA 云 PCA canonical frame，
     skewness 符号约定 → **rigid-invariant, position-aware**）。
   - `ProteinGeometryEncoder` 新增 `relative_position_features` 开关 + `relpos_proj`。
3. `protein_jepa_frame/training/train_stage_a_v2.py`
   - 新增 CLI `--displacement-direction-weight`（默认 1.0）、`--relative-position-features`（默认 true）。
   - 训练/验证 loop 同步调用新签名。
4. `tests/test_stage_a_v2_temporal.py`
   - 新增/更新 6 个测试：定值预测、梯度到 encoder/predictor、anchor 非 detached、
     head 输入=dH、canonical-position rigid-invariant / position-aware / 改变 H。

---

## 6. 下一步（方向 3）：用最好的 encoder 跑下游 B1a

- 选用 encoder：**`artifacts/stage_a_disp_relpos/checkpoint_latest.pt`**（struct_recon 最好、位置 R² 最高）。
- 做法：`train_stage_b1a.py --stage-a-ckpt <relpos> --output artifacts/stage_b1a_relpos ...`
- 验收指标：B1a 日志中的 **`b1a/diag_ca_rmsd_real`**（real-H 的 Kabsch CA RMSD），
  对比 `diag_ca_rmsd_zeroH` / `diag_ca_rmsd_persist`。
- 目标：real-H 的 CA RMSD 显著低于 zeroH/persist，并朝 **< 1.2 Å** 收敛。

---

## 7. 方向 3 首轮尝试：B1a on relpos encoder —— 卡住（未收敛）

### 已完成的准备
- relpos encoder 可被 `train_stage_b1a.py` **strict 加载**（`relative_position_features=True` 自动带入），
  H 统计健康（mean≈0, std≈1, absmax≈3.8）。
- 确认旧 Stage A ckpt（`abc_v3/stage_a_mixed`）**不兼容**（缺 `delta_target`），无法用作对照。

### 结果：真实数据上 decoder 不收敛
| 数据 | 模式 | `oracle_ca_mse` 轨迹 | `diag_ca_rmsd_real` vs `zeroH` |
|---|---|---|---|
| **real mdCATH** | oracle | ~200 **持平**（step10→350 无下降） | 23.6 vs 23.6（**real 不比 zeroH 好**） |
| **real mdCATH** | predicted_adaptation | ~200 持平 | 23.3 vs 23.3 |
| **synthetic** | oracle | 75 → 27（**下降**） | 14.9 → 8.5 vs 24.1（real 好于 zeroH） |
| — | persistence (real) | — | 6.2–8.1 |

**关键事实**：同一 decoder 在 **synthetic 上能学**（ca_mse 降、real-H 赢 zeroH），
但在 **real mdCATH 上完全不动**（ca_mse≈200 平、real-H ≡ zeroH）。
→ 问题在 **real 数据路径 / 任务难度**，不是 encoder 或 decoder 代码。

### 已排除
- **不是残基太多**：真实样本 L=110（crop 生效；之前 880 是"8 帧求和"的计数错误）。
- **不是模式**：oracle 与 predicted_adaptation 同样不收敛。
- **不是超参**：改用仓库自带 `run_b1a_anchorfree.sh` 的权重同样持平。
- **不是 encoder 尺度**：relpos H mean≈0/std≈1。

### 待查（下一轮）
1. real 的 `future` 目标与 anchor 的**内部位移尺度**（早期笔记：32–64 ns 时 Kabsch 内部 RMSD 8.8–12.5 Å、
   raw 偏移 30–70 Å）→ 可能超出 decoder `coord_scale=40` 的可表示范围，
   或远超其 coil prior（L=110 → 半径 ~40 Å）。
2. real H 对 **不同帧** 是否可区分（`wrongH≈real` 提示 H 的时间可分性差）——
   但注意这可能又落回 §2 的"单帧→未来不可预测"结论。
3. horizon 选择：B1a 默认 horizons 到 96 帧（96 ns），目标位移极大；
   应考虑**短 horizon**（1–8 ns）先让 decoder 收敛。

---

## 8. 方向 3 卡点**已解决**：B1a decoder 其实会收敛，只是"慢启动 + warmup"

### 决定性实验：单批过拟合（2 个 real 蛋白，300 步，lr 5e-4）
`.tmpwork/b1a_overfit_probe.log`（GPU5，`--paired-horizons-frames 4 8`）：

| step | oracle_ca_mse | diag_real | diag_zeroH | diag_persist |
|---|---|---|---|---|
| 20 | 139.6 | 20.47 | 20.24 | 9.51 |
| 60 | 118.7 | 18.87 | 19.72 | 9.51 |
| 100 | 58.6 | 13.24 | 16.34 | 9.51 |
| 140 | 31.3 | 9.67 | 15.51 | 9.51 |
| 180 | 11.4 | 5.81 | 15.07 | 9.51 |
| 220 | 2.75 | 2.85 | 14.90 | 9.51 |
| **280** | **0.47** | **1.17** | 14.82 | 9.51 |

### 结论（推翻 §7 的"不收敛"判断）
- decoder **确实会收敛**，且最终 **real-H 1.17 Å ≪ persist 9.5 Å ≪ zeroH 14.8 Å**，
  H 完全被利用。真实数据的路径/任务**没有问题**。
- 之前 2k/300 步"持平 ~200"的原因是**观测太早**：
  1. `--warmup-steps 300` 时 LR 仍在爬升，前 ~200 步基本没学；
  2. 初始结构 = `_coil_prior`（与真实折叠差几十 Å），`tanh` 位移头从 ~0 爬到
     大位移是**慢启动**，随后 ca_mse 才雪崩下降（100→280 步：58→0.47）。
- 结论：**不需要改数据路径**；需要的是**跑够步数**（≥2k，最好 10k）让 decoder 越过慢启动。
  这正是方向 3 的直接执行路径。


---

## 9. 决定性约束：mdCATH 帧间隔 = 1 ns，persistence 地板 ≈ 3–4 Å ⇒ **<1.2 Å 不可达**

从 manifest 读出 `frame_interval_ps = 1000.0`（见 `artifacts/mdcath_manifest_200dom/*.jsonl`）：
**mdCATH 相邻帧相差 1 ns，没有亚 ns 分辨率可用。**

在 val 20 个蛋白上直接测 Kabsch CA RMSD（frame0 → frame_t，stride=1，即 t ns）：

| 时间跨越 | mean | median | min | max |
|---:|---:|---:|---:|---:|
| 1 ns | 3.08 | 2.80 | 0.85 | 8.64 |
| 2 ns | 3.85 | 3.24 | 1.06 | 14.01 |
| 4 ns | 4.36 | 3.52 | 1.18 | 18.65 |
| 8 ns | 4.26 | 3.46 | 0.98 | 15.34 |

**推论**：
- 即使只预测**紧邻的下一帧**（1 ns），蛋白相对 anchor 也已移动 **~3 Å**（中位 2.8 Å）。
- §2 已证：单帧→未来位移的 held-out R² 仅 0.02–0.05（不可预测的热/混沌动力学主导）。
- 因此 **单帧 anchor → 未来帧的最优可达 CA RMSD 的下界 ≈ persistence ≈ 3 Å（1 ns）**。
- **`< 1.2 Å` 在这个数据设置下物理上不可达**（除非 anchor 与目标时间上重合，或改用亚 ns 轨迹）。

### 要对用户目标重定 scope
- 若坚持 mdCATH + 单帧 anchor：现实目标应设为 **real-H ≤ persistence（~3 Å @1 ns）**，并尽量逼近。
- 若要 <1.2 Å：需要 (a) 亚 ns 帧间隔数据；或 (b) 允许用**多帧/多时间 anchor**（把 anchor 定为紧邻目标的短窗口），
  此时 1 ns 内的位移更小；或 (c) 目标改为**ensemble/系综**意义的结构（RMSF/接触图），而非单帧绝对坐标。
- B1a 的 `diag_ca_rmsd_real`（用真实未来帧 H 解码）可到 ~1 Å（§8 过拟合探针 1.17 Å），
  说明 **decoder 能重建给定 H 的结构**；瓶颈是**预测 H**（forecast），不是 decoder。

---

## 10. 方向 3 最终结论：改回 **anchored decoder** 后正确收敛并泛化，但天花板 = persistence

### 关键修正：anchor-free decoder 会过拟合、不泛化
- anchor-free decoder 从零构建整个折叠（只靠 H + 序列），train ca_mse 降到 ~70，
  但 **val real-H ≈ 27 Å（远差于 persist 3.66）** → 对未见蛋白完全失败。
- 原因：它必须从 H 恢复折叠，而 §2 已证 H→折叠不可解码；anchor 只被当作事后刚性放置（R0,c0），
  **丢掉了 anchor 的真实逐残基坐标**。

### 修正方案：回到 anchored decoder（`--no-anchor-free`）
- anchored 路径 `X_hat = R_anchor @ d_local + t_anchor`，从 **anchor 的真实逐残基 CA** 出发，
  只预测有界的构象变化 `d_local`（anchor-local，≤12 Å）—— 这正是 forecast 任务定义。
- 代码改动：放宽 `train_stage_b1a.py` 里 `--no-anchor-free` 的 argparse 硬门（原被"必须 anchor-free"挡住）。
- 启动脚本：`run_b1a_relpos_anchored.sh`；日志 `.tmpwork/logs/stage_b1a_relpos_anchored.log`。

### 结果（relpos encoder，horizons 1/2/4/8 frames = 1–8 ns）

| 指标 | train（step~650） | **val@250** | **val@500** | **val@750** |
|---|---:|---:|---:|---:|
| `diag_ca_rmsd_real` | 3.47 | 4.501 | 4.495 | 4.497 |
| `diag_ca_rmsd_zeroH` | 3.63 | 4.710 | 4.651 | 4.610 |
| `diag_ca_rmsd_wrongH` | 3.47 | 4.499 | 4.501 | 4.505 |
| `diag_ca_rmsd_persist` | 3.50 | 4.540 | 4.540 | 4.540 |

**读法**：
1. **泛化**：val 4.50 ≈ train 3.47（无过拟合鸿沟）→ anchored decoder 正确工作。
2. **H 有用但极弱**：real (4.495) < zeroH (4.651) —— 确实用了 H；但
   **real ≈ wrongH (4.50)** —— H 携带的**时间预测信息几乎为零**。
3. **仅比 persistence 好 ~1%**（4.495 vs 4.540）—— 与 §2 探针（held-out R²≈0.02–0.05）**完全一致**。
4. val 在 step 250→750 几乎不动（4.501→4.497）→ **已收敛到可预测性天花板**。

### 对目标 <1.2 Å 的最终判定
- **在 mdCATH（1 ns/帧）+ 单帧 anchor 下，<1.2 Å 物理不可达。**
  最优可达 ≈ persistence ≈ **3–4.5 Å**（取决于 horizon），H 只能再挤出 ~1%。
- 瓶颈**不在 Stage A encoder / 不在 decoder 代码**（两者都已验证正确），
  而在**任务本身的可预测性**（单帧→未来被热/混沌动力学主导）。

---

## 11. 忠实实验记录：Stage A + B1a 现状快照（2026-10-18 收尾）

> 本节为一次性的"现状快照"，所有数字直接取自 artifacts 与日志，不改动前文。

### 11.1 直接指标：CA RMSD（对应 <1.2 Å 目标）

B1a **anchored** decoder（`artifacts/stage_b1a_relpos_anchored/`，encoder = relpos），
验证集，指标 `b1a/diag_ca_rmsd_{real,zeroH,wrongH,persist}`：

| step | real-H | zeroH | wrongH | persistence |
|---:|---:|---:|---:|---:|
| 250 | 4.501 | 4.710 | 4.499 | 4.540 |
| 500 | 4.495 | 4.651 | 4.501 | 4.540 |
| 750 | 4.497 | 4.610 | 4.505 | 4.540 |

**读法**：real-H (≈4.495) 仅比 persistence (4.540) 好 **~1%**；real ≈ wrongH →
**H 几乎不含未来时间信息**。250→750 步不动 → **已收敛到天花板**。

### 11.2 Stage A 本身的探针指标

同蛋白时间切分 Ridge 探针（`scripts/probe_h_readout.py`，val 16 蛋白，
`within_ratio` = 探针误差 / persistence 误差，<1.0 才赢 persistence）：

| run | steps | within ratio (h=1/2/4) | cross ratio | struct_recon |
|---|---:|---|---:|---:|
| `stage_a_disp_v2` | 20000 | 0.946 / 0.942 / 0.939 | 1.14–1.20 | ~0.3 |
| `stage_a_disp_fix` | 2000 | 0.949 / 0.945 / 0.937 | 1.21–1.29 | ~0.75 |
| **`stage_a_disp_relpos`（最好）** | 2000 | **0.954 / 0.957 / 0.943** | 1.30–1.38 | **0.0195** |

- relpos 的 `struct_recon` 训练末期 = **0.0195**（结构保真度好）；
  但 within ratio 仍 ~0.94–0.96 → **位移可预测性没提升**。
- 探针结论：Stage A encoder **会重建结构，但单帧→未来位移只比 persistence 好 5%**。

### 11.3 物理约束（已确证，见 §9）

- mdCATH `frame_interval_ps = 1000.0`（**1 ns/帧**）。
- val 20 蛋白 Kabsch CA RMSD（frame0→t）：1 ns 中位 **2.80 Å**、4 ns 中位 3.52 Å。
- 用**真实 anchor 坐标**预测未来位移，留出 R² 仅 **0.02–0.05**。
- ⇒ **persistence 地板 ≈ 3–4.5 Å；`<1.2 Å` 在"mdCATH + 单帧 anchor"下物理不可达。**

### 11.4 总判定

- **Stage A encoder 本身质量 OK**（struct_recon 0.019）；问题不在代码。
- **瓶颈是任务可预测性**：单帧→未来绝对坐标的上限 ≈ persistence。
- 方向 3（B1a 单帧绝对坐标）**已触顶，死路**；故转向 **系综级目标**（`PLAN_ENSEMBLE_20261018.md`）。

---

## 12. 方向 4：ATLAS 数据集评估（2026-10-18 起）

### 12.1 目的
在 `atlas` trajectory 数据集上跑 evaluation，检验 Stage A / B1a 在**另一套 MD 数据**
（不同采样、不同时长/帧间隔）上的泛化表现。

### 12.2 前置检查（必须先确认）
**我们有拿 atlas dataset 做 training 吗？eval 的 protein 是否是未见过的？**
→ 见 §12.3（本节填入核查结果）。

### 12.3 ATLAS 现状核查 ✅（关键结论）

**核查问题：我们有拿 atlas 做 training 吗？eval 的 protein 是否未见过？**

1. **我们的 Stage A（relpos）只在 mdCATH 上训练**：
   `artifacts/stage_a_disp_relpos/checkpoint_latest.pt` 的 `train_args.manifest`
   = `artifacts/mdcath_manifest_1000dom/train.jsonl`（800 train 蛋白，val 100，overlap=0）。
   → **完全没有用 ATLAS 训练**，ATLAS 是彻底未见（unseen）的数据。
2. **ATLAS manifest 划分（蛋白级 disjoint）**：
   - `artifacts/atlas_manifest/train.jsonl`：270 条轨迹 / **90 个蛋白**
   - `artifacts/atlas_manifest/val.jsonl`：30 条轨迹 / **10 个蛋白**
   - `test.jsonl`：**空文件（0 行）**
   - **train ∩ val 蛋白 = 空集**（已核对），disjoint ✅
3. **ATLAS 数据格式兼容**：npz（`coords[101,415,4,3]`，`frame_interval_ps=1000`，
   300 K，`npz_path` 字段）→ `load_manifest` 能识别，我们的 paired dataset
   能正常构建（val 8880 paired windows），batch/collate 正常，coords 有限。
   **注意：ATLAS 帧间隔同样是 1 ns，与 mdCATH 一致。**
4. ⚠️ 历史遗留：`artifacts/abc_v3/stage_b1b_atlas/`（Oct 17）是**旧 lineage**（abc_v3），
   不是当前 relpos/anchored 线路，**不能**作为对照。

**⇒ 结论**：可以在 ATLAS val（10 个未见蛋白）上直接跑 evaluation，
用当前 relpos encoder + anchored decoder。这是干净的跨数据集泛化测试。

### 12.4 ATLAS val 评估（跨数据集泛化，忠实记录 2026-10-18）

**协议**：冻结 `stage_a_disp_relpos/checkpoint_latest.pt`（mdCATH 训练）+ 冻结
`stage_b1a_relpos_anchored/checkpoint_best.pt`（mdCATH 训练），在 **ATLAS val**
（10 个未见蛋白 × 3 replicas = 30 条轨迹）上跑同一诊断，脚本为
`scripts/eval_b1a_manifest.py`（NEW 文件，只读 ckpt，不动精确坐标解码代码）。

**采样修复（重要）**：初版 DataLoader `shuffle=False` 顺序迭代，前 20 batch
**全部落在第 1 条轨迹**（`1ah7_A_R1`），metric 由单一蛋白主导 → 数字不可信。
已改为**全 dataset 随机抽样**（`torch.randperm`，seed=0，覆盖所有蛋白与所有
horizon）。修复后 60 batch / batch_size 4 = 240 窗口，覆盖 **30 条轨迹（10 蛋白）**、
全部 4 个 horizon。

**结果**（`artifacts/atlas_b1a_eval.json`）：

| 指标 | 全局 | h=1 | h=2 | h=4 | h=8 |
| --- | --- | --- | --- | --- | --- |
| persist | **0.239** | 0.215 | 0.234 | 0.252 | 0.260 |
| real-H (encoder+decoder) | 0.465 | 0.446 | 0.478 | 0.474 | 0.466 |
| wrongH (wrong context) | 0.465 | 0.441 | 0.479 | 0.469 | 0.473 |
| zeroH (H=0) | 2.006 | 2.053 | 2.039 | 1.938 | 1.975 |

**读数（关键 negative result）**：
1. **persist ≈ 0.24 Å**，比 mdCATH val 的 3–4 Å **小一个数量级**。
   已用独立脚本直接对 ATLAS npz 算 frame0→frame t 的 Kabsch CA RMSD 复核：
   `1ah7_A_R1` 0.21–0.28、`1btk_A_R2` 0.21–0.36、`1iom_A_R3` 0.55–1.66 → **属实**。
   ⇒ ATLAS 的轨迹极其"刚性"（短/平衡态 MD 簇），persistence 地板本身就很低。
2. **real-H (0.465) 反而差于 persistence (0.239) 约 0.23 Å**，且
   **real ≈ wrongH**（0.465 vs 0.465）→ 冻结的 Stage A encoder+decoder 在
   "未来几乎不动"的 ATLAS 上，**相对 anchor 引入的是噪声而非信息**：
   anchor 已给出 0.24 Å 的近乎完美预测，decoder 的 H 分支把误差抬到 0.47 Å。
3. **zeroH = 2.0 Å 远差于 persist 0.24** → H 通道**确实携带结构信息**
   （去掉它退化到 2 Å），但在 ATLAS 这种低位移情形下，该信息的增益
   被 anchor 的强先验淹没，且方向不利。
4. **horizon 单调性正确**：persist 随 horizon 从 0.215→0.260 单调上升，
   说明 horizon/帧索引管线**无 bug**（早先怀疑的 "只用了 1 ns" 是采样覆盖 bug，
   不是 horizon bug）。

**⇒ 结论**：ATLAS 是干净未见集；在此数据上，当前 Stage A+B1a 冻结栈
**未能超过 persistence**（差 0.23 Å）。这说明：
- 我们的模型在 mdCATH 上"≈persistence（好 ~1%）"的能力**不能跨数据集保持**；
- 对低位移刚性轨迹，**首要 baseline 就是 persistence 本身**，
  任何 decoder 必须先"学会不动"，否则净负增益；
- 这也是系综级 loss（T1–T6）要解决的动机之一：单帧确定性解码在此无效。

### 12.5 ATLAS 上的系综级评估（T6，2026-10-18 补充）

训练好的 ensemble head（mdCATH 训练，见 `ENSEMBLE_RESULTS_20261018.md`）在
ATLAS val 上跑 `scripts/eval_ensemble_manifest.py`（40 batch，27 轨迹）：

| metric | model | persistence | model−persist | 95% CI | 结论 |
| --- | --- | --- | --- | --- | --- |
| total | 0.408 | **0.181** | −0.227 | [−0.242, −0.212] | model 显著更差 |
| rmsf | 0.174 | **0.068** | −0.106 | [−0.123, −0.090] | model 显著更差 |
| dist | 0.761 | **0.374** | −0.388 | [−0.407, −0.369] | model 显著更差 |

**⇒ 在 ATLAS（近刚性）上，系综头全面输给 persistence。** 与 §12.4 的单帧结论一致：
mdCATH 学到的柔性先验不能迁移到"几乎不动"的 ATLAS 轨迹。
详见 `ENSEMBLE_RESULTS_20261018.md` §E.2。

### 12.6 待跑
- 在 ATLAS train（与 val 蛋白 disjoint）上微调 ensemble head 的泛化实验（下一轮）。
