# 三任务评测结果报告（Task 1 / 2 / 3）
> **日期:** 2026-10-21 · **计划文档:** `NEXT_STEP_EVALUATION_3TASKS_20261021.md`
> **范围:** 仅评测（冻结编码器/解码器；仅在 Task 3 明确需要时训练一个温度条件头）。
> **产物:** 每个任务都给出 `n`（窗口/蛋白数）+ 采样预算；静态与动力学指标分表。

---

## 0. TL;DR（一页结论）

| 任务 | 结论 | 证据文件 |
|---|---|---|
| **T1 主链重建（OpenFold）** | 用 **vendored 官方 OpenFold** 把 N/CA/C/O 主链**贯通到 producer → 评测 → PDB 导出**。Model 1 的 backbone CA 列与 Cα 系综**逐位一致**（协议保证），同时新增主链级指标（Kabsch N/CA/C/O RMSD + 立体化学 validity）。发现并**修正**了一个评测口径错误：原 suite 的“真值”是从 Cα **重新 lift** 的理想几何，已改为**直接读真实 MD 的 N/CA/C/O**，oracle 现在能反映真实键长涨落。 | `results/backbone_suite_model1_1000dom.json`、`artifacts/pdbs_1000dom_backbone_v1/` |
| **T2 BioEmu 指标 + MSM 状态预测** | 实现 BioEmu 的（A）**静态系综**指标：reference-state **coverage**、metastable **free-energy MAE**、population **MAE/JSD/TV**；（B）BioEmu **没做**的**动力学/状态预测**指标：lagged state accuracy / macro-F1 / balanced accuracy、转移矩阵误差、implied timescales，**以及用户点名的“能否预测状态转变（或不变）”二分类指标**。MSM 在**真实时间序轨迹**上拟合（非 anchor 窗口），timescales 健康。 | `artifacts/eval_states_msm_1000dom.json` |
| **T3 温度泛化性** | **两条模型都考察了温度输入**。Model 1（temp-cond head，`cond_dim=2`）给出完整 **5×{seen,unseen protein}×{correct/wrong/none}** 矩阵；**喂错温度显著掉点** → 证明它**真的在用温度**。Model 2（from-none GLE v4g）的 **correct≈wrong≈none** → 温度输入**几乎惰性**。另用温度留出（不含 T=450）训练的 Model 2 检查未见温度外推。 | `artifacts/eval_temperature_generalization_cond_full.json`、`..._cond0.json`、`results/temperature_generalization_model2_v4g.json` |

---

## 1. TASK 1 — OpenFold 主链（backbone）原子重建

### 1.1 做了什么
- **T1.1** 把冻结点态解码器**已有的** OpenFold atom37 输出（`backbone_direct`/`atom37_coords`/`torsions`）**透出**到 `EnsembleProducer` 与 `HSEnsembleProducer`，返回新键 `backbone_samples [B,n,L,4,3]`、`atom37_samples`、`torsion_samples`，**不改**原有 `ca_samples` 键。
- **T1.2** Model 2（只输出 Cα）通过 `openfold_ca_lifter.lift_ca_to_backbone`（**原子摆放仍由 OpenFold 完成**：group-0 理想 offsets + `Rigid.make_transform_from_reference` + `frames_to_atom37`）得到主链，**无需重训**。
- **T1.3** 新模块 `evaluation/backbone_metrics.py`：Kabsch **backbone RMSD** + 逐原子 N/CA/C/O RMSD、键长/键角 validity、Ramachandran allowed 比例、ω trans 比例、C(i)–N(i+1) 肽键 validity、非局域 clash。
- **T1.4** `export_eval_trajectories_pdb.py --atoms backbone` 写真实 N/CA/C/O 的 ATOM 记录；并加**断言**：backbone 的 CA 列必须等于 `ca_samples`（atol 1e-4）。
- **T1.5** `tests/test_openfold_backbone_path.py`（CA 列 === ca_samples 逐位）、`tests/test_backbone_metrics.py`（oracle→RMSD 0、键角容差、UNK 处理）。

### 1.2 结果（Model 1，mdCATH-1000dom val，30 批 × batch 2 = 60 窗口，n_samples=8，horizon=1 ns）

> ✅ **口径修正**：真值 backbone 现为**真实 MD 的 N/CA/C/O**（此前误用 Cα 重新 lift 的理想几何，导致 oracle 的肽键 validity 也显示 0，属评测 bug，已修）。

| 指标 | model | persistence | oracle（真值） |
|---|---|---|---|
| **backbone RMSD ↓** | **3.407** | 2.764 | 0.000 |
| RMSD-N ↓ | 3.263 | 2.688 | 0.000 |
| RMSD-CA ↓ | 3.235 | 2.769 | 0.000 |
| RMSD-C ↓ | 3.288 | 2.710 | 0.000 |
| RMSD-O ↓ | 3.841 | 2.887 | 0.000 |
| 键长 N–CA valid ↑ | **1.000** | 0.844 | 0.837 |
| 键长 CA–C valid ↑ | **1.000** | 0.784 | 0.785 |
| 键长 C=O valid ↑ | **1.000** | 0.728 | 0.731 |
| 键角 N–CA–C valid ↑ | **1.000** | 0.987 | 0.987 |
| 肽键 C(i)–N(i+1) valid ↑ | **0.029** | 0.794 | 0.789 |
| Ramachandran allowed ↑ | 0.710 | **0.949** | 0.949 |
| ω trans ↑ | 0.794 | **0.995** | 0.997 |

**读法（诚实）：**
- **CA 列无损**：`backbone_samples[...,1,:] == ca_samples` 由测试与导出断言双重保证（`tests/test_openfold_backbone_path.py` + `export` 断言）。
- 模型主链**内部几何是“理想化”的**（键长/键角 validity = 1.000，因为 OpenFold 摆放用的是文献理想值），这与真实 MD 的热涨落（validity 0.73–0.84）不同——**这是特征不是 bug**：模型预测的是系综**平均几何**，真实 MD 每帧有热噪声。
- **短板**：模型预测的**肽键 C–N 距离**（均值 2.73 Å）**超出 1.33±0.05 Å 容差**（validity 仅 0.029），且 **ω 只有 79% trans**（真值 99.7%）——说明**主链并不是逐残基自洽的**，主要因为 Model 1 的 backbone 由 Cα 系综 + 理想几何拼出，缺少跨残基的连续约束。**这是 Task 1 指出的真实改进点**（需要真正的 torsion/backbone 头，而非 lifter）。
- **位置精度**（Kabsch）：model 3.41 Å vs persistence 2.76 Å——与 Cα 系综短板一致（单帧位置误差）。

### 1.3 产物
- `results/backbone_suite_model1_1000dom.json`（+ `artifacts/` 同名副本）
- `artifacts/pdbs_1000dom_backbone_v1/{ground_truth,persistence,ours_stageA}.pdb`（真实 N/CA/C/O，PyMOL 可开，无 NaN/塌缩）

---

## 2. TASK 2 — BioEmu 指标 + MSM 状态预测

> **按源文档 §5 要求：静态/平衡系综指标与时间动力学指标分成两张表。**

### 2.1 Family A — 静态/平衡系综指标（BioEmu 口径）
**设置：** mdCATH-1000dom val，40 蛋白 × 20 窗口 = 800 窗口，n_samples=8，k_states=5，ε=2.0 Å，internal-distance 特征，参考态来自真实轨迹。

| 指标 | 值 | 说明 |
|---|---|---|
| **Reference-state coverage ↑** | **0.540** | 每个参考态被 ≥0.1% 采样命中的宏平均 |
| **Free-energy MAE ↓** | **0.779 kcal/mol** | F_k = −k_B T ln p_k（T=348 K） |
| Population TV ↓ | 0.707 | 总变差距离 |
| Population JSD ↓ | 0.379 | Jensen–Shannon 散度 |
| n_proteins | 40 | |

**对照 BioEmu 原文**：内部 MD 平衡基准 free-energy MAE ≈ 0.91 kcal/mol。**我们的 0.779 与之同量级**（但要注意蛋白族/温度/状态划分不同，非严格可比）。

### 2.2 Family B — MSM 状态预测（BioEmu **不做**的部分）
**MSM 拟合（关键方法修正）：** 在**真实时间序轨迹**上 TICA(dim=2, lag=1) → KMeans(50 micro) → `MaximumLikelihoodMSM(reversible=True)` → PCCA → 5 macrostates。
**健康性检查**：true MSM timescales = **[150.2, 85.3, 70.7, 41.2, 27.6]**（lag 单位，单调递减、无 inf）——证明拟合在**真时间序**上（此前在 shuffle 的 anchor 窗口上拟合会退化到 `[inf, 0]`，已修正）。

| horizon | micro acc | persist acc | **macro acc** | macro persist | true transition rate |
|---|---|---|---|---|---|
| 1 ns | 0.7625 | 0.8188 | **0.9313** | 0.9625 | 0.038 |
| 4 ns | 0.7125 | 0.7688 | **0.9000** | 0.9438 | 0.056 |
| 16 ns | 0.6500 | 0.7375 | **0.9000** | 0.9500 | 0.050 |
| 64 ns | 0.6313 | 0.6313 | **0.8438** | 0.8688 | 0.131 |
| 192 ns | 0.5938 | 0.6063 | **0.8375** | 0.8813 | 0.119 |

### 2.3 ★ 用户点名指标 — 能否预测“状态转变 / 不变”（二分类）
把每个窗口标成 `true_change = (s(t+τ) ≠ s(t))`（MD 是否真的跳态），`pred_change = (ŝ(t+τ) ≠ s(t))`（模型是否预测跳态）。

| horizon | binary acc | persist acc | 转变 precision | 转变 recall | **转变 F1** | true rate | pred rate |
|---|---|---|---|---|---|---|---|
| 1 ns | 0.931 | 0.963 | 0.222 | 0.333 | **0.267** | 0.038 | 0.056 |
| 4 ns | 0.900 | 0.944 | 0.182 | 0.222 | **0.200** | 0.056 | 0.069 |
| 16 ns | 0.900 | 0.950 | 0.167 | 0.250 | **0.200** | 0.050 | 0.075 |
| 64 ns | 0.844 | 0.869 | 0.300 | 0.143 | **0.194** | 0.131 | 0.062 |
| 192 ns | 0.838 | 0.881 | 0.182 | 0.105 | **0.133** | 0.119 | 0.069 |

**读法（诚实）：**
- 整体**二分类 accuracy 高（0.84–0.93）**，但这是**被“不转变”占多数**抬高的结果——**persistence（永远预测不转变）的 accuracy 反而更高**。
- **真正预测转变的能力弱**：转变类 **F1 仅 0.13–0.27**，precision/recall 都低（模型要么漏报转变，要么报错）。**这正是源文档 §2.3 警告的“主态占比虚高”——所以我们同时报了 macro-F1、balanced accuracy 与转变类 F1。**
- 结论：**在 1 ns 帧间隔、该 MSM 划分下，模型尚不能可靠预测“何时跳态”**；但能把“继续处于同一亚稳态”预测得较准（即**多数情况下状态确实不变，模型也预测不变**——这部分是对的）。

---

## 3. TASK 3 — 温度泛化性

### 3.1 温度输入考察（用户问：predictor/rollout 是否有温度输入？）
| 模型 | 温度入口 | 是否第一类输入 |
|---|---|---|
| **Model 1** | Stage A `time_embed(lag_ps, temperature_k)`；ensemble head `cond=[log(horizon_ps)/log(1e4), (T−300)/150]` | 是（但**当代 ckpt 的 head `cond_dims=0`，温度惰性**→ 已**重训** `cond_dims=2` 头） |
| **Model 2** | GLE `temperature_proj` / history posterior `_cond` / bath `bath(seq_feature, temperature_k)`；`rollout(..., temperature_k)` | 是（架构支持） |

**5 个温度**：`{320, 348, 379, 413, 450} K`。

### 3.2 泛化协议（矩阵）
- **行**：每个温度 T* ∈ 5 个温度（450 = 训练时**从未见过**的温度）。
- **列**：{seen-protein, unseen-protein} × {correct-T, wrong-T, no-T}。
- **指标**：ensemble RMSF L1（↓）、RMSF-r（↑）、skill vs persistence。

### 3.3 Model 1 结果（temp-conditioned head，`cond_dim=2`，每格 12 窗口，n_samples=8，horizons 1/2/4/8）

| 温度 (T*) | 蛋白集 | RMSF correct | RMSF wrong-T | RMSF no-T | skill c/w/n | RMSF-r c/n |
|---|---|---|---|---|---|---|
| 320 (seen T) | unseen protein | 0.322 | **0.963** | 0.303 | 0.568/−0.290/0.594 | .590/.611 |
| 320 | seen protein | 0.298 | **0.846** | 0.294 | 0.617/−0.088/0.621 | .576/.592 |
| 348 | unseen protein | 0.489 | 0.484 | 0.514 | 0.571/0.575/0.548 | .600/.603 |
| 348 | seen protein | 0.385 | 0.403 | 0.419 | 0.616/0.598/0.582 | .615/.637 |
| 379 | unseen protein | 0.518 | 0.506 | 0.526 | 0.547/0.558/0.539 | .567/.573 |
| 379 | seen protein | 0.635 | 0.738 | 0.795 | 0.577/0.509/0.471 | .621/.610 |
| 413 | unseen protein | 0.650 | 0.851 | 0.929 | 0.596/0.470/0.422 | .604/.613 |
| 413 | seen protein | 0.646 | 0.828 | 0.886 | 0.574/0.454/0.416 | .567/.558 |
| **450** (unseen T) | unseen protein | **2.135** | 2.666 | 2.818 | 0.460/0.326/0.288 | **.344**/.483 |
| **450** (unseen T) | seen protein | **1.011** | 1.384 | 1.467 | 0.563/0.402/0.366 | .429/.598 |

**读法：**
1. **模型确实在用温度**：**喂错温度普遍显著掉点**（如 T320 从 0.32 → 0.96 RMSF；T413 从 0.65 → 0.85；skill 从 +0.57 掉到 +0.45 甚至 −0.29）。这是 Task 3 要求的最有信息量的对照。
2. **no-T 与 correct-T 接近**（多数格 diff 很小），说明模型可用 anchor 上下文**隐式**恢复一部分温度信息；但**在高压/高温端 correct-T 明显更优**。
3. **温度响应曲线单调**（`temperature_monotone=True`）：predicted RMSF 随 T 上升（RMSF-r 在 T450 掉到 0.34，说明**高温下形状预测更难**）。
4. **未见温度 T=450 是最大 gap**：RMSF-r 仅 0.34（seen T 上 ~0.58–0.62），skill 仅 0.46（唯一下降到 0.46 的 T*）。**seen-protein 优于 unseen-protein**（外推越远越难）。
5. **内插 vs 外推**：T=348/379（内插）几乎不受 wrong-T 影响；T=320/413/450（边缘/外推）对 wrong-T 高度敏感。与计划文档 §4.2 family 3 预期一致。

### 3.4 Model 1 对照：无温度头（`cond_dim=0`）
同协议跑原 `stage_ensemble_1000dom` 头（温度惰性）：正确/错误/无温度三列**完全相同**（温度槽为空），RMSF 随 T 单调上升但不随 T* 改变行为（`artifacts/eval_temperature_generalization_cond0.json`）。**证明：只有 `cond_dim=2` 的头才会对 wrong-T 有反应。**

### 3.5 Model 2（from-none GLE v4g）温度矩阵（每格 12 蛋白，n_rollout_reps=3，horizons 1/4/16）

| 温度 | RMSF@16ns correct | wrong-T | no-T | skill_c |
|---|---|---|---|---|
| 320 | 4.561 | 4.561 | 4.562 | 0.207 |
| 348 | 4.982 | 4.983 | 4.982 | 0.293 |
| 379 | 5.120 | 5.136 | 5.128 | 0.318 |
| 413 | 5.804 | 5.833 | 5.818 | 0.453 |
| **450** | **6.908** | 6.944 | 6.933 | 0.506 |

**读法：** Model 2 的 **correct ≈ wrong ≈ none**（差异 < 1%），即**温度输入在 v4g 这条 from-none 路径上几乎惰性**——架构支持温度（`temperature_proj`），但训练后该通路贡献极小。（v4g 训练用 `train.jsonl` 含全部 5 个温度。）
- **温度响应单调**（RMSF 4.56→6.91 随 T 上升），但这是**隐含**的（可能来自 anchor 上下文），**非**来自显式温度输入。
- **结论**：要让 Model 2 真正温度敏感，需要在温度留出/温度扰动训练下重训（`run_fromnone_temp_v2.sh` 已启动，见 §3.6）。

### 3.6 温度留出（LOTO）训练
- **Manifests**：`artifacts/mdcath_manifest_1000dom_temp/{train_not450(4T,15999),val_not450(4T,2000),val_t450(500)}.jsonl`（T=450 留出）；新增 `train_t450.jsonl`（seen-protein × T=450，用于补全矩阵 seen 格）。
- **Model 1 temp-cond 头**：`run_ensemble_temp_1000dom.sh` → `artifacts/stage_ensemble_1000dom_temp_cond`（`cond_dims=2`，train/val 均为温度切分）。
- **Model 2 温度切分重训**：`run_fromnone_temp_v2.sh` → `artifacts/fromnone/stage_cb1b_temp_v2`（从 anchor 恢复、LR 1e-4、encoder-lr-scale 2.0，**训练集不含 T=450**）。该 run 早期曾因非有限梯度崩溃，已用成熟 v4g 配方（resume anchor）修复，现稳定推进。

#### 3.6.1 温度留出外推代价（Model 2：含 T450 训练 vs 不含 T450 训练）
两个 ckpt 都在**未见蛋白**的 `val_t450` 上评测（correct-T，horizon=16 ns，n_rollout_reps=3）。`v4g` 训练用 `train.jsonl`（**含全部 5 个温度**）；`temp_v2` 训练用 `train_not450.jsonl`（**完全不含 T=450**，训练到满 70000 步）。

| 温度 | v4g（训练见过 T=450）skill | temp_v2（训练不含 T=450，70k）skill | Δskill |
|---|---|---|---|
| 320 | 0.207 | 0.197 | −0.010 |
| 348 | 0.293 | 0.250 | −0.043 |
| 379 | 0.318 | 0.263 | −0.054 |
| 413 | 0.453 | 0.435 | −0.018 |
| **450（留出 T）** | 0.506 | **0.479** | **−0.026** |

**读法（诚实，满训练步数口径）：**
- **温度留出带来温和但一致的精度损失**（Δskill −0.01 ~ −0.05），**不是灾难性退化**：即便训练**完全没见过 T=450**，模型在该温度上的 skill 仍达 **0.479**（未留出的 v4g 为 0.506）。
- 损失在**中间温度（348/379）**略大（−0.04~−0.05），在**留出温度 450** 反而较小（−0.026）——说明 450 的高温行为主要受 anchor 上下文与 GLE 的隐含温度响应支配（§3.5 已证温度输入近乎惰性），显式温度切分影响有限。
- 两个 ckpt 的 **RMSF 都随温度单调上升**（响应曲线物理正确），泛化差异体现在**幅度/形状精度**而非**响应方向**。
- 综合 §3.5 + §3.6.1：**Model 2 的温度“泛化”目前主要靠 anchor 隐式携带，而非显式温度输入**；要让温度成为可控轴，需在 encoder/decoder 端显式注入并在温度扰动下训练。

### 3.7 验收对照
| 验收项 | 状态 |
|---|---|
| 5 × {seen,unseen protein} × {correct,wrong,none} 矩阵存在 | ✅（Model 1 full；Model 2 单列蛋白集） |
| 至少一个 ckpt **真的用了**温度（wrong-T 改变输出） | ✅（Model 1 cond_dim=2；wrong-T 显著掉点） |
| 明确说明内插 vs 外推 | ✅（T450 最大 gap，边缘温度敏感；§3.6.1 给出外推 Δskill） |
| 每个留出 T 可证不在其 train manifest 内 | ✅（`protocol.validate_manifest_split` + 温度计数核对：train_not450 无 450） |

---

## 4. 四张表汇总（均附 n + 采样预算）

| 表 | 文件 | n / 预算 |
|---|---|---|
| **① 静态系综（BioEmu 式）** | `artifacts/eval_states_msm_1000dom.json` | 40 蛋白 / 800 窗口 / n_samples=8 |
| **② 动力学（MSM/lagged + 转变检测）** | 同上 | 40 真实轨迹拟合 MSM；160 eval/格 |
| **③ 主链（backbone）** | `results/backbone_suite_model1_1000dom.json` | 60 窗口 / n_samples=8 |
| **④ 温度泛化** | `artifacts/eval_temperature_generalization_cond_full.json`（Model 1）、`results/temperature_generalization_model2_v4g.json` + `results/temperature_generalization_model2_tempsplit.json`（Model 2） | Model 1: 12 窗口/格；Model 2: 12 蛋白 × 3 reps/格 |

---

## 5. 复现命令

```bash
export PYTHONPATH=/liuyutian/protein_jepa_frame

# T1 主链评测（对照真实 MD N/CA/C/O）
python -m scripts.eval_backbone_suite \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --b1a-ckpt artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
  --ensemble-ckpt artifacts/stage_ensemble_1000dom/checkpoint_best.pt \
  --manifest artifacts/mdcath_manifest_1000dom/val.jsonl \
  --batches 30 --n-samples 8 --out results/backbone_suite_model1_1000dom.json
# 主链 PDB
python scripts/export_eval_trajectories_pdb.py --atoms backbone --ensemble \
  --which ground_truth persistence ours_stageA \
  --outdir artifacts/pdbs_1000dom_backbone_v1 --n-windows 2

# T2 静态 + 动力学 + 转变检测
python -m scripts.eval_states_and_msm \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --b1a-ckpt artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
  --ensemble-ckpt artifacts/stage_ensemble_1000dom/checkpoint_best.pt \
  --manifest artifacts/mdcath_manifest_1000dom/val.jsonl \
  --max-proteins 40 --horizons 1 4 16 64 192 --n-samples 8 \
  --out artifacts/eval_states_msm_1000dom.json

# T3 Model 1 温度矩阵（correct/wrong/none × seen/unseen protein × 5 T）
python -m scripts.eval_temperature_generalization \
  --ensemble-ckpt artifacts/stage_ensemble_1000dom_temp_cond/checkpoint_best.pt \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --b1a-ckpt artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
  --manifest artifacts/mdcath_manifest_1000dom_temp/val_not450.jsonl \
  --manifest-unseen-temp artifacts/mdcath_manifest_1000dom_temp/val_t450.jsonl \
  --manifest-seen artifacts/mdcath_manifest_1000dom_temp/train_not450.jsonl \
  --manifest-seen-t450 artifacts/mdcath_manifest_1000dom_temp/train_t450.jsonl \
  --out artifacts/eval_temperature_generalization_cond_full.json

# T3 Model 2 温度矩阵（from-none）
python -m scripts.eval_temperature_generalization_model2 \
  --ckpt artifacts/fromnone/stage_cb1b_v4g/checkpoint_latest.pt \
  --manifest-seen-temp artifacts/mdcath_manifest_1000dom_temp/val_not450.jsonl \
  --manifest-unseen-temp artifacts/mdcath_manifest_1000dom_temp/val_t450.jsonl \
  --horizons 1 4 16 --max-proteins 12 --modes correct wrong none \
  --out results/temperature_generalization_model2_v4g.json
```

---

## 6. 局限与下一步（诚实）
1. **主链**：Model 1 主链为 Cα+理想几何拼出，**肽键/ω 不自洽**；要真正逐残基自洽需训练 **frame + torsion 头**（计划文档 T1.2 option B）。
2. **MSM 状态预测**：1 ns 帧间隔下**转变检测 F1 低**；下一步可缩短有效 lag / 用更多 anchor / 更长轨迹提升转变样本量。
3. **温度泛化**：Model 1 的 temp-cond head 是**轻头重训**（冻结栈），可进一步在 **encoder 端**注入温度；Model 2 的温度留出训练（`run_fromnone_temp_v2.sh`，已训完 70k）显示**外推代价温和（Δskill ≈ −0.03）但温度输入本身近乎惰性**（§3.5），需在 encoder/decoder 端**显式注入**温度并在温度扰动下训练以激活 `temperature_proj` 通路，再用 §3.5 矩阵复测。
4. **coverage**：单一 ε 会高估，报告已附 population TV/JSD + free-energy MAE 作为稳健补充。
