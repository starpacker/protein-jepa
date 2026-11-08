# 评测方法与复现命令总报告（Protein JEPA Frame）
> **日期:** 2026-10-21（v2 全面扩充） · **配套:** [`PROJECT_REPORT_20261021.md`](PROJECT_REPORT_20261021.md)（模型/训练细节）
> **范围:** 本文件是**评测方法（method）与复现命令（command）的权威描述 + 全部评测结果汇总**。既包含三个最新任务（T1 主链重建 / T2 BioEmu+MSM / T3 温度泛化），也包含此前所有评测指标（CA-RMSD、Root-Mean W2、Variance W2、MD-PCA W2、接触 Jaccard、动力学能量分数、几何有效性、轨迹级 order-aware 指标等）。
> **约定:** 每个数字都标注 **n（窗口/蛋白数）+ 采样预算 + 方向（↑/↓）**；分布/静态指标与动力学指标分表。
> **被评测的方法（5 个 + 2 个参考基线）见 §1.2**；每条结果的复现命令见 §9（15 条，逐条校验过参数）。
> **五方法同一 harness 的完整横评**（unseen 蛋白 / 温度泛化 / 1–64 ns / rollout）在
> [`PROJECT_REPORT_20261021.md`](PROJECT_REPORT_20261021.md) **§13**，本文件 §3 是历史口径记录。

---

## 0. TL;DR（一页结论）

### 0.1 全部评测指标家族（本文件覆盖）

| 家族 | 指标 | 方向 | 章节 |
|---|---|---|---|
| **位置/位移** | common-anchor CA-RMSD、Kabsch CA-RMSD、per-atom (N/CA/C/O) RMSD、displacement MSD、shape RMSD | ↓ | §2.1 |
| **系综分布** | RMSF（幅度 L1）、RMSF-r（形态 Pearson）、**Root-Mean W2**、**Variance W2**、**MD-PCA W2**、internal-distance EMD、Weak-Contact J、Transient-Contacts J | RMSF-r/J↑，其余↓ | §2.2 |
| **能量分数** | physical-distance energy score（系综）、trajectory energy score（路径） | ↓ | §2.3 |
| **几何有效性（CA）** | CA bond mean/invalid frac、nonlocal clash frac、Rg、internal-dist RMSD/contact-error、CA pseudo-torsion MAE | 见表 | §2.4 |
| **主链原子级（T1）** | backbone RMSD (N/CA/C/O)、键长/键角 validity、Ramachandran allowed、ω trans、肽键 C–N validity、backbone clash | validity↑，RMSD/clash↓ | §2.5 |
| **静态系综（BioEmu 口径，T2A）** | reference-state coverage、free-energy MAE、population MAE/TV/JSD | coverage↑，其余↓ | §2.6 |
| **动力学/状态预测（T2B）** | lagged state acc/macro-F1/balanced-acc、transition-matrix error (TV/JSD/Frobenius)、implied timescales、MFPT、**transition-detection**（Y/N） | acc/F1↑，误差↓ | §2.7 |
| **轨迹级（order-aware）** | per-step RMSD、path_length/end_to_end/tortuosity、Rg curve、ACF lag-1、ACF-corr、walk-RMSF/RMSF-r、traj energy score | 漂移类↓，一致性↑ | §2.8 |
| **温度泛化（T3）** | 5×{seen/unseen}×{correct/wrong/none} 矩阵（RMSF/RMSF-r/skill）、temperature_monotone | skill/RMSF-r↑，RMSF↓ | §2.9 |

### 0.2 任务结论（三个新任务 + 五方法横评）

| 任务 | 结论 | 证据文件 |
|---|---|---|
| **T1 主链重建（OpenFold）** | 用 **vendored 官方 OpenFold** 把 N/CA/C/O 主链**贯通到 producer → 评测 → PDB 导出**。Model 1 的 backbone CA 列与 Cα 系综**逐位一致**（协议保证）。修正了一个评测口径错误：原 suite 的“真值”是从 Cα **重新 lift** 的理想几何，已改为**直接读真实 MD 的 N/CA/C/O**。 | `results/backbone_suite_model1_1000dom.json`、`artifacts/pdbs_1000dom_backbone_v1/` |
| **T2 BioEmu 指标 + MSM 状态预测** | 实现 BioEmu 的（A）**静态系综**指标：reference-state **coverage**=0.540、free-energy **MAE**=0.779 kcal/mol、population **TV/JSD**；（B）BioEmu **没做**的**动力学/状态预测**指标：lagged state accuracy/macro-F1、转移矩阵误差、implied timescales，**以及“能否预测状态转变（或不变）”二分类指标**（转变 F1 0.13–0.27，诚实负结果）。 | `artifacts/eval_states_msm_1000dom.json` |
| **T3 温度泛化性** | **两条模型都考察了温度输入**。Model 1（temp-cond head）给出完整 **5×{seen,unseen protein}×{correct/wrong/none}** 矩阵；**喂错温度显著掉点** → 真在用温度。Model 2（from-none GLE v4g）**correct≈wrong≈none** → 温度输入**几乎惰性**。另用温度留出训练检查未见温度外推（Δskill≈−0.03）。 | `artifacts/eval_temperature_generalization_cond_full.json`、`results/temperature_generalization_model2_v4g.json` |
| **五方法同口径横评** | Model 1 / Model 2 / Lasp-final / MDGen / bioemu2 放进**同一 harness、同一 140 窗口计划**（5 温度 × {seen,unseen} × 1/4/16/64 ns）。结论：**系综指标 bioemu2 领先我们（19:4）**，但我们**Kabsch 位置精度 4 个时标全胜**、**64 ns rollout 更稳**；MDGen 对 bioemu2 是 0:20。 | `results/five_method/all.jsonl`（→ 主报告 §13） |

---

## 1. 评测协议（所有指标共享的骨架）

### 1.1 数据与时间口径

| 项 | 设定 |
|---|---|
| 数据集 | **mdCATH**，帧间隔 **1 ns/帧**（`step_ps=1000`） |
| 温度 | **5 个**：`{320, 348, 379, 413, 450} K` |
| 划分 | **1000dom**：800 训练域 / 100 验证域 / 0 重叠（`protocol.validate_manifest_split`） |
| 窗口 | anchor 帧 → 未来 horizon 帧（horizons ∈ {1,4,16,64,192} ns）|
| 温度切分 | `train_not450`(4T,15999) / `val_not450`(4T,2000) / `val_t450`(500) / `train_t450`（seen-protein × 450K）|

### 1.2 被评测的方法清单（5 个模型 + 2 个参考基线）

> **本节是“到底评了哪些 method”的权威清单。** 每个 method 都有一张同口径结果表 + 一条 §9 复现命令。

| # | 方法 | 是什么 | 评测入口 | 结果（本文件） |
|---|---|---|---|---|
| M1 | **模型 1**（Stage-A ensemble） | mdCATH 训练的 Stage-A encoder + B1a 解码器 + Gaussian 系综头 | `scripts/eval_ensemble_metrics_suite.py`（§9.1） | §3.1 / §7-① |
| M2 | **模型 2**（from-none GLE v4e） | 不经 Stage-A/B1a，直接从 anchor 用 GLE 生成系综 | `evaluation/run_eval_fromnone_ensemble.py`（§9.3） | §3.2 / §7-③ |
| M3 | **Lasp-final**（解冻解码器的 GLE 变体） | Stage-A 低 LR 联合微调 + **解冻** B1a 解码器 + GLE kernel | `scripts/eval_lasp_final.py`（§9.6） | §3.8 / §7-⑧ |
| M4 | **H+S（Plan A）** | Stage-A+GLE 直连、clean paired 对模型 1 | `scripts/paired_hs_vs_model1.py` / `scripts/paired_walk_metrics.py`（§9.6） | §3.6 / §3.7 / §7-⑥⑦ |
| M5 | **MDGen**（zero-shot + mdCATH 短训） | 外部基线；官方 ATLAS 权重 / 2 个短训 ckpt | `scripts/eval_mdgen_baseline.py`（§9.4） | §3.3 / §7-④ |
| M6 | **bioemu2**（zero-shot，`bioemu2-preview`） | Microsoft 全原子扩散系综模型，**零微调** | `run_scripts/run_bioemu2_zeroshot{,_rep}.sh`（§9.5） | §3.4 / §7-⑤ |

两个**参考基线**（不是竞争方法，但每张表都要有）：

| 基线 | 定义 | 作用 |
|---|---|---|
| **persistence** | anchor 帧复制成系综（delta 系综） | “必须打赢”的诚实对手；短时标天花板 |
| **oracle** | 真值作预测（`pred = true`） | 指标实现的**正确性校验**（RMSD/W2=0、RMSF-r=1） |

> ⚠️ **五方法「同一 harness、同一窗口计划」的完整横评不在本文件**，而在
> [`PROJECT_REPORT_20261021.md`](PROJECT_REPORT_20261021.md) **§13（结论）与 §13.8（T0–T8 大表）**。
> 本文件 §3 的各表来自**不同的历史 harness / 窗口集**，**只能上下比较方向，不能跨节比数字**；
> 要同表比较请用 §13.8。

### 1.3 公平性/因果契约（`evaluation/COMPARISON_PROTOCOL.md`）

- 所有 learnable variant 共用**同一份**内容校验的 encoder + train-only normalizer + 冻结 raw-H decoder。
- **H+S 与 H-Markov 独立训练**：同窗口、同监督、同 optimizer 族、同更新数、同样本预算、同 loss 权重、同选点指标；**同时报参数量与算力**。
- **clean paired 评测**四条：①同一冻结栈（模型 1 与 H+S 各持 deepcopy）②同一窗口集合 ③同一 M/horizon，每窗口取 Δ 做**配对 bootstrap** ④全局 RNG 播种、跨进程可复现。
- 保留全部 finite/非 finite 结果；非 finite 预测直接 fail；**无样本拒绝 / best-of-M / 对齐性挑选**。

### 1.4 聚合与不确定度（`evaluation/metrics.py`）

- 原始行是**窗口**，不是独立蛋白；**先在每个温度内对 anchor/replica 求平均，再按蛋白做 cluster bootstrap**。
- 输出 `mean ± SEM`、配对 `Δ` 的 **95% protein-bootstrap CI**、`win_rate`。
- 主比较量：**H+S − H-Markov**，在每个 horizon 分别报告，不做 favorable-horizon 挑选。

### 1.5 几何约定（LOCKED，`evaluation/geometry.py` 单一来源）

全 evaluation 包的约定是**不重实现 Kabsch**；`common_anchor_rmsd` = 由**真值**拟合的 anchor→target 变换（所有样本共享同一 gauge，无泄漏）；`aligned_rmsd` = 去质心 Kabsch 最小 RMSD。**2026-11-01 审计例外：T1 v1 实际存在独立 Kabsch 实现且不符合其 RMSD 命名，见 §2.5；不能据此段为旧 T1 表背书。新增 v2 的 shape 对齐调用共享 `evaluation.geometry.kabsch_align`。**

---

## 2. 指标全集（定义 + 单位 + 方向 + 代码 + 校验）

> 代码真源在 `evaluation/`。下表逐指标给出**精确定义**，便于复现与引用。

### 2.1 位置 / 位移精度类

| 指标 | 定义 | 单位 | 方向 | 代码 |
|---|---|---|---|---|
| **common-anchor CA-RMSD** | anchor→真值变换（**由真值拟合**）下预测 CA 的 RMSD；所有样本共用一个 gauge，**无泄漏** | Å | ↓ | `geometry.common_anchor_rmsd` |
| **Kabsch CA-RMSD**（aligned） | 去质心 Kabsch 对齐后最小 RMSD（消除刚体自由度） | Å | ↓ | `geometry.aligned_rmsd` |
| **displacement MSD** | 逐残基位移平方均值（排除无效残基） | Å² | ↓ | `geometry.displacement_msd` |
| **sample RMSD / shape RMSD** | 每个采样构象的 anchor-RMSD / shape-RMSD（用于分布） | Å | ↓ | `run_eval.py` |
| **物理距离 energy score** | 经验 CA 距离预测的 energy score（非局域对 ≥3、固定 Å 缩放、无 target 归一化） | Å | ↓ | `geometry.physical_distance_energy_score` |

**Oracle 校验**：`pred=true` → 全部 RMSD/MSD ↔ **0**（backbone 实测 8.6e−15）。

### 2.2 系综分布类（order-free bag）

> 设真值系综 `X_true ∈ [T,L,3]`、预测系综 `X_pred ∈ [N,L,3]`。代码：`evaluation/ensemble_metrics.py`（原文件）+ `evaluation/ensemble_metrics_ext.py`（新增）。

| 指标 | 定义 | 单位 | 方向 | 代码 |
|---|---|---|---|---|
| **RMSF（幅度）** | 对齐后逐残基 RMSF 谱的 masked **L1** | Å | ↓ | `rmsf_profile_error.rmsf_mae` |
| **RMSF-r（形态）** | 逐残基 RMSF 谱 vs 真值的 **Pearson r** | — | ↑ | `rmsf_profile_error.rmsf_corr` |
| **ROOT-MEAN W2** | 逐残基 CA 位置分布 W2 的 RMS：`sqrt(mean_i W2_i²)` | Å | ↓ | `root_mean_w2` |
| **VARIANCE W2** | 逐残基**涨落幅度 σ²** 分布之 W2 | Å² | ↓ | `variance_w2` |
| **MD-PCA W2** | 真值协方差前 **k=3** 慢模投影分布 W2 之 RMS | Å | ↓ | `md_pca_w2` |
| **internal-distance EMD** | 逐对距离分布的一阶 Wasserstein（EMD，越低越好） | Å | ↓ | `distance_distribution_emd` |
| **WEAK-CONTACT J** | `P(d<8Å)∈[0.05,0.5)` 位点集合的 **Jaccard** | — | ↑ | `weak_contact_jaccard(_full)` |
| **TRANSIENT J** | `P(d<8Å)∈(0.01,0.99)` 位点集合的 **Jaccard** | — | ↑ | `transient_contact_jaccard(_full)` |

**所有 W2 均为精确一维 Wasserstein-2**（排序分位函数、**无分箱**）：`W2² = ∫₀¹ (F_a⁻¹(q) − F_b⁻¹(q))² dq`（`w2_1d`）。

**实现校验（`PROJECT_REPORT` §2.1）**：
1. oracle → 4×W2 = **0.000**、2×Jaccard = **1.000**、RMSF-r = **1.000**、RMSF_mae = **0**。✔
2. persistence 的 **RMSF-r 恒为 0**（delta 系综 RMSF 谱是常数 → Pearson 退化）。✔
3. persistence 的 contact-J/transient-J = 0，因 delta 系综 contact-prob 图只有 {0,1}，弱/瞬时带**一个都落不进**（**这是 delta 基线的数学性质，非 mdCATH 性质**）；`*_full` 变体返回 `n_true/n_pred` 可追溯。✔

> ⚠️ **引用注意**：接触类 Jaccard 对 delta/近刚体基线退化，**必须附 `n_true/n_pred`**。综合分**不是单一标量**，须按「分布形态 / 位置精度 / 接触」三类分别陈述。

### 2.3 能量分数

| 指标 | 定义 | 单位 | 方向 | 代码 |
|---|---|---|---|---|
| **physical-distance energy score** | 非局域 CA 距离的 multivariate energy score（V-statistic，含对角） | Å | ↓ | `geometry.physical_distance_energy_score` |
| **trajectory energy score** | 整条路径集合的经验分布 energy score（order-aware） | Å | ↓ | `trajectory_metrics.traj_energy_score` |

### 2.4 CA 几何有效性（诊断，`geometry.py`）

| 指标 | 定义 | 容差/约定 | 代码 |
|---|---|---|---|
| **CA bond mean / invalid frac** | 相邻 CA 键长均值 / 越界比例 | 有效带 **3.3–4.3 Å** | `ca_geometry_summary` |
| **nonlocal CA clash frac** | 非局域 CA 对 < cutoff 的比例 | clash cutoff **2.5 Å** | `ca_geometry_summary` |
| **CA radius of gyration** | 有效残基 Rg | Å | `ca_geometry_summary` |
| **distance RMSE** | 非局域对距离 RMS 误差 | Å | `ca_observable_errors` |
| **contact-error fraction** | 8 Å 接触集合不一致比例 | — | `ca_observable_errors` |
| **CA pseudo-torsion MAE** | 有符号 CA 伪二面角端点误差 | rad（wrap） | `ca_observable_errors` |

> 这些是 **CA-only 有效性诊断**（键长带、clash cutoff 是固定诊断约定），**不替代**完整主链立体化学 → 见 §2.5。

### 2.5 主链原子级指标（T1；历史 v1 与 2026-11-01 v2 分列）

> 输入主链张量 `bb [.., L, 4, 3]`（顺序 N, CA, C, O）。预测摆放可由 **vendored 官方 OpenFold** 完成；评分真值必须是**真实 MD 的 N/CA/C/O**，不能用真实 CA 的重新 lift 替代。

**历史 v1 定义表保留如下，供追溯旧命名，不代表实现已经满足表内定义。** `evaluation/backbone_metrics.py` 的“RMSD”实际先独立全主链 Kabsch，再取原子欧氏距离的算术均值；它不是共同 anchor gauge 下的均方根误差。旧 φ/ψ 配对错位一个残基，clash 的链/序号条件错误地使用 AND，且仅检查 CA；旧 suite 仅取每窗第一个预测样本的几何。旧结果和旧代码保留，不能与 v2 数值直接比较。

| 指标 | 定义 | 单位 | 方向 | 代码 |
|---|---|---|---|---|
| **backbone RMSD** | 共同 anchor gauge 下 N/CA/C/O 的 RMSD | Å | ↓ | `backbone_rmsd` |
| **per-atom RMSD (N/CA/C/O)** | 逐原子类型 RMSD | Å | ↓ | `backbone_per_atom_rmsd` |
| **bond-length mean / valid frac** | N–CA / CA–C / C=O / C–N 键长统计 | Å / frac | ↓ | `backbone_geometry_summary` |
| **bond-angle valid frac** | N–CA–C 键角有效性 | frac | ↑ | 同上 |
| **Ramachandran allowed frac** | φ/ψ 落于 allowed 区比例 | frac | ↑ | 同上 |
| **ω trans frac** | 肽键 ω 处于 trans 的比例 | frac | ↑ | 同上 |
| **peptide C(i)–N(i+1) valid** | 肽键 C–N 距离有效性 | frac | ↑ | 同上 |
| **backbone clash frac** | 非局域主链 clash 比例 | frac | ↓ | 同上 |

**历史统一几何参考**：`IDEAL_BOND = {N_CA:1.458, CA_C:1.525, C_N:1.329, C_O:1.231}`（Å）。这些统一常数及容差属于旧诊断约定，不等于 OpenFold 对所有残基的完整参考分布。

**2026-11-01 新版基础指标**：[`evaluation/backbone_metrics_v2.py`](evaluation/backbone_metrics_v2.py)，`schema_version=backbone_metrics_v2`。每个样本均计算完整几何并保留 `per_sample`；`mean` 是样本等权平均，计数另存 `totals`，后续仍需按蛋白聚合与估计不确定度。

| v2 输出 | 定义 / 范围 | 单位 |
|---|---|---|
| `coordinate_rmsd_angstrom`、`coordinate_rmsd_N/CA/C/O_angstrom` | `sqrt(mean(sum_xyz(error²)))`；调用方只用真实 future CA→observed anchor CA 拟合变换，并将其用于**全部真实主链原子**；预测保留 anchor gauge，主指标不拟合预测 | Å |
| `shape_kabsch_rmsd_angstrom`、`shape_rmsd_N/CA/C/O_angstrom` | 每样本独立全主链 Kabsch 后的真 RMS；各原子类型共用该次对齐，与主指标分列 | Å |
| `bond_*_mean_angstrom`、`angle_*_mean_deg` | N–CA、CA–C、C–O、C–N 及 N–CA–C、CA–C–N、C–N–CA；按实际所需原子、链边界和序号 gap 掩码 | Å / ° |
| `*_legacy_valid_frac` | 明示沿用历史键长/键角窗口的诊断比例，不新增验收阈值 | frac |
| `ramachandran_box_diagnostic_frac` | **同一残基**的 φ/ψ 落入历史矩形框；需左右两条肽键有效。没有正式残基类别统计参考，**不是 MolProbity allowed** | frac |
| `omega_trans_diagnostic_frac` | 实际 CA–C–N–CA 二面角满足历史 `abs(ω)>150°`；另报偏离 trans 的均值与有效数 | frac / ° |
| `ca_nonlocal_clash_frac` | 同链序号差≥3 **或**跨链的唯一 CA 对，距离<2.5 Å；这是 CA 诊断，**没有评估全主链或全原子 clash** | frac |
| `openfold_*_mean_abs_z` | 提供 aatype 时，使用 vendored OpenFold 的逐残基均值/标准差及 next-PRO C–N 参考；连续绝对 z 偏差，不设新 z 阈值 | 无量纲 |

有效原子含 NaN/Inf 会失败；被掩码原子的非有限值排除。键角/二面角退化计数单列；无有效观测返回 `None`。测试见 [`tests/test_backbone_metrics_v2.py`](tests/test_backbone_metrics_v2.py)（36 项 CPU 数值测试通过）。这补齐了可信基础指标，**不等于 T1 已完成**；正式 Ramachandran 统计参考、全主链原子 clash 与当前模型的完整对照评测仍有缺口。

### 2.6 静态/平衡系综指标（T2A，BioEmu 口径，`evaluation/states_metrics.py`）

| 指标 | 定义 | 单位 | 方向 | 代码 |
|---|---|---|---|---|
| **reference-state coverage** | 每个参考态被 ≥0.1% 采样命中的宏平均 | — | ↑ | `reference_state_coverage` |
| **free-energy MAE** | `F_k = −k_B T ln p_k` 的宏平均误差 | kcal/mol | ↓ | `free_energy_mae` |
| **population MAE / TV / JSD** | 宏状态布居的 MAE / 总变差 / JS 散度 | — | ↓ | `state_populations` |
| **assignment** | 用**同 centroids** 最近邻分配（无 model 自由度） | — | — | `assign_to_centroids` |

- 特征空间：**internal-distance**（成对 CA 距离，可 mask、旋转不变）。
- 状态定义 = 对**真实** MD 系综聚类（pairwise Cα 距离）→ 模型样本用同 centroids 投影（BioEmu §2.2 的“shared feature + shared state”公平规则）。
- `k_B = 1.9872041e-3 kcal/(mol·K)`。

### 2.7 动力学 / MSM 状态预测（T2B，BioEmu **不做**的部分，`evaluation/msm_metrics.py`）

**MSM 拟合（关键）**：在**真实时间序轨迹**上 `TICA(dim=2, lag=1)` → `KMeans(50 micro)` → `MaximumLikelihoodMSM(reversible=True)` → `PCCA` → **5 macrostates**。true 与 pred 轨迹**投影通过同一 TICA + centroids**，标签空间 model-independent。

| 指标 | 定义 | 方向 | 代码 |
|---|---|---|---|
| **lagged state accuracy** | `ŝ(t+τ) == s(t+τ)` 命中率（micro / macro） | ↑ | `lagged_state_prediction` |
| **macro-F1 / balanced accuracy** | 对类别不均衡稳健的版本 | ↑ | 同上 |
| **persistence acc** | `ŝ(t+τ) = s(t)` 基线（永远不跳态） | — | 同上 |
| **transition-matrix error** | 行 TV / JSD / Frobenius | ↓ | `transition_matrix_error` |
| **implied timescales** | MSM 隐含时间尺度（单调递减、无 inf 为健康） | — | `implied_timescales` |
| **MFPT** | 平均首次通过时间矩阵 | — | `mfpt_matrix` |
| **★ transition-detection** | 二分类：`true_change=(s(t+τ)≠s(t))` vs `pred_change=(ŝ(t+τ)≠s(t))` → precision/recall/**F1**、binary acc、`binary_skill_vs_persist` | F1↑ | `transition_detection` |

### 2.8 轨迹级（order-aware walk，`evaluation/trajectory_metrics.py`）

> 系综指标把轨迹当**无序构象袋**，**看不到轨迹顺序**；GLE rollout 是**序列**，需 order-aware 指标。输入 `pred [M,T,L,3]`、`true [T,L,3]`、`anchor [L,3]`。

| 指标 | 定义 | 方向 | 代码 |
|---|---|---|---|
| **per-step RMSD** | 每个未来帧的 common-anchor RMSD（无泄漏） | ↓ | `per_step_rmsd` |
| **path_length** | 路径内距离总长 | ↓（越短越少漂移） | `path_shape` |
| **end_to_end** | 首末帧距离 | ↓ | `path_shape` |
| **tortuosity** | 路径形状（符号之比，越接近真值越好） | ~ | `path_shape` |
| **Rg curve / rg_mean** | 每帧半径回旋 | Å | `rg_curve` |
| **ACF lag-1** | 内距向量自相关 lag-1 | ↑ | `autocorr_time` |
| **ACF-corr** | 预测 ACF 与真值 ACF 的相关 | ↑ | `walk_summary` |
| **walk RMSF / RMSF-r** | 全 walk 池化帧的 RMSF 幅度/形态 | 幅度↓、形态↑ | `walk_rmsf` |
| **traj energy score** | 整条路径集合 energy score | ↓ | `traj_energy_score` |

### 2.9 温度泛化（T3）

| 项 | 设定 |
|---|---|
| 矩阵 | 行 = 5 温度 T*；列 = {seen,unseen protein} × {correct-T, wrong-T, no-T} |
| 指标 | ensemble **RMSF L1**（↓）、**RMSF-r**（↑）、**skill vs persistence**、`temperature_monotone` |
| 机制检验 | **wrong-T 是否掉点** → 判定模型是否**真的在用温度** |

---

## 3. 结果：分布形态 / 位置精度（历史主表）

> 数据口径：**mdCATH-1000dom val（100 蛋白）**，帧间隔 1000 ps。模型 1 / 模型 2 / MDGen / bioemu2 同在 1000dom 家族（唯 MDGen 帧长 OOD）。

### 3.1 模型 1（Stage-A ensemble，1000dom val，50 批 × 4 = 200 窗口，n_samples=8）

> 最终 B1a ckpt + 重训系综头。来源 `artifacts/eval_model1_1000dom/suite_v2.json`。

| 指标 | model | persistence | oracle |
|---|---|---|---|
| RMSF 幅度 ↓ | **0.837** | 1.835 | 0.000 |
| **RMSF-r ↑** | **0.512** | 0.000 | 1.000 |
| ROOT-MEAN W2 ↓ | **16.540** | 16.752 | 0.000 |
| VARIANCE W2 ↓ | **7.878** | 10.712 | 0.000 |
| MD-PCA W2 ↓ | **0.0665** | 0.1174 | 0.0000 |
| WEAK-CONTACT J ↑ | **0.0781** | 0.0000 | 1.000 |
| TRANSIENT J ↑ | **0.1832** | 0.0000 | 1.000 |

**RMSF-r 三形式**：global **0.673**；per-target 均值 **0.553**（83 个域，**98.8% > 0**）。
**Kabsch CA RMSD（rollout 100 帧）**：model **8.39 Å** vs persistence **3.87 Å**（自回归回填漂移，**已知短板**）。

> 对比 R7 之前（B1a 停在 step-250 的系综头）：RMSF-r global 0.538→**0.673**、per-target 0.293→**0.553**、RMSF 1.154→**0.837**、variance_w2 9.31→**7.88**、pca_w2 0.088→**0.067**。**用最终 B1a 重训系综头带来全面且显著提升。**

### 3.2 模型 2（from-none GLE v4e，1000dom val，200 窗口，n_samples=12）

| horizon | RMSF-r ↑ | RMSF↓ | root_W2↓ | var_W2↓ | pca_W2↓ | weak_J↑ | trans_J↑ |
|---|---|---|---|---|---|---|---|
| 1 ns | 0.283 | 1.888 | 6.576 ✗ | 7.253 | 0.182 | 0.400 | 0.012 |
| 4 ns | **0.513** | 1.226 | 3.798 ✗ | 9.790 | 0.099 | 0.137 | 0.275 |
| 16 ns | **0.497** | 1.984 | 2.986 ✓ | 20.514 | 0.149 | 0.195 | 0.345 |
| 64 ns | **0.502** | 2.923 | 4.025 ✓ | 32.405 | 0.192 | 0.152 | 0.383 |
| persist(4ns) | 0.015 | 3.020 | 2.592 | 14.366 | 0.190 | 0.000 | 0.000 |

- **稳定胜**：RMSF 幅度、RMSF-r（4–64 ns）、variance/pca W2、transient-J（100% 窗口）。
- **已知失败**：`root_mean_w2` **短时标（1 ns）** 差（模型位移 1.5 Å ≪ 真实 3.6–7.4 Å），16/64 ns 转正。

### 3.3 Baseline — MDGen zero-shot（mdCATH，32 窗口）

| 模型 | 物理跨度 | RMSF↓ | RMSF-r↑ | root_W2↓ | var_W2↓ | pca_W2↓ | weak_J↑ | trans_J↑ | **Kabsch↓** |
|---|---|---|---|---|---|---|---|---|---|
| persistence | 瞬时 | 1.840 | −0.014 | 12.276 | 11.089 | 0.1345 | 0.000 | 0.000 | **4.014** |
| **MDGen zero-shot** | 1 ns | 4.650 | ~0 | **6.679** | 27.069 | 0.1387 | 0.000 | 0.000 | 7.344 |
| ↳ seed=7 复核 | 1 ns | 4.779 | ~0 | 6.896 | 27.371 | 0.1437 | 0.000 | 0.000 | 7.841 |
| MDGen mdCATH 短训 e1/s2000 | 100 ns | 16.061 | ~0 | 11.252 | 275.31 | 0.4760 | 0.000 | 0.000 | 19.691 |
| MDGen mdCATH 短训 e5/s6000 | 100 ns | 22.525 | ~0 | 14.213 | 541.27 | 0.6736 | 0.000 | 0.000 | 26.761 |
| **ours rollout（模型1）** | 100 ns | **1.455** | **0.334** | 11.993 | **10.211** | **0.0801** | **0.066** | **0.134** | 6.335 |

**结论**：MDGen 官方权重 **zero-shot 在 mdCATH 上劣于 persistence**（Kabsch 7.34 vs 4.01 Å），主因 **10 ps/帧 → 1000 ps/帧 步长 OOD + 蛋白族不同**；唯一亮点是 root_W2 最低（形状分布尚可，位置跑偏）。MDGen mdCATH 短训**远未收敛且震荡**，此预算下不构成有意义的对手。

### 3.4 Baseline — bioemu2 zero-shot（`bioemu2-preview`，1000dom val，200 窗口，n_samples=12）

> Microsoft Research diffusion 全原子系综模型，**零微调**，conditioned on anchor 帧；跑在同一套 `ensemble_metrics_ext.py` + 同一 window 迭代 + 同一 persistence 上。

| horizon | RMSF-r ↑ | RMSF↓ | root_W2↓ | var_W2↓ | pca_W2↓ | weak_J↑ | trans_J↑ |
|---|---|---|---|---|---|---|---|
| 1 ns | **0.431** | 2.501 | 5.588 | 19.783 | **0.087** | 0.000 | 0.165 |
| 4 ns | **0.595** | **2.243** | 3.699 | 24.336 | **0.098** | 0.163 | 0.281 |
| 16 ns | 0.579 | **1.817** | **3.449** | **20.228** | **0.086** | **0.237** | **0.348** |
| 64 ns | **0.609** | **1.715** | **4.172** | **18.552** | **0.092** | **0.209** | **0.377** |
| persistence @64 ns | 0.031 | 5.564 | 4.702 | 41.109 | 0.316 | 0.000 | 0.000 |

**读法**：bioemu2 在**长时标**显著优于 persistence（RMSF-r ~0.6、RMSF 更低、pca_W2 更小、接触 Jaccard 更高）；但**短时标（1 ns）** RMSF/root_W2 反而差于 persistence（扩散模型生成“平衡分布样本”而非“短时确定性演化”）。这是**系综生成 vs 轨迹推进**两类任务的口径差异。

**3 次重复（seed 0/1/2，固定 `PYTHONHASHSEED`）mean±std**（`results/bioemu2_reps/summary.json`）：

| horizon | RMSF-r ↑ | RMSF↓ | root_W2↓ | var_W2↓ | pca_W2↓ | weak_J↑ | trans_J↑ |
|---|---|---|---|---|---|---|---|
| 1 ns | 0.4440±0.0032 | 2.164±0.011 | 5.601±0.003 | 15.156±0.221 | 0.0807±0.0008 | 0.0000±0.0000 | 0.1730±0.0014 |
| 4 ns | 0.5950±0.0094 | 1.846±0.049 | 3.661±0.015 | 18.136±0.602 | 0.0819±0.0021 | 0.1772±0.0020 | 0.2983±0.0030 |
| 16 ns | 0.5971±0.0111 | 1.548±0.042 | 3.308±0.058 | 14.241±0.749 | 0.0771±0.0032 | 0.2433±0.0013 | 0.3611±0.0024 |
| 64 ns | 0.6129±0.0183 | 1.622±0.041 | 4.060±0.049 | 17.617±0.249 | 0.0876±0.0014 | 0.2115±0.0009 | 0.3823±0.0033 |
| persistence @64 ns | 0.0306 | 5.564 | 4.702 | 41.109 | 0.316 | 0.000 | 0.000 |

> **可复现性修复**：原默认配置有三重不确定性（bf16-mixed+TF32+cuDNN 非确定性 / `PYTHONHASHSEED` 未固定）→ 已改用 `--precision 32-true --matmul-precision highest --deterministic` + 按窗口 sha256 派生 seed + 每窗口 `torch.random.fork_rng`。修复后同 seed 跨进程**逐位一致**。
> **与单次结果对照**：RMSF-r / root_W2 / 接触 J 两表基本一致；**RMSF 幅度 / var_W2 / pca_W2 单次系统性偏高 10–30%**（超出噪声）→ 引用建议以 **3 次 mean±std** 为准。

### 3.5 汇总对照（1000dom）

| 模型 | RMSF-r(global)↑ | RMSF↓ | pca_W2↓ | Kabsch↓ |
|---|---|---|---|---|
| **模型 1（Stage-A ens，最终）** | **0.673** | **0.837** | **0.067** | 8.39 |
| **模型 2（from-none GLE v4e）** | 0.51（4 ns） | 1.23（4 ns） | 0.099（4 ns） | — |
| bioemu2 zero-shot（单次） | 0.60（64 ns） | 1.72（64 ns） | 0.092（64 ns） | — |
| bioemu2 zero-shot（3 次 mean） | 0.61（64 ns） | 1.62（64 ns） | 0.088（64 ns） | — |
| MDGen zero-shot | ~0 | 4.65 | 0.139 | 7.34 |
| persistence | 0.000 | 1.77 | 0.111 | 3.31 |

> **读法**：在**各自的历史口径**下，模型 1 的 8 帧窗数字（RMSF-r 0.673）高于 bioemu2 的 64 ns 单点（0.60）。
> **但这不是同口径胜负**——协议、窗口、时标都不同。**在同 harness 的同口径横评里，
> bioemu2 在 RMSF-r / pca_W2 / 接触 J 上反超我们**（见 `PROJECT_REPORT_20261021.md` §13.2/§13.3）。
> 请以 §13 为准，本节此表只保留历史记录。
> **位置精度（Kabsch）仍是短板**——模型 1 rollout 8.39 Å 差于 persistence 3.87 Å。
>
> ⚠️ **本表只能横着看方向，不能跨行比数字**：各行取自不同窗口集与不同时标
> （模型 1 是 1/2/4/8 帧窗的 8 帧未来窗、bioemu2 是 1–64 ns 单点、MDGen 是 100 帧 rollout）。
> **同口径五方法对比**见 `PROJECT_REPORT_20261021.md` §13.8。
> 第五个方法 **Lasp-final** 的单列结果见 §3.8。

### 3.6 Stage A + GLE 直连（Plan A，terminal-only） vs 模型 1

> 同一冻结栈（Stage-A encoder 低 LR 联合微调 + B1a 解码器冻结），GLE kernel 开放训练；`--rollout-steps == --paired-horizons-frames`。clean paired 评测（`scripts/paired_hs_vs_model1.py`）。

| 指标 | 模型 1 | H+S（Plan A） | Δ | 95% CI | 判定 |
|---|---|---|---|---|---|
| RMSF ↓ | **0.8699** | 0.8414 | −0.0284 | [−0.0808, +0.0246] | 平 |
| RMSF-r ↑ | 0.5045 | **0.5361** | +0.0316 | [+0.0020, +0.0629] | ✅显著 |
| ROOT-MEAN W2 ↓ | **13.4780** | 13.4927 | +0.0147 | [−0.0128, +0.0445] | 平 |
| VARIANCE W2 ↓ | **7.7866** | 7.9003 | +0.1138 | [−0.2807, +0.5316] | 平 |
| MD-PCA W2 ↓ | **0.0697** | 0.0719 | +0.0022 | [−0.0009, +0.0053] | 平 |
| WEAK-CONTACT J ↑ | 0.0748 | **0.0780** | +0.0032 | [−0.0033, +0.0095] | 平 |
| TRANSIENT J ↑ | 0.1768 | **0.1855** | +0.0087 | [+0.0017, +0.0157] | ✅显著 |

**读法**：**terminal-only Plan A ≈ 模型 1**（7 项中仅 RMSF-r 与 TRANSIENT-J 显著更好，其余打平，无显著变差）。它是一个与模型 1 同级的可用 GLE 版本，**尚未稳定超越**。
> **两处评测 bug 已修**（全局 RNG 未播种 / 模型 1 与 H+S 共用 encoder 对象），上一轮“7/7 全胜”为假象。**数字以 `results/plan_a_hs_vs_model1.json` 为准**；`plan_a_vs_model1.json` / `plan_a_dense_*.json` 为修正前产物，**勿引用**。

### 3.7 轨迹级（order-aware walk）：H+S **显著优于** 模型 1

> `evaluation/trajectory_metrics.py` + `scripts/paired_walk_metrics.py`；协议 = **contiguous**（`K+H` clip，rollout 第 i 步 ↔ 真实第 i 帧，`step_ps` 匹配），40 窗口、M=6。

| 轨迹指标 | 模型 1 | H+S | Δ | 95% CI | 判定 |
|---|---|---|---|---|---|
| per-step RMSD 均值 ↓ | 9.6941 | 4.4846 | **−5.2095** | [−5.5743, −4.8200] | ✅显著 |
| path_length ↓ | 33.4020 | 7.6145 | **−25.7875** | [−26.9901, −24.4682] | ✅显著 |
| end_to_end ↓ | 15.3347 | 1.8701 | **−13.4646** | [−14.0441, −12.8672] | ✅显著 |
| Rg ↓ | 17.5951 | 16.0821 | −1.5129 | [−1.6685, −1.3532] | ✅显著 |
| traj energy score ↓ | 5.8987 | 2.8326 | **−3.0661** | [−3.3245, −2.7976] | ✅显著 |
| walk RMSF ↓ | 3.5605 | 1.2311 | **−2.3295** | [−2.9661, −1.6787] | ✅显著 |
| walk RMSF-r ↑ | 0.2672 | 0.5789 | **+0.3117** | [+0.2605, +0.3617] | ✅显著 |
| acf lag-1 ↑ | 0.2969 | 0.4075 | +0.1106 | [+0.1035, +0.1174] | ✅显著 |
| acf_corr ↑ | 0.9191 | 0.9048 | −0.0143 | [−0.0202, −0.0086] | 平（均近满分） |

**读法**：**整条轨迹上 H+S 全面、显著优于模型 1**——模型 1 的自回归 rollout 累积**严重全局漂移**（path_length 33 Å、end_to_end 15 Å），而 GLE 轨迹**物理上合理**（7.6 Å / 1.9 Å / 每步 4.5 Å）。这补上了系综指标看不到的维度：**GLE 的价值在“轨迹推进”，而非“终点系综”**。结果：`results/walk_plan_a_contig.json`。

### 3.8 Lasp-final 系综指标（Lasp vs 模型 1，同窗口同批量）

> 来源 `results/lasp_final_vs_model1_bioemu2.json`；**200 窗口 / n_samples=12**。
> ⚠️ **该文件里只有 `persist` / `model1` / `lasp` 三列，没有 bioemu2 列**
> （文件名是历史遗留，bioemu2 的同口径列在 `PROJECT_REPORT_20261021.md` §13.8 T7/T8）。
> 本表与 §3.1 的窗口集不同，**只在本表内部可比，不要与 §3.1–§3.5 逐格相减**。
> Lasp = §3.7b 的"解冻解码器"最终版。命令见 §9.6。

| horizon | 指标 | 模型 1 | **Lasp-final** | 谁更好 |
|---|---|---|---|---|
| 1 ns | RMSF-r ↑ | 0.2851 | **0.3368** | **Lasp** ✅显著 |
| 1 ns | RMSF ↓ | 1.1741 | **1.1028** | **Lasp**（不显著） |
| 1 ns | root_W2 ↓ | **6.0098** | 6.1767 | 模型 1（不显著） |
| 1 ns | var_W2 ↓ | 5.6502 | **5.3598** | **Lasp**（不显著） |
| 1 ns | pca_W2 ↓ | 0.0858 | **0.0832** | **Lasp**（不显著） |
| 1 ns | weak_J ↑ | 0.0000 | 0.0000 | 平（两边都是 0） |
| 1 ns | trans_J ↑ | 0.1224 | **0.1245** | **Lasp**（不显著） |
| 4 ns | RMSF-r ↑ | 0.3645 | **0.4789** | **Lasp** ✅显著 |
| 4 ns | RMSF ↓ | 1.0756 | **1.0718** | **Lasp**（不显著） |
| 4 ns | root_W2 ↓ | 3.9634 | **3.9433** | **Lasp**（不显著） |
| 4 ns | var_W2 ↓ | 8.0140 | **7.5231** | **Lasp**（不显著） |
| 4 ns | pca_W2 ↓ | 0.0843 | **0.0721** | **Lasp** ✅显著 |
| 4 ns | weak_J ↑ | 0.1476 | **0.1548** | **Lasp**（不显著） |
| 4 ns | trans_J ↑ | 0.2367 | **0.2431** | **Lasp**（不显著） |
| 16 ns | RMSF-r ↑ | 0.3476 | **0.4090** | **Lasp** ✅显著 |
| 16 ns | RMSF ↓ | 1.9652 | **1.6003** | **Lasp** ✅显著 |
| 16 ns | root_W2 ↓ | **3.2547** | 3.2882 | 模型 1（不显著） |
| 16 ns | var_W2 ↓ | 20.3343 | **17.2545** | **Lasp** ✅显著 |
| 16 ns | pca_W2 ↓ | 0.1528 | **0.1310** | **Lasp** ✅显著 |
| 16 ns | weak_J ↑ | 0.2055 | **0.2319** | **Lasp** ✅显著 |
| 16 ns | trans_J ↑ | 0.3140 | **0.3206** | **Lasp** ✅显著 |
| 64 ns | RMSF-r ↑ | **0.3483** | 0.3234 | 模型 1（不显著） |
| 64 ns | RMSF ↓ | 3.0579 | **2.1136** | **Lasp** ✅显著 |
| 64 ns | root_W2 ↓ | 4.4359 | **4.3723** | **Lasp** ✅显著 |
| 64 ns | var_W2 ↓ | 33.3190 | **27.2152** | **Lasp** ✅显著 |
| 64 ns | pca_W2 ↓ | 0.2041 | **0.1607** | **Lasp** ✅显著 |
| 64 ns | weak_J ↑ | 0.1815 | **0.2225** | **Lasp** ✅显著 |
| 64 ns | trans_J ↑ | 0.3015 | **0.3369** | **Lasp** ✅显著 |

> **读法（28 项 = 7 指标 × 4 时标，配对 bootstrap 95% CI）**：
> **Lasp 显著赢 15 项、显著输 0 项、13 项无显著差异** —— **没有任何一项显著变差**。
> 优势随时标拉长**变大**：64 ns 的 RMSF 2.11 vs 3.06、var_W2 27.2 vs 33.3、
> pca_W2 0.161 vs 0.204 —— **解冻解码器显著抑制长时标过度离散**。
> 唯一均值变差的两项是 **64 ns RMSF-r（0.323 vs 0.348）与 16 ns root_W2**，两者**都不显著**。
> 结论：**Lasp-final 是模型 1 的无显著退步改进版**，可作为主推 ckpt。

---

## 4. TASK 1 — OpenFold 主链（backbone）原子重建

### 4.1 做了什么

> 以下 T1.1–T1.5 是历史实现记录。2026-11-01 审计确认：Model 1 已有 learned frame/torsion decoder；CA-only Model 2 的 deterministic lift 是另一条路径。原子坐标由 OpenFold 摆放不能单独证明学得了完整主链。

- **T1.1** 把冻结点态解码器**已有的** OpenFold atom37 输出（`backbone_direct`/`atom37_coords`/`torsions`）**透出**到 `EnsembleProducer` 与 `HSEnsembleProducer`，返回新键 `backbone_samples [B,n,L,4,3]`、`atom37_samples`、`torsion_samples`，**不改**原有 `ca_samples` 键。
- **T1.2** Model 2（只输出 Cα）通过 `openfold_ca_lifter.lift_ca_to_backbone`（**原子摆放仍由 OpenFold 完成**：group-0 理想 offsets + `Rigid.make_transform_from_reference` + `frames_to_atom37`）得到主链，**无需重训**。
- **T1.3** 新模块 `evaluation/backbone_metrics.py`（指标定义见 §2.5）。
- **T1.4** `export_eval_trajectories_pdb.py --atoms backbone` 写真实 N/CA/C/O 的 ATOM 记录；并加**断言**：backbone 的 CA 列必须等于 `ca_samples`（atol 1e-4）。
- **T1.5** `tests/test_openfold_backbone_path.py`（CA 列 === ca_samples 逐位）、`tests/test_backbone_metrics.py`（oracle→RMSD 0、键角容差、UNK 处理）。

### 4.2 结果（Model 1，mdCATH-1000dom val，30 批 × batch 2 = 60 窗口，n_samples=8，horizon=1 ns）

> ✅ **口径修正**：真值 backbone 现为**真实 MD 的 N/CA/C/O**（此前误用 Cα 重新 lift 的理想几何，导致 oracle 的肽键 validity 也显示 0，属评测 bug，已修）。

> **2026-11-01 二次审计：下表为保留的 v1 历史数值和列名，未重算、未修改历史 JSON。** 上述真值修正没有解决 §2.5 的 RMSD 定义、φ/ψ 配对、clash 掩码和只计首样本几何问题。下表不能当作当前 v2 成绩或完整 T1 验收。

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

**读法（2026-11-01 审计纠正）：**

- **CA 列无损**：`backbone_samples[...,1,:] == ca_samples` 由测试与导出断言双重保证。
- 模型主链的部分**残基内部几何由 OpenFold 理想参数固定**，故旧表中键长/键角 validity = 1.000 与真实 MD 热涨落不同。这只能说明摆放约束，不能由此推出原文所称“模型预测系综平均几何”，也不能证明相邻残基连续。
- **历史几何短板保留**：肽键 C–N 均值 2.73 Å、旧容差下 validity 0.029、ω trans 约 79% 是严重链连续性问题的线索；旧 suite 只计首样本，完整数值需 v2 对所有样本重新评估。
- **纠正原文成因**：原文称“Model 1 backbone 由 Cα 系综 + 理想几何拼出，需要真正的 torsion/backbone 头”，与代码及检查点不符。`StageBOracleDecoder.decode_structure` 调用的 `StructureDecoder` 已学习局部位置、quaternion 旋转和 torsion，再从预测的 R/t/torsion 经 OpenFold 输出 N/CA/C/O。`artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt`（step 6000）保存了这些头的权重；实际缺口是独立残基 frames 的跨残基连续约束，原 B1a 目标没有显式 C–N/键角/clash 连续性项。预测的 φ/ω 也不自动保证相邻 frames 的实际 φ/ω。
- **当前 LASP 接线边界**：`CAComponentBundle` 只输出 CA；`CADynamics.sample` 返回 CA 和 generated raw H。将 CA 送入 `lift_ca_to_backbone` 可产生可评分的 N/CA/C/O，但必须标为 **CA ideal-lift readout**，不是新训练的主链头。将当前 H 接旧 B1a 只能先作为**旧 learned readout 的兼容性诊断**：校准后 encoder 已变化，latent 维度相同不能保证兼容，旧 B1a 输出 CA 也未必等于当前 CA。不能仅替换旧输出的 CA 列来拼接主链。
- **lifter 的额外限制**：它不接收 chain/gap/mask，需由新调用方逐有效连续链段调用；端点/共线 triplet 使用 identity frame fallback，存在旋转等变性例外。默认 torsion 与理想残基内部几何均不保证肽键闭合，应报告端点/退化影响。

最小可训练路线是先冻结当前 encoder、动力学和 CA readout，用真实 MD N/CA/C/O 监督新的 orientation/O-torsion readout，位置锚定当前 CA，并加入实际跨残基连续性约束；旧 B1a 可作为待检验初始化。先以真实 future H 做**oracle 校准诊断**，再用 generated H 训练/评估读出迁移；未来 H 不得进入预测输入。固定 CA 若本身不支持可实现的主链几何，还需联合约束 CA，不能据 oracle 重建改善宣称长程预测改善。

### 4.3 产物
- `results/backbone_suite_model1_1000dom.json`（+ `artifacts/` 同名副本）
- `artifacts/pdbs_1000dom_backbone_v1/<window>/{ground_truth,persistence,ours_stageA}.pdb`（真实 N/CA/C/O，PyMOL 可开，无 NaN/塌缩）。
  实际窗口：`1nhwC00_320_R0_anchor7/`（L=60）、`1ka1A01_320_R0_anchor7/`（L=128）；索引见 `index.json`。

---

## 5. TASK 2 — BioEmu 指标 + MSM 状态预测

> **静态/平衡系综指标（§5.1）与时间动力学指标（§5.2）分成两张表**（指标定义见 §2.6 / §2.7）。

### 5.1 Family A — 静态/平衡系综指标（BioEmu 口径）
**设置：** mdCATH-1000dom val，40 蛋白 × 20 窗口 = 800 窗口，n_samples=8，k_states=5，ε=2.0 Å，internal-distance 特征，参考态来自真实轨迹。

| 指标 | 值 | 说明 |
|---|---|---|
| **Reference-state coverage ↑** | **0.540** | 每个参考态被 ≥0.1% 采样命中的宏平均 |
| **Free-energy MAE ↓** | **0.779 kcal/mol** | F_k = −k_B T ln p_k（T=348 K） |
| Population TV ↓ | 0.707 | 总变差距离 |
| Population JSD ↓ | 0.379 | Jensen–Shannon 散度 |
| n_proteins | 40 | |

**对照 BioEmu 原文**：内部 MD 平衡基准 free-energy MAE ≈ 0.91 kcal/mol。**我们的 0.779 与之同量级**（但蛋白族/温度/状态划分不同，非严格可比）。

### 5.2 Family B — MSM 状态预测（BioEmu **不做**的部分）
**MSM 拟合（关键方法修正）：** 在**真实时间序轨迹**上 TICA(dim=2, lag=1) → KMeans(50 micro) → `MaximumLikelihoodMSM(reversible=True)` → PCCA → 5 macrostates。
**健康性检查**：true MSM timescales = **[150.2, 85.3, 70.7, 41.2, 27.6]**（lag 单位，单调递减、无 inf）——证明拟合在**真时间序**上（此前在 shuffle 的 anchor 窗口上拟合会退化到 `[inf, 0]`，已修正）。

| horizon | micro acc | persist acc | **macro acc** | macro persist | true transition rate |
|---|---|---|---|---|---|
| 1 ns | 0.7625 | 0.8188 | **0.9313** | 0.9625 | 0.038 |
| 4 ns | 0.7125 | 0.7688 | **0.9000** | 0.9438 | 0.056 |
| 16 ns | 0.6500 | 0.7375 | **0.9000** | 0.9500 | 0.050 |
| 64 ns | 0.6313 | 0.6313 | **0.8438** | 0.8688 | 0.131 |
| 192 ns | 0.5938 | 0.6063 | **0.8375** | 0.8813 | 0.119 |

### 5.3 ★ 用户点名指标 — 能否预测“状态转变 / 不变”（二分类）
把每个窗口标成 `true_change = (s(t+τ) ≠ s(t))`（MD 是否真跳态），`pred_change = (ŝ(t+τ) ≠ s(t))`（模型是否预测跳态）。

| horizon | binary acc | persist acc | 转变 precision | 转变 recall | **转变 F1** | true rate | pred rate |
|---|---|---|---|---|---|---|---|
| 1 ns | 0.931 | 0.963 | 0.222 | 0.333 | **0.267** | 0.038 | 0.056 |
| 4 ns | 0.900 | 0.944 | 0.182 | 0.222 | **0.200** | 0.056 | 0.069 |
| 16 ns | 0.900 | 0.950 | 0.167 | 0.250 | **0.200** | 0.050 | 0.075 |
| 64 ns | 0.844 | 0.869 | 0.300 | 0.143 | **0.194** | 0.131 | 0.062 |
| 192 ns | 0.838 | 0.881 | 0.182 | 0.105 | **0.133** | 0.119 | 0.069 |

**读法（诚实）：**
- 整体**二分类 accuracy 高（0.84–0.93）**，但这是**被“不转变”占多数**抬高的结果——**persistence（永远预测不转变）的 accuracy 反而更高**。
- **真正预测转变的能力弱**：转变类 **F1 仅 0.13–0.27**。**这正是“主态占比虚高”问题——所以同时报了 macro-F1、balanced accuracy 与转变类 F1。**
- 结论：**在 1 ns 帧间隔、该 MSM 划分下，模型尚不能可靠预测“何时跳态”**；但能把“继续处于同一亚稳态”预测得较准。

---

## 6. TASK 3 — 温度泛化性

### 6.1 温度输入考察（predictor/rollout 是否有温度输入？）
| 模型 | 温度入口 | 是否第一类输入 |
|---|---|---|
| **Model 1** | Stage A `time_embed(lag_ps, temperature_k)`；ensemble head `cond=[log(horizon_ps)/log(1e4), (T−300)/150]` | 是（但**当代 ckpt 的 head `cond_dims=0`，温度惰性**→ 已**重训** `cond_dims=2` 头） |
| **Model 2** | GLE `temperature_proj` / history posterior `_cond` / bath `bath(seq_feature, temperature_k)`；`rollout(..., temperature_k)` | 是（架构支持） |

**5 个温度**：`{320, 348, 379, 413, 450} K`。

### 6.2 泛化协议（矩阵）
- **行**：每个温度 T* ∈ 5 个温度（450 = 训练时**从未见过**的温度）。
- **列**：{seen-protein, unseen-protein} × {correct-T, wrong-T, no-T}。
- **指标**：ensemble RMSF L1（↓）、RMSF-r（↑）、skill vs persistence、`temperature_monotone`。

### 6.3 Model 1 结果（temp-conditioned head，`cond_dim=2`，每格 12 窗口，n_samples=8，horizons 1/2/4/8）

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
1. **模型确实在用温度**：**喂错温度普遍显著掉点**（如 T320 从 0.32 → 0.96 RMSF；T413 从 0.65 → 0.85；skill 从 +0.57 掉到 +0.45 甚至 −0.29）。这是 Task 3 最有信息量的对照。
2. **no-T 与 correct-T 接近**（多数格 diff 很小），说明模型可用 anchor 上下文**隐式**恢复一部分温度信息；但**在高压/高温端 correct-T 明显更优**。
3. **温度响应曲线单调**（`temperature_monotone=True`）：predicted RMSF 随 T 上升（RMSF-r 在 T450 掉到 0.34，说明**高温下形状预测更难**）。
4. **未见温度 T=450 是最大 gap**：RMSF-r 仅 0.34（seen T 上 ~0.58–0.62），skill 仅 0.46。**seen-protein 优于 unseen-protein**（外推越远越难）。
5. **内插 vs 外推**：T=348/379（内插）几乎不受 wrong-T 影响；T=320/413/450（边缘/外推）对 wrong-T 高度敏感。

### 6.4 Model 1 对照：无温度头（`cond_dim=0`）
同协议跑原 `stage_ensemble_1000dom` 头（温度惰性）：正确/错误/无温度三列**完全相同**（温度槽为空），RMSF 随 T 单调上升但不随 T* 改变行为（`artifacts/eval_temperature_generalization_cond0.json`）。**证明：只有 `cond_dim=2` 的头才会对 wrong-T 有反应。**

### 6.5 Model 2（from-none GLE v4g）温度矩阵（每格 12 蛋白，n_rollout_reps=3，horizons 1/4/16）

| 温度 | RMSF@16ns correct | wrong-T | no-T | skill_c |
|---|---|---|---|---|
| 320 | 4.561 | 4.561 | 4.562 | 0.207 |
| 348 | 4.982 | 4.983 | 4.982 | 0.293 |
| 379 | 5.120 | 5.136 | 5.128 | 0.318 |
| 413 | 5.804 | 5.833 | 5.818 | 0.453 |
| **450** | **6.908** | 6.944 | 6.933 | 0.506 |

**读法：** Model 2 的 **correct ≈ wrong ≈ none**（差异 < 1%），即**温度输入在 v4g 这条 from-none 路径上几乎惰性**——架构支持温度（`temperature_proj`），但训练后该通路贡献极小。（v4g 训练用 `train.jsonl` 含全部 5 个温度。）
- **温度响应单调**（RMSF 4.56→6.91 随 T 上升），但这是**隐含**的（可能来自 anchor 上下文），**非**来自显式温度输入。
- **结论**：要让 Model 2 真正温度敏感，需在温度留出/温度扰动训练下重训（`run_scripts/run_fromnone_temp_v2.sh`）。

### 6.6 温度留出（LOTO）训练
- **Manifests**：`artifacts/mdcath_manifest_1000dom_temp/{train_not450(4T,15999),val_not450(4T,2000),val_t450(500)}.jsonl`（T=450 留出）；新增 `train_t450.jsonl`（seen-protein × T=450，用于补全矩阵 seen 格）。
- **Model 1 temp-cond 头**：`run_scripts/run_ensemble_temp_1000dom.sh` → `artifacts/stage_ensemble_1000dom_temp_cond`（`cond_dims=2`，train/val 均为温度切分）。
- **Model 2 温度切分重训**：`run_scripts/run_fromnone_temp_v2.sh` → `artifacts/fromnone/stage_cb1b_temp_v2`（从 anchor 恢复、LR 1e-4、encoder-lr-scale 2.0，**训练集不含 T=450**）。该 run 早期曾因非有限梯度崩溃，已用成熟 v4g 配方（resume anchor）修复，现稳定推进至满 70000 步。

#### 6.6.1 温度留出外推代价（Model 2：含 T450 训练 vs 不含 T450 训练）
两个 ckpt 都在**未见蛋白**的 `val_t450` 上评测（correct-T，horizon=16 ns，n_rollout_reps=3）。

| 温度 | v4g（训练见过 T=450）skill | temp_v2（训练不含 T=450，70k）skill | Δskill |
|---|---|---|---|
| 320 | 0.207 | 0.197 | −0.010 |
| 348 | 0.293 | 0.250 | −0.043 |
| 379 | 0.318 | 0.263 | −0.054 |
| 413 | 0.453 | 0.435 | −0.018 |
| **450（留出 T）** | 0.506 | **0.479** | **−0.026** |

**读法（诚实，满训练步数口径）：**
- **温度留出带来温和但一致的精度损失**（Δskill −0.01 ~ −0.05），**不是灾难性退化**：即便训练**完全没见过 T=450**，模型在该温度上 skill 仍达 **0.479**（未留出的 v4g 为 0.506）。
- 损失在**中间温度（348/379）**略大（−0.04~−0.05），在**留出温度 450** 反而较小（−0.026）——说明 450 的高温行为主要受 anchor 上下文与 GLE 的隐含温度响应支配（§6.5 已证温度输入近乎惰性），显式温度切分影响有限。
- 两个 ckpt 的 **RMSF 都随温度单调上升**（响应曲线物理正确），泛化差异体现在**幅度/形状精度**而非**响应方向**。
- 综合 §6.5 + §6.6.1：**Model 2 的温度“泛化”目前主要靠 anchor 隐式携带，而非显式温度输入**；要让温度成为可控轴，需在 encoder/decoder 端显式注入并在温度扰动下训练。

### 6.7 验收对照
| 验收项 | 状态 |
|---|---|
| 5 × {seen,unseen protein} × {correct,wrong,none} 矩阵存在 | ✅（Model 1 full；Model 2 单列蛋白集） |
| 至少一个 ckpt **真的用了**温度（wrong-T 改变输出） | ✅（Model 1 cond_dim=2；wrong-T 显著掉点） |
| 明确说明内插 vs 外推 | ✅（T450 最大 gap，边缘温度敏感；§6.6.1 给出外推 Δskill） |
| 每个留出 T 可证不在其 train manifest 内 | ✅（`protocol.validate_manifest_split` + 温度计数核对：train_not450 无 450） |

---

## 7. 全部表汇总（均附 n + 采样预算）

> 每行的**文件**是唯一权威来源；**命令**列指向 §9 里能重跑它的那一节。
> 「n / 预算」一列的数字都已在对应 JSON 里核实过（`n_windows` / `n_samples` / `n_batches`）。

| # | 表 | 指标家族 | 文件（权威来源） | n / 预算 | 复现命令 |
|---|---|---|---|---|---|
| ① | 模型 1 系综（主表） | RMSF/W2/Jaccard | `artifacts/eval_model1_1000dom/suite_v2.json` | 50 批 × 4 = **200 窗口** / n_samples=8 | §9.1 |
| ② | 模型 1 轨迹 rollout（Kabsch） | 位置精度 | `artifacts/eval_model1_1000dom/ours_rollout_100f_v2.json` | 32 批 × 4 = **128 窗口** / 100 帧 | §9.2 |
| ③ | 模型 2 系综 | RMSF/W2/Jaccard | `results/v4e_ensemble_v2.json`（**不是** `v4e_ensemble_eval.json`） | **200 窗口** / n_samples=12 | §9.3 |
| ④ | MDGen zero-shot / 短训 | 同上 + Kabsch | `artifacts/eval_mdgen/mdgen_zeroshot_atlas_mdcath{,_seed7}.json`、`mdgen_mdcath_short_s{2000,6000}.json` | 32 窗口 / n_samples=8 / 100 帧 | §9.4 |
| ⑤ | bioemu2 zero-shot（单次 / 3 次 mean±std） | 同上 | `results/bioemu2_zeroshot.json` / `results/bioemu2_reps/summary.json` | 200 窗口 / n_samples=12；3 seeds | §9.5 |
| ⑥ | Plan A 系综配对（Model 1 vs H+S） | RMSF/W2/Jaccard | `results/plan_a_hs_vs_model1.json` | 40 配对窗口 / M=8 | §9.6 |
| ⑦ | Plan A 轨迹配对（walk） | order-aware | `results/walk_plan_a_contig.json` | 40 窗口 / M=6 | §9.6 |
| ⑧ | **Lasp-final vs Model 1** | RMSF/W2/Jaccard（同窗口配对） | `results/lasp_final_vs_model1_bioemu2.json` | 200 窗口 / n_samples=12 | §9.6 |
| ⑨ | **主链（bone，T1）** | N/CA/C/O RMSD + validity | `results/backbone_suite_model1_1000dom.json` | 30 批 × 2 = 60 窗口 / n_samples=8 | §9.7 |
| ⑩ | **静态系综（BioEmu 口径，T2A）** | coverage/free-energy/population | `artifacts/eval_states_msm_1000dom.json` → `static` | 40 蛋白 / 800 窗口 / n_samples=8 | §9.8 |
| ⑪ | **动力学（MSM/lagged + 转变检测，T2B）** | state pred / transition | 同上 → `dynamics` | 40 真实轨迹拟合 MSM；160 eval/格 | §9.8 |
| ⑫ | **温度泛化（T3）** | 5×2×3 矩阵 | `artifacts/eval_temperature_generalization_cond_full.json`（M1 cond2）、`..._cond0.json`（M1 对照）、`results/temperature_generalization_model2_v4g.json` + `..._tempsplit.json`（M2） | M1: 12 窗口/格；M2: 12 蛋白 × 3 reps/格 | §9.9 / §9.10 |

---

## 8. 产物与 PDB（可视化检查）

> 脚本：`scripts/export_eval_trajectories_pdb.py`（原精确坐标解码代码未改动）。每个窗口 6 条多 MODEL PDB（16 帧），可直接在 PyMOL/VMD 叠看。

- **1000dom 版（推荐）**：`artifacts/pdbs_1000dom_20261020/`，窗口 `1nhwC00`(L=60)、`1ka1A01`(L=219)、`1kafD00`(L=104)，T=320 K、anchor frame 7。
- **主链版（T1）**：`artifacts/pdbs_1000dom_backbone_v1/<window>/{ground_truth,persistence,ours_stageA}.pdb`（真实 N/CA/C/O）。

### 8.1 ⚠️ 关键澄清：系综 PDB vs rollout PDB（“SOTA 却 RMSD 很差”的真相）

| 目录 | 任务 | 生成代码 | 典型 CA-RMSD |
|------|------|----------|--------------|
| **`artifacts/pdbs_1000dom_ensemble_20261021/`** | **单步系综**（`sample_future_ca` 调 1 次，无回馈） | `eval_ensemble_metrics_suite.py` | **与 §3 报告一致（SOTA）** |
| `artifacts/pdbs_1000dom_aligned_20261021/` | 自回归 **rollout**（把预测喂回去） | `eval_ours_trajectory.py` | ours ≈ 8.4 vs persistence 3.9 |

- §3 全部 SOTA 指标**只**来自**单步系综**任务 —— 它就是报告里的那张表。
- rollout 目录里的“差 RMSD”是**另一个更难的任务**（误差 ~2 Å/帧 累积），**不等于**报告口径。
- **结论：复现报告不需要 rollout。**

---

## 9. 复现命令

> 全部命令在 `cd /liuyutian/protein_jepa_frame` 下执行。**每节标题后的 `→ 产物` 是它写出的文件**，
> 与 §7 的表编号一一对应。所有 ckpt 路径均为仓库内相对路径。

```bash
export TMPDIR=/liuyutian/protein_jepa_frame/.tmpwork
export PYTHONPATH=/liuyutian/protein_jepa_frame
```

### 9.1 主表：Model 1 系综（§3.1 / §7-①）→ `artifacts/eval_model1_1000dom/suite_v2.json`

> `--batches 50 --batch-size 4` → 200 窗口（`suite_v2.json` 里 `n_batches=50, n_samples_total=200`）。
> 不传 `--out` 只打印不落盘；**必须显式给 `--out`**。

```bash
python -m scripts.eval_ensemble_metrics_suite \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --b1a-ckpt artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
  --ensemble-ckpt artifacts/stage_ensemble_1000dom/checkpoint_best.pt \
  --manifest artifacts/mdcath_manifest_1000dom/val.jsonl \
  --device cuda --batches 50 --batch-size 4 --n-samples 8 \
  --horizons 1 2 4 8 --pca-k 3 \
  --out artifacts/eval_model1_1000dom/suite_v2.json
```

### 9.2 轨迹 rollout 的 Kabsch / 位置精度（§3.1 末、§8.1）→ `artifacts/eval_model1_1000dom/ours_rollout_100f_v2.json`

> 这是 `8.39 Å` 这唯一一个位置精度数字的来源（100 帧自回归 rollout，32 批 = 128 样本）。
> **与 9.1 的「单步系综」是两个不同任务**，不要混引（见 §8.1）。

```bash
python -m scripts.eval_ours_trajectory \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --b1a-ckpt artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
  --ensemble-ckpt artifacts/stage_ensemble_1000dom/checkpoint_best.pt \
  --manifest artifacts/mdcath_manifest_1000dom/val.jsonl \
  --device cuda --batches 32 --batch-size 4 --n-samples 8 --num-frames 100 \
  --out artifacts/eval_model1_1000dom/ours_rollout_100f_v2.json
```

### 9.3 Model 2 系综（§3.2 / §7-②）→ `results/v4e_ensemble_v2.json`

> ⚠️ **必须是 `v4e_ensemble_v2.json`（200 窗口 / n_samples=12）**。
> 同目录的 `v4e_ensemble_eval.json` 是另一条 400 窗口 / n_samples=32 的 run，**不是本表来源**。

```bash
python -m evaluation.run_eval_fromnone_ensemble \
  --ckpt artifacts/fromnone/stage_cb1b_v4e/checkpoint_latest.pt \
  --manifest artifacts/mdcath_manifest_1000dom/val.jsonl \
  --device cuda --max-windows 200 --n-samples 12 \
  --horizons-ns 1 4 16 64 \
  --out results/v4e_ensemble_v2.json
```

### 9.4 MDGen zero-shot / 短训 baseline（§3.3 / §7-③）

```bash
# zero-shot（官方 atlas 权重）
python -m scripts.eval_mdgen_baseline \
  --mdgen-ckpt /liuyutian/mdgen_run/hf_mdgen/atlas.ckpt \
  --manifest artifacts/mdcath_manifest/val.jsonl \
  --device cuda --n-windows 32 --n-samples 8 --num-frames 100 --horizon 1 \
  --out artifacts/eval_mdgen/mdgen_zeroshot_atlas_mdcath.json
# 同 seed=7 复核
python -m scripts.eval_mdgen_baseline \
  --mdgen-ckpt /liuyutian/mdgen_run/hf_mdgen/atlas.ckpt \
  --manifest artifacts/mdcath_manifest/val.jsonl \
  --device cuda --n-windows 32 --n-samples 8 --num-frames 100 --horizon 1 --seed 7 \
  --out artifacts/eval_mdgen/mdgen_zeroshot_atlas_mdcath_seed7.json
# mdCATH 短训两个 ckpt（e1/s2000、e5/s6000）
for STEP in 2000 6000; do
  python -m scripts.eval_mdgen_baseline \
    --mdgen-ckpt "/liuyutian/mdgen_src/mdgen-master/workdir/mdcath_short/epoch=$( [ $STEP = 2000 ] && echo 1 || echo 5 )-step=${STEP}.ckpt" \
    --manifest artifacts/mdcath_manifest/val.jsonl \
    --device cuda --n-windows 32 --n-samples 8 --num-frames 100 --horizon 1 \
    --out "artifacts/eval_mdgen/mdgen_mdcath_short_s${STEP}.json"
done
```

### 9.5 bioemu2 zero-shot（§3.4 / §7-④）

```bash
bash run_scripts/run_bioemu2_zeroshot.sh        # -> results/bioemu2_zeroshot.json（单次，200 窗口 / n_samples=12）
bash run_scripts/run_bioemu2_zeroshot_rep.sh    # -> results/bioemu2_reps/seed{0,1,2}.json（3 次重复）
python -m evaluation.baselines.summarize_reps \
  --glob "results/bioemu2_reps/seed*.json" --out results/bioemu2_reps/summary.json
```

### 9.6 Lasp / H+S（GLE）系综 + 轨迹（§3.6 / §3.7 / §7-⑤⑥）

```bash
# 系综配对（Model 1 vs H+S；clean paired，40 窗口）
bash run_scripts/run_paired_eval.sh             # -> results/plan_a_hs_vs_model1.json
                                                #    + results/plan_a_dense_vs_model1.json
# order-aware 轨迹配对（§3.7）
python -m scripts.paired_walk_metrics --mode hs \
  --kernel-ckpt artifacts/plan_a_hs_1000dom/checkpoint_best.pt \
  --ensemble-ckpt artifacts/stage_ensemble_1000dom/checkpoint_best.pt \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --b1a-ckpt artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
  --manifest artifacts/mdcath_manifest_1000dom/val.jsonl \
  --protocol contiguous --horizons 8 --n-samples 6 \
  --out results/walk_plan_a_contig.json
# Lasp-final vs Model 1 vs bioemu2 三方同口径（报告 §3.7b 的 Lasp 列）
CKPT=artifacts/lasp_mh_1000dom/checkpoint_best.pt \
OUT=results/lasp_final_vs_model1_bioemu2.json \
  bash run_scripts/run_lasp_final_eval.sh
```

### 9.7 T1 主链（§4 / §7-⑨）→ `results/backbone_suite_model1_1000dom.json`

> 历史 v1 复现命令，保留用于追溯。它不会调用 v2，不能用于新版 T1 验收；新实验须使用新输出目录，勿覆盖下述冻结结果。

> `--batches 30 --batch-size 2` → 60 窗口（与 §4.2 表头一致）。

```bash
python -m scripts.eval_backbone_suite \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --b1a-ckpt artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
  --ensemble-ckpt artifacts/stage_ensemble_1000dom/checkpoint_best.pt \
  --manifest artifacts/mdcath_manifest_1000dom/val.jsonl \
  --device cuda --batches 30 --batch-size 2 --n-samples 8 --horizons 1 \
  --out results/backbone_suite_model1_1000dom.json
# 主链 PDB（真实 N/CA/C/O）
python scripts/export_eval_trajectories_pdb.py --atoms backbone --ensemble \
  --which ground_truth persistence ours_stageA \
  --outdir artifacts/pdbs_1000dom_backbone_v1 --n-windows 2
```

### 9.8 T2 静态系综 + MSM 状态预测（§5 / §7-⑦⑧）→ `artifacts/eval_states_msm_1000dom.json`

> `--max-proteins 40 --horizons 1 4 16 64 192 --n-samples 8` → `n_windows=800`（与产物一致）。

```bash
python -m scripts.eval_states_and_msm \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --b1a-ckpt artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
  --ensemble-ckpt artifacts/stage_ensemble_1000dom/checkpoint_best.pt \
  --manifest artifacts/mdcath_manifest_1000dom/val.jsonl \
  --device cuda --max-proteins 40 --horizons 1 4 16 64 192 --n-samples 8 \
  --out artifacts/eval_states_msm_1000dom.json
```

### 9.9 T3 温度泛化：Model 1（§6.3 / §7-⑩）→ `artifacts/eval_temperature_generalization_cond_full.json`

> ⚠️ **必须加 `--max-windows-per-temp 12`**（脚本默认 40）。产物里每格 `n_windows=12`，
> 用默认值会跑出 40 窗口/格，**复现不出 §6.3 的表**。

```bash
python -m scripts.eval_temperature_generalization \
  --ensemble-ckpt artifacts/stage_ensemble_1000dom_temp_cond/checkpoint_best.pt \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --b1a-ckpt artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
  --manifest artifacts/mdcath_manifest_1000dom_temp/val_not450.jsonl \
  --manifest-unseen-temp artifacts/mdcath_manifest_1000dom_temp/val_t450.jsonl \
  --manifest-seen artifacts/mdcath_manifest_1000dom_temp/train_not450.jsonl \
  --manifest-seen-t450 artifacts/mdcath_manifest_1000dom_temp/train_t450.jsonl \
  --device cuda --max-windows-per-temp 12 --n-samples 8 --horizons 1 2 4 8 \
  --out artifacts/eval_temperature_generalization_cond_full.json
# cond_dim=0 对照（§6.4）
python -m scripts.eval_temperature_generalization \
  --ensemble-ckpt artifacts/stage_ensemble_1000dom/checkpoint_best.pt \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --b1a-ckpt artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
  --manifest artifacts/mdcath_manifest_1000dom_temp/val_not450.jsonl \
  --manifest-unseen-temp artifacts/mdcath_manifest_1000dom_temp/val_t450.jsonl \
  --manifest-seen artifacts/mdcath_manifest_1000dom_temp/train_not450.jsonl \
  --manifest-seen-t450 artifacts/mdcath_manifest_1000dom_temp/train_t450.jsonl \
  --device cuda --max-windows-per-temp 12 --n-samples 8 --horizons 1 2 4 8 \
  --out artifacts/eval_temperature_generalization_cond0.json
```

### 9.10 T3 温度泛化：Model 2（§6.5 / §6.6.1 / §7-⑩）→ `results/temperature_generalization_model2_v4g.json`

> ⚠️ **必须加 `--n-rollout-reps 3`**（脚本默认 4）与 `--max-proteins 12`。
> 产物里 `n_rollout_reps=3`、每格 `n_proteins=12`。

```bash
# v4g（训练见过 T=450）
python -m scripts.eval_temperature_generalization_model2 \
  --ckpt artifacts/fromnone/stage_cb1b_v4g/checkpoint_latest.pt \
  --manifest-seen-temp artifacts/mdcath_manifest_1000dom_temp/val_not450.jsonl \
  --manifest-unseen-temp artifacts/mdcath_manifest_1000dom_temp/val_t450.jsonl \
  --horizons 1 4 16 --max-proteins 12 --n-rollout-reps 3 --modes correct wrong none \
  --out results/temperature_generalization_model2_v4g.json
# 温度留出 ckpt（训练不含 T=450；§6.6.1 的 Δskill 列）
python -m scripts.eval_temperature_generalization_model2 \
  --ckpt artifacts/fromnone/stage_cb1b_temp_v2/checkpoint_latest.pt \
  --manifest-seen-temp artifacts/mdcath_manifest_1000dom_temp/val_not450.jsonl \
  --manifest-unseen-temp artifacts/mdcath_manifest_1000dom_temp/val_t450.jsonl \
  --horizons 1 4 16 --max-proteins 12 --n-rollout-reps 3 --modes correct wrong none \
  --out results/temperature_generalization_model2_tempsplit.json
```

### 9.12 标准评测（每次评测都必须带：分温度 + T3 机制矩阵 + T2A）

> 2026-10-24 起，**任何** LASP CA dynamics checkpoint 的评测都走同一条命令，
> 使 §2.6（T2A，BioEmu 口径）与 §2.9（T3 温度泛化）**不再可选**。

```bash
bash run_scripts/run_lasp_standard_eval.sh <tag> <ckpt_best> [<ckpt_best> ...]
# 例：
bash run_scripts/run_lasp_standard_eval.sh round2 \
  artifacts/lasp_full_v1/dynamics_seed42/checkpoint_best.pt \
  artifacts/lasp_full_v1/dynamics_seed43/checkpoint_best.pt \
  artifacts/lasp_full_v1/dynamics_seed44/checkpoint_best.pt
```

每个 checkpoint 产出两类结果（各自独立进程，可指定 GPU）：

| 产物 | 内容 |
|---|---|
| `<name>.ensemble.json` | 200 窗口 × 12 样本的统一 BioEmu2 协议系综指标（7 项）+ 几何；位置指标由 suite 提供 |
| `<name>.suite.json` | 分温度扫描、**correct/wrong/none** 机制矩阵、**T2A**（coverage / free-energy MAE / population MAE·TV·JSD） |

```bash
# 等价的底层调用（suite 单个 checkpoint 时直接用它）
python -m scripts.eval_lasp_ca_suite --checkpoint CK --out OUT.json --t2a \
  --modes correct wrong none --wrong-temp 320 \
  --proteins-per-temp 12 --anchors-per-trajectory 2 --n-samples 12 \
  --horizons 1 4 16 64 --max-residues 96
python -m scripts.report_lasp_ca_suite OUT_DIR/*.suite.json --out OUT_DIR/suite_report.json
```

**两条硬性防退化约束（写进代码 + 回归测试）：**

1. **wrong 温度不得等于观测温度**。默认 `--wrong-temp 320` 恰好等于 mdCATH 最低温，
   若不设防，T\*=320 的 wrong 臂会**逐位复现** correct 臂，该格机制检验无效。
   现改为自动替换为最远 mdCATH 温度（450），并在每个窗口记录
   `temperature_used_k`。已在飞行中的退化格用
   `python -m scripts.merge_lasp_ca_suite` 以补跑单元格替换（aggregate /
   by_horizon 由合并后的逐窗口行**重算**，不留旧平均）。
2. **T2A 用预裁剪数组 + `mask=None`**。全长 mask 与已压缩数组长度不匹配，
   现在直接抛错，防止错误的特征选择。

**范围与噪声配对**：当前扫描的五个温度均在训练中出现，不能据此声称未见温度外推；
完整 seen/unseen-protein 矩阵仍需独立实验。Round 1 使用旧版含 mode 的随机种子，
三臂噪声不同；Round 2 及后续 `window_seed()` 只由窗口身份（protein/trajectory/anchor/horizon）决定，
**不含 mode**，因此 correct/wrong/none 三臂抽取**同一噪声**、只差喂给模型的温度
（温度同时进入 drift 与噪声尺度，共享噪声是真实的方差缩减）。

**bioemu2 没有温度输入**：安装包源码中 `temperature` 出现 **0** 次，
`propagate` / `propagate_with_embeddings` 均不接受温度参数。故
correct/wrong/none 消融**只对本文模型可定义**；bioemu2 只做分温度 + T2A：

```bash
bash run_scripts/run_bioemu2_t2a_samples.sh    # 5 温度并行，每温度 160 窗口，--save-samples
python -m scripts.t2a_from_samples --samples DIR --manifest ... --out bioemu2_t2a.json
```

bioemu2 侧取每温度**前 8 个蛋白**（已验证是 12 蛋白集合的子集）× 5 副本 × 1 anchor × 4 horizons
= 每温度 160 窗口。LASP 必须另跑同样的 8 蛋白 / 1 anchor 协议并核对完整窗口身份、
残基 mask 与参考构造，才可直接比较；12 蛋白 / 2 anchor 汇总不能直接替代。
`oracle` 参考半分诊断使用不同的聚类参考，不是严格上限，不能据其归一化模型分数。

### 9.11 训练（复现 ckpt 用；不是评测）

```bash
bash run_scripts/run_ensemble_1000dom.sh       # Model 1 的 Gaussian 系综头
bash run_scripts/run_ensemble_temp_1000dom.sh  # cond_dim=2 温度条件头
bash run_scripts/run_fromnone_temp_v2.sh       # Model 2 温度留出重训
```

---

## 10. 局限与下一步（诚实）

1. **主链（T1，2026-11-01 纠正）**：原文“Model 1 主链为 Cα+理想几何拼出、尚需 frame + torsion 头”不成立：已训练 B1a 有这两类头，短板是跨残基连续性及其与当前 H 的兼容性。旧肽键 validity 0.029、ω_trans 0.79 保留为 v1 历史线索；新 v2 修复基础指标后仍需真实 MD、逐样本完整对照。CA ideal-lift 只能作为标注清楚的读出诊断；无正式统计 Ramachandran 与全主链原子 clash，**T1 尚未完成**。
2. **MSM 状态预测（T2）**：1 ns 帧间隔下**转变检测 F1 低**（0.13–0.27）；下一步可缩短有效 lag / 用更多 anchor / 更长轨迹提升转变样本量。
3. **位置精度（共同短板）**：模型 1 rollout Kabsch **8.39 Å** 差于 persistence **3.87 Å**；模型 2 `root_mean_w2` 短时标（1 ns）失败（位移过小）。
   ⚠️ **"分布形态已稳赢"只在历史口径（对 persistence / MDGen）成立**——**同 harness 对 bioemu2 我们实际上输 19:4**
   （主报告 §13.2）。真实状况是：**锚定位置精度我们领先，分布形态 bioemu2 领先**，不能笼统说"已稳赢"。
4. **温度泛化（T3）**：Model 1 temp-cond head 是**轻头重训**（冻结栈），可进一步在 **encoder 端**注入温度；Model 2 温度留出训练显示**外推代价温和（Δskill ≈ −0.03）但温度输入本身近乎惰性**，需在 encoder/decoder 端**显式注入**温度并在温度扰动下训练以激活 `temperature_proj` 通路，再用 §6.3 矩阵复测。
5. **覆盖度**：单一 ε 会高估 coverage，报告已附 population TV/JSD + free-energy MAE 作为稳健补充。
6. **baseline 口径**：MDGen zero-shot 是 **OOD**（10 ps vs 1000 ps/帧）；bioemu2 是**系综采样**而非轨迹推进，短时标输 persistence 属任务口径差异，非模型缺陷。
7. **接触类 Jaccard 对 delta/近刚体基线退化**——引用时必须附 `n_true/n_pred`。
8. **协议一致性**：训练用 contiguous、部分评测用 paired 两个协议；只要双方同用评测口径就自洽，若要比 horizon 曲线需统一。
9. **本文件 §3 各表不可跨行比较**：来自不同 harness / 窗口集 / 时标（详见 §1.2 与 §3.5 警告）。**要同口径比较，用主报告 §13.8 的 T0–T8。**
10. **历史覆盖缺口与当前 T1 边界**：§3.8 当时只覆盖 7 项系综指标，原记“Lasp 未接入 T1/T2/T3”。后续 CA 标准评测已增加 §9.12 的 T2A/温度矩阵；这不补齐 T1。2026-11-01 已完成主链基础指标 v2 与200窗 CA→主链读出诊断（§12）：真实 CA 的 ideal lift 仍有约2.61 Å的跨残基 C–N 距离，不能宣布 learned backbone、完整立体化学或全指标目标达成。

---

## 11. 与主报告的对应（不做重复）

| 内容 | 位置 |
|---|---|
| 模型/训练细节（Stage-A / B1a / ensemble / GLE 配方） | [`PROJECT_REPORT_20261021.md`](PROJECT_REPORT_20261021.md) §5/§7 |
| 复现性（bioemu2 三重非确定性修复） | [`PROJECT_REPORT_20261021.md`](PROJECT_REPORT_20261021.md) §3.5b |
| Plan A 代码路径 / 失败模式（dense） | [`PROJECT_REPORT_20261021.md`](PROJECT_REPORT_20261021.md) §10 |
| **五方法同一 harness 横评（结论 + T0–T8 大表）** | [`PROJECT_REPORT_20261021.md`](PROJECT_REPORT_20261021.md) **§13 / §13.8** |
| **温度留出 / 时间尺度 / rollout 的多尺度对比** | [`PROJECT_REPORT_20261021.md`](PROJECT_REPORT_20261021.md) §13.8（T3–T6） |
| **全部评测指标定义 + 结果（本文件）** | 本文 §1–§10 |
| **本文件 §9 的可执行复现命令（15 条，已逐条校验）** | 本文 §9.1–§9.11 |

## 12. 2026-11-01 新增诊断版本（不改历史表格）

[主链与有序路径报告](RESULTS_BACKBONE_AND_PATHS_20261101.md)记录新的实际结果、来源与限制；[四臂报告](RESULTS_HS_JOINT_20261101.md)记录固定500步的 H 读出保持消融。它们均为开发验证，不是独立确认。所有命令必须使用新输出路径，runner 拒绝覆盖已有结果/协议/快照。

### 12.1 主链读出 v2

```bash
PYTHONPATH=/tmp/lasp_stability_deps:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -m scripts.eval_ca_backbone_diagnostic \
  --checkpoint artifacts/hs_joint_20261101/spatial_probe_v1/spatial_exact.pt \
  --out results/hs_joint_20261101/backbone_v2/reproduction_new.json \
  --device cuda:0 --max-windows 200 --torch-threads 2
```

实际成功工件为 [`spatial_exact_allres200_v2.json`](results/hs_joint_20261101/backbone_v2/spatial_exact_allres200_v2.json)：50蛋白、200窗、每窗12预测样本、8种预测/oracle/reference读出。主 RMSD 在观察 anchor 坐标系按平方误差开根号计算，另存 Kabsch shape 诊断；所有样本参与几何统计。只有 oracle 分支使用未来 H，真实标签不进入预测。

数据集编号 `ACDEFGHIKLMNPQRSTVWYX` 与 OpenFold 化学编号 `ARNDCQEGHILKMFPSTWYVX` 不同；新 lift/化学指标显式转换，已训练 embedding 仍使用其原编号。旧 B1a 作为历史对照保留原 placement 语义。UNK 使用已有 GLY 主链模板，不删除观察有效的未知残基；类型相关 z 参考另报有效计数。详见[化学边界审计](docs/RESIDUE_TYPE_AUDIT_20261101.md)。这既不修复旧 checkpoint 的训练，也没有补齐正式 Ramachandran、全主链/全原子 clash 或与 BioEmu2 的新 T1 对照。

### 12.2 连续有序路径 v2

```bash
PYTHONPATH=/tmp/lasp_stability_deps:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -m scripts.eval_ca_ordered_paths \
  --checkpoint artifacts/hs_joint_20261101/spatial_probe_v1/spatial_exact.pt \
  --out results/hs_joint_20261101/ordered_paths/reproduction_standard200_new.json \
  --device cuda:0 --torch-threads 2

# 独立的192 ns面板；不混入standard200
PYTHONPATH=/tmp/lasp_stability_deps:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -m scripts.eval_ca_ordered_paths \
  --checkpoint artifacts/hs_joint_20261101/spatial_probe_v1/spatial_exact.pt \
  --out results/hs_joint_20261101/ordered_paths/reproduction_long192_new.json \
  --device cuda:0 --horizons 192 --torch-threads 2
```

默认50蛋白×4 horizons=200窗；192 ns使用独立50窗，均为每窗12条连续 H/S 路径。4窗 smoke 需显式 `--max-windows 4`。每个窗口只调用一次采样器，固定观察 anchor，保留一次采样的 S₀ 及连续状态；未来真值不用于重编码、重置或初始化。

`trajectory_metrics_v2.py` 的 joint U-energy score 对整个未来路径计算；不能与旧逐时刻 marginal energy 的均值混称。每条路径分别计算 ACF/MSD，成员边界不拼接。真实 MD 每窗只有一条，单条 reference 的 U-score 为 null。reverse/shuffle 仅是顺序诊断，会改变时间边际并可能改变终点；不能据它们证明 S 的增量价值。短轨迹 ACF 与内部距离 MSD 不直接给出平衡松弛时间或原子扩散常数。这里仍没有 MSM/T2B、配对温度泛化或 BioEmu2 动力学优越性的完成证据。

### 12.3 固定 CA 的 learned backbone 受控读出实验

[预注册](docs/BACKBONE_READOUT_PILOT_V2_20261101.md)固定500个不同训练蛋白、两组各500步和同一个零更新初始化。仅训练独立 N/CA/C/O 读出头；原始 H、CA 头及 H/S 动力学冻结。coordinate 组最小化实际 N/C/O 坐标误差，geometry 组另加显式 PSF 连接的四项 v3 几何损失。系数仅用前32个训练窗口的初始参数梯度计算；不使用验证指标选择权重或 checkpoint。

`backbone_topology_v2.py` 保留完整残基 UID 和 PSF 肽键边，恢复明确的 His 别名化学身份，同时逐位保留旧 encoder/embedding 的 aa、residue_index、chain_id。几何边不再由丢失插入码的整数编号推断；历史 v2 指标仍保留，并标明它与实际 PSF 图不一致的窗口。

已保存缓存为 `results/hs_joint_20261101/backbone_repair/readout_train500_v2.pt` 与 `readout_val200_v2.pt`。训练与验证蛋白无交集；验证缓存每窗只读取8个观察帧，先用既有采样器生成全部12个预测成员，再读取未来标签。oracle H/CA 为明确标注的未来信息诊断，不进入 forecast。完整窗口、来源与逐位一致性核验见[缓存复验](results/hs_joint_20261101/backbone_repair/readout_caches_v2_verification.json)。

```bash
PYTHONPATH=/tmp/lasp_stability_deps:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -m scripts.eval_backbone_readout_pilot \
  --cache results/hs_joint_20261101/backbone_repair/readout_val200_v2.pt \
  --init-checkpoint artifacts/hs_joint_20261101/backbone_repair_v2/initial.pt \
  --coordinate-checkpoint artifacts/hs_joint_20261101/backbone_repair_v2/coordinate/checkpoint_latest.pt \
  --geometry-checkpoint artifacts/hs_joint_20261101/backbone_repair_v2/geometry/checkpoint_latest.pt \
  --out results/hs_joint_20261101/backbone_repair_v2/reproduction_readout200_new.json \
  --device cuda:0 --torch-threads 2
```

评测包含14条路径：三个头各自的 oracle/forecast、初始化与geometry头的六个固定oracle CA的 H 替换对照，以及真实MD persistence/target。所有预测成员都评分；CA须与输入逐位相同，指标使用CPU float64。主要坐标指标是逐成员实际 N/C/O RMSD，另保留完整主链 RMSD；不能将N/C/O主指标与此前四原子RMSD直接比较。按蛋白配对bootstrap的95%区间为单seed开发集上的边际探索结果，无多重比较校正。v3宽松软约束不是完整T1、全原子clash或MolProbity标准；本实验也不构成新的BioEmu2比较。

补充[固定 CA 可闭合性诊断](results/hs_joint_20261101/backbone_repair_v2/fixed_ca_capacity.json)使用三角不等式和官方残基模板半径，为 C–N 太长侧的损失给出下界；单独登记后仅消费冻结缓存，不使用主评测结果。64 ns forecast 中满足该充分不可行条件的边比例为等蛋白平均0.4653%，其余边不能据零下界认证为可行。该诊断忽略角度、过近距离和碰撞约束，并使用预先固定的浮点误差裕量；它不是完整几何最优值、T1门槛或对允许移动CA的架构的限制。

### 12.4 无参数局部frame搬运与相同预算读出

当前simplified IPA只使用当前CA的两两距离，忽略输入R。[新协议](docs/BACKBONE_TRANSPORT_PILOT_V1_20261101.md)用显式PSF图≤2-hop的观察/当前CA做中心残基锚定proper Kabsch，以 `Q @ observed_R` 提供当前局部方向；退化时保留observed frame。它不移动CA、不增加可训练参数，也不使用target atoms/mask。训练与§12.3 coordinate臂共用初始化、500窗顺序、seed和优化器，仅替换frame seed；固定500步latest。

实际200窗、50蛋白、全部12 forecast成员的overall NCO RMSD从oracle1.661962→1.442264 Å、forecast4.632646→4.575603 Å；forecast C–N软损失从.600562→.259931 Å。四个horizon的NCO配对区间均支持改善，但forecast clash变化区间跨0，且NCO仍差于persistence3.488854 Å。所有150个预登记区间均为单seed、重复开发集的边际探索性结果；完整表格及训练集几何容量诊断见[报告§6](RESULTS_BACKBONE_AND_PATHS_20261101.md#6-局部-frame-搬运相同500步预算的新读出实验)。该实验未改变CA级BioEmu2指标，不能宣布完整T1或总体目标完成。

复现使用独立新评估入口，输出必须取新路径：

```bash
export PYTHONPATH=/tmp/lasp_stability_deps:.
python -m scripts.eval_backbone_transport_pilot \
  --checkpoint artifacts/hs_joint_20261101/backbone_transport_v1/coordinate/checkpoint_latest.pt \
  --out results/hs_joint_20261101/backbone_transport_v1/reproduction200_new.json \
  --device cuda:0
```

入口要求800个旧observed control的骨架哈希及数值严格复现，因此保持原GPU推理环境。首版评估因live tuple与历史JSON list的容器差异误拒绝，原失败记录保留；[v2修订](docs/BACKBONE_TRANSPORT_EVAL_SERIALIZATION_V2_20261101.md)仅统一JSON表示，不放宽数值容忍。[成功结果](results/hs_joint_20261101/backbone_transport_v1/evaluation200_v2.json)、[执行记录](results/hs_joint_20261101/backbone_transport_v1/evaluation_v2_run.execution.json)、[独立统计与来源核验](results/hs_joint_20261101/backbone_transport_v1/independent_statistics_verification.json)均已保存。全量回归756 passed、1 skipped；不会读取test、重跑BioEmu2或覆盖受保护历史结果。
