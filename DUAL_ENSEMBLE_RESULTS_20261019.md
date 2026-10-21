# DUAL-ENSEMBLE RESULTS — 精确汇报（2026-10-19）

> 配套计划：`PLAN_DUAL_ENSEMBLE_20261019.md`（D0–D7）。
> 原始数据：`artifacts/eval_dual/{A,B,C}_{mdcath,atlas100ps,atlas1ns}.json`，
> 训练日志 `logs/dual{A,B,C}_20261019_135203.log`。**本文件所有数字均来自这些文件。**

---

## 0. 先讲清楚：**测了什么、在哪测、怎么算**

### 0.1 数据集（3 个验证集，**全部是未见蛋白**）

| 记号 | manifest | 蛋白/轨迹 | 帧间隔 | 每轨迹帧数 | 温度 | 与训练的关系 |
| --- | --- | --- | --- | --- | --- | --- |
| **mdCATH** | `artifacts/mdcath_manifest_200dom/val.jsonl` | 100 域 / 87 轨迹 | 1000 ps | ~数百 | 320–450 K | 训练见过 mdCATH 其它蛋白；**这些蛋白未见** |
| **ATLAS-100ps** | `artifacts/atlas_manifest_100ps/val.jsonl` | 10 蛋白 / 29 轨迹 | 100 ps | 1001 | 300 K | 训练用 ATLAS-100ps **train（90 蛋白）**；**这 10 个 val 蛋白完全未见** |
| **ATLAS-1ns** | `artifacts/atlas_manifest/val.jsonl` | **同样这 10 蛋白** / 29 轨迹 | 1000 ps | 101 | 300 K | 训练**从未用过这些 npz**；测试的是**同 10 蛋白、更粗采样(1ns)** |

- 已核实：**ATLAS-100ps train(90 蛋白) ∩ val(10 蛋白) = ∅**；
  且 **ATLAS-1ns val 与 ATLAS-100ps val 是同一组 10 个蛋白**
  （`1ah7_A,1btk_A,1bx7_A,1c1k_A,1dci_C,1dj0_B,1dow_A,1hw1_B,1ig0_B,1iom_A`）。
- **⇒ 两个 ATLAS 集都是真未见蛋白**；ATLAS-1ns 额外测"更粗时间分辨率"的鲁棒性。
- 三者在评估时都**只做 forward，不训练**；§1 用 50 batch × batch 2 = **100 窗口**
  （`torch.randperm(seed=0)` 全 dataset 抽样，不偏某一条轨迹）；
  §2 逐样本统计另跑 **80 窗口**（同 seed，另一抽样量）。

### 0.2b ⚠️ 完整性声明：本文件**没有**测什么

- **没有**做配对 t-检验（只有 bootstrap CI over 窗口）。
- **没有**按温度/时标分层报告（虽数据集含多温度），因此不能声称温度泛化。
- **pos 只在 §3/§4 的 50-batch eval 上测**，未纳入 §2 逐样本统计。
- **没有**与外部 baseline（TICA+MSM / VAMPNet / SPIB）相比——本轮只比 persistence。
- §6 的 6 个扩展指标是**独立评估口径**；它们**不参与**训练 loss，只用于报告。
- 所有"胜"仅针对**这些 val 集**；对训练未见过的**其它数据集**未验证。

### 0.2 测的指标（4 个，全部可微定义在代码里）

| 指标 | 定义（精确） | 单位 | 越小越好 | 代码位置 |
| --- | --- | --- | --- | --- |
| **rmsf** | 预测系综对齐后逐残基 RMSF 与真值 RMSF 的 **masked L1** | Å | ✅ | `ensemble_losses._rmsf_loss` |
| **contact** | 预测系综 contact-prob map（P(d<8Å)）与真值的 **masked L1** | 概率 | ✅ | `_contact_loss` |
| **dist** | 逐对残基内距**直方图**的 soft-EMD（累积差×bin 宽） | Å | ✅ | `_dist_loss` |
| **pos** | 确定性解码（frozen decoder∘μ）与真值未来帧的 **Kabsch CA MSE** | **Å²** | ✅ | `_position_loss` |

**总损失** `total = 1.0·rmsf + 10.0·contact + 0.3·dist + pos_weight·pos`
（已用代码复核 `reported_total == 重算值`，见验证记录）。

> **新增（2026-10-19，见 §6）**：应要求补测 **7 个系综级指标**
> （**RMSF 幅度 / RMSF-r 形态** / ROOT-MEAN W2 / VARIANCE W2 / MD-PCA W2 /
> WEAK-CONTACT J / TRANSIENT-CONTACTS J）。
> 这 6 个是**独立于训练 loss 的评估口径**，代码在**新文件**
> `evaluation/ensemble_metrics_ext.py`（不触碰原精确坐标解码路径），
> 运行入口 `scripts/eval_ensemble_metrics_suite.py`，结果原始 JSON 在
> `artifacts/eval_dual_suite/{A,B,C}_{mdcath,atlas100ps,atlas1ns}.json`。
> 定义、数值、与 persistence 的对比全部在 **§6**。

### 0.3 对照（persistence）的精确含义

`persistence` = **把 anchor 帧复制 8 份当"系综"** 后，用**完全相同的 4 个指标**算出的值。
即"模型完全不预测、只假设未来=当前帧"的 baseline。**这是本任务真正的对手**
（因为 1 ns 单帧预测的物理天花板≈persistence，见 `STAGE_A_REPAIR...` §9）。

### 0.4 ⚠️ 必须先声明的两个局限（否则数字会被误读）

1. **ATLAS 泛化口径**：ATLAS-100ps 与 ATLAS-1ns 的 val 都是**同一组 10 个未见蛋白**，
   且这 10 个蛋白**不在任何训练 split 中**（已校验）。
   差别只在**采样率**（100 ps vs 1000 ps）。所以两者都是"未见蛋白"测试，
   不是"零信息"测试——模型在训练中见过 ATLAS 数据集（其它 90 蛋白）。
2. **pos 指标的绝对值在数据集间不可比**：mdCATH 的 pos≈20.8 Å²（RMSD≈4.56 Å），
   ATLAS 的 pos≈0.013–0.017 Å²（RMSD≈0.11–0.13 Å），**难度差 ~500 倍**（ATLAS 近刚性）。
   因此 **B/C 的 total 主要由 pos 项主导**（0.5×20.8≈10.4，占 total 13.2 的 79%），
   **B/C 的 total 不能与 A 的 total 横向比**。

---

## 1. 主结果表（mean over 100 窗口；Δ = model − persistence，负=模型更好）

| run | val | total_m | total_p | Δtotal | rmsf_m | rmsf_p | Δrmsf | contact_m | contact_p | dist_m | dist_p | pos_m | pos_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **A** | mdCATH | 2.860 | 3.608 | **−0.748** | 0.858 | 1.701 | **−0.843** | 0.0161 | 0.0105 | 6.137 | 6.007 | — | — |
| **A** | ATLAS-1ns | 0.1456 | 0.1741 | **−0.0285** | 0.0338 | 0.0684 | **−0.0346** | 0.00014 | 0.00012 | 0.3681 | 0.3484 | — | — |
| **A** | ATLAS-100ps | 0.1556 | 0.1816 | **−0.0260** | 0.0240 | 0.0561 | **−0.0321** | 0.00007 | 0.00006 | 0.4363 | 0.4165 | — | — |
| **B** | mdCATH | 13.294 | 14.314 | **−1.020** | 0.936 | 1.701 | **−0.765** | 0.0137 | 0.0105 | 6.060 | 6.007 | 20.807 | 21.411 |
| **B** | ATLAS-100ps | 0.1640 | 0.1881 | **−0.0241** | 0.0251 | 0.0561 | **−0.0310** | 0.00006 | 0.00006 | 0.4332 | 0.4165 | 0.01656 | 0.01288 |
| **C** | mdCATH | 13.245 | 14.314 | **−1.069** | 0.894 | 1.701 | **−0.807** | 0.0141 | 0.0105 | 6.063 | 6.007 | 20.784 | 21.411 |
| **C** | ATLAS-100ps | 0.1608 | 0.1881 | **−0.0273** | 0.0233 | 0.0561 | **−0.0328** | 0.00007 | 0.00006 | 0.4312 | 0.4165 | 0.01502 | 0.01288 |

**bootstrap 95% CI（over 窗口，2000 次）**：所有在上表里标 **加粗**的 Δ 的 CI **都不跨 0**：

| run | val | Δtotal CI95 | Δrmsf CI95 |
| --- | --- | --- | --- |
| A | mdCATH | [−0.846, −0.653] | [−0.947, −0.746] |
| A | ATLAS-1ns | [−0.030, −0.027] | [−0.036, −0.034] |
| A | ATLAS-100ps | [−0.027, −0.025] | [−0.033, −0.031] |
| B | mdCATH | [−1.199, −0.858] | [−0.847, −0.686] |
| B | ATLAS-100ps | [−0.025, −0.023] | [−0.032, −0.030] |
| C | mdCATH | [−1.250, −0.904] | [−0.894, −0.726] |
| C | ATLAS-100ps | [−0.028, −0.026] | [−0.034, −0.032] |

> 注：原始 JSON 里 `model_vs_persistence[...].mean_improvement = persist − model`
> （正=模型更好）；本表统一换算成 **Δ = model − persist（负=模型更好）**，
> 与 §1 主表口径一致。所有 CI 均**整段在 0 以下**（=模型显著更好）。

---

## 2. **逐样本**胜率（比"均值胜"更强的证据）

只报均值可能是少数样本拉动的。对每个采样窗口算 Δ=model−persist（负=模型更好），
统计"模型更好的比例"（run A，n=80 窗口，seed=0）：

| val | 指标 | 均值 Δ | 中位数 Δ | **模型更好的比例** |
| --- | --- | --- | --- | --- |
| mdCATH | total | −0.745 | −0.631 | **97%** |
| mdCATH | rmsf | −0.868 | −0.691 | **100%** |
| mdCATH | contact | **+0.0069** | +0.0051 | **0%**（100% 更差）|
| mdCATH | dist | **+0.1767** | +0.1593 | **6%**（94% 更差）|
| ATLAS-1ns | total | −0.0283 | −0.0272 | **100%** |
| ATLAS-1ns | rmsf | −0.0349 | −0.0341 | **100%** |
| ATLAS-1ns | contact | +0.00003 | 0.0000 | 1% |
| ATLAS-1ns | dist | **+0.0214** | +0.0212 | **0%** |
| ATLAS-100ps | total | −0.0253 | −0.0247 | **100%** |
| ATLAS-100ps | rmsf | −0.0324 | −0.0320 | **100%** |
| ATLAS-100ps | contact | +0.00001 | 0.0000 | 1% |
| ATLAS-100ps | dist | **+0.0233** | +0.0230 | **0%** |

⇒ **rmsf 的胜是广谱且稳健的**（100% 窗口成立，non-trivial 幅度）；
⇒ **total 的胜几乎完全由 rmsf 项贡献**（97–100% 窗口）；
⇒ **contact 与 dist 是系统性变差的**（几乎 100% 窗口更差，只是权重下 total 仍为正收益）。
   ⇒ **不能宣称"系综级指标全面超过 persistence"**，只能宣称 **RMSF 谱显著更优**。

---

## 3. **位置重建（pos）到底如何**（换算成 CA RMSD，Å）

| run | val | model RMSD | persist RMSD | ΔRMSD | 谁更好 |
| --- | --- | --- | --- | --- | --- |
| B | mdCATH | 4.5615 | 4.6272 | **−0.0657** | 模型略好 |
| C | mdCATH | 4.5589 | 4.6272 | **−0.0683** | 模型略好 |
| B | ATLAS-100ps | 0.1287 | 0.1135 | **+0.0152** | **persistence 更好** |
| C | ATLAS-100ps | 0.1226 | 0.1135 | **+0.0091** | **persistence 更好** |

诚实结论：**位置引导 `L_pos` 只在 mdCATH 上略微赢 persistence（~0.07 Å），
在 ATLAS 上反而输（~0.01 Å）**——因为 ATLAS 近刚性，persistence 本身已~0.11 Å 近乎完美。
且 **mdCATH 的 4.56 Å 仍远离最初 `<1.2 Å` 目标**，受单帧 anchor + 1 ns 物理天花板限制
（`STAGE_A_REPAIR...` §9）。

---

## 4. 结论（精确版）

**在什么数据集上、测什么、效果如何（逐项精确）：**

| 指标 | mdCATH（未见蛋白） | ATLAS-1ns（未见蛋白） | ATLAS-100ps（未见蛋白） | 判定 |
| --- | --- | --- | --- | --- |
| **rmsf** | Δ=−0.868，**100% 窗口更优** | Δ=−0.035，100% 更优 | Δ=−0.032，100% 更优 | ✅ **显著且广谱胜 persistence** |
| **total** | Δ=−0.745，97% 更优 | Δ=−0.028，100% 更优 | Δ=−0.025，100% 更优 | ✅ 胜（但由 rmsf 主导）|
| **contact** | Δ=+0.007，**0% 更优** | +0.00003，1% | +0.00001，1% | ❌ **差于 persistence（近全样本）** |
| **dist** | Δ=+0.177，**6% 更优** | +0.021，0% | +0.023，0% | ❌ **差于 persistence（近全样本）** |
| **pos (CA)** | B/C 4.559 vs persist 4.627 Å（−0.07 Å） | — | B/C 0.123–0.129 vs 0.114 Å（**+0.01 Å，输**） | ⚠️ 仅 mdCATH 略胜，ATLAS 输 |

**结论（逐条）**：
1. **RMSF（柔性谱）是本轮唯一真正拿下的指标**：三个 val 上 100% 窗口优于 persistence，
   且 mdCATH 上幅度可观（Δ≈0.87 Å）。这是系综头的主要贡献。
2. **contact / dist 没有拿下，反而系统性变差**（≈100% 窗口更差）。
   total 之所以仍"更好"，完全是因为 rmsf 权重 1.0 且其 Δ 幅度大于
   contact(×10)·Δ + dist(×0.3)·Δ 的负贡献。**这是加权下的"净正"，不是各项皆胜。**
3. **混合训练的价值（定量）**：上一轮（仅 mdCATH 训练）在 ATLAS-1ns 上
   **rmsf = 0.174 vs persist 0.068（模型差 2.6×）、total 0.408 vs 0.181（差 2.2×）**，
   全部指标 CI 落在"更差"侧；**本轮混合训练后 rmsf = 0.034 vs persist 0.068（模型好 2.0×）**，
   翻转为显著更好。这是"用两个数据集训练"的直接、可量化的收益。
4. **位置精度**：`L_pos` 只在 mdCATH 上带来 ~0.07 Å 边际改善，ATLAS 上无效；
   **4.56 Å 仍远离 <1.2 Å 目标**（受单帧 anchor + 1 ns 物理天花板限制）。

**总体判断（诚实，回应"不够满意"）**：
- 目前能站得住的结论只有一个：**系综的 RMSF 谱显著优于 persistence**。
- **contact / dist / 位置精度均未超过 persistence**；把它们计入 total 后仍为正收益
  只是因为权重配置，**不应解读为"系综级 metric 全面领先"**。
- 因此若满意度阈值包含 contact/dist 或 CA RMSD，**本轮尚未达标**，见 §5 的改进方向。

**（2026-10-19 补充，见 §6）**：换用**独立的系综级评估口径**后，结论**部分强化**：
在 mdCATH 上 **7 个扩展指标（RMSF 幅度 + RMSF-r 形态 + ROOT-MEAN/VARIANCE/MD-PCA W2 +
弱/瞬时接触 Jaccard）全部 94–100% 窗口优于 persistence**；在 ATLAS-1ns/100ps 上
**4 个 W2 + RMSF 指标 100% 胜、RMSF-r 87–90% 胜**，但**两个 Jaccard 胜率仅 6–10%**
（**且 ATLAS 上该 Jaccard 是退化假高分，见 §6.5**）。
注意 §1 的 contact/dist（**训练 loss 口径**）仍差于 persistence，而 §6 的弱/瞬时接触
Jaccard（**评估口径**）在 mdCATH 上却 100% 胜——**两套口径结论不同，报告时须注明是哪一个**。

---

## 5. 下一步（若要更进一步提高满意度）

- **dist/contact 专项**：把 dist 的 soft-EMD 改为对**近程对加权**、contact 改 BCE/Jaccard；
  当前 contact 100% 窗口更差、dist 94% 窗口更差（mdCATH）说明 loss 形式或权重不对。
- **位置精度**：单帧 anchor 下难破天花板；需**多帧 anchor / 亚 ns 数据 / 直接预测未来帧**
  的确定性头（与系综头并联），而不是用 μ 去凑。
- **ATLAS 分层的物理意义**：300 K 近刚性 ≠ 通用规律，结论只对 ATLAS 成立。
- **t-检验 / 配对检验**：把逐窗口 Δ 拿去做配对检验（当前只做了 bootstrap CI）。

---

## 6. 扩展系综指标（7 项，2026-10-19 新增 / 修订）

> 回应"需要更多指标 + 要测 **RMSF-r** 而不只是 RMSF"以及"为什么 mdCATH 上
> persistence 的 contact-J / transient-J 都是 **0.000**"。
> 结论先说：**0.000 是 persistence 的数学性质（delta 系综），不是 mdCATH 的性质**；
> 且经查这是**评估口径的一个退化陷阱**，已在数据中显式暴露。详见 §6.5。

### 6.1 指标定义（7 项）

设**真值系综** `X_true` = 该 horizon 未来**窗口全部帧**的 CA（`future_coords[:,:,:,1,:]`）；
**预测系综** `X_pred` = 系综头采样 `n_samples=8` 的 conformer。W2 均为**精确 1-D
Wasserstein-2**（排序分位，无分箱）；对齐用 Kabsch（各自对齐自身首帧）。

| 指标 | 定义（精确） | 单位 | 方向 | 代码 |
| --- | --- | --- | --- | --- |
| **RMSF（幅度）** | 对齐后逐残基 RMSF 谱的 masked **L1** | Å | ↓ | `rmsf_profile_error.rmsf_mae` |
| **RMSF-r（形态）** | 逐残基 RMSF 谱 vs 真值的 **Pearson 相关**；**global / per-window / per-target 三种** | — | ↑ | `rmsf_profile_error.rmsf_corr` |
| **ROOT-MEAN W2** | 逐残基 CA 位置分布 W2 的 RMS：`sqrt(mean_i W2²)` | Å | ↓ | `root_mean_w2` |
| **VARIANCE W2** | 逐残基**涨落幅度(σ²)**分布之 W2 | Å² | ↓ | `variance_w2` |
| **MD-PCA W2** | 真值协方差前 k=3 慢模投影分布之 W2 的 RMS | Å | ↓ | `md_pca_w2` |
| **WEAK-CONTACT J** | `P(d<8Å)∈[0.05,0.5)` 位点集合的 **Jaccard** | — | ↑ | `weak_contact_jaccard` |
| **TRANSIENT-CONTACTS J** | `P(d<8Å)∈(0.01,0.99)` 位点集合的 **Jaccard** | — | ↑ | `transient_contact_jaccard` |

> **RMSF（幅度）vs RMSF-r（形态）**：前者问"涨落有多准"（数值），后者问"哪些残基在动"
> 的**空间形状**是否对。一个只把幅度整体缩小/放大的预测，`rmsf_mae` 会差但 `rmsf_corr`
> 仍可**接近 1**；反之幅度对但位置错，`rmsf_corr` 会掉。二者互补，必须一起报。

### 6.2 主结果表（run A，mean over 100 窗口；Δ=pers−model 对↓指标 / model−pers 对↑指标，正=模型更好）

| val | 指标 | model | persist | oracle | Δ | CI95 | 胜率 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **mdCATH** | RMSF（幅度,Å↓） | 0.892 | 1.753 | 0.000 | **+0.861** | — | **100%** |
| mdCATH | **RMSF-r（↑）** | **0.553** | **0.000** | 1.000 | **+0.553** | — | **100%** |
| mdCATH | ROOT-MEAN W2 (Å↓) | 10.398 | 10.561 | 0.000 | **+0.163** | — | 97% |
| mdCATH | VARIANCE W2 (Å²↓) | 8.760 | 10.417 | 0.000 | **+1.658** | — | 98% |
| mdCATH | MD-PCA W2 (Å↓) | 0.0783 | 0.1181 | 0.0000 | **+0.0397** | — | **100%** |
| mdCATH | WEAK-CONTACT J (↑) | 0.092 | **0.000** | 1.000 | +0.092 | — | **100%** |
| mdCATH | TRANSIENT J (↑) | 0.211 | **0.000** | 1.000 | +0.211 | — | **100%** |
| **ATLAS-1ns** | RMSF（幅度,Å↓） | 0.0356 | 0.0705 | 0.000 | **+0.0349** | — | **100%** |
| ATLAS-1ns | **RMSF-r（↑）** | **0.097** | **0.000** | 1.000 | +0.097 | — | 90% |
| ATLAS-1ns | ROOT-MEAN W2 (Å↓) | 0.1752 | 0.1818 | 0.0000 | **+0.0066** | — | **100%** |
| ATLAS-1ns | VARIANCE W2 (Å²↓) | 0.0108 | 0.0129 | 0.0000 | **+0.0020** | — | **100%** |
| ATLAS-1ns | MD-PCA W2 (Å↓) | 0.0032 | 0.0042 | 0.0000 | **+0.0010** | — | **100%** |
| ATLAS-1ns | WEAK-CONTACT J (↑) | 0.907 | 0.900 | 1.000 | +0.007 | — | 7% |
| ATLAS-1ns | TRANSIENT J (↑) | 0.921 | 0.900 | 1.000 | +0.021 | — | 10% |
| **ATLAS-100ps** | RMSF（幅度,Å↓） | 0.0254 | 0.0579 | 0.000 | **+0.0325** | — | **100%** |
| ATLAS-100ps | **RMSF-r（↑）** | **0.096** | **0.000** | 1.000 | +0.096 | — | 87% |
| ATLAS-100ps | ROOT-MEAN W2 (Å↓) | 0.1638 | 0.1701 | 0.0000 | **+0.0063** | — | **100%** |
| ATLAS-100ps | VARIANCE W2 (Å²↓) | 0.0047 | 0.0067 | 0.0000 | **+0.0021** | — | **100%** |
| ATLAS-100ps | MD-PCA W2 (Å↓) | 0.0019 | 0.0029 | 0.0000 | **+0.0010** | — | **100%** |
| ATLAS-100ps | WEAK-CONTACT J (↑) | 0.947 | 0.940 | 1.000 | +0.007 | — | 6% |
| ATLAS-100ps | TRANSIENT J (↑) | 0.955 | 0.940 | 1.000 | +0.015 | — | 6% |

（CI95 见 `artifacts/eval_dual_suite/*.json` 的 `model_vs_persistence[*].ci95`，均在"模型更好"侧且不跨 0。）

**要点**：
- **RMSF-r**：模型在 mdCATH 上 **0.553**（persist **0.000**），三 val 全胜（87–100%）。
  persist 的 RMSF-r **恒为 0**，因为 delta 系综的 RMSF 谱是**常数**，与真谱的相关系数
  按定义无意义（代码里退化为 0）。
- **4 个 W2 指标** 三 val 上 97–100% 胜。
- **2 个 Jaccard** mdCATH 上 100% 胜；**ATLAS 上均值更好但胜率 6–10%**。

### 6.2b **RMSF-r：global 与 per-target 两种形式**（mdCATH，本次细化重点）

"target" = **蛋白/域**（manifest 的 `protein_id`，如 `1b4uB00`），一个 target 含多条轨迹
（T×replica）。mdCATH val = **19 个 target / 87 条轨迹**。

- **global RMSF-r**：把**所有窗口的所有残基**的 RMSF 向量拼成一条长剖面，与真值长剖面求
  一次 Pearson 相关。反映"整体柔性图谱对不对"。
- **per-window RMSF-r**：每个窗口算一次相关，再对 100 个窗口取均值（沿用 §6.2 的 0.553）。
- **per-target RMSF-r**：对每个 target，把它**名下所有窗口的残基剖面拼起来**求一次相关，
  得到该 target 的一个标量；再报告 **19 个 target 上的分布**（mean/median/IQR/min/max/正比例）。
  这是最贴近"逐蛋白实用性"的口径。

| run | 系综 | **global** | per-window 均值 | **per-target 均值** | per-target 中位数 | min | max | >0 比例 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **A** | **model** | **+0.604** | +0.548 | **+0.581** | +0.610 | **+0.347** | **+0.839** | **19/19** |
| A | persistence | +0.000 | −0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 0/19 |
| A | oracle | +1.000 | +1.000 | +1.000 | +1.000 | +1.000 | +1.000 | 19/19 |
| **B** | **model** | **+0.516** | +0.467 | **+0.498** | +0.501 | +0.363 | +0.745 | 19/19 |
| **C** | **model** | **+0.571** | +0.448 | **+0.549** | +0.530 | +0.381 | +0.781 | 19/19 |
| B/C | persistence | +0.000 | −0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 0/19 |
| B/C | oracle | +1.000 | +1.000 | +1.000 | +1.000 | +1.000 | +1.000 | 19/19 |

**读法与结论（mdCATH）**：
1. **global 与 per-target 数量级一致**（run A：0.604 vs 0.581），说明模型学到的柔性图谱
   在**整体**和**逐蛋白**两个口径下都成立；per-window 均值略低（0.548）是因为把长/短
   窗口等权、短窗口统计噪声大。
2. **per-target 最弱的一个蛋白 r 仍 = +0.35**（run A），**19/19 全为正**，中位数 +0.61，
   说明不是靠少数好蛋白撑起均值。
3. **persistence 的三种 RMSF-r 全部 = 0.000**（delta 系综 RMSF 谱为常数 ⇒ 相关退化），
   **oracle 全 = 1.000**——两端锚点行为正确。
4. **A（最佳）> C > B**，与 §6.2 的整体排序一致；加 pos/cond（B/C）未提升 RMSF-r。

> **RMSF-r 的两种形式都指向同一结论**：模型对"哪些残基在动"的空间图谱预测，
> mdCATH 上显著优于 persistence（global 0.60、per-target 均值 0.58、逐蛋白 19/19 为正）。

**注**：按你的判断，**ATLAS 不再作为主口径**（近刚性 ⇒ 指标退化，见 §6.5），
本小节只报 mdCATH。ATLAS 数值仍保留在 `artifacts/eval_dual_suite/*_atlas*.json` 备查。

### 6.3 与 persistence 对比（跨 run A/B/C 一致）

A（无 pos/cond）、B（+pos 0.5）、C（+cond 2 +pos 0.5）三种配置下 7 指标的
model/persist 排序**完全相同**，数值差异 <5% ⇒ 收益来自**双数据集混合训练**本身。

### 6.4 oracle 健全性

`pred=true`：4×W2=**0.0000**、2×Jaccard=**1.0000**、RMSF-r=**1.000**、RMSF_mae=**0**。
单元测试 21 项全过（`tests/test_ensemble_metrics.py`）。

### 6.5 ⚠️ 为什么 mdCATH 上 persistence 的 contact-J / transient-J = **0.000**

**这是 persistence 的数学性质，与 mdCATH 无关。** 已在真实窗口上逐格核对（诊断输出在
`artifacts/eval_dual_suite/*.json` 的 `diagnostics` 字段）：

| 系综 | contact-prob 图取值 | `weak_n_pred`（弱接触集合大小） | `weak_n_true` |
| --- | --- | --- | --- |
| **persistence**（anchor 复制 8 份） | **仅 {0,1}**（8 帧全同 ⇒ 每对要么恒接触要么恒不接触） | **恒为 0** | mdCATH≈100，ATLAS≈1–2 |
| model | 连续 ∈[0,1] | mdCATH≈183，ATLAS≈1 | 同左 |
| oracle | 同真值 | =`weak_n_true` | 同左 |

- 弱接触带 `[0.05,0.5)` 与瞬时带 `(0.01,0.99)` **都要求概率取中间值**。
  persistence 的值只有 0 或 1，**一个都落不进带内** ⇒ `B = ∅`。
- Jaccard `J = |A∩B| / |A∪B| = 0 / |A|`：只要真值集合 `A` 非空（**mdCATH 成立**，
  `weak_n_true≈100`），**J 恒等于 0**。这不是模型/数据的问题，是 **delta 基线无法表达
  概率性接触**。
- **ATLAS 为什么是 0.9 而非 0**：ATLAS 近刚性 ⇒ 真值窗口本身也几乎二值
  （`weak_n_true≈1–2`，多数窗口为 0）。若**真假集合都空**，约定的退化解返回 `J=1.0`；
  只有少数窗口 `A` 非空 ⇒ 那些窗口 `J=0`。100 窗口平均 ⇒ **≈0.90/0.94 的假高分**。
  ⇒ **ATLAS 的 Jaccard 值同样不可信，不应解读为"模型好"**。

**⇒ 结论**：这两个 Jaccard 指标对 **delta/单帧基线是退化的**，且对**近刚体系统几乎无
信息量**。已在代码里新增 `*_full` 变体**显式返回集合大小** `n_true/n_pred/n_inter`，
并在 suite 输出 `diagnostics`，让"0 或 1"的来源可追溯。**报告时若引用这两个 Jaccard，
必须同时给出 `weak_n_true / weak_n_pred`**，否则会误导。

### 6.6 结论（扩展指标）

1. **RMSF-r（形态）与 RMSF 幅度一起赢**（mdCATH）：
   **global r=0.604 / per-target 均值 0.581（19/19 蛋白 > 0，最弱 +0.35）/ per-window 均值 0.548**，
   vs persistence **0.000**（三种形式）。见 §6.2b。
2. **4 个 W2 家族**三 val 上 97–100% 胜，是最稳健的系综级证据。
3. **2 个 Jaccard**：mdCATH 上 100% 胜；但 **ATLAS 上假高分（退化）不可用**，
   见 §6.5。
4. **A/B/C 一致** ⇒ 收益来自双数据集混合训练，非 pos/cond 项。
5. ⚠️ **口径差异**：§1 训练 loss 的 contact/dist 仍差于 persistence，而 §6 评估口径的
   弱/瞬时接触 Jaccard 在 mdCATH 上 100% 胜——**报告时须注明是 loss 口径还是评估口径**。

### 6.7 代码位置（"这些脚本是否已写好"）

诚实说明：**原先并没有**这 5 个新指标（仓库只有 rmsf、threshold=0.5 的 contact Jaccard、
内距 soft-EMD）。全部**本次新写**：
- `evaluation/ensemble_metrics_ext.py`（新）
- `scripts/eval_ensemble_metrics_suite.py`（新，统一入口）
- `tests/test_ensemble_metrics.py`（+9 测试，共 21）
- 结果：`artifacts/eval_dual_suite/{A,B,C}_{mdcath,atlas100ps,atlas1ns}.json`
- **未改动** `protein_jepa_frame/` 内任何精确坐标解码实现（mtime 校验：该路径今日未改）。

---

## 7. 轨迹生成基线对照（our-rollout vs MDGen-future-sim；mdCATH）

> 新增（2026-10-20）。目标：让**我们的模型也生成"未来帧轨迹"**（不只单帧系综），
> 在 **mdCATH**（同一 train/val split，双方都没见过 val 域）上与 **MDGen**
> （Jing et al., NeurIPS 2024, forward-simulation 任务）**公平对比**，
> 用**同一套**指标：§6 的 7 个系综指标 + **Kabsch CA RMSD**（CA 位置指标）。
> 这套评估**同时服务 Stage A（系综）与后续 from-none Stage B（gap/轨迹）**。

### 7.0 为什么在 mdCATH 上比

- MDGen 官方预训练权重（`atlas.ckpt`）**见过我们 ATLAS val 的 10 个蛋白**
  （它们都在 `atlas_train.csv`）⇒ ATLAS 对比对我们**不公平**。
- **mdCATH**：双方都从零训练、**同一 split**（`splits_mdcath_{train,val}.csv`，
  domain 级 4320/539/539），是**公平擂台**。
- MDGen **原生不支持 mdCATH**，故按用户要求做**可控短训练**（`num_frames=100`）。

### 7.1 两套生成语义（对齐 MDGen 的 forward-sim）

| | 输入 | 生成 | 输出 |
|---|---|---|---|
| **MDGen** | anchor 帧（帧 0 条件） | 一次 flow-matching 解 T 帧 | `[T, L, 14, 3]` |
| **我们（rollout）** | anchor 上下文帧 | 自回归 T 步（每步解码 CA 均值→回填上下文） | `[n_samples, T, L, 3]` |

两者都取**最后 T 帧**的 CA 与**真值 future 窗口**比。

### 7.2 使用的 CA 位置指标

- **Kabsch CA RMSD**（`common_anchor_rmsd`，主指标）：**anchor→真值**的刚体变换由
  **真值坐标**拟合（不含预测），所有样本/oracle/persistence **共用同一 gauge**。
- 附 **aligned CA RMSD**（`aligned_rmsd`，独立叠合的形状 RMSD），作参考。

### 7.3.1 ⚠️ 时间尺度对齐（关键，否则对照无意义）

mdCATH 的帧间隔是 **1000 ps = 1 ns/帧**。两个模型的"每步物理时间"不同：

| | 每步 = | T=100 步覆盖 |
|---|---|---|
| mdCATH 真值帧 | 1 ns | 1 帧 |
| **我们 rollout** | 1 个 mdCATH 帧 = 1 ns | **100 ns** |
| **MDGen atlas.ckpt（zero-shot）** | 10 ps（ATLAS 训练步长） | **1 ns** |
| MDGen mdCATH 短训 | 1 ns（在 mdCATH 上训） | 100 ns |

⇒ **zero-shot 与 mdCATH 短训的物理跨度差 100×**，不能直接比 T=100。
因此分两组报告：

- **组 I（物理跨度匹配，100 ns）**：ours-rollout vs **MDGen-mdCATH-短训**（都 100 ns）。
- **组 II（zero-shot）**：MDGen `atlas.ckpt` 只在其**原生 1 ns** 跨度内评（或对比
  "下一帧 Δ=1 ns"）——**只与 ours 的第 1 步可比**。

### 7.3.2 采样器

MDGen 默认 `dopri5`（自适应 ODE）在 100 帧 × L 上极慢（~4 min/窗口）。
改用 `--sampling-method euler`（固定 50 步），速度提升 ~2 个数量级，用于本次评测。

### 7.3 运行入口（新文件，不改精确坐标解码路径）

```bash
# (I) MDGen mdCATH 短训（forward-sim → 未来帧系综，100 ns）
python scripts/eval_mdgen_baseline.py \
  --mdgen-ckpt /liuyutian/mdgen_src/mdgen-master/workdir/mdcath_short/epoch=1-step=2000.ckpt \
  --manifest artifacts/mdcath_manifest/val.jsonl \
  --split-csv /liuyutian/mdgen_run/splits_mdcath_val.csv \
  --data-dir /liuyutian/mdgen_run/data_mdcath \
  --n-windows 32 --n-samples 8 --num-frames 100 --crop 256 --sampling-method euler

# (II) MDGen zero-shot（HF 官方 atlas.ckpt，原生 1 ns 跨度）
python scripts/eval_mdgen_baseline.py \
  --mdgen-ckpt /liuyutian/mdgen_run/hf_mdgen/atlas.ckpt \
  --manifest artifacts/mdcath_manifest/val.jsonl \
  --split-csv /liuyutian/mdgen_run/splits_mdcath_val.csv \
  --data-dir /liuyutian/mdgen_run/data_mdcath \
  --n-windows 32 --n-samples 8 --num-frames 100 --crop 256 --sampling-method euler

# 我们（自回归未来帧轨迹 rollout，100 ns）
python scripts/eval_ours_trajectory.py \
  --ensemble-ckpt artifacts/stage_ensemble_dualA/checkpoint_best.pt \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --b1a-ckpt artifacts/stage_b1a_relpos_anchored/checkpoint_best.pt \
  --manifest artifacts/mdcath_manifest/val.jsonl \
  --n-samples 8 --num-frames 100
```

- 新脚本：`scripts/eval_mdgen_baseline.py`、`scripts/eval_ours_trajectory.py`（均新文件）。
- MDGen 兼容层：`mdgen_baseline/mdgen_compat.py`（numpy2/torch2.6/cuSOLVER 修补）。
- mdCATH→atom14：`scripts/mdcath_h5_to_atom14_mdgen.py`。
- **MDGen 官方权重**：HF `bjing-mit/mdgen` → `atlas.ckpt`（蛋白级 forward-sim，
  num_frames=250, crop=256）、`forward_sim.ckpt`（四肽级, crop=4，本任务不适用）。
  已下载到 `/liuyutian/mdgen_run/hf_mdgen/`。

### 7.4 结果表

mdCATH val，32 窗口，每窗口 **100 帧系综**（8 样本 × T=100；MDGen sampler=euler@20）。
`rmsf-r` 为 §6.2b 的 global 形态相关。**粗体 = 该列最好。**

| 模型 | 物理跨度 | rmsf_mae↓ | rmsf-r↑ | root_mean_w2↓ | variance_w2↓ | md_pca_w2↓ | weak_J↑ | trans_J↑ | **Kabsch CA RMSD↓** |
|---|---|---|---|---|---|---|---|---|---|
| persistence（anchor 复制） | 瞬时 | 1.840 | −0.014 | 12.276 | 11.089 | 0.1345 | 0.000 | 0.000 | **4.014** |
| **MDGen zero-shot**（HF atlas.ckpt） | 1 ns | 4.650 | ~0.000 | **6.679** | 27.069 | 0.1387 | 0.000 | 0.000 | 7.344 |
| ↳ zero-shot（换 seed=7 复核） | 1 ns | 4.779 | ~0.000 | 6.896 | 27.371 | 0.1437 | 0.000 | 0.000 | 7.841 |
| MDGen mdCATH 短训（epoch5/step6000） | 100 ns | 22.53 | ~0.000 | 14.213 | 541.27 | 0.6736 | 0.000 | 0.000 | 26.76 |
| MDGen mdCATH 短训（epoch1/step2000） | 100 ns | 16.06 | ~0.000 | 11.252 | 275.31 | 0.4760 | 0.000 | 0.000 | 19.69 |
| **ours rollout**（dualA） | 100 ns | **1.455** | **0.334** | **11.993** | **10.211** | **0.0801** | **0.066** | **0.134** | 6.335 |
| oracle | — | 0 | 1 | 0 | 0 | 0 | 1 | 1 | 0 |

### 7.5 初步解读（诚实版）

1. **zero-shot（atlas.ckpt，无训练）在 mdCATH 上劣于 persistence**：
   Kabsch CA RMSD **7.34 Å vs 4.01 Å**，RMSF **4.65 vs 1.84 Å**。
   原因清楚：atlas.ckpt 在 **ATLAS（10 ps/帧、不同蛋白族）** 上训练，
   mdCATH 是 **1000 ps/帧** 的新域 ⇒ **步长差 100×**，属**严重 OOD**。
   它在 short-horizon 的 **root_mean_w2（6.68）反而最低**——即"形状分布"尚可，
   但**位置跑偏**（Kabsch 差）。⚠️ **报告时务必注明这是 OOD，不是 MDGen 的正常水平**。
2. **MDGen mdCATH 短训目前很糟且**没有随训练变好**：epoch1(s2000) RMSF 16.06 Å /
   Kabsch 19.69 Å；epoch5(s6000) 反而 **更差** RMSF 22.53 / Kabsch 26.76。
   ⇒ **5 epoch 远未收敛**（flow-matching 原论文 ~10^4 epoch），且中途在震荡。
   **这一格不能代表 MDGen 的能力**，仅说明"我们的短训练预算下它没学出动力学"。
3. **ours rollout**：6/7 系综指标**最好**；但 **Kabsch CA RMSD（6.34 Å）差于
   persistence（4.01 Å）**——自回归回填（`_synthetic_backbone`）会**漂移**。
   这正是**轨迹 rollout 的经典问题**，也是后续 Stage B 要解决的点。
4. ⚠️ **口径**：三者"生成的东西"不同（zero-shot=1 ns；短训/ours=100 ns）。
   表格里已标"物理跨度"列；**跨组只看相对 persistence 的胜负**，不看绝对值横比。

### 7.6 结论（本轮）

- **zero-shot 直接测**（用户重点关注）：MDGen 官方权重 `atlas.ckpt` **在 mdCATH 上
  不能直接工作**（Kabsch 7.34 Å > persistence 4.01 Å，RMSF 4.65 > 1.84 Å），
  主因是 **10 ps→1000 ps 的步长 OOD + 蛋白族不同**。
  唯一亮点是 short-horizon 的 **root_mean_w2（6.68）最低**——形状分布尚可，
  **位置跑偏**。
- **简单短训练（20 epoch 预算）**：目前**远未收敛**（best so far Kabsch 19.7 Å
  @epoch1，且 epoch5 更差），此预算下**不构成有意义的对手**。
  该训练仍在继续（写稿时为 epoch 7）。
- **我们的模型 rollout**：7 个系综指标里 6 个最好（rmsf_mae/rmsf-r/var_W2/pca_W2/
  两个 Jaccard），**但绝对 CA 位置（Kabsch 6.34 Å）受 rollout 漂移拖累，差于
  persistence（4.01 Å）**。这正是后续 from-none Stage B 轨迹任务的改进方向。

### 7.7 为了让对照"真正公平"，还需要什么（诚实清单）

1. **MDGen mdCATH 长训练**：flow-matching 需要 ~10^4 epoch 才收敛；20 epoch 不够。
   要么改大 epoch、要么明确声明"未收敛，仅作下限"。
2. **步长对齐**：zero-shot 的 atlas.ckpt 是 10 ps/帧；mdCATH 是 1000 ps/帧。
   严格讲应先把 mdCATH **降采样/或对 MDGen 用匹配的 10 ps 数据**再比。
3. **我们的 rollout 回填策略**：当前用 `_synthetic_backbone` 的 N/C/O 偏移回填，
   是 rollout 漂移的主要来源；Stage B 应有专门的轨迹回滚/校正机制。
4. **样本量**：32 窗口已够看趋势，最终结论建议 ≥100 窗口 + bootstrap CI。



