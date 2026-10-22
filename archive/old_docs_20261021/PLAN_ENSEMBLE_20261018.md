# PLAN: 系综级 (ensemble-level) 目标引导 JEPA —— 2026-10-18

> 背景：`STAGE_A_REPAIR_RESULTS_20261018.md` §9–§10 已证：**在 mdCATH（1 ns/帧）+ 单帧 anchor 下，
> 单帧绝对 CA RMSD 的地板是 persistence ≈ 3–4.5 Å，`<1.2 Å` 物理不可达**（H 的 held-out R²≈0.02–0.05）。
> 因此本 plan 转向 **系综级目标**：不追求逐帧绝对坐标，而是让模型预测**分布/系综可观测统计量**
> （RMSF、接触图、内部距离分布、VAMP-2 动力学、态概率/转移），并在 mdCATH 上与已有 baseline 对比。

---

## 0. 目标与验收（一句话）

**让我们的 JEPA（Stage A encoder + future-state producer + decoder）在 mdCATH 上，用系综级 loss 训练后，
在系综级指标上显著优于 persistence / TICA+MSM / VAMPNet / SPIB 基线。**

核心指标（详见 §4）：**RMSF profile 相关性**、**contact-map Jaccard/Precision**、
**内部距离分布 (ensemble) 的 Wasserstein/JS**、**VAMP-2**、**态转移 Brier skill**。

---

## 1. 别人在 mdCATH 上做的 baselines（现状盘点）

### 1.1 仓库内已有（可直接跑，`baselines/`）
| 方法 | 类型 | 实现文件 | 在 mdCATH 上测的指标 | 已有结果 |
|---|---|---|---|---|
| **TICA + MSM** | 经典 | `baselines/tica_msm.py` | VAMP-2 @lag 1/2/5/10/20；态 top-1/top-2/balanced/persistence | ✅ `baselines/results/tica_msm/` |
| **VAMPnets** | 神经 | `baselines/vampnets.py` | 同上 | ✅ `baselines/results/vampnet/` |
| **SPIB** | 神经 | `baselines/spib.py` | 同上（+熵瓶颈 β） | ✅ `baselines/results/spib/` |
| **V4-D (ours, 前身)** | 神经 | `baselines/run_v4d_vamp2*.py` | 64D/128D latent VAMP-2 | ✅ `baselines/results/v4d_test/` |
| **Persistence** | 平凡 | `evaluation/geometry.py:displacement_msd` | anchor→future RMSD/MSD | 隐含在各 eval |
| **Oracle / anchor_decode** | 诊断 | `evaluation/run_eval.py` | 上界/下界 | ✅ 已接入 |

**已有 baseline 关键数字（`baselines/results/comparison_report.md`，val 30 蛋白 / 354 轨迹）**：
- VAMP-2 (6-state, lag=1)：TICA+MSM **1.967**、VAMPNet **2.226**、SPIB **2.923**（SPIB 最佳）。
- 态精度：TICA+MSM top-1 **31.67%**；VAMPNet/SPIB top-2 **67.12%**；
  **persistence 态精度 93.64%**（→ 态预测任务本身被 persistence 主导，Brier skill 才是有意义的指标）。
- Brier skill：V4-D **+41.37%**（唯一正值的软概率方法）；硬标签 baseline 退化 (NaN/-554%)。

### 1.2 文献中 mdCATH 上的相关工作（需引用/对齐）
- **mdCATH 数据集论文**：Majewski et al., *"Machine learning coarse-grained models..."* /
  mdCATH (2024) — 提供数据集与原子级 MD，是 benchmark 定义源。
- **VAMP / VAMPnets**：Mardt et al. 2018（`baselines/vampnets.py` 已实现）。
- **SPIB**：Wang et al. 2020（已实现）。
- **TICA+MSM**：Pérez-Hernández et al. 2013（已实现）。
- **需补的外部队列（下一步检索/对齐）**：
  - 蛋白动力学**取代/生成类**：Timewarp、Str2Str、AlphaFlow/ESMFlow（系综采样）。
  - **RMSF/柔性预测**：ESM-Flex / flexible-ESM 系列。
  - **接触图预测**：ESM-2 contact、trRosetta 类（作为系综级结构统计上界）。
  - **温度泛化**：mdCATH 含 5 温度，需按温度分层评估。

> 本节先以 **仓库内可复现的 3+1 baseline** 为对照主力（它们已在同一 val split、同一 CA 特征上算过），
> 外部方法是 **下一轮补强项**（§6 T0）。

---

## 2. 什么是"系综级 loss"（本 plan 的定义）

单帧目标：`L = ‖X̂_CA(future frame) − X_true_CA(该帧)‖`（已证不可达）。
系综级目标：对**一段窗口 / 多个未来帧 / 多次采样**求**统计量**，用统计量匹配做监督：

| 统计量 | 定义（可微） | 物理含义 |
|---|---|---|
| **RMSF profile** | `r_i = sqrt(mean_t ‖x_i(t) − mean_t x_i‖²)`，按残基 | 每残基柔性 |
| **Contact map** | `C_ij = P(d_ij < 8Å)`（对窗口/采样平均） | 系综接触概率 |
| **内部距离分布** | 对采样/未来帧的 `{d_ij}` 直方图/KDE | 系综几何分布 |
| **协方差 / 慢模** | `Cov(x)` 的前 k 主特征值/特征向量 | 动力学自由度 |
| **VAMP-2** | 由 H 轨迹算（冻结 encoder 下可微） | 动力学信息量 |
| **态分布/转移** | soft state 概率、`P(z_{t+1}|z_t)` | 动力学生成质量 |

**关键设计**：系综统计量的"真值"来自**未来窗口的多个真实帧**（mdCATH 一帧 1 ns，窗口 = 未来 K 帧），
模型侧来自 **future-state producer 的多次采样/rollout**（H+S 变体天然支持）或**直接对 H_future 解码的ensemble**。

---

## 3. 架构改动（最小侵入，保持 JEPA 主体）

1. **Stage A 不动**（relpos encoder 保持冻结/或极小学习率）——系综 loss 主要是**引导 producer + decoder**。
2. **Future-state producer**：从"单点 `H_future`"改为可输出 **分布/多采样**：
   - 已有 `h_plus_s` 的 **GLE rollout**（`evaluation/variants.py`）已能出 `ca_samples` → 直接复用为系综。
   - 新增轻量路径：对 `H_future` 加可学习的高斯头 `H_future ~ N(μ(H_ctx), Σ(H_ctx))`，采样 N 次。
3. **Decoder 不动**（anchored decoder，已验证正确、泛化）。
4. **Loss**：新增 `EnsembleLoss`（§2 表的可微版），与现有 `ca_mse`/`phys_future` 并列，权重可调。

---

## 4. 指标与评估协议（固定，所有方法一致）

**Split**：与 baseline 完全一致——val 30 蛋白 / 354 轨迹；test 29 蛋白 / 342 轨迹（作最终锁定）。
**温度**：mdCATH 5 温度分层报告（320/348/379/413/450 K）。
**Horizon**：τ ∈ {1, 2, 4, 8, 16, 32} ns（帧 offset）。

| 指标 | 主/辅 | 对照 baseline | 说明 |
|---|---|---|---|
| RMSF profile 相关/MAE | **主** | persistence、ESM-Flex(外部) | 每残基柔性 |
| Contact-map Jaccard / Precision | **主** | persistence、oracle(上界) | 已有 `geometry.contact_jaccard` |
| 内部距离分布 Wasserstein/JS | **主** | persistence、oracle | 对新指标，需新增 |
| VAMP-2（latent） | **主** | TICA/VAMPNet/SPIB | 直接可比 |
| 态转移 Brier skill | **辅** | V4-D(+41.37%) | 已有流水线 |
| ensemble CA RMSD（多采样均值） | 诊断 | oracle / anchor_decode | 已有 `sample_rmsds` |

**必带下界/上界**：`persist`（不懂动力学）、`oracle`（真 H 解码，上界）、`anchor_decode`。

---

## 5. Task list（step-by-step）

- **T0 [ ] 外部 baseline 检索与对齐**：整理 mdCATH 上 Timewarp/Str2Str/ESM-Flex 等的公开报告数字，
  写进本 md §1.2 表格（标明可复现 vs 引用）。
- **T1 [x] 系综评估指标实现**：
  - `evaluation/ensemble_metrics.py`：`rmsf_profile`、`contact_prob_map`、`distance_distribution_emd`、
    `covariance_slow_modes`；每个都返回标量 + 与 persist/oracle 的对比。
  - 单测 `tests/test_ensemble_metrics.py`（解析可验证的 toy case）。
- **T2 [x] 系综目标生成（数据侧）**：从 mdCATH 未来窗口（多帧）算真值 RMSF/contact/距离分布，
  产出 cache：`artifacts/ensemble_targets/{train,val}.npz`。
- **T3 [x] 模型侧系综采样**：
  - 复用 `h_plus_s` rollout 的 `ca_samples`；补一个 `GaussianH` head 版 producer（可微、快）。
  - 单测：采样可微、维度正确、zero-H 退化为某固定分布。
- **T4 [x] `EnsembleLoss`**：`L_ens = w_rmsf·L_rmsf + w_contact·L_contact + w_dist·L_dist`
  （全部可微，smooth-L1 / 交叉熵 / EMD 的 soft 版）。
- **T5 [x] 训练脚本**：`protein_jepa_frame/training/train_stage_ensemble.py`（NEW 文件）：
  - 冻结 relpos encoder + anchored decoder；只训练 `GaussianHHead` + EnsembleLoss。
  - **NaN-backward 已修**：`evaluation/ensemble_metrics.py::rmsf_profile` 的
    `sqrt` 在方差=0（masked 残基）处梯度无穷 → 改为 `sqrt(x.clamp_min(1e-12))`；
    `pair_distance_hist` 距离改 `sqrt(d²+1e-12)`。修复后 forward/backward 均有限。
  - 40-step smoke 通过；正式 4000-step 运行见 `ENSEMBLE_RESULTS_20261018.md`。
  - **首轮 val（step500）已超 persistence**：val total 2.949 vs persist 3.540，
    其中 **RMSF 0.865 vs 1.512（好 43%）**；contact/dist ≈ persistence。
- **T6 [x] 评估**：`scripts/eval_ensemble_manifest.py`（NEW），跑 mdCATH val 与 ATLAS val，
  model / persistence / oracle 三行 + bootstrap 95% CI。
  - **mdCATH val：total +0.728 [0.63,0.82]、RMSF +0.835 [0.74,0.94] → 显著优于 persistence（GO）**；
    contact/dist 略差。
  - **ATLAS val：全部指标显著变差**（model 全面输 persistence）→ 不跨数据集泛化。
- **T7 [x] 结果文档**：`ENSEMBLE_RESULTS_20261018.md`（含 mdCATH/ATLAS 对照表 + 结论）。

---

## 6. 验收标准（go / no-go）

1. **GO**：在 ≥1 个主指标（RMSF 相关 或 contact Jaccard 或 VAMP-2）上，
   系综训练的 JEPA **显著优于 persistence**（95% CI 不跨 0）**且 ≥ SPIB baseline**。
2. 若仅 ≈ persistence → 说明系综统计量也被单帧 H 限制，回到 §9 结论：
   需要**真多帧 anchor / 亚 ns 数据 / 更长历史**（写进本 md 的 "负结果" 段）。
3. 所有报告必须带 `persist` 与 `oracle` 两端锚点，禁止无对照地报单一数字。

---

## 7. 风险与依赖

- **可微性**：EMD/接触图需 soft 版本（sinkhorn / sigmoid 软化），否则梯度断。
- **VAMP-2 在冻结 encoder 下**：需用 latent 轨迹重算，勿与 6-state baseline 混比维度（见 comparison_report 注释）。
- **温度**：不同温度动力学差异大，须分层，勿混池。
- **计算**：8×A800，GPU 5 常空；rollout 多采样开销大，先小 N(=8) 探路。

---

## 8. 与已有工作的衔接

- 复用：`baselines/{tica_msm,vampnets,spib}.py`、`baselines/metrics.py`（VAMP-2/Brier）、
  `evaluation/geometry.py`（`contact_jaccard`/`displacement_msd`/`aligned_rmsd`）、
  `evaluation/variants.py`（`h_plus_s` rollout → 系综样本）。
- 复用产物：`artifacts/stage_a_disp_relpos/`（encoder）、`artifacts/stage_b1a_relpos_anchored/`（decoder）。
