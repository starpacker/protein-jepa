# Stage B "H+S (GLE) iteration" on top of Model 1 — 结果（2026-10-21）

> **问题**：在**当前最强的 Model 1**（1000dom，静态 Gaussian 系综头）之上，
> 把静态采样换成 **H+S（GLE bath）随机迭代**，能否**超过 Model 1**？
>
> **一句话答案**：**不能**（配对检验下 Model 1 在 RMSF-r 等主指标上仍胜；
> H+S 只在 transient-contact Jaccard 与 rollout RMSF 上略胜）。但这是一个**干净的负结果**，
> 且 H+S 在若干"分布展开"指标上确实更接近真值——见下。

---

## 1. 方法（与 Model 1 的唯一区别）

| 组件 | Model 1 | **本实验（H+S）** |
| --- | --- | --- |
| Stage-A encoder | 冻结 | **同（复用）** |
| B1a anchored decoder | 冻结 | **同（复用）** |
| latent 采样 | `H = mu + sigma*eps`（静态高斯头） | **`StageB1b` GLE：从 `H_anchor` 自主演化 `rollout_steps` 步** |
| 可训练参数 | GaussianHHead | **GLE kernel（H-space energy U(H) + per-residue bath S）** |
| 损失 | RMSF-L1 + contact-L1 + dist-EMD | **完全同（复用 `compute_ensemble_loss`）** |
| 数据 | 1000dom train/val | **同** |

- 新驱动：`protein_jepa_frame/training/train_stage_b_hs.py`（`HSEnsembleProducer`）。
- **未改动**精确坐标解码路径；Model 1 ckpt 只读。
- 训练：`artifacts/stage_hs_1000dom/checkpoint_best.pt`（6000 步，GPU 3，best val ens/total **1.9583**）。
- 评估：扩展 `eval_ensemble_metrics_suite.py` / `eval_ours_trajectory.py` 支持 `--kernel-ckpt`；
  另有配对检验脚本 `scripts/paired_hs_vs_model1.py`。

## 2. 关键结果 —— **配对检验（同一批窗口，最严格的判据）**

`n=200` 窗口，n_samples=8，bootstrap 2000 次 95% CI。Δ = H+S − Model1。

| 指标 | Model 1 | H+S | Δ | 95% CI | 显著? | 胜者 |
| --- | --- | --- | --- | --- | --- | --- |
| RMSF 幅度 ↓ | 0.8398 | **0.8310** | −0.009 | [−0.028, +0.010] | 否 | 平（略偏 H+S）|
| **RMSF-r ↑** | **0.5151** | 0.4957 | −0.019 | [−0.036, −0.002] | **是** | **Model 1** |
| ROOT-MEAN W2 ↓ | **11.071** | 11.083 | +0.012 | [+0.003, +0.020] | **是** | **Model 1** |
| VARIANCE W2 ↓ | **8.522** | 8.737 | +0.215 | [+0.125, +0.308] | **是** | **Model 1** |
| MD-PCA W2 ↓ | **0.0701** | 0.0734 | +0.0033 | [+0.0026, +0.0041] | **是** | **Model 1** |
| WEAK-CONTACT J ↑ | 0.0869 | 0.0873 | +0.0004 | [−0.003, +0.004] | 否 | 平 |
| **TRANSIENT J ↑** | 0.2039 | **0.2141** | +0.0102 | [+0.006, +0.015] | **是** | **H+S** |
| RMSF-r(逐窗 profile) ↑ | — | — | −0.019 | [−0.034, −0.004] | **是** | **Model 1** |

**结论**：**Model 1 在 4 个指标上显著更好**（RMSF-r、root/var/pca W2），**H+S 只在 transient-contact Jaccard 上显著更好**。

## 3. rollout（Kabsch）对比

| | RMSF↓ | RMSF-r↑ | pca_W2↓ | **Kabsch↓** |
| --- | --- | --- | --- | --- |
| Model 1 | 2.197 | 0.363 | 0.067 | **8.39** |
| H+S | **2.046** | **0.429** | **0.064** | 8.74 |
| persistence | — | — | — | 3.87 |

H+S 在 rollout 的 RMSF/RMSF-r 上**更好**，Kabsch 略差；两者都远不及 persistence 的 Kabsch（已知 rollout 漂移）。

## 4. 为什么 H+S 没赢？（诚实分析）

1. **Model 1 已很强**：它用同一条冻结解码路径，把 RMSF 谱/位置方差/pca 拟得很准。
   GLE 的随机迭代**引入了额外的动力学噪声**，反而**抬高了 variance_w2 / pca_W2**（分布过宽）。
2. **静态高斯头其实是"对的归纳偏置"**：mdCATH 1 ns/帧下，从 anchor 出发的**系综**更接近
   "围绕 anchor 的静态高斯云"，而非"沿时间扩散的轨迹"——这与 bioemu2 短时标输给 persistence
   是同一个现象（§ `PROJECT_REPORT_20261021.md` §3.5）。
3. **H+S 的优势在"接触事件"**：transient-contact Jaccard 显著更高，说明 GLE 更能生成
   "偶发成键/断键"的构象，这对**瞬态接触**类指标有利。

## 5. 一个重要的方法学教训（已在评估里修正）

- 两次**独立** `eval_ensemble_metrics_suite.py` 运行（Model 1 vs H+S）会因
  `build_dataset` 的窗口顺序/子集不同而**用到不同窗口**（实测 target 交集仅 77/90），
  导致**不可比的池化 global RMSF-r**（H+S 0.741 "看着更高" 其实是采样差异）。
- **必须用配对检验**（同批同窗，`scripts/paired_hs_vs_model1.py`）。配对结果才推翻了
  "H+S 更好"的假象。

## 6. 交付物

| 类别 | 路径 |
| --- | --- |
| H+S 训练驱动 | `protein_jepa_frame/training/train_stage_b_hs.py` |
| H+S ckpt | `artifacts/stage_hs_1000dom/checkpoint_best.pt` |
| 配对检验脚本 | `scripts/paired_hs_vs_model1.py` |
| 配对结果 | `artifacts/eval_hs_1000dom/paired_vs_model1.json` |
| H+S 系综指标 | `artifacts/eval_hs_1000dom/suite.json` |
| H+S rollout 指标 | `artifacts/eval_hs_1000dom/rollout_100f.json` |
| 运行脚本 | `run_stage_hs_1000dom.sh` |

## 7. 下一步建议（若想继续）

- **别只走 1 步**：当前 `rollout_steps=1`。可试 `rollout_steps` ∈ {2,4,8}，让 GLE 真正"迭代"；
  但要防分布过宽（同时调低 diffusion/或加 variance floor）。
- **混合方案**：保留静态头为主 + GLE 只负责"接触事件"（residual ensemble），取两者之长。
- **调 loss**：H+S 抬高 variance_w2 → 加 `--variance-contr` / 降低 `contact_weight`。
