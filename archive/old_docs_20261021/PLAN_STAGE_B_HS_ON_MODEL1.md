# Plan — Stage B "H+S (GLE) 迭代" on top of Model 1

> 目标：拿 **当前最强 Model 1**（1000dom，RMSF-r global **0.673**、RMSF **0.837**，
> 静态 Gaussian 系综头）做基座，**在它之上加 H+S（GLE bath）随机迭代**，
> 看能否**超过 Model 1**。直接复用现有 `StageB1b` GLE kernel + `train_stage_ensemble`
> 的系综损失 + `ensemble_producer` 的冻结解码路径。

## 0. 动机与关键假设

- Model 1 的系综头是 **静态** 的：`H^{(n)} = mu + sigma*eps_n`（i.i.d. 高斯，**无时间演化**）。
  它能拟分布形状（RMSF 谱），但**不能显式表达"沿时间推进"的动力学**。
- Model 2（from-none GLE）证明 **GLE + bath S 的 latent 随机演化**可训练且能拟 RMSF-r（~0.5）；
  但它用的是**随机 encoder**，表征弱。
- **本实验 = 强表征（Model 1 冻结 encoder/decoder）+ GLE H+S 迭代**，两全其美？
  - 关键风险（已有诊断）：**冻结 Stage-A 的 ΔH 对 CA 位移不可解码**（`B1B_DIAGNOSIS`）。
    但那是"**同一帧 descriptor 的差分**"；本实验是"**在 H 空间加 GLE 噪声再用同一 decoder 解码**"，
    与 Model 1 的高斯头**同一条解码路径**（同 decoder、同 anchor basis），因此**没有那个不可解码问题**
    ——Model 1 已经用这条路拿到 0.673，只是采样是静态的。

## 1. 方法（与 Model 1 的唯一区别）

| | Model 1（现状） | **本实验（Model 1 + GLE）** |
|---|---|---|
| encoder | 冻结 Stage-A | 同（复用） |
| decoder | 冻结 B1a | 同（复用） |
| latent 采样 | `H = mu + sigma*eps`（静态高斯头） | **GLE 从 anchor H 出发自主演化 M 步 → H_τ 轨迹** |
| 可训练参数 | GaussianHHead | **StageB1b GLE kernel（能量 U(H) + bath S 参数）** |
| 解码 | 同 decoder | 同 decoder |
| 损失 | RMSF L1 + contact L1 + dist EMD | **同（完全复用 `compute_ensemble_loss`）** |

**GLE 起点**：`H_anchor = Encoder(anchor frame)`（冻结），与 `H_anchor` 的差 `ΔH_τ` 作为状态。
**GLE 采样**：用 `StageB1b.rollout()` 从 `H_anchor` 出发演化 `horizon_steps` 步，得 `N` 条 H 轨迹；
每条轨迹的**末帧 H**（或全轨迹）→ 冻结 decoder → CA 云 → 系综损失。

## 2. 实现步骤

- [x] **S1** 新训练驱动 `protein_jepa_frame/training/train_stage_b_hs.py`
      （**新文件**，不改 Model 1 的 `train_stage_ensemble.py`；旧精确解码路径不动）
- [x] **S2** 复用冻结组件：Stage-A encoder + B1a decoder（`EnsembleProducer` 的加载逻辑）
- [x] **S3** 复用 `StageB1b` GLE kernel（能量 + bath + `rollout`），trainable
- [x] **S4** 复用 `compute_ensemble_loss`（RMSF/contact/dist），与原 Model 1 **同损失同权重**
- [x] **S5**  — 冒烟通过（20 步，val ens/total 2.35→2.21 已优于 persistence）；`train_stage_b_hs.py` 冒烟测试（--smoke）：1 GPU，跑通前向/反向/评估
- [x] **S6** 正式训练完成（GPU 3，6000 步，best val ens/total **1.9583**）→ `artifacts/stage_hs_1000dom/` 正式训练：1000dom train，GPU 3，与 Model 1 同口径
- [x] **S7** 同口径评估（`eval_ensemble_metrics_suite.py`），与 Model 1 **直接对比**
- [x] **S8** 轨迹 PDB 已导出（`artifacts/pdbs_hs_vs_model1_20261021/`，含 ours_hs vs ours_stageA）；报告 `STAGE_B_HS_RESULTS_20261021.md`

## 3. 验收标准 与 结论

- **主判据**：RMSF-r 是否显著 > Model 1。
- **结论（配对检验，n=200）**：❌ **未通过**。Model 1 在 **RMSF-r（0.515 vs 0.496，显著）**、
  variance_w2、md_pca_w2、root_mean_w2 上更好；**H+S 仅在 transient-contact Jaccard
  （0.214 vs 0.204，显著）**与 rollout RMSF/RMSF-r 上更好。
- **这是干净负结果**：说明在 mdCATH 1ns/帧、anchor 系综任务上，**静态高斯头是更合适的归纳偏置**，
  GLE 的额外随机迭代反而使分布过宽（variance/pca W2 升高）。
- 完整结果见 `STAGE_B_HS_RESULTS_20261021.md`。

## 4. 关键复用文件（不改动）

- `protein_jepa_frame/models/stage_b1b/model.py`（GLE kernel）
- `protein_jepa_frame/models/ensemble_producer.py`（冻结解码封装）
- `protein_jepa_frame/training/ensemble_losses.py`（系综损失）
- `protein_jepa_frame/training/train_stage_ensemble.py`（参考驱动的数据/评估循环）
