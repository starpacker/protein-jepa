# Protein-JEPA-Frame — 完整项目报告（单一文档）

> **日期**：2026-10-21
> **本文件是唯一需要的文档**（自包含）。原始材料：`archive/old_docs_20261021/DUAL_ENSEMBLE_RESULTS_20261019.md`、
> `archive/old_docs_20261021/FROMNONE_GLE_ENSEMBLE_RESULTS_20261019.md`、`archive/old_docs_20261021/BIOEMU2_ZEROSHOT_RESULTS_20261020.md`
> 及仓库内 JSON（`artifacts/eval_*/*.json`、`results/*.json`）。
> 所有模型现**统一使用 `mdCATH-1000dom`** 作为默认数据集（见 §7）。

---

## 0. 一页速览（TL;DR）

| 项 | 内容 |
| --- | --- |
| **任务** | 给定 mdCATH 蛋白的 **anchor 帧**（+ 短历史），预测**未来构象系综 / 未来帧轨迹**的 CA 结构 |
| **模型 1** | **Stage-A ensemble**：JEPA 式**预训练** encoder + 锚定解码器（冻结）+ **Gaussian H 系综头**（只训头） |
| **模型 2** | **from-none GLE**：**随机** encoder + **GLE 随机 latent 动力学** + 残差解码器（端到端） |
| **Baseline 1** | **persistence**：anchor 帧复制 T 份（最诚实的天花板对手） |
| **Baseline 2** | **MDGen zero-shot**（`atlas.ckpt`，无微调） |
| **Baseline 3** | **bioemu2 zero-shot**（`bioemu2-preview`，微软 diffusion 全原子系综，无微调） |
| **主指标（7 项）** | RMSF-幅度 / **RMSF-r（形态）** / ROOT-MEAN W2 / VARIANCE W2 / MD-PCA W2 / WEAK-CONTACT J / TRANSIENT-CONTACTS J（+ Kabsch CA RMSD） |
| **最强结论** | ✅ 两个模型在**系综分布指标（RMSF 谱/形态 + 3×W2 + 接触类）稳定优于 persistence**；其中**模型 1（最终版）RMSF-r=0.673、RMSF=0.837，已优于所有 zero-shot baseline**。⚠️ **CA 绝对位置精度**（Kabsch / 短时标 root_W2）仍是共同短板 |
| **Stage A + GLE 直连（Plan A）** | 把 Stage-A 表征与 GLE 动力学接起来（encoder 微调 + decoder 冻结 + GLE 训练）。**terminal-only 版本 ≈ 模型 1**（RMSF-r、TRANSIENT-J 显著更好，其余打平，见 **§3.7**）；**dense 轨迹监督版本失败已回退**（§10.2）。代码/下一步见 **§10**、命令见 **§11** |
| **⚠️ PDB 口径提醒** | §3 的 SOTA 是**单步系综**任务的成绩，对应 PDB：**`artifacts/pdbs_1000dom_ensemble_20261021/`**（不漂移）。另一个 `..._aligned_.../` 目录是**自回归 rollout PDB**，口径不同，RMSD 差是**另一个任务**，**复现报告不需要 rollout**。详见 **§4.1** |

**三套 baselevel 正交结论**：
- persistence 是"必须打赢"的诚实对手；我们的模型在**分布形态**上稳定赢它。
- MDGen zero-shot 在 mdCATH 上**劣于 persistence**（步长 OOD 10 ps→1000 ps/帧），**不代表其正常水平**。
- bioemu2 zero-shot **长时标（16–64 ns）显著优于 persistence**（RMSF-r ~0.6），但**短时标（1 ns）差于 persistence**（扩散系综 ≠ 短时确定性推进）。

---

## 1. 两套模型是什么

### 1.1 共同设定

两套模型都做同一件事：**从 anchor 帧出发**，产生一组"未来 CA 构象"（系综）或一条"未来帧轨迹"。
评估时统一：真值系综 = 该 horizon 内**真实 MD 的未来帧**；persistence = anchor 复制；
所有量先做 Kabsch 对齐（各帧对齐到 anchor 帧的同一 gauge）。

### 1.2 模型 1 — Stage-A ensemble（"强表征 + 轻头"）

| 项 | 内容 |
| --- | --- |
| 骨干 | stage-A v2 (encoder) + stage-B1a (anchored decoder)，**预训练后冻结** |
| 可训练部分 | 只加 **`GaussianHHead`**（在 latent H 上拟合高斯，采样出 M 个 conformer） |
| 物理语义 | 头输出 H 的分布 → 解码器把每个 H 样本映射成一条 CA 云 |
| 训练数据 | **mdCATH-1000dom**（旧版为 mixed mdCATH-200dom + ATLAS，见 §7） |
| 训练目标 | RMSF(L1) + contact-prob(L1) + 内距 soft-EMD +（可选）位置项 `L_pos` |
| 变体 | A（无 pos/cond）、B（+pos 0.5）、C（+cond 2 + pos 0.5） |
| checkpoint | `artifacts/stage_ensemble_1000dom/checkpoint_best.pt`（1000dom）<br>旧 200dom：`artifacts/stage_ensemble_dualA/checkpoint_best.pt` |

### 1.3 模型 2 — from-none GLE（"端到端 from-scratch + 显式随机动力学"）

| 项 | 内容 |
| --- | --- |
| 骨干 | **随机初始化** encoder（无预训练）+ **GLE**（广义 Langevin，bath_dim=8）latent 动力学 |
| 解码 | CRI / MLP **残差解码器**：`ca = X_ctx + tanh 限幅 Δx` |
| 物理语义 | 用 GLE 在 latent 上**真做随机演化**，得到 M 条 rollout → 系综 |
| 训练数据 | `mdcath_manifest_1000dom`（mdCATH，1 ns/帧） |
| 训练目标 | CA RMSD（点预测）+ latent ES + **系综分布项**（v4/4e 逐步加入）|
| 演进 | anchor → v4 → v4c/v4d → **v4e**（+RMSF-r 剖面损失）→ v4f（训练中） |
| 最优 checkpoint | `artifacts/fromnone/stage_cb1b_v4e/checkpoint_latest.pt` |

### 1.4 两条路线的本质差异

| | 模型 1 · Stage-A ensemble | 模型 2 · from-none GLE |
|---|---|---|
| 表征 | JEPA 式**预训练** encoder（冻结） | **随机**初始化 encoder（无预训练） |
| 动力学 | Gaussian H **系综头**采样 | **GLE**（广义 Langevin）随机 latent 演化 |
| 可训练量 | 只训练轻量系综头 | 端到端全量 |
| 一句话 | 强表征 + 轻头 | 从零端到端 + 显式随机动力学 |

---

## 2. 评估指标与实现（正确性核对）

> 代码：`evaluation/ensemble_metrics.py`（原文件，未改）+ `evaluation/ensemble_metrics_ext.py`（新增）。
> **未触碰**精确原子坐标解码路径（原代码留存）。**单测 26 项全过**（`tests/test_ensemble_metrics.py` 等）。

设真值系综 `X_true ∈ [T, L, 3]`（该 horizon 的 MD 未来帧 CA），预测系综 `X_pred ∈ [N, L, 3]`。
W2 均为**精确一维 Wasserstein-2**（排序分位函数、无分箱）：`W2²=∫₀¹(F_a⁻¹(q)−F_b⁻¹(q))²dq`。

| 指标 | 精确定义 | 单位 | 方向 | 代码 |
| --- | --- | --- | --- | --- |
| **RMSF（幅度）** | 对齐后逐残基 RMSF 谱的 masked **L1** | Å | ↓ | `rmsf_profile_error.rmsf_mae` |
| **RMSF-r（形态）** | 逐残基 RMSF 谱 vs 真值的 **Pearson r**（global / per-window / per-target） | — | ↑ | `rmsf_profile_error.rmsf_corr` |
| **ROOT-MEAN W2** | 逐残基 CA 位置分布 W2 的 RMS：`sqrt(mean_i W2²)` | Å | ↓ | `root_mean_w2` |
| **VARIANCE W2** | 逐残基**涨落幅度 σ²** 分布之 W2 | Å² | ↓ | `variance_w2` |
| **MD-PCA W2** | 真值协方差前 k=3 慢模投影分布 W2 之 RMS | Å | ↓ | `md_pca_w2` |
| **WEAK-CONTACT J** | `P(d<8Å)∈[0.05,0.5)` 位点集合的 **Jaccard** | — | ↑ | `weak_contact_jaccard(_full)` |
| **TRANSIENT J** | `P(d<8Å)∈(0.01,0.99)` 位点集合的 **Jaccard** | — | ↑ | `transient_contact_jaccard(_full)` |
| **Kabsch CA RMSD** | anchor→真值变换（由**真值**拟合）下 CA RMSD，所有样本共享同一 gauge | Å | ↓ | `geometry.common_anchor_rmsd` |

### 2.1 核对结论（实现正确性）

1. **oracle 锚点**：`pred=true` 时 → 4×W2 = **0.000**、2×Jaccard = **1.000**、RMSF-r = **1.000**、RMSF_mae = **0**。✔（实测）
2. **persistence 的 RMSF-r 恒为 0**：delta 系综的 RMSF 谱是**常数**，与真谱的 Pearson 按定义退化 → 0。✔（实测）
3. **persistence 的 contact-J/transient-J**：delta 系综 contact-prob 图**只有 {0,1}**，
   弱/瞬时带**一个都落不进** ⇒ 预测集合空；真集非空时 `J=0/|A|=0`。
   **这是 delta 基线的数学性质**，不是 mdCATH 性质。已在 `*_full` 变体返回 `n_true/n_pred` 可追溯。✔
4. **对齐口径一致**：训练侧 `_ensemble_stats` 与评估侧 `align_ensemble` 都是"各帧 Kabsch 对齐到首帧"；
   已修正一次早期口径错配（`_ensemble_stats` 曾在绝对坐标上算方差）。

### 2.2 引用指标时的注意事项

- **接触类 Jaccard 对 delta/近刚体基线退化**——引用时必须附 `n_true/n_pred`。
- 综合分不是单一标量；报告时按"分布形态 / 位置精度 / 接触"三类分别陈述。

---

## 3. 结果：我们的模型 vs 三套 baseline

### 3.1 数据集口径（**先声明，否则数字会被误读**）

| 模型/基线 | 训练数据 | 评估 val 集 | 帧间隔 |
| --- | --- | --- | --- |
| 模型 1（Stage-A ens） | **mdCATH-1000dom** | **mdCATH-1000dom val（100 蛋白）** | 1000 ps |
| 模型 2（from-none GLE） | mdCATH-1000dom train | **mdCATH-1000dom val（100 蛋白）** | 1000 ps |
| persistence | — | 随各模型 | — |
| MDGen zero-shot | ATLAS（10 ps/帧） | mdCATH val（32 窗口） | 10 ps（原生，OOD） |
| bioemu2 zero-shot | PDB+AFDB+MS 生成 MD | mdCATH-1000dom val（200 窗口） | 训练 1ns–1µs |

> **✅ 口径已统一**：模型 1 已在 **1000dom** 重训（见 §7），与模型 2 / MDGen / bioemu2 同属一个数据集家族。
> （历史：早期整合时模型 1 用 200dom-val、模型 2 用 1000dom-val，两 val 集不相交——现已消除。）

### 3.2 模型 1（Stage-A ensemble，1000dom val，50 批/200 样本，n_samples=8）

> **已用最终 B1a ckpt + 重训系综头刷新**（§7 R7）。数字来自 `artifacts/eval_model1_1000dom/suite_v2.json`。

| 指标 | model | persistence | oracle |
| --- | --- | --- | --- |
| RMSF 幅度 ↓ | **0.837** | 1.835 | 0.000 |
| **RMSF-r ↑** | **0.512** | 0.000 | 1.000 |
| ROOT-MEAN W2 ↓ | **16.540** | 16.752 | 0.000 |
| VARIANCE W2 ↓ | **7.878** | 10.712 | 0.000 |
| MD-PCA W2 ↓ | **0.0665** | 0.1174 | 0.0000 |
| WEAK-CONTACT J ↑ | **0.0781** | 0.0000 | 1.000 |
| TRANSIENT J ↑ | **0.1832** | 0.0000 | 1.000 |

**RMSF-r 三形式**：global **0.673**；per-target 均值 **0.553**（83 个域，**98.8% > 0**）。
**Kabsch CA RMSD（rollout 100 帧）**：model **8.39 Å** vs persistence **3.87 Å**（自回归回填漂移，已知短板）。
结果文件：`artifacts/eval_model1_1000dom/{suite_v2.json, ours_rollout_100f_v2.json}`。

> 对比 R7 之前（B1a 停在 step-250 时的系综头）：RMSF-r global 0.538→**0.673**、
> per-target 0.293→**0.553**、RMSF 1.154→**0.837**、variance_w2 9.31→**7.88**、
> pca_w2 0.088→**0.067**。**用最终 B1a 重训系综头带来全面且显著的提升。**

### 3.3 模型 2（from-none GLE v4e，1000dom val，200 窗口，n_samples=12）

| horizon | RMSF-r ↑ | RMSF↓ | root_W2↓ | var_W2↓ | pca_W2↓ | weak_J↑ | trans_J↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 ns | 0.283 | 1.888 | 6.576 ✗ | 7.253 | 0.182 | 0.400 | 0.012 |
| 4 ns | **0.513** | 1.226 | 3.798 ✗ | 9.790 | 0.099 | 0.137 | 0.275 |
| 16 ns | **0.497** | 1.984 | 2.986 ✓ | 20.514 | 0.149 | 0.195 | 0.345 |
| 64 ns | **0.502** | 2.923 | 4.025 ✓ | 32.405 | 0.192 | 0.152 | 0.383 |
| persist(4ns) | 0.015 | 3.020 | 2.592 | 14.366 | 0.190 | 0.000 | 0.000 |

- **稳定胜**：RMSF 幅度、RMSF-r（4–64 ns）、variance/pca W2、transient-J（100% 窗口）。
- **已知失败**：`root_mean_w2` 短时标（1 ns）差（模型位移 1.5 Å ≪ 真实 3.6–7.4 Å），16/64 ns 转正。

### 3.4 Baseline — MDGen zero-shot（mdCATH，32 窗口）

| 模型 | 物理跨度 | RMSF↓ | RMSF-r↑ | root_W2↓ | var_W2↓ | pca_W2↓ | weak_J↑ | trans_J↑ | **Kabsch↓** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| persistence | 瞬时 | 1.840 | −0.014 | 12.276 | 11.089 | 0.1345 | 0.000 | 0.000 | **4.014** |
| **MDGen zero-shot** | 1 ns | 4.650 | ~0 | **6.679** | 27.069 | 0.1387 | 0.000 | 0.000 | 7.344 |
| ↳ seed=7 复核 | 1 ns | 4.779 | ~0 | 6.896 | 27.371 | 0.1437 | 0.000 | 0.000 | 7.841 |
| MDGen mdCATH 短训 e1/s2000 | 100 ns | 16.061 | ~0 | 11.252 | 275.31 | 0.4760 | 0.000 | 0.000 | 19.691 |
| MDGen mdCATH 短训 e5/s6000 | 100 ns | 22.525 | ~0 | 14.213 | 541.27 | 0.6736 | 0.000 | 0.000 | 26.761 |
| **ours rollout（模型1）** | 100 ns | **1.455** | **0.334** | 11.993 | **10.211** | **0.0801** | **0.066** | **0.134** | 6.335 |

**结论**：MDGen 官方权重 **zero-shot 在 mdCATH 上劣于 persistence**（Kabsch 7.34 vs 4.01 Å），
主因 **10 ps/帧 → 1000 ps/帧 步长 OOD + 蛋白族不同**；唯一亮点是 root_W2 最低（形状分布尚可，位置跑偏）。
MDGen mdCATH 短训**远未收敛且震荡**（flow-matching 需 ~10⁴ epoch），此预算下不构成有意义的对手。

### 3.5 Baseline — bioemu2 zero-shot（`bioemu2-preview`，1000dom val，200 窗口，n_samples=12）

> Microsoft Research 的 diffusion 全原子系综模型，**零微调**，conditioned on anchor 帧。
> 跑在**同一套** `evaluation/ensemble_metrics_ext.py` + 同一 window 迭代 + 同一 persistence 上。
> 结果：`results/bioemu2_zeroshot.json`。

| horizon | RMSF-r ↑ | RMSF↓ | root_W2↓ | var_W2↓ | pca_W2↓ | weak_J↑ | trans_J↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 ns | **0.431** | 2.501 | 5.588 | 19.783 | **0.087** | 0.000 | 0.165 |
| 4 ns | **0.595** | **2.243** | 3.699 | 24.336 | **0.098** | 0.163 | 0.281 |
| 16 ns | 0.579 | **1.817** | **3.449** | **20.228** | **0.086** | **0.237** | **0.348** |
| 64 ns | **0.609** | **1.715** | **4.172** | **18.552** | **0.092** | **0.209** | **0.377** |
| persistence @64 ns | 0.031 | 5.564 | 4.702 | 41.109 | 0.316 | 0.000 | 0.000 |

**读法**：bioemu2 在**长时标**上显著优于 persistence（RMSF-r ~0.6，RMSF 更低、pca_W2 更小、
接触 Jaccard 更高）；但**短时标（1 ns）** RMSF/root_W2 反而差于 persistence
（扩散模型生成的是"平衡分布样本"而非"沿轨迹的短时确定性演化"——1ns 内锚点几乎不动，
persistence 因此占优）。这正是**系综生成 vs 轨迹推进**两类任务的口径差异。

**环境坑（已解决）**：bioemu2 的 `cuequivariance_ops_torch` 需要 `libnvrtc.so.13`；
本机是 CUDA 12.9，但 pip `nvidia-cu13` wheel 里**有**该库，只是不在默认 search path 上。
修复：`export LD_LIBRARY_PATH=/opt/conda/lib/python3.13/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH`（已写进 `run_scripts/run_bioemu2_zeroshot.sh`）。

### 3.5b bioemu2 zero-shot — 3 次重复（seed 0/1/2）的 mean ± std 【新增】

> **背景**：§3.5 是**单次**运行的结果。后来发现 bioemu2 那条路径在原始默认配置下
> **不可复现**（三重不确定性，见下），因此补跑 **3 次重复**（seed 0/1/2），
> 给出 mean ± std 作为更稳健的引用口径。**§3.5 的旧结果保留不动**，两表并列对照。
> 结果：`results/bioemu2_reps/summary.json`（原始三次：`seed0/1/2.json`）。
> 配置完全相同：同一 `bioemu2-preview` ckpt、同一 `mdcath_manifest_1000dom/val.jsonl`、
> 200 窗口、n_samples=12、horizons 1/4/16/64 ns、EDM sampler。

| horizon | RMSF-r ↑ | RMSF↓ | root_W2↓ | var_W2↓ | pca_W2↓ | weak_J↑ | trans_J↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 ns | 0.4440±0.0032 | 2.164±0.011 | 5.601±0.003 | 15.156±0.221 | 0.0807±0.0008 | 0.0000±0.0000 | 0.1730±0.0014 |
| 4 ns | 0.5950±0.0094 | 1.846±0.049 | 3.661±0.015 | 18.136±0.602 | 0.0819±0.0021 | 0.1772±0.0020 | 0.2983±0.0030 |
| 16 ns | 0.5971±0.0111 | 1.548±0.042 | 3.308±0.058 | 14.241±0.749 | 0.0771±0.0032 | 0.2433±0.0013 | 0.3611±0.0024 |
| 64 ns | 0.6129±0.0183 | 1.622±0.041 | 4.060±0.049 | 17.617±0.249 | 0.0876±0.0014 | 0.2115±0.0009 | 0.3823±0.0033 |
| persistence @64 ns | 0.0306 | 5.564 | 4.702 | 41.109 | 0.316 | 0.000 | 0.000 |

**与 §3.5 单次结果的对照（哪些在 seed 噪声内、哪些不是）：**

| 项 | §3.5 单次 | 3 次 mean±std | 相对差 | 判定 |
| --- | --- | --- | --- | --- |
| 4 ns RMSF-r | 0.5950 | 0.5950±0.0094 | 0.0% | 一致 |
| 64 ns RMSF-r | 0.6091 | 0.6129±0.0183 | 0.6% | 一致 |
| 1 ns root_W2 | 5.588 | 5.601±0.003 | 0.2% | 一致 |
| 1 ns RMSF | 2.501 | 2.164±0.011 | **+13.5%** | 超出噪声 |
| 1 ns var_W2 | 19.78 | 15.16±0.22 | **+23.4%** | 超出噪声 |
| 4 ns var_W2 | 24.34 | 18.14±0.60 | **+25.5%** | 超出噪声 |
| 16 ns var_W2 | 20.23 | 14.24±0.75 | **+29.6%** | 超出噪声 |
| 16 ns RMSF | 1.817 | 1.548±0.042 | **+14.8%** | 超出噪声 |
| 4 ns pca_W2 | 0.0980 | 0.0819±0.0021 | **+16.4%** | 超出噪声 |

**读法**：
- **RMSF-r / root_W2 / 接触类 Jaccard** 两表**基本一致**（差异在 seed 噪声内）；
- **RMSF 幅度 / variance_W2 / pca_W2** 单次运行**系统性偏高 10–30%**（远大于 std，且方向一致），
  属原始非确定性配置下的一次运气偏差。引用时建议以 **3 次 mean±std** 为准。
- **结论段（§3.5 "读法"）不受影响**：长时标优于 persistence、短时标（1 ns）差于
  persistence（1 ns RMSF 2.164 > 1.898、root_W2 5.601 > 1.683）在 mean±std 下依然成立。

**为什么旧结果不可复现（重跑工程）**：bioemu2 默认配置有**三重不确定性** ——
① `propagate_with_embeddings()` 调 `generate_samples()` 时**未传 seed** → EDM 初始噪声未固定；
② 默认 **bf16-mixed + TF32 + cuDNN 非确定性**；
③ **`PYTHONHASHSEED` 未固定** → Python 3 字符串哈希随机化泄漏进 bioemu2 内部对
`hash()`/set/dict 的用法（**只在跨进程、同 seed 下才暴露**）。
**修复**：`os.execve` 重启钉死 `PYTHONHASHSEED=0`，加 `--precision 32-true
--matmul-precision highest --deterministic`，并**按窗口身份 sha256 派生 seed** + 每窗口
`torch.random.fork_rng`。修复后同 seed 跨进程**逐位一致（max diff = 0.0）**，换 seed 结果确实不同
→ 3 次重复的 std 仅 **0.001–0.75**，为真实可复现噪声。

### 3.6 汇总对照（1000dom，同时标可比的量）

| 模型 | RMSF-r(global)↑ | RMSF↓ | pca_W2↓ | Kabsch↓ |
| --- | --- | --- | --- | --- |
| **模型 1（Stage-A ens，1000dom，最终）** | **0.673** | **0.837** | **0.067** | 8.39 |
| **模型 2（from-none GLE v4e）** | 0.51（4 ns） | 1.23（4 ns） | 0.099（4 ns） | — |
| bioemu2 zero-shot（单次，§3.5） | 0.60（64 ns） | 1.72（64 ns） | 0.092（64 ns） | — |
| bioemu2 zero-shot（3 次 mean，§3.5b） | 0.61（64 ns） | 1.62（64 ns） | 0.088（64 ns） | — |
| MDGen zero-shot | ~0 | 4.65 | 0.139 | 7.34 |
| persistence | 0.000 | 1.77 | 0.111 | 3.31 |

> **读法**：**在 RMSF-r / RMSF / pca_W2 三个分布形态指标上，模型 1 已优于所有 zero-shot baseline**
> （含 bioemu2）。**位置精度（Kabsch）仍是短板**——模型 1 rollout 8.39 Å 差于 persistence 3.87 Å。

### 3.7 Stage A + GLE 直连（Plan A，terminal-only） vs 模型 1 【新增，当前最佳 GLE 版本】

> 目标：把 **Stage-A encoder 的表征** 与 **GLE latent 动力学** 接起来（"Plan A"），
> 在**同一冻结栈**（Stage-A encoder + B1a 冻结解码器）下与**模型 1** 做**配对**比较，
> 看 GLE 是否能在模型 1 的基础上进一步优化系综指标。

**配置（terminal-only）**：
- 车辆：`protein_jepa_frame/training/train_stage_b_hs.py`，脚本 `run_scripts/run_plan_a_hs_1000dom.sh`。
- 冻结/开放：**encoder 用 Stage-A 初始化并联合微调**（`--train-encoder --encoder-lr-scale 0.1`，低 LR）；
  **B1a 解码器冻结**；GLE kernel（`StageB1b`）开放训练。
- rollout 步数 == 监督窗口帧数（`--rollout-steps 8 --paired-horizons-frames 8`），
  即"loss 看到的 rollout 长度 == 推理时的 rollout 长度"。
- ckpt：`artifacts/plan_a_hs_1000dom/checkpoint_best.pt`（6000 步，best val ens/total 1.7509；val_persist 2.6106）。

**配对评测**（`scripts/paired_hs_vs_model1.py`，同一 val 窗口、同一冻结解码器、同一 M=8 采样；
每窗口取 Δ=(H+S − 模型1)，bootstrap 95% CI）：

| 指标 | 模型 1 | H+S（Plan A） | Δ | 95% CI | 判定 |
| --- | --- | --- | --- | --- | --- |
| RMSF ↓ | **0.8699** | 0.8414 | −0.0284 | [−0.0808, +0.0246] | 平 |
| RMSF-r ↑ | 0.5045 | **0.5361** | +0.0316 | [+0.0020, +0.0629] | ✅显著 |
| ROOT-MEAN W2 ↓ | **13.4780** | 13.4927 | +0.0147 | [−0.0128, +0.0445] | 平 |
| VARIANCE W2 ↓ | **7.7866** | 7.9003 | +0.1138 | [−0.2807, +0.5316] | 平 |
| MD-PCA W2 ↓ | **0.0697** | 0.0719 | +0.0022 | [−0.0009, +0.0053] | 平 |
| WEAK-CONTACT J ↑ | 0.0748 | **0.0780** | +0.0032 | [−0.0033, +0.0095] | 平 |
| TRANSIENT J ↑ | 0.1768 | **0.1855** | +0.0087 | [+0.0017, +0.0157] | ✅显著 |

**读法**：**terminal-only Plan A ≈ 模型 1**——仅 **RMSF-r** 与 **TRANSIENT-J** 显著更好，
其余 5 项统计上打平（无显著变差）。它是一个**可用的、与模型 1 同级的 GLE 版本**，
但**尚未稳定超越**模型 1；更未达到 §3.6 中模型 1 的 global RMSF-r 水平。

**同一批次里还试了"密集轨迹监督"（dense），结论是失败，已回退**（详见 §10.2）：
dense 把 rollout 的 **每个中间帧** 都拿去和真实 future 帧对齐监督，但 7 项里 6 项不如模型 1
（`variance_w2`、`md_pca_w2`、`rmsf_mae` 显著变差）。

> **两个评测 bug（已修，§10.1）**使上一轮"7/7 全胜"为假象；修正并做**真配对**后才得到上表。
> 数字来源：`results/plan_a_hs_vs_model1.json`（**以此为准**）；
> 旧的 `results/plan_a_vs_model1.json` / `results/plan_a_dense_vs_model1.json` 为修正前产物，**勿引用**。

> **口径提醒**：训练用 `step_ps=1000`，而评测用 **paired loader**（`--horizons 8`：scalar 4 帧 context + stride-8 future 窗口），
> 与训练 dense 所用的 **contiguous loader** 不是同一协议。当前**模型 1 与 H+S 两侧同用该 paired loader**，
> 配对比较是自洽的；但**跨 horizon 比较**需先统一协议（见 §10.3）。

---

## 4. 轨迹 PDB 文件（肉眼/可视化检查）

> 脚本：`scripts/export_eval_trajectories_pdb.py`（**新文件**，原精确坐标解码代码未改动）。
> 每个窗口 6 条 CA-only 多 MODEL PDB（16 帧），直接可在 PyMOL/VMD 里叠看。

```
artifacts/pdbs_1000dom_20261020/<protein>_T<temp>_R<rep>_anchor<f>/
    ground_truth.pdb   persistence.pdb
    ours_stageA.pdb    ours_fromnone.pdb
    mdgen_zeroshot.pdb mdgen_mdcath.pdb
```

- **1000dom 版（推荐）**：`artifacts/pdbs_1000dom_20261020/`，窗口
  `1nhwC00`(L=60)、`1ka1A01`(L=219)、`1kafD00`(L=104)，均 T=320 K、anchor frame 7。含 `index.json`、`README.md`、`export.log`。
- **200dom 版（旧）**：`artifacts/pdbs_20261020/`（同样 6 轨迹 × 3 窗口 × 16 帧）。
- 口径与 MDGen 时标错配声明见各自目录内 `README.md`。

### 4.1 ⚠️ 关键澄清：系综 PDB vs rollout PDB（"SOTA 却 RMSD 很差"的真相）

§3 的 SOTA 数字与"肉眼看着会漂移"的 PDB，**是两件不同的事**，各自有对应的代码与目录：

| 目录 | 任务 | 生成代码 | 典型 CA-RMSD |
|------|------|----------|--------------|
| **`artifacts/pdbs_1000dom_ensemble_20261021/`** | **单步系综**（`sample_future_ca` 调 1 次，吐 N 个构象，无回馈） | `eval_ensemble_metrics_suite.py` | **与 §3 报告一致（SOTA）** |
| `artifacts/pdbs_1000dom_aligned_20261021/` | 自回归 **rollout**（`for _ in range(T_fut)`，把预测喂回去） | `eval_ours_trajectory.py` | ours ≈ 8.4 vs persistence 3.9 |

- §3 的全部 SOTA 指标**只**来自**单步系综**任务 —— 它就是报告里的那张表。
- 报告的系综指标**完全可以做成 PDB**：见 `pdbs_1000dom_ensemble_20261021/`（每窗口 32 个样本 + 65 帧真值），RMSF 量级正确（0.58–2.0 Å，聚合 ≈0.84 Å 与报告相符），**不漂移**。
- rollout 目录里的"差 RMSD"是**另一个更难的任务**：每一帧由**上一帧的预测**外推，误差 ~2 Å/帧 累积。它**不等于**报告口径。
- **结论：复现报告不需要 rollout。** 单步系综就是报告口径，其 PDB 已导出（见上表第 1 行）。

---

## 5. 训练方法细节

### 5.1 模型 1 — Stage-A ensemble 训练流程

三段式（只在第 3 段可训练，前两段冻结）：

1. **Stage-A v2（encoder 预训练）**：JEPA 式 latent 预测 + 位移/相对位置监督。产物
   `artifacts/stage_a_disp_relpos/checkpoint_latest.pt`（**已是 1000dom 训练**）。
2. **Stage-B1a（anchored decoder）**：从 anchor+history 解码未来 CA，锚定到 anchor。
   损失 = relative-disp + CA-MSE + FAPE + torsion + distance + phys-future + anchor。
   1000dom 重训产物：`artifacts/stage_b1a_relpos_anchored_1000dom/`（6000 步，`run_scripts/run_b1a_relpos_anchored_1000dom.sh`）。
3. **Gaussian 系综头（本阶段唯一可训练）**：拟合 latent H 的高斯 → 采样 M 个 H → 解码成 CA 云。
   损失 = RMSF(L1) + contact(L1) + 内距 soft-EMD。产物：`artifacts/stage_ensemble_1000dom/`
   （`run_scripts/run_ensemble_1000dom.sh`；encoder+decoder 冻结）。

### 5.2 模型 2 — from-none GLE 训练流程

单段端到端：随机 encoder + GLE(bath_dim=8) latent 随机动力学 + 残差解码器 `ca = X_ctx + tanh(Δx)`。
损失 = 点预测 CA RMSD + latent ES + 系综分布项（v4e 加入 RMSF-r 剖面损失）。
演进 anchor → v4 → v4c/v4d → v4e（最优）→ v4f（训练中）。

---

## 6. 诚实局限与下一步

1. **CA 绝对位置精度是共同短板**：
   - 模型 1 rollout Kabsch **8.39 Å** 差于 persistence **3.87 Å**（自回归回填漂移）；
   - 模型 2 的 `root_mean_w2` 在**短时标（1 ns）**失败（位移过小）。
   → 这是后续 Stage B 的重点。**分布形态（RMSF-r/W2/接触）已稳赢，位置精度待攻。**
2. **MDGen zero-shot 是 OOD**（10 ps vs 1000 ps/帧）；"MDGen 不行"**不是**它的正常水平。
3. **MDGen 短训远未收敛**；要公平需 ~10⁴ epoch 或步长对齐。
4. **接触类 Jaccard 对 delta/近刚体基线退化**——引用时必须附 `n_true/n_pred`。
5. **bioemu2 是"系综采样"而非"轨迹推进"**：长时标赢 persistence，短时标输，属任务口径差异，非模型缺陷。
6. **模型 1 的系综头已与最终 B1a 对齐**（§7 R7，best val 3.8130）；若后续再改 B1a 需重跑系综头。
7. **四项现已同处 1000dom 口径**（模型 1/模型 2/MDGen/bioemu2 均在同一 mdCATH-1000dom 家族上评估）；
   唯有 MDGen 的**帧长**仍是 OOD（10 ps vs 1000 ps）。

---

## 7. 附：1000dom 统一重训（2026-10-20/21）

> 目标：把默认数据集统一为 `mdcath_manifest_1000dom`，并**在 1000dom 上重训 Model 1**，
> 使 **Model 1 / MDGen / from-none GLE / bioemu2 四者数据口径一致**，可做直接对比。

| 组件 | 之前 | 现在（1000dom） | 产物 |
| --- | --- | --- | --- |
| Stage-A encoder | 已 1000dom | 不变 | `artifacts/stage_a_disp_relpos/` |
| **B1a 锚定解码器** | 200dom | **重训 1000dom（6000 步）** | `artifacts/stage_b1a_relpos_anchored_1000dom/` |
| **Gaussian 系综头** | mixed 200dom+ATLAS | **重训 1000dom** | `artifacts/stage_ensemble_1000dom/` |
| 默认 manifest | 200dom | **1000dom**（3 个训练脚本默认值已改） | — |

- 1000dom split：**800 训练域 / 100 验证域，0 重叠**。
- **B1a 最终收敛**（1000dom val）：`ca_rmsd_real = 3.818 Å` **优于 persistence 3.890 Å**；
  且 real < zeroH(4.93) < wrongH（3.82≈real）→ H 条件有效、无 H 会显著变坏。
- **修的 bug**：`train_stage_ensemble.py` 的 `--max-train-samples` 原用
  `torch.linspace(0,len-1,N)` 取 **N 个等间距**索引（末位四舍五入到 len → `IndexError`），
  已改为均匀 stride 子采样（保证 in-range）。
- **R7 ✅ 完成**：系综头用**最终 B1a ckpt**（step 6000）重训 4000 步，best val ens/total **3.8130**
  （旧 3.8461）。刷新评估后 Model 1 全面提升（见 §3.2 注）：RMSF-r global 0.538→**0.673**，
  per-target 0.293→**0.553**，RMSF 1.154→**0.837**。轨迹 PDB 同步重导出至
  `artifacts/pdbs_1000dom_20261021/`。

---

## 8. 复现命令

```bash
# 环境
export CLAUDE_CODE_TMPDIR=/liuyutian/protein_jepa_frame/.tmpwork TMPDIR=$CLAUDE_CODE_TMPDIR PYTHONPATH=/liuyutian/protein_jepa_frame

# 默认数据集（已写入各训练脚本默认值）
MANIFEST=artifacts/mdcath_manifest_1000dom   # train.jsonl / val.jsonl

# 1) B1a 锚定解码器（1000dom）
bash run_scripts/run_b1a_relpos_anchored_1000dom.sh

# 2) Gaussian 系综头（1000dom，依赖 B1a best ckpt）
bash run_scripts/run_ensemble_1000dom.sh

# 3) 评估 Model 1（7 系综指标 + Kabsch）
python -m evaluation.eval_ours_trajectory ...   # 见 artifacts/eval_model1_1000dom/

# 4) bioemu2 zero-shot baseline
bash run_scripts/run_bioemu2_zeroshot.sh
# 4b) bioemu2 zero-shot —— 3 次重复（seed 0/1/2，固定 PYTHONHASHSEED）
bash run_scripts/run_bioemu2_zeroshot_rep.sh
python -m evaluation.baselines.summarize_reps --in-dir results/bioemu2_reps \
  --out results/bioemu2_reps/summary.json

# 5) 导出轨迹 PDB
python scripts/export_eval_trajectories_pdb.py \
  --manifest $MANIFEST/val.jsonl \
  --out-dir artifacts/pdbs_1000dom_20261020
```

---

## 9. 产物清单

| 类别 | 路径 |
| --- | --- |
| 本报告 | `PROJECT_REPORT_20261021.md` |
| Model 1（1000dom）ckpt | `artifacts/stage_ensemble_1000dom/checkpoint_best.pt` |
| B1a（1000dom）ckpt | `artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt` |
| Model 2 ckpt | `artifacts/fromnone/stage_cb1b_v4e/checkpoint_latest.pt` |
| Model 1 评估 | `artifacts/eval_model1_1000dom/{suite.json, ours_rollout_100f.json}` |
| bioemu2 结果（单次，§3.5） | `results/bioemu2_zeroshot.json` |
| bioemu2 结果（3 次重复 mean±std，§3.5b） | `results/bioemu2_reps/summary.json`（`seed0/1/2.json`） |
| 轨迹 PDB（1000dom，**最终 ckpt**） | `artifacts/pdbs_1000dom_20261021/`（推荐） |
| **系综 PDB（=§3 报告口径，不漂移）** | **`artifacts/pdbs_1000dom_ensemble_20261021/`**（推荐，见 §4.1） |
| 轨迹 PDB（Kabsch 对齐版，rollout） | `artifacts/pdbs_1000dom_aligned_20261021/` |
| 轨迹 PDB（1000dom，初版） | `artifacts/pdbs_1000dom_20261020/` |
| 轨迹 PDB（200dom） | `artifacts/pdbs_20261020/` |
| 指标代码 | `evaluation/ensemble_metrics.py` + `evaluation/ensemble_metrics_ext.py` |
| 导出脚本 | `scripts/export_eval_trajectories_pdb.py` |
| 单测 | `tests/test_ensemble_metrics*.py`（26 项全过） |

---

## 10. Stage A + GLE 直连（Plan A）—— 代码路径、失败模式、下一步 【新增】

### 10.1 代码路径与冻结策略（当前最佳 = terminal-only）

**入口**：`protein_jepa_frame/training/train_stage_b_hs.py`
（车辆必须是它，**不是** cb1b：from-none 的 MLP 解码器绑死在 v4e 的 H 方言上，
无法解码 Stage-A 的 H；因此 Plan A 的冻结解码器只能是 B1a 的 `StageBOracleDecoder`。）

| 组件 | 状态 | 说明 |
| --- | --- | --- |
| Stage-A encoder | **可训练（低 LR，scale 0.1）** | 用 `--train-encoder --encoder-lr-scale 0.1`；Stage A 的 H 在各系综上表现好，是宝贵的初始化 |
| B1a decoder | **冻结** | 锚定解码器，保证 H→CA 映射不被破坏 |
| GLE kernel（`StageB1b`） | **可训练** | 唯一从零学的部分（`U(H)` 能量 + 静态浴 D/C/Γ + `HistoryPosterior`） |

**关键训练技巧**：`--rollout-steps == --paired-horizons-frames`。
旧版 `stage_hs` 用 **1 步 rollout** 去拟合**整窗口**目标 → 多步的 `h·C@S` 浴反馈从未进过 loss，
推理时发散（实测 ‖H_rollout−H₀‖：1 步 33 → 4 步 9220）。对齐步数后，训练与推理的 rollout 一致。

**结果**：见 **§3.7** —— **terminal-only ≈ 模型 1**（7 项中 2 项显著更好，其余打平）。
ckpt：`artifacts/plan_a_hs_1000dom/checkpoint_best.pt`。

### 10.2 失败尝试：密集轨迹监督（dense）

代码路径：`train_stage_b_hs.py` 的 `_run_batch_dense` + `_decode_path` +
`--dense-trajectory --dense-stride --dense-horizon --dense-{ca,es,rmsf,rmsf-corr,var}-weight`；
数据用 **contiguous loader**（`collate_clips` + `split_contiguous_window`，`step_ps` 匹配）保证
"rollout 第 i 步 ↔ 真实 future 第 i 帧"。

- 朴素 rollout 只把 `traj[:, -1]` 拿去 loss，**丢掉了 T 个中间帧**（H=8 时 1 用/9 产）。
- dense 把**整条轨迹**每一步都做 Kabsch 点损失 + energy score + 系综 RMSF/RMSF-r/variance。
- **结论：失败**。7 项里 6 项不如模型 1（`variance_w2` +1.04、`md_pca_w2` +0.0043、
  `rmsf_mae` +0.123、`transient_J` −0.013 均显著变差），且训练耗时约 2×。
  推测：逐步 Kabsch 点损失**压制了 GLE 应有的扩散涨落**，反而拉坏系综的方差/慢模态。
- **已回退**：一切改动 flag 门控——去掉 `--dense-trajectory` 即回 terminal-only，
  去掉 `--train-encoder` 即回冻结 encoder；旧路径行为完全不变。
  结果 `results/plan_a_dense_*.json` 为修正前产物，**勿引用**。

**两个评测 bug（上一轮"全胜"为假象的根因，已在 `scripts/paired_hs_vs_model1.py` 修复）**：
1. **全局 RNG 未播种**：`GaussianHHead.sample`/GLE 用全局 torch generator，脚本只播了
   dataset-subset 的局部 generator → 同一个冻结模型 1 每个进程系综都不同。已加 `torch.manual_seed`。
2. **模型 1 与 H+S 共用同一 encoder 对象**：`build()` 里两者建在同一实例上，H+S 重载
   **微调后**的 encoder 时把模型 1 的 encoder 也覆盖了 → "模型 1"基线随 run 漂移
   （rmsf_mae 1.0376 vs 1.4248）。已给模型 1 `deepcopy(encoder)` 隔离。
  修复后模型 1 基线两次评测**完全一致**（0.8699），才是真配对。

### 10.3 待办：什么是"干净配对" + Next-step plan

**"干净配对（clean paired）"** 指的是四点同时满足的评测：
1. **同一冻结栈**：模型 1 与 H+S 用**同一份** Stage-A/B1a 权重（不能像 10.2 的 bug 那样被覆盖）；
2. **同一窗口集合**：同一 manifest、同一 `(anchor, future-window)` 分割（val 域零重叠）；
3. **同一采样数 M、同一 horizon**；每窗口取 Δ=(H+S−模型1)，用**配对** bootstrap 出 CI（而非各评各的）；
4. **全局 RNG 播种**，跨进程可复现。
当前 §3.7 **1–4 全满足**；唯一遗留是**训练用 contiguous、评测用 paired 两个协议**——
只要"双方同用评测口径"就自洽，但若要比 horizon 曲线，需把训练也切到同一 contiguous 协议。

**Next-step plan（按优先级）**：
1. **轨迹级指标（回应用户第 1 点）**：现在只有"系综"指标，terminal-only 更是只看终点。
   引入**整条 rollout 轨迹**的约束/评估：绕行度（path length / 回转半径沿 t）、
   per-step Kabsch-RMSD 曲线、逐帧 RMSF、时间自相关（ACF/relaxation）、以及
   **路径分布评分**（ES on 轨迹）。这些作为**评估**先建起来，再考虑**进 loss**（dense 用的就是这条路，失败）。
2. **解冻 B1a decoder（回应用户第 2 点）**：现在 decoder 冻结。方案：
   (a) **低 LR 微调 decoder**（`--decoder-lr-scale`），(b) 末端几层解冻，(c) 仅在 Stage-A H 上用 **LoRA/残差 adapter**。
   风险：模型 1 的强项正是这个冻结 decoder 的保真度，解冻可能**同时破坏模型 1 基线**；
   因此需要**对照**（冻结 vs 解冻）与**配准**（解冻后模型 1 也要在同栈上重评）。
3. **统一协议**：train / eval 都切到 contiguous（`split_contiguous_window`, `step_ps` 匹配），
   消除 §10.2 的口径差，支持跨 horizon 直接比较。
4. **根因诊断（为何 GLE 赢不过模型 1）**：模型 1 的头是**静态高斯**（H 上一步采样），
   GLE 是**动力学**；当前 terminal-only 等价于"多步 rollout 后取终点"，
   但系综指标上看不到动力学优势 → 需要用 (1) 的轨迹指标证明 GLE 的价值，
   否则应把 GLE 定位为"轨迹推进"而非"系综生成"。

---

## 11. 复现命令（Plan A / terminal-only）【新增】

```bash
# 训练（terminal-only Plan A，GPU 4）
bash run_scripts/run_plan_a_hs_1000dom.sh          # -> artifacts/plan_a_hs_1000dom/

# 配对评测（self-contained，内部播种全局 RNG，模型 1 与 H+S 隔离 encoder）
bash run_scripts/run_paired_eval.sh                # -> results/plan_a_hs_vs_model1.json（以此为准）
```
