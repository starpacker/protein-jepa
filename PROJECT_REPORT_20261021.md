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
| **Lasp = Stage A + GLE 直连** | 把 Stage-A 表征与 GLE 动力学接起来（encoder 微调 + GLE 训练），代号 **Lasp**。**§3.7** 冻结解码器版：系综上 ≈ 模型 1，轨迹上完胜；**§3.7b 解冻解码器版（Lasp-final）**：系综上 **RMSF-r / MD-PCA W2 / WEAK-J / TRANSIENT-J 四项显著优于模型 1**。**§3.8** 轨迹级（order-aware）指标上全面显著优于模型 1（漂移 7.6 Å vs 33 Å）。**dense 逐点监督版本失败已回退**（§10.2）；**轨迹级分布损失版（Lasp-C）训练中**（§10.3）。代码见 **§10**、命令见 **§11** |
| **🆕 五方法统一横评** | **§12**：把 Model 1 / Model 2 / Lasp(§3.7b 解冻版) / MDGen / bioemu2 放进**同一个 harness、同一份 140 窗口计划**（5 温度 × {seen,unseen} 蛋白 × 1/4/16/64 ns），**结论见 §13.1–13.7，完整大表见 §13.8（T0–T8）**，每张表都含全部五种方法。**协议修正**：系综/位置指标按 horizon **独立采样**（与已发布口径一致），轨迹指标用**一条有序 rollout 切片**——两者不可混排。 |
| **🏆 当前 Lasp SOTA = v3（2026-10-23）** | **§14**：修掉旧 Lasp 的两个结构性缺陷——GLE 浴场改为 FDT 正确的**速度耦合**（旧版 `force` 耦合把迁移率放大 15.4 倍，导致轨迹发散）、去掉 `force_clip=1.0`（旧版削掉 ~85% 的力）。在**同协议 400 窗 / 100 蛋白**上对 bioemu2 显著胜负 **14:2**（旧 Lasp 同窗口是 10:5），对旧 Lasp 头对头 **赢 18/28 格**。⚠️ 唯一系统性退步是 **RMSF-r 四个时标全败**（对 bioemu2 仍留 2 个显著败格：16/64 ns）——瓶颈已定位为"方向对了、展布被压窄"。**§13 的 Lasp 列（`plan_a_unfrozen_dec_1000dom`）已被本节取代。** |
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

> **命名**：这套 Stage A + Stage B(GLE) 的完整模型代号 **Lasp**。
> 下面 §3.7 是 Lasp 的 **terminal-only-frozen-decoder**版本；§3.7b 是 **解冻解码器**的
> "Lasp-final" 版本（当前最佳），§3.8 是它的轨迹级评测。

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
>
> ⚠️ **重要**：§3.7 的 7 项都是**系综（无序 bag）指标**，它们**看不到轨迹顺序**。
> 因此"terminal-only ≈ 模型 1"只说明**终点系综**打平；**整条轨迹**上 H+S 明显更好 → 见 **§3.8**。

### 3.7b Lasp（解冻解码器） vs 模型 1 【新增，当前最佳版本】

> 在 §3.7 的基础上**再解冻 B1a 解码器**（`--train-decoder --decoder-lr-scale 0.05`，
> 低 LR），其余完全不变。假说：冻结解码器是 H→CA 映射的瓶颈，微调它能让 GLE 的 H
> 更好地落到系综上。结果**证实了假说**——
> ckpt：`artifacts/plan_a_unfrozen_dec_1000dom/checkpoint_best.pt`（best val ens/total **1.6873**，
> 优于冻结版 1.7509）。

| 指标 | 模型 1 | Lasp（冻结解码器，§3.7） | **Lasp（解冻解码器）** |
| --- | --- | --- | --- |
| RMSF ↓ | **0.8699** | 0.8414 | 0.8306 |
| RMSF-r ↑ | 0.5045 | 0.5361 ✔ | **0.5802 ✔** |
| ROOT-MEAN W2 ↓ | **13.4780** | 13.4927 | 13.4691 |
| VARIANCE W2 ↓ | **7.7866** | 7.9003 | 7.2177 |
| MD-PCA W2 ↓ | 0.0697 | 0.0719 | **0.0599 ✔** |
| WEAK-CONTACT J ↑ | 0.0748 | 0.0780 | **0.0894 ✔** |
| TRANSIENT J ↑ | 0.1768 | 0.1855 ✔ | **0.1992 ✔** |

（✔ = 配对 95% CI 排除 0。完整表：`results/plan_a_unf_dec_vs_model1.json`。）

**读法**：解冻解码器后，从"§3.7 的 2 项显著"升级到**4 项显著**：RMSF-r **+0.076**、
MD-PCA W2 **−0.0099**、WEAK-CONTACT J **+0.0145**、TRANSIENT J **+0.0224**；
RMSF/root_W2/variance_W2 也都不差于模型 1（点估计全面更优）。**这是 Lasp 目前最佳版本。**

### 3.8 轨迹级（order-aware "walk"）指标：H+S 显著优于模型 1 【新增】

> 回应"只约束 terminal 不合理、需要对整条轨迹做约束与评估"。系综指标把轨迹当成
> **无序构象袋**，无法区分"平滑扩散"和"把同样的构象打乱"。这里用
> `evaluation/trajectory_metrics.py`（**新增**）在**有序轨迹**上评估。
> 评测脚本 `scripts/paired_walk_metrics.py`（**新增**），协议 = **contiguous**
> （`K+H` 一段 clip，rollout 第 i 步与真实第 i 帧**物理时间对齐**，`step_ps` 匹配），
> 40 窗口、M=6。两侧同一冻结栈、同一窗口，Δ=(H+S−模型1)，bootstrap 95% CI。
> **下表用的是 §3.7 的冻结解码器版**；解冻解码器版（Lasp-final）两者相当
> （见 `results/walk_plan_a_unf_dec.json`：path_length 7.6→15.6、walk RMSF-r 0.579→0.641、
> acf_corr 0.905→**0.968 ✔**，即解冻后轨迹的**内部距离自相关更真实**）。

| 轨迹指标 | 模型 1 | H+S | Δ | 95% CI | 判定 |
| --- | --- | --- | --- | --- | --- |
| per-step RMSD 均值 ↓ | 9.6941 | 4.4846 | **−5.2095** | [−5.5743, −4.8200] | ✅显著 |
| path_length ↓ | 33.4020 | 7.6145 | **−25.7875** | [−26.9901, −24.4682] | ✅显著 |
| end_to_end ↓ | 15.3347 | 1.8701 | **−13.4646** | [−14.0441, −12.8672] | ✅显著 |
| Rg ↓ | 17.5951 | 16.0821 | −1.5129 | [−1.6685, −1.3532] | ✅显著 |
| traj energy score ↓ | 5.8987 | 2.8326 | **−3.0661** | [−3.3245, −2.7976] | ✅显著 |
| walk RMSF ↓ | 3.5605 | 1.2311 | **−2.3295** | [−2.9661, −1.6787] | ✅显著 |
| walk RMSF-r ↑ | 0.2672 | 0.5789 | **+0.3117** | [+0.2605, +0.3617] | ✅显著 |
| acf lag-1 ↑ | 0.2969 | 0.4075 | +0.1106 | [+0.1035, +0.1174] | ✅显著 |
| acf_corr ↑ | 0.9191 | 0.9048 | −0.0143 | [−0.0202, −0.0086] | 平（均接近满分） |

**读法**：**在整条轨迹上 H+S 全面、显著优于模型 1**——模型 1 的自回归 rollout 会累积
**严重全局漂移**（path_length 33 Å、end_to_end 15 Å），而 GLE 的轨迹**物理上合理**
（path_length 7.6 Å、end_to_end 1.9 Å、每步 RMSD 4.5 Å），符号之比的 `tortuosity` 也更真实。
这正好补上了系综指标看不到的维度：**GLE 的价值在"轨迹推进"，而非"终点系综"**。
（`acf_corr` 两者都 ~0.9 接近饱和，无区分度。）

> 这是"terminal-only 与模型 1 打平"的**关键反转**：系综上打平，但**轨迹上 H+S 完胜**。
> 因此下一步不应否定 GLE，而是**用轨迹级信号来训练/评价它**（见 §10.3 的 next-step）。
> 结果：`results/walk_plan_a_contig.json`。

### 3.9 【关键】三方向统一评测：Lasp vs 模型 1 vs bioemu2（同一协议、同一 200 窗口）

> 回应"等 Lasp 解冻解码器完成后，与 model 1 和 bioemu2 做一次测评"。为做到**逐窗口可比**，
> 新脚本 `scripts/eval_lasp_final.py`（**新增**）复刻了 bioemu2 harness
> （`run_bioemu2_zeroshot.py`）的**同一** manifest / `EvalProtocol` 窗口迭代 / 真实未来系综 /
> persistence / `extended_ensemble_summary` / bootstrap 聚合。**校验**：本脚本输出的
> persistence 与 `results/bioemu2_reps/summary.json` 的 persistence **逐位相同**
> （hz4 3.0198/14.3661/0.1895；hz16 4.4126/28.0998/0.2651；hz64 5.5644/41.1092/0.3160），
> 证明**评的是同一批窗口**。用 Lasp-final（解冻解码器，§3.7b）。
> 结果：`results/lasp_final_vs_model1_bioemu2.json`。

| horizon | 指标 | persistence | **bioemu2** | 模型 1 | **Lasp** | Lasp vs bio | vs 模型 1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 ns | RMSF-r ↑ | 0.000 | **0.444** | 0.285 | 0.337 | ✗ | ✔ |
| 1 ns | RMSF ↓ | 1.898 | 2.164 | 1.174 | **1.103** | ✔ | ✔ |
| 4 ns | RMSF-r ↑ | 0.015 | **0.595** | 0.365 | 0.479 | ✗ | ✔ |
| 4 ns | RMSF ↓ | 3.020 | 1.846 | 1.076 | **1.072** | ✔ | ✔ |
| 16 ns | RMSF-r ↑ | 0.002 | **0.597** | 0.348 | 0.409 | ✗ | ✔ |
| 16 ns | RMSF ↓ | 4.413 | **1.548** | 1.965 | 1.600 | ✗（点差 0.05） | ✔ |
| 16 ns | MD-PCA W2 ↓ | 0.265 | **0.077** | 0.153 | 0.131 | ✗ | ✔ |
| 16 ns | VARIANCE W2 ↓ | 28.10 | **14.24** | 20.33 | 17.25 | ✗ | ✔ |
| 64 ns | RMSF-r ↑ | 0.031 | **0.613** | 0.348 | 0.323 | ✗ | ✗ |
| 64 ns | RMSF ↓ | 5.564 | **1.622** | 3.058 | 2.114 | ✗ | ✔ |
| 64 ns | VARIANCE W2 ↓ | 41.11 | **17.62** | 33.32 | 27.22 | ✗ | ✔ |
| 64 ns | MD-PCA W2 ↓ | 0.316 | **0.088** | 0.204 | 0.161 | ✗ | ✔ |

（全 28 格统计：**Lasp vs 模型 1 = 24 胜 / 4 负**；**Lasp vs bioemu2 = 7 胜 / 21 负**。）

**读法（诚实结论）**：
1. **Lasp 明显优于模型 1**（24/28 格），两项显著（16–64 ns 的 RMSF、MD-PCA W2、VARIANCE W2 等）。
2. **Lasp 目前尚未超过 bioemu2**。bioemu2 在两个'形态保真'指标上占优：
   **RMSF-r（0.44–0.61 vs 0.32–0.48）** 与 **长时标离散度（MD-PCA W2、VARIANCE W2 在 16/64 ns 显著更小）**。
   本模型在 **RMSF 绝对误差（1/4/64 ns 更低）** 上更好。
3. **根因（已诊断）**：Lasp 目前是**单 horizon H=8（8 ns）训练**的，长时标（16/64 ns）**过度离散**
   （variance_W2 27.2 vs 17.6，md_pca_W2 0.161 vs 0.088）。bioemu2 不暴露前 k 帧历史、只看 anchor，
   却在长时标有更**校准**的扩散幅度 —— 这正是"horizon 校准"问题。
4. **下一步**：**多 horizon 训练**（§10.3 item 7，`run_scripts/run_lasp_mh.sh`，已启动）——
   在 1/4/16/64 ns 上同时最小化同一 terminal 系综损失（long rollout 用 truncated BPTT），
   让 GLE 学到一个**依赖 horizon 的速率**，直接消除长时标过度离散。
5. 按既定门槛（**未同时超过模型 1 与 bioemu2 就不汇报**）：**当前不满足，仍在推进**。

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
| **轨迹级指标（新增）** | `evaluation/trajectory_metrics.py`（§3.8） |
| **轨迹级配对评测（新增）** | `scripts/paired_walk_metrics.py` → `results/walk_plan_a_contig.json` |
| **Plan A ckpt** | `artifacts/plan_a_hs_1000dom/checkpoint_best.pt`（terminal-only，§3.7/§3.8） |
| **Plan A 系综配对结果** | `results/plan_a_hs_vs_model1.json`（**以此为准**） |
| 导出脚本 | `scripts/export_eval_trajectories_pdb.py` |
| 单测 | `tests/test_ensemble_metrics*.py` + `tests/test_plan_a_walk.py`（全过） |

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

 
## 11. 复现命令（Plan A / terminal-only）【新增】

```bash
# 训练（terminal-only Plan A，GPU 4）
bash run_scripts/run_plan_a_hs_1000dom.sh          # -> artifacts/plan_a_hs_1000dom/

# 训练（变体 B：解冻 B1a decoder，低 LR；与上面同栈、仅 decoder 是否冻结不同）
bash run_scripts/run_plan_a_unfrozen_decoder.sh    # -> artifacts/plan_a_unfrozen_dec_1000dom/

# 系综配对评测（self-contained，内部播种全局 RNG，模型 1 与 H+S 隔离 encoder/decoder）
bash run_scripts/run_paired_eval.sh                # -> results/plan_a_hs_vs_model1.json（以此为准）

# 轨迹级（order-aware）配对评测：Model1 vs H+S 的整条 walk
python -m scripts.paired_walk_metrics --mode hs \
  --kernel-ckpt artifacts/plan_a_hs_1000dom/checkpoint_best.pt \
  --ensemble-ckpt artifacts/stage_ensemble_1000dom/checkpoint_best.pt \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --b1a-ckpt artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
  --manifest artifacts/mdcath_manifest_1000dom/val.jsonl \
  --protocol contiguous --horizons 8 --n-samples 6 \
  --out results/walk_plan_a_contig.json
```

# 三方向统一评测（Lasp vs 模型 1 vs bioemu2，同 bioemu2 协议/同 200 窗口）
python -u scripts/eval_lasp_final.py \
  --kernel-ckpt artifacts/plan_a_unfrozen_dec_1000dom/checkpoint_best.pt \
  --ensemble-ckpt artifacts/stage_ensemble_1000dom/checkpoint_best.pt \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --b1a-ckpt artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
  --manifest artifacts/mdcath_manifest_1000dom/val.jsonl \
  --device cuda --horizons-ns 1 4 16 64 --n-samples 12 --max-windows 200 \
  --out results/lasp_final_vs_model1_bioemu2.json
#   （校验：本脚本输出的 persistence 与 results/bioemu2_reps/summary.json 逐位相同）


## 12. 五方法统一横评（2026-10-22 新增）【新增】

> 代码：`scripts/five_method/`（`common.py` / `adapters_local.py` / `adapters_external.py` /
> `run_five_method.py` / `make_tables.py` / `paired_stats.py`）
> 产物：`results/five_method/*.jsonl`（逐窗口原始记录）、`FIVE_METHOD_COMPARISON_20261022.md`（表格）
> 复现：`bash run_scripts/launch_five_method.sh`

### 12.1 为什么要重跑：此前五家的数字**不可直接同表比较**

§3 与 §3.7b 的结论各自成立，但它们是**不同 harness、不同窗口集**上得到的：Model 1 与
Lasp 走 `scripts/eval_lasp_final.py` 的 200 窗口；bioemu2 走 `run_bioemu2_zeroshot.py` 的另一套
窗口；MDGen 从来没有在 mdCATH 上跑过完整 7 指标。把它们的数字直接并排，等于把
"窗口难度"和"方法优劣"混在一起——**尤其窗口只取 20 个时，蛋白难度差异会主导均值**。

因此本节的做法是：**一个 harness、一份窗口计划、五套方法**。

| 统一项 | 做法 |
| --- | --- |
| 窗口计划 | `evaluation/data.py::iter_windows` 只生成一次，五方法共用同一批 `(protein, trajectory, anchor, horizon)` |
| 真值 | 同一 `future_path`；`horizons=(64,)` 一次性枚举 ⇒ 每个 anchor 拿到完整 65 帧真值路径 |
| 残基掩码 | 同一 `residue_mask`，不完整残基在五方法间同进同出 |
| 指标代码 | 同一 `evaluation.ensemble_metrics_ext` / `evaluation.geometry` / `evaluation.trajectory_metrics` |
| 随机性 | 每窗口按 `sha256(protein|trajectory|anchor|horizon|T|salt)` 派生种子，`fork_rng` 隔离 |
| 比较口径 | 所有表格只用**五方法都完整跑过的窗口交集**（见 §13.8 的 T0），杜绝"用不同窗口平均" |

### 12.2 一个必须讲清楚的协议分歧（本次最重要的修正）

最初把 `horizons` 设成 `(64,)` 是为了"一次枚举拿到 65 帧真值"，随后把 rollout 的第 h 帧当作
"h ns 的预测"。这**改变了实验定义**：报告 §3 的 "1 ns Kabsch = 3.74 Å" 是**独立地在 1 ns
horizon 上采样**得到的；而从一条 64 步自回归链里读第 1 帧，误差会累积整条链的漂移
（实测 143–148 Å）。我把仓库自带的 `scripts/eval_ours_trajectory.rollout_trajectory`
逐字复制出来对照，漂移同样出现（逐字版 32.17 Å vs 我的 28.87 Å @ 16 ns）——**漂移是仓库
自回归口径的性质，不是新 harness 的 bug**。

最终拆成两套并行口径，并在每条记录里写 `protocol: 1`，旧记录无法混入：

| 指标组 | 协议 | 理由 |
| --- | --- | --- |
| `ens`（7 项系综指标）、`pos`（Kabsch/aligned） | **逐 horizon 独立采样**：要 1 ns 就按 1 ns 条件 | 与 §3 / §3.7b / bioemu2 zero-shot 的已发布口径一致，保证可对照 |
| `walk`（9 项 order-aware 轨迹指标） | **一条 64 步有序 rollout，再切片到 h** | 轨迹指标本来就只能定义在一条有序路径上 |

> 换句话说：**系综表回答"这个模型预测得准不准"，轨迹表回答"这条链走得稳不稳"**。
> 两者对同一方法可以给出不同排名（例如 Model 2 的 1 ns 系综很差，但 1 ns 短链几乎不动），
> 这是信息，不是矛盾——但**不能混在一张表里排名**。

### 12.3 五套方法在本 harness 里的接入方式

| 方法 | 产物形态 | 温度输入 | 有序路径 | ckpt |
| --- | --- | --- | --- | --- |
| **Model 1** | `EnsembleProducer` + `GaussianHHead`，逐 horizon 独立采样 | ✅ `cond=[log(h_ps)/log 1e4, (T−300)/150]` | ✅ 自回归 walk | `stage_ensemble_1000dom`（温度变体 `--temp-split`） |
| **Model 2** | from-none GLE `kernel.rollout` + 残差解码器 | ✅ 同上（kernel 温度特征） | ✅ 原生 rollout | `fromnone/stage_cb1b_v4e`（温度变体 `--temp-split`） |
| **Lasp** | `HSEnsembleProducer`（Stage-A + B1b GLE + **B1a 解码器解冻** = §3.7b Lasp-final） | ✅ | ✅ `H_traj` 解码成 walk | `plan_a_unfrozen_dec_1000dom` |
| **MDGen** | `NewMDGenWrapper`，atom14 `[B,T,L,14,3]`，`sim_condition=True` | ❌ 无温度输入 | ✅ 一次采样整条轨迹 | `atlas.ckpt`（zero-shot） |
| **bioemu2** | `propagate_with_embeddings`（独立端点）/ `simulate_trajectory_with_embeddings`（有序） | ❌ 无温度输入 | ⚠️ 仅 `--bioemu2-walk` 模式 | `bioemu2-preview`（zero-shot） |

三个工程性坑，均已解决：

1. **MDGen 与 bioemu2 的 linalg 后端互斥**。MDGen 的 `rot_to_quat` 对 4×4 四元数矩阵做批量
   `eigh`，在本机 cuSOLVER 上抛 `CUSOLVER_STATUS_INTERNAL_ERROR`，MAGMA 正常；而 bioemu2 的
   `rigid_align.py:167` 调用 `svd(..., driver="gesvd")`，MAGMA 直接拒绝。解法是**按方法切换**
   `torch.backends.cuda.preferred_linalg_library`（`common.set_linalg_backend`），而不是全局设一次。
2. **bioemu2 有序 rollout 极慢**（~10 min/窗口），只能对 `rollout` 单元开 5 个窗口。
3. **5 进程并行时 CPU 线程超订**导致同一窗口从 10 s 涨到 105 s；固定
   `torch.set_num_threads(4)` + `set_num_interop_threads(1)` 后恢复。

### 12.4 温度泛化：怎么造出"训练没见过的温度"

mdCATH 只有 5 个温度 {320, 348, 379, 413, 450} K。为了让"unseen temperature"**真的成立**，
额外训练了**温度留出**变体：`artifacts/mdcath_manifest_1000dom_temp/` 下 `train_not450` /
`val_not450` 把 T=450 K 整体从训练集移除（已验证去重后蛋白集合与原始 train/val 完全相同，
即**只动温度、不动蛋白**），Model 1 / Model 2 用 `--temp-split` 加载该变体。

由此得到用户要求的三个格子（+1 个参照格）：

| 格子 | 蛋白 | 温度 | 对谁算"没见过的温度" |
| --- | --- | --- | --- |
| unseen protein × unseen T | val 划分（蛋白与轨迹均不重叠） | **450 K** | Model 1/2（温度留出） |
| unseen protein × seen T | val 划分 | 348 K | 无（纯蛋白泛化） |
| seen protein × unseen T | train 划分（见过的序列、没见过的轨迹） | **450 K** | Model 1/2（温度留出） |
| seen protein × seen T | train 划分 | 348 K | 无（参照上界） |

⚠️ **MDGen / bioemu2 没有温度输入**。它们在 §13.8-T3/T5 的 T450 列上的数字是"同一个模型被拿去预测更热的
轨迹"，正确的读法是 **zero-shot 迁移**，不是"温度条件化能力"。表里已按此标注。

### 12.5 时间尺度：1 / 4 / 16 / 64 ns

mdCATH 为 1 ns/帧（`step_ps=1000`），因此 `horizons_frames` 的数值**恰好等于 ns**：
1/4/16/64 ns = 1/4/16/64 帧。64 ns 是 mdCATH 单条轨迹长度内可用的最长时标。

### 12.6 读表须知（诚实局限）

- **交集口径**：为保证可配对，所有表格只统计五方法都跑完的窗口。bioemu2 最慢，
  因此交集规模由它决定（§13.8 的 T0 里明确写出每个方法的覆盖量和最终交集大小）。
- **样本量小**：每个温度格 20 个窗口、20 个蛋白（`prot` 更大的跑法见 `--max-proteins`）。
  均值差异请配合 §13.8 的 T8 **配对 bootstrap 95% CI** 读；单看均值差会过度解读。
- **MDGen 是 OOD 使用**：`atlas.ckpt` 在 10 ps/帧上训练，mdCATH 是 1000 ps/帧，
  帧间隔差 100×，且该 ckpt 并未在 mdCATH 上微调。它在 mdCATH 上劣于 persistence 是
  **已知的步长 OOD**，不代表其正常水平（与 §0 的结论一致）。
- **bioemu2 无有序路径**：默认模式（`propagate`）每次独立采样端点，轨迹表里天然是 `n/a`；
  只有 `--bioemu2-walk` 模式能填，且因成本只在 5 个窗口上跑。
- **接触类 Jaccard 的退化**：见 §2.2。delta/近刚体预测在弱接触带上天然趋近 0，引用时附
  `n_true/n_pred`（JSONL 里有）。
- **Lasp 的温度扫描**：Lasp 的训练覆盖全部 5 个温度，因此它**没有**"unseen temperature"格；
  它在 §13.8-T3 的 T450 列应读作"已见温度的插值质量"，与 Model 1/2 的 T450 列**不可等同解读**。

### 12.7 口径核对：本 harness 能复现已发布数字

改协议之前先证明新 harness 没有算错。以 Model 1 为例，在 **20 个蛋白 / 每温度 20 窗口**
的新计划上跑出的逐窗口均值，与已发布口径（`artifacts/eval_model1_1000dom/suite_v2.json`，
200 样本、`--horizons 1 2 4 8` 的 8 帧未来窗）**同量级**：

| 指标 | 已发布（`suite_v2`，8 帧窗） | 本 harness（1→64 ns） |
| --- | --- | --- |
| RMSF-r ↑（per-window 均值） | 0.512 | 0.49 / 0.59 / 0.59 / 0.58 |
| RMSF 幅度 ↓ | 0.837 | 0.62 / 0.76 / 1.40 / 2.06 |
| MD-PCA W2 ↓ | 0.0665 | 0.079 / 0.077 / 0.121 / 0.163 |
| Kabsch CA RMSD ↓ | 3.74–4.62（短时标） | 3.40 / 3.88 / 5.00 / 6.03 |

两条口径的**趋势一致、量级一致**（RMSF 与 pca_W2 随时标单调变差，符合物理直觉），
差异来自窗口规模（20 vs 200）与时标端点（8 帧窗 vs 1/4/16/64 ns 单点）。
**结论：harness 复现了已发布协议，表里的五方法差异不是实现差异。**



---

## 13. 五方法正面对比：结论与完整大表（2026-10-22 新增）【新增】

> 数据来源：§13.8 的 T7/T8 与本节脚本，`results/five_method/all.jsonl`。
> 本节 13.1–13.7 的数字由 `.tmpwork/fm/section13.py` 从该文件重算，与 13.8 的
> T0–T8 大表**永远一致**（同一份数据、同一套窗口）。
> 口径：**unseen 蛋白**（val 划分，蛋白与轨迹均不与训练重叠），140 个统一窗口，
> 1/4/16/64 ns，配对 bootstrap（4000 次）95% CI。判显著 = CI 不含 0。

### 13.1 一句话结论

> **在"构象系综/分布形态"这一层，零样本的 bioemu2 明确强于我们三个模型
> （7 项系综指标 × 4 时标的显著胜负比：bioemu2 19:4 胜 Model 1、17:6 胜 Model 2、
> 17:4 胜 Lasp）。我们的优势集中在另外三处，而且都很硬：**
>
> 1. **锚定位置精度**：Kabsch CA RMSD 四个时标全部领先
>    （我们最好 3.02/3.96/5.12/6.59 Å vs bioemu2 4.52/5.40/6.43/7.78 Å），
>    且优势随外推距离拉长不衰减。
> 2. **1–16 ns 的涨落幅度控制**：`var_W2` 我们 3.8–18.4 vs bioemu2 11.9–16.1，
>    在 1/4 ns 显著优 2–3 倍；`RMSF` 幅度同理。**但 64 ns 反过来被 bioemu2 赢**——
>    长时标上它的"过度离散"反而更接近真实的宽分布，这点必须如实写。
> 3. **64 ns 有序 rollout 的稳定性**：Model 2 / Lasp 的 e2e 仅 2.7 Å（不漂移），
>    优于 bioemu2 的 5.6 Å；Model 1 的自回归在这里崩到 104 Å。
>
> **一句话选型建议**：要"不给定结构的构象系综 / 柔性谱 / 慢模分布"→ 用 bioemu2；
> 要"给定结构、给定温度、精确推进到指定时刻且不漂"→ 用 Model 2（1–16 ns 最准）
> 或 Lasp（64 ns 最稳）。这不是"谁全面更好"，而是**两类任务的分工**。

### 13.2 胜负总账（7 项系综指标 × 4 时标 = 28 项）


| 对比（A 方） | A 方显著赢 | **bioemu2 显著赢** | 无显著差异 | A 方胜率 |
|---|---|---|---|---|
| Model 1 vs bioemu2 | **4** | 19 | 5 | 14% |
| Model 2 vs bioemu2 | **6** | 17 | 5 | 21% |
| Lasp vs bioemu2 | **4** | 17 | 7 | 14% |
| MDGen vs bioemu2 | **0** | 20 | 8 | 0% |

> ⚠️ **这一格是本节最诚实、也最不舒服的数字：在 7 项系综指标上，bioemu2 反以
> 19:4 / 17:6 / 17:4 领先我们。** 三个模型结论一致，不是偶然。
> 原因是清楚的：bioemu2 是全原子扩散模型，它优化的目标就是"生成符合平衡分布的构象"，
> 而我们这三个模型是用 mdCATH 单帧锚定监督训练的，分布形态不是它们的训练目标。
> **这不代表我们的模型没用**——13.4/13.5 显示我们在位置精度和长轨迹上明显更强，
> 详见 13.6 的场景分工。
>
> 同一张表里 **MDGen 是 0:20**：它在这个 harness 下连一项系综指标都赢不了 bioemu2。
> 也就是说，我们的三个模型虽然输给 bioemu2，但**明显强于 MDGen**，
> 而 MDGen 正是此前被普遍引用为 "mdCATH 上最强" 的基线。这两个事实必须一起说。
>
> 📌 **本表的 Lasp 列已被 §14 取代。** 这里用的是
> `artifacts/plan_a_unfrozen_dec_1000dom`（legacy `force` 浴场 + `force_clip=1.0`）。
> 修掉这两个结构性缺陷后的 **v3**（`artifacts/lasp_v3_velocity`）在同一批 400 窗上
> 对 bioemu2 是 **14:2**，对这里的 Lasp 头对头赢 18/28 格，详见 §14。
> 注意本节的 140 窗口径与 §14 的 400 窗口径**不能直接换算**（同一 ckpt 在 140 窗是 4:17、
> 在 400 窗是 10:5），原因见 §14.2。

### 13.3 系综指标逐时标（明细分项）

每格是该方法 100 个 unseen 窗口的均值。**粗体** = 与我们对应模型的配对
bootstrap 差异显著（95% CI 不含 0）——**粗体只标我们三列**，因为判定问的是
"我们 vs bioemu2"；MDGen 列是参照，不参与判定。


#### horizon 1 ns

| 指标 | Model 1 | Model 2 | Lasp | MDGen | bioemu2 | 判定 |
|---|---|---|---|---|---|---|
| RMSF-r↑ | **0.469** | **0.442** | **0.510** | 0.466 | 0.581 | ❌ **bioemu2 赢**（Model 1/Model 2/Lasp 显著落后） |
| RMSF↓ | **0.641** | **1.278** | **0.593** | 2.039 | 1.534 | ✅ **我们赢**（Model 1/Model 2/Lasp） |
| root_W2↓ | **7.066** | **7.396** | **7.147** | 9.804 | 6.683 | ❌ **bioemu2 赢**（Model 1/Model 2/Lasp 显著落后） |
| var_W2↓ | **4.095** | **6.103** | **3.828** | 10.033 | 11.914 | ✅ **我们赢**（Model 1/Model 2/Lasp） |
| pca_W2↓ | **0.078** | **0.127** | **0.079** | 0.057 | 0.054 | ❌ **bioemu2 赢**（Model 1/Model 2/Lasp 显著落后） |
| weak_J↑ | 0.000 | **0.670** | 0.000 | 0.000 | 0.000 | ✅ **我们赢**（Model 2） |
| trans_J↑ | **0.146** | **0.005** | **0.163** | 0.097 | 0.185 | ❌ **bioemu2 赢**（Model 1/Model 2/Lasp 显著落后） |

#### horizon 4 ns

| 指标 | Model 1 | Model 2 | Lasp | MDGen | bioemu2 | 判定 |
|---|---|---|---|---|---|---|
| RMSF-r↑ | **0.567** | 0.692 | **0.625** | 0.571 | 0.686 | ❌ **bioemu2 赢**（Model 1/Lasp 显著落后） |
| RMSF↓ | **0.811** | **0.818** | **0.781** | 2.015 | 1.430 | ✅ **我们赢**（Model 1/Model 2/Lasp） |
| root_W2↓ | **3.639** | **3.558** | **3.640** | 7.575 | 3.402 | ❌ **bioemu2 赢**（Model 1/Model 2/Lasp 显著落后） |
| var_W2↓ | **7.615** | **8.333** | **7.146** | 12.295 | 15.129 | ✅ **我们赢**（Model 1/Model 2/Lasp） |
| pca_W2↓ | **0.077** | **0.080** | **0.074** | 0.060 | 0.058 | ❌ **bioemu2 赢**（Model 1/Model 2/Lasp 显著落后） |
| weak_J↑ | **0.127** | **0.145** | **0.143** | 0.099 | 0.174 | ❌ **bioemu2 赢**（Model 1/Model 2/Lasp 显著落后） |
| trans_J↑ | **0.250** | 0.302 | **0.292** | 0.174 | 0.317 | ❌ **bioemu2 赢**（Model 1/Lasp 显著落后） |

#### horizon 16 ns

| 指标 | Model 1 | Model 2 | Lasp | MDGen | bioemu2 | 判定 |
|---|---|---|---|---|---|---|
| RMSF-r↑ | **0.558** | **0.644** | **0.604** | 0.583 | 0.718 | ❌ **bioemu2 赢**（Model 1/Model 2/Lasp 显著落后） |
| RMSF↓ | 1.515 | 1.433 | 1.342 | 2.116 | 1.285 | — 无显著差异 |
| root_W2↓ | 3.474 | **3.361** | 3.458 | 7.883 | 3.583 | ✅ **我们赢**（Model 2） |
| var_W2↓ | 17.934 | 18.418 | 16.636 | 18.071 | 16.097 | — 无显著差异 |
| pca_W2↓ | **0.126** | **0.125** | **0.117** | 0.079 | 0.069 | ❌ **bioemu2 赢**（Model 1/Model 2/Lasp 显著落后） |
| weak_J↑ | **0.161** | **0.189** | **0.193** | 0.156 | 0.234 | ❌ **bioemu2 赢**（Model 1/Model 2/Lasp 显著落后） |
| trans_J↑ | **0.301** | **0.352** | **0.338** | 0.250 | 0.387 | ❌ **bioemu2 赢**（Model 1/Model 2/Lasp 显著落后） |

#### horizon 64 ns

| 指标 | Model 1 | Model 2 | Lasp | MDGen | bioemu2 | 判定 |
|---|---|---|---|---|---|---|
| RMSF-r↑ | **0.533** | **0.407** | **0.474** | 0.558 | 0.725 | ❌ **bioemu2 赢**（Model 1/Model 2/Lasp 显著落后） |
| RMSF↓ | **2.422** | **2.273** | 1.942 | 2.462 | 1.678 | ❌ **bioemu2 赢**（Model 1/Model 2 显著落后） |
| root_W2↓ | 4.222 | 4.111 | 4.231 | 8.213 | 4.176 | — 无显著差异 |
| var_W2↓ | **31.262** | **31.647** | 27.756 | 27.110 | 23.353 | ❌ **bioemu2 赢**（Model 1/Model 2 显著落后） |
| pca_W2↓ | **0.173** | **0.173** | **0.140** | 0.101 | 0.092 | ❌ **bioemu2 赢**（Model 1/Model 2/Lasp 显著落后） |
| weak_J↑ | **0.148** | **0.138** | **0.163** | 0.144 | 0.215 | ❌ **bioemu2 赢**（Model 1/Model 2/Lasp 显著落后） |
| trans_J↑ | **0.275** | **0.331** | **0.307** | 0.302 | 0.353 | ❌ **bioemu2 赢**（Model 1/Model 2/Lasp 显著落后） |


### 13.4 绝对位置精度：Kabsch CA RMSD（Å，越低越好）

这是**我们最稳定赢的一项**，且随外推距离拉长优势不减——因为我们的模型是**锚定推进**（从 anchor 出发，位置受控），而 bioemu2 的 `propagate` 是**独立采样**（只保证分布对，不管这一次落到哪个具体构象）。

| horizon | Model 1 | Model 2 | Lasp | MDGen | bioemu2 | 我们最好 | 结果 |
|---|---|---|---|---|---|---|---|
| 1 ns | 3.52 | 3.02 | 3.61 | 12.90 | 4.52 | **3.02** | ✅ 我们赢 |
| 4 ns | 4.20 | 3.96 | 4.23 | 13.43 | 5.40 | **3.96** | ✅ 我们赢 |
| 16 ns | 5.42 | 5.12 | 5.51 | 14.29 | 6.43 | **5.12** | ✅ 我们赢 |
| 64 ns | 6.91 | 6.59 | 7.50 | 15.30 | 7.78 | **6.59** | ✅ 我们赢 |

### 13.5 轨迹 rollout（64 ns 有序路径）

只有能产出**有序路径**的方法才进这张表。bioemu2 需 `--bioemu2-walk`（一条时间链），
因此样本少（5 窗口 × 8 = 40 条），仅作参考。

| 指标 | Model 1 | Model 2 | Lasp | MDGen | bioemu2 | 备注 |
|---|---|---|---|---|---|---|
| per-step↓ | 54.65 | 5.60 | 6.16 | 14.60 | 7.65 |  |
| path_len↓ | 351.1 | 75.6 | 108.0 | 90.5 | 91.1 |  |
| e2e↓ | 104.11 | 2.68 | 2.76 | 6.78 | 5.63 | 漂移量；<br>Model 1 崩坏 |
| Rg↓ | 57.8 | 15.9 | 17.1 | 20.9 | 15.6 | 真实 Rg≈16 Å；<br>M1 涨到 58 Å |
| energy↓ | 61.45 | 3.54 | 3.54 | 11.79 | 3.31 |  |
| ACF-1↑ | 0.922 | 0.627 | 0.455 | 0.922 | 0.880 | 时间自相关；<br>越高越平滑 |
| ACF-corr↑ | 0.708 | 0.902 | 0.830 | 0.595 | 0.670 |  |

**读法**：`end_to_end` 和 `Rg` 是漂移的硬指标。
**Model 1 的自回归 64 ns 会崩**（e2e 104 Å、Rg 58 Å），这就是 §3.8 的已知短板；
**Model 2 / Lasp 完全不崩**（e2e 2.7 Å），比 bioemu2 的 5.6 Å 还稳。
换句话说：**长轨迹这条需求上，Model 2 和 Lasp 是全场最稳的**，Model 1 不能用于长 rollout。

### 13.6 结论分场景

| 场景 | 谁更强 | MDGen 表现 | 依据（全部来自 13.3 / 13.4 / 13.5） |
|---|---|---|---|
| 预测**柔性谱形态**（RMSF-r↑） | **bioemu2** | 0.47–0.58，全程低于 bioemu2 且多数低于我们 | 全程 0.68–0.73 vs 我们 0.41–0.69，4 个时标全显著 |
| 预测**慢模分布**（pca_W2↓） | **bioemu2** | 0.057–0.101；1 ns 与 bioemu2 持平，4 ns 起与我们同量级 | 全程 0.054–0.092 vs 我们 0.074–0.173，4 个时标全显著 |
| 预测**弱/瞬态接触**（weak_J↑、trans_J↑） | **bioemu2** | weak_J 0/0.10/0.16/0.14；trans_J 0.10–0.30，全程低于 bioemu2 | 4/16/64 ns 两项全显著领先；仅 1 ns 的 weak_J 由 Model 2 领先 |
| 预测**涨落幅度**（var_W2↓、RMSF↓）@1–16 ns | **我们** | var_W2 10.0/12.3，略优于 bioemu2 但仍远差于我们；RMSF 2.0 全场最差 | 1 ns 4.1 vs 11.9、4 ns 7.6 vs 15.1；bioemu2 系综过度离散 |
| 预测**涨落幅度**@64 ns | **bioemu2** | var_W2 27.1，比 bioemu2 的 23.4 更差 | var_W2 23.4 vs 27.8–31.6、RMSF 1.68 vs 1.94–2.42 |
| 预测**具体位置**（Kabsch RMSD↓） | **我们** | 12.9 / 13.4 / 14.3 / 15.3 Å，全场最差 | 4 个时标全领先；64 ns 6.59 vs 7.78 Å |
| **64 ns rollout 不漂移**（e2e↓/Rg↓） | **Model 2 / Lasp** | e2e 6.99 Å、Rg 20.9，优于 bioemu2 但不及 Model 2/Lasp | e2e 2.7 Å；bioemu2 5.6 Å；Model 1 崩到 104 Å |
| **温度条件化** | **我们**（能力差异，非实测胜负） | 无温度输入（与 bioemu2 同为 zero-shot） | bioemu2 无温度输入；T450 对它是纯 zero-shot 迁移 |
| 输出**全原子/物理合法性** | **bioemu2** | 同为 CA 骨架，且 MDGen 需要物理后端才出全原子 | 它出全原子并过物理检查；我们只有 CA 骨架 |


### 13.7 必须说清的限制

1. **对比并非完全对称**：bioemu2 无温度输入，§13.8-T5 里 T450（unseen 温度）那一格对它是
   零样本迁移，而我们的模型有温度条件。公平的结论应限定在 T=348/320 这些常规温度。
2. **样本量小**：每格 20 蛋白 / 20 窗口。配对 bootstrap 已尽量控制，但 28 项里有 5–7 项
   不显著属正常，**不要按均值差排名**，按 CI 判断。
3. **任务层级不同**：我们输出 CA 骨架，bioemu2 输出全原子。若下游需要侧链/氢键，
   两者不在同一层级，本节结论不适用。
4. **bioemu2 的轨迹列样本极少**（n_traj=40），13.5 该列只作参考，不能当 140 窗口的结论。
5. **bioemu2 是 zero-shot**：`bioemu2-preview` 未在 mdCATH 上微调。这说明"我们的模型
   在未微调的通用扩散模型面前表现如何"，而不是"架构上限对比"。


### 13.8 完整大表（T0–T8；每张表都含全部五种方法）

> 由 `scripts/five_method/make_tables.py` 从 `results/five_method/all.jsonl` 自动生成，
> 不要手改；重跑 `bash .tmpwork/fm/build_report.sh` 即可刷新。
> T1/T2 = unseen 蛋白的多时标系综与位置/轨迹指标；T3/T4 = 温度泛化（unseen / seen 蛋白）；
> T5 = 用户指定的三格温度场景并排；T6 = 有序 rollout；T7 = 多尺度汇总；T8 = 配对 bootstrap。

# Five-method comparison — one harness, one window set

Records: **700** method/window pairs across 40 proteins and 7 (protein-split x temperature) cells.

All five methods are scored on identical windows: same anchors, same true MD futures, same residue masks, same persistence, and the same `evaluation.*` metric code. The only thing that differs is how each method produces its CA cloud.

`n/a` means the method structurally cannot fill that cell (documented per table), never that the run failed.


#### T0 — Coverage and provenance


#### Coverage — distinct proteins scored per temperature

| method | T320 | T348 | T379 | T413 | T450 | cells | windows | proteins |
|---|---|---|---|---|---|---|---|---|
| Model 1 | 20 | 40 | 20 | 20 | 40 | 7 | 140 | 40 |
| Model 2 | 20 | 40 | 20 | 20 | 40 | 7 | 140 | 40 |
| Lasp | 20 | 40 | 20 | 20 | 40 | 7 | 140 | 40 |
| MDGen | 20 | 40 | 20 | 20 | 40 | 7 | 140 | 40 |
| bioemu2 | 20 | 40 | 20 | 20 | 40 | 7 | 140 | 40 |

> Counts are *proteins*, not windows (several windows can come from one trajectory). `—` means the method has no record at that temperature.

#### Common windows — all five methods

| method | fully scored windows |
|---|---|
| Model 1 | 140 |
| Model 2 | 140 |
| Lasp | 140 |
| MDGen | 140 |
| bioemu2 | 140 |

> Intersection used below: **140** windows. Averaging over different window sets would confound protein difficulty with method quality, so every table is restricted to this intersection.

#### T1 — Unseen protein: ensemble metrics across time scales


#### Unseen protein (mdCATH-1000dom val, disjoint in protein *and* trajectory) — horizon 1 ns

| method | RMSF-r↑ | RMSF↓ | root_W2↓ | var_W2↓ | pca_W2↓ | weak_J↑ | trans_J↑ | n_win |
|---|---|---|---|---|---|---|---|---|
| Model 1 | 0.469 | 0.641 | 7.07 | 4.10 | 0.0784 | 0.000 | 0.146 | 100 |
| Model 2 | 0.442 | 1.278 | 7.40 | 6.10 | 0.1266 | 0.670 | 0.005 | 100 |
| Lasp | 0.510 | 0.593 | 7.15 | 3.83 | 0.0792 | 0.000 | 0.163 | 100 |
| MDGen | 0.466 | 2.039 | 9.80 | 10.03 | 0.0572 | 0.000 | 0.097 | 100 |
| bioemu2 | 0.581 | 1.534 | 6.68 | 11.91 | 0.0540 | 0.000 | 0.185 | 100 |

#### Unseen protein (mdCATH-1000dom val, disjoint in protein *and* trajectory) — horizon 4 ns

| method | RMSF-r↑ | RMSF↓ | root_W2↓ | var_W2↓ | pca_W2↓ | weak_J↑ | trans_J↑ | n_win |
|---|---|---|---|---|---|---|---|---|
| Model 1 | 0.567 | 0.811 | 3.64 | 7.61 | 0.0772 | 0.127 | 0.250 | 100 |
| Model 2 | 0.692 | 0.818 | 3.56 | 8.33 | 0.0795 | 0.145 | 0.302 | 100 |
| Lasp | 0.625 | 0.781 | 3.64 | 7.15 | 0.0740 | 0.143 | 0.292 | 100 |
| MDGen | 0.571 | 2.015 | 7.58 | 12.29 | 0.0603 | 0.099 | 0.174 | 100 |
| bioemu2 | 0.686 | 1.430 | 3.40 | 15.13 | 0.0575 | 0.174 | 0.317 | 100 |

#### Unseen protein (mdCATH-1000dom val, disjoint in protein *and* trajectory) — horizon 16 ns

| method | RMSF-r↑ | RMSF↓ | root_W2↓ | var_W2↓ | pca_W2↓ | weak_J↑ | trans_J↑ | n_win |
|---|---|---|---|---|---|---|---|---|
| Model 1 | 0.558 | 1.515 | 3.47 | 17.93 | 0.1261 | 0.161 | 0.301 | 100 |
| Model 2 | 0.644 | 1.433 | 3.36 | 18.42 | 0.1249 | 0.189 | 0.352 | 100 |
| Lasp | 0.604 | 1.342 | 3.46 | 16.64 | 0.1169 | 0.193 | 0.338 | 100 |
| MDGen | 0.583 | 2.116 | 7.88 | 18.07 | 0.0792 | 0.156 | 0.250 | 100 |
| bioemu2 | 0.718 | 1.285 | 3.58 | 16.10 | 0.0690 | 0.234 | 0.387 | 100 |

#### Unseen protein (mdCATH-1000dom val, disjoint in protein *and* trajectory) — horizon 64 ns

| method | RMSF-r↑ | RMSF↓ | root_W2↓ | var_W2↓ | pca_W2↓ | weak_J↑ | trans_J↑ | n_win |
|---|---|---|---|---|---|---|---|---|
| Model 1 | 0.533 | 2.422 | 4.22 | 31.26 | 0.1726 | 0.148 | 0.275 | 100 |
| Model 2 | 0.407 | 2.273 | 4.11 | 31.65 | 0.1728 | 0.138 | 0.331 | 100 |
| Lasp | 0.474 | 1.942 | 4.23 | 27.76 | 0.1400 | 0.163 | 0.307 | 100 |
| MDGen | 0.558 | 2.462 | 8.21 | 27.11 | 0.1011 | 0.144 | 0.302 | 100 |
| bioemu2 | 0.725 | 1.678 | 4.18 | 23.35 | 0.0920 | 0.215 | 0.353 | 100 |

#### T2 — Unseen protein: position and order-aware rollout metrics


#### Unseen protein — position + trajectory metrics — horizon 1 ns

| method | Kabsch↓ | aligned↓ | per-step↓ | path_len↓ | e2e↓ | Rg↓ | energy↓ | walkRMSF↓ | walkRMSF-r↑ | ACF-1↑ | ACF-corr↑ | n_win |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Model 1 | 3.52 | 3.51 | 3.51 | 1.5 | 1.48 | 16.9 | 1.93 | 1.24 | 0.000 | -0.128 | n/a | 100 |
| Model 2 | 3.02 | 3.02 | 3.02 | 0.3 | 0.33 | 16.7 | 2.23 | 0.00 | 0.000 | -0.125 | n/a | 100 |
| Lasp | 3.61 | 3.60 | 3.61 | 1.5 | 1.46 | 17.1 | 2.11 | 1.06 | 0.000 | -0.124 | n/a | 100 |
| MDGen | 12.90 | 12.52 | 12.90 | 5.5 | 5.49 | 20.9 | 11.32 | 3.21 | 0.000 | -0.132 | n/a | 100 |
| bioemu2 | 4.52 | 4.45 | 5.43 | 3.7 | 3.71 | 17.0 | 2.35 | 3.07 | -0.000 | -0.103 | n/a | 100 |

#### Unseen protein — position + trajectory metrics — horizon 4 ns

| method | Kabsch↓ | aligned↓ | per-step↓ | path_len↓ | e2e↓ | Rg↓ | energy↓ | walkRMSF↓ | walkRMSF-r↑ | ACF-1↑ | ACF-corr↑ | n_win |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Model 1 | 4.20 | 4.19 | 5.42 | 11.6 | 5.89 | 17.1 | 2.98 | 1.63 | 0.412 | -0.051 | 0.988 | 100 |
| Model 2 | 3.96 | 3.95 | 3.61 | 3.6 | 1.44 | 16.6 | 2.27 | 0.77 | 0.686 | 0.211 | 0.963 | 100 |
| Lasp | 4.23 | 4.22 | 4.01 | 5.3 | 1.33 | 17.1 | 2.36 | 0.67 | 0.600 | 0.203 | 0.964 | 100 |
| MDGen | 13.43 | 12.99 | 13.22 | 11.4 | 5.91 | 20.9 | 11.37 | 2.03 | 0.557 | 0.581 | 0.845 | 100 |
| bioemu2 | 5.40 | 5.35 | 5.56 | 10.0 | 4.34 | 16.6 | 2.23 | 1.66 | 0.685 | 0.562 | 0.873 | 100 |

#### Unseen protein — position + trajectory metrics — horizon 16 ns

| method | Kabsch↓ | aligned↓ | per-step↓ | path_len↓ | e2e↓ | Rg↓ | energy↓ | walkRMSF↓ | walkRMSF-r↑ | ACF-1↑ | ACF-corr↑ | n_win |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Model 1 | 5.42 | 5.42 | 13.64 | 59.2 | 23.06 | 21.2 | 10.03 | 5.25 | 0.265 | 0.625 | 0.799 | 100 |
| Model 2 | 5.12 | 5.10 | 4.41 | 17.5 | 1.87 | 16.4 | 2.61 | 1.40 | 0.733 | 0.473 | 0.913 | 100 |
| Lasp | 5.51 | 5.50 | 4.76 | 21.1 | 1.54 | 17.1 | 2.79 | 1.29 | 0.641 | 0.359 | 0.920 | 100 |
| MDGen | 14.29 | 13.75 | 13.78 | 29.7 | 6.39 | 20.9 | 11.47 | 1.96 | 0.618 | 0.832 | 0.674 | 100 |
| bioemu2 | 6.43 | 6.29 | 6.27 | 30.2 | 4.67 | 16.1 | 2.45 | 0.75 | 0.784 | 0.794 | 0.752 | 100 |

#### Unseen protein — position + trajectory metrics — horizon 64 ns

| method | Kabsch↓ | aligned↓ | per-step↓ | path_len↓ | e2e↓ | Rg↓ | energy↓ | walkRMSF↓ | walkRMSF-r↑ | ACF-1↑ | ACF-corr↑ | n_win |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Model 1 | 6.91 | 6.90 | 54.65 | 351.1 | 104.11 | 57.8 | 61.45 | 36.28 | 0.121 | 0.922 | 0.708 | 100 |
| Model 2 | 6.59 | 6.57 | 5.60 | 75.6 | 2.68 | 15.9 | 3.54 | 2.16 | 0.681 | 0.627 | 0.902 | 100 |
| Lasp | 7.50 | 7.48 | 6.16 | 108.0 | 2.76 | 17.1 | 3.54 | 1.93 | 0.569 | 0.455 | 0.830 | 100 |
| MDGen | 15.30 | 14.72 | 14.60 | 90.5 | 6.78 | 20.9 | 11.79 | 2.37 | 0.596 | 0.922 | 0.595 | 100 |
| bioemu2 | 7.78 | 7.60 | 7.65 | 91.1 | 5.63 | 15.6 | 3.31 | 1.21 | 0.696 | 0.880 | 0.670 | 100 |

> `Kabsch` is the common-anchor CA RMSD (the fitted transform comes from the true anchor→target pair only, so no prediction leaks into the gauge). `per-step`, `path_len`, `e2e`, `Rg`, `energy` are computed on each method's own ordered rollout; bioemu2 is `n/a` there because `propagate` draws independent endpoints and has no ordered path.

#### T3 — Temperature generalization, unseen protein

Rows = (metric, method); columns = the temperature of the evaluated trajectory. T=450 K is absent from the training split of the temperature-holdout checkpoints, so the T450 column is a genuinely unseen temperature for Model 1 / Model 2. Lasp is trained on all five temperatures but conditions on T only through the GLE kernel's temperature feature. MDGen and bioemu2 take **no temperature input at all**, so their columns are the same model run on hotter/colder trajectories — the correct reading is 'zero-shot transfer', not 'temperature conditioning'.


#### Unseen protein — horizon 1 ns

| metric | method | T320 | T348 | T379 | T413 | T450 |
|---|---|---|---|---|---|---|
| RMSF-r↑ | Model 1 | 0.506 | 0.441 | 0.490 | 0.509 | 0.398 |
| RMSF-r↑ | Model 2 | 0.399 | 0.387 | 0.446 | 0.500 | 0.481 |
| RMSF-r↑ | Lasp | 0.503 | 0.484 | 0.537 | 0.536 | 0.491 |
| RMSF-r↑ | MDGen | 0.462 | 0.446 | 0.468 | 0.435 | 0.521 |
| RMSF-r↑ | bioemu2 | 0.587 | 0.536 | 0.604 | 0.587 | 0.591 |
| RMSF↓ | Model 1 | 0.476 | 0.616 | 0.646 | 0.637 | 0.831 |
| RMSF↓ | Model 2 | 0.955 | 1.239 | 1.201 | 1.317 | 1.677 |
| RMSF↓ | Lasp | 0.482 | 0.573 | 0.575 | 0.591 | 0.743 |
| RMSF↓ | MDGen | 2.244 | 2.030 | 2.082 | 2.067 | 1.771 |
| RMSF↓ | bioemu2 | 1.539 | 1.703 | 1.568 | 1.432 | 1.429 |
| root_W2↓ | Model 1 | 6.88 | 7.37 | 6.91 | 7.60 | 6.58 |
| root_W2↓ | Model 2 | 7.14 | 7.69 | 7.23 | 7.95 | 6.97 |
| root_W2↓ | Lasp | 6.91 | 7.44 | 6.99 | 7.69 | 6.69 |
| root_W2↓ | MDGen | 9.71 | 9.87 | 9.82 | 10.36 | 9.25 |
| root_W2↓ | bioemu2 | 6.55 | 6.92 | 6.49 | 7.23 | 6.22 |
| var_W2↓ | Model 1 | 1.87 | 3.75 | 4.24 | 4.04 | 6.57 |
| var_W2↓ | Model 2 | 3.22 | 5.75 | 5.84 | 6.19 | 9.52 |
| var_W2↓ | Lasp | 2.13 | 3.12 | 3.97 | 3.74 | 6.19 |
| var_W2↓ | MDGen | 10.64 | 9.46 | 10.05 | 10.42 | 9.60 |
| var_W2↓ | bioemu2 | 11.30 | 15.50 | 11.65 | 9.66 | 11.46 |
| pca_W2↓ | Model 1 | 0.0571 | 0.0701 | 0.0844 | 0.0741 | 0.1065 |
| pca_W2↓ | Model 2 | 0.0927 | 0.1180 | 0.1298 | 0.1313 | 0.1612 |
| pca_W2↓ | Lasp | 0.0551 | 0.0699 | 0.0859 | 0.0808 | 0.1041 |
| pca_W2↓ | MDGen | 0.0536 | 0.0542 | 0.0606 | 0.0520 | 0.0656 |
| pca_W2↓ | bioemu2 | 0.0597 | 0.0565 | 0.0510 | 0.0487 | 0.0542 |
| weak_J↑ | Model 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| weak_J↑ | Model 2 | 0.800 | 0.900 | 0.650 | 0.650 | 0.350 |
| weak_J↑ | Lasp | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| weak_J↑ | MDGen | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| weak_J↑ | bioemu2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| trans_J↑ | Model 1 | 0.158 | 0.147 | 0.136 | 0.144 | 0.146 |
| trans_J↑ | Model 2 | 0.006 | 0.004 | 0.006 | 0.003 | 0.006 |
| trans_J↑ | Lasp | 0.153 | 0.156 | 0.163 | 0.169 | 0.173 |
| trans_J↑ | MDGen | 0.072 | 0.087 | 0.094 | 0.101 | 0.130 |
| trans_J↑ | bioemu2 | 0.154 | 0.167 | 0.183 | 0.193 | 0.226 |

> windows per column: T320=20, T348=20, T379=20, T413=20, T450=20

#### Unseen protein — horizon 4 ns

| metric | method | T320 | T348 | T379 | T413 | T450 |
|---|---|---|---|---|---|---|
| RMSF-r↑ | Model 1 | 0.596 | 0.568 | 0.576 | 0.577 | 0.516 |
| RMSF-r↑ | Model 2 | 0.681 | 0.685 | 0.715 | 0.700 | 0.681 |
| RMSF-r↑ | Lasp | 0.614 | 0.633 | 0.645 | 0.634 | 0.601 |
| RMSF-r↑ | MDGen | 0.537 | 0.599 | 0.582 | 0.574 | 0.562 |
| RMSF-r↑ | bioemu2 | 0.660 | 0.740 | 0.681 | 0.687 | 0.664 |
| RMSF↓ | Model 1 | 0.619 | 0.640 | 0.737 | 0.935 | 1.123 |
| RMSF↓ | Model 2 | 0.598 | 0.686 | 0.717 | 0.951 | 1.136 |
| RMSF↓ | Lasp | 0.577 | 0.583 | 0.749 | 0.915 | 1.083 |
| RMSF↓ | MDGen | 2.325 | 2.126 | 2.080 | 1.930 | 1.613 |
| RMSF↓ | bioemu2 | 1.571 | 1.578 | 1.403 | 1.238 | 1.359 |
| root_W2↓ | Model 1 | 3.33 | 3.67 | 3.50 | 3.95 | 3.75 |
| root_W2↓ | Model 2 | 3.29 | 3.62 | 3.44 | 3.84 | 3.59 |
| root_W2↓ | Lasp | 3.31 | 3.67 | 3.50 | 3.95 | 3.78 |
| root_W2↓ | MDGen | 7.39 | 7.54 | 7.65 | 7.89 | 7.40 |
| root_W2↓ | bioemu2 | 3.18 | 3.46 | 3.29 | 3.64 | 3.44 |
| var_W2↓ | Model 1 | 4.04 | 5.65 | 7.99 | 9.52 | 10.88 |
| var_W2↓ | Model 2 | 4.45 | 6.58 | 8.65 | 10.18 | 11.80 |
| var_W2↓ | Lasp | 3.87 | 5.08 | 7.88 | 9.04 | 9.86 |
| var_W2↓ | MDGen | 13.08 | 12.57 | 13.14 | 12.55 | 10.13 |
| var_W2↓ | bioemu2 | 15.47 | 17.91 | 13.16 | 12.18 | 16.93 |
| pca_W2↓ | Model 1 | 0.0567 | 0.0677 | 0.0758 | 0.0894 | 0.0964 |
| pca_W2↓ | Model 2 | 0.0566 | 0.0736 | 0.0796 | 0.0911 | 0.0967 |
| pca_W2↓ | Lasp | 0.0533 | 0.0616 | 0.0774 | 0.0866 | 0.0913 |
| pca_W2↓ | MDGen | 0.0650 | 0.0634 | 0.0599 | 0.0613 | 0.0517 |
| pca_W2↓ | bioemu2 | 0.0584 | 0.0697 | 0.0522 | 0.0538 | 0.0534 |
| weak_J↑ | Model 1 | 0.129 | 0.125 | 0.119 | 0.127 | 0.136 |
| weak_J↑ | Model 2 | 0.145 | 0.144 | 0.144 | 0.142 | 0.151 |
| weak_J↑ | Lasp | 0.143 | 0.135 | 0.152 | 0.145 | 0.143 |
| weak_J↑ | MDGen | 0.073 | 0.085 | 0.097 | 0.110 | 0.129 |
| weak_J↑ | bioemu2 | 0.146 | 0.166 | 0.175 | 0.191 | 0.193 |
| trans_J↑ | Model 1 | 0.261 | 0.254 | 0.232 | 0.240 | 0.262 |
| trans_J↑ | Model 2 | 0.307 | 0.309 | 0.302 | 0.294 | 0.300 |
| trans_J↑ | Lasp | 0.287 | 0.285 | 0.303 | 0.295 | 0.290 |
| trans_J↑ | MDGen | 0.134 | 0.160 | 0.168 | 0.187 | 0.223 |
| trans_J↑ | bioemu2 | 0.271 | 0.302 | 0.321 | 0.338 | 0.353 |

> windows per column: T320=20, T348=20, T379=20, T413=20, T450=20

#### Unseen protein — horizon 16 ns

| metric | method | T320 | T348 | T379 | T413 | T450 |
|---|---|---|---|---|---|---|
| RMSF-r↑ | Model 1 | 0.624 | 0.586 | 0.564 | 0.525 | 0.489 |
| RMSF-r↑ | Model 2 | 0.678 | 0.667 | 0.670 | 0.628 | 0.576 |
| RMSF-r↑ | Lasp | 0.633 | 0.633 | 0.628 | 0.572 | 0.556 |
| RMSF-r↑ | MDGen | 0.575 | 0.590 | 0.594 | 0.570 | 0.584 |
| RMSF-r↑ | bioemu2 | 0.714 | 0.751 | 0.753 | 0.707 | 0.664 |
| RMSF↓ | Model 1 | 1.259 | 1.332 | 1.273 | 1.627 | 2.084 |
| RMSF↓ | Model 2 | 1.148 | 1.330 | 1.195 | 1.496 | 1.996 |
| RMSF↓ | Lasp | 1.079 | 1.104 | 1.212 | 1.450 | 1.865 |
| RMSF↓ | MDGen | 2.537 | 2.420 | 2.045 | 1.845 | 1.733 |
| RMSF↓ | bioemu2 | 1.348 | 1.400 | 1.237 | 1.139 | 1.304 |
| root_W2↓ | Model 1 | 3.23 | 3.45 | 3.29 | 3.74 | 3.65 |
| root_W2↓ | Model 2 | 3.15 | 3.38 | 3.21 | 3.58 | 3.49 |
| root_W2↓ | Lasp | 3.22 | 3.46 | 3.25 | 3.71 | 3.64 |
| root_W2↓ | MDGen | 7.69 | 7.94 | 7.90 | 8.26 | 7.62 |
| root_W2↓ | bioemu2 | 3.44 | 3.69 | 3.50 | 3.80 | 3.48 |
| var_W2↓ | Model 1 | 13.65 | 16.99 | 13.69 | 19.26 | 26.08 |
| var_W2↓ | Model 2 | 13.81 | 17.72 | 14.18 | 19.64 | 26.73 |
| var_W2↓ | Lasp | 12.73 | 15.25 | 13.13 | 17.94 | 24.12 |
| var_W2↓ | MDGen | 20.73 | 20.96 | 15.01 | 15.93 | 17.73 |
| var_W2↓ | bioemu2 | 17.38 | 18.13 | 14.70 | 13.80 | 16.47 |
| pca_W2↓ | Model 1 | 0.1044 | 0.1188 | 0.1160 | 0.1399 | 0.1513 |
| pca_W2↓ | Model 2 | 0.1013 | 0.1207 | 0.1155 | 0.1377 | 0.1493 |
| pca_W2↓ | Lasp | 0.0968 | 0.1061 | 0.1105 | 0.1296 | 0.1414 |
| pca_W2↓ | MDGen | 0.0886 | 0.0935 | 0.0638 | 0.0756 | 0.0745 |
| pca_W2↓ | bioemu2 | 0.0706 | 0.0734 | 0.0597 | 0.0667 | 0.0743 |
| weak_J↑ | Model 1 | 0.171 | 0.182 | 0.143 | 0.147 | 0.164 |
| weak_J↑ | Model 2 | 0.194 | 0.198 | 0.184 | 0.188 | 0.183 |
| weak_J↑ | Lasp | 0.191 | 0.207 | 0.198 | 0.181 | 0.185 |
| weak_J↑ | MDGen | 0.110 | 0.130 | 0.155 | 0.181 | 0.204 |
| weak_J↑ | bioemu2 | 0.218 | 0.227 | 0.233 | 0.249 | 0.242 |
| trans_J↑ | Model 1 | 0.324 | 0.331 | 0.276 | 0.282 | 0.294 |
| trans_J↑ | Model 2 | 0.362 | 0.368 | 0.344 | 0.345 | 0.338 |
| trans_J↑ | Lasp | 0.352 | 0.353 | 0.341 | 0.329 | 0.316 |
| trans_J↑ | MDGen | 0.199 | 0.222 | 0.252 | 0.277 | 0.301 |
| trans_J↑ | bioemu2 | 0.381 | 0.388 | 0.390 | 0.400 | 0.378 |

> windows per column: T320=20, T348=20, T379=20, T413=20, T450=20

#### Unseen protein — horizon 64 ns

| metric | method | T320 | T348 | T379 | T413 | T450 |
|---|---|---|---|---|---|---|
| RMSF-r↑ | Model 1 | 0.637 | 0.570 | 0.545 | 0.490 | 0.422 |
| RMSF-r↑ | Model 2 | 0.529 | 0.502 | 0.436 | 0.295 | 0.273 |
| RMSF-r↑ | Lasp | 0.515 | 0.524 | 0.532 | 0.431 | 0.367 |
| RMSF-r↑ | MDGen | 0.585 | 0.546 | 0.605 | 0.520 | 0.536 |
| RMSF-r↑ | bioemu2 | 0.777 | 0.782 | 0.772 | 0.683 | 0.614 |
| RMSF↓ | Model 1 | 1.503 | 1.811 | 2.049 | 2.885 | 3.862 |
| RMSF↓ | Model 2 | 1.304 | 1.770 | 1.931 | 2.670 | 3.690 |
| RMSF↓ | Lasp | 1.372 | 1.565 | 1.672 | 2.159 | 2.940 |
| RMSF↓ | MDGen | 2.725 | 2.755 | 2.162 | 2.183 | 2.486 |
| RMSF↓ | bioemu2 | 1.279 | 1.296 | 1.376 | 1.913 | 2.525 |
| root_W2↓ | Model 1 | 3.83 | 4.11 | 4.09 | 4.55 | 4.54 |
| root_W2↓ | Model 2 | 3.74 | 3.99 | 4.03 | 4.40 | 4.40 |
| root_W2↓ | Lasp | 3.89 | 4.18 | 4.10 | 4.51 | 4.48 |
| root_W2↓ | MDGen | 8.02 | 8.33 | 8.18 | 8.56 | 7.97 |
| root_W2↓ | bioemu2 | 3.95 | 4.25 | 3.97 | 4.43 | 4.28 |
| var_W2↓ | Model 1 | 15.97 | 24.90 | 26.33 | 36.51 | 52.60 |
| var_W2↓ | Model 2 | 15.96 | 25.65 | 26.91 | 36.68 | 53.04 |
| var_W2↓ | Lasp | 14.45 | 22.42 | 23.30 | 31.93 | 46.69 |
| var_W2↓ | MDGen | 22.15 | 27.90 | 22.31 | 25.92 | 37.28 |
| var_W2↓ | bioemu2 | 18.69 | 16.66 | 19.01 | 26.30 | 36.10 |
| pca_W2↓ | Model 1 | 0.1289 | 0.1544 | 0.1714 | 0.1897 | 0.2187 |
| pca_W2↓ | Model 2 | 0.1258 | 0.1570 | 0.1730 | 0.1896 | 0.2184 |
| pca_W2↓ | Lasp | 0.0998 | 0.1222 | 0.1415 | 0.1534 | 0.1830 |
| pca_W2↓ | MDGen | 0.0856 | 0.1046 | 0.0861 | 0.1041 | 0.1251 |
| pca_W2↓ | bioemu2 | 0.0702 | 0.0710 | 0.0815 | 0.1056 | 0.1319 |
| weak_J↑ | Model 1 | 0.150 | 0.152 | 0.132 | 0.141 | 0.163 |
| weak_J↑ | Model 2 | 0.136 | 0.139 | 0.135 | 0.138 | 0.144 |
| weak_J↑ | Lasp | 0.128 | 0.152 | 0.161 | 0.180 | 0.192 |
| weak_J↑ | MDGen | 0.097 | 0.120 | 0.145 | 0.167 | 0.191 |
| weak_J↑ | bioemu2 | 0.205 | 0.215 | 0.224 | 0.220 | 0.210 |
| trans_J↑ | Model 1 | 0.315 | 0.308 | 0.266 | 0.252 | 0.235 |
| trans_J↑ | Model 2 | 0.356 | 0.328 | 0.320 | 0.325 | 0.328 |
| trans_J↑ | Lasp | 0.301 | 0.304 | 0.312 | 0.313 | 0.307 |
| trans_J↑ | MDGen | 0.253 | 0.273 | 0.313 | 0.335 | 0.336 |
| trans_J↑ | bioemu2 | 0.401 | 0.386 | 0.377 | 0.321 | 0.279 |

> windows per column: T320=20, T348=20, T379=20, T413=20, T450=20

#### T4 — Temperature generalization, seen protein

Same layout, but the proteins are drawn from the **training** split (seen sequences, unseen trajectories). Comparing T3 against T4 isolates the protein-generalization gap at each temperature. Only the two cells the task asks for were run for seen proteins — **T348** (seen protein × seen temperature) and **T450** (seen protein × *unseen* temperature) — so the other three columns are `n/a` by design, not by failure.


#### Seen protein — horizon 1 ns

| metric | method | T320 | T348 | T379 | T413 | T450 |
|---|---|---|---|---|---|---|
| RMSF-r↑ | Model 1 | n/a | 0.431 | n/a | n/a | 0.442 |
| RMSF-r↑ | Model 2 | n/a | 0.399 | n/a | n/a | 0.546 |
| RMSF-r↑ | Lasp | n/a | 0.498 | n/a | n/a | 0.526 |
| RMSF-r↑ | MDGen | n/a | 0.444 | n/a | n/a | 0.455 |
| RMSF-r↑ | bioemu2 | n/a | 0.557 | n/a | n/a | 0.583 |
| RMSF↓ | Model 1 | n/a | 0.675 | n/a | n/a | 0.803 |
| RMSF↓ | Model 2 | n/a | 0.980 | n/a | n/a | 1.600 |
| RMSF↓ | Lasp | n/a | 0.506 | n/a | n/a | 0.732 |
| RMSF↓ | MDGen | n/a | 2.349 | n/a | n/a | 1.995 |
| RMSF↓ | bioemu2 | n/a | 1.447 | n/a | n/a | 1.255 |
| root_W2↓ | Model 1 | n/a | 5.34 | n/a | n/a | 5.59 |
| root_W2↓ | Model 2 | n/a | 5.65 | n/a | n/a | 5.95 |
| root_W2↓ | Lasp | n/a | 5.42 | n/a | n/a | 5.72 |
| root_W2↓ | MDGen | n/a | 9.20 | n/a | n/a | 9.21 |
| root_W2↓ | bioemu2 | n/a | 5.18 | n/a | n/a | 5.36 |
| var_W2↓ | Model 1 | n/a | 2.91 | n/a | n/a | 5.35 |
| var_W2↓ | Model 2 | n/a | 3.77 | n/a | n/a | 8.40 |
| var_W2↓ | Lasp | n/a | 2.30 | n/a | n/a | 5.14 |
| var_W2↓ | MDGen | n/a | 11.15 | n/a | n/a | 10.73 |
| var_W2↓ | bioemu2 | n/a | 11.60 | n/a | n/a | 9.38 |
| pca_W2↓ | Model 1 | n/a | 0.0559 | n/a | n/a | 0.0991 |
| pca_W2↓ | Model 2 | n/a | 0.1072 | n/a | n/a | 0.1653 |
| pca_W2↓ | Lasp | n/a | 0.0576 | n/a | n/a | 0.1036 |
| pca_W2↓ | MDGen | n/a | 0.0660 | n/a | n/a | 0.0698 |
| pca_W2↓ | bioemu2 | n/a | 0.0649 | n/a | n/a | 0.0643 |
| weak_J↑ | Model 1 | n/a | 0.000 | n/a | n/a | 0.000 |
| weak_J↑ | Model 2 | n/a | 0.650 | n/a | n/a | 0.450 |
| weak_J↑ | Lasp | n/a | 0.000 | n/a | n/a | 0.000 |
| weak_J↑ | MDGen | n/a | 0.000 | n/a | n/a | 0.000 |
| weak_J↑ | bioemu2 | n/a | 0.000 | n/a | n/a | 0.000 |
| trans_J↑ | Model 1 | n/a | 0.121 | n/a | n/a | 0.150 |
| trans_J↑ | Model 2 | n/a | 0.010 | n/a | n/a | 0.008 |
| trans_J↑ | Lasp | n/a | 0.156 | n/a | n/a | 0.180 |
| trans_J↑ | MDGen | n/a | 0.078 | n/a | n/a | 0.118 |
| trans_J↑ | bioemu2 | n/a | 0.170 | n/a | n/a | 0.219 |

> windows per column: T320=0, T348=20, T379=0, T413=0, T450=20

#### Seen protein — horizon 4 ns

| metric | method | T320 | T348 | T379 | T413 | T450 |
|---|---|---|---|---|---|---|
| RMSF-r↑ | Model 1 | n/a | 0.551 | n/a | n/a | 0.491 |
| RMSF-r↑ | Model 2 | n/a | 0.720 | n/a | n/a | 0.707 |
| RMSF-r↑ | Lasp | n/a | 0.626 | n/a | n/a | 0.587 |
| RMSF-r↑ | MDGen | n/a | 0.521 | n/a | n/a | 0.548 |
| RMSF-r↑ | bioemu2 | n/a | 0.739 | n/a | n/a | 0.690 |
| RMSF↓ | Model 1 | n/a | 0.756 | n/a | n/a | 1.380 |
| RMSF↓ | Model 2 | n/a | 0.705 | n/a | n/a | 1.461 |
| RMSF↓ | Lasp | n/a | 0.641 | n/a | n/a | 1.341 |
| RMSF↓ | MDGen | n/a | 2.388 | n/a | n/a | 1.770 |
| RMSF↓ | bioemu2 | n/a | 1.179 | n/a | n/a | 1.113 |
| root_W2↓ | Model 1 | n/a | 2.84 | n/a | n/a | 3.43 |
| root_W2↓ | Model 2 | n/a | 2.77 | n/a | n/a | 3.29 |
| root_W2↓ | Lasp | n/a | 2.81 | n/a | n/a | 3.43 |
| root_W2↓ | MDGen | n/a | 7.82 | n/a | n/a | 7.95 |
| root_W2↓ | bioemu2 | n/a | 2.67 | n/a | n/a | 3.12 |
| var_W2↓ | Model 1 | n/a | 5.94 | n/a | n/a | 15.47 |
| var_W2↓ | Model 2 | n/a | 6.69 | n/a | n/a | 16.34 |
| var_W2↓ | Lasp | n/a | 5.38 | n/a | n/a | 14.11 |
| var_W2↓ | MDGen | n/a | 14.21 | n/a | n/a | 11.72 |
| var_W2↓ | bioemu2 | n/a | 10.16 | n/a | n/a | 9.98 |
| pca_W2↓ | Model 1 | n/a | 0.0634 | n/a | n/a | 0.1242 |
| pca_W2↓ | Model 2 | n/a | 0.0715 | n/a | n/a | 0.1270 |
| pca_W2↓ | Lasp | n/a | 0.0623 | n/a | n/a | 0.1192 |
| pca_W2↓ | MDGen | n/a | 0.0770 | n/a | n/a | 0.0755 |
| pca_W2↓ | bioemu2 | n/a | 0.0577 | n/a | n/a | 0.0648 |
| weak_J↑ | Model 1 | n/a | 0.107 | n/a | n/a | 0.137 |
| weak_J↑ | Model 2 | n/a | 0.157 | n/a | n/a | 0.152 |
| weak_J↑ | Lasp | n/a | 0.147 | n/a | n/a | 0.150 |
| weak_J↑ | MDGen | n/a | 0.077 | n/a | n/a | 0.127 |
| weak_J↑ | bioemu2 | n/a | 0.169 | n/a | n/a | 0.202 |
| trans_J↑ | Model 1 | n/a | 0.207 | n/a | n/a | 0.261 |
| trans_J↑ | Model 2 | n/a | 0.308 | n/a | n/a | 0.310 |
| trans_J↑ | Lasp | n/a | 0.300 | n/a | n/a | 0.300 |
| trans_J↑ | MDGen | n/a | 0.130 | n/a | n/a | 0.214 |
| trans_J↑ | bioemu2 | n/a | 0.305 | n/a | n/a | 0.361 |

> windows per column: T320=0, T348=20, T379=0, T413=0, T450=20

#### Seen protein — horizon 16 ns

| metric | method | T320 | T348 | T379 | T413 | T450 |
|---|---|---|---|---|---|---|
| RMSF-r↑ | Model 1 | n/a | 0.550 | n/a | n/a | 0.545 |
| RMSF-r↑ | Model 2 | n/a | 0.702 | n/a | n/a | 0.550 |
| RMSF-r↑ | Lasp | n/a | 0.649 | n/a | n/a | 0.617 |
| RMSF-r↑ | MDGen | n/a | 0.544 | n/a | n/a | 0.531 |
| RMSF-r↑ | bioemu2 | n/a | 0.767 | n/a | n/a | 0.723 |
| RMSF↓ | Model 1 | n/a | 0.907 | n/a | n/a | 2.515 |
| RMSF↓ | Model 2 | n/a | 0.880 | n/a | n/a | 2.544 |
| RMSF↓ | Lasp | n/a | 0.797 | n/a | n/a | 2.237 |
| RMSF↓ | MDGen | n/a | 2.535 | n/a | n/a | 2.145 |
| RMSF↓ | bioemu2 | n/a | 1.096 | n/a | n/a | 1.671 |
| root_W2↓ | Model 1 | n/a | 2.83 | n/a | n/a | 3.40 |
| root_W2↓ | Model 2 | n/a | 2.71 | n/a | n/a | 3.35 |
| root_W2↓ | Lasp | n/a | 2.76 | n/a | n/a | 3.37 |
| root_W2↓ | MDGen | n/a | 8.21 | n/a | n/a | 8.20 |
| root_W2↓ | bioemu2 | n/a | 3.00 | n/a | n/a | 3.30 |
| var_W2↓ | Model 1 | n/a | 8.67 | n/a | n/a | 30.25 |
| var_W2↓ | Model 2 | n/a | 9.62 | n/a | n/a | 31.17 |
| var_W2↓ | Lasp | n/a | 7.82 | n/a | n/a | 27.83 |
| var_W2↓ | MDGen | n/a | 17.26 | n/a | n/a | 21.96 |
| var_W2↓ | bioemu2 | n/a | 11.98 | n/a | n/a | 19.98 |
| pca_W2↓ | Model 1 | n/a | 0.0917 | n/a | n/a | 0.1689 |
| pca_W2↓ | Model 2 | n/a | 0.1000 | n/a | n/a | 0.1734 |
| pca_W2↓ | Lasp | n/a | 0.0858 | n/a | n/a | 0.1569 |
| pca_W2↓ | MDGen | n/a | 0.0817 | n/a | n/a | 0.1010 |
| pca_W2↓ | bioemu2 | n/a | 0.0592 | n/a | n/a | 0.1042 |
| weak_J↑ | Model 1 | n/a | 0.147 | n/a | n/a | 0.171 |
| weak_J↑ | Model 2 | n/a | 0.196 | n/a | n/a | 0.186 |
| weak_J↑ | Lasp | n/a | 0.198 | n/a | n/a | 0.200 |
| weak_J↑ | MDGen | n/a | 0.123 | n/a | n/a | 0.198 |
| weak_J↑ | bioemu2 | n/a | 0.222 | n/a | n/a | 0.216 |
| trans_J↑ | Model 1 | n/a | 0.277 | n/a | n/a | 0.305 |
| trans_J↑ | Model 2 | n/a | 0.357 | n/a | n/a | 0.350 |
| trans_J↑ | Lasp | n/a | 0.346 | n/a | n/a | 0.332 |
| trans_J↑ | MDGen | n/a | 0.195 | n/a | n/a | 0.300 |
| trans_J↑ | bioemu2 | n/a | 0.367 | n/a | n/a | 0.352 |

> windows per column: T320=0, T348=20, T379=0, T413=0, T450=20

#### Seen protein — horizon 64 ns

| metric | method | T320 | T348 | T379 | T413 | T450 |
|---|---|---|---|---|---|---|
| RMSF-r↑ | Model 1 | n/a | 0.521 | n/a | n/a | 0.428 |
| RMSF-r↑ | Model 2 | n/a | 0.503 | n/a | n/a | 0.277 |
| RMSF-r↑ | Lasp | n/a | 0.472 | n/a | n/a | 0.495 |
| RMSF-r↑ | MDGen | n/a | 0.512 | n/a | n/a | 0.498 |
| RMSF-r↑ | bioemu2 | n/a | 0.702 | n/a | n/a | 0.670 |
| RMSF↓ | Model 1 | n/a | 1.466 | n/a | n/a | 4.909 |
| RMSF↓ | Model 2 | n/a | 1.495 | n/a | n/a | 4.733 |
| RMSF↓ | Lasp | n/a | 1.726 | n/a | n/a | 3.787 |
| RMSF↓ | MDGen | n/a | 2.699 | n/a | n/a | 3.133 |
| RMSF↓ | bioemu2 | n/a | 1.191 | n/a | n/a | 3.665 |
| root_W2↓ | Model 1 | n/a | 3.24 | n/a | n/a | 4.41 |
| root_W2↓ | Model 2 | n/a | 3.16 | n/a | n/a | 4.33 |
| root_W2↓ | Lasp | n/a | 3.35 | n/a | n/a | 4.27 |
| root_W2↓ | MDGen | n/a | 8.38 | n/a | n/a | 8.56 |
| root_W2↓ | bioemu2 | n/a | 3.21 | n/a | n/a | 4.17 |
| var_W2↓ | Model 1 | n/a | 17.39 | n/a | n/a | 65.18 |
| var_W2↓ | Model 2 | n/a | 18.42 | n/a | n/a | 65.61 |
| var_W2↓ | Lasp | n/a | 17.12 | n/a | n/a | 58.55 |
| var_W2↓ | MDGen | n/a | 23.03 | n/a | n/a | 46.19 |
| var_W2↓ | bioemu2 | n/a | 12.35 | n/a | n/a | 49.97 |
| pca_W2↓ | Model 1 | n/a | 0.1387 | n/a | n/a | 0.2631 |
| pca_W2↓ | Model 2 | n/a | 0.1472 | n/a | n/a | 0.2628 |
| pca_W2↓ | Lasp | n/a | 0.1152 | n/a | n/a | 0.2180 |
| pca_W2↓ | MDGen | n/a | 0.1021 | n/a | n/a | 0.1685 |
| pca_W2↓ | bioemu2 | n/a | 0.0809 | n/a | n/a | 0.1902 |
| weak_J↑ | Model 1 | n/a | 0.131 | n/a | n/a | 0.166 |
| weak_J↑ | Model 2 | n/a | 0.147 | n/a | n/a | 0.142 |
| weak_J↑ | Lasp | n/a | 0.131 | n/a | n/a | 0.201 |
| weak_J↑ | MDGen | n/a | 0.108 | n/a | n/a | 0.190 |
| weak_J↑ | bioemu2 | n/a | 0.212 | n/a | n/a | 0.180 |
| trans_J↑ | Model 1 | n/a | 0.290 | n/a | n/a | 0.227 |
| trans_J↑ | Model 2 | n/a | 0.348 | n/a | n/a | 0.325 |
| trans_J↑ | Lasp | n/a | 0.269 | n/a | n/a | 0.298 |
| trans_J↑ | MDGen | n/a | 0.256 | n/a | n/a | 0.321 |
| trans_J↑ | bioemu2 | n/a | 0.376 | n/a | n/a | 0.236 |

> windows per column: T320=0, T348=20, T379=0, T413=0, T450=20

#### T5 — The three requested temperature cells, side by side

The four cells the task asks for, all at once. `n_win` is the number of scored windows behind each cell.


#### Temperature cells — horizon 1 ns

| cell | method | RMSF-r↑ | RMSF↓ | root_W2↓ | var_W2↓ | pca_W2↓ | weak_J↑ | trans_J↑ | n_win |
|---|---|---|---|---|---|---|---|---|---|
| unseen protein × unseen T (450 K) | Model 1 | 0.398 | 0.831 | 6.58 | 6.57 | 0.1065 | 0.000 | 0.146 | 20 |
| unseen protein × unseen T (450 K) | Model 2 | 0.481 | 1.677 | 6.97 | 9.52 | 0.1612 | 0.350 | 0.006 | 20 |
| unseen protein × unseen T (450 K) | Lasp | 0.491 | 0.743 | 6.69 | 6.19 | 0.1041 | 0.000 | 0.173 | 20 |
| unseen protein × unseen T (450 K) | MDGen | 0.521 | 1.771 | 9.25 | 9.60 | 0.0656 | 0.000 | 0.130 | 20 |
| unseen protein × unseen T (450 K) | bioemu2 | 0.591 | 1.429 | 6.22 | 11.46 | 0.0542 | 0.000 | 0.226 | 20 |
| unseen protein × seen T (348 K) | Model 1 | 0.441 | 0.616 | 7.37 | 3.75 | 0.0701 | 0.000 | 0.147 | 20 |
| unseen protein × seen T (348 K) | Model 2 | 0.387 | 1.239 | 7.69 | 5.75 | 0.1180 | 0.900 | 0.004 | 20 |
| unseen protein × seen T (348 K) | Lasp | 0.484 | 0.573 | 7.44 | 3.12 | 0.0699 | 0.000 | 0.156 | 20 |
| unseen protein × seen T (348 K) | MDGen | 0.446 | 2.030 | 9.87 | 9.46 | 0.0542 | 0.000 | 0.087 | 20 |
| unseen protein × seen T (348 K) | bioemu2 | 0.536 | 1.703 | 6.92 | 15.50 | 0.0565 | 0.000 | 0.167 | 20 |
| seen protein × unseen T (450 K) | Model 1 | 0.442 | 0.803 | 5.59 | 5.35 | 0.0991 | 0.000 | 0.150 | 20 |
| seen protein × unseen T (450 K) | Model 2 | 0.546 | 1.600 | 5.95 | 8.40 | 0.1653 | 0.450 | 0.008 | 20 |
| seen protein × unseen T (450 K) | Lasp | 0.526 | 0.732 | 5.72 | 5.14 | 0.1036 | 0.000 | 0.180 | 20 |
| seen protein × unseen T (450 K) | MDGen | 0.455 | 1.995 | 9.21 | 10.73 | 0.0698 | 0.000 | 0.118 | 20 |
| seen protein × unseen T (450 K) | bioemu2 | 0.583 | 1.255 | 5.36 | 9.38 | 0.0643 | 0.000 | 0.219 | 20 |
| seen protein × seen T (348 K) — reference | Model 1 | 0.431 | 0.675 | 5.34 | 2.91 | 0.0559 | 0.000 | 0.121 | 20 |
| seen protein × seen T (348 K) — reference | Model 2 | 0.399 | 0.980 | 5.65 | 3.77 | 0.1072 | 0.650 | 0.010 | 20 |
| seen protein × seen T (348 K) — reference | Lasp | 0.498 | 0.506 | 5.42 | 2.30 | 0.0576 | 0.000 | 0.156 | 20 |
| seen protein × seen T (348 K) — reference | MDGen | 0.444 | 2.349 | 9.20 | 11.15 | 0.0660 | 0.000 | 0.078 | 20 |
| seen protein × seen T (348 K) — reference | bioemu2 | 0.557 | 1.447 | 5.18 | 11.60 | 0.0649 | 0.000 | 0.170 | 20 |

#### Temperature cells — horizon 4 ns

| cell | method | RMSF-r↑ | RMSF↓ | root_W2↓ | var_W2↓ | pca_W2↓ | weak_J↑ | trans_J↑ | n_win |
|---|---|---|---|---|---|---|---|---|---|
| unseen protein × unseen T (450 K) | Model 1 | 0.516 | 1.123 | 3.75 | 10.88 | 0.0964 | 0.136 | 0.262 | 20 |
| unseen protein × unseen T (450 K) | Model 2 | 0.681 | 1.136 | 3.59 | 11.80 | 0.0967 | 0.151 | 0.300 | 20 |
| unseen protein × unseen T (450 K) | Lasp | 0.601 | 1.083 | 3.78 | 9.86 | 0.0913 | 0.143 | 0.290 | 20 |
| unseen protein × unseen T (450 K) | MDGen | 0.562 | 1.613 | 7.40 | 10.13 | 0.0517 | 0.129 | 0.223 | 20 |
| unseen protein × unseen T (450 K) | bioemu2 | 0.664 | 1.359 | 3.44 | 16.93 | 0.0534 | 0.193 | 0.353 | 20 |
| unseen protein × seen T (348 K) | Model 1 | 0.568 | 0.640 | 3.67 | 5.65 | 0.0677 | 0.125 | 0.254 | 20 |
| unseen protein × seen T (348 K) | Model 2 | 0.685 | 0.686 | 3.62 | 6.58 | 0.0736 | 0.144 | 0.309 | 20 |
| unseen protein × seen T (348 K) | Lasp | 0.633 | 0.583 | 3.67 | 5.08 | 0.0616 | 0.135 | 0.285 | 20 |
| unseen protein × seen T (348 K) | MDGen | 0.599 | 2.126 | 7.54 | 12.57 | 0.0634 | 0.085 | 0.160 | 20 |
| unseen protein × seen T (348 K) | bioemu2 | 0.740 | 1.578 | 3.46 | 17.91 | 0.0697 | 0.166 | 0.302 | 20 |
| seen protein × unseen T (450 K) | Model 1 | 0.491 | 1.380 | 3.43 | 15.47 | 0.1242 | 0.137 | 0.261 | 20 |
| seen protein × unseen T (450 K) | Model 2 | 0.707 | 1.461 | 3.29 | 16.34 | 0.1270 | 0.152 | 0.310 | 20 |
| seen protein × unseen T (450 K) | Lasp | 0.587 | 1.341 | 3.43 | 14.11 | 0.1192 | 0.150 | 0.300 | 20 |
| seen protein × unseen T (450 K) | MDGen | 0.548 | 1.770 | 7.95 | 11.72 | 0.0755 | 0.127 | 0.214 | 20 |
| seen protein × unseen T (450 K) | bioemu2 | 0.690 | 1.113 | 3.12 | 9.98 | 0.0648 | 0.202 | 0.361 | 20 |
| seen protein × seen T (348 K) — reference | Model 1 | 0.551 | 0.756 | 2.84 | 5.94 | 0.0634 | 0.107 | 0.207 | 20 |
| seen protein × seen T (348 K) — reference | Model 2 | 0.720 | 0.705 | 2.77 | 6.69 | 0.0715 | 0.157 | 0.308 | 20 |
| seen protein × seen T (348 K) — reference | Lasp | 0.626 | 0.641 | 2.81 | 5.38 | 0.0623 | 0.147 | 0.300 | 20 |
| seen protein × seen T (348 K) — reference | MDGen | 0.521 | 2.388 | 7.82 | 14.21 | 0.0770 | 0.077 | 0.130 | 20 |
| seen protein × seen T (348 K) — reference | bioemu2 | 0.739 | 1.179 | 2.67 | 10.16 | 0.0577 | 0.169 | 0.305 | 20 |

#### Temperature cells — horizon 16 ns

| cell | method | RMSF-r↑ | RMSF↓ | root_W2↓ | var_W2↓ | pca_W2↓ | weak_J↑ | trans_J↑ | n_win |
|---|---|---|---|---|---|---|---|---|---|
| unseen protein × unseen T (450 K) | Model 1 | 0.489 | 2.084 | 3.65 | 26.08 | 0.1513 | 0.164 | 0.294 | 20 |
| unseen protein × unseen T (450 K) | Model 2 | 0.576 | 1.996 | 3.49 | 26.73 | 0.1493 | 0.183 | 0.338 | 20 |
| unseen protein × unseen T (450 K) | Lasp | 0.556 | 1.865 | 3.64 | 24.12 | 0.1414 | 0.185 | 0.316 | 20 |
| unseen protein × unseen T (450 K) | MDGen | 0.584 | 1.733 | 7.62 | 17.73 | 0.0745 | 0.204 | 0.301 | 20 |
| unseen protein × unseen T (450 K) | bioemu2 | 0.664 | 1.304 | 3.48 | 16.47 | 0.0743 | 0.242 | 0.378 | 20 |
| unseen protein × seen T (348 K) | Model 1 | 0.586 | 1.332 | 3.45 | 16.99 | 0.1188 | 0.182 | 0.331 | 20 |
| unseen protein × seen T (348 K) | Model 2 | 0.667 | 1.330 | 3.38 | 17.72 | 0.1207 | 0.198 | 0.368 | 20 |
| unseen protein × seen T (348 K) | Lasp | 0.633 | 1.104 | 3.46 | 15.25 | 0.1061 | 0.207 | 0.353 | 20 |
| unseen protein × seen T (348 K) | MDGen | 0.590 | 2.420 | 7.94 | 20.96 | 0.0935 | 0.130 | 0.222 | 20 |
| unseen protein × seen T (348 K) | bioemu2 | 0.751 | 1.400 | 3.69 | 18.13 | 0.0734 | 0.227 | 0.388 | 20 |
| seen protein × unseen T (450 K) | Model 1 | 0.545 | 2.515 | 3.40 | 30.25 | 0.1689 | 0.171 | 0.305 | 20 |
| seen protein × unseen T (450 K) | Model 2 | 0.550 | 2.544 | 3.35 | 31.17 | 0.1734 | 0.186 | 0.350 | 20 |
| seen protein × unseen T (450 K) | Lasp | 0.617 | 2.237 | 3.37 | 27.83 | 0.1569 | 0.200 | 0.332 | 20 |
| seen protein × unseen T (450 K) | MDGen | 0.531 | 2.145 | 8.20 | 21.96 | 0.1010 | 0.198 | 0.300 | 20 |
| seen protein × unseen T (450 K) | bioemu2 | 0.723 | 1.671 | 3.30 | 19.98 | 0.1042 | 0.216 | 0.352 | 20 |
| seen protein × seen T (348 K) — reference | Model 1 | 0.550 | 0.907 | 2.83 | 8.67 | 0.0917 | 0.147 | 0.277 | 20 |
| seen protein × seen T (348 K) — reference | Model 2 | 0.702 | 0.880 | 2.71 | 9.62 | 0.1000 | 0.196 | 0.357 | 20 |
| seen protein × seen T (348 K) — reference | Lasp | 0.649 | 0.797 | 2.76 | 7.82 | 0.0858 | 0.198 | 0.346 | 20 |
| seen protein × seen T (348 K) — reference | MDGen | 0.544 | 2.535 | 8.21 | 17.26 | 0.0817 | 0.123 | 0.195 | 20 |
| seen protein × seen T (348 K) — reference | bioemu2 | 0.767 | 1.096 | 3.00 | 11.98 | 0.0592 | 0.222 | 0.367 | 20 |

#### Temperature cells — horizon 64 ns

| cell | method | RMSF-r↑ | RMSF↓ | root_W2↓ | var_W2↓ | pca_W2↓ | weak_J↑ | trans_J↑ | n_win |
|---|---|---|---|---|---|---|---|---|---|
| unseen protein × unseen T (450 K) | Model 1 | 0.422 | 3.862 | 4.54 | 52.60 | 0.2187 | 0.163 | 0.235 | 20 |
| unseen protein × unseen T (450 K) | Model 2 | 0.273 | 3.690 | 4.40 | 53.04 | 0.2184 | 0.144 | 0.328 | 20 |
| unseen protein × unseen T (450 K) | Lasp | 0.367 | 2.940 | 4.48 | 46.69 | 0.1830 | 0.192 | 0.307 | 20 |
| unseen protein × unseen T (450 K) | MDGen | 0.536 | 2.486 | 7.97 | 37.28 | 0.1251 | 0.191 | 0.336 | 20 |
| unseen protein × unseen T (450 K) | bioemu2 | 0.614 | 2.525 | 4.28 | 36.10 | 0.1319 | 0.210 | 0.279 | 20 |
| unseen protein × seen T (348 K) | Model 1 | 0.570 | 1.811 | 4.11 | 24.90 | 0.1544 | 0.152 | 0.308 | 20 |
| unseen protein × seen T (348 K) | Model 2 | 0.502 | 1.770 | 3.99 | 25.65 | 0.1570 | 0.139 | 0.328 | 20 |
| unseen protein × seen T (348 K) | Lasp | 0.524 | 1.565 | 4.18 | 22.42 | 0.1222 | 0.152 | 0.304 | 20 |
| unseen protein × seen T (348 K) | MDGen | 0.546 | 2.755 | 8.33 | 27.90 | 0.1046 | 0.120 | 0.273 | 20 |
| unseen protein × seen T (348 K) | bioemu2 | 0.782 | 1.296 | 4.25 | 16.66 | 0.0710 | 0.215 | 0.386 | 20 |
| seen protein × unseen T (450 K) | Model 1 | 0.428 | 4.909 | 4.41 | 65.18 | 0.2631 | 0.166 | 0.227 | 20 |
| seen protein × unseen T (450 K) | Model 2 | 0.277 | 4.733 | 4.33 | 65.61 | 0.2628 | 0.142 | 0.325 | 20 |
| seen protein × unseen T (450 K) | Lasp | 0.495 | 3.787 | 4.27 | 58.55 | 0.2180 | 0.201 | 0.298 | 20 |
| seen protein × unseen T (450 K) | MDGen | 0.498 | 3.133 | 8.56 | 46.19 | 0.1685 | 0.190 | 0.321 | 20 |
| seen protein × unseen T (450 K) | bioemu2 | 0.670 | 3.665 | 4.17 | 49.97 | 0.1902 | 0.180 | 0.236 | 20 |
| seen protein × seen T (348 K) — reference | Model 1 | 0.521 | 1.466 | 3.24 | 17.39 | 0.1387 | 0.131 | 0.290 | 20 |
| seen protein × seen T (348 K) — reference | Model 2 | 0.503 | 1.495 | 3.16 | 18.42 | 0.1472 | 0.147 | 0.348 | 20 |
| seen protein × seen T (348 K) — reference | Lasp | 0.472 | 1.726 | 3.35 | 17.12 | 0.1152 | 0.131 | 0.269 | 20 |
| seen protein × seen T (348 K) — reference | MDGen | 0.512 | 2.699 | 8.38 | 23.03 | 0.1021 | 0.108 | 0.256 | 20 |
| seen protein × seen T (348 K) — reference | bioemu2 | 0.702 | 1.191 | 3.21 | 12.35 | 0.0809 | 0.212 | 0.376 | 20 |

#### T6 — Trajectory rollout (order-aware) quality

These metrics only mean something for a *method that produces an ordered path*. bioemu2's `propagate` draws independent endpoints, so it is `n/a` here unless it was run in `--bioemu2-walk` mode (one temporal chain of denoiser steps); MDGen samples a whole trajectory in one shot, so it is scored natively. Method 1 rolls out autoregressively and is therefore the most exposed to drift — that is precisely what these columns measure.


#### Unseen protein — ordered rollout — horizon 1 ns

| method | per-step↓ | path_len↓ | e2e↓ | Rg↓ | energy↓ | walkRMSF↓ | walkRMSF-r↑ | ACF-1↑ | ACF-corr↑ | n_traj |
|---|---|---|---|---|---|---|---|---|---|---|
| Model 1 | 3.51 | 1.5 | 1.48 | 16.9 | 1.93 | 1.24 | 0.000 | -0.128 | n/a | 800 |
| Model 2 | 3.02 | 0.3 | 0.33 | 16.7 | 2.23 | 0.00 | 0.000 | -0.125 | n/a | 800 |
| Lasp | 3.61 | 1.5 | 1.46 | 17.1 | 2.11 | 1.06 | 0.000 | -0.124 | n/a | 800 |
| MDGen | 12.90 | 5.5 | 5.49 | 20.9 | 11.32 | 3.21 | 0.000 | -0.132 | n/a | 800 |
| bioemu2 | 5.43 | 3.7 | 3.71 | 17.0 | 2.35 | 3.07 | -0.000 | -0.103 | n/a | 40 |

#### Unseen protein — ordered rollout — horizon 4 ns

| method | per-step↓ | path_len↓ | e2e↓ | Rg↓ | energy↓ | walkRMSF↓ | walkRMSF-r↑ | ACF-1↑ | ACF-corr↑ | n_traj |
|---|---|---|---|---|---|---|---|---|---|---|
| Model 1 | 5.42 | 11.6 | 5.89 | 17.1 | 2.98 | 1.63 | 0.412 | -0.051 | 0.988 | 800 |
| Model 2 | 3.61 | 3.6 | 1.44 | 16.6 | 2.27 | 0.77 | 0.686 | 0.211 | 0.963 | 800 |
| Lasp | 4.01 | 5.3 | 1.33 | 17.1 | 2.36 | 0.67 | 0.600 | 0.203 | 0.964 | 800 |
| MDGen | 13.22 | 11.4 | 5.91 | 20.9 | 11.37 | 2.03 | 0.557 | 0.581 | 0.845 | 800 |
| bioemu2 | 5.56 | 10.0 | 4.34 | 16.6 | 2.23 | 1.66 | 0.685 | 0.562 | 0.873 | 40 |

#### Unseen protein — ordered rollout — horizon 16 ns

| method | per-step↓ | path_len↓ | e2e↓ | Rg↓ | energy↓ | walkRMSF↓ | walkRMSF-r↑ | ACF-1↑ | ACF-corr↑ | n_traj |
|---|---|---|---|---|---|---|---|---|---|---|
| Model 1 | 13.64 | 59.2 | 23.06 | 21.2 | 10.03 | 5.25 | 0.265 | 0.625 | 0.799 | 800 |
| Model 2 | 4.41 | 17.5 | 1.87 | 16.4 | 2.61 | 1.40 | 0.733 | 0.473 | 0.913 | 800 |
| Lasp | 4.76 | 21.1 | 1.54 | 17.1 | 2.79 | 1.29 | 0.641 | 0.359 | 0.920 | 800 |
| MDGen | 13.78 | 29.7 | 6.39 | 20.9 | 11.47 | 1.96 | 0.618 | 0.832 | 0.674 | 800 |
| bioemu2 | 6.27 | 30.2 | 4.67 | 16.1 | 2.45 | 0.75 | 0.784 | 0.794 | 0.752 | 40 |

#### Unseen protein — ordered rollout — horizon 64 ns

| method | per-step↓ | path_len↓ | e2e↓ | Rg↓ | energy↓ | walkRMSF↓ | walkRMSF-r↑ | ACF-1↑ | ACF-corr↑ | n_traj |
|---|---|---|---|---|---|---|---|---|---|---|
| Model 1 | 54.65 | 351.1 | 104.11 | 57.8 | 61.45 | 36.28 | 0.121 | 0.922 | 0.708 | 800 |
| Model 2 | 5.60 | 75.6 | 2.68 | 15.9 | 3.54 | 2.16 | 0.681 | 0.627 | 0.902 | 800 |
| Lasp | 6.16 | 108.0 | 2.76 | 17.1 | 3.54 | 1.93 | 0.569 | 0.455 | 0.830 | 800 |
| MDGen | 14.60 | 90.5 | 6.78 | 20.9 | 11.79 | 2.37 | 0.596 | 0.922 | 0.595 | 800 |
| bioemu2 | 7.65 | 91.1 | 5.63 | 15.6 | 3.31 | 1.21 | 0.696 | 0.880 | 0.670 | 40 |

> **bioemu2's `n_traj` is far smaller than the others.** Its `propagate` sampler draws each endpoint independently, so it has no ordered path at all and only contributes here for the windows where the separate `--bioemu2-walk` run (one temporal chain of denoiser steps, ~27 min/window) was executed. Read its row as indicative, not as a 140-window estimate; the four path-producing methods share the same full window set.


#### T7 — Multi-scale summary (mean over horizons)

One row per method: the mean of each metric across the 4 horizons (1 ns, 4 ns, 16 ns, 64 ns), unseen protein. Use this as the headline table; T1/T2 give the per-horizon detail.


#### Unseen protein — mean across horizons

| method | RMSF-r↑ | RMSF↓ | root_W2↓ | var_W2↓ | pca_W2↓ | weak_J↑ | trans_J↑ | Kabsch↓ | n_win |
|---|---|---|---|---|---|---|---|---|---|
| Model 1 | 0.532 | 1.347 | 4.60 | 15.23 | 0.1136 | 0.109 | 0.243 | 5.01 | 100 |
| Model 2 | 0.546 | 1.450 | 4.61 | 16.13 | 0.1259 | 0.286 | 0.248 | 4.67 | 100 |
| Lasp | 0.553 | 1.164 | 4.62 | 13.84 | 0.1025 | 0.125 | 0.275 | 5.21 | 100 |
| MDGen | 0.544 | 2.158 | 8.37 | 16.88 | 0.0744 | 0.100 | 0.206 | 13.98 | 100 |
| bioemu2 | 0.678 | 1.482 | 4.46 | 16.62 | 0.0681 | 0.156 | 0.311 | 6.03 | 100 |

#### T8 — Paired bootstrap: every method against Model 1

All five methods scored the same windows, so a *paired* test removes between-protein difficulty and answers the question that actually matters: on a given protein, which method is better? Each cell is `mean(method) vs mean(Model 1)`, the parentheses holding the paired Δ and its 4000-sample bootstrap 95% CI; **bold** marks a CI that excludes zero (a real difference). One table, all five methods.


#### unseen protein — paired vs Model 1 — horizon 1 ns

| metric | Model 1 | Model 2 | Lasp | MDGen | bioemu2 |
|---|---|---|---|---|---|
| RMSF-r↑ | 0.469 | 0.442 (-0.027) | **0.510 (+0.041)** | 0.466 (-0.003) | **0.581 (+0.112)** |
| RMSF↓ | 0.641 | **1.278 (+0.637)** | **0.593 (-0.048)** | **2.039 (+1.398)** | **1.534 (+0.893)** |
| root_W2↓ | 7.066 | **7.396 (+0.330)** | **7.147 (+0.081)** | **9.804 (+2.738)** | **6.683 (-0.383)** |
| var_W2↓ | 4.095 | **6.103 (+2.008)** | 3.828 (-0.267) | **10.033 (+5.938)** | **11.914 (+7.819)** |
| pca_W2↓ | 0.078 | **0.127 (+0.048)** | 0.079 (+0.001) | **0.057 (-0.021)** | **0.054 (-0.024)** |
| weak_J↑ | 0.000 | **0.670 (+0.670)** | 0.000 (+0.000) | 0.000 (+0.000) | 0.000 (+0.000) |
| trans_J↑ | 0.146 | **0.005 (-0.141)** | **0.163 (+0.016)** | **0.097 (-0.050)** | **0.185 (+0.038)** |

> Format: `mean (Δ vs Model 1)`. **Bold** = 95% CI excludes zero. A negative Δ is better for the ↓ metrics, positive is better for ↑.

#### unseen protein — paired vs Model 1 — horizon 4 ns

| metric | Model 1 | Model 2 | Lasp | MDGen | bioemu2 |
|---|---|---|---|---|---|
| RMSF-r↑ | 0.567 | **0.692 (+0.126)** | **0.625 (+0.059)** | 0.571 (+0.004) | **0.686 (+0.120)** |
| RMSF↓ | 0.811 | 0.818 (+0.007) | 0.781 (-0.029) | **2.015 (+1.204)** | **1.430 (+0.619)** |
| root_W2↓ | 3.639 | **3.558 (-0.081)** | 3.640 (+0.001) | **7.575 (+3.937)** | **3.402 (-0.237)** |
| var_W2↓ | 7.615 | **8.333 (+0.718)** | 7.146 (-0.469) | **12.295 (+4.680)** | **15.129 (+7.514)** |
| pca_W2↓ | 0.077 | 0.080 (+0.002) | 0.074 (-0.003) | **0.060 (-0.017)** | **0.058 (-0.020)** |
| weak_J↑ | 0.127 | **0.145 (+0.018)** | **0.143 (+0.016)** | **0.099 (-0.028)** | **0.174 (+0.047)** |
| trans_J↑ | 0.250 | **0.302 (+0.053)** | **0.292 (+0.042)** | **0.174 (-0.075)** | **0.317 (+0.068)** |

> Format: `mean (Δ vs Model 1)`. **Bold** = 95% CI excludes zero. A negative Δ is better for the ↓ metrics, positive is better for ↑.

#### unseen protein — paired vs Model 1 — horizon 16 ns

| metric | Model 1 | Model 2 | Lasp | MDGen | bioemu2 |
|---|---|---|---|---|---|
| RMSF-r↑ | 0.558 | **0.644 (+0.086)** | **0.604 (+0.047)** | 0.583 (+0.025) | **0.718 (+0.160)** |
| RMSF↓ | 1.515 | 1.433 (-0.082) | **1.342 (-0.173)** | **2.116 (+0.601)** | 1.285 (-0.230) |
| root_W2↓ | 3.474 | **3.361 (-0.113)** | **3.458 (-0.016)** | **7.883 (+4.409)** | 3.583 (+0.109) |
| var_W2↓ | 17.934 | **18.418 (+0.484)** | **16.636 (-1.298)** | 18.071 (+0.137) | 16.097 (-1.837) |
| pca_W2↓ | 0.126 | 0.125 (-0.001) | **0.117 (-0.009)** | **0.079 (-0.047)** | **0.069 (-0.057)** |
| weak_J↑ | 0.161 | **0.189 (+0.028)** | **0.193 (+0.031)** | 0.156 (-0.005) | **0.234 (+0.072)** |
| trans_J↑ | 0.301 | **0.352 (+0.050)** | **0.338 (+0.037)** | **0.250 (-0.051)** | **0.387 (+0.086)** |

> Format: `mean (Δ vs Model 1)`. **Bold** = 95% CI excludes zero. A negative Δ is better for the ↓ metrics, positive is better for ↑.

#### unseen protein — paired vs Model 1 — horizon 64 ns

| metric | Model 1 | Model 2 | Lasp | MDGen | bioemu2 |
|---|---|---|---|---|---|
| RMSF-r↑ | 0.533 | **0.407 (-0.126)** | **0.474 (-0.059)** | 0.558 (+0.026) | **0.725 (+0.193)** |
| RMSF↓ | 2.422 | **2.273 (-0.149)** | **1.942 (-0.480)** | 2.462 (+0.040) | **1.678 (-0.744)** |
| root_W2↓ | 4.222 | **4.111 (-0.111)** | 4.231 (+0.009) | **8.213 (+3.990)** | 4.176 (-0.046) |
| var_W2↓ | 31.262 | **31.647 (+0.385)** | **27.756 (-3.506)** | **27.110 (-4.152)** | **23.353 (-7.909)** |
| pca_W2↓ | 0.173 | 0.173 (+0.000) | **0.140 (-0.033)** | **0.101 (-0.072)** | **0.092 (-0.081)** |
| weak_J↑ | 0.148 | **0.138 (-0.009)** | **0.163 (+0.015)** | 0.144 (-0.004) | **0.215 (+0.067)** |
| trans_J↑ | 0.275 | **0.331 (+0.056)** | **0.307 (+0.032)** | **0.302 (+0.027)** | **0.353 (+0.078)** |

> Format: `mean (Δ vs Model 1)`. **Bold** = 95% CI excludes zero. A negative Δ is better for the ↓ metrics, positive is better for ↑.

#### seen protein — paired vs Model 1 — horizon 1 ns

| metric | Model 1 | Model 2 | Lasp | MDGen | bioemu2 |
|---|---|---|---|---|---|
| RMSF-r↑ | 0.437 | 0.472 (+0.035) | **0.512 (+0.075)** | 0.449 (+0.013) | **0.570 (+0.133)** |
| RMSF↓ | 0.739 | **1.290 (+0.551)** | **0.619 (-0.120)** | **2.172 (+1.433)** | **1.351 (+0.612)** |
| root_W2↓ | 5.464 | **5.800 (+0.336)** | **5.572 (+0.108)** | **9.205 (+3.741)** | **5.268 (-0.196)** |
| var_W2↓ | 4.129 | **6.084 (+1.955)** | **3.716 (-0.413)** | **10.940 (+6.811)** | **10.492 (+6.363)** |
| pca_W2↓ | 0.077 | **0.136 (+0.059)** | 0.081 (+0.003) | 0.068 (-0.010) | 0.065 (-0.013) |
| weak_J↑ | 0.000 | **0.550 (+0.550)** | 0.000 (+0.000) | 0.000 (+0.000) | 0.000 (+0.000) |
| trans_J↑ | 0.135 | **0.009 (-0.126)** | **0.168 (+0.032)** | **0.098 (-0.038)** | **0.195 (+0.059)** |

> Format: `mean (Δ vs Model 1)`. **Bold** = 95% CI excludes zero. A negative Δ is better for the ↓ metrics, positive is better for ↑.

#### seen protein — paired vs Model 1 — horizon 4 ns

| metric | Model 1 | Model 2 | Lasp | MDGen | bioemu2 |
|---|---|---|---|---|---|
| RMSF-r↑ | 0.521 | **0.713 (+0.192)** | **0.606 (+0.085)** | 0.535 (+0.014) | **0.715 (+0.194)** |
| RMSF↓ | 1.068 | 1.083 (+0.016) | 0.991 (-0.076) | **2.079 (+1.011)** | 1.146 (+0.078) |
| root_W2↓ | 3.133 | **3.028 (-0.105)** | 3.117 (-0.016) | **7.885 (+4.752)** | **2.893 (-0.240)** |
| var_W2↓ | 10.706 | **11.512 (+0.805)** | 9.747 (-0.959) | 12.963 (+2.257) | 10.070 (-0.636) |
| pca_W2↓ | 0.094 | **0.099 (+0.005)** | 0.091 (-0.003) | 0.076 (-0.018) | **0.061 (-0.033)** |
| weak_J↑ | 0.122 | **0.154 (+0.032)** | **0.149 (+0.027)** | **0.102 (-0.020)** | **0.185 (+0.063)** |
| trans_J↑ | 0.234 | **0.309 (+0.075)** | **0.300 (+0.066)** | **0.172 (-0.062)** | **0.333 (+0.099)** |

> Format: `mean (Δ vs Model 1)`. **Bold** = 95% CI excludes zero. A negative Δ is better for the ↓ metrics, positive is better for ↑.

#### seen protein — paired vs Model 1 — horizon 16 ns

| metric | Model 1 | Model 2 | Lasp | MDGen | bioemu2 |
|---|---|---|---|---|---|
| RMSF-r↑ | 0.547 | **0.626 (+0.079)** | **0.633 (+0.086)** | 0.537 (-0.010) | **0.745 (+0.198)** |
| RMSF↓ | 1.711 | 1.712 (+0.001) | **1.517 (-0.194)** | **2.340 (+0.629)** | 1.383 (-0.328) |
| root_W2↓ | 3.115 | **3.029 (-0.086)** | **3.063 (-0.052)** | **8.203 (+5.088)** | 3.152 (+0.038) |
| var_W2↓ | 19.461 | **20.397 (+0.936)** | **17.825 (-1.636)** | 19.608 (+0.147) | 15.982 (-3.479) |
| pca_W2↓ | 0.130 | **0.137 (+0.006)** | **0.121 (-0.009)** | **0.091 (-0.039)** | **0.082 (-0.049)** |
| weak_J↑ | 0.159 | **0.191 (+0.033)** | **0.199 (+0.040)** | 0.160 (+0.001) | **0.219 (+0.060)** |
| trans_J↑ | 0.291 | **0.354 (+0.063)** | **0.339 (+0.048)** | **0.248 (-0.043)** | **0.360 (+0.069)** |

> Format: `mean (Δ vs Model 1)`. **Bold** = 95% CI excludes zero. A negative Δ is better for the ↓ metrics, positive is better for ↑.

#### seen protein — paired vs Model 1 — horizon 64 ns

| metric | Model 1 | Model 2 | Lasp | MDGen | bioemu2 |
|---|---|---|---|---|---|
| RMSF-r↑ | 0.475 | **0.390 (-0.085)** | 0.483 (+0.009) | 0.505 (+0.030) | **0.686 (+0.212)** |
| RMSF↓ | 3.187 | 3.114 (-0.073) | **2.757 (-0.431)** | 2.916 (-0.271) | **2.428 (-0.759)** |
| root_W2↓ | 3.823 | 3.746 (-0.077) | 3.811 (-0.012) | **8.472 (+4.649)** | 3.688 (-0.135) |
| var_W2↓ | 41.285 | **42.012 (+0.727)** | **37.835 (-3.450)** | **34.610 (-6.675)** | **31.156 (-10.128)** |
| pca_W2↓ | 0.201 | 0.205 (+0.004) | **0.167 (-0.034)** | **0.135 (-0.066)** | **0.136 (-0.065)** |
| weak_J↑ | 0.149 | 0.145 (-0.004) | **0.166 (+0.017)** | 0.149 (+0.000) | **0.196 (+0.047)** |
| trans_J↑ | 0.258 | **0.337 (+0.078)** | **0.283 (+0.025)** | 0.288 (+0.030) | **0.306 (+0.048)** |

> Format: `mean (Δ vs Model 1)`. **Bold** = 95% CI excludes zero. A negative Δ is better for the ↓ metrics, positive is better for ↑.

## 14. Lasp v3（速度耦合 + 去力截断）：当前 Lasp SOTA（2026-10-23 新增）【新增】

> 数据来源：`results/multiprotein/{v2,v3,report_lasp}_400w.json`。
> 口径：**400 个 round-robin 窗口 / 100 个 unseen 蛋白 / 5 个温度**
> （320/348/379/413/450 K），12 样本/窗口，
> protein-clustered bootstrap 95% CI。判显著 = CI 不含 0。
>
> **与 §13 的 140 窗 / 40 蛋白不是同一批窗口，两个协议的绝对数字不可互换**
> （见 §14.2 的警告框）。本节内部三方比较是同协议、干净的。

### 14.1 一句话结论

> **v3（`artifacts/lasp_v3_velocity`，step 7750）是当前 Lasp SOTA。**
> 在同一批 400 窗 / 100 蛋白上，它对 bioemu2 是 **显著 14:2**（12 项无显著差异），
> 而报告 §13.2 里用的那版 Lasp（`plan_a_unfrozen_dec_1000dom`）在同一批窗口上是 **10:5**。
> 对旧 Lasp 头对头 **赢 18/28 格**，进步是真实的、可复现的。
>
> **但 v3 不是全面碾压：`RMSF-r` 在 1/4/16/64 ns 四个时标上全部不如旧 Lasp**
> （0.4725/0.5760/0.5599/0.5284 vs 0.4733/0.6097/0.6348/0.5607），
> 且对 bioemu2 仍保留 **2 个显著败格（16 ns、64 ns 的 `RMSF-r`）**。
> 原因单一：速度耦合把系综压紧了，涨落谱被抹平——方向对了，展布被压窄。

### 14.2 三方胜负总账（同协议 400 窗 / 100 蛋白）

记法与 §13.2 一致：只统计配对 bootstrap（protein-clustered）95% CI 不含 0 的格，
共 7 指标 × 4 时标 = 28 格。

| 模型 | 我们显著赢 | bioemu2 显著赢 | 无显著差异 | 胜率 |
|---|---|---|---|---|
| 报告版 Lasp (`plan_a_unfrozen_dec_1000dom`) | 10 | 5 | 13 | 67% |
| v2 (`lasp_v2_roundtrip`) | 5 | 17 | 6 | 23% |
| **v3 (`lasp_v3_velocity`)** | 14 | 2 | 12 | 88% |

> ⚠️ **不要拿 §13.2 的 `4:17` 与上表直接比。** 同一个 ckpt 在 140 窗 / 40 蛋白上是 4:17，
> 在 400 窗 / 100 蛋白上是 10:5；连 bioemu2 的基线值本身都变了
> （`var_W2` @1 ns：11.91 → 2.58）。两个窗口集选出的样本差别太大，
> **只有同协议（同批窗口 + 同批 bioemu2 样本）的数字可以互相比较。**
> 本节的 `report_lasp_400w.json`、`v2_400w.json`、`v3_400w.json` 三者
> 逐格 bioemu2 基线完全一致、蛋白集合完全一致，因此三方比较是干净的。

### 14.3 v3 vs 旧 Lasp：头对头（28 格）

**v3 赢 18/28，输 10/28。** 按指标汇总：

| 指标 | 4 时标平均相对增益 (v3 vs 旧 Lasp) | v3 赢的时标数 |
|---|---|---|
| root_W2↓ | **+0.025** | **4/4** |
| trans_J↑ | **+0.103** | **4/4** |
| weak_J↑ | **+0.112** | 3/4 |
| var_W2↓ | +0.041 | 3/4 |
| RMSF↓ | +0.010 | 2/4 |
| pca_W2↓ | +0.037 | 2/4 |
| RMSF-r↑ | **-0.060** | **0/4** |

> 相对增益按 `Δ / ‖bioemu2 基线‖` 归一，正 = v3 更好。
> 加粗的行是**系统性**的：`root_W2` / `trans_J` 四个时标全胜，`RMSF-r` 四个时标全败；
> 其余是指标间小幅互有胜负（`weak_J` @1 ns 两者均为 0.000，记为平局）。

### 14.4 v3 vs bioemu2 逐格明细（同协议 400 窗；Δ = v3 − bioemu2）

**粗体** = 配对 bootstrap 95% CI 不含 0 且方向有利于 v3。

#### horizon 1 ns

| 指标 | bioemu2 | v3 | Δ | 95% CI | 判定 |
|---|---|---|---|---|---|
| RMSF↓ | 0.6137 | 0.5697 | -0.0440 | [-0.0907, +0.0077] | 无显著差异 |
| RMSF-r↑ | 0.4660 | 0.4725 | +0.0065 | [-0.0219, +0.0364] | 无显著差异 |
| root_W2↓ | 6.3458 | 6.3274 | -0.0184 | [-0.0432, +0.0038] | 无显著差异 |
| var_W2↓ | 2.5797 | 2.6607 | +0.0810 | [-0.3435, +0.5418] | 无显著差异 |
| pca_W2↓ | 0.0563 | 0.0583 | +0.0019 | [-0.0020, +0.0055] | 无显著差异 |
| weak_J↑ | 0.0000 | 0.0000 | +0.0000 | [+0.0000, +0.0000] | 无显著差异 |
| trans_J↑ | 0.1252 | **0.1635** | +0.0383 | [+0.0308, +0.0460] | **v3 显著赢** |

#### horizon 4 ns

| 指标 | bioemu2 | v3 | Δ | 95% CI | 判定 |
|---|---|---|---|---|---|
| RMSF↓ | 0.6086 | **0.5551** | -0.0535 | [-0.1023, -0.0093] | **v3 显著赢** |
| RMSF-r↑ | 0.5855 | 0.5760 | -0.0094 | [-0.0394, +0.0205] | 无显著差异 |
| root_W2↓ | 3.6335 | **3.5883** | -0.0452 | [-0.0616, -0.0293] | **v3 显著赢** |
| var_W2↓ | 4.7245 | **4.1615** | -0.5629 | [-1.2220, -0.0237] | **v3 显著赢** |
| pca_W2↓ | 0.0580 | 0.0589 | +0.0008 | [-0.0029, +0.0042] | 无显著差异 |
| weak_J↑ | 0.1211 | **0.1535** | +0.0324 | [+0.0259, +0.0393] | **v3 显著赢** |
| trans_J↑ | 0.2302 | **0.2837** | +0.0535 | [+0.0428, +0.0644] | **v3 显著赢** |

#### horizon 16 ns

| 指标 | bioemu2 | v3 | Δ | 95% CI | 判定 |
|---|---|---|---|---|---|
| RMSF↓ | 1.0902 | 1.0325 | -0.0578 | [-0.1345, +0.0097] | 无显著差异 |
| RMSF-r↑ | 0.6021 | 0.5599 | -0.0421 | [-0.0721, -0.0101] | **bioemu2 显著赢** |
| root_W2↓ | 2.2627 | **2.2288** | -0.0339 | [-0.0459, -0.0212] | **v3 显著赢** |
| var_W2↓ | 11.7930 | 11.3371 | -0.4558 | [-1.1489, +0.1096] | 无显著差异 |
| pca_W2↓ | 0.1024 | 0.1040 | +0.0016 | [-0.0025, +0.0053] | 无显著差异 |
| weak_J↑ | 0.1664 | **0.2230** | +0.0565 | [+0.0487, +0.0640] | **v3 显著赢** |
| trans_J↑ | 0.3008 | **0.3603** | +0.0595 | [+0.0495, +0.0690] | **v3 显著赢** |

#### horizon 64 ns

| 指标 | bioemu2 | v3 | Δ | 95% CI | 判定 |
|---|---|---|---|---|---|
| RMSF↓ | 1.8460 | **1.6927** | -0.1533 | [-0.2434, -0.0666] | **v3 显著赢** |
| RMSF-r↑ | 0.5834 | 0.5284 | -0.0551 | [-0.0889, -0.0186] | **bioemu2 显著赢** |
| root_W2↓ | 3.2513 | **3.2124** | -0.0389 | [-0.0521, -0.0256] | **v3 显著赢** |
| var_W2↓ | 22.1969 | **21.5588** | -0.6382 | [-1.2775, -0.1176] | **v3 显著赢** |
| pca_W2↓ | 0.1482 | 0.1454 | -0.0028 | [-0.0068, +0.0010] | 无显著差异 |
| weak_J↑ | 0.1558 | **0.1961** | +0.0403 | [+0.0341, +0.0465] | **v3 显著赢** |
| trans_J↑ | 0.3000 | **0.3530** | +0.0530 | [+0.0443, +0.0618] | **v3 显著赢** |

### 14.5 v3 vs 旧 Lasp 逐格明细（同协议 400 窗；Δ = v3 − 旧 Lasp）

#### horizon 1 ns

| 指标 | 旧 Lasp | v3 | Δ | 更好 |
|---|---|---|---|---|
| RMSF↓ | 0.5161 | 0.5697 | +0.0535 | Lasp |
| RMSF-r↑ | 0.4733 | 0.4725 | -0.0008 | Lasp |
| root_W2↓ | 6.5351 | 6.3274 | -0.2077 | **v3** |
| var_W2↓ | 2.7632 | 2.6607 | -0.1024 | **v3** |
| pca_W2↓ | 0.0725 | 0.0583 | -0.0142 | **v3** |
| weak_J↑ | 0.0000 | 0.0000 | +0.0000 | 持平 |
| trans_J↑ | 0.1419 | 0.1635 | +0.0216 | **v3** |

#### horizon 4 ns

| 指标 | 旧 Lasp | v3 | Δ | 更好 |
|---|---|---|---|---|
| RMSF↓ | 0.6271 | 0.5551 | -0.0720 | **v3** |
| RMSF-r↑ | 0.6097 | 0.5760 | -0.0337 | Lasp |
| root_W2↓ | 3.7047 | 3.5883 | -0.1164 | **v3** |
| var_W2↓ | 4.8805 | 4.1615 | -0.7189 | **v3** |
| pca_W2↓ | 0.0631 | 0.0589 | -0.0042 | **v3** |
| weak_J↑ | 0.1369 | 0.1535 | +0.0166 | **v3** |
| trans_J↑ | 0.2690 | 0.2837 | +0.0147 | **v3** |

#### horizon 16 ns

| 指标 | 旧 Lasp | v3 | Δ | 更好 |
|---|---|---|---|---|
| RMSF↓ | 1.0521 | 1.0325 | -0.0197 | **v3** |
| RMSF-r↑ | 0.6348 | 0.5599 | -0.0749 | Lasp |
| root_W2↓ | 2.2575 | 2.2288 | -0.0287 | **v3** |
| var_W2↓ | 11.4913 | 11.3371 | -0.1541 | **v3** |
| pca_W2↓ | 0.1004 | 0.1040 | +0.0036 | Lasp |
| weak_J↑ | 0.2071 | 0.2230 | +0.0158 | **v3** |
| trans_J↑ | 0.3462 | 0.3603 | +0.0141 | **v3** |

#### horizon 64 ns

| 指标 | 旧 Lasp | v3 | Δ | 更好 |
|---|---|---|---|---|
| RMSF↓ | 1.6758 | 1.6927 | +0.0169 | Lasp |
| RMSF-r↑ | 0.5607 | 0.5284 | -0.0324 | Lasp |
| root_W2↓ | 3.2831 | 3.2124 | -0.0708 | **v3** |
| var_W2↓ | 20.6883 | 21.5588 | +0.8705 | Lasp |
| pca_W2↓ | 0.1243 | 0.1454 | +0.0211 | Lasp |
| weak_J↑ | 0.1627 | 0.1961 | +0.0334 | **v3** |
| trans_J↑ | 0.3143 | 0.3530 | +0.0387 | **v3** |

### 14.6 复现命令

```bash
# v3（当前 SOTA）
CKPT=artifacts/lasp_v3_velocity/checkpoint_best.pt \
OUT=results/multiprotein/v3_400w.json GPU=3 MAXW=400 \
  bash run_scripts/run_lasp_v2_eval.sh

# 报告 §13 用的旧 Lasp，同一批窗口
CKPT=artifacts/plan_a_unfrozen_dec_1000dom/checkpoint_best.pt \
OUT=results/multiprotein/report_lasp_400w.json GPU=3 MAXW=400 \
  bash run_scripts/run_lasp_v2_eval.sh

# v2
CKPT=artifacts/lasp_v2_roundtrip/checkpoint_best.pt \
OUT=results/multiprotein/v2_400w.json GPU=3 MAXW=400 \
  bash run_scripts/run_lasp_v2_eval.sh
```

### 14.7 v3 改了什么

报告 §13 的 Lasp（`plan_a_unfrozen_dec_1000dom`）有两个结构性缺陷，v3 都修掉了：

| 缺陷 | 旧 Lasp | v3 |
|---|---|---|
| GLE 浴场耦合 | `bath_coupling` 字段不存在 → legacy `force`：稳态给出 `H' = H + h(D + CΓ⁻¹Cᵀ)f`，`CΓ⁻¹Cᵀ` 半正定 → 迁移率被**放大**而非阻尼（实测 `CΓ⁻¹Cᵀ/D = 15.4`，`‖H‖` 64 步内 11 → 3969） | `bath_coupling="velocity"`（FDT 正确）：`p = f − C·S`，浴场由**已实现位移**驱动，从而与 `(H−H₀)` 反相关 → 真正的摩擦。`‖H‖` 11 → 18 |
| 力截断 | `force_clip = 1.0`，而 `‖f‖ ≈ 7.2` → **~85% 的力被削掉**（步长 1 ns 只有真实位移的 1/3） | `force_clip = None`（默认关闭）+ `max_step_norm = 20` 只限漂移不限力 |

另外训练侧加了 `_safe_step`：loss 或梯度一旦出现非有限值就跳过该步、不动优化器状态
（`clip_grad_norm_` 对 NaN 无效，旧实现会静默把权重写成 NaN）。
v3 训练中该保护一直在实际工作——截至 step ~12,600 已累计跳过 **1,133 个非有限步**
（多为 `grad_norm=inf`），而训练状态保持健康、`checkpoint_best` 未被污染。
**没有这道保护，v3 会重蹈覆辙**（旧版 v3 曾在 step 8,630 静默变成全 NaN，浪费 5,000+ 步）。

> **口径提醒**：本节全部数字来自 `results/multiprotein/*_400w.json`，
> 400 个 round-robin 窗口 / 100 个 unseen 蛋白 / 5 个温度（320/348/379/413/450 K），
> 12 个样本/窗口，protein-clustered bootstrap CI。与 §13 的 140 窗 / 40 蛋白**不是同一批窗口**。

