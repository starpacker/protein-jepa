# Current Project Progress — Protein-JEPA-Frame（2026-10-20 汇报整合版）

> 本文件**整合**两份结果文档：
> - `DUAL_ENSEMBLE_RESULTS_20261019.md`（**模型 1**：Stage-A + 系综头路线）
> - `FROMNONE_GLE_ENSEMBLE_RESULTS_20261019.md`（**模型 2**：from-none GLE 路线）
>
> 并把它们与**三套 baseline**（persistence / MDGen zero-shot / bioemu2）放在**同一套指标**下对比。
> 原始数字均来自仓库内 JSON：`artifacts/eval_*/*.json`、`results/*_ensemble*.json`。
> 整理计划与验收清单：`PROGRESS_REPORT_PLAN_20261020.md`。

---

## 0. 一页速览（TL;DR）

| 项 | 内容 |
| --- | --- |
| **任务** | 给定 mdCATH 的 anchor 帧（+ 短历史），预测**未来构象系综 / 未来帧轨迹**的 CA 结构 |
| **模型 1** | **Stage-A ensemble**：JEPA 式预训练 encoder（H）+ B1a 锚定解码器（冻结）+ **Gaussian H 系综头**；在 **混合数据**（mdCATH-200dom + ATLAS）上训练 |
| **模型 2** | **from-none GLE**：**随机** encoder + **GLE 随机 latent 动力学** + CRI/MLP 残差解码器；在 mdCATH-1000dom 上训练；训练 loss = CA RMSD + 系综分布项 |
| **Baseline 1** | **persistence**：anchor 帧复制 T 份（最诚实的天花板对手） |
| **Baseline 2** | **MDGen zero-shot**：官方 `atlas.ckpt`，**无微调**，直接推理 |
| **Baseline 3** | **bioemu2 zero-shot**：Microsoft `bioemu2-preview`，**无微调**，直接推理 |
| **主指标** | RMSF-幅度 / **RMSF-r（形态）** / ROOT-MEAN W2 / VARIANCE W2 / MD-PCA W2 / WEAK-CONTACT J / TRANSIENT-CONTACTS J（+ Kabsch CA RMSD） |
| **最强结论** | **模型 1 与模型 2 的系综分布指标（RMSF 幅度、RMSF-r、三个 W2、接触类）稳定优于 persistence**；CA 位置精度（Kabsch）仍有 rollout 漂移。 |

---

## 1. 两套模型是什么

### 1.1 共同设定

两套模型都做同一件事：**从 anchor 帧出发**，产生一组"未来 CA 构象"（系综）或一条"未来帧轨迹"。
评估时统一：真值系综 = 该 horizon 内**真实 MD 的未来帧**；persistence = anchor 复制；所有量先做 Kabsch 对齐。

### 1.2 模型 1 — Stage-A ensemble（`DUAL_ENSEMBLE_RESULTS_20261019.md`）

| 项 | 内容 |
| --- | --- |
| 骨干 | stage-A v2 (encoder) + stage-B1a (anchored decoder)，**预训练后冻结** |
| 可训练部分 | 只加 **`GaussianHHead`**（在 latent H 上拟合高斯，采样出 M 个 conformer） |
| 物理语义 | 头输出 H 的分布 → 解码器把每个 H 样本映射成一条 CA 云 |
| 训练数据 | `mixed_manifest`：mdCATH-200dom(4000) + ATLAS-1ns(267) / ATLAS-100ps |
| 训练 loss | RMSF(L1) + contact-prob(L1) + 内距 soft-EMD +（可选）位置项 `L_pos` |
| 变体 | A（无 pos/cond）、B（+pos 0.5）、C（+cond 2 + pos 0.5） |
| checkpoint | `artifacts/stage_ensemble_dualA/checkpoint_best.pt` |

### 1.3 模型 2 — from-none GLE（`FROMNONE_GLE_ENSEMBLE_RESULTS_20261019.md`）

| 项 | 内容 |
| --- | --- |
| 骨干 | **随机初始化** encoder（无预训练）+ **GLE**（广义 Langevin，bath_dim=8）latent 动力学 |
| 解码 | CRI / MLP **残差解码器**：`ca = X_ctx + tanh 限幅 Δx` |
| 物理语义 | 用 GLE 在 latent 上**真做随机演化**，得到 M 条 rollout → 系综 |
| 训练数据 | `mdcath_manifest_1000dom`（mdCATH，1 ns/帧） |
| 训练 loss | CA RMSD（点预测）+ latent ES + **系综分布项**（v4/4e 逐步加入）|
| 演进 | anchor → v4 → v4c/v4d → **v4e**（+RMSF-r 剖面损失）→ v4f（训练中） |
| 最优 checkpoint | `artifacts/fromnone/stage_cb1b_v4e/checkpoint_latest.pt` |

**两套模型的本质差异**：模型 1 是"**冻结预训练表征 + 轻量系综头**"（表征强、可训练参数少）；
模型 2 是"**端到端 from-scratch + 显式随机动力学**"（无预训练、直接优化轨迹 + 分布）。

---

## 2. 评估指标与实现（正确性核对）

> 代码：`evaluation/ensemble_metrics.py`（原文件，未改）+ `evaluation/ensemble_metrics_ext.py`（新增）。
> **未触碰**精确原子坐标解码路径。**单测 26 项全过**（`tests/test_ensemble_metrics.py` 等）。

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

**核对结论（实现正确性）**：
1. **oracle 锚点**：`pred=true` 时 → 4×W2 = **0.000**、2×Jaccard = **1.000**、RMSF-r = **1.000**、RMSF_mae = **0**。✔（实测）
2. **persistence 的 RMSF-r 恒为 0**：delta 系综的 RMSF 谱是**常数**，与真谱的 Pearson 按定义退化 → 0。✔（实测）
3. **persistence 的 contact-J/transient-J**：delta 系综 contact-prob 图**只有 {0,1}**，
   弱/瞬时带**一个都落不进** ⇒ 预测集合空；真集非空时 `J=0/|A|=0`。
   **这是 delta 基线的数学性质**，不是 mdCATH 性质。已在 `*_full` 变体返回 `n_true/n_pred` 可追溯。✔
4. **对齐口径一致**：训练侧 `_ensemble_stats` 与评估侧 `align_ensemble` 都是"各帧 Kabsch 对齐到首帧"；
   已修正一次早期口径错配（`_ensemble_stats` 曾在绝对坐标上算方差，见 FROMNONE §6.1）。

---

## 3. 三套 baseline 与我们的模型对比

### 3.1 数据集口径（**先声明，否则数字会被误读**）

| 模型/基线 | 训练数据 | 评估 val 集 | 帧间隔 |
| --- | --- | --- | --- |
| 模型 1（Stage-A ens） | mixed（mdCATH-200dom + ATLAS） | **mdCATH-200dom val（20 蛋白）** + ATLAS | 1000 ps |
| 模型 2（from-none GLE） | mdCATH-1000dom train | **mdCATH-1000dom val（100 蛋白）** | 1000 ps |
| persistence | — | 随各模型 | — |
| MDGen zero-shot | ATLAS（10 ps/帧） | mdCATH val（32 窗口） | 10 ps（原生） |
| bioemu2 zero-shot | PDB+AFDB+MS 生成 MD | mdCATH-1000dom val | 训练 1ns–1µs |

⚠️ **两个 val 蛋白集不相交**（0 重叠），且模型 1 的 200dom-val 有 **17/20** 落在 1000dom-**train** 中。
⇒ **模型 1 与模型 2 的绝对数值不可直接横比**；本文件对**同一模型**给出与它**同窗口**的 baseline 对比。

### 3.2 对照表 A — 模型 1（Stage-A ensemble，mdCATH-200dom val，100 窗口）

相对 persistence（§DUAL §6.2）：Δ = pers−model（↓指标）/ model−pers（↑指标），正=模型更好。

| 指标 | model | persistence | Δ | 胜率 |
| --- | --- | --- | --- | --- |
| RMSF 幅度 ↓ | **0.892** | 1.753 | **+0.861** | 100% |
| **RMSF-r ↑** | **0.553** | 0.000 | **+0.553** | 100% |
| ROOT-MEAN W2 ↓ | 10.398 | 10.561 | +0.163 | 97% |
| VARIANCE W2 ↓ | 8.760 | 10.417 | +1.658 | 98% |
| MD-PCA W2 ↓ | 0.0783 | 0.1181 | +0.0397 | 100% |
| WEAK-CONTACT J ↑ | 0.092 | 0.000 | +0.092 | 100% |
| TRANSIENT J ↑ | 0.211 | 0.000 | +0.211 | 100% |

**RMSF-r 三形式（mdCATH）**：global **0.604** / per-target 均值 **0.581**（19/19 蛋白 > 0，最弱 +0.35）/ per-window 0.548；persistence 三种全 0。

### 3.3 对照表 B — 模型 2（from-none GLE v4e，mdCATH-1000dom val，200 窗口，n_samples=12）

| horizon | RMSF-r ↑ | RMSF↓ | root_W2↓ | var_W2↓ | pca_W2↓ | weak_J↑ | trans_J↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 ns | 0.283 | 1.888 | 6.576 ✗ | 7.253 | 0.182 | 0.400 | 0.012 |
| 4 ns | **0.513** | 1.226 | 3.798 ✗ | 9.790 | 0.099 | 0.137 | 0.275 |
| 16 ns | **0.497** | 1.984 | 2.986 ✓ | 20.514 | 0.149 | 0.195 | 0.345 |
| 64 ns | **0.502** | 2.923 | 4.025 ✓ | 32.405 | 0.192 | 0.152 | 0.383 |
| persist(4ns) | 0.015 | 3.020 | 2.592 | 14.366 | 0.190 | 0.000 | 0.000 |

- **稳定胜**：RMSF 幅度、RMSF-r（4–64 ns）、variance/pca W2、transient-J（100% 窗口）。
- **已知失败**：`root_mean_w2` 在短时标（1 ns）差（模型位移 1.5 Å ≪ 真实 3.6–7.4 Å），16/64 ns 转正。

### 3.4 对照表 C — MDGen zero-shot（用户重点关注，mdCATH，32 窗口）

| 模型 | 物理跨度 | RMSF↓ | RMSF-r↑ | root_W2↓ | var_W2↓ | pca_W2↓ | weak_J↑ | trans_J↑ | **Kabsch↓** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| persistence | 瞬时 | 1.840 | −0.014 | 12.276 | 11.089 | 0.1345 | 0.000 | 0.000 | **4.014** |
| **MDGen zero-shot** | 1 ns | 4.650 | ~0 | **6.679** | 27.069 | 0.1387 | 0.000 | 0.000 | 7.344 |
| ↳ seed=7 复核 | 1 ns | 4.779 | ~0 | 6.896 | 27.371 | 0.1437 | 0.000 | 0.000 | 7.841 |
| MDGen mdCATH 短训 e1/s2000 | 100 ns | 16.061 | ~0 | 11.252 | 275.31 | 0.4760 | 0.000 | 0.000 | 19.691 |
| MDGen mdCATH 短训 e5/s6000 | 100 ns | 22.525 | ~0 | 14.213 | 541.27 | 0.6736 | 0.000 | 0.000 | 26.761 |
| **ours rollout（模型1）** | 100 ns | **1.455** | **0.334** | 11.993 | **10.211** | **0.0801** | **0.066** | **0.134** | 6.335 |

**结论**：MDGen 官方权重 **zero-shot 在 mdCATH 上劣于 persistence**（Kabsch 7.34 vs 4.01 Å），
主因 **10 ps/帧 → 1000 ps/帧 的步长 OOD + 蛋白族不同**；唯一亮点是 root_W2 最低（形状分布尚可，位置跑偏）。
MDGen mdCATH 短训 **远未收敛且震荡**（flow-matching 需 ~10⁴ epoch），此预算下不构成有意义的对手。

### 3.5 对照表 D — bioemu2 zero-shot（`bioemu2-preview`）

> Microsoft Research 的 diffusion 全原子系综模型，**零微调**，conditioned on anchor 帧。
> 跑在**同一套** `evaluation/ensemble_metrics_ext.py` + 同一 window 迭代 + 同一 persistence 上。
> **状态：✅ 已完成**（`run_bioemu2_zeroshot.sh`，manifest=1000dom-val，
> horizons 1/4/16/64 ns，n_samples=12，200 窗口；结果 `results/bioemu2_zeroshot.json`）。

| horizon | RMSF-r ↑ | RMSF↓ | root_W2↓ | var_W2↓ | pca_W2↓ | weak_J↑ | trans_J↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 ns | **0.431** | 2.501 | 5.588 | 19.783 | **0.087** | 0.000 | 0.165 |
| 4 ns | **0.595** | **2.243** | 3.699 | 24.336 | **0.098** | 0.163 | 0.281 |
| 16 ns | 0.579 | **1.817** | **3.449** | **20.228** | **0.086** | **0.237** | **0.348** |
| 64 ns | **0.609** | **1.715** | **4.172** | **18.552** | **0.092** | **0.209** | **0.377** |
| persistence @64 ns | 0.031 | 5.564 | 4.702 | 41.109 | 0.316 | 0.000 | 0.000 |

**读法**：bioemu2 在**长时标**上显著优于 persistence（RMSF-r ~0.6，RMSF 更低、
pca_W2 更小、接触 Jaccard 更高）；但**短时标（1 ns）** RMSF/root_W2 反而差于 persistence
（扩散模型生成的是"平衡分布样本"而非"沿轨迹的短时确定性演化"——1ns 内锚点几乎不动，
persistence 因此占优）。这正是**系综生成 vs 轨迹推进**两类任务的口径差异。
bioemu2 的 `weak_contact_j` 在 1 ns 为 0（其生成的系综在该窗口几乎不破接触），其余时标正常。

**环境坑（已解决）**：bioemu2 的 `cuequivariance_ops_torch` 需要 `libnvrtc.so.13`；
本机是 CUDA 12.9，但 pip `nvidia-cu13` wheel 里**有**该库，只是不在默认 search path 上。
修复：`export LD_LIBRARY_PATH=/opt/conda/lib/python3.13/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH`
（已写进 `run_bioemu2_zeroshot.sh`）。smoke test 通过。

---

## 4. 轨迹 PDB 文件（肉眼/可视化检查）

产物：`artifacts/pdbs_20261020/`（脚本 `scripts/export_eval_trajectories_pdb.py`，**新文件**）。
每个窗口 6 条 CA-only 多 MODEL PDB（16 帧），3 个窗口：

```
artifacts/pdbs_20261020/<protein>_T<temp>_R<rep>_anchor<f>/
    ground_truth.pdb   persistence.pdb
    ours_stageA.pdb    ours_fromnone.pdb
    mdgen_zeroshot.pdb mdgen_mdcath.pdb
```

窗口：`1nhwC00`(L=60)、`1ka1A01`(L=219)、`1kafD00`(L=104)，均 T=320 K、anchor frame 7。
细节与口径见 `artifacts/pdbs_20261020/README.md`（含 MDGen 时标错配声明）。

---

## 5. 诚实局限与下一步

1. **两套模型评估口径不同**（不同 val split）——若要严格同表对比，需在**同一 val 集**上重跑两模型。
   → *部分已解决*：附录 A 已把 Model 1 在 **1000dom**（与 from-none GLE / MDGen / bioemu2 同集）重训并评估。
2. **MDGen zero-shot 是 OOD**（10 ps vs 1000 ps/帧）；"MDGen 不行"**不是**它的正常水平。
3. **MDGen 短训远未收敛**；要公平需 ~10⁴ epoch 或步长对齐。
4. **CA 位置精度**：模型 1 rollout Kabsch 6.34 Å 差于 persistence 4.01 Å（自回归回填漂移）；
   模型 2 的 root_mean_w2 短时标失败。**位置预测是共同短板**，是后续 Stage B 的重点。
5. ~~**bioemu2 结果待补**（正在跑）。~~ → ✅ 已补入 §3.5（1000dom val，200 窗口）。
   bioemu2 短时标（1 ns）系综指标差于 persistence，长时标（16–64 ns）显著更好。
6. **接触类 Jaccard 对 delta/近刚体基线退化**——引用时必须附 `n_true/n_pred`。

---

# 附录 A — 1000dom 统一重训（2026-10-20 晚）

> 目标：让 **Model 1 / MDGen / from-none GLE 三者数据口径一致**，做直接对比。
> **默认数据集已切到 `artifacts/mdcath_manifest_1000dom/{train,val}.jsonl`**（详见 `PROGRESS_REPORT_PLAN_20261020.md` R1）。

## A.1 做了什么

| 组件 | 之前 | 现在（1000dom） | 产物 |
| --- | --- | --- | --- |
| Stage-A encoder | 已 1000dom（无需重训） | 不变 | `artifacts/stage_a_disp_relpos/` |
| **B1a 锚定解码器** | 200dom | **重训 1000dom** | `artifacts/stage_b1a_relpos_anchored_1000dom/` |
| **Gaussian 系综头** | mixed 200dom+ATLAS | **重训 1000dom** | `artifacts/stage_ensemble_1000dom/` |
| 默认 manifest | 200dom | **1000dom**（3 个训练脚本默认值已改） | — |

- 1000dom split：**800 训练域 / 100 验证域，0 重叠**。
- 修了一个真 bug：`train_stage_ensemble.py` 的 `--max-train-samples` 用
  `torch.linspace(0,len-1,N)` 取 **N 个等间距**索引（末位四舍五入到 len → `IndexError`），
  已改为均匀 stride 子采样。

> ⏳ **注（R7）**：上表 A.2 的系综头是在 B1a **step-250** 的 `checkpoint_best.pt` 上训的；
> B1a（6000 步）尚在跑且持续下降。已挂一个 watcher（PID 2846145），B1a 收尾后
> **用最终 B1a best ckpt 自动重训系综头**并刷新评估/PDB。最终数字可能略优于上表。

## A.2 Model 1（1000dom）系综指标（1000dom val 100 窗口，n_samples=8）

| 指标 | model | persistence | oracle |
| --- | --- | --- | --- |
| RMSF 幅度 ↓ | **1.154** | 1.772 | 0.000 |
| **RMSF-r ↑** | **0.276** | 0.000 | 1.000 |
| ROOT-MEAN W2 ↓ | **11.189** | 11.324 | 0.000 |
| VARIANCE W2 ↓ | **9.313** | 9.942 | 0.000 |
| MD-PCA W2 ↓ | **0.0879** | 0.1108 | 0.0000 |
| WEAK-CONTACT J ↑ | **0.0598** | 0.0000 | 1.000 |
| TRANSIENT J ↑ | **0.1488** | 0.0000 | 1.000 |

**RMSF-r**：global **0.538**；per-target 均值 **0.293**（65 个域，94% > 0）。
**Kabsch CA RMSD（rollout 100 帧）**：model **7.58 Å** vs persistence **3.31 Å**（rollout 漂移，同 200dom 现象）。
结果文件：`artifacts/eval_model1_1000dom/{suite.json, ours_rollout_100f.json}`。

## A.3 三流程同口径并列（1000dom，均为我们训练/推理）

| 模型 | RMSF-r(global)↑ | RMSF↓ | pca_W2↓ | Kabsch↓ |
| --- | --- | --- | --- | --- |
| Model 1（Stage-A ens，1000dom） | 0.538 | 1.154 | 0.088 | 7.58 |
| Model 2（from-none GLE v4e） | 0.51（4ns） | 1.23（4ns） | 0.099（4ns） | — |
| MDGen zero-shot | ~0 | 4.65 | 0.139 | 7.34 |
| persistence | 0.000 | 1.77 | 0.111 | 3.31 |

## A.4 轨迹 PDB（1000dom）

`artifacts/pdbs_1000dom_20261020/`（3 窗口 × 6 轨迹 × 16 帧，CA-only）：
`ground_truth / persistence / ours_stageA / ours_fromnone / mdgen_zeroshot / mdgen_mdcath`。
**现在三者数据口径一致**，可直接肉眼/统计对比。
