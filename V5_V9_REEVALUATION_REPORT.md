# v5 / v9 重新评估报告（LSP-WM Stage B 最新一轮）

评估日期：2026-09-16
评估对象：

| 标签 | checkpoint | 训练步数 | 说明 |
|---|---|---:|---|
| v3（旧参照，报告基线） | `artifacts/lspwm_v44_stage_b_v3_improved_thermodynamics/checkpoint_latest.pt` | 3000 | `EVALUATION_COMPARISON_REPORT.md` 引用的模型 |
| v5 | `artifacts/lspwm_v44_stage_b_v5_fixed_pmf/checkpoint_latest.pt` | 4600 / 计划 5000 | adaptive KDE mean-force + BAOAB + 4-step trajectory NLL |
| v9 | `artifacts/lspwm_v44_stage_b_v9_load_bath/checkpoint_latest.pt` | 3000 | 从 Stage A 载入已训练 bath（唯一改动） |

评估协议与 `EVALUATION_COMPARISON_REPORT.md` 保持一致：M1 严格时序 holdout（seed 6303、
`manifests/tmjepa_v3.validation.jsonl`、`dt=8000 ps`、history window 8、kernel length 4、
2000 次 protein bootstrap）。所有实验都在同一环境（`/opt/conda/bin/python3`，torch 2.8.0+cu129）
下重跑，v5 也以 seed 6303 重跑（仓库里旧的 `m1_strict_v5.json` 用的是 seed 6448 与另一个 venv，
与 v3 不可比，已弃用）。

---

## 1. 结论摘要（TL;DR）

**没有。v9 和 v5 都还没有超过 baseline；v9 目前是一个失效的训练 run，M1 实验甚至无法完成。**

| 实验 | 旧报告（v3） | v5（本轮） | v9（本轮） | 是否超过 baseline |
|---|---|---|---|---|
| **M1** 热力学闭合 | 完成，但从属（输给 GenericMarkov/GRU/CT） | 完成，从属（`stationary_jsd`、`acf_error` 显著更差） | **模型 30/30 rollout 全部发散，`Full` 无任何有效指标** | ❌ 否 |
| **R1** 表示闭合 | 归档数值不可比（协议不同） | Δ_hist 显著 > 0（相对增益 30–34%） | Δ_hist 显著 > 0（相对增益 31–37%） | ⚠️ 在“相对尺度”上优于 PCA/tICA/VAMP，但协议存在混杂，不能算胜出 |
| **K1** FastFolding 动力学 | JEPA 全面垫底（acc 0.274） | 好于 v3，但全面垫底（acc 0.650 vs 旧协议基线 0.98+）† | 全面垫底（acc 0.509） | ❌ 否 |
† 基线 0.98+ 来自旧 `state` 标签（98.9% 类不平衡），已退役。正确 q_native-label 协议下 MSM accuracy = 0.490。

核心事实：

1. **v9 训练发散**：多步 trajectory NLL 从第 960 步开始爆炸，第 1700 步后达到 1e12 量级，
   结束时训练 loss 为 `1.37e15`；dynamics NLL 在整个训练过程中**恒定在 235.3，从未下降**；
   LR 在 3000 步内衰减到 `2.93e-10`，优化实际已停止。作为直接后果，M1 里 `Full` /
   `FullCalibrated` 在 30 条测试轨迹上全部触发 `rollout became non-finite`。
2. **v5 训练能收敛但物理项仍然很弱**：dynamics NLL 从 ~397 降到 ~106（dynamics 阶段最好），
   但 force loss 周期性出现 ~1e9 的尖峰；`Full` 的 `stationary_jsd` 与 `acf_error` 在
   protein-bootstrap 下**显著差于** `GenericMarkov`。
3. **R1 是唯一有正面信号的实验**，但那条正面信号来自“相对尺度”解读，且当前探针协议
   （current 用线性探针、history 用 MLP 探针）不可作为与坐标基线比较的严格证据。
4. 评测管线本身有两个必须修掉的缺陷（见第 6 节），其中一个会让 K1 出现**假胜**。

---

## 2. 训练健康度诊断（为什么结果是这样）

### 2.1 v9：多步 NLL 发散 + 学习率提前归零

| 指标 | v5 | v9 |
|---|---|---|
| `lspwm_dynamic_nll`（thermodynamic 阶段首/末） | 397.0 → 280.4 | 235.3 → 235.3 |
| `lspwm_dynamic_nll`（dynamics 阶段最好） | **106.2** | 235.3（无变化） |
| `lspwm_dynamic_nll`（训练最后一步） | 170.9 | 235.3 |
| `lspwm_trajectory_nll` 首次 > 1e6 | 从未发生 | **step 960** |
| `lspwm_trajectory_nll` > 1e12 的步数 | 0 | **130 步（自 step 1700 起）** |
| 训练最后一步 `loss` | 16.96 | **1.37e15** |
| 最后一步 LR | 1.64e-07 | **2.93e-10** |
| `lspwm.world_model.potential.network.4.weight` L2 | 0.0869 | **0.0018** |

- v9 的 `lspwm_curriculum_steps = [0, 200, 700, 1200, 1700, 3000]` 把全部课程压进 3000 步，
  而学习率在同样的 3000 步内从 1e-5 衰减到 ~1e-10。也就是说 **observation 阶段
  （step 1700–3000）几乎是在零学习率下空转**，而且这一阶段正是 trajectory NLL 已经
  爆炸、最需要修正的时候。
- 训练结束时 learned potential 的输出层 L2 只有 `0.0018`（v5 为 `0.0869`，v3 为 `0.0389`），
  即 **learned mean force 基本为 0**，模型没有学到可用的自由能梯度。
- 唯一“变好”的是 native bath：`||K[1:]||_F` 从 v3/v5 的 `3.87e-10` 提升到 `3.48e-3`，
  说明 Stage A 的 bath 确实被载入了。但由于 drift/diffusion 与之不匹配，rollout 立刻发散。

### 2.2 v5：能收敛但物理项不稳定

- `lspwm_force_loss` 在 step 1620、2340、2880、3060、3600、3960、4140、4320、4500
  出现 `1e8–1e10` 的尖峰（`loss` 同步跳到 `1e8–1e9`），说明 KDE mean-force 目标仍然
  存在数值不稳定。
- 训练实际停在 step 4660（计划 5000），`checkpoint_latest.pt` 为 step 4600。
- `paired_skill` 稳定在 0.8 左右，`lspwm_koopman_loss` 长期停在 -6.9，说明表征侧
  （Stage A 冻结）没问题，问题全在物理动力学侧。

---

## 3. M1：热力学闭合（严格时序 holdout）

协议与 `artifacts/reviewer_experiments/thermodynamic_closure/m1_strict.json` 完全一致，
seed = 6303。表中数值是 30 条测试轨迹的均值；每项指标都是**越小越好**。

### 3.1 v5（`artifacts/reviewer_experiments/m1_strict_v5_seed6303.json`）

| Method | prediction MSE | innovation W | FDT residual | stationary JSD | ACF error | rollout norm | semigroup defect |
|---|---:|---:|---:|---:|---:|---:|---:|
| Markov | 0.0023771 | 273.14 | 11.531 | 0.51141 | 0.77279 | 12.196 | 3.76e-06 |
| FreeEnergy | 0.0023771 | 273.14 | 14.488 | 0.49712 | 0.77277 | 11.565 | 4.83e-05 |
| Mori | 0.0052023 | 251.13 | 22.456 | **0.08309** | 0.73575 | 2.85e+05 | 2.357 |
| **Full（本模型）** | 0.0023772 | 273.14 | **1.0491** | 0.59660 | 0.76403 | 51.558 | 3.10e-04 |
| FullCalibrated | 0.0023771 | 273.14 | 1.3104 | 0.48130 | 0.76678 | 12.289 | 5.21e-06 |
| **GenericMarkov**（24/30 条轨迹有效） | **0.0022818** | 254.13 | 1.1581 | 0.47876 | 0.57022 | 7.693 | 0.35754 |
| GRU | 0.0024128 | **248.87** | 1.2870 | 0.35886 | **0.34947** | 8.333 | 0.62911 |
| CausalTransformer | 0.0026584 | 249.37 | 1.1972 | 0.38511 | 0.48063 | 8.573 | 0.63495 |

配对 protein-bootstrap（Full − baseline，负值 = Full 更好；“显著”指 95% CI 不含 0）：

| 对比 | 显著占优的指标 | 显著变差的指标 |
|---|---|---|
| Full vs GenericMarkov | `semigroup_mean_defect` | `stationary_jsd`、`acf_error` |
| Full vs GRU | `semigroup_mean_defect` | `innovation_whiteness`、`stationary_jsd`、`acf_error` |
| Full vs CausalTransformer | `semigroup_mean_defect` | `innovation_whiteness`、`stationary_jsd`、`acf_error` |
| Full vs Markov | `fdt_residual` | `stationary_jsd` |
| Full vs FreeEnergy | `fdt_residual` | `stationary_jsd` |
| Full vs Mori | `prediction_mse`、`fdt_residual`、`semigroup_mean_defect` | `innovation_whiteness`、`stationary_jsd`、`acf_error` |

**结论**：v5 在 `prediction_mse`（配对 diff −0.00133）、`fdt_residual`（配对 diff −0.514）
上的点估计偏向 Full，但 95% CI 都含 0，**不显著**；而 `stationary_jsd`
（`+0.0725`，CI `[0.0311, 0.1239]`）与 `acf_error`（`+0.1871`，CI `[0.1134, 0.2939]`）
是**显著更差**，且 GRU / CausalTransformer 在这两项上同样显著更好。
M1 不成立。

> 注：`GenericMarkov` 只有 24/30 条轨迹有效（其余 rollout 发散），所以上表的原始均值
> 与配对 bootstrap 的 diff 符号可能不一致——配对比较只使用两者共有的轨迹。

### 3.2 v9（`artifacts/reviewer_experiments/m1_strict_v9_seed6303.json`）

| Method | prediction MSE | innovation W | FDT residual | stationary JSD | ACF error | rollout norm | semigroup defect |
|---|---:|---:|---:|---:|---:|---:|---:|
| Markov | 0.0022593 | 273.33 | 11.617 | 0.51316 | 0.77071 | 12.171 | 5.38e-09 |
| FreeEnergy | 0.0022593 | 273.33 | 14.324 | 0.50404 | 0.77062 | 11.576 | 6.04e-08 |
| Mori | 0.0050702 | 250.41 | 23.750 | **0.05941** | 0.71110 | 9.41e+06 | 2.428 |
| **Full（本模型）** | **无法评估** | — | — | — | — | — | — |
| FullCalibrated | **无法评估** | — | — | — | — | — | — |
| GenericMarkov | 0.0021946 | 255.55 | 1.1488 | 0.50298 | 0.52921 | 7.686 | 0.36082 |
| GRU | 0.0024904 | 250.17 | 1.3261 | 0.38289 | 0.38956 | 8.637 | 0.64852 |
| CausalTransformer | 0.0027199 | 249.74 | 1.2128 | 0.38363 | 0.40240 | 8.431 | 0.62818 |

v9 的 `Full`/`FullCalibrated` 在 **全部 30 条测试轨迹**上抛出
`Full rollout became non-finite`（v9 日志共 67 条 WARNING，其中 60 条来自 Full/FullCalibrated），
因此 `n_test_trajectories = 0`、`per_protein = {}`、全部 summary 为 NaN、
`pairwise_full_minus_baseline` 全部为空。**这是本模型在 M1 上的硬失败，而不是“指标略差”。**

### 3.3 v3 / v5 / v9 对照

| 模型 | native `‖K[1:]‖_F` | Full prediction MSE | Full stationary JSD | Full ACF error | Full rollout norm | 结论 |
|---|---:|---:|---:|---:|---:|---|
| v3 | 3.87e-10 | 0.003013 | 0.58365 | 0.76861 | 124.43 | 完成，输给 neural 基线 |
| v5 | 3.87e-10 | 0.002377 | 0.56101 | 0.77137 | 78.23 | 完成，输给 neural 基线 |
| v9 | **3.48e-03** | 不可评估 | 不可评估 | 不可评估 | 不可评估 | 硬失败 |

> ⚠️ `scripts/evaluate_m1_strict.py:876` 把 `"closure_gate"` **硬编码为 `False`**，
> 它并不是由数据算出来的。旧报告“closure gate failed”的结论在数值上恰好成立，
> 但该字段本身目前不具备自动判定能力，见第 6 节。

---

## 4. R1：表示闭合（History Gap）

协议：`scripts/evaluate_history_gap.py`，20 个验证蛋白，hierarchical sampler，
100 batches × 8，seed 6303；current 探针 = `LinearProbe`，history 探针 = MLP `HistoryProbe`，
`Δ_hist = MSE_current − MSE_history`（> 0 表示 history 有帮助），
protein-level 2000 次 bootstrap。产物：
`artifacts/reviewer_experiments/representation_closure_v5/results.json`、
`.../representation_closure_v9/results.json`。

### 4.1 horizon = 1（结果最有意义的一档）

| 表示 | H | current loss | Δ_hist | 95% CI | 相对增益 Δ/current | 旧 gate（CI_low > 0.02） |
|---|---:|---:|---:|---|---:|---|
| **JEPA v3**（旧模型） | 2 | 0.025610 | **+0.008280** | [0.005980, 0.011065] | **+32.3%** | fail |
| **JEPA v5** | 2 | 0.017891 | **+0.005429** | [0.004082, 0.007012] | **+30.4%** | fail |
| JEPA v5 | 8 | 0.017891 | +0.006080 | [0.004560, 0.007910] | +34.0% | fail |
| **JEPA v9** | 2 | 0.017760 | **+0.005522** | [0.004102, 0.007280] | **+31.1%** | fail |
| JEPA v9 | 8 | 0.017760 | +0.006532 | [0.004790, 0.008710] | +36.8% | fail |
| PCA（CA 坐标） | 2 | 31.3099 | +0.7066 | [0.0822, 1.3196] | +2.26% | **PASS** |
| tICA（CA 坐标） | 2 | 0.25980 | **−0.0431** | [−0.0615, −0.0284] | −16.6% | fail（history 反而变差） |
| VAMP（CA 坐标） | 2 | 3.10870 | −0.0530 | [−0.2170, 0.0583] | −1.7% | fail（含 0） |

在 h = 2 / 4 / 8 / 16 / 32 / 64 上，v5 与 v9 的 `Δ_hist` **在所有 H∈{1,2,4,8} 上都是正值
且 CI 不含 0**；tICA 与 VAMP 在所有 horizon 上都是负值或与 0 交叉。

### 4.2 怎么解读（重要）

1. **正面信号（保留但需降级）**：在**相对尺度**上，JEPA 的 history gap（~30%）远大于
   CA 坐标 PCA（2.3%），而 tICA/VAMP 的 history gap 是负的。也就是说
   「JEPA 学到的状态已经接近 history 闭合，而坐标表示还需要 history」这一叙述
   **方向上是成立的**。但这一行对比里 PCA 用的是 CA 坐标 + 线性 ridge（另一套协议），
   JEPA 用的是 MLP history 探针，因此**不能直接读成「JEPA 优于 PCA」**。
2. **但不能算胜出**，原因有三：
   - 旧 gate 的阈值 `CI_lower > 0.02` 是**绝对 MSE 单位**，而 JEPA 的 loss 尺度
     （0.0179）比 PCA（31.3）小约 1750 倍，这个阈值在跨表示比较时没有意义。
     用相对准则（Δ/current > 2%）反而 JEPA 全面通过、PCA 只有 h=1/H=2 通过、
     tICA/VAMP 全 fail。**gate 需要重新定义。**
   - 该脚本的 current 探针是线性层、history 探针是 2 层 MLP，**两者容量不对等**，
     正 Δ_hist 里有多少来自表示、多少来自 MLP 容量，目前无法区分。
     仓库里另一套 `evaluate_r1_coordinate_baselines.py` 用的是**同一确定性 ridge 探针**，
     但那是 CA 坐标版本，两条协议没有打通（这正是旧报告 "Priority Improvements #1"）。
   - 两套协议的数据窗口也不同（坐标版按整条轨迹 80/20 切，JEPA 版按采样窗口池化）。
3. 为了量化“探针容量”这一混杂因素，本轮额外做了一个对照实验：
   用**同样的神经探针协议**，但把表示换成对 JEPA `frame_states` 做 PCA(16) 投影
   （见 4.3 节）。这是判断“Δ_hist 是表示性质还是探针性质”的关键对照。

### 4.3 对照实验：同一神经探针协议下的 PCA(JEPA frame_states)

产物：`artifacts/reviewer_experiments/representation_closure_v5_control_pca/results.json`

| 表示（同一神经探针协议） | h | H | current loss | Δ_hist | 95% CI | 相对增益 | 旧 gate |
|---|---:|---:|---:|---:|---|---:|---|
| PCA(JEPA frame_states, 16-d) | 1 | 2 | 7.13809 | +7.01908 | [6.25672, 7.92808] | **+98.3%** | **PASS** |
| PCA(JEPA frame_states, 16-d) | 8 | 8 | 7.12341 | +7.01356 | [6.25465, 7.91402] | +98.5% | **PASS** |
| PCA(JEPA frame_states, 16-d) | 64 | 8 | 7.15274 | +7.04264 | [6.27967, 7.95183] | +98.5% | **PASS** |
| JEPA v5（256-d） | 1 | 2 | 0.01789 | +0.00543 | [0.00408, 0.00701] | +30.4% | fail |
| JEPA v5（256-d） | 64 | 8 | 0.01776 | +0.00650 | [0.00478, 0.00865] | +36.6% | fail |

（PCA 对照在全部 28 个 (h, H) 组合上都 PASS，相对增益都在 98.2%–98.5%。）

**这个对照决定性地说明**：在当前探针协议下，**任意 16 维线性投影都会“通过” gate**，
因为 history 探针（MLP）比 current 探针（线性）容量大得多。因此不能拿
「JEPA 的 Δ_hist 显著为正」去论证 JEPA 优于 PCA/tICA/VAMP。

不过它同时给出一个**正面且可用的信息**：
JEPA 的 256 维状态已经比 16 维 PCA 投影**更接近 history 闭合**
（相对 gap 30% vs 98%，即 JEPA 状态本身携带了更多未来信息，history 的边际价值小得多），
而 tICA/VAMP 的 Δ_hist 为负，说明那两个表示的历史反而有害。
这个方向值得用**统一探针**再确认一次。

### 4.4 R1 结论

**R1 目前不能作为“优于 baseline”的证据**；但它提供了唯一一条正向线索，
且指明了下一步该做什么：把两套协议统一成**同一确定性 ridge 探针 + 相对尺度准则**，
再重跑 JEPA / PCA / tICA / VAMP / SRV。

---

## 5. K1：FastFolding 蛋白动力学

> **PROTOCOL NOTE:** The entire K1 section in this file uses the OLD `state` labels (`unfolded/transition/folded`) under 98.9% class imbalance. All accuracy values of 0.98-0.99 for MSM/tICA+MSM/Persistence are artifacts of the imbalanced label distribution — a constant `transition` predictor achieves accuracy 0.989. These are NOT the corrected FastFolding (scaffold-disjoint, q_native-label) protocol numbers. For the correct K1 protocol results (MSM accuracy = 0.490, tICA+MSM accuracy = 0.419, JEPA v6 accuracy = 0.564, Persistence accuracy = 0.733), see `CORRECTED_FINDINGS_REPORT.md`.

协议：4 个测试蛋白（`2F4K-0`、`CLN025-0`、`GTT-0`、`GTT-1`）、1372 个 held-out event sample、
scaffold-disjoint split、context 8 frames、horizons {1,4,8,16}、seed 6303、2000 次 bootstrap。
所有方法的标签与 split 完全相同（`labels_sha256 = 15d0ce73…`）。

### 5.1 关键发现：旧流水线的 “calibrated readout” 可以让模型权重归零

`scripts/generate_fastfolding_predictions.py` 的默认 readout 会在 validation 上搜索
`(ridge, temperature, prior_blend)`，目标是最小化 validation NLL/Brier。由于三分类严重不平衡，
搜索经常选中 **`prior_blend = 1.0`**，此时

```python
test_probabilities = (1.0 - prior_blend) * test_raw + prior_blend * prior[None, :]
```

**模型特征被乘了 0，预测退化成“每个 horizon 一个常数先验向量”**。证据：

| 文件 | 来源 checkpoint | 1372 行中不同行数 | 说明 |
|---|---|---:|---|
| `lspwm_v44_stage_b_v3_improved_thermodynamics/eval_improved/fastfolding_predictions_context_calibrated.jsonl` | v3 | **4** | 每个 horizon 一条常数 |
| `lspwm_v44_stage_b_detskill_fix_v3/fastfolding_predictions_context_calibrated.jsonl` | detskill_fix_v3 | **4** | 与上一行**逐位相同** |

两个**不同 checkpoint** 产生逐位相同的预测，只能是模型被旁路。而
`artifacts/reviewer_experiments/protein_kinetics/k1_context_calibrated.json` 里
「JEPA 反超所有 baseline」的漂亮数字（NLL 0.159、accuracy 0.990）正是这个常数先验预测器的成绩，
**不是模型成绩**，必须从比较中撤下。

对应的 `readout_calibration` 字段可自证：

```json
{"prior_blend": 1.0, "ridge": 0.01, "temperature": 0.5, "validation_n": 460}
```

### 5.2 本轮采取的做法

给 `scripts/generate_fastfolding_predictions.py` **新增一个可选开关**
`--readout-mode {validated,fixed}`（默认 `validated`，即原来的行为不变）：

- `fixed`：直接用 `--ridge`（本轮取 1.0）的 train-only ridge 打分，
  `temperature = 1.0`、`prior_blend = 0.0`，**不做任何 validation 校准**，
  因此不可能退化为常数预测器。
- 输出里记录 `readout_calibration.mode = "fixed"`，可区分。

在本轮所有 v3/v5/v9 的 prediction 文件里，`state_probs` 都有 1372 个互不相同的行。

### 5.3 结果（fixed readout，`--feature-source context_endpoint`）

> **PROTOCOL NOTE:** The numbers below use the OLD `state` labels (unfolded/transition/folded) under 98.9% class imbalance. The accuracies of 0.98-0.99 reflect the imbalanced label distribution, not true kinetic prediction performance — a constant `transition` predictor achieves accuracy 0.989. These are NOT the corrected q_native-label FastFolding protocol results. For the correct K1 protocol numbers (MSM accuracy = 0.490, tICA+MSM accuracy = 0.419, JEPA v6 accuracy = 0.564), see `CORRECTED_FINDINGS_REPORT.md`.

| Method | NLL ↓ | Brier ↓ | Accuracy ↑ | Row-JS ↓ | Stationary-L1 ↓ | ITS rel.err ↓ | MFPT rel.err ↓ | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 0.54035 | **0.01304** | 0.98044 | **0.01390** | **0.00737** | 0.41734 | **0.33652** | **0.01956** |
| MSM（train 拟合三态） | 0.22920 | 0.02954 | 0.98787 | 0.01787 | 0.54341 | 0.46007 | 0.53678 | 0.17193 |
| tICA+MSM | **0.21789** | 0.02780 | **0.98991** | 0.01457 | 0.51779 | **0.40002** | 0.52454 | 0.16565 |
| **JEPA v3** | 1.58182 | 0.29073 | 0.42855 | 0.10269 | 1.09614 | 0.91114 | 0.64262 | 0.42618 |
| **JEPA v5** | **0.99292** | **0.18660** | **0.65047** | **0.07110** | **0.97396** | 0.96518 | **0.62031** | **0.26018** |
| **JEPA v9** | 1.71195 | 0.26299 | 0.50884 | 0.09307 | 1.03931 | 0.93134 | 0.62195 | 0.37359 |
| JEPA v3（旧报告 legacy `forecast` readout，仅供参考） | 1.96124 | 0.34214 | 0.27439 | 0.11995 | 1.16273 | 0.94252 | 0.65563 | 0.51517 |

分 horizon 的 JEPA 表现：

| 模型 | h=1 acc / NLL | h=4 acc / NLL | h=8 acc / NLL | h=16 acc / NLL |
|---|---|---|---|---|
| v3 | 0.2418 / 2.3491 | 0.1573 / 1.8293 | 0.7176 / 0.7314 | 0.5974 / 1.4175 |
| v5 | 0.9810 / 0.2827 | 0.4803 / 1.2346 | 0.6471 / 0.9362 | 0.4935 / 1.5181 |
| v9 | 0.5245 / 0.8678 | 0.3455 / 1.8070 | 0.9706 / 0.4471 | 0.1948 / 3.7259 |

**结论**：三者在相同 readout 下的排序是 **v5 > v9 ≈ v3**，
即 v5 相对 v3 有实质提升（accuracy 0.429 → 0.650，NLL 1.582 → 0.993），
但**三者都全面垫底**：accuracy 0.43–0.65 vs baseline 0.980–0.990；
NLL 高出 4–8 倍；`stationary_l1`（0.97–1.10 vs Persistence 0.007）
说明模型预测的稳态分布与真实分布几乎不重叠。
committor 相关在 h=1 上甚至是**负的**（v5 Spearman −0.687，v9 −0.648，v3 +0.143），
意味着模型的 committor 与参考 committor 反向——这正是旧报告 K1 gate 失败的原因，本轮没有改善。

### 5.4 v3 对照（fixed readout）

> **PROTOCOL NOTE:** Same as above — these numbers use the OLD `state` labels with 98.9% class imbalance. The accuracy values of 0.98-0.99 are artifacts of the imbalanced labels, not genuine kinetic prediction performance. For corrected q_native-label results, see `CORRECTED_FINDINGS_REPORT.md`.

产物：`artifacts/reviewer_experiments/protein_kinetics/k1_v3_fixed_readout.json`

| Method | NLL ↓ | Brier ↓ | Accuracy ↑ | Row-JS ↓ | Stationary-L1 ↓ | ITS rel.err ↓ | MFPT rel.err ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 0.54035 | **0.01304** | 0.98044 | **0.01390** | **0.00737** | 0.41734 | **0.33652** |
| MSM（train 拟合三态） | 0.22920 | 0.02954 | 0.98787 | 0.01787 | 0.54341 | 0.46007 | 0.53678 |
| tICA+MSM | **0.21789** | 0.02780 | **0.98991** | 0.01457 | 0.51779 | **0.40002** | 0.52454 |
| JEPA v3（fixed readout） | 1.58182 | 0.29073 | 0.42855 | 0.10269 | 1.09614 | 0.91114 | 0.64262 |

v3 在相同 readout 下同样全面垫底，因此**“v5 比 v3 好”是真实提升，但没有改变相对 baseline 的败局**。

---

## 6. 本轮在评测管线里发现的两个缺陷（必须先修）

### 6.1 K1 的 “validated readout” 可以退化成常数先验预测器（会导致假胜）

- 位置：`scripts/generate_fastfolding_predictions.py::_fit_validated_readout`
- 机制：在 validation 上按 `(NLL, Brier)` 字典序选 `(ridge, temperature, prior_blend)`；
  三分类严重不平衡时最优解是 `prior_blend = 1.0`，此时
  `test_probabilities = 1.0 * validation_prior`，**模型特征的权重被清零**。
- 后果：`k1_context_calibrated.json` / `k1_detskill_context_calibrated.json` 里
  “JEPA 全面反超 baseline”的数字，实际上是一个常数预测器的成绩；
  两份来自不同 checkpoint 的预测文件**逐位相同**，是最直接的证据。
- 已处理：
  1. 给生成脚本新增 `--readout-mode {validated,fixed}`（默认 `validated`，行为不变）；
     本轮所有结论都基于 `fixed`。
  2. 建议把 `prior_blend = 1.0` 直接判为无效配置并抛错，
     或至少把 `prior_blend >= 0.5` 的结果标记为 `degenerate = true`。
- **请把 `k1_context_calibrated.json`、`k1_detskill_context_calibrated.json`
  从所有正在进行的对比材料里删除或标注为无效。**

### 6.2 M1 的 `closure_gate` 是硬编码常量

- 位置：`scripts/evaluate_m1_strict.py:876` → `"closure_gate": False`
- 它不读任何指标，无论模型多好都永远是 `False`。
- 旧报告「closure gate failed」的**结论**在数值上依然成立（Full 显著输在
  `stationary_jsd` / `acf_error`），但该字段本身没有判定能力，不能作为“gate”引用。
- 另外 M1 现在把 “rollout 发散” 静默降级为 per-trajectory `continue`，
  最终产出 `n_test_trajectories = 0` 但 `gates.strict_run_complete = True`。
  建议把「模型方法的有效轨迹数为 0」提升为硬错误，
  避免这种失败被读者误读成“指标略差”。

### 6.3 R1 的 gate 阈值是绝对尺度

- 位置：`scripts/evaluate_history_gap.py:216` → `if lo > 0.02`
- 0.02 是绝对 MSE，而 JEPA 的 loss 尺度（0.018）比 CA 坐标 PCA（31.3）小 1750 倍，
  跨表示比较时该阈值没有意义（见第 4 节）。
- 建议改为相对准则（如 `lo / mean_current_loss > 0.05`），
  并把 current / history 探针换成同族探针（都线性，或都 MLP），消除容量混杂。

---

## 7. 建议的下一步（按优先级）

1. **停掉 v9 这条线，先修训练**（当前最大的时间黑洞）：
   - `lspwm_curriculum_steps` 与 3000 步的 LR 衰减不匹配 → 把总步数提到 5000–8000，
     或让 LR 在 observation 阶段保持 ≥ 1e-6；
   - 给 `lspwm_trajectory_nll` 加**数值保护**（对增量做 clip、Huber 化，
     或对 `sᵀCs` 项做 spectral clip），并加“连续 N 步 traj_nll 超阈值就回滚 checkpoint”的保护；
   - 加训练中的 **rollout 有限性断言**（现在要等到 M1 评测阶段才发现发散）。
2. **bath 载入要带耦合一致性检查**：v9 已把 `‖K[1:]‖_F` 从 1e-10 提到 3.5e-3，
   但 Full rollout 立刻发散，说明 memory 项与当步 drift/diffusion 在 `dt = 8000 ps` 上不自洽。
   建议重新标定物理 `dt`，或先冻结 bath 只训 diffusion。
3. **统一 R1 探针协议**（旧报告 Priority #1，本轮再次确认这是唯一有正面信号的实验）：
   同一确定性 ridge 探针 + 相对尺度 gate，重跑 JEPA / PCA / tICA / VAMP / SRV。
4. **K1 换掉退化 readout**，并考虑直接评估模型的**转移算子**（`operator` 块）
   而不只是分类 readout；同时必须在报告里保留 `stationary_l1` 与 committor 相关，
   因为目前模型在这两项上与 baseline 差距最大。
5. **M1 的 `Full` 必须显式报告有效轨迹数**，任何 `n_test_trajectories < 30` 的运行
   都不应进入比较表。

> 关于已经存在的 `configs/lspwm_v44_stage_b_v10_balanced_weights.yaml`：
> 它把 `lspwm_trajectory_weight` 从 0.5 降到 0.02、`lspwm_dynamic_weight` 提到 0.3，
> 这对“traj NLL 主导总 loss”是正确的方向，但它**保留了
> `epochs: 3` / `lspwm_curriculum_steps: [0, 200, 700, 1200, 1700, 3000]`**，
> 也就是说第 2.1 节里“LR 在 3000 步内衰减到 ~1e-10、observation 阶段空转”
> 这个根因**没有修**。建议 v10 至少把总步数提到 5000–8000，
> 或让 `observation` 阶段的 LR 保持 ≥ 1e-6，再开跑。

---

## 8. 本轮产物与复现命令

### 8.1 新增/更新的产物

| 产物 | 内容 |
|---|---|
| `artifacts/reviewer_experiments/m1_strict_v5_seed6303.json` + `.log` | M1 严格评测（v5, seed 6303, `/opt/conda/bin/python3`） |
| `artifacts/reviewer_experiments/m1_strict_v9_seed6303.json` + `.log` | M1 严格评测（v9, seed 6303） |
| `artifacts/reviewer_experiments/representation_closure_v3/results.json` | R1（v3，本协议） |
| `artifacts/reviewer_experiments/representation_closure_v5/results.json` | R1（v5，本协议） |
| `artifacts/reviewer_experiments/representation_closure_v9/results.json` | R1（v9，本协议） |
| `artifacts/reviewer_experiments/representation_closure_v5_control_pca/results.json` | R1 对照：PCA(JEPA frame_states)+同一神经探针 |
| `artifacts/reviewer_experiments/protein_kinetics/k1_v3_fixed_readout.json` | K1（v3, fixed readout） |
| `artifacts/reviewer_experiments/protein_kinetics/k1_v5_fixed_readout.json` | K1（v5, fixed readout） |
| `artifacts/reviewer_experiments/protein_kinetics/k1_v9_fixed_readout.json` | K1（v9, fixed readout） |
| `artifacts/lspwm_v44_stage_b_{v3,v5,v9}/eval_improved/fastfolding_predictions_fixed_readout.jsonl` | 非退化 FastFolding 预测 |
| `artifacts/lspwm_v44_stage_b_{v5,v9}/eval_improved/eval_config.json` | 新增的评测配置 |

建议把 `/tmp` 下的两个 M1 JSON 归档到
`artifacts/reviewer_experiments/`（例如 `m1_strict_v5_seed6303.json`、
`m1_strict_v9_seed6303.json`），`/tmp` 会在重启后丢失。

### 8.2 代码改动

| 文件 | 改动 |
|---|---|
| `scripts/generate_fastfolding_predictions.py` | 新增 `--readout-mode {validated,fixed}`（默认值保持原行为）；`fixed` 时使用 train-only ridge、`temperature=1.0`、`prior_blend=0.0`；输出里记录 `readout_mode` 与 `readout_calibration.mode` |

### 8.3 复现命令

```bash
cd /liuyutian/protein_jepa_frame

# 1) M1（v5 / v9）
/opt/conda/bin/python3 -u scripts/evaluate_m1_strict.py \
  --checkpoint artifacts/lspwm_v44_stage_b_v5_fixed_pmf/checkpoint_latest.pt \
  --config artifacts/lspwm_v44_stage_b_v5_fixed_pmf/eval_improved/eval_config.json \
  --manifest manifests/tmjepa_v3.validation.jsonl \
  --output /tmp/m1_v5_seed6303.json \
  --max-batches 50 --batch-size 32 --device cuda:0 \
  --clip-frames 24 --window-stride 8 --history-window 8 --kernel-length 4 \
  --neural-epochs 200 --rollout-steps 256 --seed 6303

# 2) R1
/opt/conda/bin/python3 -u scripts/evaluate_history_gap.py --model lspwm \
  --checkpoint artifacts/lspwm_v44_stage_b_v5_fixed_pmf/checkpoint_latest.pt \
  --config artifacts/lspwm_v44_stage_b_v5_fixed_pmf/eval_improved/eval_config.json \
  --manifest manifests/tmjepa_v3.validation.jsonl \
  --output-dir artifacts/reviewer_experiments/representation_closure_v5 \
  --horizons 1,2,4,8,16,32,64 --history-lengths 1,2,4,8 \
  --max-batches 100 --batch-size 8 --device cuda:0 --representations lspwm

# 3) K1 预测（非退化 readout）+ K1 评测
/liuyutian/protein_jepa/.venv/bin/python -u scripts/generate_fastfolding_predictions.py \
  --checkpoint v5=artifacts/lspwm_v44_stage_b_v5_fixed_pmf/checkpoint_latest.pt \
  --labels artifacts/fastfolding/frame_labels.full.jsonl \
  --output artifacts/lspwm_v44_stage_b_v5_fixed_pmf/eval_improved/fastfolding_predictions_fixed_readout.jsonl \
  --context-frames 8 --horizons 1 4 8 16 \
  --feature-source context_endpoint --readout-mode fixed --ridge 1.0 \
  --batch-size 4 --device cuda --seed 6303

/liuyutian/protein_jepa/.venv/bin/python -u scripts/evaluate_k1_fastfolding.py \
  --labels artifacts/fastfolding/frame_labels.full.jsonl \
  --predictions artifacts/lspwm_v44_stage_b_v5_fixed_pmf/eval_improved/fastfolding_predictions_fixed_readout.jsonl \
  --output artifacts/reviewer_experiments/protein_kinetics/k1_v5_fixed_readout.json \
  --context-frames 8 --horizons 1 4 8 16 \
  --train-fraction 0.6 --validation-fraction 0.2 \
  --tica-lag 1 --tica-clusters 8 --bootstrap-samples 2000 --seed 6303
```

辅助分析脚本（临时用，未纳入正式管线）：
`.tmp_m1_compare.py`、`.tmp_r1_compare.py`、`.tmp_detach_launch.py`、
`.tmp_v9_ckpt_health.py`、`.tmp_v9_ckpt_health2.py`

---

## 9. 一句话回答“是否已经超过 baseline”

**还没有。**

- **M1**：v5 完整跑完但显著输在 `stationary_jsd` / `acf_error`；
  **v9 的动力学完全发散，`Full` 无法评测（30/30 rollout non-finite）**。
- **R1**：JEPA 在相对尺度上的 history 闭合性是三个模型里最好的方向性证据，
  但当前探针协议（线性 current vs MLP history）与绝对阈值都能让 16 维 PCA 通过，
  因此**不构成“优于 baseline”的证据**。
- **K1**：v5 比 v3 有真实进步（accuracy 0.429 → 0.650 in the old `state`-label protocol），
  但即使按正确的 q_native-label 协议（MSM accuracy = 0.490），JEPA 的 accuracy = 0.564
  仍落后于 Persistence baseline 的 0.733，且 committor 相关为负。

**当前第一优先级不是继续调 v9，而是先修掉 v9 的训练发散，再把第 6 节的三个评测缺陷补上。**

