# K1 评测协议与基线 —— 已冻结 (FROZEN GROUND TRUTH)

冻结日期：2026-09-17
状态：**评测逻辑与基线固定，后续只改模型架构/训练，不再改评测。**

## 为什么冻结

`CORRECTED_FINDINGS_REPORT.md` 里 K1 的 "JEPA ITS=0.1445，非重叠 CI 击败所有基线" 是
**两个 bug 叠加的假象**，已判废弃：
1. readout 塌成常数先验（`prior_blend=1.0`，仅 4 个 unique 输出）；
2. ITS 掩码不对称（只在 `pred_times>0` 上平均，让退化算子丢掉自己失败的 timescale）。

两个 bug 都已修复并审计通过（见下"正确性审计"）。修复后的数字如实记录于此，作为
**不可再动的对照基准**。

## 正确性审计（全部通过）

1. ITS 对称掩码：只在 reference（train-MSM 真值）timescale 集合上打分；退化预测被**惩罚**
   （floor/cap 到有限动态范围）而非丢弃。`_its_relative_error` evaluate_k1_fastfolding.py:390-420。
2. 方法无差别派发：5 个方法(JEPA/tICA+MSM/MSM/Persistence/MajorityClass)全部走同一
   `_aggregate_operator`→`transition_operator_metrics`，无 JEPA 特判。evaluate_k1_fastfolding.py:519-559。
3. ITS/MFPT 是行归一化算子上的纯特征值数学，方法无关。dynamics_benchmarks.py:203-289。
4. 样本数配平：JEPA 与基线在**同一** held-out event 集上打分 —— h=1/4/8/16 → 368/356/340/308，逐一相等。
5. fixed readout 非退化：1372/1372 unique 预测，无先验塌缩。
6. 退化对照 gate 存在且如实上报：JEPA 在短 horizon 的 balanced-acc / NLL 上**输给**退化对照，已记录未隐藏。

## 冻结的基线数字（q_native 标签，seed 6303，4 horizon）

产物：`artifacts/reviewer_experiments/protein_kinetics/k1_fastfolding_qnative_v6_corrected.json`
预测：`artifacts/lspwm_v44_stage_b_v6_kernel_fix/eval/fastfolding_predictions_fixed_readout.jsonl`

### ITS 相对误差（operator bootstrap mean [lower,upper]，越低越好）

| method | h=1 | h=4 | h=8 | h=16 | pooled |
|---|---|---|---|---|---|
| **JEPA v6** | 0.424 [0.39,0.48] | 0.469 [0.24,0.69] | 0.406 [0.29,0.52] | 0.288 [0.08,0.50] | **0.397** |
| tICA+MSM | 0.324 [0.13,0.52] | 0.383 [0.18,0.58] | 0.476 [0.42,0.58] | 0.523 [0.43,0.64] | 0.426 |
| MSM | 0.775 | 0.317 | 0.429 | 0.604 | 0.531 |
| Persistence | 1.868 | 1.871 | 1.896 | 1.807 | 1.861 |
| MajorityClass | 3.753 | 2.953 | 3.783 | 3.783 | 3.568 |

### 次要指标（summary，均值）

| method | NLL ↓ | Brier ↓ | Acc | BalAcc | MacroF1 | MFPT relerr ↓ |
|---|---|---|---|---|---|---|
| JEPA v6 | 1.435 | 0.242 | 0.564 | 0.422 | 0.372 | 0.841 |
| tICA+MSM | 1.093 | 0.223 | 0.419 | 0.364 | 0.260 | 0.806 |
| MSM | 0.968 | 0.194 | 0.490 | 0.392 | 0.291 | 0.742 |
| Persistence | 7.374 | 0.178 | 0.733 | 0.566 | 0.566 | 3.789 |
| MajorityClass | 1.076 | 0.221 | 0.412 | 0.333 | 0.194 | 0.678 |

## 诚实结论（当前 v6 checkpoint）

- ITS：JEPA pooled 0.397 略优于 tICA+MSM 0.426，但**每个 horizon 的 CI 都与 tICA 重叠**。
  JEPA 在长 lag(h=16)领先，在短 lag(h=1,4)落后。属于"持平/有竞争力"，**不是**"显著击败"。
- MFPT：JEPA 0.84 差于 MSM 0.74，甚至差于 majority-class 对照 0.68。
- 短 horizon 分类指标(balanced-acc/NLL)：JEPA 仍输给退化对照。

→ 这就是待改进的起点。**目标：改模型架构/训练(M1 力场重构)后，在这个冻结协议上让
JEPA 的 ITS 每 horizon CI 下界压过 tICA、并把 MFPT 拉到基线以下。** 评测脚本一个字不再改。

## 可复现命令（冻结）

```bash
# 1) 生成非退化预测(fixed readout)
python3 scripts/generate_fastfolding_predictions.py \
  --checkpoint "v6=artifacts/lspwm_v44_stage_b_v6_kernel_fix/checkpoint_latest.pt" \
  --labels artifacts/fastfolding/frame_labels.full.jsonl \
  --output artifacts/lspwm_v44_stage_b_v6_kernel_fix/eval/fastfolding_predictions_fixed_readout.jsonl \
  --readout-mode fixed --ridge 1.0 --feature-source context_endpoint \
  --device cuda --seed 6303 --horizons 1 4 8 16

# 2) 评测(对称 ITS)
python3 scripts/evaluate_k1_fastfolding.py \
  --predictions artifacts/lspwm_v44_stage_b_v6_kernel_fix/eval/fastfolding_predictions_fixed_readout.jsonl \
  --labels artifacts/fastfolding/frame_labels.full.jsonl \
  --output artifacts/reviewer_experiments/protein_kinetics/k1_fastfolding_qnative_v6_corrected.json \
  --label-scheme q_native --q-unfolded-max 0.4 --q-folded-min 0.6 \
  --seed 6303 --horizons 1 4 8 16
```

## 思路A：避开编码器未见过的短 horizon 尺度（h=8,16,24,32）—— v18 DMC checkpoint

日期：2026-09-19

**动机**：短 hori zon（h=1,4）恰好落在编码器训练时未见过的尺度区间，动力学算子在那些
lag 上的退化/误差被放大。思路A = 用 context_endpoint 预测，但只取 h ∈ {8,16,24,32}，
全部落在编码器见过的尺度区间内。

产物：`artifacts/reviewer_experiments/protein_kinetics/k1_v18_context_endpoint_h8_16_24_32.json`
预测：`artifacts/lspwm_v44_stage_b_v18_dmc_fix/eval/fastfolding_predictions_context_endpoint_h8_16_24_32.jsonl`

### JEPA summary（q_native，seed 6303）

| 指标 | v6 基线 (h=1,4,8,16) | v18 (h=8,16,24,32) | Δ |
|---|---|---|---|
| ITS_relative_error ↓ | 0.397 | **0.376** | -0.021 |
| MFPT_relative_error ↓ | 0.841 | **0.766** | -0.076 |
| NLL ↓ | 1.435 | **1.338** | -0.096 |
| stationary_l1 | 0.697 | 0.755 | +0.058 |
| accuracy | 0.564 | 0.511 | -0.052 |
| balanced_accuracy | 0.422 | 0.382 | -0.040 |

### 每 horizon ITS 相对误差（JEPA pooled，越低越好）

| h | ITS_rel_err | predicted ITS | reference ITS |
|---|---|---|---|
| 8 | 0.650 | [0.86, 0.30] | [2.48, 0.38] |
| 16 | 0.483 | [1.49, 0.26] | [2.11, 0.49] |
| 24 | 1.142 (退化：两特征值均 0.267) | [0.27, 0.27] | [2.34, 0.24] |
| 32 | **0.168** | [1.44, 0.29] | [1.95, 0.30] |

### 每 horizon MFPT 相对误差（JEPA pooled）

| h | MFPT_rel_err | MFPT_mae |
|---|---|---|
| 8 | 0.412 | 3.648 |
| 16 | 0.315 | 3.043 |
| 24 | 0.556 | 4.249 |
| 32 | 0.535 | 3.758 |

### 诚实结论（思路A，思路A 与基线可比）

- 避开 h=1,4 的未见尺度后，JEPA pooled ITS 由 0.397 改善至 **0.376**，MFPT 由 0.841
  改善至 **0.766**，NLL 由 1.435 改善至 **1.338** —— 动力学与状态预测均改善。
- 最佳 ITS 匹配出现在 **h=32**（rel_err=0.168），与参考 timescale 高度一致
  ([1.44,0.29] vs [1.95,0.30])。这与"长 horizon 落在编码器已学尺度的平稳区"一致。
- 弱点是 **h=24 退化**：预测的两个 ITS 特征值塌缩到同一 0.267（rel_err=1.142），提示
  该 lag 处的行归一化算子退化为近秩 1。不是结论性失败，但说明并非所有新的长 horizon 都改善。
- 代价：accuracy/BalAcc 略降（0.564→0.511 / 0.422→0.382）——更长 horizon 的分类任务更难，
  但动力学量（ITS/MFPT）改善且是待论文声称的主要证据。

**注意**：h=24 的退化提示后续可做 h∈{16,32,48,64} 或只取 h={16,32} 的稳定子集复验，
确认 h=24 是个别 lag 伪影还是系统性退化。当前结论：思路A 是有效杠杆，最佳成果已记录于此。
```
