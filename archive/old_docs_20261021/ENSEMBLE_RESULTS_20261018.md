# ENSEMBLE RESULTS — 系综级目标（2026-10-18）

> 配套计划：`PLAN_ENSEMBLE_20261018.md`（T1–T7）。
> 忠实记录：所有数字均来自本机真实运行，附命令与日志路径。
> 对照协议（`PLAN_ENSEMBLE_20261018.md` §6）：**任何报告必须带 `persist` 两端锚点。**

---

## 0. 一句话结论

用**冻结的** Stage-A relpos encoder + anchored decoder，只训练一个新的
`GaussianHHead`（T5），以系综级 loss 对齐 **未来窗口** 的 RMSF / contact-prob /
internal-distance 分布：

- **首轮即超过 persistence（mdCATH val）**：`val total 2.949 vs persist 3.540`，
  **RMSF 项 0.865 vs 1.512（好 43%）**；contact/dist ≈ persistence。
- 说明单帧 H **能**通过一个小的采样头解码出**比"不动"更真实的柔性分布**，
  这是单帧确定性坐标解码（real-H 仅好 persistence ~1%，见 §A）**做不到**的。

---

## A. 前置事实（单帧确定性解码的负结果，来自 STAGE_A_REPAIR_RESULTS_20261018.md）

| 数据集 | persist | real-H | wrongH | zeroH |
| --- | --- | --- | --- | --- |
| mdCATH val（1–8 ns） | 4.540 | **4.495** | 4.497 | 14.8 |
| ATLAS val（1–8 ns） | **0.239** | 0.465 | 0.465 | 2.006 |

- mdCATH：real-H 仅比 persistence 好 ~1%；real ≈ wrongH ⇒ H 相对 anchor 几乎无增量。
- ATLAS：real-H (0.465) **反而差于** persistence (0.239) ⇒ 对"近刚性"轨迹，
  anchor 已是近完美预测，decoder 的 H 分支净负增益。
- 已独立复核 ATLAS persistence：直接对 npz 算 Kabsch CA RMSD，`1ah7_A_R1` 0.21–0.28、
  `1btk_A_R2` 0.21–0.36、`1iom_A_R3` 0.55–1.66 Å ⇒ 属实，非 bug。

⇒ **动机**：单帧确定性坐标解码在这两个数据集上都打不过 persistence 地板，
因此转向"预测系综统计量"而非"预测单帧坐标"。

---

## B. 方法（T3 + T4 + T5）

- **producer**（`models/ensemble_producer.py`）：冻结 encoder→H_anchor，
  `GaussianHHead` 输出 `(mu, log_sigma)`，`log_sigma∈[1e-3,5]`（bias 初始化 -4）；
  采样 `H_s = mu + sigma·eps`，`s=1..n`（n=8），经**冻结** anchored decoder
  解码为 `ca_samples [B,n,L,3]`。
- **目标**（`training/ensemble_targets.py`）：从**未来窗口**（多帧真值）算
  `rmsf [B,L]`、`contact [B,L,L]`（P(d<8Å)）、`dist_hist [B,P,bins]`。
- **loss**（`training/ensemble_losses.py`）：
  `L = w_rmsf·L_rmsf + w_contact·L_contact + w_dist·L_dist`
  （rmsf: masked L1；contact: masked L1；dist: per-pair 累积差分 soft-EMD）。
- **只训练 head**，encoder/decoder 冻结；绝不覆盖已有 ckpt（只读）。
- **persistence 对照**：同协议把 anchor 复制 n 次当 ensemble（`persistence_ensemble`）。

---

## C. NaN-backward 修复（T5 前置）

现象：forward `total=12.3680` 有限，但 step0 `gnorm=nan` ⇒ NaN 在 **backward**。

根因：`evaluation/ensemble_metrics.py::rmsf_profile` 用 `sqrt(var)`，
在 `var=0`（masked 残基方差恰为 0）处导数为 ∞；`NaN×0=NaN`。

修复：
- `rmsf_profile`: `torch.sqrt(sq.clamp_min(1e-12))`（forward 对真实残基精确，
  梯度在数值零处有限且为 0）。
- `pair_distance_hist`: 距离改 `sqrt(d²+1e-12)`。

验证：`_rmsf_loss` backward 的 grad `isfinite().all() == True`（修复前 NaN count=105）；
T1–T4 全部 **30 tests pass**。

---

## D. T5 训练（mdCATH，正式运行）

命令（后台，GPU 5）：

```
python -m protein_jepa_frame.training.train_stage_ensemble \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --b1a-ckpt     artifacts/stage_b1a_relpos_anchored/checkpoint_best.pt \
  --manifest     artifacts/mdcath_manifest_200dom/train.jsonl \
  --val-manifest artifacts/mdcath_manifest_200dom/val.jsonl \
  --output artifacts/stage_ensemble_relpos \
  --device cuda:5 --steps 4000 --batch-size 4 \
  --rmsf-weight 1.0 --contact-weight 10.0 --dist-weight 0.3 \
  --val-every 250 --val-batches 8
```

日志：`logs/t5_ensemble_20261019_114541.log`

### D.1 验证集曲线（model vs persistence）

| step | val total | val rmsf | val contact | val dist | pers total | pers rmsf |
| --- | --- | --- | --- | --- | --- | --- |
| 250 | **2.993** | **0.890** | 0.0128 | 6.581 | 3.540 | 1.512 |
| 500 | **2.949** | **0.865** | 0.0118 | 6.554 | 3.540 | 1.512 |
| 750 | **2.979** | **0.850** | 0.0138 | 6.636 | 3.540 | 1.512 |

- **RMSF（柔性谱）显著优于 persistence（~43%）**：这是系综级目标的核心收益。
- contact 略差于 persistence（0.012 vs 0.008），dist ≈ persistence。
- ⇒ 在 §6 的 "≥1 个主指标显著优于 persistence" 上：**RMSF 满足**（待 T6 给 95% CI）。

---

## E. T6 评估（model / persistence / oracle + bootstrap 95% CI）

脚本：`scripts/eval_ensemble_manifest.py`（NEW）。每个窗口作为独立样本，
model-vs-persistence 的差做 **2000 次 bootstrap**，报 95% CI（CI 不跨 0 才算显著）。
`n_samples=8`，权重 rmsf/contact/dist = 1/10/0.3。
`oracle` = 用真值未来窗口当系综（下界 ~0，验证实现正确）。

### E.1 mdCATH val（同分布，500 蛋白 val manifest）

结果文件：`artifacts/mdcath_ensemble_eval.json`（60 batch）。

| metric | model | persistence | oracle | model−persist (improvement) | 95% CI | 显著？ |
| --- | --- | --- | --- | --- | --- | --- |
| **total** | **2.844** | 3.572 | ~0 | **+0.728** | [0.630, 0.825] | ✅ 不跨 0 |
| **rmsf** | **0.813** | 1.648 | ~0 | **+0.835** | [0.736, 0.936] | ✅ 不跨 0 |
| contact | 0.0168 | 0.0105 | 0 | −0.0062 | [−0.0072, −0.0054] | ❌ 显著变差 |
| dist | 6.210 | 6.062 | 0 | −0.149 | [−0.176, −0.123] | ❌ 显著变差 |

⇒ **GO 判据满足**（`PLAN` §6.1：≥1 个主指标显著优于 persistence 且 95% CI 不跨 0）：
**total 与 RMSF 两项均显著更优**（RMSF 好 ~49%）。contact/dist 略差。

### E.2 ATLAS val（未见，10 蛋白 × 3 replicas；近刚性）

结果文件：`artifacts/atlas_ensemble_eval.json`（40 batch，27 轨迹）。

| metric | model | persistence | oracle | model−persist | 95% CI | 显著？ |
| --- | --- | --- | --- | --- | --- | --- |
| total | 0.408 | **0.181** | ~0 | −0.227 | [−0.242, −0.212] | ❌ 显著变差 |
| rmsf | 0.174 | **0.068** | ~0 | −0.106 | [−0.123, −0.090] | ❌ 显著变差 |
| contact | 0.00055 | **0.00015** | 0 | −0.00041 | [−0.0007, −0.0002] | ❌ 显著变差 |
| dist | 0.761 | **0.374** | ~0 | −0.388 | [−0.407, −0.369] | ❌ 显著变差 |

⇒ **在 ATLAS 上，model 全面且显著地输给 persistence。** 与 §A 的单帧结果一致：
ATLAS 轨迹近刚性（persistence 仅 0.24 Å），mdCATH 上训得的采样头在此**过预测**
（引入本不存在的柔性），净负增益。

### E.3 结论

1. **系综级 loss 在 mdCATH（同分布）上确实有效**：RMSF 谱显著优于 persistence
   （导入的是**真实柔性信息**，而单帧确定性解码做不到，见 §A）。
2. **但完全不跨数据集泛化到 ATLAS**：对近刚性轨迹，模型"想太多"。
   ⇒ 需要 **per-dataset / 温度分层** 校准，或让 head 学一个"柔性门控"
   （位移小时自动退化为 persistence）。
3. oracle 端为 ~0（实现正确），persistence 与 model 两端锚点齐全，符合 §6.3 报告纪律。

---

## F. 待办（T7 收尾 / T0）

- **T7**：把 §E 表并入 "vs 外部 baselines"（TICA+MSM / VAMPNet 1.967/2.226、SPIB 2.923 等；
  **latent 维度与 6-state 协议不同，需单独说明，勿混比**，见 `PLAN` §7）。
- **T0**（下一轮）：外部 mdCATH baseline（Timewarp/Str2Str/ESM-Flex）公开数字对齐。
- **泛化补强**：ATLAS 失败表明需要 (a) 温度/数据集分层校准，
  (b) 在 ATLAS train 上微调 head（ATLAS train 与 val 蛋白 disjoint，可行），
  作为下一轮实验。

