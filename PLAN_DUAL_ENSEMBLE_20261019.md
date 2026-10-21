# PLAN — 双数据集训练 + 系综级目标 (+ 原子位置重建引导) 2026-10-19

> 承接：`PLAN_ENSEMBLE_20261018.md`（T1–T7 已全部完成）、
> `ENSEMBLE_RESULTS_20261018.md`、`STAGE_A_REPAIR_RESULTS_20261018.md`。
> 约束（继续有效）：**原先精确原子坐标解码代码留存、只读**；新代码开新文件；
> 不引入过量监督；不 `pkill -f`（用精确 PID）；用 `CLAUDE_CODE_TMPDIR`/`TMPDIR` 规避 tmpfs。

---

## 0. 用户需求（本轮）

1. **用两个数据集一起训练**（mdCATH + ATLAS）。
2. **target 系综级 metrics**（RMSF / contact-prob / 内距分布 / …）。
3. **原子位置重建（CA RMSD）作为引导指标**（auxiliary），不是唯一目标。

---

## 1. 现状盘点（2026-10-19 核实）

### 1.1 数据
| 数据 | manifest | 帧间隔 | 帧数/轨迹 | train | val | 温度 |
| --- | --- | --- | --- | --- | --- | --- |
| mdCATH 200dom | `artifacts/mdcath_manifest_200dom/{train,val}.jsonl` | 1000 ps | ~few hundred | 4000 | 500 | 320–450 K |
| **已混合** | `artifacts/mixed_manifest/{train,val}.jsonl` | 混合 | — | 4267 | 530 | 见下 |
| ATLAS 1ns | `artifacts/atlas_manifest/{train,val}.jsonl` | 1000 ps | 101 | 270 | 30 | 300 K |
| **ATLAS 100ps** | `artifacts/atlas_manifest_100ps/{train,val}.jsonl` | **100 ps** | **1001** | 270 | 30 | 300 K |

- `mixed_manifest` = mdCATH(4000) + ATLAS-1ns(267) train、mdCATH(500)+ATLAS(30) val。
- `load_manifest` **透明支持** npz(mdCATH h5 / ATLAS npz)，`time_ps` 按各自
  `frame_interval_ps` 推算 ⇒ 不同帧间隔可同池。
- **val（含 ATLAS）蛋白未见**：mdCATH train/val disjoint；ATLAS train/val 蛋白 disjoint；
  我们对 ATLAS 从未训练 Stage A/B1a/ensemble，**ATLAS 是彻底未见**。

### 1.2 关键观察（决定设计）
- **ATLAS-1ns 近刚性**：persistence 地板仅 0.24 Å（§ `STAGE_A_REPAIR...` §12.4）。
- **ATLAS-100ps 更好 posed**：1001 帧 / 0.1 ns 分辨率。同一物理时间(≈800 ps)下
  persistence 0.19–0.64 Å，但**时间维度精细 10×**，单帧→未来的可预测性显著提高
  （1 ns 分辨率时 8000 ps 已达 1.66 Å）。
- **系综级 loss 在同分布有效**（mdCATH：RMSF 显著优 persistence 49%），
  但 **mdCATH→ATLAS 不迁移**（近刚性上全面输给 persistence）。
  ⇒ 本轮用**混合训练**直接让模型见过 ATLAS 的刚性，预期修掉这个失配。

### 1.3 代码资产（全部为 NEW，不动精确坐标解码）
- `evaluation/ensemble_metrics.py`（T1，已含 NaN-safe 修复）
- `protein_jepa_frame/training/ensemble_targets.py`（T2）
- `protein_jepa_frame/models/ensemble_producer.py`（T3）
- `protein_jepa_frame/training/ensemble_losses.py`（T4）
- `protein_jepa_frame/training/train_stage_ensemble.py`（T5，已修 NaN-backward）
- `scripts/eval_ensemble_manifest.py`（T6）

---

## 2. 设计

### 2.1 数据：混合 + 时间分辨率正确性
- 主用：**新混合 manifest**（mdCATH 200dom + **ATLAS-100ps**），
  以 `horizon_time_ps`（而非 frame）对齐目标时标。
  - mdCATH horizon_frames [1,2,4,8] × 1000 ps = 1/2/4/8 ns。
  - ATLAS-100ps horizon_frames 需**放大**以匹配时标（如 [10,20,40,80]×100ps
    = 1/2/4/8 ns）**或**保持小 horizon 做"近期动力学"任务（二者作为 ablation）。
- **不平衡处理**：ATLAS 只占 ~6%（1ns 版）。提供两种：
  (a) 用 `dataset_weights`（**需给 PairedTrajectoryDataset 补上，目前未实现**）；
  (b) 简单过采样 ATLAS 行（复制 manifest 行）——先做 (b) 快、可控。
- **温度分层**：mdCATH 320–450 K、ATLAS 300 K。报告时分层，勿混池（`PLAN` §7）。

### 2.2 目标：系综级为主 + 位置重建为辅
损失：
```
L = w_ens · L_ens  +  w_pos · L_pos
L_ens = w_rmsf·L_rmsf + w_contact·L_contact + w_dist·L_dist   (T4，已有)
L_pos = CA-RMSD(Kabsch) 对 ensemble 均值 / 或对 anchor-local 位移   (NEW 辅助项)
```
- **L_pos 的角色**：**引导指标**。防止 Gaussian 头学到"只会输出大柔性"的退化解；
  要求**系综均值**同时靠近真值未来帧（Kabsch CA MSE，复用 `kabsch_ca_mse_loss`，只读）。
- 只在 `train_stage_ensemble.py` 里加一个 `--pos-weight` 分支（新文件/新参数，
  不改 `stage_b1a/objectives.py`）。
- **对齐一致性**：系综与位置都用 `align_ensemble`（首帧参考）口径一致。

### 2.3 模型：冻结栈 + 可训练 head（沿用 T3/T5）
- frozen：Stage-A relpos encoder + B1a anchored decoder。
- trainable：`GaussianHHead` + （可选）轻量时标条件（`horizon_time_ps` 注入 head）。
- **NEW 可选**：`log_sigma` 用 `horizon_time_ps` / `temperature_k` 条件化
  ⇒ 学"柔性随超前时间/温度增长"，直接针对 §1.2 的失配。

### 2.4 评估协议（对齐 `PLAN_ENSEMBLE` §4/§6）
- 三个 val：**mdCATH val**、**ATLAS-100ps val**、**ATLAS-1ns val**。
- 每个都报 model / **persistence** / oracle + bootstrap 95% CI（T6 已有）。
- 分层：按 `dataset`、`temperature_k`、`horizon_time_ps`。
- **GO 判据**：在 **ATLAS val** 上不再显著输给 persistence（本轮核心新增目标），
  且 mdCATH val 上保持 ≥ 上一轮（RMSF 显著优）。

---

## 3. 任务清单

- **D0 [x] 混合 manifest 构建**：
  - `scripts/build_mixed_manifest.py`（NEW）：拼接 mdCATH 200dom + ATLAS-100ps，
    按 `horizon_time_ps` 语义对齐 horizon；支持 `--atlas-repeat N` 过采样。
  - 产出 `artifacts/mixed_manifest_100ps/{train,val}.jsonl`（NEW 目录，不覆盖旧 mixed）。
  - 校验：train/val 蛋白 disjoint；两数据集行数、温度分布打印。

- **D1 [x] producer 增补：时标/温度条件（NEW 参数，`cond_dim` 默认 0 不变行为）**：
  - `models/ensemble_producer.py`：`GaussianHHead.forward` 增可选
    `cond`（`[B, c]`：`log(horizon_time_ps)`, `T/300`）拼到 H_anchor 末维。
  - 默认关（`--cond-dims 0` 行为不变），开时新线性层。

- **D2 [x] loss 增补：位置引导项（NEW，`_position_loss` + `pos_weight`）**：
  - `training/ensemble_losses.py`：加 `_position_loss(pred_ca_mean, true_ca, mask)`
    （Kabsch CA MSE；只读调用 `kabsch_ca_mse_loss` 或自带实现）。
  - `EnsembleLossWeights` 加 `pos_weight`；`EnsembleLossComponents` 加 `pos`。
  - 单测：`tests/test_ensemble_losses.py` 增 2 例（pos 项方向正确 / 可微）。

- **D3 [x] 训练脚本扩展（`--pos-weight` / `--cond-dims`，validate 含 persist pos）**：
  - `train_stage_ensemble.py`：加 `--pos-weight`、`--cond-dims`、`--dataset-weights`
    透传；日志加 `ens/pos`、`val/ens/pos`。
  - 保持只读 ckpt、只优化 head；不改精确坐标路径。

- **D4 [x] 训练运行（A/B/C 完成，4000 steps；best val 见 `DUAL_ENSEMBLE_RESULTS_20261019.md` §3）**：
  - Run A（基线对齐）：混合 manifest（无 ATLAS 过采样），`w_pos=0` → 复现并超越上轮。
  - Run B（+位置引导）：`w_pos>0`，看 CA RMSD 是否下降且系综指标不退化。
  - Run C（+时标条件 + ATLAS 过采样）：针对 §1.2 失配。
  - 每 run 4000 steps，GPU 5（或空卡）。

- **D5 [x] 评估**：`scripts/eval_ensemble_manifest.py` 跑 3 个 val，分层 + CI。
  产出对照表（model/persist/oracle × mdCATH/ATLAS-100ps/ATLAS-1ns）。

- **D6 [x] 结果文档**：`DUAL_ENSEMBLE_RESULTS_20261019.md`（NEW），
  表格化 vs baselines（TICA+MSM/VAMPNet/SPIB；注意协议维度差异，勿混比）。

- **D7 [x] 收尾**：更新 `REPAIR_PROGRESS.md`、`PLAN_ENSEMBLE_20261018.md`（关联）。

---

## 4. 验收（go / no-go）

1. **主 GO**：ATLAS val 上 model 的系综 metrics **不再显著差于 persistence**
   （95% CI 不跨 0 或落在"更好"侧），**且** mdCATH val 保持 RMSF 显著优。
2. **辅 GO**：加入 `L_pos` 后，**ensemble 均值 CA RMSD 下降**（位置引导有效）
   而系综指标不退化。
3. **负结果也记录**：若 ATLAS 仍输 → 说明刚性数据需 per-dataset 校准 / 门控，
   写进 §5 风险与"负结果"段。

---

## 5. 风险与依赖

- **不平衡**：ATLAS 行数少（1ns 版 267）。→ D0 的 `--atlas-repeat`。
- **时标错配**：两数据集帧间隔不同，horizon 必须以 **ps** 对齐，否则目标时标混乱。
- **温度混淆**：ATLAS 全 300 K。报告必须分层，勿把 300 K 的刚性当成通用规律。
- **过拟合 head**：head 小（~800 KB），但仍可能对 mdCATH 过拟合 → 用 val + CI 监控。
- **计算**：8×A800，GPU 5 常用；每 run ~25–30 min。

---

## 6. 复用

- T1–T7 全部 infra（metrics/targets/producer/losses/train/eval）。
- `load_frozen_stack`（只读 Stage-A + B1a）。
- `_truncate_history`、`build_paired_batch`、`collate_paired_windows`、`apply_protocol`。
- 本文件与 `PLAN_ENSEMBLE_20261018.md` 互补：后者是"单数据集 + 系综"，
  本文件是"双数据集 + 系综 + 位置引导"。
