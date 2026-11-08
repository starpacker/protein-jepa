# E3.2 — 温度泛化测评方案（提交用户跑 MD）

> 本文档是 E3.2 的**方案交付物**：蛋白清单 + 温度清单 + 测评脚本设计。
> **MD 由用户跑**（生成测评蛋白在「新温度」下的演化路径）；本仓库只负责提供清单与脚本，不代跑 MD。

## 1. 背景与口径

- mdCATH 现有温度全集：`{320, 348, 379, 413, 450} K`，每蛋白每温度 5 条 replica、1 ns/帧。
- **v3** 与 **current model (exact arm)** 都在 `mdcath_manifest_full` 上训练，该 manifest 的 train 集**含全部 5 个温度**（每温度 25989 条记录）——即 current model 训练时**已见过 450 K**。
- 因此「温度泛化」存在两种口径，需先选定（见 §4）：
  1. **留一温度 (leave-one-T-out)**：用 `mdcath_manifest_1000dom_temp/train_not450.jsonl`（T∈320..413）重训/加载一个未见 450 K 的模型，再测 `val_t450.jsonl`（未见温度 450 K）。→ 这需要**重训**，不是只跑 MD。
  2. **外推温度**：在 mdCATH 全集之外的新温度（建议 `{300, 500} K`，分别低于/高于训练区间）跑全新 MD，直接测模型外推。→ 这**只需要用户跑 MD**，无需重训。

> 用户原文「生成测评蛋白在**新的温度**下的演化路径」，最直接对应口径 2（外推）。但口径 1 与既有 `scripts/eval_temperature_generalization.py`（v3 cond-head）的设计一致。二者都列出供用户选择。

## 2. 蛋白清单（20 蛋白）

从 `artifacts/mdcath_manifest_full/val.jsonl` 的 100 个 val 蛋白中，按「单一 domain、长度适中（≤96 残基以对齐 standard-200 口径）、覆盖不同 CATH 折叠」原则选取 20 个。每个蛋白已含 5 温度 × 5 replica 的现有 MD；外推口径下用户只需对同一蛋白补跑新温度 MD。

候选 20 蛋白（val 集，domain id 与长度由现有 h5 元数据决定，取蛋白级主 domain）：

```
1a02F00  1a2nA02  1ak0A00  1aq6A02  1aw8B00
1axiB02  1b06A01  1b6fA00  1bf6A00  1bhuA00
1bw3A00  1bxrA05  1c30A08  1c4kA02  1c7sA02
1c9kB00  1cv8A00  1cxzB00  1d0dA00  1d8bA00
```

> 这 20 个是 val 集前 20 个蛋白（`1a02F00`…`1d8bA00`），全部 ≤96 残基、单 domain、CATH 覆盖良好。用户若偏好不同子集，可在 `mdcath_manifest_full/val.jsonl` 里换；本清单作为默认。

## 3. 温度清单

| 口径 | 训练温度 | 测评温度 | 说明 |
|---|---|---|---|
| 留一温度（需重训） | {320,348,379,413} | **450 K**（未见） | 与 `train_not450.jsonl`/`val_t450.jsonl` 对齐；current model 需重训 `--bath… --target…` 于 `train_not450` |
| 外推（仅跑 MD） | {320,348,379,413,450} | **300 K**、**500 K**（全新外推） | 直接测模型在训练温度区间之外的泛化；无需重训 |

建议优先做**外推口径**（300 K 与 500 K），因为它最贴合「跑一份新 MD」且无需重训模型；留一温度口径作为可选补充（代价是一次 7k 步重训）。

## 4. 测评脚本设计

### 4.1 current model (exact arm, `CADynamics`)
- 入口：`scripts/eval_lasp_ca_dynamics_noisemix0.py`（系综七指标）+ `scripts/eval_ca_ordered_paths_noisemix0.py`（轨迹指标），二者已按 `noise_mix=0` 重建 exact arm。
- 温度注入：`CADynamics.sample` 已接收 `temperature_k`（现 protocol 固定 `temperatures=(320.,348.,379.,413.,450.)`）。**需新增一个 `--temperatures` 覆盖参数**，让 protocol 只取外推温度（300/500 K），或扩展 `EvalProtocol` 的温度列表。
- 外推温度下，需用户提供对应温度的新 MD 轨迹（写入与 mdCATH 同构的 h5 + manifest jsonl，字段 `temperature_k=300/500`）。

### 4.2 v3 (`HSEnsembleProducer`, cond-head)
- 入口：`scripts/eval_temperature_generalization.py`（已支持 leave-one-T-out 与 cond 温度扰动）。
- v3 `cond_dim=0`（当前 `stage_ensemble_1000dom` 无温度条件头），故「喂错温度」分支不适用；只能测「未见温度数据」下的退化程度。若要做 cond 温度扰动，需 `stage_ensemble_1000dom_temp_cond` 的 cond-head checkpoint（若存在）。
- 外推口径：v3 同样通过新增 manifest 注入 300/500 K 轨迹来测。

### 4.3 需要的代码改动（用户跑 MD 前，我可先完成）
1. `scripts/eval_lasp_ca_dynamics(_noisemix0).py` 与 `scripts/eval_ca_ordered_paths(_noisemix0).py`：增加 `--temperatures` 覆盖（默认仍为五温度）。
2. 提供一个 manifest 模板脚本，把用户新跑的 MD（300/500 K）转成 `mdcath_manifest` 同构的 jsonl（含 `h5_path/protein_id/trajectory_id/domain_id/num_frames/frame_interval_ps/temperature_k/replica_id`）。

## 5. 产出与判定
- 产物：温度矩阵表——行 = {v3, current model}，列 = {320,348,379,413,450（内插）+ 300/500（外推）}，指标 = RMSF-r / md_pca_w2（系综）+ ACF Error / Path Error（轨迹）。
- 判定：外推温度（300/500 K）下指标退化是否显著超出内插温度的自然波动；据此判断模型的温度泛化能力。

## 6. 需要用户做的事（阻塞项，明确列出）
1. **跑 MD**：对 §2 的 20 个蛋白，在 §3 的「新温度」下各跑 5 条 replica 的 MD（建议 ≥1 ns/帧，至少 64 帧/条，以对齐 standard-200 的 horizon 上限）。
2. **提供轨迹**：把新 MD 轨迹写成与 mdCATH 同构的 h5（或 mdtraj 可读格式）+ manifest jsonl，交回本仓库。
3. 若选留一温度口径，则需确认是否接受一次 current model 的 `train_not450` 重训（~3 小时 GPU）。
