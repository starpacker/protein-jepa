# 投稿前实验清单（重写版）— 2026-11-06

> 本文档重写自 `new_target_1107.md`，把「要做什么」落到「每一步的命令、checkpoint、产物文件与判定标准」。
> 原始目标文档（`new_target_1107.md`）仍保留作语义来源；本文档是**执行版**。
>
> **执行约定：**
> - 每个实验一个 checkbox；完成后打勾，并把结果数字写入 `RESULTS_TODO_1107.md`（结果汇总文档）。
> - 未打勾的实验 = 未完成，不可声称完成。
> - 涉及评测，一律**不打开 `test.jsonl`**（保持密封）；开发评测用 `val` manifest。
> - 权威评测方法见 `EVALUATION_METHODS_AND_COMMANDS_20261021.md`；系综七指标入口 `scripts/eval_lasp_ca_dynamics.py`，序感知轨迹入口 `scripts/eval_ca_ordered_paths.py`。

---

## 前置任务 P0（最高优先级，先于一切实验）

> 用户原文：「重新确认 current model 是否合理实现 —— 我们的 current model 是在 V3 基础上，将 S 利用历史信息正确初始化，但是现在效果居然不如 V3 了，需要检查代码确认 current model 被正确实现。」

**关键背景（代码调研已确认，供审查）：** current model 与 v3 **不是**同一条训练路径，二者差异远超「S 初始化方式」：

| 维度 | v3 | correct-framework（"current model"） |
|---|---|---|
| 训练脚本 | `protein_jepa_frame/training/train_stage_b_hs.py` | `protein_jepa_frame/training/train_lasp_ca_dynamics.py` |
| encoder / decoder | **train-encoder + train-decoder** + roundtrip 损失 | **冻结**（`CAComponentBundle` 全部 `requires_grad_(False)`） |
| bath 耦合 | `velocity`（FDT-correct，单步 `Cᵀ·(hDp)` 驱动） | `velocity` + `collective_mobility` |
| S₀ 来源 | GRU（历史默认，但历史 recipe 里 `history_conditioning=False` → 从未训练） | `exact`（闭式 Kalman 后验）/ `prior` / `gru` 三臂 |
| 上游表征 | `stage_a_disp_relpos` + `stage_b1a_relpos_anchored_1000dom` | `lasp_full_v1/representation/checkpoint_best.pt`（经 `lasp_v10_geometry` 初始化的**校准后** encoder） |
| 数据 | full manifest（5398 域） | `lasp_full_v1` 的 stateless 训练窗口流 |
| checkpoint | `artifacts/lasp_v3_velocity/` | `artifacts/lasp_s0_arms_20261031/{exact,prior,gru}_seed{42,43,44}/` |

**因此「current model 效果不如 v3」最可能的解释不是 S 初始化的 bug，而是两者训练路径/数据/表征/目标都不同。** P0 的任务就是把这条链条逐环查清，确认 current model「确实是在 v3 基础上只改 S 初始化」还是「实际是另一条独立路径」。

### P0.1 — 确认「current model 是否在 v3 基础上」这个前提是否成立
- [x] 比对 v3 与 exact arm 的 checkpoint 内嵌组件：encoder/decoder/kernel 的权重来源是否同源（v3 → `train_stage_b_hs` 内嵌 encoder/decoder；exact → `lasp_full_v1/representation` 内嵌 encoder/decoder）。
- [x] 用 `scripts/diagnose_lasp_checkpoint_state.py`（或等价脚本）打印两个 checkpoint 的 `args`、`kernel_config`、`s0_mode`、`history_conditioning`、`collective_mobility`、`noise_mix`、`bath_coupling`，确认差异清单。
- [x] 产物：一份「v3 vs current model 差异清单」表格，写进 `RESULTS_TODO_1107.md` 的 P0 节。

### P0.2 — 验证 exact S₀ 后验在 current model 里被正确接通
- [x] 跑 `pytest tests/test_exact_bath_posterior.py tests/test_ca_dynamics.py tests/test_lasp_prediction_semantics.py -q`，确认 S₀ 后验实现回归通过。
- [x] 用 `scripts/diagnose_s0_mechanism.py` 复核 exact arm 的：有效耦合 `|C|`、`|C·m₀|/|f|`、S₀ 杠杆（h=1/4/16/64），确认「S 真的进入动力学」而非退化回惰性浴场（对照 `RESULTS_LASP_CORRECT_FRAMEWORK_20261031.md` §6：exact 有效耦合应 ≈ 1.003e-02，杠杆 0.07–0.13 Å）。
- [x] 产物：exact arm 的机制核验数字，写进 `RESULTS_TODO_1107.md` P0 节。

### P0.3 — 确认评测入口对两条路径口径一致（这是 E0 的前提）
- [x] 确认 v3 只能用 `scripts/eval_lasp_final.py`（它读 `train_stage_b_hs` 产出的 `HSEnsembleProducer`），而 exact arm 用 `scripts/eval_lasp_ca_dynamics.py`（它读 `CADynamics`）。二者 `extended_ensemble_summary` 同源，但模型加载路径不同。
- [x] 记录：E0 若要同口径比较 v3 与 correct，必须固定同一 `EvalProtocol`（同 manifest、同 `round_robin_proteins=True`、同 `round_robin_conditions=True`、同 `max_residues=96`、同 `n_samples=12`、同 `seed=0`）。
- [x] 产物：E0 统一协议的一段命令模板，写进 `RESULTS_TODO_1107.md` P0 节。

**P0 判定：** 只有当上述三项全部完成，才能回答用户的核心疑问「current model 是否正确实现」。若 P0.1 发现 current model 并非「v3 + 正确 S 初始化」，则 P0 的输出就是一份明确的差异报告，供用户决定是否要「在 v3 的真实基础上重做 S 初始化」——这是后续实验的前提。

---

## 实验 E0 — 统一评测 v3 与 correct-framework

> 目标：把 v3、Correct H+S、Correct H-only 放进同一评测协议，产出一张可横向比较的表。

**前置说明：** 原文档表格列了 5 个指标（RMSF-r、Contact J、ACF Error、Path Error、Latent Stable）。其中前两个是系综指标（`eval_lasp_ca_dynamics.py`），后三个是序感知轨迹指标（`eval_ca_ordered_paths.py`）。二者是**分开的评测入口**，需分两步跑，再合并到一张表。

### E0.1 — 系综七指标（RMSF-r / Contact J 等）统一评测
- [x] 对以下 checkpoint 各跑 `scripts/eval_lasp_ca_dynamics.py`（同协议：`--manifest artifacts/mdcath_manifest_full/val.jsonl --max-windows 200 --n-samples 12 --seed 0 --max-residues 96`）：
  - v3：`artifacts/lasp_v3_velocity/checkpoint_best.pt`（**注意**：v3 是 `train_stage_b_hs` 产物，需走 `eval_lasp_final.py` 的加载，或先确认 `eval_lasp_ca_dynamics.py` 能否加载；若不能，用 `eval_lasp_final.py --kernel-ckpt artifacts/lasp_v3_velocity/checkpoint_best.pt`）。
  - Correct H+S：`artifacts/lasp_s0_arms_20261031/exact_seed42/checkpoint_best.pt`（+ seed43/44 做 3-seed 均值）。
  - Correct H-only：`bath_dim=0` 的 H-only 对照（若 `lasp_s0_arms` 无 H-only arm，需新训一个 `--bath-dim 0` 的 arm，或确认 `lasp_v13_no_bath` 是否可作为 H-only 对照——需先确认其 recipe 是否与 exact arm 对齐）。
  - Persistence / MD Oracle：`eval_lasp_ca_dynamics.py` 已内置 anchor-target（persistence）与 future-path-mixture 参考，直接取输出。
- [x] 指标列：`rmsf_corr`（RMSF-r↑）、`weak_contact_j`/`transient_contacts_j`（Contact J↑）。
- [x] 产物：E0 系综表，写进 `RESULTS_TODO_1107.md`。

### E0.2 — 序感知轨迹指标（ACF Error / Path Error / Latent Stable）统一评测
- [x] 对同一组 checkpoint 跑 `scripts/eval_ca_ordered_paths.py`（`--horizons 1 4 16 64`，12 paths，seed 0），取：
  - ACF Error（lag-1 自相关误差，对照真实 MD）
  - Path Error（路径长度 vs MD，或逐步 RMSD）
  - Latent Stable（H/S rollout 的稳定性：`max_norm`、有限性比例）
- [x] 注意：`eval_ca_ordered_paths.py` 是「连续轨迹」口径（一次性连续 rollout，不逐帧重编码），与系综口径不同，需明确标注不可混排。
- [x] 产物：E0 轨迹表，写进 `RESULTS_TODO_1107.md`。

### E0.3 — 合并成原文档要求的五列总表
- [x] 把 E0.1（RMSF-r、Contact J）+ E0.2（ACF Error、Path Error、Latent Stable）合并成一张表，行 = {v3, Correct H+S, Correct H-only, Persistence, MD Oracle}，列 = 五指标。
- [x] 明确标注：系综指标与轨迹指标来自不同入口，Latent Stable 对 Persistence/MD Oracle 标「不适用」。

---

## 实验 E1 — Representation Sufficiency Probe（表征充分性探针）

> 目的：回答「当前 JEPA 表示 H 是否丢失了对未来动力学有用的信息」。**不训练新 propagator**，纯探针。

### E1.1 — 构建三种输入
- [x] P1（当前单帧 `H_t`）：Stage A encoder 编码 anchor 帧。
- [x] P2（历史 latent `H_{t-K:t}`）：编码最后 K 帧历史。
- [x] P3（联合 `(H_t, S_t)`）：H + 浴场状态 S（用 exact 后验得到 S₀，或 rollout 得到的 S_t）。

### E1.2 — 三类目标 × 三个时间跨度（1 / 4 / 16 ns）
- [x] 目标 A：未来 CA 内部距离变化。
- [x] 目标 B：未来接触状态变化。
- [x] 目标 C：未来低维集体运动特征（MD-PCA 慢模投影）。

### E1.3 — 用同一 proper scoring rule 评价
- [x] 对每个 (输入 × 目标 × 时标) 组合，训练一个线性/岭探针（**独立选 λ**，遵守 2026-11-01 审计修正：对照必须条件化 H₀、λ 与评分蛋白不重叠、双向 fold）。
- [x] 计算 `G_τ^history = R(P1) − R(P2)`；若 `G > 0`，说明历史含 H 未保留的额外预测信息。
- [x] 参照 `scripts/test_history_memory.py`（注意其已知缺陷：anchor-only 对照只用截距、λ 同蛋白选取），重写为**正确的条件化对照**。
- [x] 产物：每 (目标 × 时标) 的 `G_τ^history` 表 + 判读，写进 `RESULTS_TODO_1107.md`。

**E1 判定：** 若所有 `G_τ^history ≤ 0`（在正确对照下），则历史无额外信息 → 支持「H 已充分」；若某目标/时标 `G > 0` 显著，则 H 有信息缺口 → 支持继续 Kinetic-JEPA。

---

## 实验 E2 — Frozen JEPA + Kinetic Supervision

> 目的：固定 JEPA（encoder 冻结），只训练 corrected H/S propagator，换一个**明确时间顺序**的训练目标，看是否足以纠正时间尺度失真。

### E2.1 — 实现 multi-lag joint observable score
- [x] 在 `train_lasp_ca_dynamics.py` 实现 `target=temporal` 目标：`temporal_observable`（内部距离 + 软接触 + 回转半径）、`joint_temporal_score`（multi-lag joint Energy Score）、`CADynamics.sample_temporal`、`--target temporal` + `--temporal-lags`。
- [x] 关键：这是**联合评价同一轨迹多时间点**，而非逐时间点的无序系综误差。

### E2.2 — 训练对照（保持其余 recipe 不变）
- [x] arm A：现有 endpoint/path-mixture 目标（baseline，用 exact arm 的 recipe）= `lasp_s0_arms_20261031/exact_seed42`。
- [x] arm B：新 `L_temporal`（multi-lag joint energy score）= `e2_temporal_20261106/temporal_seed42`。
- [x] 两 arm 同 seed(42)、同预算(7000 步)、同 encoder（冻结）、同 bath 尺度、同 `s0_mode=exact`（已核验 run.json：仅 target 不同）。

### E2.3 — 评测与判定
- [x] 用 standard-200 协议评测两 arm（`eval_ca_ordered_paths_noisemix0.py` + `eval_lasp_ca_dynamics_noisemix0.py`，200 窗口/50 蛋白/12 samples/seed 0/horizons 1/4/16/64）。
- [x] 判定：`L_temporal` **未**显著改善时间尺度失真（ACF 0.27/0.37/0.36 vs 0.29/0.40/0.38 边际；Path 反而 3.42/16.29/79.11 vs 2.74/13.53/67.10 更差；md_pca_w2 略差；RMSF-r 明显下降）。
- [x] 产物：两 arm 对比表 + 判定已写入 `RESULTS_TODO_1107.md` E2 节。

---

## 实验 E3 — 几何合理性 / 温度泛化 / bioemu 基线

### E3.1 — 键长键角合理性（人工检查 PDB + 百分比统计）
- [x] 用 `scripts/export_eval_trajectories_pdb.py` 导出 v3（`ours_hs`，3 窗口 CA PDB）与 current model（`scripts/export_ca_pdb_current_model.py`，3 窗口 × 4 sample CA PDB）的预测构象 PDB。
- [x] 用 `scripts/eval_backbone_suite.py`（v3）+ `scripts/eval_ca_backbone_diagnostic_noisemix0.py`（current model）统计键长/键角合法百分比（N/CA/C/O 键长、肽键 C–N、Rama、ω trans）。
- [x] 产物：键长/键角合法百分比表（v3 vs current model）已写入 `RESULTS_TODO_1107.md` E3.1 节。**PDB 人工检查由用户本人完成**（原文要求）。

### E3.2 — 温度泛化测评（V3 与 current model）
- [x] 整理一份**测评脚本 + 蛋白清单 + 温度清单**文档（`RESULTS_E32_TEMPERATURE_PLAN_20261106.md`），交用户跑 MD 生成新温度下的演化路径：20 蛋白清单、温度口径 {300/500 K 外推 或 450 K 留一}。
- [x] 参照 `scripts/eval_temperature_generalization.py`（cond head）结构，在方案文档中给出 v3 与 current model 的温度矩阵脚本改动点（`--temperatures` 覆盖 + 外推温度 manifest 注入）。
- [x] 产物：温度泛化测评方案文档 + 蛋白/温度清单已写入 `RESULTS_TODO_1107.md`，并明确「MD 由用户跑」。

### E3.3 — bioemu 作为新 baseline 部署 + zero-shot 测评
- [x] 解压 `bioemu-main.zip` 到独立目录 `/liuyutian/protein_jepa_frame/bioemu_extract/`（不混入项目源码）。
- [x] 读 `bioemu-main/README.md` + `MODEL_CARD.md`，确认其为 **BioEmu v1 平衡态系综采样器**（非条件动力学），zero-shot 推理入口 `python -m bioemu.sample --sequence ...`（需 MSA 检索或 A3M）。
- [x] 部署：装依赖（隔离 venv `pip install bioemu` 成功，`bioemu 1.4.1` 可 import）。下载权重 + smoke case：❌ 阻塞（CUDA 13 wheel vs 驱动 12.9 → CPU-only；MSA 检索 `a3m.mmseqs.com` 不可达；AF2/bioemu 权重未缓存）。
- [x] 参照 `run_scripts/run_bioemu2_zeroshot.sh` 的评测口径，确认 bioemu v1 仅系综三指标可比（RMSF-r/Contact J/md_pca_w2），轨迹指标「不适用」（平衡态系综、无有序轨迹）。
- [ ] 产物：bioemu zero-shot 结果表 —— ❌ 阻塞（见上，`RESULTS_TODO_1107.md` E3.3 节已如实记录部署成功 + zero-shot 阻塞原因）。

---

## 汇总与提交

- [x] 全部实验完成后，在 `RESULTS_TODO_1107.md` 顶部写一段「一句话结论 + 各实验判读摘要」，并汇报用户。
- [x] 确认 `test.jsonl` 全程未打开；所有评测用 val manifest。
