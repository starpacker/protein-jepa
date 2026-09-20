# 三大主实验逻辑重梳 + 实现/基线审计 + SOTA 判定报告

日期：2026-09-16
仓库：`/liuyutian/protein_jepa_frame`
审计范围：`scripts/evaluate_history_gap.py`（主实验 1）、`scripts/evaluate_m1_strict.py`（主实验 2）、
`scripts/evaluate_k1_fastfolding.py`（主实验 3）、`scripts/evaluate_r1_coordinate_baselines.py`、
`artifacts/reviewer_experiments/_shared/*`、`protein_jepa_frame/fastfolding/benchmark.py`、
`protein_jepa_frame/models/lspwm.py`、`protein_jepa_frame/training/train.py`、
以及 `artifacts/` 下全部 Stage-B 训练与评测产物。

关联文档：
`ICLR_PAPER_PLAN.md`（论文选题计划）、`V5_V9_REEVALUATION_REPORT.md`（上一轮 v5/v9 重评）、
`EVALUATION_COMPARISON_REPORT.md`、`docs/reviewer_experiment_todo.md`。

> **本报告与之前所有报告的关系：** 之前的报告是在**当前实现的前提下**解读数字；
> 本报告是**先验证实现本身**，再判断哪些数字可以被引用。
> 所有结论都附 `file:line` 与可复现命令，见附录 A。

---

## 0. 结论速览

1. **主实验 1（History Gap）当前没有一条可引用的 JEPA 数字。**
   `evaluate_history_gap.py` 的 horizon 循环**不改变数据**（`hf` 只用作 dict key），
   current 用线性探针、history 用 MLP 探针（容量不对等），train/val 是**帧级随机划分**（存在时间泄漏），
   且 `pca/tica/vamp` 三条"基线"是在 **JEPA 自己的 latent** 上做的，不是坐标。
   ⇒ `representation_closure*_v5/v9/results.json` 里的 `Δ_hist ≈ -1.20`（5 个 horizon 几乎同一个数）不能用于任何 claim。
2. **主实验 1 有一条已经写好、但被弃用的正确实现**：
   `scripts/evaluate_r1_coordinate_baselines.py::_fit_predict_gap`（确定性 ridge、容量对等、
   train-prefix/test-suffix、horizon 真正生效、protein 级 bootstrap）。
   它已经跑出可用基线：**PCA(16D, CA) 在 h=1/H=2 的 `Δ_hist = +0.707`（95% CI `[+0.082,+1.320]`），
   tICA = -0.043，VAMP = -0.053。** ⇒ 修复方式是把 LSP-WM latent 接进这套探针，而不是新写一套。
3. **主实验 2 的 2×2 消融有 3 个结构性缺陷，其中两个是"格子根本不存在"**：
   `Markov` 与 `FreeEnergy` **共用同一个力场**（`native_lspwm_potential`），只差常数/局部扩散；
   **"有 memory、无自由能"这一格从未实现**；
   `Full` 与 `Mori` 用了**两套不同的 memory 约定**（一个是 `kernel[lag]` 不中心化，一个是 `kernel[lag-1]@(h-center)`）。
   ⇒ 现在这张 8 方法表不是 2×2，而是一组不可比的模型。
4. **`Full` 的 memory ≈ 0 不是"训练不够"，而是架构数量级封顶。**
   `GaugeFixedBath`（`lspwm.py:302-380`）的 `amplitude = coupling_scale * softplus(raw_amplitude)`，
   在默认 `coupling_scale=1e-3`、`raw_amplitude=0` 下我复现出 **`‖K(0)‖_F = 9.6e-7`、`‖K[1:]‖_F = 2.0e-7`（dt=8000 ps）**——
   比拟合出来的 `Mori` kernel（`4.37e-4`）低 **3 个数量级**；而实测 `Full.kernel_norm = 3.87e-10`，
   又比这个上界低 **500 倍**，说明 Stage-B 训练**把 bath 主动压向 0**。
   另外 `dt=8000 ps` 配 `τ∈{1000,2000,4000,8000} ps` 使 lag≥2 的权重 ≤ 0.135 ⇒ **名义 4-lag memory 实际只有 1 lag。**
5. **主实验 2 的三个"主指标"可以被退化模型通过。**
   `FullCalibrated.mobility_scale = 0.0`（v5、v9 都是）⇒ `mean = z`，**即纯随机游走**，
   却拿到全场最好的 `semigroup_mean_defect = 5.10e-6` 和不错的 `fdt_residual = 1.298`。
   ⇒ `semigroup_mean_defect` 实际度量的是"一步映射离恒等有多近"；`fdt_residual` 度量的是"噪声协方差能否从 train 迁移到 test"。
6. **主实验 2 的样本量被一行 CLI 参数砍掉了 8 倍。**
   归档 run 用 `--max-batches 50`；`50×32/53 = 30.19` 与日志 `Extracted 30 trajectories` 完全吻合。
   于是评测集只剩 **8 个 protein / 30 条轨迹 / 299 个 test transition**，
   每条轨迹的 reference 只有 **10–11 帧**，而 `stationary_jsd` 要用 21 个 bin 去比较这 11 帧和 256 步 rollout，
   并且**只用了 256 维 latent 的前 16 维**。
   manifest 里其实有 `20 proteins × 4 temperatures × 53 windows = 4240 clips`（132 个 batch）可用。
7. **主实验 3 已经可以定案：所谓 baseline 的 0.98–0.99 accuracy 100% 是类别不平衡。**
   我直接复现了 K1 的划分（seed 6303，scaffold-disjoint）：h=1 的 368 个测试样本里 **364 个是 `transition`（98.91%）**。
   常数预测器（永远输出 `transition`）得到 **acc 0.9891 / bal 0.3333**；
   而 `tICA+MSM` 得到 **0.9891 / 0.3333（完全相同）**，`MSM` 得到 0.9810/0.3306，`Persistence` 0.9783/0.3297
   —— **三个基线全部 ≤ 常数预测器**。
   同时 `frame_labels.full.jsonl` 全部 2600 帧里 `unfolded` 只有 **8 帧** ⇒ "unfolded→folded 动力学"在当前标签下**根本不可估计**。
8. **SOTA 判定：v5 不是最好的模型，只是"唯一被完整评测过"的模型。**
   v9 训练发散（final loss `1.37e15`，`Full` M1 30/30 non-finite）⇒ 不可用。
   但 **v6（`lspwm_v44_stage_b_v6_kernel_fix`）的训练健康度明显好于 v5**：
   `trajectory_nll` 397.7 → 157.6（有界、单调下降），`dynamic_nll` 399.9 → **-31.2**（v5 从未低于 170.9），
   `force_loss` 0.94（v5 为 2.17），且 v5 的 `trajectory_nll` 在全部 467 条日志里**恒等于 0**（多步 path loss 从未真正生效）。
   ⇒ 当前"冠军"应标为 **v5（已被评测中最稳）**，并**立刻补评 v6**；在 v6 评测完成前，任何 SOTA 声明都是不完整的。


---


## 1. 实验 1：R1 History Gap —— 审计结论 + 修复后重跑

### 1.1 原实现的逻辑错误（逐条定位）

| # | 问题 | 位置 | 为什么是逻辑错误 |
|---|---|---|---|
| R1-a | **horizon 循环完全不改变数据** | 旧 `scripts/evaluate_history_gap.py:151` `for hf in horizons:`，循环体内 `hf` 只被用作 dict key；真正的样本对由 `paired_horizons_frames=[24,48,96]` 采样决定 | 7 个"horizon"是**同一个混合 lag 实验**；"Δ_hist 随 horizon 单调衰减"这一 claim 没有任何数据支撑 |
| R1-b | **train/val 是帧级随机划分** | 旧 `split_frames_random`（对所有帧做 permutation 再切 20%） | 同一轨迹的相邻帧同时进入 train 与 val ⇒ **时间泄漏**；"val loss" 实质是插值误差，不是外推误差 |
| R1-c | **probe 容量不对等** | 旧 `LinearProbe`（单层 `nn.Linear`）对比旧 `HistoryMLP`（`evaluate_history_gap.py:52` `flatten -> 256 -> ReLU -> 256 -> ReLU -> D`） | 测出来的 Δ_hist 里混入了**参数量差异**，无法归因到"多出来的历史信息" |
| R1-d | **所谓 `pca/tica/vamp` 基线是在 JEPA latent 上做的** | 旧 `_fit_representation` 的输入是 latent 张量，不是 CA 坐标 | 三行"坐标基线"回答不了"JEPA 表示是否优于坐标表示"这个唯一有意义的问题；K1 的 `tICA+MSM` 也因此在两个实验间不可比 |
| R1-e | **表示拟合使用了全部帧（含评测帧）** | 旧 `fit_representation(all_frames)` 在划分之前执行 | 基线表示层"看过"评测数据 ⇒ 不是 held-out 基线 |
| R1-f | **聚合单位是帧，不是 protein** | 旧 summary 对所有帧求均值与 std | 同一 protein 的 4 条温度轨迹被当作独立样本，置信区间被人为压窄 |

### 1.2 修复内容（已落地）

新脚本 `scripts/evaluate_history_gap_v2.py`（协议名 **`history_gap_ordered_v2`**）：

* **状态 = ordered native `q` 轨迹**，由共享模块
  `artifacts/reviewer_experiments/_shared/latent_trajectories.py::extract_ordered_latent_trajectories`
  提取（连续因果 clip 的端点、按 trajectory 分组、与 M1 共用同一实现）。
  物理步长 `dt_ps = frame_interval_ps × window_stride`（默认 **8000 ps**），
  horizon `h` 因此是**真实的单调时间 lag**（`physical_horizon_ps` 写进 JSON）。
* **单一探针**：`current` 与 `history` 使用**同一个**确定性 ridge 学习器（`_ridge`，
  自适应 L2：`reg = 1e-3 × trace(GᵀG)/D`，截距项不正则化），只有特征集合不同 ⇒ 容量对等。
* **两个容量对照**（把"多参数"从"多信息"里分离）：
  * `current_tiled_H` —— 特征数与 history 相同，但把当前态复制 H 次（**无新信息**）；
  * `history_shuffled` —— 同样的历史实现，但**打乱时间对齐**。
* **零信息对照** `mse_train_mean`：用训练集目标均值预测。
* **划分**：每条轨迹**前 80% 帧（train prefix）/ 后 20% 帧（test suffix）**，
  表示与 probe 都只用 prefix 拟合。
* **聚合**：先按 protein 平均，再对 protein 做 **bootstrap（4000 次）**，
  输出 `delta_*` 的均值与 95% CI 以及 `..._ci_excludes_zero`。
* **坐标基线真正落在 CA 坐标上**：`_coordinate_sequences` 用
  `baselines.data.extract_features_for_protein(frame_stride=1)`（Kabsch 对齐后的
  `(F, R*3)` CA 特征），再按 clip 端点抽样，使坐标与 latent **共享同一时间原点与物理步长**。
* **口径 gate**：`{name}_history_beats_current_and_tiled` 要求
  `Δ_hist_vs_current` 与 `Δ_hist_vs_tiled` 的 CI 都排斥 0，且 `Δ_hist_vs_tiled > 0`。

### 1.3 修复过程中被本轮重跑捕获的**第二个**实现错误

第一次 v2 重跑（`/tmp/r1v2_v5.log`）在坐标阶段崩溃，暴露出修复本身的两个缺陷 —— 都已修好：

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| R1-g | `ValueError: ... array at index 0 has size 636 and the array at index 4 has size 738` | `_fit_transform_representation` 把 **20 个不同长度 protein** 的 `3*n_residues` 维 CA 特征拼成一个矩阵，做**一次**全局 PCA/tICA/VAMP | 拆成 `_fit_transform_one_protein`（单 protein 拟合）+ `_fit_transform_representation`（逐 protein 循环），只在该 protein 的 train prefix 上拟合；返回 `dim_per_protein` / `skipped_proteins` |
| R1-h | 同一环境下 `tica`/`vamp` 把 **20/20 protein** 全部跳过（`ModuleNotFoundError: No module named 'deeptime'`），却仍写出 JSON，`n_traj=0` 且**比较集被悄悄变成空集** | `deeptime` 只在 `/liuyutian/protein_jepa/.venv` 里存在；而 `_fit_transform_representation` 用 `except Exception` 逐 protein 吞掉错误 | ① 启动时若请求了 `tica/vamp` 而 `deeptime` 不可用则**直接 `SystemExit`** 并给出正确解释器路径；② 某表示 0 个可用 protein 时标记 `status: "failed"`、写 `errors`、**从交叉表示比较集中排除**（不再一票否决整轮）；③ 若 `errors` 非空，JSON 照写但**进程返回码 2** |

> 这两条本身是审计链的证据：旧的"静默降级"模式（`except: continue` + 空集照写结果）
> 会让一个**完全失败的基线**在 JSON 里看起来像"结果不显著"。

---


## 2. 实验 2：M1 有限记忆闭合 —— 审计结论 + 修复后重跑

### 2.1 原实现的逻辑错误（逐条定位）

| # | 问题 | 位置 | 为什么是逻辑错误 |
|---|---|---|---|
| M1-a | **所谓 2×2 消融并不存在**：`Markov` 与 `FreeEnergy` 共用 `native_lspwm_potential` 力场 | 旧 `MarkovClosure.fit` 与旧 `FreeEnergyClosure.fit` 都是 `self.force_fn = adapter.get_force_fn()` | "no free energy" 这一格没有被实现。两格只差"常数扩散 vs 局部 kNN 扩散"，不是"有力场 vs 无力场" |
| M1-b | **"有 memory、无自由能"这一格完全不存在** | 旧方法表 `{Markov, FreeEnergy, Mori, Full, FullCalibrated, GenericMarkov, GRU, CausalTransformer}` | `Mori`/`Full` 都带力场；没有任何格子是"memory only"，于是"memory 的贡献"无法从"力场的贡献"里分离 |
| M1-c | **`Full` 与 `Mori` 的 memory 约定不一致** | 旧 `FullClosure.predict_step` 用 `memory += self.kernel[lag] @ history[-lag] * dt`；旧 `MoriClosure` 用 `kernel[lag-1] @ (history[-lag] - center)` | `Full` 用 `K(1·dt)…K(L·dt)`（**跳过 `K(0)`**）且**不中心化**；`Mori` 用拟合出的 lag `0..L-1`（含 `K(0)`）并中心化。两格的 memory 不是同一个数学对象，`kernel_norm` 也不可比 |
| M1-d | `closure_gate` 硬编码 `False` | 旧 `"closure_gate": False` | 有 gate 字段但没有 gate 定义；论文里"闭合通过/未通过"无从复现 |
| M1-e | rollout 发散被静默吞掉 | 旧 `except Exception as e: print(...); continue` | `n_test_trajectories` 只统计成功的，**发散数**不上报。v9 的 `Full` 30/30 non-finite 在 JSON 里完全看不出来 |
| M1-f | `semigroup_mean_defect` 拟合的是**无截距**线性算子 | 旧 `np.linalg.lstsq(states, one_step)` | 有常数偏移时，偏移被算进 defect；而且 `T1@T1` 与 `T2` 的复合在纯线性约定下定义域不一致（仿射映射不能用纯线性矩阵复合） |
| M1-g | `stationary_jsd` 只用前 16 维，且没有噪声地板 | 旧 `range(min(reference.shape[1], 16))` | 256 维 latent 只比较前 16 维；用 ~10 帧 reference 对 256 步 rollout 做 21-bin 直方图，JSD 被有限样本噪声主导。没有 `reference vs reference` 地板，任何 JSD 值都无法解释 |
| M1-h | 评测集被一行 CLI 参数砍掉 8 倍，且**不报警** | `--max-batches 50`（`scripts/run_m1_full.sh`）vs manifest 容量 140 batch | 评测集只剩 30 轨迹 / 8 protein / 299 test transition（详见 §0.6） |
| M1-i | 没有任何"退化模型对照" | 旧方法表 | `FullCalibrated.mobility_scale=0`（即 `mean=z`，纯随机游走）却是全场最好的 `semigroup_mean_defect`；表里没有 `TrainMean`/随机游走这种零信息对照，读者无法识别 |

### 2.2 修复内容（已落地）

`scripts/evaluate_m1_strict.py`（协议名 `strict_temporal_holdout_v1` → **`strict_temporal_holdout_v2`**）：

* **统一状态表示与时间栅格**：新增 `artifacts/reviewer_experiments/_shared/latent_trajectories.py`，
  R1 与 M1 共用同一个 `extract_ordered_latent_trajectories`（ordered clip endpoint、同 `dt_ps`、同 trajectory 分组）。
  旧 `extract_native_q_trajectories` 变成薄包装，`split_temporal` 也改为从共享模块导入 ⇒ 不会出现两套实现漂移。
* **补齐真正的 2×2**：新增 `TwoByTwoClosure`，四格只由 `use_force` / `use_memory` 两个布尔决定，使用**同一条**更新规则

  ```
  z_{t+1} = z_t + s_f · D F(z_t) · dt + dt · Σ_{j=0}^{L-1} K_j (z_{t-j} − c) + ε
  ```

  * `M_None`（force=0, memory=0）—— 新增真空对照
  * `M_Force`（force=1, memory=0）
  * `M_Memory`（force=0, memory=1）—— **新增的"有 memory 无自由能"格**
  * `M_Full`（force=1, memory=1）
* **统一 memory lag 约定**：`fit_memory_kernel` 重写，特征改为 `lag = 0..L-1`（含当前态）；
  `predict_step` 中 `kernel[0]` 乘当前态、`kernel[j]` 乘 `history[-j]`。`Full` / `FullCalibrated` / `M_*` 全部同一约定。
* **新增零信息对照** `TrainMeanClosure`（训练均值状态 + 训练方差）。
* **`semigroup_mean_defect` 改为仿射算子**：在齐次坐标 `[z,1]` 上拟合，比较 `T1∘T1` 与 `T2`。
* **`stationary_jsd` 全维 + 噪声地板**：默认 64 维，新增 `stationary_jsd_noise_floor`（reference 前半 vs 后半）与 `stationary_jsd_excess_over_floor`。
* **真实的 `closure_gate`**：要求
  ① `M_Full − M_Force` 在 `prediction_mse` 上 protein-bootstrap 区间排斥 0 且为负（memory 在力场之外有增量）；
  ② `M_Full − M_Memory` 同理（力场在 memory 之外有增量）；③ `M_Full − M_None` 显著。
  同时把全部中间量写入 `closure_gate_details`。
* **失败不再静默**：每个方法输出 `n_failed_trajectories`。
* **样本量告警**：`loader_stats.truncated_by_max_batches` 若为 true，则写 warning 且 gate `no_truncated_evaluation_set=false`。
  本次重跑使用 `--max-batches 0`（**不截断**，78/78 轨迹全部进入评测）。
* **结构指纹去重（第二轮加强）**：每个方法写入
  `(effective_force_source, diffusion_source, kernel_source, effective_drift_increment_norm, kernel_norm, architecture)`，
  检查两两不同；重复则 warning + gate 置 false。**这一条正是本次审计中 `Markov == FreeEnergy` 事故的自动捕获器。**
  第二轮把"名义力场"改成"**有效**漂移增量范数"后，指纹第一次抓到了真实塌缩：
  v5 的 `M_Force` 拟合出的 `force_scale = 0.0`（原始力增量范数 `574.4`，但与观测增量**反相关**被 clip 到 0），
  于是 `M_Force` 与 `M_None` **逐位相同**、`M_Full` 与 `M_Memory` **逐位相同** ⇒ 指纹给出
  `{'M_None': ['M_Force'], 'M_Memory': ['M_Full']}`。
* **行为指纹 + 近似同一对**：除结构指纹外，再比较每个方法的**评测指标向量**
  （`prediction_mse … rollout_mean_norm`）：
  * `behavioural_fingerprint_duplicates`（逐位相同 ⇒ 同一个模型）；
  * `behaviourally_near_identical_pairs`（最大相对差 < 1e-3 ⇒ 蒙特卡洛噪声级同一模型），
    并单独列出其中的**消融格对**，gate `ablation_cells_behaviourally_separated`。
* **memory 的 ridge 由内层验证集选择**：`fit_memory_kernel` 现在返回 `(kernel, diagnostics)`，
  在训练 prefix 内部再切 60%/40%（按轨迹切，绝不跨边界），在
  `λ ∈ {1e-3,1e-2,1e-1,1,10,100} × trace(XᵀX)/D` 上选验证误差最小的；
  `kernel_diagnostics` 写入 JSON。原因：`4 lag × 256 dim = 1024` 个系数在强共线的 lagged latent 上
  求解会**过拟合**（旧固定 `1e-2` 时 `M_Memory` 的 train/test 误差比为 3.23e-3 / 7.38e-3），
  不修就无法公平回答"memory 有没有用"。
* **native bath 可诊断**：输出 `native_kernel_lag_relative_norm`、`native_effective_memory_lags`
  （相对范数 > 0.05 的 lag 数）与 `native_kernel_ratio_excluding_zero_lag`，暴露"名义 4-lag、实际 1-lag"；gate `native_memory_informative`。

---

## 3. 实验 3：K1 FastFolding —— 审计结论 + 修复后重跑

### 3.1 原实现的逻辑错误（逐条定位）

| # | 问题 | 位置 | 为什么是错误 |
|---|---|---|---|
| K1-a | **committor 指标是恒等式，不是预测能力** | `scripts/evaluate_k1_fastfolding.py:114` `_committor_metrics`：`predicted = probabilities @ committor`，`target = committor[source_states]`；而 `Persistence` 的 `probabilities = I`（同文件 `_persistence_probabilities`） | `I @ q_train = q_train[source]` ⇒ `predicted ≡ target` ⇒ **brier 恒为 0、pearson 恒为 1**。所谓"Persistence 的 committor 完美"是代数恒等式 |
| K1-b | 同上，`benchmark.py` 的 committor 参考也来自同一个函数 | `protein_jepa_frame/fastfolding/benchmark.py:225` `committor = {"unfolded":0.0,"folded":1.0}.get(source, 0.5)` | 对 98.9% 的 `transition` 样本输出 0.5，与 `committor_target`（`build_event_samples` 里按"先到 folded 记 1"）不同源，但与 K1 的 train-MSM 参考同源 |
| K1-c | **没有任何"无动力学"对照** | `scripts/evaluate_k1_fastfolding.py:412`（旧）`methods = {"Persistence","MSM","tICA+MSM","JEPA"}` | 缺少常数预测器（majority class），于是 98.9% 的类别不平衡被误读为"基线很强" |
| K1-d | 只有 `accuracy` 进 summary，`balanced_accuracy`/`macro_f1` 被丢弃 | `evaluate_k1_fastfolding.py:496`（旧） | `multiclass_forecast_metrics` 明明已经返回 `balanced_accuracy`/`macro_f1`，但 summary 只取 `nll,brier,accuracy,ece` |
| K1-e | 没有输出类别分布，也没有输出标签文件的类别统计 | 整个文件 | 无法发现 `unfolded` 只有 8 帧 |

### 3.2 修复内容（已落地）

* `scripts/evaluate_k1_fastfolding.py`
  * 新增 `_fold_forecast_metrics`：**非恒等式**的 committor 代理——预测 `state_probs[:, folded]`，目标是**held-out 观测到的** `1[target_state == folded]`，附 `constant_base_rate_brier` 对照与 AUC。`Persistence` 在这套指标下只能输出 `P(source==folded)`（98.9% 为 0），无法再白拿 0 分。
  * `_committor_metrics` 保留原列但**显式输出退化证据**：`identity_operator_brier`、`constant_prior_brier`、`n_distinct_target_values`、`auc`。
  * 新增 `MajorityClass` 基线（训练集状态边缘分布，常数输出）。
  * summary 增加 `balanced_accuracy`、`macro_f1` 与全部 `fold_forecast_*` 列。
  * 每个 horizon 输出 `class_imbalance`（source/target 三类计数、majority class 及其占比）。
  * 新增 `degenerate_metric_audit`：对每个 headline 指标，给出 `Persistence/MajorityClass` 里最好的那个值，以及 JEPA 是否超过它。
  * 新增 gates：`majority_class_control_present`、`unfolded_kinetics_estimable`、`jepa_beats_degenerate_control_on_nll`、`jepa_beats_degenerate_control_on_balanced_accuracy`。

> 注意：这次**没有改标签、没有改划分**（`labels_sha256 = 15d0ce73…` 前后一致，scaffold-disjoint seed 6303），
> 所以新旧结果可以直接对照，差异全部来自评测逻辑的修正。

### 3.3 重跑结果（`k1_v2_{v3,v5,v9}.json`）

> **PROTOCOL NOTE:** The numbers below use the OLD `state` labels (unfolded/transition/folded) under 98.9% class imbalance. The accuracies of 0.98-0.99 reflect the imbalanced label distribution — a constant `transition` predictor achieves accuracy 0.989. For corrected q_native-label FastFolding results where MSM accuracy = 0.490 and tICA+MSM accuracy = 0.419, see `CORRECTED_FINDINGS_REPORT.md`.

4 个 horizon（1/4/8/16）的均值：

| method | nll ↓ | brier ↓ | accuracy | balanced_acc ↑ | macro_f1 ↑ | fold_forecast folded_brier ↓ | folded AUC ↑ |
|---|---|---|---|---|---|---|---|
| Persistence | 0.5404 | 0.0130 | 0.9804 | 0.3714 | 0.3300 | **0.0145** | 0.1430 |
| **MajorityClass（新增对照）** | 0.3038 | 0.0438 | **0.9899** | 0.3750 | 0.3316 | 0.0631 | 0.1464 |
| MSM | 0.2292 | 0.0295 | 0.9879 | 0.3743 | 0.3313 | 0.0415 | 0.1430 |
| tICA+MSM | **0.2179** | **0.0278** | **0.9899** | **0.3750** | **0.3316** | 0.0383 | **0.6960** |
| JEPA v5 | 0.9929 | 0.1866 | 0.6505 | 0.3850 | 0.2638 | 0.1819 | 0.2242 |
| JEPA v9 | 1.7119 | 0.2630 | 0.5088 | 0.2072 | 0.2100 | 0.3120 | 0.1651 |
| JEPA v3 | 1.5818 | 0.2907 | 0.4286 | 0.1693 | 0.1873 | 0.3378 | 0.1600 |

类别不平衡（测试集，各 horizon）：

| horizon | n | `unfolded` | `transition` | `folded` | majority 占比 |
|---|---|---|---|---|---|
| 1 | 368 | 1 | 364 | 3 | 0.9891 |
| 4 | 356 | 1 | 352 | 3 | 0.9888 |
| 8 | 340 | 1 | 336 | 3 | 0.9882 |
| 16 | 308 | 0 | 306 | 2 | 0.9935 |

**结论（可以写进论文的版本）：**

1. `tICA+MSM` 的 0.9899 / 0.3750 与**常数预测器完全相等**；`MSM`、`Persistence` 均 ≤ 常数预测器。
   即：**当前 K1 协议下没有任何方法展示出 folding state 预测能力**。0.98–0.99 的 accuracy 是 98.9% 类别占比的直接后果。
2. gate `jepa_beats_degenerate_control_on_nll = false`、`..._on_balanced_accuracy = false`（v3/v5/v9 全部为 false）⇒
   **JEPA 在 K1 上不能声称优于任何基线**。唯一可说的是 `folded AUC`：`tICA+MSM` 0.696 有弱排序能力，JEPA ≤ 0.23（低于随机）。
3. gate `unfolded_kinetics_estimable = false`：标签文件 2600 帧里 `unfolded` 仅 **8 帧**，
   `unfolded → folded` 的动力学量（MFPT、速率）**在数据上不可估计**。K1 若要在论文里成立，
   必须换标签来源（例如按 `q_native` 阈值自行定义 native-contact 展开态与折叠态，而不是用预标注的 `state`）。

---

---
