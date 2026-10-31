# LASP 正确框架重建计划 (2026-10-31)

## 前提：旧 SOTA 已冻结

- `results/standard_eval/stopped_best/dynamics_seed{42,43,44}/checkpoint_best.pt`
  sha256 前缀 `138909db… / 93e90975… / 99d51a4a…`，与 `selection.json` 全部一致。
- 9 份评测产物（ensemble / matched8 / suite × 3 seed）齐全。
- 只读备份：`results/sota_frozen_20261024_backup`（362 MB，校验通过，`chmod a-w`）。
  后续任何实验**不得**写入上述两个路径。

## 已确证的缺陷

1. **两套不一致的动力学**（本计划的主线）
   - `sample_step(collective_mobility=True)`（训练+部署）：
     `H' = H + h·D·(f − C·S) + noise`, `S' = S + CᵀΔH − hΓS + √(2hΓ)ε`
   - `kalman_nll`（诊断，且被用来论证设计）：
     `mu = H + h·D·f + h·C·m`, `S' = (I−hΓ)S + h·Cᵀf + √(2hΓ)ε`
   - 差异：耦合项差因子 `D ≈ 2×10⁻⁵`，且符号相反；Kalman 缺少 `CᵀDC` 阻尼、
     缺少由 H 噪声注入的相关噪声。
   - **结论：sampler 是 FDT 自洽的（联合 mobility 由 `[I;Cᵀ]M[I,C]+diag(0,Γ)` 分解证明 PSD），
     Kalman 是错的。修复方向 = 让诊断与 sampler 对齐，而不是反过来。**

2. **S₀ 用 GRU 摊销推理**，无任何逼近真后验的保证，梯度份额 ~0.1–0.2%。

3. `rmsf` loss 在 `horizon=1` 目标恒为 0（已修，`target_x.shape[0]>=2` 守卫）。

## 关键推导：精确后验是闭式的

给定观测到的 H 路径，把 sampler 的联合噪声按 H 观测条件化：

```
S_{k+1} = S_k + CᵀΔH_k − hΓ S_k + [Cᵀ√(2hD)ε + √(2hΓ)ε_S]
```

若 H_{k+1} 已知，则 `√(2hD)ε_H = ΔH_k − hD(f_k − C S_k)` 被完全揭示，
于是给定历史路径：

```
S_{k+1} = (1 − hΓ) S_k + Cᵀ ΔH_k + η,     η ~ N(0, 2hΓ)
```

即 **零参数、闭式、精确**：
- `m₀ = Σ_j (1−hΓ)^{N−j} · Cᵀ ΔH_j`
- `P₀ = 对角`，递推 `P_{k+1} = (1−hΓ)² P_k + 2hΓ`，稳态 `≈ 1/(1−hΓ/2)`。

这条式子与 sampler 的 S 更新**逐项一致**，因此不会再次出现两套模型。
它同时给出 `kalman_nll` 的正确似然：
`H_{k+1}|H_k ~ N(H_k + hD(f_k − C m_k), R + hDC·P_k·(hDC)ᵀ)`。

## 第 1 步结果（2026-10-31）

**真正的 bug 是 Kalman 增益的符号，不是协方差。**

观测模型：`ΔH − hDf = −B_h S + noise`，故有效观测矩阵 `C_obs = −B_h`。
标准增益 `K = P C_obsᵀ V⁻¹ = −P B_hᵀ V⁻¹`（带负号）；旧代码用 `+P B_hᵀ V⁻¹`
且加到均值上，方向反了。Joseph 协方差对符号不敏感（`K C_obs` 两个负号抵消），
所以只有均值更新错、协方差一直是对的——这也解释了为什么单步 NLL 曾经"看起来对"。

- `K = −P B_hᵀ V⁻¹`，`m += K e`，`KB = −K B_h = K C_obs`。
- 修复后 `kalman_nll` 与独立 MC 验证过的 F/Q 递推一致到 **5.4e-5**（修复前差 1.888）。
- `tests/test_stage_b1b_kalman.py` 3 项通过；相关子集 132 passed / 1 skipped。
- 条件预测矩对采样器 MC 验证（`scripts/verify_conditional_predictive.py`）：
  `V_model 0.290 vs V_mc 0.289`，`mu` 三位小数一致，逐步稳定。
  注意：F/Q 的"边缘方差随时间增长"是无条件边缘，**不是**滤波的条件预测方差，
  两者不应相等——之前用错了对照对象。

**更正一条此前的错判：** 旧报告说 `bath_var/R ≈ 140×（高于 R）`。那是用错误的
`B_h = h·C`（漏了 D≈2e-5）算的。用正确 `B_h = h·D·C` 重测，三个 SOTA seed 的
`bath_var/R` 分别为 **4.5e-8 / 2.9e-8 / 6.0e-8**——即 bath 比 R 小约 **1e8 倍**，
彻底惰性。这加强了「旧框架没真正用 S」的判断，也意味着 S₀ 条件化在旧 kernel
下本就不可能起作用：必须先让 S 机制活起来（×50–100 耦合 + Γ×0.1），S₀ 才有空间。

## 第 2 步结果（2026-10-31）

**原先的「drive-only」闭式后验是错的，已替换为正确的 Kalman 后验。**

`exact_bath_posterior` 最初按 `S_{k+1} = (1−hΓ)S_k + CᵀΔH_k + η` 实现。它把均值传播
对了，但不是后验：观测 `H_{k+1}` 还通过新息**约束** `S_k`（每步信息量
`hDC/√(2hD)`，并不为零）。在真实锚点上，drive-only 均值与精确滤波均值相差：

| 尺度 | K=2 | K=4 | K=8 |
|---|---|---|---|
| 训练尺度 | 76% | 80% | 83% |
| ×100 耦合 + Γ×0.1 | 92% | 95% | 93% |

（标准差在训练尺度只差 ~1e-7，因为那时 bath 无信息、两者都退化为先验；×100 时差 0.4–2%。）

现在实现为完整 Kalman：观测 `z = ΔH − hDf = C_obs S + noise`，`C_obs = −hDC`，
`R = 2hD`，Joseph 更新后接采样器自己的时间更新。

验证（`tests/test_exact_bath_posterior.py`，3 项全过）：
- `C=0` 时后验严格等于 OU 先验递推（与历史无关）；
- 与独立手写递推一致到 **1e-10**（float64）；
- 与 **40 万样本**采样器联合分布的回归条件矩一致到 2e-2。

全套测试：**332 passed / 1 skipped**；4 个 `test_msm_metrics` 失败是环境缺 `sklearn`，与本次改动无关。

## 第 3 步结果（2026-10-31）

### 3a. 模型无关检验：历史里到底有没有超过 H₀ 的记忆？

`scripts/test_history_memory.py`，val split 200 窗口，蛋白 2 折交叉（测试蛋白不参与拟合），
逐残基、蛋白内中心化（去掉蛋白各自的平均漂移），岭回归从最近 7 帧相对位移预测未来位移：

| horizon | 零预测 MSE | 历史特征最优 | 降低 |
|---|---|---|---|
| 1 | 7.761 | 7.584 | **2.28%** |
| 4 | 23.056 | 22.529 | **2.29%** |
| 16 | 57.007 | 57.309 | **−0.53%** |
| 64 | 113.089 | 108.673 | **3.90%** |

**结论：在这批数据上，历史对未来的线性可提取记忆只有约 2–4%，h=16 甚至为负。**
这直接回答了「历史里绝对有超过 H₀ 的可预测记忆」：确实存在，但量级是百分之几，
远不足以支撑「用 S₀ 条件化显著改变预测」的期待。（线性是下界；但 3.9% 已经很小。）

### 3b. no-update 探针（`scripts/probe_s0_leverage.py`，seed42，60 窗口，同种子同噪声）

SOTA checkpoint 本身 `history_conditioning=False`，且其 `history_posterior.head`
权重**恰好全零**（初始化特征），所以未训练的 GRU 恒等于先验——这是旧框架
「名义上有历史、实际没用」的直接证据。

**训练尺度（S 惰性）**：prior/gru/exact 三者预测完全相同（杠杆 0.0000 Å），指标逐位相同。

**活跃尺度（×100 耦合、Γ×0.1）**：exact 相对 prior 的集合均值位移 =

| h | 1 | 4 | 16 | 64 |
|---|---|---|---|---|
| 杠杆 (Å) | 0.138 | 0.287 | 0.465 | 0.470 |

指标变化**大多略差**：h=4 `weak_contact_j` 0.1126 vs 0.1235、`transient` 0.2650 vs 0.2721；
h=16 `rmsf_corr` 0.5813 vs 0.5831；仅 h=64 `rmsf_corr`(0.5279 vs 0.5229) 和
`md_pca_w2`(9.566 vs 9.590) 略好。**杠杆是真实的，但没有转化为指标改进。**

## 执行步骤

| # | 步骤 | 产物 | 判据 |
|---|---|---|---|
| 1 | 修复 D 不一致：新增 `exact_bath_posterior()`，重写 `kalman_nll` 使其与 `sample_step` 同模型 | `model.py` + `tests/` | 新旧似然在 `C=0` 极限下逐点相等；单测通过 |
| ~~1~~ | **✅ 已完成 2026-10-31** | — | 见「第 1 步结果」 |
| 2 | 用精确滤波替换 GRU 作 S₀（保留 GRU 为可选对照） | `model.py`, `train_lasp_ca_dynamics.py` 新 flag | 与逐帧手工递推数值一致（1e-6） |
| 3 | no-update 探针：精确 S₀ vs 先验 vs GRU，测预测偏移与 7 项指标 | `results/.../exact_s0_probe.json` | 预测偏移须显著 > 0.065 Å（旧 S₀ 的量级） |
| 4 | 训练对照臂（同 recipe，仅 S₀ 来源不同）：`exact` / `prior` / `gru`，3 seed | `artifacts/lasp_exact_s0_20261031/{arm}_seed{42,43,44}` | 训练有限值，val 收敛 |
| 5 | standard-200 配对评测 + protein-cluster bootstrap | `results/lasp_exact_s0_20261031/` | 98.33% 单侧上界 |
| 6 | 决策：是否超 SOTA | 报告 | 逐指标 + 复合，按 horizon 分解 |

## 预注册判据（避免"理论正确"变成不可证伪）

- 主指标（事先定）：`rmsf_corr`、`md_pca_w2`、`variance_w2` 在 h∈{1,4,16,64} 上的配对差。
- 显著性：protein-cluster bootstrap（10000 draws, seed 20261024），98.33% 单侧上界。
- 若 `exact` 相对 `prior` 的改善 **在所有主指标上都不超过 seed 噪声**，则结论为
  "该数据/窗口下历史不含超出 H₀ 的可预测记忆"，而不是"框架不对"。
- 任何 oracle（用到未来帧）结果不计为模型胜利。

## 约束

- `test.jsonl` 保持不打开。
- 不重新评测 BioEmu2、不重评 v3。
- 仅用本会话创建的进程可 kill，且 kill 前 `ps -p <pid> -lstart,args` 核验。
- 不设 `HF_ENDPOINT`。临时目录用 `/tmp/lasp_review_runtime`。

## 第 4 步：训练对照臂（进行中）

### 训练预算的合法性：SOTA 本身在 7000–9000 步达到最优

冻结 SOTA 的 val 曲线（17–21 个验证点）显示 val 损失在 **step 7000–9000** 触底，
之后**单调恶化**（过拟合），checkpoint 由 val 选择：

| seed | val 点数 | 最优 step | 最优 val | 最终 val | 后悔 |
|---|---|---|---|---|---|
| 42 | 20 | 7000 | 4.4230 | 5.2460 | +0.823 |
| 43 | 21 | 9000 | 4.4006 | 5.1656 | +0.765 |
| 44 | 17 | 7000 | 4.4370 | 5.1541 | +0.717 |

**因此 SOTA 冻结权重本质上是一个 ~7k 步的模型**，`--steps 30000` 只是训到 30000
再回头选最优。本计划的 7000 步预算因此是**与 SOTA 对齐**的，不是打折。
（对照臂 val-every 500，比 SOTA 的 1000 更细，选择更准。）

### 臂的设计

三臂同 recipe、同 seed、同噪声，**唯一差别是 S₀ 来源**：
`exact`（完整 Kalman 后验）/ `prior`（恒为零均值、单位协方差）/ `gru`（旧的摊销推理）。
全部 `--history-conditioning` 打开、活跃尺度（`coupling_scale 0.1`、`gamma_scale 2e-5`，
即 ×100 耦合、Γ×0.1）。

**`prior` 是主对照**：它与 `exact` 的差别**只有 S₀**，因此 `exact vs prior` 是
唯一干净的因果对比。`exact vs SOTA` 混杂了步数/PyTorch 尺度/历史开关，只作次要参考。

### seed 噪声地板（重要的先验校准）

在**任何** S₀ 实验之前，先量化了 SOTA 自己的 seed 间差异（同 200 窗口、同配对协议、
protein-cluster bootstrap 10000 次 / seed 20261024）：

| h | 显著 win | 显著 loss | 不确定 | 共 21 项 |
|---|---|---|---|---|
| 1 | 5 | 3 | 13 | 3 对 seed × 7 指标 |
| 4 | 5 | 1 | 15 | |
| 16 | 1 | 1 | 19 | |
| 64 | 3 | **10** | 8 | |

h=64 上 SOTA 的一个 seed 相对另一个 seed 在 10/21 项上**显著更差**。
seed 间 `|Δ|` 的量级：`rmsf_corr` 0.003/0.005/0.009/0.026，
`md_pca_w2` 0.05/0.06/0.10/0.25，`variance_w2` 0.11/0.10/0.10/1.10（h=1/4/16/64）。

**这意味着：S₀ 效应必须在 3-seed 均值上越过这个地板才算数。**
单 seed 的"显著"在这个数据上不具备说服力。

### 预注册判据（在训练前固定）

- 主指标：`rmsf_corr`、`md_pca_w2`、`variance_w2`。
- 配对：窗口键完全相同（protein/traj/temperature/anchor/horizon），
  且 sampler 噪声由窗口键派生的种子决定 —— 两臂看到**同一条噪声路径**。
- 聚簇 bootstrap 重采样**蛋白**（非窗口），10000 次，seed 20261024，单侧 98.33%。
- **WIN** 需同时满足：3-seed 均值改善 > 0 **且** ≥2/3 seed 的单侧下界 > 0。
  **LOSS** 对称。其余记 INCONCLUSIVE。
- 主指标全部 INCONCLUSIVE ⇒ 结论是"该数据下历史不含超出 H₀ 的可预测记忆"，
  而不是"框架不对"（3a 的 2–4% 线性记忆已预示这个方向）。

### 工具

- `run_scripts/run_lasp_s0_arms.sh` — 训练网格（2 GPU 波次调度）。
- `run_scripts/run_lasp_s0_arms_eval.sh` — 每臂 standard-200 集成指标。
- `scripts/compare_s0_arms.py` — 单 seed 配对 + protein-cluster bootstrap（预注册参数）。
- `scripts/report_s0_arms.py` — 3-seed 聚合 + 预注册 WIN/LOSS/INCONCLUSIVE 判定。
- `results/lasp_s0_arms_20261031/sota_seed_noise_floor.{json,txt}` — seed 地板。

### 需要在训练后核验的机制诊断（防止把"S₀ 没接入"误读成"S₀ 没用"）

`exact` 与 `prior` 若收敛到同一模型，有两种完全不同的原因：

(a) 历史确实没有超出 H₀ 的记忆 → 优化器把 C 压小，两条臂自然等价（这是**结论**）；
(b) 机制没接上（梯度断了 / C 太小 / S 仍惰性）→ 两臂等价（这是**bug**）。

区分方法，在 wave-1 checkpoint 落地后立刻做：

1. **训练后的 C 量级**：比较 `c_net[-1].weight.std() * coupling_scale` 与初始化值（0.1×0.1=1e-2）。
   若 `exact` 臂把 C 训小了而 `prior` 臂没有，说明优化器在**主动关闭** S（支持 (a)）。
2. **S₀ 杠杆（在各自 checkpoint 的真实尺度上）**：`scripts/probe_s0_leverage.py
   --scale checkpoint`，对每条臂测 exact/gru/prior 的集合均值位移。
   若 exact 臂上杠杆 ≈ 0 而 C 也 ≈ 0，则 (a)；若 C 不小但杠杆仍 ≈ 0，则 (b)。
3. **梯度路径**：确认 loss 对 `c_net` 的梯度非零（初始 C 非零初始化已保证，见 model.py:282-286）。

初始化时的量级估算（active scale，C ~ N(0, 0.1²)、coupling 0.1）：
`|C m₀| ≈ 1.2e-4`（力单位），而 `|f| ≈ 4e-2`，即 **0.3%**；
单步位移贡献 `dt·D·|C m₀| ≈ 2.6e-5`，扩散 `sqrt(2dtD) ≈ 4.5e-1`，**6e-5**。
也就是说在**初始化参数**下 S₀ 几乎不可能影响预测；杠杆必须由训练把 C 放大才会出现。
这正是为什么必须看训练后的 C，而不是看初始化的 C。

### 调度修正：4 路并发（2026-10-31 17:46）

实测该机器是 **CPU 瓶颈而非 GPU 瓶颈**：单进程占 ~0.86 核，A800 利用率仅 10–25%，
显存 6.8/80 GB。176 核、load 8.3。

| 并发 | 单进程 s/step | 单位时间完成的 step 数 |
|---|---|---|
| 2（每 GPU 1 个） | 1.24 | 0.36 s/step-equivalent |
| **4（每 GPU 2 个）** | **1.17–1.49** | **0.26–0.37（≈1.66×）** |

因此改为 `run_scripts/drive_lasp_s0_arms.sh`（槽位驱动，谁空谁补，已完成/在跑的
会跳过，可中途启动）。原波次脚本 `run_lasp_s0_arms.sh` 的固定波次会在某个臂已存在时
产生只跑 1 个任务的空波，已弃用。

已停止旧 launcher 的三个 bash 进程（1362786/1362788/1362789，本会话创建，已用
`ps -o lstart` 核验），**4 个训练进程全部存活**（1362790/1362791/1386925/1386926）。

### 陷阱：`checkpoint_best.pt` 在 step 0 就存在

`save('best',0,initial)` 在训练开始时就写出 `checkpoint_best.pt`（val 选择从 step 0 起算）。
因此**"best 存在"不等于"训练完成"**——用它做完成判据会把崩溃的 run 误判为已完成。

正确的完成判据是 `checkpoint_latest.pt` 里记录的 `step >= STEPS`。
`drive_lasp_s0_arms.sh` 和监控脚本都已改用这个判据。

## 第 3c 步补充：冻结 SOTA 的耦合 C 在 30k 步里**从未长大**

`scripts/diagnose_s0_mechanism.py`（新）对冻结 SOTA seed42 的直接测量：

| 量 | 值 |
|---|---|
| `coupling_scale` | 1.000e-03 |
| c_net 输出层权重 std × coupling_scale（**训练后**） | **9.999e-05** |
| 初始化应为 c_init_std × coupling_scale | 1.000e-04 |
| `|C m₀| / |f|` | 3.2e-05 |
| 单步 bath 均值位移 | 7.4e-06 |
| 单步扩散 | 1.02e-01 |
| S₀ 杠杆（三个来源） | 全部 0.0000 Å |

**训练后有效耦合与初始化值相同到 4 位有效数字**（9.999e-05 vs 1.000e-04）。
也就是说 30 000 步训练完全没有把 C 推离初始化——不是"训小了"，是**从来没动过**。
配合 bat h 方差仅为 R 的 ~1e-8，这解释了旧框架里 S 为何彻底惰性：
S 不进 H 的演化，S₀ 换成什么都不可能改变预测。

这条也给出第 4 步的判读基准：**如果 `exact` 臂训练后 effective coupling 仍是
9.999e-05 量级，说明 active scale 也没能让优化器使用 S**，那么 exact≈prior
是机制问题（(b)），不能当作"S₀ 没用"（(a)）的证据。

**6 路实测（2026-10-31 17:52，6 个训练进程同时）：**
exact_seed42 从 step 1020（elapsed 1627.4s）到 step 1040（elapsed 1652.0s）——
**20 步用 24.6 s = 1.23 s/step**，与 2 路时的 1.24 s/step **几乎相同**，
即并发从 2 提到 6 吞吐约 3×，单任务不减速。此时 GPU 利用率已达 99%，为上限。
显存 14/80 GB，176 核 load ~9，均有余量；瓶颈是 GPU 而非 CPU（与 2 路时的判断相反，
2 路时 GPU 只有 10–25% 是因为任务本身没喂满）。

结论：`MAX_JOBS=6` 是这台机器的合适工作点。

### 修正后的排期

9 个 run × 7000 step × 1.23 s = 77 490 s 的 GPU 工时，6 路并发 ≈ **3.6 h**
（自 17:52 起算，约 21:30 完成训练）。评测 9 个 checkpoint（standard-200，
每臂 ~10 min，2 路）≈ 45 min，预计 **22:30 前**出全部数字与判定。

**6 路下的真实单步时间（用新启动的 exact_seed44 排除验证停顿的污染）：**
step 130 完成于 elapsed 142.8 s → **1.10 s/step**（旧臂的"s/step"列因包含
val（每 500 步一次、每次采样 64 窗口）而系统性偏高，不能直接用来算排期）。

据此：7000 × 1.10 ≈ 2.15 h/run，9 run / 6 槽 ≈ 1.5 波 ≈ **3.2 h**，
自 17:52 起算约 **21:10** 训完，评测 9 个 checkpoint 约 45 min，22:00 前可出判定。

**最终实测（exact_seed44，100 步干净窗口）：1.167 s/step @ 6 路。**

**全套测试（2026-10-31 18:01）：332 passed / 1 skipped / 4 failed。**
4 个失败全部是 `tests/test_msm_metrics.py` 的 `ModuleNotFoundError: No module named 'sklearn'`
（`evaluation/msm_metrics.py:142`），是本机环境缺包，与本次 LASP 改动无关，改动前即如此。

### 早期观察：优化器在 warmup 内没有推动耦合 C

在 step 500–1000（warmup=1500 尚未结束）时，两臂的 effective coupling 仍在

| run | step | eff coupling | 初始化 |
|---|---|---|---|
| exact_seed42 | 1000 | 1.0036e-02 | 1.0000e-02 |
| exact_seed43 | 1000 | 9.9710e-03 | 1.0000e-02 |
| prior_seed42 | 500 | 1.0037e-02 | 1.0000e-02 |
| prior_seed43 | 500 | 9.9734e-03 | 1.0000e-02 |

与冻结 SOTA 30k 步后仍是初始化值一致。这不必是缺陷：warmup 期间 lr 很小，
且 C 的变化速率取决于"S 是否有用"。但它意味着**判读时必须看训练结束时的 C**：
若 7000 步后 C 仍 ≈ init，则 exact vs prior 的差别完全来自 S₀ 在固定耦合下的
直接杠杆（探针在 1e-2 耦合处已测得 0.14–0.47 Å），而不是"学到了怎么用历史"。
这两种情形都要如实写进报告。

**尺度等价性核验：** 对照臂的 effective coupling 1.0036e-02 与探针的 "active"
设定（SOTA 的 C × 100 = 1.0000e-02）在数值上等价。因此第 3b 步在冻结 SOTA 上
测到的 0.138/0.287/0.465/0.470 Å 杠杆，可以**直接**用来预期这些臂的行为。
