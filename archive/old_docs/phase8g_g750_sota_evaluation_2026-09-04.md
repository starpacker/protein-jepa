# Phase 8G / G-750 内部 SOTA 评估报告

**评估日期：** 2026-09-04

**Checkpoint：** `artifacts/phase8g_tmjepa_v3_overdamped_longhorizon/checkpoint_step750.pt`

**结论：** G-750 是当前 Phase-8 TM-JEPA V3 中 endpoint energy score 和长程预测的内部 SOTA checkpoint，但不是全指标 SOTA，尚未通过完整 promotion gate，也不能据此宣称外部 benchmark SOTA。

## 1. 结论摘要

在统一评估代码、验证集、采样 seed 和 Monte Carlo 设置下，G-750 获得：

- 三个独立评估 seed 的五 horizon macro energy skill：**20.22 +/- 0.75%**；
- 旧主 baseline C1-500：**18.64 +/- 0.82%**；
- 8 ns skill：**16.93 +/- 0.30%**，C1 为 **14.48 +/- 0.18%**；
- 16 ns skill：**3.77 +/- 2.09%**，C1 为 **0.23 +/- 2.12%**；
- seed 6303 下，G-750 在 2、4、8、16 ns 均为所比较 Phase-8 checkpoints 的最优值，1 ns 排名不是第一；
- 相同随机样本下，G-750 在三个 seed、全部五个 horizon 上均略优于 G-500，说明继续训练到 750 steps 的方向一致，但增益较小。
- 在 legacy 30-protein MSM label benchmark 上，G-750 的 aligned frozen probe 达到 **51.30% accuracy / 43.79% balanced accuracy**，显著高于旧 V5.1 probe 的 **36.22% / 26.88%**；但 aligned VAMP-CV lag1 dim20 normalized 为 **0.455-0.457**，未超过 centered-CA baseline **0.462**。

G-750 解决了旧 C1 在 16 ns 上不优于 persistence 的主要失败，并显著改善了扩散矩阵的实用非退化性和条件数。然而，以下问题仍然存在：

- 90% coverage 在 8/16 ns 普遍过高，长程分布过宽；
- 320 K、348 K 的 innovation covariance closure 未达门限；
- lag-1 innovation correlation 约 0.11，明显高于 0.05 门限；
- path NLL 比 C1 差；
- 16 ns 的最弱温度单元仍接近零，正 skill 尚不稳健；
- numerical CK distance 随 horizon 增大，长程半群一致性仍弱。

所以准确表述应为：

> **G-750 是当前固定 learned-CV / PMF 条件下，Phase-8 概率 endpoint rollout 与长程 energy skill 的内部 SOTA；它是下一轮改进的 champion checkpoint，而不是已经完成物理闭合与校准的最终模型。**

## 2. 模型与训练设置

G-750 使用分阶段训练，而不是把所有模块同时更新：

| 模块 | G 阶段状态 | 来源/作用 |
|---|---:|---|
| Context encoder | frozen | Phase 8A2 JEPA representation |
| Causal temporal encoder | frozen | 固定长度 causal history representation |
| Learned CV projection | frozen | Phase 8A2 learned 4D slow CV |
| Thermodynamic potential / PMF | frozen | Phase 8B2 multi-noise DSM PMF |
| Kinetics / mobility / diffusion | trainable | G 阶段唯一主要更新的动力学模块 |
| Mori memory | disabled | 当前数据未证明 Mori memory 带来稳定优势 |

关键配置如下：

| 配置 | 值 |
|---|---:|
| G 阶段初始化 | C1 checkpoint step 500 |
| G 阶段训练步数 | 750 |
| Optimizer learning rate | `2e-5` |
| Energy-score weight | `0.5` |
| Path-NLL weight | `1.0` |
| Diffusion floor | `5e-7` |
| Base integration step | 1000 ps |
| Forecast horizons | 1/2/4/8/16 ns |
| Ensemble size | 8 |

动力学主体是 overdamped thermodynamic diffusion：使用 learned PMF 提供自由能梯度，kinetics head 预测依赖状态和温度的 mobility/diffusion，通过 Eyring-Einstein temperature clock、exact Ito divergence 和 local exact OU/FDT integrator 进行 rollout。此版本禁用了 Mori bath，因此本报告不把 Mori memory 作为 G-750 的贡献。

## 3. 统一评估协议

正式比较使用修复后的 `scripts/evaluate_tmjepa_v3.py`：

| 项目 | 设置 |
|---|---:|
| Validation manifest | `manifests/tmjepa_v3.validation.jsonl` |
| Validation trajectories | 80 |
| Proteins | 20/20 covered |
| Temperatures | 320/348/379/413 K |
| Sampled contexts per seed | 200 |
| Endpoint rows per seed | 1000（200 contexts x 5 horizons） |
| Closure transitions per seed | 3400 |
| Batch size / batches | 4 / 50 |
| Independent evaluation seeds | 6303, 6304, 6305 |
| CK evaluation | seed 6303, 8 samples |

Energy skill 的定义为：

`1 - model_energy_score / persistence_energy_score`

因此 skill 大于零表示概率预测优于“未来 CV 与当前 CV 不变”的 persistence baseline。Energy score 越低越好，skill 越高越好。

评估器同时修复了一个 RNG 污染问题：旧实现中的 CK sampling 会消耗全局随机流，使开启 CK 后的后续 endpoint samples 发生变化。现在 CK 使用隔离的随机流；同一 G-750 运行的 CK0 和 CK8 endpoint metrics 已逐值验证完全一致。结果文件也新增了 `evaluation_protocol`，记录 seed、batch 数、sampled contexts、CK samples 和 lags。

## 4. Phase-8 checkpoints 对比

以下为严格统一的 seed 6303 结果。表中数字均为相对 persistence 的 energy skill，越高越好。

| Checkpoint | 1 ns | 2 ns | 4 ns | 8 ns | 16 ns | Macro |
|---|---:|---:|---:|---:|---:|---:|
| C1-500, old overdamped | 28.562% | 27.058% | 22.053% | 14.600% | -0.262% | 18.402% |
| C2-500, longer-schedule branch | **28.805%** | 27.118% | 21.247% | 12.723% | -3.531% | 17.273% |
| D2-1000, Mori floor | 28.661% | 26.738% | 21.428% | 13.218% | -2.614% | 17.486% |
| E-1000, overdamped high floor | 28.667% | 26.571% | 21.082% | 12.553% | -3.592% | 17.056% |
| G-500, long-horizon objective | 28.543% | 27.608% | 23.404% | 16.967% | 3.221% | 19.949% |
| **G-750** | 28.546% | **27.626%** | **23.445%** | **17.047%** | **3.329%** | **19.999%** |

G-750 的优势集中在中长程。1 ns 与最优 checkpoint 的差距只有 0.26 percentage point，但它在 4/8/16 ns 上明显更强。D2 的 Mori memory 没有转化为更好的 endpoint skill，因此当前数据支持继续把 overdamped G 作为 champion。

## 5. G-750 对旧主 baseline C1

以下仍为完全配对的 seed 6303 结果。G 与 C1 使用相同 learned CV，因此 persistence score 完全相同，可以直接比较 model energy score。

| Horizon | Persistence ES | C1 model ES | G-750 model ES | C1 skill | G-750 skill | Skill change |
|---|---:|---:|---:|---:|---:|---:|
| 1 ns | 0.727363 | 0.519617 | 0.519731 | 28.562% | 28.546% | -0.016 pp |
| 2 ns | 0.804699 | 0.586967 | 0.582395 | 27.058% | 27.626% | +0.568 pp |
| 4 ns | 0.888395 | 0.692474 | 0.680107 | 22.053% | 23.445% | +1.392 pp |
| 8 ns | 0.919183 | 0.784985 | 0.762486 | 14.600% | 17.047% | +2.448 pp |
| 16 ns | 1.010848 | 1.013494 | 0.977200 | -0.262% | 3.329% | +3.590 pp |

G-750 相对 C1 的 model energy score 在 8 ns 降低 2.87%，在 16 ns 降低 3.58%。1 ns 基本不变，说明 G 的改进不是用短程性能大幅交换长程性能。

## 6. 三个评估 seed 的稳定性

| Checkpoint | 1 ns | 2 ns | 4 ns | 8 ns | 16 ns | Macro |
|---|---:|---:|---:|---:|---:|---:|
| C1-500 | 28.43 +/- 0.40% | 27.68 +/- 0.55% | 22.39 +/- 1.55% | 14.48 +/- 0.18% | 0.23 +/- 2.12% | 18.64 +/- 0.82% |
| G-500 | 28.45 +/- 0.39% | 28.19 +/- 0.51% | 23.71 +/- 1.46% | 16.86 +/- 0.29% | 3.66 +/- 2.10% | 20.17 +/- 0.75% |
| **G-750** | **28.45 +/- 0.39%** | **28.21 +/- 0.51%** | **23.75 +/- 1.46%** | **16.93 +/- 0.30%** | **3.77 +/- 2.09%** | **20.22 +/- 0.75%** |

这里的标准差同时包含不同 validation contexts 和 ensemble Monte Carlo sampling 的变化。G-750 相对 G-500 的增益较小，但配对差值在三个 seed 中方向完全一致：2/4/8/16 ns 平均分别为 `+0.019/+0.039/+0.075/+0.103 pp`。

16 ns 的方差仍较大。G-750 在三次评估中的 16 ns aggregate skill 为 `3.329/6.042/1.924%`；因此可以报告“平均为正且优于 C1”，但不应声称长程结果已经低方差收敛。

## 7. Legacy MSM/VAMP 表示评估

用户关心的旧指标不能直接从 Phase-8 endpoint validation 推断，因为两套数据不同：

| Protocol | Proteins | Trajectories | 作用 |
|---|---:|---:|---|
| Phase-8 endpoint validation | 20 | 80 | 评估 1/2/4/8/16 ns probabilistic rollout |
| Legacy MSM/VAMP benchmark | 30 | 354 | 评估 MSM label decodability 与 VAMP slow-mode readout |

因此这里额外把 G-750 放回 legacy validation manifest 和 MSM labels 下做 frozen-readout probe。Readout 均只使用当前/历史 context，不使用 future target、rollout endpoint 或 MSM labels 参与模型编码。

### 7.1 MSM trajectory-heldout probe

Aligned legacy protocol：`mdcath.deeptime_lcc_pcca_v2.full_lag1.validation.filtered.jsonl`，对应 filtered labels，24-frame context，`frame_stride=2`，`window_stride=8`，最多 8000 clips，per-protein trajectory holdout，ridge lambda 在 train trajectories 内部选择。

| Readout | Accuracy | Balanced accuracy | Macro-F1 | Majority baseline | Proteins above baseline |
|---|---:|---:|---:|---:|---:|
| G-750 `tmjepa_cv` | 50.77% | 42.91% | 38.09% | 37.90% | 21/30 |
| G-750 `tmjepa_cv_condition_kinetics` | **51.30%** | **43.79%** | **39.43%** | 37.90% | 21/30 |
| V5.1 `v51_state_memory` old probe | 36.22% | 26.88% | 22.42% | 37.90% | not recorded here |
| TICA+MSM direct assignment | 31.67% | 18.99% | - | - | - |
| VAMPNet direct assignment | 26.13% | 18.91% | - | - | - |
| SPIB direct assignment | 24.82% | 20.27% | - | - | - |

这说明 G-750 的 learned/fixed CV 坐标里确实包含更强的 metastable-state 可解码信息。需要注意两点：第一，G-750 数字是 frozen representation + ridge probe，不是模型自身输出 6-state MSM classifier；第二，传统 baseline 行是 direct assignment 指标，协议语义不同，只能作为 landscape 参照，不能当作严格同类模型 head 对比。

为了诊断 readout 来源，也跑了 G 自身训练采样协议 `frame_stride=1`：

| Readout | Accuracy | Balanced accuracy | Macro-F1 | Majority baseline | Proteins above baseline |
|---|---:|---:|---:|---:|---:|
| `tmjepa_cv` | 53.13% | 46.09% | 42.94% | 37.98% | 25/30 |
| `tmjepa_condition` | 38.28% | 27.71% | 15.82% | 37.98% | 4/30 |
| `tmjepa_cv_condition` | 52.89% | 45.72% | 43.25% | 37.98% | 24/30 |
| `tmjepa_kinetics` | 48.93% | 41.96% | 37.88% | 37.98% | 20/30 |
| `tmjepa_cv_condition_kinetics` | **53.39%** | **46.18%** | 42.38% | 37.98% | 23/30 |

这里 `tmjepa_condition` 单独接近 majority baseline，说明 MSM probe 的提升主要来自 CV/kinetic state，而不是 protein ID 或 temperature condition 泄漏。`tmjepa_kinetics` 单独有效但弱于 CV，说明 kinetics head 学到了状态依赖速率结构，但尚未把它转化为超过 CV 本身的慢态表示。

### 7.2 VAMP-CV

Aligned legacy VAMP protocol：50 protein-stratified trajectories，24-frame context，`frame_stride=2`，`window_stride=8`，blocked 5-fold CV，gap 20，deeptime VAMP，normalized score 为 `(VAMP2 - 1 constant mode) / kinetic_dim`。

| Readout / baseline | lag1 dim5 | lag1 dim20 | lag10 dim5 | lag10 dim20 |
|---|---:|---:|---:|---:|
| G-750 `tmjepa_cv` | 0.5483 | 0.4570 | 0.6552 | 0.3744 |
| G-750 `tmjepa_cv_condition_kinetics` | 0.5468 | 0.4551 | 0.6579 | 0.3810 |
| `geometry_stats` | 0.5325 | 0.4575 | 0.6473 | 0.3755 |
| `centered_ca` | **0.5731** | **0.4624** | **0.6668** | **0.3920** |

VAMP 的结论比 MSM 更严格：G-750 的 CV readout 高于 compact geometry stats 的 lag1/lag10 dim5，但没有超过 centered-CA baseline，也没有超过旧 V5.1 lag1 dim20 `0.4565` 的水平。融合 condition/kinetics 没有系统提升 VAMP，说明当前 kinetics readout 并没有形成更清晰的 Koopman slow modes。

综合 legacy 指标：G-750 的 state decodability 很强，但 slow-mode/VAMP 仍弱。这与 endpoint evaluation 中的 lag-1 correlation、path NLL 和 CK distance 问题一致：表示空间已有可用慢态信息，转移核和噪声闭合还没有把这些信息变成严格的动力学一致性。

## 8. Calibration、closure 与物理诊断

### 8.1 90% marginal coverage

| Horizon | C1 | G-750 | 目标区间 |
|---|---:|---:|---:|
| 1 ns | 84.97% | 85.25% | 85-95% |
| 2 ns | 89.83% | 90.22% | 85-95% |
| 4 ns | 93.62% | 93.95% | 85-95% |
| 8 ns | 95.88% | 96.17% | 85-95% |
| 16 ns | 97.02% | 97.12% | 85-95% |

G-750 在 2/4 ns 附近校准合理，但 8/16 ns 明显过覆盖，表示预测分布随 rollout 累积得过宽。另一方面，分温度后 320 K 和 348 K 的 1 ns coverage 分别只有 83.56% 和 82.45%，说明不能用一个全局 variance multiplier 同时解决短程欠覆盖和长程过覆盖；下一版需要 horizon-conditioned、可能再加 temperature-conditioned 的 calibration。

### 8.2 Closure 与 likelihood

以下为三 seed mean +/- sample standard deviation：

| Metric | C1-500 | G-750 | 判读 |
|---|---:|---:|---|
| Innovation mean RMS | 0.01762 +/- 0.00141 | **0.01652 +/- 0.00133** | G 略好 |
| Innovation covariance RMSE | 0.14831 +/- 0.00646 | **0.14523 +/- 0.00264** | G overall 略好 |
| Path NLL / dim | **-0.89541 +/- 0.02161** | -0.84257 +/- 0.01851 | G 明显更差 |
| Lag-1 correlation RMS | **0.10894 +/- 0.00076** | 0.11247 +/- 0.00175 | G 更差且均未闭合 |

G 的 overall covariance RMSE 小于 0.15，但 seed 6303 下 320 K、348 K 分别为 0.1769、0.1757，因此 per-temperature covariance gate 失败。Lag-2/4/8 correlation 已接近 0.03，主要问题集中在 lag 1，说明当前一步转移核仍遗漏短时可预测结构，而不是所有时间尺度都存在同等强度的 memory。

### 8.3 Diffusion diagnostics

| Metric | C1-500 | G-750 | 约束 |
|---|---:|---:|---:|
| Minimum diffusion eigenvalue | `9.14e-8` | **`3.48e-7`** | > 0 |
| Minimum increment std | 0.0135 | **0.0264** | >= 0.01 |
| Mean condition number | 251.4 | **68.0** | 越低越好 |
| Maximum condition number | 365.4 | **80.4** | <= 100 |

G-750 同时通过 PSD、non-degenerate increment 和 condition-number gates。这是 G 相对旧 C1 最明确的物理数值改进之一：它不再依赖近退化、病态条件的扩散矩阵来获得短程 likelihood。

### 8.4 Numerical CK diagnostic

修复 RNG 隔离后的 G-750 CK8 energy distance：

| Horizon | 1 ns | 2 ns | 4 ns | 8 ns | 16 ns |
|---|---:|---:|---:|---:|---:|
| CK energy distance | 0.245 | 0.332 | 0.480 | 0.649 | 0.870 |

该距离随 horizon 单调增大，说明直接 rollout 与分段组合 rollout 的分布差异持续累积。当前 evaluator 尚未给 CK distance 设置 promotion threshold，因此它只能作为重要诊断，不能被表述为“CK 已通过”。

## 9. Promotion gate 状态

| Gate | G-750 | 原因 |
|---|---:|---|
| Complete temperature x horizon grid | PASS | 4 temperatures x 5 horizons 均有样本 |
| Complete protein coverage | PASS | 20/20 proteins |
| Positive skill in every temperature x horizon cell | FAIL | seed 6303 的 379 K / 16 ns 为 -0.170% |
| Coverage calibrated per temperature | FAIL | 短程局部欠覆盖，长程普遍过覆盖 |
| Covariance closed per temperature | FAIL | 320/348 K covariance RMSE > 0.15 |
| Multi-lag white per temperature | FAIL | lag-1 correlation RMS 约 0.11 |
| Diffusion PSD | PASS | minimum eigenvalue > 0 |
| Diffusion nondegenerate | PASS | minimum increment std > 0.01 |
| Diffusion well-conditioned | PASS | maximum condition number < 100 |
| Mori filter predictive | N/A | G 禁用 memory；代码中的 true 是 bypass，不是 Mori 成功证据 |

`promotion_pass = false`。这与“endpoint/long-horizon internal SOTA”并不矛盾：G 是当前最好的候选，但尚未满足我们为最终模型设定的完整可靠性标准。

## 10. 下一轮改进优先级

1. **保留 G-750 作为初始化点。** 它已经建立了更好的长程 skill 和实用 diffusion，不应回到高 diffusion floor 的 D2/E。
2. **直接优化 lag-1 closure。** 加入小权重 innovation autocorrelation loss，并监控 path NLL，目标是降低 lag-1 correlation，同时不损失 8/16 ns skill。
3. **分 horizon 校准方差。** 学习受限的 horizon-conditioned variance scale；必要时加入温度项，避免全局缩放伤害 320/348 K 的 1 ns coverage。
4. **加入适度 covariance closure loss。** 优先修复 320/348 K，不以 overall average 掩盖条件失败。
5. **加入 representation-preserving kinetic regularization。** Legacy MSM 显示 CV 可分性已经强，VAMP 未过 centered-CA；下一版应小权重加入 VAMP/lag-correlation 正则或只解冻 learned CV projection 的最后层，防止 endpoint skill 提升时慢模态退化。
6. **把 CK consistency 变成训练与 promotion 指标。** 先定义可重复的直接/组合 rollout 比较和阈值，再决定是否加入 semigroup distribution loss。
7. **每个候选先做统一 CK0 screening，再做 CK8。** Endpoint 指标必须使用隔离随机流；所有结果至少报告三个 seed，16 ns 必须报告方差和最弱温度单元。

下一轮不能只延长相同 G 配置的训练步数。G-500 到 G-750 的增益稳定但很小，当前瓶颈已经从“是否有长程 skill”转移到“短时 closure、conditional calibration 和 CK consistency”。

### 10.1 Phase-8H/I follow-up probes

在本轮 legacy 评估后，追加跑了两个从 G-750 初始化的 closure probe：

| Branch | 目标 | 训练 | 结论 |
|---|---|---:|---|
| Phase-8H closure-calibrated | `energy=0.45, path=0.8, cov=0.1, autocorr=0.25` | 750 steps | covariance/path 略好，但 8/16 ns endpoint 明显退化，不 promote |
| Phase-8I micro-closure | `energy=0.5, path=1.0, cov=0.02, autocorr=0.75` | 250 steps | 基本保住 endpoint，closure 只有微小改善，可作为后续小扰动方向 |

Seed 6303、同一 endpoint validation protocol 的直接对比：

| Checkpoint | 1 ns | 2 ns | 4 ns | 8 ns | 16 ns | Macro | Cov RMSE | Path NLL | Lag1 corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G-750 | **28.546%** | **27.626%** | **23.445%** | **17.047%** | **3.329%** | **19.999%** | 0.145625 | -0.842227 | 0.110687 |
| H-250 | 28.618% | 27.400% | 22.869% | 15.948% | 1.580% | 19.283% | 0.140004 | -0.846311 | 0.110747 |
| H-750 | 28.616% | 27.412% | 22.880% | 15.961% | 1.590% | 19.292% | **0.139853** | **-0.848499** | 0.110841 |
| I-250 | 28.550% | 27.612% | 23.405% | 16.959% | 3.191% | 19.943% | 0.144907 | -0.843201 | **0.110610** |

Interpretation：H 证明较强 covariance closure 会换掉长程 skill；I 证明低 LR、小 covariance、保留原 path/energy 主目标基本不会破坏 G。当前 champion 仍为 G-750，因为 I-250 没有超过 macro 和 16 ns skill。下一步如果继续这个方向，应在 I 的基础上只做更细的 loss schedule，而不是上调 covariance 权重。

## 11. 可复现实验产物

- 正式 G-750 CK8：`artifacts/sota_eval_20260904/g750_seed6303_ck8.json`
- G-750 三 seed CK0：`artifacts/sota_eval_20260904/g750_seed{6303,6304,6305}_ck0.json`
- G-500 三 seed CK0：`artifacts/sota_eval_20260904/g500_seed{6303,6304,6305}_ck0.json`
- C1 三 seed CK0：`artifacts/sota_eval_20260904/c1_seed{6303,6304,6305}_ck0.json`
- Phase-8 single-seed controls：`artifacts/sota_eval_20260904/{c2,d2,e}_seed6303_ck0.json`
- Legacy MSM/VAMP：`artifacts/paper_eval/phase8g_g750_legacy/`
- Phase-8H diagnostic：`artifacts/phase8h_tmjepa_v3_overdamped_closure_calibrated/`
- Phase-8I micro-closure：`artifacts/phase8i_tmjepa_v3_overdamped_microclosure/`
- G config：`configs/phase8g_tmjepa_v3_overdamped_longhorizon.yaml`
- H/I configs：`configs/phase8h_tmjepa_v3_overdamped_closure_calibrated.yaml`，`configs/phase8i_tmjepa_v3_overdamped_microclosure.yaml`
- Evaluator：`scripts/evaluate_tmjepa_v3.py`
- Paper evaluator readouts：`protein_jepa_frame/eval/paper_eval.py`
