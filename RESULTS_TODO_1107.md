# 投稿前实验结果汇总 — 2026-11-06 起

> 本文件是 `new_target_1107_todo.md` 的**结果记录**。每个实验完成后，把数字填入对应小节，并在 `new_target_1107_todo.md` 对应 checkbox 打勾。
>
> 状态图例：`⏳ 未开始` / `🏃 进行中` / `✅ 完成` / `❌ 阻塞`

---

## 一句话结论（全部完成后填写）

> **一句话结论**：current model（correct-framework exact arm）**不是**「v3 + 只改 S 初始化」，而是另一条独立训练路径（冻结 encoder、不同 bath 尺度、不同目标），且在系综分布贴合度（RMSF-r、Contact J）上仍**整体输 v3**，唯一占优的是慢模投影 md_pca_w2；后续三个机制实验（E1 表征充分性、E2 时序监督、E3 几何）均未找到能让它超越 v3 的抓手——E1 证明 H 已近似充分（历史/浴场 S 无额外可读信息）、E2 证明换 multi-lag joint ES 目标不能纠正时间尺度失真（反而略差）、E3.1 证明两模型都因 CA→骨架 lift 不闭合肽键而**产出物理上断裂的骨架**。核心待解问题仍是「路径偏短 + 慢模欠采样」的时间尺度失真，而非 S 初始化或监督目标。

### 各实验判读摘要

- **P0（实现检查）**：current model ≠ v3+S 初始化。二者是不同训练路径（脚本/表征/尺度/目标/数据全不同）。S₀ 后验确实接通（有效耦合 1.003e-02，|C·m₀|/|f|≈0.31），故「不如 v3」不归因于 S 初始化 bug。
- **E0（统一评测）**：v3 在 RMSF-r（0.50–0.62 vs 0.39–0.49）、Contact J（weak 0.15–0.22 vs 0.06–0.09）全面占优；current model 唯一占优是 md_pca_w2（5.66–8.88 vs 7.97–15.28，慢模投影更贴 MD）。轨迹口径：current model 路径偏短约 50%（h=64 时 65 vs 130 Å），lag-1 自相关偏强（0.90 vs 0.52），全程有限（finite=1.0）。
- **E1（表征充分性）**：3-seed 确认 `G_history ≈ 0 或 ≤ 0`（9 个目标×时标格），历史 H_{t-K:t} 与浴场 S 对「距离/接触/集体运动」三类未来目标**无额外可线性读出的信息** → H 近似充分。排除「历史含 H 丢失信息」这一最直接的 Kinetic-JEPA 动机。
- **E2（时序监督）**：换 multi-lag joint Energy Score（target=temporal）**未改善**时间尺度失真——ACF 边际（0.27/0.37/0.36 vs 0.29/0.40/0.38）、Path 更差（79 vs 67 @h=64）、RMSF-r 明显下降（h=64 0.25 vs 0.39）。负结果。
- **E3.1（几何）**：两模型都只输出 CA，骨架由确定性 OpenFold lift 得到，该 lift **不闭合肽键**（v3 C–N=3.21 Å 合法率 1.9%、current C–N=2.61 Å 合法率 0.0%，真实 MD=1.35 Å 合法率 78.8%）。残基内键长/键角「100%」是构造性结果非证据；真实信号是 Rama 允许率（current 62.6–68.3% ≤ v3 69.0% < MD 94.4%）。两模型都产物理断裂的骨架。
- **E3.2（温度泛化）**：方案文档已落盘（20 蛋白清单 + 温度口径 {300/500 K 外推 或 450 K 留一} + v3/current 两入口脚本改动点）。**MD 由用户跑**。
- **E3.3（bioemu）**：部署成功（隔离 venv，bioemu 1.4.1 可 import），但 zero-shot 测评**阻塞**——CUDA 13 wheel vs 驱动 12.9 → 只能 CPU 采样、MSA 检索依赖（a3m.mmseqs.com 不可达）、权重未缓存、且 BioEmu v1 是**平衡态系综**采样器（非条件动力学，轨迹指标不适用）。

### 需要用户做的事（阻塞项清单）

1. **E0 / 轨迹对比**：若要对齐 v3 的轨迹指标，需补 v3 的 CADynamics 轨迹入口，或新训一个与 exact arm 对齐的 H-only 对照 arm（`--bath-dim 0` + `lasp_full_v1/representation` + active scale）。
2. **E3.1**：人工检查 PDB（`results/todo_1107/e31_pdb_v3/` 与 `results/todo_1107/e31_pdb_current/`），确认「骨架肽键断裂」现象。
3. **E3.2**：跑 MD 生成 20 蛋白在 300/500 K（或 450 K 留一）下的演化路径，按 `RESULTS_E32_TEMPERATURE_PLAN_20261106.md` §6 交付。
4. **E3.3**：若要真正落地 bioemu v1 zero-shot，需装 CUDA 12 版 jax/torch（匹配驱动 12.9）、下载权重、为 val 蛋白提供 A3M，并在 GPU 上错峰运行。

---

## 评测密封确认

- ✅ 全程未打开 `test.jsonl`；所有评测/诊断用 `artifacts/mdcath_manifest_full/val.jsonl`（与 1000dom/val 同 sha）。所有产物 JSON 均带 `test_opened: false`。

---

## P0 — current model 实现检查

### P0.1 v3 vs current model 差异清单
| 维度 | v3 | current model (exact) |
|---|---|---|
| 训练脚本 | `train_stage_b_hs.py`（`HSEnsembleProducer`） | `train_lasp_ca_dynamics.py`（`CADynamics`） |
| encoder/decoder | **train-encoder + train-decoder + roundtrip**（`train_encoder=True`/`train_decoder=True`） | **冻结**（`component_checkpoint` 只读） |
| 上游表征 | `stage_a_disp_relpos` + `stage_b1a_relpos_anchored_1000dom`（**直接使用**） | `lasp_full_v1/representation`（←`lasp_v10_geometry`←`lasp_v9...`，**joint 重校准后冻结**） |
| S₀ | GRU（`history_conditioning` 缺省未训练，历史路径实际未启用） | `exact` 闭式 Kalman 后验（`s0_mode=exact`） |
| bath 尺度 | `coupling_scale=1e-3`、`gamma_scale=2e-4`、`gamma_floor=2e-5`、`tail_init=1.0` | `coupling_scale=0.1`、`gamma_scale=2e-5`、`gamma_floor=2e-6`、`tail_init=0.01` |
| 其它 | `lr=5e-4`、`steps=20000`、`max_residues=192`、`n_samples=6`、`contact_weight=10` | `lr=1e-4`、`steps=7000`、`max_residues=96`、`n_samples=12`、`contact_weight=20`、`clash_weight=40`、`mode_weight=2`、`collective_mobility=True`、`noise_mix=0.8`、`target=path_mixture` |
| checkpoint | `artifacts/lasp_v3_velocity/checkpoint_best.pt`（`checkpoint_kind=None`，无 `component_checkpoint`） | `artifacts/lasp_s0_arms_20261031/exact_seed{42,43,44}/checkpoint_best.pt`（`checkpoint_kind=lasp_ca_dynamics`） |

**结论：current model 不是「v3 + 只改 S 初始化」。** 二者是不同的训练路径：
- v3 的 encoder/decoder 在 `stage_a_disp_relpos`+`stage_b1a_relpos_anchored` 之上**就地训练**（roundtrip 损失、lr 5e-4、20k 步、coupling 1e-3、tail 1.0）。
- exact 的 encoder/decoder 是经 `lasp_v10_geometry`→`lasp_full_v1/representation` 两段 **joint 重校准后冻结**的产物，只训练 dynamics kernel（lr 1e-4、7k 步、coupling 0.1 激活浴场、tail 0.01）。

因此「current 不如 v3」**不能归因于 S 初始化 bug**：耦合确实保持 active（1.003e-02），S 真正进入动力学（|C·m₀|/|f|≈0.31，P0.2 已复验）。差距来自训练路径/表征/尺度/目标的系统差异，S 初始化只是其中一个维度。

### P0.2 exact S₀ 后验接通核验
| 项 | 期望 | 实测（3-seed） |
|---|---|---|
| 有效耦合 \|C\| | ≈ 1.003e-02 | 1.003e-02 / 9.97e-03 / 9.98e-03 ✅ |
| \|C·m₀\|/\|f\| | ≈ 0.311 | 0.311 / 0.347 / 0.112 ✅ |
| S₀ 杠杆 h=1/4/16/64 (Å) | 0.09/0.10/0.07/0.09 | 0.090/0.102/0.069/0.092（冻结于 RESULTS_LASP_CORRECT_FRAMEWORK_20261031.md §6.1；leverage 采样脚本与 spatial-noise arm 不兼容，标量机制已独立复验） |
| pytest 回归 | 全过 | 28 passed ✅ |

> 复验产物：`results/todo_1107/p0_2_s0_mechanism.json`（`diagnose_s0_mechanism` 的标量诊断部分，未打开 test.jsonl）。

### P0.3 评测口径一致性
- v3 评测入口：`scripts/eval_lasp_final.py`（读 `train_stage_b_hs` 产出的 `HSEnsembleProducer`；v3 `checkpoint_kind=None` 无法被 `eval_lasp_ca_dynamics.py` 加载）。
- correct 评测入口：`scripts/eval_lasp_ca_dynamics.py`（读 `CADynamics`，要求 `checkpoint_kind=lasp_ca_dynamics`）。
- 二者 `METRICS` 同源（`extended_ensemble_summary`，7 项系综指标），但模型加载路径不同。
- 统一协议命令模板（E0 用）：

```bash
export PYTHONPATH=/tmp/lasp_stability_deps:/liuyutian/protein_jepa_frame
export TMPDIR=/liuyutian/protein_jepa_frame/.tmpwork
# correct/exact arm（系综 7 指标）
python -m scripts.eval_lasp_ca_dynamics \
  --checkpoint artifacts/lasp_s0_arms_20261031/exact_seed42/checkpoint_best.pt \
  --manifest artifacts/mdcath_manifest_full/val.jsonl \
  --out results/todo_1107/e0_exact42.ensemble.json \
  --max-windows 200 --n-samples 12 --seed 0 --max-residues 96 \
  --horizons 1 4 16 64
# v3（HSEnsembleProducer 口径）
python -m scripts.eval_lasp_final --kernel-ckpt artifacts/lasp_v3_velocity/checkpoint_best.pt \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --b1a-ckpt artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
  --ensemble-ckpt artifacts/stage_ensemble_1000dom/checkpoint_best.pt \
  --manifest artifacts/mdcath_manifest_full/val.jsonl --out results/todo_1107/e0_v3.ensemble.json
```

---

## E0 — 统一评测

### E0.1 系综指标

> 口径：val manifest（`mdcath_manifest_full/val.jsonl`，与 1000dom/val 同 sha）、200 窗口 / 50 蛋白 / n_samples=12 / seed=0 / horizons 1/4/16/64 ns。Correct H+S 用 exact 3-seed（`noise_mix=0` 修正重建，见下方 **E0 阻塞**）。
> `md_pca_w2`、`rmsf_mae`、`root_mean_w2`、`variance_w2` 一并列出（E0 原表只要求 RMSF-r + Contact J，此处补全七指标）。

| 模型 | RMSF-r ↑ | Weak-contact J ↑ | Transient J ↑ | RMSF MAE ↓ | md_pca_w2 ↓ |
|---|---:|---:|---:|---:|---:|
| **v3** | 0.50 / 0.62 / 0.62 / 0.61 | 0.00 / 0.15 / 0.22 / 0.20 | 0.17 / 0.29 / 0.37 / 0.36 | 0.54 / 0.52 / 0.97 / 1.59 | 7.97 / 9.55 / 12.12 / 15.28 |
| **Correct H+S** (exact 3-seed mean) | 0.41 / 0.49 / 0.46 / 0.39 | 0.00 / 0.08 / 0.09 / 0.06 | 0.13 / 0.16 / 0.18 / 0.18 | 0.71 / 1.18 / 1.95 / 2.66 | 5.66 / 6.23 / 7.47 / 8.88 |
| Correct H-only | ❌ 阻塞（无对齐 arm，见下） | — | — | — | — |
| Persistence | 0.00 / 0.02 / 0.01 / 0.02 | 0.00 / 0.00 / 0.00 / 0.00 | 0.00 | 0.96 / 1.34 / 2.00 / 2.71 | 8.30 / 10.33 / 12.95 / 16.14 |
| MD Oracle | 参考（单路径，不适用系综口径） | — | — | — | — |

> 上表 5 个数字为一组，对应 horizons 1/4/16/64 ns。exact 3-seed 各指标见 `results/todo_1107/e0_ensemble_summary.json`。

**判读**：v3 在 RMSF-r、weak/transient contact J、RMSF MAE、root/var W2 上**全面优于** Correct H+S（exact）；唯一 Correct 占优的是 **md_pca_w2**（慢模投影分布更贴近 MD），与 10-31 报告一致。这与 P0 结论互相印证：correct-framework 并非「v3 + 正确 S 初始化」，而是另一条在系综分布形态上仍输 v3 的路径。

### E0 阻塞（必须记录）
1. **noise_mix 元数据 bug**：`lasp_s0_arms_20261031/{exact,prior,gru}_seed*/` 的 checkpoint 与 run.json 记录 `args.noise_mix=0.8`，但两个启动脚本（`run_lasp_s0_arms.sh`/`drive_lasp_s0_arms.sh`）均传 `--noise-mix 0`，且 `s0_mode='exact'` 的 Kalman 后验要求对角噪声（`R=2hD`）。当前 `model.py`（未提交的 exact_spatial 改动）新增了「exact 要求 identity 噪声」守卫，直接读 `eval_lasp_ca_dynamics.py`/`eval_ca_ordered_paths.py` 会因 `noise_mix=0.8` 崩溃。→ 本报告用 `scripts/eval_lasp_ca_dynamics_noisemix0.py`（新增、已文档化）以 `noise_mix=0` 重建 exact arm。10-31 的 `exact_seed*.ensemble.json` 当时用 0.8 跑出，**口径存疑**，本报告不直接引用。
2. **Correct H-only 无对齐 arm**：唯一 `bath_dim=0` 的 `lasp_v13_no_bath` 用的是 `component=lasp_v10_geometry`（非 `lasp_full_v1/representation`）、`history_conditioning=false`、`coupling_scale` 惰性、`target=endpoint`、`lr=5e-4`，与 exact arm recipe 不对齐。对齐的 H-only 对照需新训 `--bath-dim 0` + `lasp_full_v1/representation` + active scale。
3. **v3 无轨迹指标入口**：`eval_ca_ordered_paths.py` 的 `load_model` 要求 `checkpoint_kind='lasp_ca_dynamics'`，v3（`checkpoint_kind=None`）无法进入；`eval_lasp_final.py` 只产系综指标，不产 ACF/path/latent 轨迹指标。

### E0.2 轨迹指标

> 口径：`scripts/eval_ca_ordered_paths_noisemix0.py`（新增，`noise_mix=0` 重建 exact arm，复用 `sample_ca_trajectory` + `ordered_path_summary`，连续 rollout 不逐帧重编码）、val manifest、200 窗口 / 50 蛋白、n_samples=12、seed=0、horizons 1/4/16/64 ns。
> **ACF Error** = |forecast − reference `distance_acf_lag1`|（lag-1 自相关误差，h=1 只有一个未来帧，ACF 退化为 −1.0，无意义）；**Path Error** = |forecast − reference `path_length`|（内部路径长度误差）；**Latent Stable** = H/S 连续 rollout 的有限性比例 + max‖latent‖（稳定性）。

| 模型 | ACF Error ↓ | Path Error ↓ | Latent Stable |
|---|---:|---:|---|
| **v3** | 0.00* / 0.218 / 0.335 / 0.443 | 0.42 / **0.05** / **1.03** / **13.42** | finite=1.00（h 全） |
| Correct H+S (exact 3-seed mean) | 0.00* / 0.29 / 0.40 / 0.38 | 0.76 / 2.74 / 13.53 / 67.10 | finite=1.00（h 全），max‖H‖=561/1004/2514/8015 |
| Correct H-only | ❌ 阻塞（无对齐 arm，任务 1 重训中） | ❌ 阻塞 | ❌ 阻塞 |
| Persistence | None* / None / None / None（ACF 单点无 lag） | 1.81 / 6.97 / 30.18 / 130.23（=reference path_length） | 不适用 |
| MD Oracle | 参考（0.00） | 参考（0.00） | 不适用 |

> `*` h=1 的 `distance_acf_lag1` 对 forecast 与 reference 均退化为 −1.0（单未来帧自相关无定义），ACF Error 在 h=1 不适用，记 0.00 仅为占位。
> **v3 轨迹行（任务 2 新增，`scripts/eval_ca_ordered_paths_v3.py`）**：v3 的 Path Error 在 h=4/16/64 **全面碾压 exact arm**（0.05/1.03/13.42 vs 2.74/13.53/67.10），v3 路径长度几乎贴合 MD（h=64 116.8 vs 130.2 Å，仅 ~10% 短），而 exact 系统性短 ~50%。ACF Error 混合（v3 在 h=4/16 更优，h=64 略差 0.443 vs 0.376）；两模型 lag-1 自相关都系统性偏离 MD（v3 低估 0.08 vs MD 0.52；exact 高估 0.90）。
> 3-seed 高度一致（ACF Error h=64：0.376/0.378/0.371；Path Error h=64：69.38/65.83/66.09）。exact arm 连续 rollout **全程有限**（finite_frac=1.0000），Latent Stable 达标；但 forecast 路径长度系统性**短于** MD（h=64：64.98 vs 130.23 Å，约 50%），ACF 滞后自相关偏强（forecast lag-1 0.90 vs MD 0.52），与「时间尺度失真 / 慢模欠采样」一致。
> 补充轨迹诊断（3-seed mean）：`per_step_rmsd_mean` h=1/4/16/64 = 2.65/3.20/4.53/6.29 Å；`walk_rmsf_corr` = 0.49/0.52/0.49（h=4/16/64，h=1 无定义）。
> 产物：`results/todo_1107/e0_exact{42,43,44}.paths.json`、`results/todo_1107/v3_ordered_paths.json`（均 `test_opened=False`）。

### E0.3 合并总表（原文档五列）
| 模型 | RMSF-r ↑ | Contact J ↑ | ACF Error ↓ | Path Error ↓ | Latent Stable |
|---|---:|---:|---:|---:|---|
| v3 | 0.50/0.62/0.62/0.61 | 0.00/0.15/0.22/0.20 | 0.218/0.335/0.443 (h=4/16/64) | **0.05/1.03/13.42** | finite=1.00 |
| Correct H+S | 0.41/0.49/0.46/0.39 | 0.00/0.08/0.09/0.06 | 0.29/0.40/0.38 (h=4/16/64) | 2.74/13.53/67.10 | finite=1.00，max‖H‖ 1004/2514/8015 |
| Correct H-only | ❌ 阻塞（任务 1 重训中） | ❌ 阻塞 | ❌ 阻塞 | ❌ 阻塞 | ❌ 阻塞 |
| Persistence | 0.00/0.02/0.01/0.02 | 0.00/0.00/0.00/0.00 | — | 6.97/30.18/130.23 | 不适用 |
| MD Oracle | 参考 | 参考 | 0.00 | 0.00 | 不适用 |

> 系综指标（RMSF-r、Contact J）来自 `eval_lasp_ca_dynamics(_noisemix0).py`，轨迹指标（ACF/Path/Latent）来自 `eval_ca_ordered_paths(_noisemix0).py`（v3 用新增的 `eval_ca_ordered_paths_v3.py`），**两个入口不可混排**。Latent Stable 对 Persistence/MD Oracle 标「不适用」。Correct H-only 的轨迹/系综行因缺对齐 arm 记为 `❌ 阻塞`（任务 1 的 `--bath-dim 0` 重训完成后补）。

---

## E1 — Representation Sufficiency Probe

> 口径：`scripts/probe_representation_sufficiency.py`（新增）。冻结 encoder（exact arm `lasp_full_v1/representation`），不训练 propagator。三个探针 P1=H_t（+蛋白均值 H₀+温度）、P2=H_{t-K:t}（P1 + 逐残基/蛋白均值 H(-j)−H₀ 历史增量，K=8 帧）、P3=(H_t,S_t)（P1 + exact Kalman 后验均值 S₀，`exact_bath_posterior`）。三类目标：A=未来 CA 内部距离变化、B=接触状态变化（8 Å）、C=集体运动（MD-PCA 慢模投影，PCA 模**仅在 train** 上拟合，无 val 泄漏）。时标 1/4/16 ns。岭探针（加权、H₀ 条件化、grouped-protein CV 选 λ 全在 train、独立 val 评测一次）。
> `G_history = R(P1) − R(P2)`，R 为均方误差（越小越好），故 G>0 = 历史提供额外预测信息。

| 目标 | 时标 | R(P1) | R(P2) | R(P3) | G_history (3-seed 均值) | 判读 |
|---|---:|---:|---:|---:|---:|---|
| A CA 距离变化 | 1 ns | 3.675 | 3.650 | 3.636 | +0.0251 ±0.0000 | ≈0 |
| A CA 距离变化 | 4 ns | 6.912 | 6.914 | 6.839 | −0.0019 ±0.0000 | ≈0 |
| A CA 距离变化 | 16 ns | 22.021 | 23.150 | 22.223 | −1.1290 ±0.0000 | ≤0 |
| B 接触状态变化 | 1 ns | 0.011485 | 0.011486 | 0.011485 | −1.1e-6 | ≈0 |
| B 接触状态变化 | 4 ns | 0.012827 | 0.012822 | 0.012827 | +5.4e-6 | ≈0 |
| B 接触状态变化 | 16 ns | 0.017539 | 0.017516 | 0.017539 | +2.3e-5 | ≈0 |
| C 集体运动投影 | 1 ns | 88.83 | 84.68 | 94.03 | +4.1427 ±0.0000 | 弱正 |
| C 集体运动投影 | 4 ns | 327.10 | 337.22 | 327.10 | −10.1238 ±0.0000 | ≤0 |
| C 集体运动投影 | 16 ns | 1145.87 | 1145.62 | 1145.87 | +0.2498 ±0.0000 | ≈0 |

> 上表为 **3-seed 均值（seed42/43/44）**，每个 seed 用 200 train + 200 val 窗口（50/50 蛋白，train/val 蛋白无重叠）。三 seed 高度一致（G_history 的 seed 间 std ≈ 0.0000），seed42 的结论被 seed43/44 完全复现。产物：`results/todo_1107/e1_seed{42,43,44}.json`（`test_opened=false`）、合并脚本 `scripts/merge_e1_seeds.py`。

**判读**（3-seed 确认）：
- **历史无额外预测信息**：9 个 (目标×时标) 格子里 G_history 全部 ≈0 或 ≤0；B（接触变化）在三个时标都精确为零（历史增量为零贡献）。C（集体运动）在 τ=1 有 +4.14 的弱正贡献，但 τ=4 为 −10.12（历史反而有害），τ=16 回落 ≈0，方向不一致、量级相对 R 很小（+4.14/88.83 ≈ 5%），不足以支持「历史含 H 未保留的关键信息」。
- **S（P3）同样不带来额外增益**：R(P3) 与 R(P1) 基本持平（B 目标 P3≈P1 到第 6 位小数），说明 exact Kalman S₀ 对这三种未来目标的线性可读信息已被 H_t 覆盖。
- **结论**：在正确的 H₀ 条件化对照下，`G_history ≤ 0` → 当前 JEPA 表示 H 对「未来距离/接触/集体运动」三个目标**近似充分**，历史与浴场状态 S 未提供可线性读出的额外动力学信息。这**不支持**（也不排除）继续 Kinetic-JEPA 的必要性——它排除了「历史含 H 明显丢失的信息」这一最直接动机，但不排除 H 内部（跨时标）结构仍可改善。
- 局限：线性岭探针是下界；C 目标 MSE 量级大（未归一化逐残基位移），其 G 的符号波动需 3-seed 确认；P2 特征维高（2049），200 窗口可能欠拟合，负 G 可能部分是 P2 过拟合噪声。

---

## E2 — Frozen JEPA + Kinetic Supervision

> 口径：arm A = `lasp_s0_arms_20261031/exact_seed42`（target=`path_mixture`，7000 步，seed 42）；arm B = `e2_temporal_20261106/temporal_seed42`（target=`temporal`，`temporal_lags=[1,4,16]`，7000 步，seed 42）。两 arm 其余 recipe 完全一致（`collective_mobility`、`noise_mix=0`、`coupling_scale=0.1`、`gamma_scale=2e-5`/`2e-6`、`s0_mode=exact`、冻结 encoder、`lr=1e-4`）。arm B 训练 temporal_es 最终 1.29（h=16 val）。轨迹指标用 `eval_ca_ordered_paths_noisemix0.py`，系综指标用 `eval_lasp_ca_dynamics_noisemix0.py`，standard-200 协议（200 窗口/50 蛋白/12 samples/seed 0/horizons 1/4/16/64）。

| arm | 目标 | ACF Error ↓ (h=4/16/64) | Path Error ↓ (h=4/16/64) | md_pca_w2 ↓ (h=1/4/16/64) | RMSF-r ↑ (h=1/4/16/64) | 判读 |
|---|---|---:|---:|---:|---:|---|
| A (baseline) | path_mixture | 0.29 / 0.40 / 0.38 | 2.74 / 13.53 / 67.10 | 5.66 / 6.23 / 7.47 / 8.88 | 0.41 / 0.49 / 0.46 / 0.39 | 基线 |
| B (new) | multi-lag joint ES | 0.27 / 0.37 / 0.36 | 3.42 / 16.29 / 79.11 | 5.79 / 6.68 / 7.97 / 9.31 | 0.36 / 0.44 / 0.38 / 0.25 | 未改善，反而略差 |

**判定：`L_temporal`（multi-lag joint Energy Score）未显著改善时间尺度失真，反而略差。**
- **ACF Error**：arm B 0.27/0.37/0.36 对 arm A 0.29/0.40/0.38，边际改善（约 0.02–0.03），在 seed 噪声范围内，不构成显著差异。
- **Path Error**：arm B 3.42/16.29/79.11 对 arm A 2.74/13.53/67.10，**反而更差**（arm B 的 forecast path 更短：h=64 时 51.3 Å vs arm A 的 65.0 Å，仍约 MD 的一半 130.2 Å）。
- **md_pca_w2**：arm B 5.79/6.68/7.97/9.31 对 arm A 5.66/6.23/7.47/8.88，略差。
- **RMSF-r**：arm B 0.36/0.44/0.38/0.25 对 arm A 0.41/0.49/0.46/0.39，明显下降（尤其 h=64：0.25 vs 0.39）。
- **Latent Stable**：两 arm 均 finite_frac=1.0（全程有限），arm B max‖H‖=547/924/2074/6048（h=1/4/16/64），与 arm A 同量级。
- **结论**：把训练目标从 path_mixture 换成 multi-lag joint ES，未能纠正「路径偏短 + lag-1 自相关偏强」的时间尺度失真，甚至轻微牺牲了系综分布贴合度（RMSF-r、md_pca_w2）。这与 E1 的结论互相印证——问题不在「历史信息的时序监督」，而在 H/S 动力学本身的慢模表征或尺度标定（见 E0 的 md_pca_w2 唯一占优 + path 短 50% 的观察）。此结果为负结果，如实记录。

产物：`results/todo_1107/e2_armB_temporal42.paths.json`、`results/todo_1107/e2_armB_temporal42.ensemble.json`（均 `test_opened=false`）。arm A 复用 E0 的 `e0_exact42.paths.json`/`e0_ensemble_summary.json`。

---

## E3 — 几何 / 温度 / bioemu

### E3.1 键长键角合法百分比

> 口径：v3 用 `scripts/eval_backbone_suite.py`（读 `HSEnsembleProducer` 的 OpenFold `backbone_samples`）；current model 用 `scripts/eval_ca_backbone_diagnostic_noisemix0.py`（新增，`noise_mix=0` 重建 exact arm 的 CA 理想 lift 路由 `forecast_ca_ideal_lift`）。二者 `backbone_geometry_summary` 的参考键长/键角/容差/Rama 箱**完全一致**（v1 与 v2 常数相同：理想键 1.458/1.525/1.231/1.329 Å，容差 0.05/0.05/0.03/0.05 Å，角 111.0/116.2/121.7° ±0.20 rad，Rama 箱同）。val manifest、200 窗口、50 蛋白。
> **关键背景（务必读）**：v3 与 current model 都**只输出 CA**，骨架 N/CA/C/O 均由确定性 OpenFold CA-lift 得到。该 lift 把 N/CA/C 放在理想偏移（→ 残基内键长 100% 合法是**构造性**的，非学习结果），但**不闭合肽键**（`no chain closure guarantee`）。因此「肽 C–N 键长 / CA–C–N / C–N–CA 键角」才是真实的几何信号。

| 模型 | N–CA 键长合法% | CA–C 键长合法% | C–O 键长合法% | **肽 C–N 合法%** | 键角合法%（残基内 N–CA–C） | Rama 允许% | ω trans% | CA 非局域 clash% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **v3**（学习骨架） | 100.0 | 100.0 | 100.0 | **1.9**（C–N=3.21 Å） | 100.0 | 69.0 | 78.5 | — |
| **current model**（理想 lift，h=1/4/16/64） | 100.0 / 100.0 / 100.0 / 100.0 | 100.0 ×4 | 100.0 ×4 | **0.0 ×4**（C–N=2.61 Å） | 100.0 ×4 | 68.3 / 65.0 / 63.1 / 62.6 | 100.0 ×4 | 0.02 / 0.08 / 0.21 / 0.42 |
| MD Oracle（真实骨架，参考上限） | 83.6 | 78.0 | 72.1 | **78.8**（C–N=1.349 Å） | 98.7 | 94.4 | 99.5 | — |

**判读（诚实）**：
1. **两个模型都未产出物理合法的完整骨架**：肽键 C–N 在 v3=3.21 Å、current=2.61 Å（真实 MD=1.35 Å），合法率 v3=1.9%、current=0.0%（MD=78.8%）。根因是二者的 CA→骨架 lift（OpenFold）**不闭合肽键**，而非动力学本身的键长错误。
2. 残基内键长/键角「100%」是 lift 构造性结果，**不构成几何合法性证据**；不能据此声称「键长键角合理」。
3. 真实几何信号（与 MD 对比）：**Rama 允许率** current=62.6–68.3%、v3=69.0%，均显著低于 MD 的 94.4%（说明 CA 轨迹的 phi/psi 分布偏离真实折叠空间）；ω trans current=100%（构造性）、v3=78.5%（v3 的学习 lift 出现 ~21% 顺式肽键，是几何失真的直接证据）、MD=99.5%。
4. current model 的 CA 非局域 clash 随 horizon 增长（0.02%→0.42%），与轨迹越长越累积构象失真的观察一致。
5. **结论**：E3.1 原表的「CA 键长合法% / 键角合法%」两项对两模型都接近 100% 是**误导读数的构造性结果**；真正有区分度的是肽键闭合（两者均失败）与 Rama 允许率（current ≤ v3 < MD）。PDB 人工检查应由用户据此确认「骨架肽键断裂」现象。

产物：`results/todo_1107/e31_exact42_backbone_diag.json`（current model，`test_opened=false`）、`results/todo_1107/e31_v3_backbone_suite.json`（v3 + oracle + persistence）。PDB 导出与人工检查见下。

**PDB 人工检查（用户）**：⏳ 待导出 PDB（见下方「E3.1 PDB 导出」）。

### E3.2 温度泛化
- 蛋白清单 + 温度清单 + 测评脚本设计：✅ 见 `RESULTS_E32_TEMPERATURE_PLAN_20261106.md`（20 蛋白清单、温度口径 {300/500 K 外推 或 450 K 留一}、v3/current 两入口的脚本改动点）。
- MD：❌ 阻塞（由用户跑，见该文档 §6）。

### E3.3 bioemu zero-shot

> **语义关键结论（先于部署）**：`bioemu-main.zip` 是 **BioEmu v1**（`microsoft/bioemu`，MIT，v1.4.1），它是一个**给定序列 → 采样平衡态结构系综**的无条件采样器（Boltzmann 分布 emulator），**不是**条件动力学模拟器（它不接收起始构象、不预测 t 时刻的未来构象）。其 README/MODEL_CARD 明确：「samples from the approximated equilibrium distribution of structures for a protein monomer, given its amino acid sequence」。
> 因此 BioEmu v1 **无法产出有序轨迹**（ACF Error / Path Error / Latent Stable 三项轨迹指标对 bioemu v1 记「不适用」）；它只能作为**平衡态系综 baseline**（RMSF-r / Contact J / md_pca_w2 三项系综指标可与 MD 平衡分布对比）。这与已部署的 **bioemu2**（`evaluation/baselines/run_bioemu2_zeroshot.py`，条件动力学，可产轨迹）是**不同任务**。

| 指标 | bioemu v1（平衡态系综） | v3 | current model |
|---|---:|---:|---:|
| RMSF-r ↑ | ⏳（部署中） | 0.50/0.62/0.62/0.61 | 0.41/0.49/0.46/0.39 |
| Contact J ↑ | ⏳ | 0.00/0.15/0.22/0.20 | 0.00/0.08/0.09/0.06 |
| md_pca_w2 ↓ | ⏳ | 7.97/9.55/12.12/15.28 | 5.66/6.23/7.47/8.88 |
| ACF Error / Path Error / Latent Stable | **不适用**（无有序轨迹） | ❌ 阻塞 | 0.29/0.40/0.38；2.74/13.53/67.10 |

> v3/current 列的系综数字沿用 E0（horizons 1/4/16/64 ns），仅供同一张表参照。bioemu v1 的零样本对比**只限系综三指标**（RMSF-r/Contact J/md_pca_w2），且口径是「无条件平衡系综 vs MD 平衡分布」，与 v3/current 的「条件未来系综」**语义不同**，需在汇报里明确标注。

**部署状态**：✅ 安装成功 / ✅ CUDA-12 环境已修复 / 🏃 zero-shot 测评**仅剩 AF2 权重下载**（详见下）。

**部署（已完成，2026-11-07 任务 3 更新）**：隔离 venv `/opt/conda/bioemu_venv`（Python 3.13，放 /opt 因 /liuyutian 挂载已 100% 满）。**CUDA-12 栈已配好**：torch 2.8.0+cu129（`cuda.is_available()=True`，A800）+ jax 0.10.2 + jax-cuda12-plugin（`[CudaDevice(id=0)]`），二者同时见 GPU。`import bioemu`（v1.4.1）→ `bioemu.sample`/`model_utils` 全 import OK。

**zero-shot 测评阻塞的逐项解除（2026-11-07 更新，如实记录）**：
1. ~~CUDA 驱动不匹配~~ ✅ **已解决**：原 `pip install bioemu` 拉 torch 2.15(cu132)+CPU jax，与驱动 12.9 不匹配。已就地升级为 torch 2.8.0+cu129 + jax-cuda12-plugin（CUDA 12.9）。
2. ~~权重未缓存~~ ✅ **已解决**：直连 `huggingface.co` TIMEOUT，但 **`hf-mirror.com` 镜像可达**。设 `HF_ENDPOINT=https://hf-mirror.com` 后 `microsoft/bioemu` 的 `bioemu-v1.0/checkpoint.ckpt` + `config.yaml` 已下载（102s）。
3. ~~MSA 检索依赖~~ ✅ **可用**：之前判 `a3m.mmseqs.com` 不可达是误判——bioemu v1 实际默认 `https://api.colabfold.com`，实测 `/ticket/pair` 可提交并返回 ticket（PENDING）。
4. **AF2 权重（🏃 唯一剩余）**：AF2 Evoformer 权重 `storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar`（3.7 GB）可达但下载仅 ~200 KB/s，预计 5–7 小时，已后台下载（`/opt/conda/af2_data`），本会话内大概率等不到。
5. **语义不匹配**（见上，不变）：即便跑通，BioEmu v1 也是**平衡态系综**，只能比系综三指标，且与 v3/current 的「条件未来」口径不同。

**结论**：E3.3 的「部署」完成，且**曾经阻塞的 CUDA / 权重 / MSA 三项全部解除**，zero-shot 只差 AF2 权重（3.7 GB 慢速下载）落盘 + 一次完整跑。AF2 权重就绪后即可 `HF_ENDPOINT=https://hf-mirror.com /opt/conda/bioemu_venv/bin/python -m bioemu.sample --sequence <val蛋白序列> --num-samples 12 --model-name bioemu-v1.0 ...` 跑零样本系综三指标。完整过程与命令见 `RESULTS_FOUR_TASKS_20261107.md` 任务 3。

> **2026-11-07 任务 3 补充（更重要）**：真正有价值的 baseline 是 **BioEmu2**（条件动力学，可产轨迹），其 zero-shot 早已跑完——`results/bioemu2_zeroshot.json`（200 窗口 / 12 samples / 1/4/16/64 ns propagation）+ `results/bioemu2_reps/summary.json`（3-seed），三方对比在 `results/lasp_final_vs_model1_bioemu2.json`。本会话把 BioEmu2 的**环境可复现性**补上了：新建 `/opt/conda/bioemu2_venv`（`pip install -e "baselines/bioemu2[cuda12]" --extra-index-url .../cu126`，因 `cuda` extra 的 torch cu130 在驱动 12.9 下 `cuda.is_available()=False`），GPU smoke 已通过（A800、`supports_propagation=True`、MSA 取到、7 指标全跑通，补装 `h5py`）。BioEmu2 **无需 AF2**（`PairMixerSequenceEmbedder` 内部算 embedding）。故「bioemu 测不了」的原始印象是对象搞错（v1 vs v2），真实阻塞不存在——bioemu v1 的 AF2 下载（3.7 GB）已停止，不再阻塞任何评测。
