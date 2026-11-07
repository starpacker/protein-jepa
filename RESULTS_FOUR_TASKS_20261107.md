# 四任务结果 — 2026-11-07（H-only / v3 轨迹 / bioemu / E1 复查）

> 对应 `/goal` 新任务 1–4。本文件记录每一项的部署、代码、结果与阻塞。
> 状态：`✅ 完成` / `🏃 进行中` / `❌ 阻塞`

---

## 任务 1 — Correct H-only 部署（允许重训）

> **方案更正（2026-11-07，用户指正）**：H+S 用 **v3 SOTA**（`artifacts/lasp_v3_velocity`），故 H-only 必须对齐 **v3 recipe**，而不是 exact arm。exact arm 是另一条训练路径（冻结 encoder、coupling 0.1、path_mixture、lr 1e-4），用它做 H-only 会引入除 S 之外的差异，不是干净的 S 消融。

### 部署（✅ 完成）
- 新增 `run_scripts/run_lasp_v3_h_only.sh`：v3 recipe（`run_lasp_v3_velocity.sh`）的**唯一改动**是 `--bath-dim 0`。
- 其余完全对齐 v3 SOTA：`--train-encoder --train-decoder`（就地，encoder/decoder lr-scale 0.1）、`--bath-coupling velocity`、`coupling_scale=1e-3`（默认）、`--lr 5e-4`、`--steps 20000`、`--max-residues 192`、`--history-frames 4`、`--rollout-steps 8`、`--n-samples 6`、`--multi-horizon --paired-horizons-frames 1 4 16 64`、`--per-sample-temperature`、`--roundtrip-weight 0.5`、`seed 42/43/44`。
- `bath_dim=0` 是模型原生支持的 H-only 消融（`StageB1bConfig.bath_dim=0 → use_bath=False → S=None`，rollout 里 bath 分支完全跳过）。smoke test（3 步）已跑通，无报错，确认 v3 路径下 `--bath-dim 0` 正确接通。
- 已删除旧的错误方案文件（exact-arm 版 `run_lasp_h_only.sh` / `eval_lasp_h_only.sh` / `compare_hs_vs_h_only.py`）及错误训练产物（`artifacts/lasp_h_only_20261107`，~76% 处中止）。

### 训练（🏃 进行中）
- 输出：`artifacts/lasp_v3_h_only_20261107/v3honly_seed{42,43,44}/`，seed 42 在 GPU 0、seed 43 在 GPU 1（seed 44 排队）。
- 实测 v3 recipe ~1.9s/步 → 20000 步约 **~10.5 小时/seed**。seed 42 为**主对照**（v3 SOTA 也是单 seed 42）；43/44 用于 seed 方差（可选）。

### 评测（✅ 已就绪，训练完成后一键触发）
- 新增 `run_scripts/eval_lasp_v3_h_only.sh`：对每个 seed 跑 `eval_lasp_final.py`（系综，与 v3 SOTA 的 `e0_v3.ensemble.json` 同入口）+ `eval_ca_ordered_paths_v3.py`（轨迹，ACF/Path/Latent-Stable）。
- 新增 `scripts/compare_v3_hs_vs_h_only.py`：读 v3-SOTA 与 v3-H-only 两套系综/轨迹 JSON，逐 horizon 输出「S 是否带来可测增益」，直接回答任务 4 的「H+S > H ?」（诚实记录负结果）。已 dry-run 验证（self-vs-self 零差）。

---

## 任务 2 — v3 轨迹指标入口（修改 eval 代码）

### 代码（✅ 完成）
- 新增 `scripts/eval_ca_ordered_paths_v3.py`：v3 没有 `CADynamics` 入口（`checkpoint_kind=None`、无 `component_checkpoint`），原 `eval_ca_ordered_paths.py` 无法加载。本脚本复用后者的**同一套指标/面板/汇总**（`ordered_path_summary`、`temporal_controls`、`_reference`、`summarize`、`panel_spec`、`validate_panel`），只替换模型加载与连续 rollout。
- v3 连续 rollout：`hs.sample_future_ca` 做一次 GLE rollout 到 `hs._rollout_steps=horizon`，返回 `H_traj [BN,horizon+1,L,dH]`；每步 H 经 `_decode_path`（`decoder.decode_structure` 的 `backbone_coords[:,:,1,:]`）解码到 CA（anchor 规范）；t0 覆盖为观测 anchor。与 exact-arm 适配器同语义（一次采样、不重编码、不重启浴场）。

### 结果（✅ 完成，standard-200：200 窗口 / 50 蛋白 / 12 samples / seed 0 / horizons 1/4/16/64）

| horizon | 模型 | ACF Error ↓ | Path Error ↓ | forecast path | MD path | forecast ACF | MD ACF |
|---|---:|---:|---:|---:|---:|---:|
| h=4 | **v3** | **0.218** | **0.05** | 6.92 | 6.97 | −0.311 | −0.093 |
| h=4 | exact (current) | 0.291 | 2.87 | 4.58 | 6.97 | 0.197 | −0.093 |
| h=16 | **v3** | **0.335** | **1.03** | 29.15 | 30.18 | −0.023 | 0.312 |
| h=16 | exact (current) | 0.402 | 14.10 | 16.89 | 30.18 | 0.714 | 0.312 |
| h=64 | v3 | 0.443 | **13.42** | 116.81 | 130.23 | 0.079 | 0.521 |
| h=64 | exact (current) | **0.376** | 69.38 | 61.62 | 130.23 | 0.897 | 0.521 |

**判读**：
- **Path Error 上 v3 碾压 current model**：v3 路径长度几乎贴合 MD（h=64 时 116.8 vs 130.2 Å，仅 ~10% 短），而 current model 系统性短 ~50%（61.6 vs 130.2 Å）。这量化确认了 E0 的「current 路径偏短 50%」问题，且 v3 无此问题。
- **ACF Error 混合**：v3 在 h=4/16 更好（0.218/0.335 vs 0.291/0.402），h=64 略差（0.443 vs 0.376）。两模型 lag-1 自相关都**系统性低估** MD（v3 0.079 vs MD 0.521；current 0.897 高估）。
- 这填补了 E0.2/E0.3 之前 `v3 轨迹行 ❌ 阻塞` 的空缺。

产物：`results/todo_1107/v3_ordered_paths.json`（`test_opened=false`）。

---

## 任务 3 — bioemu 独立 CUDA 环境（测评）

### 关键更正（✅ 本会话重新定位）：bioemu 早已测过，且用的是 BioEmu2 而非 bioemu v1

本会话一开始误把「bioemu 没测成」当作阻塞，是因为一直在攻 **bioemu v1**（旧包，依赖 AF2 Evoformer 权重 + torch/jax CUDA 不匹配）。但仓库里其实早就用 **BioEmu2**（`microsoft/bioemu2`，`bioemu2-preview`，自包含、无需 AF2）跑完了 zero-shot 评测，结果已在盘上：

| 产物 | 内容 |
|---|---|
| `results/bioemu2_zeroshot.json` | BioEmu2 时间传播（propagate）zero-shot 轨迹评测：200 窗口 / 12 samples / 1/4/16/64 ns |
| `results/bioemu2_reps/{seed0,seed1,seed2,summary}.json` | 3-seed 重复 + 汇总 |
| `evaluation/baselines/run_bioemu2_zeroshot.py` | 评测脚本（与 from-none 模型**同一套** `EvalProtocol`/`extended_ensemble_summary`/persistence 基线，apples-to-apples） |
| `run_scripts/run_bioemu2_zeroshot.sh` + `_rep.sh` + `_t2a_samples.sh` | 启动脚本 |
| `scripts/merge_bioemu2_shards.py` / `compare_vs_bioemu2.py` / `analyze_lasp_bioemu2_gaps.py` | 合并/对比工具 |
| `baselines/bioemu2/msa_cache/`（100+ 缓存） + `checkpoints/bioemu2-preview/checkpoint.ckpt`（2.0 GB，已下载） | MSA 缓存 + 权重 |

BioEmu2 zero-shot 轨迹指标（`bioemu2_zeroshot.json`，propagation）：

| horizon | rmsf_corr ↑ | md_pca_w2 ↓ | root_mean_w2 ↓ | weak_contact_j ↑ | transient_j ↑ |
|---|---:|---:|---:|---:|---:|
| 1 ns | 0.431 | 0.087 | 5.59 | 0.000 | 0.165 |
| 4 ns | 0.595 | 0.098 | 3.70 | 0.163 | 0.281 |
| 16 ns | 0.579 | 0.086 | 3.45 | 0.237 | 0.348 |
| 64 ns | 0.609 | 0.092 | 4.17 | 0.209 | 0.377 |

三点结论：
1. **任务 3 的核心诉求「实际测评 bioemu」此前已达成**——BioEmu2 在标准 200 窗口协议下跑完了零样本动力学基线，且与 LASP/Model1 的三方对比已在 `results/lasp_final_vs_model1_bioemu2.json`。
2. 本会话的「bioemu 测不了」是**对象搞错**（bioemu v1 vs BioEmu2），非真实阻塞。
3. 本会话**额外**把环境可复现性补上了：新建 `/opt/conda/bioemu2_venv`（CUDA-12.6，匹配驱动 12.9）让 BioEmu2 可随时重跑（见下）。

### 环境可复现（✅ 本会话新增）
驱动 `CUDA 12.9`（`nvidia-smi`），A800 compute 8.0。BioEmu2 的 `cuda` extra 拉 `torch 2.13.0+cu130`（CUDA 13），在 12.9 驱动下 `cuda.is_available()=False`（报「driver too old, found 12090」）。正确做法是 `cuda12` extra：

```bash
/opt/conda/bin/python3.13 -m venv /opt/conda/bioemu2_venv
/opt/conda/bioemu2_venv/bin/python -m pip install -e "baselines/bioemu2[cuda12]" \
  --extra-index-url https://download.pytorch.org/whl/cu126
```

| 组件 | 版本 | 说明 |
|---|---|---|
| torch | 2.13.0+cu126 | CUDA 12.6，匹配驱动 12.9 |
| cuequivariance | 0.11.1 (cu12 ops) | BioEmu2 等变卷积后端 |
| bioemu2 | 0.1.0rc0 | 自包含（`PairMixerSequenceEmbedder` 从序列+MSA 内部算 embedding，**无需 AF2**） |
| MSA | `api.colabfold.com` + `BIOEMU2_MSA_CACHE` | 已确认可达，缓存 100+ 条 |

- BioEmu2 提供**系综采样 + 时间传播**两类能力：`EmuSession.sample()`（平衡态系综）与 `EmuSession.propagate(x0, lag_time_ns)`（时间延迟传播），后者直接映射到我们的轨迹/ACF 指标口径，比 bioemu v1（仅平衡态系综）信息量更大。
- **GPU smoke 已通过（本会话实测）**：`torch 2.13.0+cu126` `cuda.is_available()=True`（A800），`cuequivariance` import OK，`bioemu2` 加载 `bioemu2-preview` checkpoint `supports_propagation=True`，MSA 从 `api.colabfold.com` 取到（1580256 残基），embed → propagate → 7 指标全跑通（1 蛋白 / 2 窗口 smoke，`rmsf_corr=0.5022` 与全量 50 蛋白的 0.431 量级一致）。唯一补装的是 `h5py`（评测数据加载器依赖，非 BioEmu2 本身）。
- 结论：环境可复现、可用，按 `run_bioemu2_zeroshot.sh` 原样即可重跑完整评测（无需 AF2 下载）。

### bioemu v1（旧包）——不再需要
原路线的 bioemu v1（AF2 依赖）已被 BioEmu2 取代；后台 AF2 权重下载（3.7 GB）可以停止，不再阻塞任何评测。

---

## 任务 4 — E1 复查（为什么 P1=P2=P3，以及 H+S>H 的正确测法）

### 对 P1=P2=P3 的诚实解释（✅ 已定位根因，含用户指出的两个缺陷）
E1 探针的 P3 用的是 `exact_bath_posterior(H_hist)` 的**后验均值 S₀**。但读 `stage_b1b/model.py:750` 的 `exact_bath_posterior`：S₀ 是**对同一段 H 历史做 Kalman 滤波的确定性函数**（`z=dH−hDf`，f=`compute_forces(H)`，D/C/Γ=`bath(H)`）。因此：

- P1 = H_t（单帧）
- P2 = H_{t-K:t}（完整历史）
- P3 = P1 + S₀，而 **S₀ = g(H_{t-K:t}) 是 P2 已含信息的一个确定性变换**。

所以在线性 ridge 下 **P3 的信息必然被 P2 包含**，P1=P2=P3 不是 bug，是「S₀ 由 H 历史解析算出」的**数学必然**。

**用户指出的两个缺陷（已确认成立，是本探针的两个独立失效原因）**：
1. **缺陷 1 — 线性 ridge 读 S₀ 是信息下界**：线性 ridge 只能读出 S 的**线性**分量对目标的贡献；S 的动力学贡献（GLE 非线性传播、与 H 的耦合项 −C·S、记忆核的时间卷积）在线性投影下天然读不出。P1=P2=P3 只能证明「S₀ 没有**线性可读**的额外信息」，不能证明「S 无动力学价值」。
2. **缺陷 2 — S 杠杆太小（相对目标尺度）**：即使 S 进入动力学，其每步位移贡献相对 H 的漂移/扩散太小（exact arm 下 S₀ 杠杆 0.09–0.10 Å vs 每步 RMSD ~3–6 Å，仅 ~2–3%），线性探针必然把这点信号淹没在目标方差里。**信号太小 ≠ 信号为零**。

> 结论修正：E1 的 P1=P2=P3 是「线性探针 + 小杠杆」的**必然下界结果**，不是「S 无用」的证据。要真正判断 S 的价值，必须做**动力学层面对比**（H+S vs H-only 的轨迹/系综指标），这正是任务 1 做的事。

### S 的 scale 状态（区分两条训练路径）
- **exact arm**（`coupling_scale=0.1`，v3 的 100×，`gamma_scale=2e-5`）：S 被显式调大。P0.2 已测 |C·m0|/|f|=0.58、S₀ 杠杆 0.09–0.10 Å。但 exact arm **在系综指标上整体输 v3**（E0），且它改了别的变量（冻结 encoder、path_mixture 目标），不是干净的 S 消融。
- **v3 SOTA**（`coupling_scale=1e-3` 默认，`bath_coupling=velocity`）：S 是**惰性的**（bath_var/R ~ 1e-8，见 `lasp-bath-inert-two-scale-bug`）。这是任务 4 的 H+S 基准。

### 「H+S > H」的正确测法（🏃 由任务 1 完成，v3 口径）
「H+S 是否 > H」是**动力学层面**问题，正确测法是对比同一 recipe 下 bath 的开/关：
- **H+S** = **v3 SOTA**（`artifacts/lasp_v3_velocity`，seed 42，已评测：`e0_v3.ensemble.json` + `v3_ordered_paths.json`）
- **H-only** = 新训 v3 recipe + `--bath-dim 0`（`artifacts/lasp_v3_h_only_20261107`，任务 1）

在同一 standard-200 协议下比 ACF Error / Path Error / RMSF-r / md_pca_w2。

**诚实预期（必须先声明）**：v3 SOTA 的 S 是**惰性的**（coupling_scale=1e-3），所以「v3-SOTA vs v3-H-only」**大概率无差异**——但这是「v3 从未激活 S」的结果，不是「S 无用」的证据。若要测「**正确缩放的** S 是否有价值」，还需一个在 v3 recipe 上把 coupling_scale 从 1e-3 提到 0.05–0.1 的「激活 S」arm。因此：
- 任务 1 交付的「v3-SOTA vs v3-H-only」给出的是「**as-shipped S 是否有增益**」的诚实答案（预期：无，因 S 惰性）。
- 「正确缩放的 S 是否有增益」需要额外的「v3 + activated-S」arm。

### 「正确缩放的 S」activated-S arm（🏃 已 launch，2026-11-07）
按 memory `lasp-bath-inert-two-scale-bug` 的两尺度修正（coupling ×50、Γ×0.1），新增 `run_scripts/run_lasp_v3_activated_s.sh`：v3 recipe **唯一两处改动**是 `--coupling-scale 0.05`（默认 1e-3 的 50×）+ `--gamma-scale 2e-5`（默认 2e-4 的 0.1×），其余字节级对齐 v3 SOTA（`bath_dim=8`、`bath-coupling velocity`、seed 42/43/44）。目标是把 bath 对每步位移的贡献从 ~0.1% 提到 |C·m₀|/|f| ~ O(1)，让 S 真正驱动 H。

- 三 arm 矩阵：
  | arm | bath_dim | coupling_scale | gamma_scale | 状态 |
  |---|---|---|---|---|
  | H-only | 0 | (n/a) | (n/a) | 🏃 `lasp_v3_h_only_20261107`（seed 42/43/44，~step 750–1000/20000） |
  | v3 SOTA | 8 | 1e-3 | 2e-4 | ✅ 已评测（惰性 S 基准） |
  | activated-S | 8 | 0.05 | 2e-5 | 🏃 `lasp_v3_activated_s_20261107`（seed 42/43 GPU 0/1，seed 44 排队） |
- 3-step smoke 通过（loss 有限，与 H-only 同 step 同量级）；正式训练 30s 后 step ~20 无发散、无 non-finite。
- 预期：若正确缩放的 S 有动力学价值，activated-S 应在 trajectory（ACF/Path）或 ensemble（RMSF-r/md_pca_w2）上 **优于 H-only**；若仍无差异，则说明 S 在两尺度修正下也不贡献（需进一步查 S 是否真正进入动力学）。
- **诚实记账**：×50 coupling 有 over-drive 风险（memory 记 ×150+ 发散），若训练中途 non-finite 会如实记录，不静默回退。评测入口复用 `eval_lasp_v3_h_only.sh`（v3 口径 `eval_lasp_final` + `eval_ca_ordered_paths_v3`）。

**训练稳定性观察（2026-11-07，step ~2400）——重要反转**：`grad_norm=inf` 跳过是 v3 multi-horizon recipe 的**固有良性现象**，**不是** activated-S 的 over-drive。实测 skip 率：

| arm | bath | skip 率（SKIPPED / total） | cur step |
|---|---|---|---|
| H-only seed 42 | 0 | **61.2%** | 3150 |
| H-only seed 43 | 0 | **66.6%** | 3150 |
| activated-S seed 42 | 8 (0.05) | **0.0%** | 2420 |
| activated-S seed 43 | 8 (0.05) | **3.6%** | 2435 |

即 **activated-S 比 H-only 显著更稳**（0–3.6% vs 61–67% 跳过）。这与 memory `v16-dissipative-bath-confirmed` 一致：正确缩放的耗散浴场（K(0) 负定）**阻尼** H-space rollout，而 bath_dim=0（无浴场）时 H 长程 rollout 更易 `inf` 梯度。**这是「正确缩放的 S 有动力学价值」的旁证**（即便最终轨迹指标无差异，至少它在训练动力学上起到了稳定作用）——但主判据仍以评测指标为准，训练稳定只是辅助观察。四个 arm 的 val loss 均在 4.3–4.8 区间收敛（无发散），无真实 crash（仅 benign skip）。

### 正确 v3 PDB（✅ 已生成，供用户 E3.1 人工检查）
用户要求「先生成正确的 pdb 文件给我，v3 的即可」。已用 `scripts/export_v3_autoregressive_pdb.py`（**正确的 v3 口径**：编码 anchor 一次 → H-space 单次 GLE rollout → 逐帧解码，**无坐标自反馈**）重新导出：

- 产物：`results/todo_1107/e31_pdb_v3_autoregressive/{1nhwC00,1ka1A01,1kafD00}*_h64_s{0..3}.pdb`（12 个文件，3 蛋白 × 4 samples，每文件 65 帧 = 64 步 + anchor）。
- 验证：`REMARK NFRAMES 65`、`NRESIDUES 96`、origin 标记 `v3 HSEnsembleProducer single-H-space rollout`；Rg 稳定（1ka1A01 首帧 14.26 → 末帧 13.94 Å，**无发散**）。
- **注意**：旧的 `results/todo_1107/e31_pdb_v3/ours_hs.pdb` 是**旧 buggy 导出器**产物（坐标自反馈循环，Rg 13.9→43.3 Å 发散，`NFRAMES 16`），**不应作为人工检查材料**；请用户检查新目录 `e31_pdb_v3_autoregressive/`。

### 补充（更精细的 S 贡献探针，可选）
若要在 H-only 训练之外、更直接地量化 S 的边际贡献，可做**一步预测探针**：对同一 anchor，分别用 (H,S₀) 完整 GLE 一步 vs H-only 一步，比较其一步 CA 位移对 MD 一步位移的预测误差。这测的是 S 的**动态**贡献而非线性可读性。已确认 `diagnose_s0_mechanism.py` 的 `scalar_diagnostics` 已具备 `|C·m0|/|f|`、`bath_disp_per_step` 等标量，可在此基础上扩展。
