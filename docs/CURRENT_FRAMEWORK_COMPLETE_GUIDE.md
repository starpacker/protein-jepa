# Protein-JEPA-Frame / LSP-WM 当前框架完整指南

> 本文档以当前工作区中的代码为准，描述当前 ABC pipeline 的实际数据流、模型、训练目标、坐标约定、OpenFold 解码链路和 evaluation 协议。它不是早期设计文档的复述，也不把历史实验报告中的数值自动当作当前模型的最终性能。

本文的目标是让读者可以回答以下问题：

1. 一个 mdCATH 轨迹样本从 HDF5 到模型输入经历了什么变换？
2. Stage A 学习的到底是什么表示？
3. Stage B 中 `Q`、`G`、`S`、energy landscape、force、diffusion 和 transition 各自承担什么角色？
4. Stage C 如何把 `H + Q` 和 anchor backbone 还原成 residue frames、torsions 以及 OpenFold atom37？
5. evaluation 的 oracle、direct、autonomous、persistence、qshuffle 分别测量什么，哪些 metric 是单结构指标，哪些必须使用 ensemble 协议？
6. 一个报告什么时候可以被称为完整、可信，什么时候只是 smoke test 或诊断结果？

---

## 1. 当前框架的定位

### 1.1 一句话概括

当前框架是一个面向蛋白质分子动力学轨迹的三阶段 latent world-model pipeline：

```text
MD backbone trajectory
        |
        v
Stage A: SE(3)-invariant JEPA representation learning
        |
        v
Stage B: per-residue latent dynamics + conservative energy + finite bath
        |
        v
Stage C: latent-to-structure decoder
        |
        v
OpenFold atom37 structure / PDB
        |
        v
single-structure, rollout, intervention, ensemble evaluation
```

当前实现的最清晰的边界是：

- Stage A 负责学习对蛋白 backbone geometry 和时间变化有用的 `H` 表示；
- Stage B 负责把 `H` 投影为 per-residue latent dynamical state `Q`，并在 `Q` 上建立具有能量梯度、扩散和有限 memory bath 的闭环 transition；
- Stage C 负责把任意来源的目标时刻 `Q`（oracle、direct 或 autonomous）与 context `H`、anchor frame 合成为结构；
- evaluation 必须区分“预测 Q 的能力”“decoder 的能力”“closed-loop rollout 的能力”和“什么都不做的 persistence baseline”。

### 1.2 当前实现和完整 LSP-WM 目标的关系

仓库的顶层 README 还包含更广义的 LSP-WM 目标：

```text
u_t = (q_t^protein, e_t^environment, m_t^memory)
```

其中 `e_t` 是环境状态，`m_t` 是 memory state，`a_t` 是 intervention。当前 ABC pipeline 对这一目标的实现是一个受控子集：

- protein state 以 per-residue `Q` 表达；
- finite bath `S` 充当显式的有限 memory/auxiliary state；
- environment 目前主要由温度条件和一个很小的显式 `e` 容器体现；
- intervention evaluation 目前以 temperature intervention 等实验脚本为主，而不是完整的多环境联合 world model；
- Stage C 当前从 backbone 轨迹训练，不能声称已经用完整 atom37 原子轨迹学习了真实 side-chain dynamics。

因此，“LSP-WM”在本文中表示当前实现的架构方向，而“完整 protein-environment world model”仍然是更大的目标，不能由当前 ABC 代码自动推出。

### 1.3 当前 canonical 协议

当前修复后的 canonical 约定如下：

| 项目 | canonical 约定 |
|---|---|
| 内部坐标单位 | Ångström |
| 内部时间单位 | ps |
| backbone 原子顺序 | `[N, CA, C, O]` |
| Stage A spatial output | `H: [B,T,L,D_H]`，通常 `D_H=128` |
| Stage B state | `Q: [B,L,8]` |
| Stage B finite bath | `S: [B,L,16]` |
| Stage B global summary | `G = masked_mean(Q): [B,8]` |
| Stage C anchor | context window 的最后一帧 |
| Q 与 future Q 的编码 | 都使用 `encode_state(..., use_temporal=True)` |
| atom37 canonical ordering | OpenFold `residue_constants.atom_types` ordering |
| single-structure fit | OpenFold/Biopython SVD Kabsch，CA fit |
| ensemble metrics | AlphaFlow official scripts；不能用单结构指标冒充 |
| 无效样本 | fail-fast 或明确非 `ok` 状态，不能静默变成 0/NaN 的“完整报告” |

### 1.4 代码地图

核心文件：

- 数据：[`protein_jepa_frame/data/schema.py`](../protein_jepa_frame/data/schema.py)、[`dataset.py`](../protein_jepa_frame/data/dataset.py)、[`collate.py`](../protein_jepa_frame/data/collate.py)、[`manifest.py`](../protein_jepa_frame/data/manifest.py)
- Stage A：[`models/stage_a/model.py`](../protein_jepa_frame/models/stage_a/model.py)、[`models/stage_a/objectives.py`](../protein_jepa_frame/models/stage_a/objectives.py)、[`training/stage_a.py`](../protein_jepa_frame/training/stage_a.py)
- Stage B：[`models/stage_b/model.py`](../protein_jepa_frame/models/stage_b/model.py)、[`state.py`](../protein_jepa_frame/models/stage_b/state.py)、[`transition.py`](../protein_jepa_frame/models/stage_b/transition.py)、[`objectives.py`](../protein_jepa_frame/models/stage_b/objectives.py)
- Energy/LaSP：[`models/lasp/energy.py`](../protein_jepa_frame/models/lasp/energy.py)、[`bath.py`](../protein_jepa_frame/models/lasp/bath.py)、[`integrator.py`](../protein_jepa_frame/models/lasp/integrator.py)
- Stage C：[`models/stage_c/model.py`](../protein_jepa_frame/models/stage_c/model.py)、[`structure_decoder.py`](../protein_jepa_frame/models/stage_c/structure_decoder.py)、[`geometry.py`](../protein_jepa_frame/models/stage_c/geometry.py)、[`direct_head.py`](../protein_jepa_frame/models/stage_c/direct_head.py)、[`rollout.py`](../protein_jepa_frame/models/stage_c/rollout.py)、[`training/stage_c.py`](../protein_jepa_frame/training/stage_c.py)
- OpenFold 解码：[`models/stage_c/openfold_atom37.py`](../protein_jepa_frame/models/stage_c/openfold_atom37.py)
- 单结构指标：[`eval/structure_metrics.py`](../protein_jepa_frame/eval/structure_metrics.py)
- checkpoint 兼容加载：[`eval/stage_c_checkpoint.py`](../protein_jepa_frame/eval/stage_c_checkpoint.py)
- canonical world-model evaluation：[`scripts/eval_world_model.py`](../scripts/eval_world_model.py)
- AlphaFlow/MDTraj adapter：[`eval/stage_c_evaluator.py`](../protein_jepa_frame/eval/stage_c_evaluator.py)
- side-chain/clash evaluation：[`scripts/eval_sidechain_quality.py`](../scripts/eval_sidechain_quality.py)

### 1.5 当前代码状态和报告可信度

当前代码已经完成一轮针对 ABC pipeline 的 evaluation/geometry audit。与本手册直接相关的测试集合包括：

```bash
pytest -q \
  tests/test_structure_metrics.py \
  tests/test_openfold_atom37.py \
  tests/test_gate5_ensemble_metrics.py \
  tests/test_stage_c_rollout.py \
  tests/test_model_train.py \
  tests/test_gate3_atom37_layer.py
```

这组相关测试此前通过为 `80 passed`（有 warnings）。全量 pytest 仍可能因为 dirty worktree 中删除了大量与当前 ABC 无关的历史脚本而出现 collection errors；这两种状态要分开记录。

当前 `artifacts/eval_audit_canonical1/full_report.json` 是一份 1-sample canonical smoke report。它的意义是验证模型加载、canonical state path、OpenFold geometry 和 fail-fast schema，不是最终 held-out quality report。历史上出现过的“oracle≈qshuffle≈persistence”结果是 decoder/Q-path 的诊断证据；除非明确注明 checkpoint、split、protocol 和 status，否则不能把旧报告中的数值称作当前最终性能。

---

## 2. 数据：mdCATH 从文件到训练样本

### 2.1 数据来源和 split

`scripts/build_mdcath_manifest.py` 针对 mdCATH HDF5 数据生成 JSONL manifest。每一行描述一条轨迹：

```json
{
  "h5_path": "/path/to/mdcath_dataset_xxx.h5",
  "protein_id": "domain_id",
  "trajectory_id": "domain_id_T348_R0",
  "domain_id": "domain_id",
  "format": "mdcath-hdf5",
  "dataset": "mdCATH",
  "num_frames": 1000,
  "frame_interval_ps": 1000.0,
  "temperature_k": 348,
  "replica_id": "0",
  "metadata": {"hdf5_group": "domain_id/348/0"}
}
```

默认按 domain 做 80/10/10 train/validation/test split，而不是随机把同一 domain 的不同 replica 分到不同 protein split。这个约束很重要：如果同一 CATH domain 的轨迹同时出现在 train 和 test，模型可能只是在记忆结构，而不是泛化动力学。

当前生成脚本使用的温度标签为 `320, 348, 379, 413, 450 K`，replica 为 `0..4`。默认 frame interval 是 `1000 ps`。实际值写入 manifest，并由 dataset 计算 `time_ps` 和 horizon duration；模型不应该把原始 frame index 当成物理时间。

### 2.2 HDF5 的读取路径

`TrajectoryClipDataset` 支持三类格式：

1. `npz`：要求 `coords` 形状为 `[F,L,4,3]`；
2. `protein-jepa-hdf5`：读取标准化的 `coords`、`atom_mask`、`time_ps` 和 topology；
3. `mdcath-hdf5`：从 domain 的 PDB topology 解析原子索引，再从全原子坐标中抽取 backbone。

mdCATH 路径的关键步骤：

```text
HDF5 full atom coordinates
        |
        +-- domain PDB topology
        |       |
        |       +-- residues in PDB order
        |       +-- N/CA/C/O atom indices
        |       +-- aatype, residue_index, chain_id
        |
        +-- gather N/CA/C/O
                |
                v
coords [T,L,4,3], atom_mask [T,L,4]
```

PDB topology parser 处理：

- `ATOM` 和 `HETATM` 记录；
- altloc 空值和 `A`；
- chain name、residue number、insertion code；
- 三字母到一字母氨基酸映射；
- 不存在的 N/CA/C/O 用 mask 标记为 invalid，而不是把零坐标当作真实坐标。

读取后所有非有限的 backbone 坐标会被置零并从 `atom_mask`/`residue_mask` 中排除。后续 evaluation 的有效性规则更严格：要评分的 residue 出现非有限值时直接报错，而不是在 aggregate 中偷偷删除。

### 2.3 样本字段

单个 clip 的主要字段：

| 字段 | shape/类型 | 含义 |
|---|---|---|
| `coords` | `[T,L,4,3]` | `[N,CA,C,O]` backbone coordinates |
| `atom_mask` | `[T,L,4]` | atom-level validity |
| `residue_mask` | `[T,L]` | residue 是否完整有效 |
| `aatype` | `[L]` | 20 AA + unknown index 20 |
| `residue_index` | `[L]` | 原始 residue index，不强制从 0 连续 |
| `chain_id` | `[L]` | chain integer id |
| `time_ps` | `[T]` | 绝对物理时间 |
| `temperature_k` | scalar | trajectory 温度 |
| `protein_id` | string | domain/protein identity |
| `trajectory_id` | string | domain + temperature + replica identity |
| `dataset` | string | 例如 `mdCATH` |
| `forces` | optional `[T,L,4,3]` | 若启用 force loading |

`collate_clips` 对不同长度蛋白进行 residue padding：

- 坐标 padding 为 0；
- mask padding 为 false；
- `aatype` padding 为 unknown index 20；
- `residue_index`/`chain_id` padding 为 -1；
- 计算 `dt_ps = diff(time_ps)`，供 dynamic encoder 使用。

### 2.4 物理时间和 paired window

Stage A/B/C 的主训练都使用 paired-window protocol。设 context offsets 为：

```text
o_0, o_1, ..., o_{T-1}
```

给定 raw-frame horizon `h`，一个样本是：

```text
context frames = start + offsets
future  frames = start + h + offsets
horizon_time_ps = future[0].time_ps - context[0].time_ps
```

默认训练配置常用：

```text
num_frames = 8
frame_stride = 4
paired_horizons_frames = [32, 64, 96]
```

因此 raw frame horizon 不是 32 ps，而是 `32 × frame_interval_ps`。在 mdCATH 默认 frame interval 为 1000 ps 时，分别大致对应 32 ns、64 ns、96 ns。模型的时间 embedding 接收 `horizon_time_ps`，而不是把 `32/64/96` 直接当作秒或 ps。

### 2.5 防止 context/future 泄漏

`validate_real_paired_window_protocol` 对有 manifest 的真实数据执行 overlap audit。它计算 context offset 集和 future offset 集，要求：

```text
{context offsets} ∩ {horizon + context offsets} = ∅
```

如果发生共享 frame，真实数据训练会直接拒绝。synthetic dataset 仍用于 shape/gradient smoke test，但不能把 synthetic smoke 的通过当作 mdCATH 协议已经通过。

### 2.6 `paired_all_horizons`

有两种 paired sampling 方式：

- `paired_all_horizons=false`：每个 dataset item 选择一个 horizon；
- `paired_all_horizons=true`：一个 context 对多个 horizon，返回 `pairs` 列表，`collate_paired_windows` 将其 flatten，并保留 group index/size。

后者适合在同一 context 上同时监督多个物理时间尺度，但会改变每个 loader item 的有效 batch 数，不应和普通 batch 统计混淆。

---

## 3. 坐标、frame、mask 和 SE(3) 约定

### 3.1 坐标单位

代码内部约定：

- 坐标为 Å；
- 时间为 ps；
- 温度为 K；
- Stage B 的 `Q` 是无量纲/latent coordinate，不能直接解释成 Å；
- Stage C 的 translation update 才是 Å 量纲的局部位移。

`coordinate_scale` 可以对输入坐标做统一缩放，但 canonical mdCATH 配置为 `1.0`。如果改变 scale，必须同时重新理解 bond lengths、distance cutoffs、decoder update bounds 和 evaluation 输出单位。

### 3.2 Backbone 顺序

项目内部的 backbone tensor 是：

```text
[N, CA, C, O]
```

OpenFold atom37 的 backbone index 仍然是：

```text
N  = 0
CA = 1
C  = 2
O  = 4
```

不要把 `[N, CA, C, O]` 的第四个位置误认为 atom37 index 3；atom37 index 3 是 CB。

### 3.3 Project atom37 和 OpenFold atom37

历史项目 schema 的 atom37 tail ordering 与 OpenFold 不同。OpenFold 的 authoritative order 来自：

```python
openfold.np.residue_constants.atom_types
```

当前 `openfold_atom37.py` 暴露：

- `OPENFOLD_ATOM37_NAMES`
- `PROJECT_ATOM37_TO_OPENFOLD`
- `OPENFOLD_ATOM37_TO_PROJECT`
- `to_project_ordering`
- `to_openfold_ordering`

canonical decoder 输出是 OpenFold ordering。若调用者拥有 project-schema ordering，必须显式转换；不允许按“看起来差不多”的 index 直接比较或写 PDB。

### 3.4 residue frame

项目的 `RigidBodyFrame.from_backbone(N, CA, C)`：

- origin 为 CA；
- local x 轴沿 `CA -> C`；
- local y/z 由 N-CA-C 平面和右手系构成；
- 返回 `R,t`，其中

```text
x_global = R @ x_local + t
```

Stage C 的 frame update 在 anchor frame 的 local coordinate 中预测：

```text
R_new = R_anchor @ R_update
t_new = t_anchor + R_anchor @ dt_local
```

这里 `dt_local` 是每个 residue 的局部 translation residual。`q_translation_head` 的输出也先经过 `tanh` 和最大幅度 bound，再转换到 global 坐标。

### 3.5 SE(3) 的含义和限制

Stage A 的 `ProteinGeometryEncoder` 不读取绝对坐标，而读取：

- residue 内部四原子距离；
- residue 间 CA 距离及 RBF；
- aatype/residue/chain embeddings；
- 可选 orientation/torsion features。

因此 Stage A 表示设计为对全局旋转和平移不敏感。Stage C 则必须重新使用真实 anchor frame，把预测的内部/相对运动放回输入坐标系。两者不是矛盾：

- `H/Q` 描述 conformation/dynamics；
- anchor frame 提供当前观察的 gauge 和 coarse placement；
- target-relative loss 去除不可识别的全局 COM gauge，但保留 residue-wise deformation。

---

## 4. Stage A：JEPA representation pretraining

### 4.1 Stage A 的目的

Stage A 不直接输出坐标。它学习一个对后续动力学和结构解码有用的 representation：

```text
backbone clip -> spatial residue features H -> temporal state representation
```

它同时解决两个问题：

1. 局部/残基级 masked representation prediction；
2. 多物理 horizon 的 endpoint forecasting。

这样得到的表示不应只编码静态 sequence identity，也应该包含未来 dynamics 可预测的内容。

### 4.2 Spatial encoder

`ProteinGeometryEncoder` 输入：

```text
coords       [B,T,L,4,3]
residue_mask [B,T,L]
aatype       [B,L]
residue_index[B,L]
chain_id     [B,L]
token_mask   [B,T,L] 或 None
```

对每个 residue 建立初始 embedding：

```text
geom_proj(internal distances)
+ aatype embedding
+ residue_index embedding
+ chain_id embedding
```

`internal distances` 是四个 backbone 原子六种 pair distance：

```text
(N,CA), (CA,C), (C,O), (N,C), (N,O), (CA,O)
```

若 `orientation_features=true`，还会加入由相邻 residue 的 backbone torsions 和 pair orientation 构造的 features。之后使用多层 distance-biased graph attention：

- q/k/v 是 residue feature；
- pair CA distance 经过 RBF 后作为 head bias；
- invalid residue 作为 key mask；
- 层内残差、LayerNorm、MLP；
- 输出 `H [B,T,L,D_H]`。

Spatial encoder 不是 OpenFold IPA，也不直接产生 rigid frame；它是 Stage A 的 invariant geometric encoder。

### 4.3 Temporal encoder

`TemporalStateEncoder` 先把 residue-level H pool 成 frame-level feature：

```text
mean pooling 或 kinetic query pooling -> [B,T,D_H]
```

随后使用 Transformer temporal blocks。可选信息包括：

- learned frame position embedding；
- physical time embedding（log-time）；
- temperature/dataset condition；
- frame delta features；
- causal attention；
- motion tower。

输出：

```text
frame_states [B,T,D_H]
clip_state   [B,D_H]
```

Stage A 默认可以有 EMA target temporal encoder，与 online temporal encoder 保持同构。

### 4.4 EMA online/target 结构

Stage A 的 canonical JEPA 结构：

```text
online context encoder  <- masked context, receives gradient
EMA target encoder       <- full visible target, no gradient
online predictor         <- predicts target latent
```

对于 temporal branch 同样有 target tower。训练中 target encoder 通过 momentum update：

```text
target <- momentum * target + (1 - momentum) * online
```

target encoder 在 `.train()` 调用后仍被强制 `.eval()`，避免 dropout 或 batch-dependent behavior 使 target 漂移。

### 4.5 Mask objective

`SpatiotemporalBlockMasker` 在 `[T,L]` token 网格上生成：

- `context_mask`：可见 token；
- `target_mask`：被遮盖、需要预测的 token。

默认 block masking 会随机选择 temporal span 和 residue span，并确保：

- 有足够 context；
- target 不为空；
- target 与 context 不重叠。

也支持 `future` mask：只看前半段或 context 区间，预测未来区间。这种模式更接近 causal forecasting。

online branch 得到 `context_latents`；EMA target branch 对完整 clip 得到 `target_latents`。`SpatiotemporalPredictor` 将 masked token 和 visible token 放进 Transformer，预测 residue-level target。若启用 temporal encoder，`StatePredictor` 额外预测 frame-level target state。

masked objective 的形式是 masked Smooth L1 或 MSE：

```text
L_mask = w_residue * mean_masked(pred_residue, target_residue)
       + w_state   * mean_masked(pred_state, target_state)
```

### 4.6 Paired endpoint objective

`forward_paired` 的数据是明确命名的 context/future namespace，避免把 context 和 target 的字段混用。

流程：

1. context 通过 online spatial encoder；
2. context 通过 online temporal encoder，得到 `context_endpoint`；
3. context 和 future 都通过 EMA spatial/temporal encoders，得到 teacher endpoints；
4. `PairedDeltaPredictor` 接收 context endpoint、`horizon_time_ps`、temperature/dataset condition；
5. 输出 latent delta；
6. 根据 `paired_teacher_space_delta` 的设定，把 delta 加到 online 或 teacher context anchor 上。

主要目标：

```text
delta_target = future_endpoint_ema - context_endpoint_ema
delta_pred   = PairedDeltaPredictor(context_endpoint, horizon_time_ps, T)
L_delta      = SmoothL1(delta_pred, delta_target)
```

也可以记录 endpoint prediction loss 和 persistence baseline，但 canonical Stage A 主要用 delta loss 训练。

### 4.7 Stage A joint training loop

`StageAJointBatchIterator` 在 mask loader 和 paired loader 之间按固定 `paired_ratio` 交错采样：

```text
mask batch   -> StageA.forward_mask_prediction
paired batch -> StageA.forward_paired
```

每个 step 的训练过程包括：

1. 物理时间和 mask/paired batch 准备；
2. online forward；
3. EMA target forward；
4. loss aggregation；
5. backward、gradient clipping；
6. optimizer step；
7. EMA update；
8. JSONL logging；
9. periodic checkpoint。

Stage A checkpoint 通常包含完整 `model` state、optimizer、epoch/step、config。Stage B 初始化时把 `context_encoder.*` remap 成 `spatial_encoder.*`。

### 4.8 Stage A evaluation

[`scripts/eval_stage_a_heldout.py`](../scripts/eval_stage_a_heldout.py) 在 held-out protein 上做 paired endpoint evaluation：

```text
prediction = context_endpoint + delta_prediction
target     = future_endpoint
persistence= context_endpoint
```

对每个 horizon 计算：

- model MSE；
- persistence MSE；
- skill = `1 - model_mse / persistence_mse`；
- skill > 0 的样本比例；
- horizon-level mean skill。

Stage A 的 skill 只能说明 latent endpoint prediction 是否超过 latent persistence，不能直接说明 Stage C 坐标预测会超过 persistence。

---

## 5. Stage B：per-residue LaSP latent dynamics

### 5.1 Stage B 的角色

Stage B 将 Stage A 表示变成可演化的 latent dynamics：

```text
H_history -> Q_t -> energy landscape -> conservative force
                              |
                              +-> diffusion/noise
                              +-> finite bath coupling
                              +-> closed-loop transition
```

当前 canonical Stage B 不是一个“从 Q 直接回归 future Q 的普通 MLP”。它明确包含：

- per-residue `Q`；
- `G` global pooled state；
- per-residue finite bath `S`；
- CA distance graph；
- conservative energy；
- force = `-dA/dQ`；
- FDT-like diffusion/noise；
- Euler-Maruyama + exact-OU bath update。

### 5.2 Stage B 的表示

给定 context clip：

```text
H_history = Stage A spatial + temporal history
Q = projection(H_last)
G = masked_mean(Q)
S_0 = bath_initializer(H_last)
```

典型 shape：

```text
H_last [B,L,128]
Q      [B,L,8]
G      [B,8]
S      [B,L,16]
```

`encode_state` 是唯一 canonical state encoding entry point。它返回：

- `q`；
- `h_last`；
- 2D residue mask；
- optional `bath_s0`；
- optional CA pair distances；
- optional forces/energy。

训练和所有 evaluation 都应使用同一方法，以保证 context Q、future Q、oracle Q 在同一 representation space。

### 5.3 H→Q projection

当前 projection 是三层 MLP：

```text
Linear(D_H,256) -> SiLU
Linear(256,128)  -> SiLU
Linear(128,8)
```

Stage B canonical training 中，Stage A spatial/temporal encoder 冻结；projection、energy graph、diffusion、bath initializer、bath parameter nets 等 Stage B 模块训练。

### 5.4 `G` global state

```python
G = masked_mean(Q, residue_mask)
```

`G` 的定义必须在 energy/transition 内重新计算，调用方不能单独传一个与 Q 不一致的 G。这避免了 `Q` 已变而 `G` 仍是旧值的内部不一致。

### 5.5 Energy graph

Stage B 使用 `ResidueMessagePassingPotential`。每个 residue 的能量依赖：

- 自身 `q_i`；
- 邻居 `q_j`；
- CA distance 的 RBF；
- 可选 condition；
- 可选 temperature feature `log(T/T_ref)`。

边特征大致是：

```text
[q_i, q_j, RBF(CA_distance_ij), condition]
```

message passing 对 valid residues 进行多层更新，最后生成每 residue scalar energy：

```text
energy_i = energy_head(node_i)
energy_total = sum_i energy_i * mask_i
```

其中 RBF centers 覆盖约 `1..30 Å`，canonical energy graph 使用 8 个 RBF、3 层 message passing 和约 128 hidden width。超参数不是框架逻辑的核心，但 distance graph 和 mask 是核心。

### 5.6 Conservative force

Force 由总能量对 Q 的 autograd 得到：

```text
F_i = -∂ A(Q, CA_distances, T) / ∂ Q_i
```

关键点：

- 不是独立训练一个任意 force head；
- force 必须来自 scalar energy 的梯度；
- training 时 `create_graph=True`，这样 force loss 可以反向影响 energy parameters；
- evaluation 时可以关闭高阶 graph；
- mask 会在 force 上清零。

这保证了 energy/force 在数学上保持 conservative consistency，但不自动保证 energy landscape 学得好。force floor、force magnitude regularization 和 rollout diagnostics 才是判断 landscape 是否有用的手段。

### 5.7 Diffusion

当前 canonical path 使用 scalar per-protein mobility，而不是依赖 Q 的真正 state-dependent diagonal diffusion。原因是：若 `D(Q)` 改变，就必须加入对应的 Ito divergence correction；在没有完整 divergence term 时，state-dependent diffusion 会破坏 transition 和 integrator 的一致性。

实现保留了 diffusion head 以兼容 checkpoint 和 likelihood calibration，但其 Q dependence 被 detach，最终 diffusion 近似为：

```text
D(T) = diffusion_scale * positive_scalar * T/T_ref
```

其中 `diffusion_scale` canonical 配置约为 `1e-6`，temperature 通过参考温度缩放。

### 5.8 Finite bath `S`

有限 bath 的 shape 是 `[B,L,16]`。初始化：

```text
S_0 = bath_initializer(H_last)
```

bath 参数从 Q 产生：

```text
Gamma(Q) = softplus(bath_gamma_net(Q)) + 0.01
C(Q)      = coupling_scale * bath_coupling_net(Q)
```

其中 `C` shape 为 `[B,L,D_Q,D_S]`。系统受到 `C @ S` 的额外 forcing，bath 受到 `C^T @ F` 的 back-action。

### 5.9 Closed-loop transition

当前主 transition 可以概括为：

```text
dQ = D(Q/T) * F(Q) dt + C(Q) S dt + sqrt(2 D dt) ξ
dS = -Gamma(Q) S dt + back_action(Q,F) dt + OU_noise
```

代码实际以 sub-stepping 实现：

1. 将 requested `dt_ps` 拆成不超过 `max_substep_dt_ps` 的小步；
2. 每个小步重新计算 energy、force、diffusion；
3. 更新 Q 的 drift 和可选 FDT noise；
4. 用 exact OU decay 更新 S，并加入 back-action/noise；
5. 对 invalid residue 应用 mask；
6. 返回 Q、S 和 diagnostics。

当前实现移除了“最多只做 10 个 substeps”的旧 cap。如果 requested dt 需要数百个 substeps，会给 warning；这通常意味着调用者给了不合理的 horizon/step 组合。

### 5.10 Stage B transition NLL

给定观测到的 `Q_next`，teacher-forced one-step mean 为：

```text
mean = Q + dt * D * F
if bath exists:
    mean += dt * C^T S
cov = 2 * dt * D
```

然后计算 Gaussian-like per-residue NLL，并在 valid residues 上平均。为防止小 variance 造成 raw continuous-density NLL 大量为负并硬饱和，当前实现对 raw elementwise NLL 使用 softplus transform 后再聚合。

训练和 integrator 使用同一组 drift/covariance 定义，这是 transition likelihood 与 rollout 物理路径一致性的必要条件。

### 5.11 Stage B 训练目标

`stage_b_total_loss` 主要包含：

#### 5.11.1 Transition NLL

让 one-step latent transition 对 future endpoint 有似然。

#### 5.11.2 Force floor

```text
f_norm = ||F||
penalty = relu(target_threshold - f_norm)^2
```

它防止 energy collapse 到几乎平坦的 surface。若 `F≈0` 而 noise 不为 0，autonomous rollout 会接近无界 random walk；但是 force floor 不是“真实物理定律”，只能作为防止平坦退化的 regularizer。

#### 5.11.3 Force magnitude regularization

当前命名为 `force_direction_weight` 的默认项实际上主要惩罚 force magnitude 平均值，充当 soft ceiling；只有额外启用 `force_cosine_weight` 时，才有 force 与 observed `ΔQ` 的 cosine alignment。

因此读取日志时，不能把 `force_mag_loss` 自动解释为方向 supervision。

#### 5.11.4 Energy regularization

对 energy total 的平方做小权重 regularization，防止 energy 数值漂移。

#### 5.11.5 Rollout MSE/NLL

可选地从 `Q_0,S_0` 做多步 rollout，与 future encoded Q 对比。当前训练实现使用 deterministic rollout（noise off），因为早期 force 太小的时候 stochastic rollout 的 noise 会淹没梯度；evaluation 的 autonomous 模式可以使用 stochastic FDT noise。

#### 5.11.6 Semigroup consistency

定义：

```text
T_{2dt}(z) ≈ T_dt(T_dt(z))
```

实现用 deterministic transition，并将 dt 放大一个 multiplier（当前 objective 默认约 5）以放大局部 curvature 的差异。中间状态会 detach，以限制 graph 深度和显存。

这项 loss 只能检查数值 transition 的局部 semigroup consistency；它不是完整的 Chapman-Kolmogorov distributional test，也不等价于真实 long-horizon accuracy。

### 5.12 Stage B 训练 freeze contract

canonical Stage B 的 optimizer 参数是除以下模块之外的所有 Stage B 参数：

- `spatial_encoder.*`
- `temporal_encoder.*`
- target/EMA 相关模块

也就是说 Stage B 训练的是：

- H→Q projection；
- energy graph；
- diffusion scalar/head；
- bath initializer；
- bath gamma/coupling；
- transition/dynamics 相关可训练模块。

`full_abc_stage_b_canonical_v1.yaml` 的重要协议：

```text
max_steps = 2760
context/future horizons = 32,64,96 raw frames
base_step_ps = 100
force floor target = 0.3
gradient clip = 10
```

用户明确接受 Stage B 只跑到 2760 steps；这不是把 B “补到理论终点”的必要条件，而是一个固定预算 checkpoint protocol。

### 5.13 Stage B 日志应该看什么

最重要的诊断不是只看 total loss，而是同时看：

- `transition_nll` 是否下降；
- force norm 是否从接近 0 学到有限、但不爆炸的范围；
- `energy_total` 是否发散；
- diffusion 是否被 noise 主导；
- rollout divergence rate；
- semigroup loss 是否有限且有变化；
- Q variance 是否 collapse；
- context Q 与 future Q 是否在同一 encoder path 上产生；
- autonomous Q norm 随 horizon 是否指数爆炸。

“loss 下降”不能单独证明 energy landscape 正常；flat energy 也可能在某些 NLL 设计下有 deceptively good 的短期数值。

---

## 6. Stage C：从 latent state 到结构

### 6.1 Stage C 的任务定义

给定：

```text
H_context   = context history 的 per-residue representation
Q_target    = 某个目标时间的 dynamics state
anchor      = context 最后一帧的 backbone frames
sequence    = aatype/residue/chain metadata
```

Stage C 输出目标时间的结构：

```text
H_context + Q_target + anchor frame
        -> residue frame update
        -> torsions
        -> OpenFold atom37
```

`Q_target` 可以来自：

1. oracle future encoder；
2. direct endpoint head；
3. autonomous Stage B rollout。

三者必须使用同一个 StructureDecoder，才能把 oracle 作为 decoder upper-bound、direct 作为 endpoint prediction、autonomous 作为 full system 进行可比分析。

### 6.2 Stage C context encoding

当前 canonical path：

```python
state = stage_b.encode_state(
    context_coords,
    context_mask,
    aatype,
    residue_index,
    chain_id,
    time_ps=context_time_ps,
    temperature_k=temperature,
    use_temporal=True,
)
h_context = state["h_last"]
q_context = state["q"]
```

这里的 `h_last` 不是单纯 spatial encoder 最后一帧的输出；当 `use_temporal=True` 时，会把 temporal frame state 加到 last-frame spatial feature 上。这个约定必须同时用于 training、oracle extraction 和 evaluation。

### 6.3 Anchor frame

anchor 是 context 的最后一帧：

```text
anchor_backbone = context_coords[:, -1]
R_anchor, t_anchor = from_backbone(N_anchor, CA_anchor, C_anchor)
```

可选 `anchor_noise_level` 只腐蚀 decoder 的 anchor translation；H 和 oracle Q 仍来自未腐蚀观察。设计动机是避免 decoder 只复制 anchor，同时保持 Q 具有解释 endpoint motion 的机会。

### 6.4 Decoder input embedding

`StructureDecoder` 对每个 residue 组合：

```text
s_aa    = aatype embedding
s_h     = h_proj(H_context)
s_q     = q_proj(Q_target)
s_frame = frame_condition_proj([s_h, s_q])
s_0     = s_aa + s_h + s_q + s_frame
```

之后使用多层 simplified IPA-style blocks：

- content attention over residue features；
- anchor CA distance 作为 geometric bias；
- contact graph 以 8 Å CA cutoff 构造；
- residual + LayerNorm + transition MLP；
- 当前实现的 IPA 是简化版，不是 OpenFold full Evoformer/IPA module。

### 6.5 Frame update

`FrameUpdate` 从最终 single representation 输出：

- quaternion `dq`，归一化后构成 `R_update`；
- local translation `dt_local`，经过 `tanh` 和 `max_translation_update` bound。

默认 frame translation final layer zero-init，以保证初始化时 decoder 接近 anchor，避免随机 translation 在早期把 FAPE 推到 clamp 饱和区。

### 6.6 显式 Q→translation 路径

为了避免 H/IPA shortcut 完全绕过 Q，decoder 有一条独立的 Q-only head：

```text
q_translation_update_local
  = max_q_translation_update * tanh(q_translation_head(Q))
```

正常 Phase 2：

```text
dt_local = dt_from_H_IPA + q_translation_update_local
```

Phase 1 Q-only warmup：

```text
dt_local = q_translation_update_local
```

这样 Phase 1 强迫 Q 解释 anchor-relative CA displacement，避免 H/IPA 用 anchor geometry 走 identity mapping 而让 Q 完全不被使用。

`q_position_head` 目前为了 checkpoint compatibility 保留，但 canonical forward 不再把它作为 COM shortcut 使用；global COM displacement 在相对位置目标下是 gauge degree of freedom，不应被当作可识别监督信号。

### 6.7 Torsion head

TorsionHead 对每个 residue 输出 7 个角的 binned logits：

```text
[phi, psi, omega, chi1, chi2, chi3, chi4]
```

canonical 默认 36 bins，即约 10° 一个 bin。初始化 bias 使用一个 Ramachandran-aware prior：

- phi 约在 -65° 附近；
- psi 约在 -45° 附近；
- omega 偏向 trans；
- chi 默认较平坦。

训练时用 target backbone 计算 phi/psi/omega，并用 binned cross entropy。decoder 输出时用 softmax expectation 得到连续角度。

注意：Stage C 训练代码中的 `_compute_target_torsions` 是当前仍保留的手写 backbone dihedral helper；它只用于从 backbone target 计算训练监督，不负责 atom placement。真正的 frame→atom37 几何已经切换到 OpenFold。

### 6.8 训练中使用的结构 loss

`stage_c_total_loss` 的默认组件：

```text
L = w_fape * L_fape
  + w_torsion * L_torsion
  + w_distance * L_CA_pair_distance
  + w_bond * L_bond_diagnostic
  + w_clash * L_clash_diagnostic
  + w_ca * L_CA_relative
```

实际 canonical 训练还会额外加：

```text
L_q_translation
L_direct_Q_MSE
L_q_shuffle_margin (optional)
```

#### FAPE

如果提供 predicted/target frames，FAPE 在各 residue local frame 中比较 backbone points，并 clamp 到默认 10 Å。它不要求预测的 absolute global translation 与 target 相同。

无 frames 时的 fallback 是 CA-centered coordinate error，但 canonical Stage C training 会提供 frames 和 target frames。

#### Torsion CE

将 target angle 映射到 `[-π,π)` 的 bin，再对 7 个 torsion 做 mask-aware CE。

#### CA pairwise distance

对 predicted/target CA 做 `torch.cdist`，忽略 diagonal，默认将距离 clamp 到 30 Å，再计算 absolute distance error。它是内部几何 loss，与 global rigid transform 无关。

#### Anchor-relative CA loss

`direct_ca_loss` 使用同一个 anchor：

```text
pred_rel   = pred_CA - anchor_CA
target_rel = target_CA - anchor_CA
```

随后分别减去 valid residue 的 mean，去除全局 COM gauge；如果提供 anchor frame，则转到 anchor local coordinates，再计算 residue-wise squared error。与旧的“pred 自己居中、target 自己居中后比较”相比，共享 anchor 保留了相对 deformation 方向。

#### Bond/clash

当前 training 中 bond/clash 主要作为 detached diagnostics，不把它们的病态 normalize/project gradient 传入共享 decoder。bond ideal lengths 来自 OpenFold stereo chemical properties；side-chain clash 的 canonical standalone metric 使用 OpenFold official clash rule。

### 6.9 Q translation supervision

对于 valid target residues，训练构造：

```text
target_disp_global = target_CA - anchor_CA
target_disp_global -= valid_residue_mean(target_disp_global)
target_disp_local  = R_anchor^T @ target_disp_global
```

然后用 Smooth L1 监督 `q_translation_update_local`。target 会按 `q_translation_target_scale` 缩放；这个 scale 是数值 conditioning，不改变 metric 的 Å 单位。

这项 supervision 是当前解决“decoder 只用 H/anchor、不用 Q”的核心机制之一。

### 6.10 Q shuffle anti-bypass loss

如果启用 `q_shuffle_weight`，将 Q 沿 residue 维 roll 一格：

```text
Q_correct = Q_oracle
Q_shuffle = roll(Q_oracle, +1, residue_dim)
```

要求正确 Q 对 target local displacement 的误差小于 shuffled Q 至少 margin：

```text
L_shuffle = relu(margin + error_correct - error_shuffle)
```

它不是主结构 reconstruction loss，而是一个 semantic anti-bypass constraint，用来检测模型是否真的使用 residue-specific Q。

### 6.11 Stage C 两阶段训练

当前 q-aware config 的逻辑：

#### Phase 1：Q-only teacher-forced warmup

- 使用 oracle future Q；
- 只允许 Q-only translation path；
- 对 Q translation 做强监督；
- 可加 Q-vs-shuffled-Q margin；
- 目标是先让显式 Q path 学会与结构 displacement 对齐。

#### Phase 2：joint decoder

- 恢复 H/IPA translation residual；
- 使用 predicted Q 或 oracle/predicted blend；
- 保留 Q translation loss；
- 可继续使用 qshuffle anti-bypass；
- 让正常 direct/autonomous 输入分布逐步接管。

`stage_c_q_only_steps` 控制 Phase 1 的 global step 数。若 `q_weight_ramp_steps>0`，则 oracle→predicted Q 按 global step 线性 ramp；canonical v1 例如使用 3000-step ramp。

### 6.12 Stage C fine-tune 选项

canonical 默认冻结 Stage A/B representation；但配置可以选择：

- `stage_c_finetune_spatial_encoder=true`；
- `stage_c_finetune_temporal_encoder=true`。

即使 fine-tune，也只开放这两个 representation encoder，Stage B projection、energy、bath transition 等仍然冻结。encoder 使用更小 learning-rate scale（例如 0.05）。

如果有任何 Stage B parameter trainable，Stage C checkpoint 会嵌入完整 `stage_b_model` state；evaluation 必须优先使用 embedded Stage B，而不能重新加载旧 external Stage B。否则训练和 evaluation 的 H/Q representation 会不一致。

### 6.13 Stage C 训练 loop

每个 paired batch：

1. 读取 context/future 最后一帧作为 anchor/target；
2. 从 context encode `H`；
3. 从 future encode `Q_oracle`；
4. direct head 从 H、horizon、temperature 预测 `Q_direct`；
5. 根据 phase/ramp 选择 `q_decode`；
6. StructureDecoder 输出 frames/torsions/atom37；
7. 计算结构 loss、Q translation loss、direct Q MSE、可选 qshuffle loss；
8. gradient accumulation；
9. gradient clipping；
10. decoder 及可选 representation encoder optimizer step；
11. JSONL logging 和 checkpoint。

### 6.14 Stage C checkpoint

Stage C checkpoint 至少包括：

- `model`：decoder + direct head state；
- `optimizer`；
- `epoch`/`step`；
- 完整 config；
- fine-tune 时的 `stage_b_model`。

旧 checkpoint 可能没有 `q_translation_head`。兼容加载器只对这一组 zero-init missing keys 补齐，其他 decoder/direct-head key missing 会直接报错，避免部分 checkpoint 被误当作完整模型。

---

## 7. OpenFold atom37 解码链路

这是当前结构生成最需要精确理解的部分。

### 7.1 为什么要使用 OpenFold

atom37 reconstruction 涉及：

- residue frame convention；
- backbone/default rigid group frames；
- torsion angle 到 sidechain frames；
- residue-specific atom14 layout；
- atom14 到 atom37 lookup；
- missing atom mask；
- PDB ordering。

手写这些表和 side-chain tree 很容易在 tail atom ordering、chi slot、残基类型 mapping 或 frame convention 上出错。当前 canonical path 尽可能调用 vendored official OpenFold：

```python
make_atom14_masks
torsion_angles_to_frames
frames_and_literature_positions_to_atom14_pos
atom14_to_atom37
residue_constants.restype_atom37_mask
Rigid / Rotation
openfold.np.protein.to_pdb
```

### 7.2 Stage C frame 到 OpenFold frame

项目的 `RigidBodyFrame.from_backbone` 与 OpenFold `Rigid.make_transform_from_reference` 的 residue frame 轴定义不同，但二者由同一 N/CA/C geometry 决定。当前 adapter 使用固定旋转矩阵：

```text
DECODER_TO_OPENFOLD_ROT =
[[ 1, 0, 0],
 [ 0, 0, 1],
 [ 0,-1, 0]]
```

具体转换：

```text
R_openfold = R_decoder @ DECODER_TO_OPENFOLD_ROT
```

translation `t` 保持为 CA origin。这个固定 mapping 已通过真实结构 frame 对齐测试，误差达到数值精度级别。

### 7.3 Stage C torsion 到 OpenFold alpha

Stage C 内部顺序：

```text
[phi, psi, omega, chi1, chi2, chi3, chi4]
```

OpenFold `torsion_angles_to_frames` 需要 canonical AlphaFold slot：

```text
[pre_omega, phi, psi, chi1, chi2, chi3, chi4]
```

因此 adapter 先重排：

```text
pre_omega <- omega
phi       <- phi
psi       <- psi
chi1..4   <- chi1..4
```

再转成：

```text
alpha[...,i,:] = [sin(angle_i), cos(angle_i)]
```

### 7.4 官方 atom14/atom37 pipeline

给定 OpenFold backbone `Rigid`、alpha torsions 和 aatype：

```text
Rigid backbone + alpha
        |
        v
torsion_angles_to_frames
        |
        v
all residue rigid-group frames
        |
        v
frames_and_literature_positions_to_atom14_pos
        |
        v
atom14 coordinates
        |
        +-- make_atom14_masks
        |
        v
atom14_to_atom37
        |
        v
atom37 coordinates + atom37_atom_exists mask
```

当前 canonical `StructureDecoder.forward` 的 `atom37_coords` 就是这个 output；`backbone_coords` 从 OpenFold atom37 的 N/CA/C/O index 提取，而不是从旧手写 sidechain tree 生成。

### 7.5 仍保留但非 canonical 的手写代码

`structure_decoder.py` 仍有：

- `IDEAL_BOND`/`IDEAL_ANGLE` 常量；
- `reconstruct_backbone_from_torsions`；
- `_rodrigues` 等函数。

它们现在只用于 legacy/diagnostic torsion-chain-extension path。canonical frame→atom37 输出不使用它们。文档和 evaluation 必须引用 `atom37_coords`，不能把 `backbone_torsion_reconstructed` 当作官方 decoder output。

### 7.6 PDB export

`stage_c_evaluator.py` 将 OpenFold atom37 arrays 封装成 `openfold.np.protein.Protein`，调用官方 `protein.to_pdb`。因此：

- 不手写 atom names；
- 不手写 residue formatting；
- 不手写 chain/PDB column；
- 输入必须显式标记是 OpenFold ordering 还是 project ordering；
- 多 sample ensemble 使用 `MODEL/ENDMDL` 包装。

---

## 8. Stage C 的三种 Q 来源

### 8.1 Oracle

```text
future coordinates
    -> Stage B encode_state(use_temporal=True)
    -> Q_oracle
    -> shared decoder(H_context, Q_oracle, anchor)
```

Oracle 不是部署模式，而是 decoder 上限/诊断模式。若 oracle 都不超过 persistence，主要问题在 decoder、loss、anchor 或表示的信息充分性，而不是 direct head 或 autonomous transition。

### 8.2 Direct endpoint

`DirectEndpointHead` 从 context H 显式接收：

- H history；
- optional static condition；
- `tau_ps`；
- temperature；
- residue mask。

它输出 per-residue Gaussian 参数：

```text
mean      [B,L,D_Q]
log_sigma [B,L,D_Q]
```

mean 和 log_sigma 都由 shared per-residue MLP 生成。time 经过 log-scale embedding，temperature 经过单独 embedding。训练中 direct head 以 `Q_oracle` 做 MSE：

```text
L_direct_Q = MSE(Q_direct_mean, Q_oracle)
```

evaluation 的 deterministic direct mode 使用 mean，不采样；如要研究 distribution，使用 `sample()` 生成 `[B,S,L,D]`。

### 8.3 Autonomous recursive

```text
context H/Q + bath S0 + anchor CA graph
    -> initialize once
    -> repeated Stage B transition
    -> Q_recursive(horizon)
    -> shared decoder
```

关键 invariant：

- history encoder 只运行一次；
- bath 只初始化一次，不刷新；
- rollout 不读 future H 或 future coordinates；
- horizon 不能整除 `delta_t_ps` 时，执行 remainder step；
- training rollout 常用 deterministic；
- canonical world model evaluation 可以 stochastic=True，并对多个 seeds 报告。

---

## 9. Evaluation 总体协议

### 9.1 Evaluation 的层级

当前 evaluation 不是单个脚本，而是若干互补层：

```text
Stage A held-out latent endpoint skill
Stage B one-step/dynamics/rollout/temperature/intervention
Stage C single-structure oracle/direct/autonomous/persistence/qshuffle
Stage C atom37/side-chain/clash validation
AlphaFlow ensemble metrics
baselines and ablations
```

### 9.2 Train/val/test 分离

training 使用 train manifest；标准评估使用 val/test manifest。不能因为某个 checkpoint 在 train paired window 上的 loss 较低就称它超过 persistence。至少要报告：

- split；
- protein/domain 数；
- trajectory/replica 数；
- horizon；
- random seeds；
- checkpoint path；
- metric protocol；
- failed sample count。

### 9.3 checkpoint pairing

Stage C 评估必须使用与其训练一致的 Stage B representation：

1. 如果 Stage C checkpoint 内嵌 `stage_b_model`，优先使用 embedded state；
2. 否则加载显式 `--stage-b-ckpt`；
3. legacy `context_encoder.*` 自动 remap 到 `spatial_encoder.*`；
4. legacy energy graph 会根据 saved layer shapes 推断 condition channel 和 temperature feature；
5. 无法严格匹配时直接失败。

这样避免“训练用 fine-tuned encoder，evaluation 却加载原始 Stage B”的 silent mismatch。

---

## 10. Canonical world-model evaluation

主脚本是 [`scripts/eval_world_model.py`](../scripts/eval_world_model.py)。它对每个 held-out sample、horizon 和 seed 生成 raw per-record 结果。

### 10.1 五个 mode

#### `oracle`

Future target window 通过与 context 相同的 `encode_state(use_temporal=True)` 得到 `Q_oracle`，再 decoder。测量的是 decoder 在获得正确 dynamics code 时的能力。

#### `direct`

context H 进入 DirectEndpointHead，显式给入物理 horizon 和 temperature，取 predicted Q mean，再 decoder。测量 endpoint predictor + decoder。

#### `autonomous`

context 得到 Q0/S0/CA graph，按 base step 做 closed-loop Stage B transition，得到 horizon Q，再 decoder。对每个 seed 运行 stochastic transition。测量完整 generative path 的结构质量和 divergence。

#### `persistence`

直接把 anchor backbone 与 future target 比较，不经过 decoder。它是最重要的 baseline：任何模型模式若不能在关键 metric 上超过它，就不能称为有效预测器。

#### `qshuffle`

使用 oracle Q，但沿 residue 维 roll 一格，再 decoder。它测量 decoder 是否使用 residue-specific Q。若 oracle 和 qshuffle 几乎相同，说明 Q 通路被忽略或 decoder 对 Q 不敏感。

### 10.2 有效 residue mask

评估 mask 是：

```text
eval_mask = last_context_residue_mask & last_future_residue_mask
```

不能只使用 future mask，否则模型可能因 anchor 缺失/被 padding 的 residue 得到不合理分数。

### 10.3 autonomous seed 处理

endpoint modes 本身是 deterministic mean/decoded output，autonomous mode 由 seeds 产生多个 samples。为了减少重复计算，脚本把 autonomous seeds batch 化运行 transition；oracle/direct/persistence/qshuffle 的 metric 在每个 horizon 只算一次，然后在 raw report 中按 seed 复制记录，以保持统一的 `(sample, seed, horizon)` 表结构。

### 10.4 fail-fast

world-model evaluation 的完整性条件：

- 每个 sample 必须处理；
- sample 内异常、NaN、Inf 坐标直接记录为 failure；
- 失败 sample 不能被写成 passed record；
- 五个 required modes 都必须有记录；
- `n_records = n_samples × n_seeds`；
- 任何 metric NaN/Inf 都会使 evaluation 失败；
- 数据集比 requested sample 数短会失败；
- `--skip-autonomous` 时不能把结果伪装成包含 autonomous 的完整五模式报告。

脚本只有在上述检查通过后才写 `full_report.json` 的 `status: passed`。

### 10.5 单结构指标

canonical `compute_structure_metrics` 返回：

- `rmsd_kabsch`；
- `kabsch_ca_point_error`；
- `dist_mae`；
- `contact_jaccard`；
- `weak_contact_j`；
- `transient_contact_j`；
- `n_valid_residues`；
- `n_nonlocal_pairs`。

这些指标按 valid residue intersection 计算，不能把失败样本当作 0。

---

## 11. Metric 的数学定义

### 11.1 CA Kabsch RMSD

给定 valid residue 的 predicted CA `P` 和 target CA `Q`，寻找最优 rigid transform `(R,t)`：

```text
min_{R in SO(3), t} || P R + t - Q ||_F^2
```

返回对齐后的 CA RMSD。实现使用 OpenFold 的 `superimpose` path，并使用 OpenFold 内部同源的 Biopython `SVDSuperimposer` 应用 transform。

单位是 Å。

### 11.2 CA-fit 后 backbone point error

Kabsch 只在 CA 上拟合一次，随后将同一个 rigid transform 应用到 `[N,CA,C,O]` 全部原子。然后：

```text
mean_{residue, atom in N/CA/C/O} ||pred_aligned - target||_2
```

它与 CA RMSD 不同：CA RMSD 只看 CA，point error 看全 backbone。

### 11.3 非局部 CA pair distance MAE

对 valid CA pair：

```text
d_pred(i,j) = ||CA_pred_i - CA_pred_j||
d_true(i,j) = ||CA_true_i - CA_true_j||
```

只对 off-diagonal、非局部 pair 计算：

- 同链且 `|residue_index_i - residue_index_j| > 3`；
- 跨链 pair 无论 residue number 差异都可纳入。

```text
dist_mae = mean |d_pred - d_true|
```

### 11.4 Contact Jaccard

对相同 nonlocal pair mask，定义 contact bins：

```text
strong contact:  [0, 8) Å
weak contact:    [8, 15) Å
transient:       [15, 25) Å
```

对 predicted set `A` 和 target set `B`：

```text
J(A,B) = |A ∩ B| / |A ∪ B|
```

如果 union 为空，canonical convention 是 `J=1`，因为两个空集合完全相同。旧的空 union 返回 0 会让“无 contact”蛋白被不合理惩罚，已被修正。

### 11.5 不报告的 RMWD

RMWD 是 ensemble distribution metric，不是单结构 Kabsch RMSD 的别名。single-structure world-model report 不报告 RMWD；要得到 RMWD、RMSF correlation、PCA W2 等，必须运行 AlphaFlow ensemble protocol。

---

## 12. Stage C ensemble evaluation 和 AlphaFlow

### 12.1 何时需要 ensemble

如果问题是“生成的结构分布是否覆盖 MD ensemble”，单个 target endpoint 不足以回答。需要：

- 多个 predicted samples；
- 对应的 reference MD ensemble；
- 相同 atom set/order；
- 相同 ensemble size protocol；
- official ensemble metric implementation。

### 12.2 OpenFold PDB export

`stage_c_evaluator.py` 的 `_atom37_to_pdb_lines`：

1. 按 `atom37_order` 判断输入是 OpenFold 还是 project order；
2. 必要时使用 audited permutation 转到 OpenFold order；
3. 构造 official `Protein`；
4. 调用 `openfold.np.protein.to_pdb`；
5. 外层加入 MODEL/ENDMDL。

reference ensemble 也必须经过同样的 export path，确保 parser/atom ordering 一致。

### 12.3 MDTraj RMSD

MDTraj adapter 要求版本不低于 1.9.9。MDTraj 返回 nm，adapter 转成 Å。

预测 PDB 和 reference PDB 的 atom index 不假设相同；因为 missing atom mask 会使写出的 atom 数量不同，adapter 按以下 identity 匹配共同 atom：

```text
(atom name, chain index, residue sequence number, insertion code)
```

共同 atom 少于 3 个直接失败。

### 12.4 AlphaFlow adapter

AlphaFlow adapter 只做：

- PDB export；
- 调用固定版本脚本；
- 解析输出；
- 保存 stdout/stderr/provenance。

它不在本仓库重新实现 AlphaFlow 的 RMSF、W2、PCA、contact math。当前锁定记录为：

```text
repo: bjing2016/alphaflow
commit: 0408d7c89dac444a43a9089d7427ce470b0a5e67
```

输出字段：

- Per-target RMSF r；
- RMWD；
- RMWD variance；
- MD-PCA W2；
- weak contacts J；
- transient contacts J。

### 12.5 Ensemble size invariant

如果指定 `ensemble_size`：

- prediction PDB model 数必须匹配；
- reference ensemble 读取出的总 frame 数必须匹配；
- 不匹配直接失败。

不能用一帧 target 冒充 reference ensemble。

### 12.6 status 和 aggregate

`evaluate_stage_c` 结果有明确 status，例如：

- `ok`；
- `mdtraj_failed`；
- `alphaflow_failed`；
- `alphaflow_skipped`；
- `metrics_incomplete`。

`aggregate_metrics` 默认只纳入 `status == "ok"` 的 protein，并同时报告：

- total proteins；
- valid proteins；
- excluded proteins；
- exclusion reason；
- mean/std；
- protein-level bootstrap confidence interval。

这和 canonical world-model evaluation 的 strict fail-fast 有细微区别：

- world-model raw endpoint report 要求全体样本完整，否则整个 report fail；
- AlphaFlow aggregate 可以在 per-protein 层面排除失败，但必须公开排除数量和 status，不能把剩余样本包装成无条件完整结果。

---

## 13. Side-chain、bond 和 clash evaluation

### 13.1 当前数据的限制

mdCATH 主 pipeline 当前从 HDF5 读取 backbone `[N,CA,C,O]`。因此：

- decoder 可以根据 OpenFold geometry 生成 atom37；
- 但 target manifest 未必包含可以与 predicted sidechains 对齐的 native atom37 target；
- 没有 native atom37 时，不能诚实地报告真实 all-heavy-atom RMSD。

`eval_sidechain_quality.py` 在这种 backbone-only 情况下会把 atom37 RMSD 标记为 unavailable/NaN，并报告可从 backbone target 推导的 torsion error 和 predicted clash diagnostics。不能把 decoder-vs-decoder 或 synthetic reconstruction 数字当作 native side-chain accuracy。

### 13.2 Official clash metric

旧的 `distance < 2 Å` all-pairs 实现会把正常 covalent/peptide geometry 也当成 clash。当前 `compute_clashes` 使用 OpenFold：

```python
make_atom14_masks
between_residue_clash_loss
residue_constants.van_der_waals_radius
```

并遵守 official peptide/disulfide exception 与 residue index 逻辑。每个 clash endpoint 在 per-atom mask 中计一次，最终除以 2 得 pair count。

### 13.3 Bond metric

training diagnostic 的 backbone bond ideal length 来自 OpenFold `stereo_chemical_props.txt`。当前被检查的 backbone pairs 包括：

```text
N-CA, CA-C, C-O
```

在 canonical Stage C 中 bond/clash 主要用于质量检查，不应被误读成 decoder 已经拥有完整可训练 side-chain force field。

---

## 14. Ablations 和 baselines

### 14.1 Structure decoder ablations

`models/stage_c/ablations.py` 提供：

- `QZeroDecoder`：将 Q 置零；
- `QShuffleDecoder`：打乱 residue-specific Q；
- `HZeroDecoder`：将 H 置零；
- `QNoiseDecoder`：向 Q 加噪声；
- `QRandomDecoder`：替换为随机 Q。

这些 ablations 用于回答：

- H 是否承担所有结构信息？
- Q 是否提供增量运动信息？
- decoder 是否真的使用 residue-specific Q？
- Q 的信息是稳定的还是只在特定 scale 上有效？

### 14.2 Persistence baseline

最基本 baseline 是最后 context frame 原样输出。它不是一个弱到可以忽略的 baseline：在短时间 horizon 或高相关 MD trajectory 中，persistence 往往很强。所有 Stage C claim 都必须至少和 persistence 比较。

### 14.3 Stage B baselines

`eval/baselines.py` 中保留：

- latent persistence predictor；
- causal Transformer world model baseline；
- conditional flow-matching predictor。

这些 baseline 主要用于 Q-space endpoint/rollout 对比，不等价于结构 decoder 的 baseline。

### 14.4 Rollout quality

`eval_rollout_quality.py` 关注 Stage B autonomous dynamics 本身，通常记录：

- 多 horizon latent norm/variance；
- 与 target latent 的误差；
- divergence rate；
- autocorrelation/ACF；
- RBF MMD 等 path/statistics diagnostics。

它回答的是“Q transition 是否稳定、是否保留动态统计”，不能直接代替 Stage C atom coordinate evaluation。

### 14.5 Temperature intervention

`eval_temperature_intervention.py` 在相同 context 或不同 temperature condition 下运行 transition/structure comparison，检查：

- temperature 是否实际影响 diffusion/noise；
- energy/force 是否对 temperature 有响应；
- intervention response 是否大于 seed noise；
- response direction 是否合理。

它是 conditional dynamics diagnostic，而不是 equilibrium free-energy proof。

### 14.6 Evaluation 脚本的完整分工

为了避免“跑了一个脚本就认为完成了 evaluation”，当前脚本可以按以下矩阵理解：

| 脚本 | 主要输入 | 主要输出 | 能回答的问题 |
|---|---|---|---|
| `eval_stage_a_heldout.py` | Stage A + held-out paired windows | latent MSE/skill | Stage A endpoint latent 是否超过 persistence |
| `eval_world_model.py` | Stage B + Stage C + val manifest | oracle/direct/autonomous/persistence/qshuffle | 五模式单结构 endpoint 对比 |
| `eval_stage_c_heldout.py` | Stage C + held-out data | canonical single-structure metrics | decoder 在 held-out endpoint 的几何质量 |
| `eval_stage_c_endtoend.py` | DirectHead + decoder | oracle/direct endpoint metrics | direct Q-to-structure path 是否工作 |
| `eval_stage_c_full_metrics.py` | Stage C + canonical adapter | RMSD/contact/distance summary | 不混入 ensemble 指标的完整单结构表 |
| `eval_ablations.py` | Stage C + ablation wrappers | H/Q zero/shuffle/noise/random comparison | decoder 的信息依赖和 shortcut |
| `eval_rollout_quality.py` | Stage B | latent rollout statistics | autonomous Q 是否稳定、是否漂移/发散 |
| `eval_full_stage_b.py` | Stage B | baselines、rollout、temperature、sidechain/world-model 子目录 | Stage B 诊断的一站式旧/兼容 runner |
| `eval_baselines.py` | Stage B representation + baseline models | baseline comparison | LaSP 与简单预测器的比较 |
| `eval_temperature_intervention.py` | Stage B + temperature conditions | intervention metrics | 温度条件是否真的改变 transition |
| `eval_sidechain_quality.py` | Stage C atom37 + backbone targets | OpenFold clash/torsion diagnostics | atom37 geometry 是否有效；native sidechain accuracy 是否可评估 |
| `stage_c_evaluator.py` | predicted/reference atom37 ensembles | MDTraj + AlphaFlow + bootstrap | distribution-level ensemble quality |

`run_full_abc_evaluation.sh` 是 orchestration wrapper。它会等待 checkpoint、检查 `canonical_state_version.txt`，依次运行 world model、rollout、baselines、temperature 和 sidechain evaluation，并在子任务失败时以非零 exit code 结束。它并不会把每个子任务的 success 自动合并成科学结论；最终必须查看各自的 raw records、status 和 protocol metadata。

### 14.7 Generic dynamics metrics 的边界

`protein_jepa_frame/eval/dynamics_benchmarks.py` 还提供与具体模型无关的动力学 benchmark primitives，包括：

- multiclass state forecast NLL、Brier、accuracy、balanced accuracy、macro-F1、ECE；
- binary event log loss、Brier、accuracy、ECE；
- trajectory-safe transition counts；
- row-normalized transition operator；
- stationary distribution；
- implied timescales；
- MFPT；
- row total variation、row JS divergence、stationary L1；
- Chapman-Kolmogorov/semigroup defect；
- integrated Brier score 和 survival curve error；
- path-feature RBF MMD；
- path-level transition/failure statistics。

这些函数服务于更广泛的 kinetic/world-model experiments。它们不自动被 `eval_world_model.py` 的单结构 summary 调用；如果在论文或最终报告中使用，必须另行记录 state definition、operator discretization、lag time、censoring rule 和 sample count。

---

## 15. 历史问题、已修复问题和为什么要记录它们

### 15.1 SE(3)-invariant loss 与绝对平移的矛盾

旧版本让 pred 和 target 各自独立 center，再比较坐标。这样 global translation 的梯度确实被消掉；如果 decoder 的任务被误写成预测 absolute position，就会出现“模型学 torsion、不学 position”。

当前正确解释是：蛋白 global translation 是 gauge，不应作为结构物理目标；应训练 shared-anchor-relative deformation，并用 anchor frame 去除任意 global gauge、保留内部 motion。

### 15.2 Q path 被 H/anchor shortcut 绕过

旧版 decoder 的 `q_proj(Q)` 虽然参与 representation，但 H/anchor-conditioned IPA 已经可以输出 identity/anchor-like structure，Q 的梯度和实际贡献很弱。表现为：

```text
oracle ≈ qshuffle ≈ persistence
```

当前修复方向包括：

- Phase 1 Q-only translation；
- explicit Q-only translation head；
- Q translation local displacement supervision；
- optional qshuffle margin；
- Phase 2 再恢复 H+Q joint residual；
- 可选 Stage C representation fine-tune。

### 15.3 B context Q 与 future Q 的状态空间不一致

如果 context Q 用 temporal history，而 future Q 只用 single-frame spatial encoding，那么 NLL/rollout/decoder 会比较两个不同编码器定义下的向量。当前所有 canonical path 都调用：

```text
encode_state(..., use_temporal=True)
```

包括：

- Stage B context；
- Stage B future target；
- Stage B rollout target；
- Stage C context H/Q；
- Stage C oracle Q；
- world-model evaluation oracle Q。

### 15.4 C training/evaluation H path 不一致

旧版 training 可能直接调用 `forward_spatial`，evaluation 却加入 temporal H；或者相反。当前 C training 和 eval 都由 `encode_state` 产生 `h_last`，以保证 H 的语义一致。

### 15.5 手写 atom37 ordering 错位

旧版手写 atom37 table 在尾部 `CZ/NH1/NH2/OH` 等顺序上容易错位。当前 OpenFold adapter：

- lookup 由 `make_atom14_masks` 提供；
- atom placement 由 official rigid group functions 提供；
- PDB formatting 由 official Protein formatter 提供；
- project/OpenFold ordering 转换显式、可测试。

### 15.6 evaluation 吞失败

旧版 evaluator 可能在某些 mode 失败后仍输出含有 NaN/0 的表面完整 report。当前 canonical world-model evaluator 在 sample/mode/record/finite metric 层级 fail-fast；AlphaFlow evaluator 在 protein status 和 aggregate metadata 中显式记录失败。

### 15.7 全量 pytest 和无关 dirty worktree

当前工作区包含大量历史文件删除和未提交的 framework changes。相关 OpenFold/structure/evaluation tests 已通过，但全量 pytest 可能因为历史删除文件造成 collection errors。这些 collection errors 不能简单解释成当前 ABC core 的 geometry/evaluation failure；运行报告时要区分：

- 本次相关测试；
- 全仓库 collection 状态；
- 与当前改动无关的历史文件状态。

---

## 16. 推荐训练流程

### 16.1 Stage A

```bash
python -m protein_jepa_frame.cli train --stage A \
  --config configs/full_abc_stage_a_from_scratch.yaml
```

训练前检查：

- train manifest 路径存在；
- train/val/test domain 不重叠；
- real paired-window protocol audit 通过；
- time_ps 是物理时间而不是 raw index；
- EMA target keys 正常保存。

### 16.2 Stage B canonical 2760 steps

```bash
python -m protein_jepa_frame.cli train --stage B \
  --config configs/full_abc_stage_b_canonical_v1.yaml
```

重点确认：

- `init_checkpoint` 指向 Stage A checkpoint；
- stage B output 有 `canonical_state_version.txt`；
- 最终 `step` 精确为 2760；
- no NaN/Inf；
- force/energy/diffusion/rollout diagnostics 有记录。

### 16.3 Stage C canonical

```bash
python -m protein_jepa_frame.cli train --stage C \
  --config configs/full_abc_stage_c_canonical_v1.yaml
```

如果目标是让 decoder 强制使用 Q，使用 q-aware two-phase 配置：

```bash
python -m protein_jepa_frame.cli train --stage C \
  --config configs/full_abc_stage_c_qaware_finetune_v2.yaml
```

如果同时开放 representation encoder，必须确认 checkpoint 嵌入了 adapted Stage B state。

### 16.4 训练中断恢复

Stage B 使用 `resume_checkpoint` 恢复 model/optimizer/global step。Stage C resume 需要同时有：

- `model`；
- `optimizer`；
- compatible decoder/direct-head keys；
- fine-tune run 则还需要 `stage_b_model`。

不要把 `stage_c_pretrained` 和 `resume_checkpoint` 混为同一语义：

- `stage_c_pretrained` 是 warm-start decoder/direct head；
- `resume_checkpoint` 是继续同一优化轨迹。

---

## 17. 推荐 evaluation 流程

### 17.1 World model 五模式

```bash
python -u scripts/eval_world_model.py \
  --stage-c-ckpt artifacts/full_abc_stage_c_canonical_v1/checkpoint_latest.pt \
  --stage-b-ckpt artifacts/full_abc_stage_b_canonical_v1/checkpoint_latest.pt \
  --val-manifest artifacts/mdcath_manifest_200dom/val.jsonl \
  --output-dir artifacts/world_model_eval \
  --n-val-samples 200 \
  --seeds 42 123 456 \
  --base-step-ps 100 \
  --max-substep-dt-ps 100
```

先跑小规模 smoke，再跑完整 held-out。smoke report 的用途是检查：

- checkpoint 是否能加载；
- atom37 是否 finite；
- mode 是否全部生成；
- metrics schema 是否完整；
- autonomous 是否 diverge。

smoke 通过不代表 quality 通过。

### 17.2 Stage C full single-structure metrics

使用 [`scripts/eval_stage_c_full_metrics.py`](../scripts/eval_stage_c_full_metrics.py) 得到 canonical single-target metrics。该脚本明确不合成 RMWD/RMSF/PCA。

### 17.3 Held-out 和 end-to-end

- `eval_stage_c_heldout.py`：held-out structure quality；
- `eval_stage_c_endtoend.py`：DirectHead Q → decoder 的 end-to-end；
- `eval_ablations.py`：H/Q ablations；
- `eval_rollout_quality.py`：Stage B autonomous Q rollout；
- `eval_temperature_intervention.py`：temperature response；
- `eval_sidechain_quality.py`：OpenFold atom37/clash/torsion diagnostics。

### 17.4 AlphaFlow ensemble evaluation

只有在以下条件满足时才运行并解释 AlphaFlow：

- prediction samples 数量明确；
- reference MD ensemble 真实存在；
- predicted/reference atom order 已统一；
- MDTraj 和 AlphaFlow environment/version 已锁定；
- per-protein status 全部保存；
- aggregate 报告 valid/excluded 数目。

没有 reference ensemble 时，只能报告 single-structure metrics。

---

## 18. 如何判断模型是否真的“超过 persistence”

### 18.1 最低判断标准

至少在 held-out protein 上，针对同一 horizon、同一 residue mask、同一 metric protocol：

```text
model metric better than persistence metric
```

对于 RMSD、point error、dist MAE，越低越好；对于 Jaccard，越高越好。

不能用：

- train loss 对 persistence loss 的比较；
- oracle 对 direct 的比较；
- 不同 mask 的比较；
- 不同 horizon 的平均数混在一起；
- 把失败样本从模型分子中删除而保留 persistence；
- 把 Kabsch RMSD 称为 RMWD。

### 18.2 mode-specific interpretation

| 现象 | 主要说明 |
|---|---|
| oracle 不超过 persistence | decoder/representation/anchor-relative loss 瓶颈 |
| oracle 超过，direct 不超 | DirectEndpointHead 或 Q prediction 瓶颈 |
| direct 超过，autonomous 不超 | Stage B transition/rollout 稳定性瓶颈 |
| oracle≈qshuffle | decoder 忽略 residue-specific Q |
| decoder≈persistence 且 q translation norm≈0 | Q→structure path 未学到 |
| autonomous Q norm 随 horizon 爆炸 | energy/force/diffusion/integrator 不稳定 |
| single structure 好但 AlphaFlow 差 | ensemble diversity/coverage/path distribution 不对 |
| AlphaFlow 无结果 | evaluation protocol/environment 不完整，不是模型质量结论 |

### 18.3 绝对坐标和相对几何必须分开

Kabsch RMSD 会消除 global rigid transform；这不是缺陷，而是结构比较的标准做法。另一方面，训练时 direct CA loss 不能把 pred/target 各自独立 center 后再声称模型学到了 anchor-relative displacement。当前协议用 shared anchor + shared gauge removal，既不监督不可识别的 global translation，也不抹掉 residue-wise internal deformation。

---

## 19. 当前已知限制

### 19.1 Backbone-only training data

当前 mdCATH loader 主路径只抽取 N/CA/C/O。OpenFold atom37 decoder 可以生成 sidechains，但 target sidechain supervision 缺失。因此：

- torsion/sidechain generation 是 decoder prior + learned torsion head；
- native atom37 RMSD 不能从 backbone-only 数据得到；
- 需要完整 atom37 manifest 才能做真实 sidechain RMSD、chi accuracy 和 per-AA atom error。

### 19.2 Simplified Stage C IPA

当前 Stage C 的 `InvariantPointAttention` 是 simplified content attention + distance bias，使用 anchor positions，但不等于 OpenFold 完整 IPA/Evoformer。OpenFold 目前主要被用于 geometry reconstruction、rigid/frame algebra、atom mapping、clash 和 PDB formatting。

### 19.3 Stage B finite bath 是受控近似

`S` 是有限 memory closure，不是从原子 solvent coordinates 严格推导的 exact Mori memory。它的物理一致性需要通过 held-out statistics、temperature response、stationarity、semigroup 和 rollout diagnostics 检验。

### 19.4 Force floor 不是物理真值

force floor 防止 flat-energy degeneration，但过强会制造非真实 force magnitude。必须同时查看：

- force alignment；
- Q endpoint skill；
- energy scale；
- stationary/rollout statistics；
- temperature response。

### 19.5 自主 rollout 仍可能成本高

每个 transition substep 需要 energy gradient；autonomous evaluation 还要对多 seed、多个 horizon 重复。当前脚本已把 seeds batch 化，输出 ETA/progress，并减少 endpoint mode 的重复 decoder/metric pass，但 long horizon 仍然可能很慢。

### 19.6 单结构 metric 不等于动力学正确

一个预测结构经 Kabsch 后 RMSD 下降，只说明该 endpoint 的几何更接近 target；不能单独证明：

- transition law 正确；
- free-energy landscape 正确；
- path ensemble 正确；
- temperature intervention 正确；
- long-horizon kinetics 正确。

---

## 20. Tensor shape 总表

### 20.1 数据和 Stage A

| 对象 | shape |
|---|---|
| clip coordinates | `[B,T,L,4,3]` |
| clip residue mask | `[B,T,L]` |
| spatial H | `[B,T,L,D_H]` |
| temporal frame state | `[B,T,D_H]` |
| temporal clip state | `[B,D_H]` |
| residue JEPA prediction | `[B,T,L,D_H]` |
| paired endpoint | `[B,D_H]` |

### 20.2 Stage B

| 对象 | shape |
|---|---|
| H last | `[B,L,D_H]` |
| Q | `[B,L,8]` |
| G | `[B,8]` |
| S | `[B,L,16]` |
| CA distance graph | `[B,L,L]` |
| force | `[B,L,8]` |
| diffusion | `[B,L]` |
| environment e | `[B,8]` |

### 20.3 Stage C

| 对象 | shape |
|---|---|
| anchor R | `[B,L,3,3]` |
| anchor t/CA | `[B,L,3]` |
| predicted R | `[B,L,3,3]` |
| predicted t | `[B,L,3]` |
| torsion logits | `[B,L,7,36]` |
| torsion angles | `[B,L,7]` |
| backbone output | `[B,L,4,3]` |
| atom37 output | `[B,L,37,3]` |
| atom37 mask | `[B,L,37]` |

---

## 21. 重要文件和职责速查

| 文件 | 职责 |
|---|---|
| `config.py` | Data/Model/Optimization/Train dataclass 与 YAML 解析 |
| `data/manifest.py` | JSONL trajectory record |
| `data/dataset.py` | npz/HDF5/mdCATH lazy window dataset |
| `data/collate.py` | clip/paired batch namespace 和 padding |
| `models/encoder.py` | Stage A SE(3)-invariant spatial encoder |
| `models/temporal.py` | temporal Transformer/history encoder |
| `models/stage_a/model.py` | online/EMA JEPA + paired endpoint |
| `models/stage_b/model.py` | canonical `encode_state`, energy, force, transition |
| `models/stage_b/objectives.py` | NLL、force regularization、rollout、semigroup |
| `models/lasp/energy.py` | scalar PMF/energy graph |
| `models/stage_c/structure_decoder.py` | H+Q→frames/torsions/atom37 |
| `models/stage_c/openfold_atom37.py` | official OpenFold geometry adapter |
| `models/stage_c/direct_head.py` | H+time+temperature→direct Q distribution |
| `models/stage_c/rollout.py` | Stage C autonomous state rollout wrapper |
| `training/stage_a.py` | joint mask/paired training |
| `training/stage_b.py` | frozen representation + LaSP training |
| `training/stage_c.py` | decoder/Q-aware/fine-tune training |
| `eval/structure_metrics.py` | canonical single-structure metric implementation |
| `eval/stage_c_checkpoint.py` | checkpoint split/load/legacy compatibility |
| `eval/stage_c_evaluator.py` | PDB, MDTraj, AlphaFlow, bootstrap |
| `scripts/eval_world_model.py` | five-mode canonical endpoint evaluation |
| `scripts/eval_ablations.py` | H/Q bypass ablations |
| `scripts/eval_sidechain_quality.py` | official clash + torsion diagnostics |

---

## 22. 最终理解框架

把当前系统压缩成一组不应混淆的关系：

```text
Stage A:
    coordinates -> H
    learns representation and latent predictability

Stage B:
    H -> Q
    Q + CA graph -> A(Q)
    -grad A -> force
    force + diffusion + bath -> transition(Q,S)

Stage C:
    H_context + Q_target + anchor frame
        -> frame residual + torsions
        -> official OpenFold atom37

Evaluation:
    oracle       isolates decoder ceiling
    direct       isolates endpoint Q prediction + decoder
    autonomous   tests full closed-loop world model
    persistence  tests whether doing nothing is already better
    qshuffle     tests whether decoder uses residue-specific Q
```

最重要的科学解释纪律是：

1. **不要把 absolute global translation 当作蛋白结构学习目标。** 结构比较应尊重 SE(3) gauge；但训练 loss 仍要使用 shared anchor 保留真实 relative deformation。
2. **不要把 Q 仅仅当作 decoder 的一个普通 embedding。** 如果没有显式 Q path 和 anti-bypass test，H/anchor 很容易让模型退化为 persistence。
3. **不要让 context Q 和 future Q 进入不同状态空间。** 所有 canonical path 必须使用相同的 temporal `encode_state`。
4. **不要把单结构 metric 和 ensemble metric 混为一谈。** Kabsch RMSD、distance MAE、contact Jaccard 与 AlphaFlow RMWD/RMSF/PCA 是不同对象。
5. **不要把失败样本吞掉。** 一个“完整 report”必须说明 status、sample count、mode count 和 excluded/failed count。
6. **不要把 OpenFold atom37 output 和 project legacy atom37 index 直接混用。** ordering 必须显式标记和转换。
7. **不要从 backbone-only target 推断真实 side-chain accuracy。** 没有 native atom37 target 时只能报告生成结构的 geometry/clash/torsion diagnostics。

只要遵守这七条，ABC pipeline 的每一层能力就可以被单独诊断，最终的“是否超越 persistence”也会有可复现、可审计的含义。
