# ICLR 论文计划：从「模型打不过 baseline」到可发表的核心主张

日期：2026-09-16
仓库：`/liuyutian/protein_jepa_frame`
关联文档：`V5_V9_REEVALUATION_REPORT.md`（本轮审计）、`docs/reviewer_experiment_todo.md`（原有实验计划）、
`EVALUATION_COMPARISON_REPORT.md`

---

## 0. 一句话现状

> **你现在的模型不是「不够好」，而是「你在用错误的指标衡量它，而且你的消融实验没有按设计实现」。**
>
> 本轮审计（全部命令都在本仓库实跑过）发现四件事：
> 1. **M1 的 2×2 消融根本没按设计实现**——`Markov` 与 `FreeEnergy` 共用同一个力场，
>    `Full` 的 memory 范数是 `3.87e-10`（等于零），`FullCalibrated` 的 mobility 被标定为 `0`
>    （退化模型）。而**唯一真正带 memory 的 `Mori` 拿到了全场最好的稳态误差**
>    （`stationary_jsd` 0.083，比其它方法好 4.3–6.8×）。
> 2. **K1 的 raw accuracy 是多数类假象**：baseline 的 0.98 其实是 balanced accuracy 0.33（=随机）。
>    换成 balanced accuracy 后，**v5 在 h=8 上以 0.5506 大胜所有 baseline（+0.217）**。
> 3. **K1 的 committor 标签是重言式**：常数预测器拿到 brier `0.00000`、ρ `1.0000`（标签泄漏）。
> 4. v9 训练发散、R1 探针容量不对等、`closure_gate` 硬编码 `False`、readout 会退化成常数先验。
>
> **所以在继续调 v10/v11 之前，必须先做的是修好测量层和消融结构。**
> 修好之后你手上至少有两个真实效应、一个从未打开的盲测赛道（450K 温度外推），
> 以及一条清晰的可发表主张：**「结构决定物理性质的可分解性」**——
> 而不是「我的 MSE 最小」。

---

## 1. 本轮审计的新发现（决定论文选题）

### 1.1 【最重要】M1 的 2×2 消融：效应是反的，而且非常干净

`artifacts/reviewer_experiments/m1_strict_v5_seed6303.json`（v5，seed 6303，
30 条测试轨迹，protein bootstrap 2000 次）的 `methods` 字段里有完整的 8 个方法：

| method | n_traj | prediction_mse | innovation_whiteness | fdt_residual | **stationary_jsd** | acf_error | semigroup_mean_defect |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Markov` | 30 | 0.0023672 | 272.39 | 11.451 | 0.51767 | 0.77161 | 3.74e-06 |
| `FreeEnergy` | 30 | 0.0023672 | 272.39 | 14.265 | 0.50755 | 0.77158 | 4.77e-05 |
| **`Mori`** | 30 | 0.0052666 | 249.71 | 22.750 | **0.083036** | 0.73463 | 2.3553 |
| `Full`（FE+Mem） | 30 | 0.0023674 | 272.39 | **1.0458** | 0.56254 | 0.77203 | **3.21e-04** |
| `FullCalibrated` | 30 | 0.0023672 | 272.39 | 1.298 | 0.51203 | 0.77566 | 5.10e-06 |
| `GenericMarkov` | 24 | 0.0036989 | 258.68 | 1.5597 | 0.49000 | 0.58491 | 0.3518 |
| `GRU` | 30 | **0.0023494** | **247.84** | 1.2578 | 0.35727 | **0.33525** | 0.63232 |
| `CausalTransformer` | 30 | 0.0025809 | 248.26 | 1.1757 | 0.38215 | 0.48078 | 0.63663 |

**这张表读出来的因果结构（和我上一份报告写的完全不同）：**

1. **`Mori`（只有 memory、没有自由能力）的 `stationary_jsd` = 0.083，是全场最好，
   比 GRU 好 4.3×、比 `Full` 好 6.8×。**
   → 「有限 memory」这一项**单独就能大幅修正稳态分布**。这正是
   `docs/reviewer_experiment_todo.md` §511 期待的「functional decomposition」。
2. **但 `Full`（自由的 bath memory + 状态依赖扩散）把 `stationary_jsd` 拉回到 0.563。**
3. `Full` 在 `fdt_residual`（1.05）和 `semigroup_mean_defect`（3.2e-4）上是全场最好。
   → 状态依赖扩散 + bath 项**单独负责 FDT / 半群一致性**。
4. `GRU`/`CT` 在 `acf_error`（0.335/0.481）和 `innovation_whiteness`（248）上最好，
   但 `semigroup`（0.63）和 `fdt`（1.26）都不如 `Full`。
   → 通用序列模型**预测得准、但物理不一致**。

#### 1.1b 【决定性证据】`fit_info` 显示当前的「2×2」根本没有按设计实现

上表的 `prediction_mse` / `innovation_whiteness` 出现**逐位相同**，
我原本怀疑是「自由能项没生效」。直接 dump `methods[*].fit_info` 后发现
**真正的原因更严重：四个格子的结构根本不是设计文档里写的那四个。**
（已实测验证，见文末 §8 Step 1）

| method | `force_source` | `diffusion_source` | `kernel_source` | **`kernel_norm`** | 其他 |
|---|---|---|---|---|---|
| `Markov` | `native_lspwm_potential` | `train_constant` (1.523e-07) | — | — | |
| `FreeEnergy` | `native_lspwm_potential` | `train_local_knn` (1.523e-07) | — | — | |
| `Mori` | `native_lspwm_potential` | `train_constant` (1.523e-07) | **`train_finite_memory_ridge`** | **4.374e-04** | |
| `Full` | `native_lspwm_potential` | `native_lspwm_state_dependent` | `native_lspwm_bath` | **3.870e-10** | |
| `FullCalibrated` | `native_lspwm_potential` | `native_lspwm_state_dependent` | `native_lspwm_bath` | 3.870e-10 | **`mobility_scale: 0.0`** |

**三个必须立刻修正的事实：**

1. **`Markov` 和 `FreeEnergy` 用的是同一个力场**（都是 `native_lspwm_potential`），
   二者只差 `diffusion_source`（`train_constant` vs `train_local_knn`）。
   ⇒ **它们不是「自由能 vs 无自由能」的对照**，而是「常数扩散 vs 局部 kNN 扩散」的对照。
   这直接解释了 `prediction_mse` 逐位相同。
   **`docs/reviewer_experiment_todo.md` §470 要求的 `M-Markov`（no free energy）
   从未被实现过。**
2. **`Full` 的 `kernel_norm = 3.87e-10`，实际上等于零** ⇒ **`Full` 根本没有 memory！**
   真正带 memory 的只有 `Mori`（ridge 拟合，`kernel_norm = 4.37e-4`，比 `Full` 大 6 个数量级）。
3. **`FullCalibrated` 的 `mobility_scale = 0.0`** ⇒ 漂移被标定为 0，
   模型几乎不动，所以它「半群一致性」极好（5.1e-6）。
   **这是一个退化模型**，和 D1 的 readout 退化是同一类问题，必须标记或重构。

**修正后的因果结论（更干净、更可辩护）：**

> 在**唯一一个真正带有限 memory 的格子（`Mori`）**里，
> `stationary_jsd` = **0.083**，比所有其它方法好 4.3–6.8×；
> 而 **`Full` 因为没有 memory（`kernel_norm≈0`），拿不到这个好处**。
> 也就是说：**「有限 memory 修正稳态分布」是本轮最干净、量级最大的效应
> （kernel 范数差 6 个数量级），而不是「自由能破坏了稳态」。**

这比原来的读法**更强**，因为：
- 它有一个 6 个数量级的机制变量（`kernel_norm`）作为"处理强度"；
- 它是**单调可检验**的（kernel 越强 → 稳态越好？可做扫描，见 P2/P3）；
- 它把 `Full` 的失败解释成**配置错误**（memory 没生效），而不是「模型思想错了」——
  **这意味着修好配置后，`Full` 有可能同时拿到稳态和 FDT 两个优势**，
  也就是 P3 要做的事。

> **结论：这是一个真实、量级大、机制清晰的效应，不是「我们全面输了」。**
> 上一份 `V5_V9_REEVALUATION_REPORT.md` §3 只报了 `Full` 一行
> （而这一行的 memory 恰好是坏的），所以把结论写成了「全面垫底」。


### 1.2 K1 的 raw accuracy 是多数类假象；balanced accuracy 下模型在 h=8 大胜

FastFolding 是 3 分类（unfolded / transition / folded）且极不平衡，
所以 **Persistence 的 raw accuracy = 0.978，但 balanced_accuracy = 0.330
（恰好 = 1/3 = 随机水平）**。也就是说旧报告里「baseline 0.980–0.990，
模型 0.43–0.65，模型惨败」这句话里，**baseline 的 0.98 是纯粹的多数类预测**，
不是能力。

`protein_jepa_frame/eval/dynamics_benchmarks.py:108` 里
`balanced_accuracy` 和 `macro_f1` 早就实现了，但报告一直在引用 `accuracy`。

**v5，balanced_accuracy（按蛋白宏平均）：**

| horizon | Persistence | MSM | tICA+MSM | **JEPA v5** | **JEPA − best baseline** |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.3297 | 0.3306 | 0.3333 | 0.3306 | −0.003（并列） |
| 4 | 0.3305 | 0.3333 | 0.3333 | 0.1619 | **−0.171（真实失败）** |
| **8** | 0.3304 | 0.3333 | 0.3333 | **0.5506** | **+0.217（大胜）** |
| 16 | 0.4951 | 0.5000 | 0.5000 | 0.4967 | −0.003（并列） |

对照 v3（`balanced_accuracy` 全 horizon 都 ≤ 0.30）和 v9（≤ 0.327）：

| h | v3 | v5 | v9 |
|---:|---:|---:|---:|
| 1 | 0.0815 | 0.3306 | 0.1767 |
| 4 | 0.0530 | 0.1619 | 0.2266 |
| **8** | 0.2421 | **0.5506** | 0.3274 |
| 16 | 0.3007 | 0.4967 | 0.0980 |

→ **v5 在 h=8 上比所有 baseline 高 0.22 个 balanced accuracy，而 v3 只有 0.24。**
这是一个**真实的、跨 checkpoint 可区分的效应**，旧报告完全没看见。
同时 h=4 是真实失败（0.162 低于随机 0.333），必须诚实报告。

### 1.3 K1 的 committor 标签是重言式（致命的评测 bug）

`protein_jepa_frame/fastfolding/benchmark.py:225`：

```python
committor = {"unfolded": 0.0, "folded": 1.0}.get(source, 0.5)
```

而 `committor_target`（同文件 109–113 行）定义为「未来窗口内**先**命中 folded 还是
unfolded」。由于状态标签是**时间粘滞**的、而未来窗口从 `end+1` 开始，
**第一个命中几乎总是当前状态本身** ⇒ `committor_target ≈ source_state`。

后果（实测，所有 horizon 一致）：

| method | committor spearman | committor brier |
|---|---:|---:|
| `Persistence` | **1.0000** | **0.00000** |
| `MSM` | 0.98–1.00 | 0.002 |
| `tICA+MSM` | 0.21 | 0.002 |
| `JEPA v5` | −0.69 … +0.71 | 0.010–0.061 |

**常数/粘滞预测器拿到 brier 0.00000、ρ=1.0000 —— 这是标签泄漏的铁证。**
所以：

- 上一份报告里「JEPA 的 committor 相关为负（−0.687），说明模型反向」这个结论
  **不成立**——参考量本身是重言式，负相关是跟一个无意义参考比出来的。
- `docs/reviewer_experiment_todo.md` §75 早就写了「必须用 reference MSM 固定边界条件
  解 committor」，但当前实现**没有**这么做。这是 **P1** 必修项。

### 1.4 其余已确认的缺陷（来自 `V5_V9_REEVALUATION_REPORT.md`）

| # | 缺陷 | 位置 | 后果 |
|---|---|---|---|
| D1 | `validated` readout 可退化成常数先验（`prior_blend=1.0`） | `scripts/generate_fastfolding_predictions.py` | 两个不同 checkpoint 产生**逐位相同**预测；`k1_context_calibrated.json` 是假的 |
| D2 | `closure_gate` 硬编码 `False` | `scripts/evaluate_m1_strict.py:876` | M1 没有真实 gate；rollout 发散被静默 `continue` |
| D3 | R1 gate 阈值是绝对 MSE 尺度 | `scripts/evaluate_history_gap.py` | 跨表示不可比（JEPA 的 loss 尺度比 PCA 小 3 个数量级） |
| D4 | R1 的 current 用线性探针、history 用 MLP 探针 | 同上 | **容量不对等**；PCA(JEPA frame_states,16d) 对照在 28/28 组合上相对增益 98% 且全 PASS ⇒ 该指标不隔离表示质量 |
| D5 | v9 训练发散（traj NLL → 1.37e15，LR→2.9e-10，dynamics NLL 恒定 235.3） | `configs/lspwm_v44_stage_b_v9_load_bath.yaml` | M1 无法评测（30/30 non-finite） |
| D6 | v10 config 不足以修根因（仍 `epochs: 3`） | `configs/lspwm_v44_stage_b_v10_balanced_weights.yaml` | LR 饥饿未修，**不要按现状启动** |

> **关键判断：D1–D4 全部是「测量层」缺陷。在它们修好之前，
> 任何新模型的正面结果和负面结果都不能写进论文。**
> 而 1.1 和 1.2 说明：修好之后，**已经有真实效应在等你**。

---

## 2. 论文选题：三个候选主张，以及为什么推荐第三个

### 候选 A：「我们的模型最好」（❌ 放弃）

现状：M1 上 `Full` 输 `Markov` 的 `prediction_mse`、输 `GRU` 的 `acf_error`；
K1 上输 Persistence 的 raw accuracy（虽然那是假象）。
**这条路需要你在准确率上打赢一个自由的 3 层 MLP / GRU，在一个
物理约束的参数化下几乎不可能。不要再走。**

### 候选 B：「闭合性是可测量的，而且通用序列模型不具备」（✅ 推荐核心）

**位置：这是唯一一条「你的模型结构上有结构性优势、而 baseline 结构上没有」的路。**

核心主张（4 条 claim 组成的证据链）：

> **C1（测量）** 现有 latent protein dynamics 评测存在系统性退化：
> 不平衡标签上的 raw accuracy、由未来首次命中导出的 committor proxy、
> 绝对尺度的 history-gap 阈值，**都能让常数预测器拿到 SOTA 级分数**。
> 我们给出一套非退化的协议（matched-capacity 探针 + 相对尺度 gate +
> balanced/skill 指标 + 由 train-only MSM 固定边界解出的 committor）。
>
> **C2（机制）** 在这套协议下，对一个 2×2 结构消融（自由能力场 × 有限 memory）
> 做映射，得到一个清晰的**分离结构**：
> **有限 memory 项**把稳态分布误差降低约 6–7 倍
> （`stationary_jsd` 0.083 vs 0.36–0.56，同时 `kernel_norm` 差 6 个数量级），
> 而**状态依赖扩散项**负责 FDT / 半群一致性
> （1.05 / 3.2e-4 vs 11.5–22.8 / 0.35–0.64）。
> 通用序列模型落在表面的另一个位置（ACF / innovation 好、FDT / 半群差）。
> **关键：现有配置从未让两项同时生效过，所以正确的配置本身就是一个未报告的结论。**
>
> **C3（方法）** 我们给出去耦参数化，使两项优势可同时取得
> （**这是你的建模贡献，也是风险最高的一步**）。
>
> **C4（迁移）** 由于自由能项显式携带温度依赖，闭合法在**温度外推**
> （348K→450K，盲测 20 蛋白）上比通用历史模型泛化更好。

**为什么 B 是可行的：** C1 我已经验证了 ~80%（1.2/1.3 是实证 + 代码）；
C2 有 1.1 的表作为种子（需要 3 seed 确认 + 力场活跃性审计）；
C4 的**数据资产已经存在且从未打开**（`manifests/tmjepa_v3.temperature_test.jsonl`，
20 蛋白 @450K），而训练是 348K。这是天然的结构性优势赛道。

**为什么 B 有风险：** C3 可能失败。缓解：把 C1+C2 作为论文骨架
（「诊断 + 机制」），C3+C4 作为 payoff。即使 C3 只拿到部分优势，
论文仍然成立（诚实的 trade-off 映射本身就是贡献）。

### 候选 C：纯「评测审计」论文（⚠️ 只作 workshop / TMLR 退路）

`ProjectedForceClosure` 的实现 + D1–D4 的发现本身有价值，但
**ICLR 主 track 对纯 negative/audit 论文接受率很低**。
把它作为 B 的 §3（协议章节），不要作为独立主线。
如果 C3 彻底失败，退到 TMLR 或 NeurIPS/ICLR workshop。

### 推荐：B 为主线，C 作为其中一节

**一句话 thesis（可直接当 abstract 第一句）：**

> Latent protein dynamics models are routinely reported as either
> state-of-the-art or physically inconsistent depending on which metric is
> used; we show that most of these verdicts are artifacts of degenerate
> metrics, and that under a non-degenerate protocol the structural terms of
> a thermodynamic latent world model control **separable** physical
> properties, with a strict trade-off that generic sequence models cannot
> escape.

---

## 3. 分期 Plan

每一期包含：目标 / 具体改动 / 产物 / 通过标准 / 预估工时。
**P0 和 P1 是硬前置，不可跳过、不可并行偷懒。**

---

### P0 —— 结构审计 + 代码缺陷修复（**1–2 天，最高优先级**）

**目的：1.1b 已经证明当前的「2×2」没按设计实现。P0 要把四个格子做对，
否则 C2/C3 全部建立在一个不存在的消融上。**

**任务：**

| ID | 任务 | 文件 | 通过标准 |
|---|---|---|---|
| P0-1 | **补上真正的 `M-Markov`**（no free energy）：`Markov` 当前用的是 `native_lspwm_potential`，与 `FreeEnergy` 同源 ⇒ 需要实现一个力场无关的对照 | `scripts/evaluate_m1_strict.py:322` | 存在一个 `force_source != native_lspwm_potential` 的格子；`prediction_mse` 不再与 `FreeEnergy` 逐位相同 |
| P0-2 | **查清 `Full` 的 `kernel_norm = 3.87e-10` 为什么≈0**：是 Stage-A bath 没训好、还是 dt 不匹配（τ=4k–128k ps vs dt=8000 ps） | `protein_jepa_frame/models/lspwm.py`（`GaugeFixedBath`）、`configs/*v5*.yaml` | 给出根因；若要修，修完后 `Full.kernel_norm` 达到 `Mori` 量级（~1e-4） |
| P0-3 | **`FullCalibrated` 的 `mobility_scale=0.0` 判为退化**：加 assert（`mobility_scale > 0`）或标记 `degenerate=true` | `scripts/evaluate_m1_strict.py` | 该行在主表中带退化标记，或被重构 |
| P0-4 | `closure_gate` 真实实现（现在硬编码 `False`） | `scripts/evaluate_m1_strict.py:876` | 有 gate 定义 + 写入 JSON + 单测 |
| P0-5 | M1 禁止静默降级：`n_test_trajectories < 30` 为硬失败；发散必须计入 failure rate（不能 `continue`） | `scripts/evaluate_m1_strict.py` | v5 确认 `n_test_trajectories == 30`，发散数显式输出 |
| P0-6 | `prior_blend=1.0` 设为错误或标 `degenerate=true`（D1） | `scripts/generate_fastfolding_predictions.py` | 单测：退化输入必须 raise 或写 flag |
| P0-7 | 删除/标废 `k1_context_calibrated.json`、`k1_detskill_context_calibrated.json` | `artifacts/reviewer_experiments/protein_kinetics/` | 移入 `_invalid/` 或加 `"valid": false` |
| P0-8 | 所有评测方法的 `fit_info` 必须自带「结构指纹」字段（force/diffusion/kernel/mobility 的来源与范数），并写一个跨方法断言：**任意两个格子的结构指纹不得相同** | `scripts/evaluate_m1_strict.py` | 该断言能捕获本次的 `Markov==FreeEnergy` 事故 |

**产物：** `artifacts/reviewer_experiments/protocol/structure_audit.json` +
`docs/structure_audit_2026-09-XX.md`

**硬门：** P0-1/P0-2 未完成前 → **冻结 C2/C3 的全部结论**（当前的 2×2 不成立）。

---

### P1 —— 测量层修复：让指标不再能被常数预测器骗过（**2–4 天**）

**目的：产出 C1。这是论文的 §3，也是后面所有结论的前提。**

| ID | 任务 | 文件 | 通过标准 |
|---|---|---|---|
| P1-1 | **保留集（retention set）**：故意放入 `ConstantPrior`、`MajorityClass`、`StickyState`、`ShuffledHistory` 四个**退化解码器**，任何指标分不出它们与真模型，该指标即判废 | 新文件 `protein_jepa_frame/eval/degeneracy_suite.py` | 每个指标的 JSON 里带 `sanity: {constant_prior: ..., best_degenerate: ...}` |
| P1-2 | **K1 全量改报 balanced_accuracy / macro_f1 / skill-vs-persistence**，raw accuracy 只能作为附录诊断 | `protein_jepa_frame/fastfolding/benchmark.py`、`scripts/evaluate_protein_kinetics.py` | 主表不再出现裸 accuracy |
| P1-3 | **committor 重写**：由 train-only 轨迹拟合 MSM（固定状态定义 + 固定边界集 A/B），解线性系统得 reference committor；**参考值必须与 source state 标签无确定性关系**（加一个 assert：`corr(source_state_indicator, committor) < 0.95`） | `protein_jepa_frame/fastfolding/`（新 `committor.py`） | Persistence 的 committor brier 不再 ≈0、ρ 不再 ≈1.0；该退化 test 写进单测 |
| P1-4 | **R1 探针容量对等**：current 与 history 探针都用**同一个 MLP**（同深度/宽度/参数量），并在 JSON 里记录 `n_params` 与差值 | `scripts/evaluate_history_gap.py` | `n_params` 相等（或差值 <1%）；PCA 对照不再 100% PASS |
| P1-5 | **R1 gate 改相对尺度**：`gate = (L_current − L_history) / L_current > 0.05`，禁止绝对阈值 | `scripts/evaluate_history_gap.py` | 所有表示（PCA/tICA/VAMP/JEPA）用同一 gate 公式 |
| P1-6 | **R1 协议统一**：current/history 都用同一个确定性 ridge 探针（作为主结果），MLP 作为附录；重跑 JEPA / PCA / tICA / VAMP / SRV | `scripts/evaluate_history_gap.py`、`scripts/evaluate_r1_coordinate_baselines.py` | 所有表示在同一份 latent 导出上评测 |
| P1-7 | M1 表格强制带 `n_effective_trajectories`；`< 30` 直接拒绝 | `scripts/evaluate_m1_strict.py` | Table 输出含该列；`GenericMarkov` 的 24/30 被显式标注 |

**产物：** 重跑的 R1 表（含容量对等 + 相对 gate）、重跑的 K1 表（balanced + 真 committor）、
`degeneracy_suite` 报告。

**这是论文最容易被攻击的地方，也是最容易拿分的地方。**
建议把 P1-1 的退化解码器保留集做成论文的一个**图**（Fig. 2）：
「哪些指标能被常数预测器通过」——视觉冲击很强。

---

### P2 —— 确认 C2：把 1.1 的 trade-off 做成论文的 Fig. 3（**3–5 天**）

**前提：P0 通过（G0 结构独立性成立）。**

| ID | 任务 | 细节 | 通过标准 |
|---|---|---|---|
| P2-1 | M1 `v5` 全 8 方法 **3 seed**（6303/6304/6305）重跑 | 现有 `m1_strict_v5_seed6303.json` 只 1 seed | 每格有均值 ± 配对 bootstrap 95% CI |
| P2-2 | 加入 `M-Memory-only`（`Mori`）与 `M-FE-only`（`FreeEnergy`）的**配对**比较，而非只比 `Full` | 用 `pairwise_full_minus_baseline` 之外的 `Mori_vs_*` / `FreeEnergy_vs_*` | 输出完整 8×8 配对矩阵 |
| P2-3 | **修 v9 的训练**（见 P2-4），使 3 个 checkpoint 家族都可评 | v9 现在 30/30 non-finite | v9 M1 可评 |
| P2-4 | **训练稳定性修复**（这是 v10 之前必须做的）| 见下方 | 无 NLL 爆炸、无 LR 饥饿 |
| P2-5 | 对 `stationary_jsd` / `acf_error` / `fdt_residual` / `semigroup` 各自做**置换检验**（label permutation）确认效应不是采样噪声 | 新脚本 `scripts/m1_permutation_test.py` | 4 个指标都有 p 值 |

**P2-4 训练稳定性修复清单（针对 v9/v10 现有 bug）：**

1. **总步数不足**：`configs/lspwm_v44_stage_b_v9_load_bath.yaml:125` `epochs: 3` = 3000 步，
   而 `lspwm_curriculum_steps: [0,200,700,1200,1700,3000]` 把 `observation` 相
   放在 1700–3000。`_lr_scale`（`protein_jepa_frame/training/train.py:126-131`）是
   **余弦退火到 0**，到 step 2500 时 LR 已是 ~1e-7 量级 ⇒ 后 1/3 的训练实际上是冻住的。
   **修法：总数提到 6000–8000 步，并把 `observation` 相放在 40%–80% 区间；
   或给 `observation` 相设 LR 下限（`max(cos, 1e-6)`）。**
2. **`lspwm_trajectory_nll` 无数值保护**：step 960 起指数爆炸到 1.37e15。
   **修法：加 `torch.clamp`/log 空间累加；训练中每 N 步做一次 8 步 rollout 的
   `torch.isfinite` 断言；保留最近 3 个 checkpoint 做回滚。**
3. **bath 与 drift/diffusion 不一致**：v9 确实加载了 Stage-A bath
   （‖K[1:]‖_F 3.87e-10 → 3.48e-3），但 memory kernel 的时间尺度
   （τ=4k…128k ps）与 `dt=8000 ps` 不符。
   **修法：用物理 dt（或把 bath 冻结、只训 diffusion），并加一个
   「bath–drift 一致性」诊断进 gate。**
4. **v10 config 不足以解决根因**：
   `configs/lspwm_v44_stage_b_v10_balanced_weights.yaml` 把
   `lspwm_trajectory_weight` 0.5→0.02、`lspwm_dynamic_weight`→0.3，
   但**仍然 `epochs: 3` + 同样的 `curriculum_steps`** ⇒ LR 饥饿未修。
   **不要按现状启动 v10。**

**产物：** `artifacts/reviewer_experiments/thermodynamic_closure/` 下的
3-seed JSON + `summary.md` + Fig. 3（trade-off surface 散点图：
x = stationary_jsd，y = fdt_residual，点 = 8 个方法，颜色 = 2×2 格子）。


---

### P3 —— 方法贡献 C3：stationary-preserving memory projection（**5–10 天，风险最高**）

**问题定义（由 1.1 的 trade-off 直接推出）：**

- `Mori`：**唯一带真实 memory 的格子**（`kernel_norm=4.37e-4`），`stationary_jsd` 最好（0.083），
  但 `fdt` 22.8、`semigroup` 2.36 最差（因为它用的是 `train_constant` 扩散，非状态依赖）；
- `Full`：**memory 实际为零**（`kernel_norm=3.87e-10`），靠状态依赖扩散拿到
  `fdt` 1.05、`semigroup` 3.2e-4 最好，但 `stationary_jsd` 退回 0.563。

⇒ 这是一个「**memory 管稳态、状态依赖扩散管 FDT**」的分离结构，
而现有配置**从没让两者同时生效过**。这正是方法要解决的问题。

**物理假说（两个都必须测，不要预设）：**

- **H1（有利）**：有限 memory 通过提供缺失的投影自由度/记忆核，
  把 rollout 的长期统计拉回真实稳态 ⇒ 稳态 JSD 大幅下降。
  （支持证据：`kernel_norm` 差 6 个数量级、`stationary_jsd` 差 6.8×；
  但**相关性 ≠ 因果**，必须做 P3-0 的剂量扫描）
- **H2（不利）**：memory 项是有偏的——它在稳态下有一个非零的平均贡献
`⟨m(q)⟩_π ≠ 0`，因此**移动了稳态分布**；但同时它提供了正确的
时间关联结构（所以 FDT 好）。反过来纯 memory 版没有自由能锚定，
稳态看起来"更对"是因为轨迹被过度阻尼、看起来更像平稳噪声（**这需要 P0-1/P2-5 验证，
不要把两种解释都排除**）。

**方法（可实现的、原理清晰的修法）：stationary-preserving memory projection。**

> 把 memory 修正项投影到稳态测度 π 的正交补上，使其**不能改变稳态**，
> 只能改变动力学（时间关联）。

具体两种等价实现，任选其一（**建议都实现，作为消融**）：

1. **损失约束（简单）**：
   `L_orth = ( E_{q~π} [ m(q) · ∇U(q) ] )² ` 或
   `L_orth = ‖ E_{q~π}[m(q)] ‖²`，其中 π 由 train-only 经验分布给出。
2. **GENERIC 结构约束（更漂亮）**：
   复用仓库里已有的 `PoissonOperator` / `DissipationOperator`
   （`protein_jepa_frame/models/lspwm.py:1619-1624`），把动力学写成
   `q̇ = −M(q)∇U(q) + L(q)∇S(q) + m_⊥(q, h_t)`，并强制 `m_⊥ ⊥ ∇U`
   （在 π 加权 L² 内积下）。这样 memory 项在结构上**不可能**改变稳态。

**可验证的预测（这就是论文的实验设计）：**

| 模型 | `kernel_norm` | stationary_jsd | fdt_residual | semigroup |
|---|---:|---:|---:|---:|
| `Mori`（基线：真 memory + 常数扩散） | 4.37e-04 | 0.083 | 22.8 | 2.36 |
| `Full`（基线：零 memory + 状态依赖扩散） | 3.87e-10 | 0.563 | 1.05 | 3.2e-4 |
| **`Full-⊥`（本方法：真 memory + 状态依赖扩散 + 正交投影）** | ~1e-4 | **期望 ≈ 0.08–0.15** | **期望 ≈ 1.0–1.5** | **期望 ≤ 1e-3** |

**外加一条必须做的剂量–响应实验（把相关性升级为因果）：**

| ID | 任务 | 目的 |
|---|---|---|
| P3-0 | **kernel 强度扫描**：把 memory 核按 `α ∈ {0, 1e-4, 1e-3, 1e-2, 1e-1}` 缩放（或扫 ridge 正则强度），画 `stationary_jsd(α)` | 若单调下降 ⇒ H1 成立，C2 从「观察」升级为「剂量–响应因果」，这是审稿人最认的形式 |

**如果预测成立，这是论文的核心结果：**
> "我们证明稳态误差与 FDT 误差分别由两个结构项控制，
> 并且用一个稳态正交投影把二者解耦，首次在同一模型上同时取得两者最优。"

**如果预测失败，诚实记录，退到 C1+C2 的「诊断 + trade-off 映射」论文（仍然可发 TMLR/workshop）。**

**任务：**

| ID | 任务 | 文件 |
|---|---|---|
| P3-1 | 实现 `lspwm_stationary_orthogonal_memory`（flag + 损失项） | `protein_jepa_frame/models/lspwm.py`（memory 项附近）、`protein_jepa_frame/config.py` |
| P3-2 | 实现 π 的经验估计（train-only，EMA 或 KDE，与已有 `_estimate_kde_bandwidth` 复用） | 同上 |
| P3-3 | 在 M1 的 8 方法里加第 9 个方法 `FullOrthogonal` | `scripts/evaluate_m1_strict.py` |
| P3-4 | 训练配置 `configs/lspwm_v45_stationary_preserving.yaml`（在修好的 v5 训练配方上） | `configs/` |
| P3-5 | 消融：`L_orth` 权重 ∈ {0, 1e-3, 1e-2, 1e-1}，画权衡曲线 | 训练 + M1 |

**注意：`lspwm_history_closure_enabled` 在每一个 config 里都是 `false`，
但 `ProjectedForceClosure`（`lspwm.py:383`）已经实现好了。**
它是一个 **bounded + O(dt²) short-time gated** 的修正项——
正好符合「memory 只应影响短时间、且有界」的物理直觉。
**建议把它打开并和 `L_orth` 一起评测**，因为 R1 的发现
（JEPA 的 latent 已经接近 history-Markov）意味着**残差应该是小的、短时的**，
而不是一个大 RNN。这也是一个漂亮的「表示 + 结构协同」故事。

---

### P4 —— 迁移主张 C4：温度外推（**3–5 天，性价比最高的「wow」实验**）

**资产（已经存在，从未打开）：**

- `manifests/tmjepa_v3.temperature_test.jsonl` —— **20 个盲测蛋白 @ 450 K**，
  训练温度是 348 K ⇒ 天然的温度外推。
- `manifests/tmjepa_v3.temperature_cv_{348,379,413}_{train,validation}.jsonl`
  ⇒ 可以画出**分级**的温度迁移曲线（348→379→413→450）。

**实验：**
固定所有模型（`Full` / `Full-⊥` / `GRU` / `CT` / `MSM` / `tICA+MSM` / `Persistence`），
在 348K 上训练/拟合，在 379/413/450K 上评测。

**预期图（论文 Fig. 4）：** x = 温度（ΔT = 0/31/65/102 K），y = balanced accuracy 或 NLL skill，
每条曲线一个方法。**物理结构模型应该退化更慢**，因为
`lspwm_reference_temperature_k` / diffusion ∝ kT / 自由能项都显式含 T，
而 GRU/CT 只见过 348K。

**任务：**

| ID | 任务 | 通过标准 |
|---|---|---|
| P4-1 | 冻结 checkpoint，在 temperature_test 上跑全部方法（**只跑一次**，不许用于选型） | 产出迁移表 |
| P4-2 | 在 348/379/413 上做同样的训练+评测，画分级曲线 | 曲线单调可解释 |
| P4-3 | 报告 `Δskill / ΔT` 斜率 + 配对 bootstrap CI | 若闭合法斜率显著更平，C4 成立 |
| P4-4 | 检查 `scripts/audit_lspwm_protocol.py` 确认 test/temperature_test 与 train 蛋白不重叠 | 零重叠 |

> ⚠️ 这是**盲测集**。按 `docs/revise_plan_implementation_2026-09-11.md:61`
> 的纪律，必须在 P1/P2/P3 全部冻结**之后**才能打开，且只能开一次。

---

### P5 —— 稳健性（**2–3 天**）

| ID | 任务 | 通过标准 |
|---|---|---|
| P5-1 | 3 seed（6303/6304/6305）× 所有主张 | 方向一致 |
| P5-2 | 按 protein family / topology 分层重跑 | 不是单一蛋白族驱动 |
| P5-3 | 长 rollout 失败率表（发散轨迹数、norm drift） | 显式报告，不隐藏 |
| P5-4 | 计算成本表（参数量、训练步、GPU 小时） | 论文附录必需 |

**算力现实：** 本机 2×A800-80GB、176 核；v5 的 4670 步训练约 2 小时
（由 checkpoint mtime 09-15 10:16→12:19 推得，≈1.5 s/step），
R1/M1 评测在**分钟级**。**3-seed 全面重跑的总成本约 1–2 天机时，
算力不是瓶颈，工程和评测正确性才是。**

---

### P6 —— 写作（**2–3 周**）

**论文骨架（对应 `docs/reviewer_experiment_todo.md` §812 的原有规划，按本轮发现重排）：**

| 章节 | 内容 | 依赖 |
|---|---|---|
| §1 Intro | 「同一模型在不同指标下既是 SOTA 又是垃圾」——引出退化指标问题 | C1 |
| §2 Protocol | matched-capacity 探针、相对 gate、balanced 指标、train-only MSM committor | P1 |
| §3 **Fig. 2** 退化指标保留集 | 常数预测器能通过哪些指标（视觉强） | P1-1 |
| §4 **Fig. 3** 2×2 trade-off surface | `stationary_jsd` vs `fdt_residual`，8 方法，颜色=格子 | P2 |
| §5 方法：stationary-preserving memory projection | 原理 + 实现 + 消融 | P3 |
| §6 **Fig. 4** 温度外推曲线 | 348→379→413→450K | P4 |
| §7 限制与负面结果 | h=4 的 balanced acc 失败、v9 发散、`GenericMarkov` 24/30 | 全部 |
| §8 附：S1 合成双井 GLE 参数恢复 | 证明能恢复已知 `A(z)+D+K(t)` | 见下方备注 |

> **S1（合成双井 GLE）强烈建议做**：它能一次性回答审稿人最狠的那句
> 「你这套参数化到底能不能恢复已知的动力学？」，而且成本很低
> （合成数据，不需要 GPU 集群）。它把论文从「benchmark 对比」
> 提升到「带 ground truth 的机制验证」。预算允许的话**优先于 P5**。

---

## 4. 论文级实验矩阵（最终要产出的表和图）

| 编号 | 实验 | 数据 | 方法 | 产出 & 支持的 claim |
|---|---|---|---|---|
| E1 | 退化指标保留集 | 任意 | ConstantPrior / MajorityClass / StickyState / ShuffledHistory vs 各指标 | **Fig. 2** → C1 |
| E2 | 2×2 结构消融 @3 seed（**修正后**：真 `M-Markov` / `M-FE` / `M-Memory` / `M-Full`，四者结构指纹必须互不相同） | mdCATH 验证集（30 轨迹） | `M-Markov` / `M-FE` / `M-Memory`(`Mori`) / `M-Full` + `GenericMarkov` / `GRU` / `CT` | **Fig. 3 + Table 1** → C2 |
| E3 | 稳态正交投影 + kernel 剂量扫描（P3-0） | 同上 | `M-Full-⊥`（`L_orth` 权重 × kernel `α` 双扫描） | **Fig. 3 新增点 + Table 2** → C3 |
| E4 | 温度外推 | `temperature_cv_{348,379,413}` + `temperature_test`(450K) | E2/E3 的方法 + 基线 | **Fig. 4** → C4 |
| E5 | S1 合成双井 GLE 参数恢复 | 合成 | `A(z)+D+K(t)` 恢复误差 | **Fig. 1** → 方法有效性 |
| E6 | 稳健性 | 3 seed × family 分层 | 全部 | 附录 |
| E7 | 长 rollout 失败率 + 算力 | 全部 | 全部 | 附录（诚实性） |

**Table 1（主表，M1）示例结构：**

| method | prediction_mse ↓ | innovation_whiteness ↓ | fdt_residual ↓ | stationary_jsd ↓ | acf_error ↓ | semigroup_defect ↓ | n_eff |
|---|---|---|---|---|---|---|---|

**关键：主表必须同时给出「预测类」和「物理类」两半，并且把
`Mori` 和 `FreeEnergy` 两行单独标出来**——因为 C2 的全部信息量在这两行的
不对称性上（memory 管稳态、自由能管 FDT/半群）。

---

## 5. Kill Criteria（明确的止损规则）

> 纪律：**gate 不过就记录失败并停止对应 claim，不要改评测器然后偷偷保留旧结果。**
> （这是 `docs/reviewer_experiment_todo.md` §788 的原话，务必遵守。）

| Gate | 判据 | 不过怎么办 |
|---|---|---|
| **G0 结构独立性** | 四个格子的 `fit_info` 结构指纹**互不相同**（力场来源/扩散来源/`kernel_norm`/`mobility_scale` 至少一项有数量级差异），且 `Mori` 与 `M-Full` 的 `kernel_norm > 1e-6` | 修结构；在修好前**冻结** C2/C3 的全部结论 |
| **G1 指标非退化** | 四个退化解码器在主导指标上**明显低于**真模型（例如 balanced_accuracy 差 > 0.1） | 该指标降级为附录；不得进入主表 |
| **G2 表示闭合** | 容量对等 + 相对 gate 下，`Δ_hist(JEPA) < Δ_hist(PCA/tICA)`（配对 bootstrap CI 不含 0） | 放弃 C1 的「JEPA 更 Markov」子主张，只保留「现有指标退化」部分 |
| **G3 结构分解** | `stationary_jsd(Mori) < stationary_jsd(Full)` 且 `fdt(Full) < fdt(Mori)`，两侧 CI 均不含 0，3 seed 方向一致 | 放弃 C2 的因果解释，退为「观察到的 trade-off」描述性结果 |
| **G4 解耦成功** | `Full-⊥` 在 `stationary_jsd` 上 ≈ `Mori` 且 `fdt` 上 ≈ `Full`，两侧都显著 | 放弃 C3（方法贡献）→ 论文降级为「诊断 + trade-off 映射」 |
| **G5 温度迁移** | `Full(-⊥)` 的 `Δskill/ΔT` 斜率显著平于 GRU/CT | 放弃 C4 |
| **G6 稳健性** | 3 seed 方向一致；family 分层不翻转 | 降级为单 seed 报告 + 明示限制 |

**只要 G1 过了、G3 或 G4 至少过一个，就有一篇可投 ICLR/ICML 的论文。
G1 + G3 都过 = 一篇「诊断 + 机制」的干净论文（不需要赢任何准确率指标）。**

---

## 6. 代码改动清单（按文件，可直接照着做）

| 文件 | 改动 | 优先级 | 关联 |
|---|---|---|---|
| `protein_jepa_frame/models/lspwm.py` | ① 结构指纹（`force_source`/`diffusion_source`/`kernel_norm`/`mobility_scale`）完整输出；② **查明并修好 `Full` 的 `kernel_norm≈0`**（Stage-A bath / dt 不匹配）；③ `memory` 项加稳态正交投影（`L_orth`）；④ 打开并适配 `ProjectedForceClosure` | P0/P3 | C2/C3 |
| `scripts/evaluate_m1_strict.py:322`（`MarkovClosure`） | **实现真正的无自由能 `M-Markov` 对照**（当前它和 `FreeEnergy` 共用 `native_lspwm_potential`） | P0 | C2 |
| `protein_jepa_frame/config.py` | 新增 `lspwm_stationary_orthogonal_memory`、`lspwm_stationary_orthogonal_weight` 等 flag | P0 | C3 |
| `protein_jepa_frame/training/train.py:126-131` | `_lr_scale` 加 LR 下限（`max(cos, 1e-6)`）；`lspwm_trajectory_nll` 数值保护 + rollout 有限性断言 + checkpoint 回滚 | P0/P2 | D5 |
| `configs/lspwm_v44_stage_b_v9_load_bath.yaml:125,118` | `epochs: 3 → 6~8`；`curriculum_steps` 重排（`observation` 放在 40%–80%） | P2 | D5 |
| `configs/lspwm_v45_stationary_preserving.yaml`（新） | 在修好的 v5 配方上 + `L_orth` | P3 | C3 |
| `scripts/evaluate_m1_strict.py` | ① `closure_gate` 真实化（:876）；② `n_eff` 列 + `<30` 硬失败；③ 新增 `FullOrthogonal` 方法；④ 输出 `Mori_vs_*` / `FreeEnergy_vs_*` 配对 | P0/P2/P3 | C2/C3 |
| `scripts/generate_fastfolding_predictions.py` | `prior_blend=1.0` → raise / `degenerate=true`（`--readout-mode` 已加） | P0 | D1 |
| `scripts/evaluate_history_gap.py` | 探针容量对等（都走同一 MLP）+ 相对尺度 gate + 记录 `n_params` | P1 | D3/D4 |
| `scripts/evaluate_protein_kinetics.py` | 主表改 balanced_accuracy / macro_f1 / skill | P1 | C1 |
| `protein_jepa_frame/fastfolding/benchmark.py:225` | **修 committor 泄漏**；参考值改由 train-only MSM 固定边界求解 | P1 | C1 |
| `protein_jepa_frame/eval/degeneracy_suite.py`（新） | 四个退化解码器保留集 | P1 | C1 |
| `protein_jepa_frame/eval/result_schema.py`（新） | 统一 JSON schema + provenance | P0 | 全部 |
| `scripts/m1_permutation_test.py`（新） | 指标置换检验 | P2 | C2 |
| `tests/` | 为 P0-3/P0-5/P1-3/G1 各加一个单测（回归保护） | P0/P1 | 全部 |

**注意：仓库的测试基础设施是 pytest（有 `tests/` 和 `.pytest_cache`），
新加的保护都要落成单测，否则下次重构会静默回退。**


---

## 7. 时间线与投稿策略（⚠️ 先核对官方 CFP 日期）

**重要提醒：`ICLR` 的投稿周期通常在每年 9 月截止。今天是 2026-09-16，
所以 ICLR 2027 的 main-track deadline 极可能已经过了或就在眼前。
请务必先核对官方 CFP，不要按我下面的推测排期。**

**推荐的排期（以「还有 ~4 个月」为前提，对应 ICML/NeurIPS 周期）：**

| 周 | 里程碑 | 对应期 |
|---|---|---|
| W1 | P0 完成（结构审计 + 8 项修复 + 单测）→ **G0 通过** | P0 |
| W2 | P1 完成（指标非退化 + committor 重写 + R1 容量对等） | P1 |
| W3–W4 | P2 完成（3 seed × 8 方法 M1 + 置换检验）→ **Fig. 3 定稿** | P2 |
| W4–W5 | E5（S1 合成双井 GLE）跑通 → **Fig. 1** | P6 备注 |
| W5–W8 | P3（`Full-⊥` 实现 + 消融）→ **Table 2** | P3 |
| W8–W9 | P4（温度外推，**最后开盲测**）→ **Fig. 4** | P4 |
| W9–W10 | P5 稳健性 | P5 |
| W10–W13 | P6 写作 + 内部 review | P6 |

**退路阶梯（按优先级）：**

1. **主目标**：ICML 2027（~1 月截止）/ NeurIPS 2027（~5 月截止）—— 时间充裕，
   可以完整做完 G0–G5。
2. **次目标**：ICLR 2028（~9 月截止）—— 如果有 G4 强结果可以等。
3. **同时投**：把 C1（指标退化审计）单独写成 workshop 论文投
   NeurIPS 2026 / ICLR 2027 workshop。**workshop 的 deadline 通常晚于 main track
   且有第二轮**，可以先把这部分成果锁定，不占用主线时间。
4. **保底**：TMLR（滚动投稿，无 deadline，对「诊断 + 诚实负面结果」友好）。

**关于「ICLR 文章」这个目标的诚实评估：**

- 如果 G4（解耦）成立 → **ICLR/ICML main track 有竞争力**（有新方法 + 有机制 + 有迁移）。
- 如果只有 G3 → **ICLR/ICML 偏弱**，因为「映射了一个 trade-off」听起来像分析而非方法；
  需要把 C1 的协议部分做得很厚，或加 E5 的 ground-truth 恢复来撑。
- 如果只有 G1 → 单投 workshop 或 TMLR。

---

## 8. 立即可执行的下一步（今天就能开始）

**Step 1（✅ 已完成，结论见 §1.1b）：`fit_info` 审计。**
这一步是整个 plan 的岔路口，我已经跑过了。命令与实测输出如下：

```bash
cd /liuyutian/protein_jepa_frame
/opt/conda/bin/python3 - <<'PYEOF'
import json
d = json.load(open('artifacts/reviewer_experiments/m1_strict_v5_seed6303.json'))
M = d['methods']
for name in ['Markov','FreeEnergy','Mori','Full','FullCalibrated',
             'GenericMarkov','GRU','CausalTransformer']:
    print(name, '->', json.dumps(M[name].get('fit_info', {}))[:400])
PYEOF
```

**实测结果（原始输出）：**

```text
Markov           -> {"force_source": "native_lspwm_potential", "diffusion_source": "train_constant",
                     "diffusion_mean": 1.5232748790695407e-07, "n_train_transitions": 1239}
FreeEnergy       -> {"force_source": "native_lspwm_potential", "diffusion_source": "train_local_knn",
                     "pilot_diffusion": 1.5232742830927997e-07, "n_train_transitions": 1239}
Mori             -> {"force_source": "native_lspwm_potential", "diffusion_source": "train_constant",
                     "kernel_source": "train_finite_memory_ridge", "kernel_norm": 0.00043737117184735843, ...}
Full             -> {"force_source": "native_lspwm_potential",
                     "diffusion_source": "native_lspwm_state_dependent",
                     "kernel_source": "native_lspwm_bath", "kernel_norm": 3.8701516991146434e-10, ...}
FullCalibrated   -> {"force_source": "native_lspwm_potential", ..., "calibration_source": "train_prefix_only",
                     "mobility_scale": 0.0, "kernel_norm": 3.8701516991146434e-10, ...}
GenericMarkov    -> {"architecture": "MLP(256,256)", "train_loss": 0.0020431370940059423, ...}
GRU              -> {"architecture": "GRU(128)", "train_loss": 0.0019827226642519236, ...}
CausalTransformer-> {"architecture": "causal_transformer(128,2 layers)", "train_loss": 0.0022580744698643684, ...}
```

**⇒ 三个结构性结论（详见 §1.1b）：**
1. `Markov` 与 `FreeEnergy` **共用同一个力场** ⇒ 「有/无自由能」的 2×2 从未实现；
2. `Full.kernel_norm = 3.87e-10` **≈ 0** ⇒ `Full` 根本没有 memory；
   唯一真 memory 是 `Mori`（`4.37e-4`，差 6 个数量级）；
3. `FullCalibrated.mobility_scale = 0.0` ⇒ **退化模型**（漂移被标定为 0）。

**Step 2（优先，1 天）：按 P0-1/P0-2/P0-3 修结构。**
现在最高优先级不是加新模型，而是把四个格子做成**真正互相独立**的结构，
并加 P0-8 的「结构指纹不得重复」断言。这一步做完，C2 才第一次真正成立。

**Step 3（半天）：把 8 方法 × 6 指标的完整表 + 配对 CI 落成一份 artifact，**
作为 Fig. 3 的数据源（现在这些数字散在 JSON 里，没有被任何报告引用过）。

**Step 4（1 天）：跑 P0-4 / P0-5 / P0-6 三个修复 + 单测。**
这三个改动小、风险低、且立刻让后续所有结果变得可信。

**Step 5：把 `V5_V9_REEVALUATION_REPORT.md` 的摘要按本文档 §1 修正。**
特别是：
- §3 的 M1 结论要**整段重写**——必须补上 `Mori` 行，并说明
  `Full` 的 memory 是坏的、`Markov`/`FreeEnergy` 不是真对照；
- K1 的结论要从「模型惨败」改成「raw accuracy 是假象；balanced 下 h=8 胜 0.22、h=4 失败」；
- committor 的所有结论标记为**待重算（标签泄漏）**。

> **一句话总结这个 plan：**
> 你不是需要一个更好的模型，你是需要先相信你的尺子。
> 修好尺子之后，你已经有两个真实效应（`Mori` 的稳态优势、h=8 的 balanced accuracy 优势），
> 加上一个从未打开的盲测赛道（450K 温度外推）。
> 论文的 claim 应该是「**结构决定物理性质的可分解性**」，
> 而不是「我的 MSE 最小」。

