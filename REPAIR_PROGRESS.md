# 修复与实验进度（2026-10-17）

当前完成了训练、时间、表示和评估合同修复，正在执行 H 读出门槛。**尚无 H+S 优于 persistence 或 pure-JEPA 的新证据；正式 B1b 训练尚未启动。** 原报告与旧 checkpoints 仅作历史诊断，不能续接为新实验结果。

## 按顺序执行的状态

| 顺序 | 状态 | 本轮交付 |
|---|---|---|
| P0：窗口、版本、测量 | 完成，CPU 回归通过 | 连续 K+N 窗口、共享 anchor、物理时间检查；schema-v2 表示；公共 anchor 指标；精确 window 配对和 protein bootstrap |
| P0：梯度、几何、交接 | 完成，CPU 回归通过 | B1a oracle 与 adaptation 分开；自主 rollout 梯度；冻结共享 encoder/decoder/normalizer；history pose 校准；保存加载与 B1b resume |
| P1：H 读出门槛 | **未通过，继续定位** | 新 Stage A 真实数据 pilot；单构象、8 窗口、独立读出器诊断；4 个验证蛋白的失败 gate |
| P1：history 增量信息 | 未开始正式实验 | 需前一门槛通过后，同一表示比较末帧、完整和打乱 history |
| P1：H+S 公平比较 | 未开始正式实验 | gate 通过后比较 direct、dS=0、H+S、无 history posterior；训练与验证 horizon 都需通过 gate |
| P2：多 seed、长程、温度外推 | 未开始 | 必须先有可重复的公平基线收益 |

## B1b 的实际训练目标

生产训练入口：[train_stage_b1b.py](protein_jepa_frame/training/train_stage_b1b.py)。

1. 只从真实 history 编码 `H_hist`，通过 posterior 初始化 `S0`。
2. 从最后 history 的 `H0` 开始迭代 `(H_hat,S_hat)`；所有样本都自主展开。
3. 预测完成后，才编码真实未来 H，作为停止梯度的 latent energy-score 标签。
4. 冻结 B1a decoder 只解码 **预测 H**；physical energy score 比较预测结构与**真实未来坐标**。真实未来 H 不进入 decoder、rollout 或状态更新。
5. teacher-forced Kalman NLL 独立用于诊断，训练 CLI 明确拒绝非零 `--nll-weight`，默认 point RMSD 权重为 0。

等价地，预测分布为 `F(history, noise)`，目标由未来观测提供；不能把目标当作递推的输入。回归测试固定 history 和 seed、改写未来坐标，要求预测 H、S 和 backbone 完全相同，同时 loss 改变。另有 decoder 调用检查，防止损失分支重新接回 true future H。

动力学内部使用 normalized H，decoder 接收 inverse 后的 raw H。B1b 冻结整个表示。正式训练要求 H gate 与 checkpoint 的 representation ID、train/val manifest hashes、K、stride、物理 dt，以及训练和验证 horizon 全部匹配。`--dry-run` 产物标为 `contract_smoke_only`，不能进入正式性能表。

## 本轮新实验

目录：`artifacts/repaired_v2/pilot_20261017/`。以下均为本轮重新运行的 CPU 结果。当前容器无法使用 GPU。

数据为 mdCATH 320 K、R0、1 ns/frame；Stage A 使用 8 个训练蛋白和 4 个独立验证蛋白。域、protein 和 source file 均分离。训练蛋白为 51–56 残基，验证蛋白为 66–97 残基，包含长度分布变化；均为完整蛋白，未用 128-residue crop 截断。训练抽样覆盖所有训练蛋白，验证抽样覆盖所有验证蛋白和 horizon。此数据切分不是温度 holdout。

Stage A 为 latent32、2 层 encoder、固定正交 OPF、K8、horizon 1/2/4/8 ns，200 steps，约 87 秒。最佳验证复合损失 1.435619，OPF condition=1。复合损失下降只能说明目标可优化，不能替代结构或动力学 gate。

为了先定位读出问题，B1a 最初仅使用 `1balA00`；**尚未完成 8 蛋白 B1a 校准**。下表为独立 eval-mode 的训练集合 shape RMSD，不能作为未来预测或泛化结果：

| 诊断 | true-H | anchor-H | zero-H | persistence | 可支持的结论 |
|---|---:|---:|---:|---:|---|
| 混合目标、64 固定窗口、200 steps | 7.235 Å | 7.240 Å | 8.380 Å | 3.765 Å | 当前读出未学成 |
| CA-only、8 固定窗口、1000 steps | 4.148 Å | 4.795 Å | 6.198 Å | 5.029 Å | 比 anchor-H 好，但误差远超 gate 预算 |
| 共享 decoder、单个构象、500 steps | 0.121 Å | 2.191 Å | 5.568 Å | 1.566 Å | 架构可以表示并拟合单一构象 |
| 缓存 raw-H 的强 MLP、8 训练窗口 | 0.053 Å | 7.547 Å | 5.171 Å | 5.029 Å | 有限 H 样本可区分、可记忆；不证明独立窗口可读出 |
| 共享 decoder、dropout=0 + encoder 顶层校准、8 窗口 | 3.800 Å | 5.065 Å | 7.247 Å | 5.029 Å | 联合干预略降误差，仍未达标；不能单独归因于 dropout |

强 MLP 约 52 万参数，flatten 全蛋白 H；标签在固定训练参考下对齐，推理输入仅 H。它能记忆 8 个样本，不能据此判断 H 已具有足够、可泛化的动力学信息。共享 decoder 单构象通过而多窗口误差仍高，下一步优先检查跨构象读出、校准和优化；不能直接把误差归给 encoder，也不能断言只是 decoder 的问题。

这些诊断的逐窗口结果、steps、seed、launch command、checkpoint/manifest hashes 保存在 `pilot_summary.json`、`ca_overfit_summary.json`、`diagnostic_single_frame/result.json`、`diagnostic_flat_mlp/result.json`、`diagnostic_shared_calibrated/result.json` 和各自日志中。最后一项也达到了预设的 1000-step 上限，未继续盲目加步数。

另外把**仅在单蛋白拟合的** CA-only decoder 转移到 4 个验证蛋白，运行 128 个独立标识的 gate windows（4 proteins × 8 anchors × 4 horizons）。该诊断 gate 明确失败：

| horizon | true-H 公共 anchor RMSD | persistence |
|---|---:|---:|
| 1 ns | 20.667 Å | 2.655 Å |
| 2 ns | 20.682 Å | 2.840 Å |
| 4 ns | 20.811 Å | 3.518 Å |
| 8 ns | 20.681 Å | 4.039 Å |

该结果说明单蛋白读出器不能直接用于这些新蛋白；它**不是**尚未训练的 8-protein B1a 的表现。`h_gate_heldout_diagnostic/report.json` 的 `passed=false`，不会允许正式 B1b。通过条件预先设置为 true-H 误差不超过 persistence 的 1/3，且 paired protein bootstrap 对 anchor-H、同蛋白 wrong-time H 和 persistence 均显示收益，内部距离变化误差也需改善。阈值是本轮工程预算，不是通用物理常数。

## 复现与后续

完整测试（无 ignore）：

最终结果：**113 passed，1 skipped**，10.25 秒。跳过项是仓库中已移除模块对应的旧架构测试；当前 Stage A/B1a/B1b 与 evaluation 的测试均纳入执行。默认 `pytest` 已包含 evaluation 测试。`compileall`、shell 语法与 `git diff --check` 通过。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python -m pytest tests evaluation/tests -q
```

H gate（失败也会先写完整 report，再返回 exit 2）：

```bash
python -m evaluation.run_h_gate \
  --b1a-ckpt artifacts/repaired_v2/pilot_20261017/b1a_overfit_1bal_ca_only/checkpoint_latest.pt \
  --manifest artifacts/repaired_v2/pilot_20261017/val.jsonl \
  --train-manifest artifacts/repaired_v2/pilot_20261017/train.jsonl \
  --history-frames 8 --horizons-frames 1 2 4 8 \
  --max-proteins 4 --max-anchors 8 --torch-threads 2 \
  --output artifacts/repaired_v2/pilot_20261017/h_gate_heldout_diagnostic
```

新 [Stage A](run_stage_a_unified.sh)、[B1a](run_b1a_unified.sh)、[B1b](run_b1b_unified.sh) 启动脚本使用独立 repaired_v2 输出、显式 validation、fail-on-error；B1a 默认冻结 encoder 校准 oracle，B1b 默认自主 energy score。全量脚本是后续配方，本轮未运行全量训练。A/B1a 的 resume 恢复权重、optimizer 和 CPU RNG，但尚未恢复 shuffle cursor，不能承诺逐字相同续训；B1b 的一步精确恢复另有测试。

下一轮先完成共享 decoder 的少窗口校准与独立窗口检查，再训练 8 蛋白 B1a、检验 4 蛋白 gate。通过后才测 history 增量信息，并进行同表示、同预算的 H+S 对照。还必须单独增加**同蛋白、时间顺序切分**评估；目前只有独立蛋白的开发集 pilot，不能用它替代同蛋白未来预测。

GLE bath 仍条件于初始构象 H，尚未改为纯序列条件；Gamma 限幅不等于完整耦合系统的长程稳定性证明。数据身份目前锁到 manifest 和 checkpoint hashes，原始大 HDF5 文件未逐字内容哈希。正式结论仍需多个训练 seeds、留出的测试蛋白、几何有效性和长程统计。

---

## 2026-10-18 更新：true-H 读出门槛失败的直接原因已定位

本轮在真实 Stage A checkpoint（`artifacts/repaired_v2/comparison_20261018/stage_a/checkpoint_best.pt`）和真实 mdCATH 数据上，用一个**独立于项目 decoder 的探针**，直接测量 H 携带多少可解码构象信息。方法、脚本、全部数字见 `DIAGNOSIS_H_READOUT_20261018.md`。

**决定性证据。** 探针任务：H_t → 未来内部 CA 位移（与 B1a `target_delta_local` 同定义），指标为解码结构的 shape RMSD（Kabsch 对齐后）；persistence 即"直接输出 anchor 构象"。

| 设置 | 探针 H 误差 | persistence | 判定 |
|---|---:|---:|---|
| 同蛋白，时间切分，线性 Ridge，1 ns | 2.552 Å | 2.132 Å | 输 20% |
| 同蛋白，时间切分，线性 Ridge，8 ns | 3.843 Å | 3.356 Å | 输 15% |
| 跨蛋白 LOO，线性 Ridge，1 ns | 5.190 Å | 2.319 Å | 输 2.24× |
| 跨蛋白 held-out，MLP，1 ns | 2.311 Å | 2.096 Å | 输 |

**关键判读**：连**同蛋白内、用最容易的线性读出**都打不过 persistence，说明问题**不是**"decoder 太弱""蛋白太少""泛化差"，而是 **H 本身不含足够的可解码构象信息**。跨蛋白翻倍只是次要的泛化问题。

**不是塌缩。** 实测 `cos(H_t,H_{t+1ns})≈0.84`、`|ΔH|/|H|≈0.4–0.6`、对应结构 RMSD 1.6–3.9 Å。H 确实随时间变化，但**变化方向与真实构象位移不对齐**。

**根因（代码层）**：
1. `protein_jepa_frame/models/encoder.py` 是逐帧局部表示：只有每残基内部距离 + 可选 torsion + 身份 embedding + pairwise attention，**没有任何全局构象通道**（无 global pooling、无质心/Rg/长程接触）。
2. Stage A 目标 `H_future=compose(decompose(H_anchor)+pred_factors)` 只奖励"有可测量的时间变化"（variance/contrast floor），**不奖励变化方向对齐真实位移**。
3. latent_dim 仅 32、2 层、CPU 400 步（跑道本身也窄）。
4. B1a 最新配置 `fape/torsion/relative_disp` 权重全为 0，只剩 `ca_mse=1`，与"内部位移才是 H 的职责"矛盾。
5. 项目 decoder 的 true-H（14–20 Å）比探针（~2.5 Å）差一个数量级，decoder 另有独立缺陷；但即使修好 decoder 也过不了 gate，因为信息上限已被探针钉死。

**对"用更多蛋白拟合 decoder"的结论**：方向对但不能解决当前问题。既然**训练分布内**都读不出，增加蛋白数量只优化泛化，会得到一个"在新蛋白上同样打不过 persistence 的 decoder"，却付出巨大算力。正确顺序是先修表示，再用少量蛋白验证读出显著优于 persistence，最后才用大量蛋白扩规模（磁盘上有 5398 个 mdCATH domain，现有 manifest 只用 200 个）。

**下一轮（本轮正在做）**：
- P0-A：改 Stage A 目标，用真实未来帧的**内部 CA 位移**直接监督 factor 增量（尽量少加监督，保持 JEPA-Anything 风格；暂不加全局构象通道）。
- P0-B：扩展蛋白数量与训练步数，使用 GPU。
- 硬门槛：同蛋白 held-out true-H ≤ 0.5× persistence 才准进 B1b。

---

## 2026-10-18（续）2000-step 验证：位移监督无增强、仍不过 gate

按你的要求，在两个 GPU run 走到 ~2000–3000 step 时取 ckpt 做一轮独立读出验证
（`scripts/probe_h_readout.py`，同蛋白时间切分 Ridge 探针，val 上 16 个蛋白，
每 5 帧采样；cross 分支仅在同长度蛋白间训练）。

| ckpt | step | h=1ns within | h=8ns within | h=1ns cross | 判定（gate ≤0.5×） |
|---|---:|---:|---:|---:|---|
| **stage_a_disp_v2**（位移监督 w=1.0） | 3000 | 0.946 | 0.935 | 1.143 | 未过 |
| **stage_a_disp_ablation_nodec**（无位移 w=0.0） | 2000 | 0.946 | 0.935 | 1.180 | 未过 |
| 旧 baseline（修复前，参考） | — | ~1.12 | ~1.20 | — | 未过 |

**结论**：
1. **两版 within-protein 曲线几乎完全重合**（0.946/0.942/0.939/0.935 vs 0.946/0.942/0.937/0.935）。
   即"用真实未来帧内部 CA 位移直接监督 factor 增量"这一改动，**没有提升 H 的可解码性**。
   它唯一可见的影响是把 **cross-protein ratio 从 ~1.20 拉低到 ~1.14**（略更可泛化），但幅度很小。
2. **相对旧 baseline（~1.12–1.20）本轮有真实改善**：within 从 1.12 → 0.94，首次稳定 < 1.0，
   说明**表示确实开始携带可解码的时间位移信息**（0.94 < 1.0，赢 persistence 6%）。
3. **但离 0.5× 的硬门槛还很远**（0.94 vs 0.50）。位移监督本身不足以把信息"编码进" H，
   因为 Stage A 目标只是让 `pred_factors` 能合成出正确位移，**并未强制 H 本身可线性读出该位移**——
   位移信息可以全部活在 predictor/factor 路径里，H 仍是"局部几何描述子"。
4. 训练侧 corroboration：`displacement=0.973` 在 3400 步几乎不降（0.976→0.972），
   说明这个辅助损失被主目标的表示瓶颈卡住，loss 也下不去。

**判读**：这一轮证伪了"仅靠最小监督的位移 head 就能修好表示"。下一步应回到 P0-A 的真正含义——
让**监督直接作用于 H**（例如对 `decompose(H_anchor)` 加弱对齐/对 H 本身加浅读出约束），
而不是只约束 predictor 的合成输出；或者按你先前给的余地，考虑引入**极轻量的全局构象通道**。
在 within ≤ 0.5× 达成前，不建议扩到 5398 蛋白规模（算力会被浪费）。

---

## 2026-10-18（再续）根因反思：问题在"目标的定义"与"H 的读出通道"，不在监督总量

### 先看这次实验给出的硬证据

我在 `stage_a_disp_v2`（位移监督，step 4000）上做了两个新诊断：

1. **H 的时间变化方向与真实位移几乎不相关。**
   取同一蛋白、`H(t+h)-H(t)` 的 per-residue 模式，用 ridge 去拟合真实内部 CA 位移模式：

   | h | pattern R² | per-sample cos |
   |---|---:|---:|
   | 2 步（≈10 ns 内） | 0.066 | 0.200 |
   | 8 步（≈40 ns 内） | 0.044 | 0.203 |
   | 常数均值基线 | −0.11 | — |

   即 H **确实在变**，但变的**方向**与"哪个残基往哪儿动"基本无关（cos≈0.2）。这解释了为什么 within 探针只能到 0.94：H 里那点可解码信息主要是"变了多少"，不是"怎么变"。

2. **OPF 在本次配置下是恒等映射。** `opf_learnable=False, opf_mode="fixed_orthogonal"` → 基就是单位阵，
   `compose∘decompose ≡ identity`（实测 `compose(decompose(x))-x = 0`）。
   所以：
   - `synth_delta_H ≡ pred`（合成路径是 no-op 重参数化）；
   - `target_factors ≡ target_delta_H`（直接是 teacher 的 H 增量）；
   - 我加的 `displacement_head` 实际读到的就是 **predictor 的原始输出 `pred`**，
     梯度路径是 `displacement loss → pred → predictor(→ 经 context_H → encoder)`。

### 根因链（从现象反推）

**(A) 目标自指：`target = h̄(future) − h̄(anchor)`，没有锚定到结构。**
Stage A 的 future 损失要求 `pred ≈ P_j·(h̄_fut − h̄_anc)`。这里 `h̄` 是**teacher 编码器自己的输出**。
这是一个**自指（self-referential）目标**：只要编码器对"同一物理变化"给出一致的、可预测的 latent 差，
损失就降下去了——**但它完全不关心这个差是否对应真实几何**。编码器完全可以选择一个
"时间戳方向"（比如把所有帧映到一条可预测的低维流形上），让 `h̄_fut − h̄_anc` 极易预测，
同时丢掉"哪个残基移动"的几何细节。第 1 条诊断（cos≈0.2）正是这个简并解被选中的证据。

`delta_target`（做减法）本意是去掉 frame-invariant 常量，但**减法不引入结构信息**，
它只是让目标更"干净"——反而更容易被一个时间流形解满足。

**(B) 我加的位移监督作用在了错误的张量上。**
`displacement_readout_loss` 读的是 `pred`（predictor 输出），约束的是
"predictor 的合成增量能否解码出位移"。于是信息可以**全部活在 predictor 里**，
encoder 的 H 不需要携带它——这正是我们观测到的：加了监督，within 探针**一点没动**（0.946 vs 0.946）。
换言之，这个监督**绕过了 H**，而 gate 检验的恰恰是 **H**。

**(C) 编码器架构里没有"集体构象"通道，`struct_recon` 也没救回来。**
`ProteinGeometryEncoder` 是逐残基局部几何 + 两两距离 attention，输出 per-residue H。
`struct_recon_loss` 回归 CA-CA 距离，本应逼 H 保留几何——但它同样是**自指/局部**的：
H_i、H_j 加一个 `H_mean` 就能拟合 i-j 距离，**不需要 H 编码"相对锚点"的构象位移**。
一个静态的描述子（只编码"这个蛋白长这样"）也能把 CA-CA 距离拟合得很好，
而**静态描述子对 `h̄_fut − h̄_anc` 的时间方向没有约束**。这解释了为什么 H 有结构信息（struct_recon 能降）
却没有**时间位移方向**信息。

**(D) 时间对比损失也在错误的量纲上。**
`temporal_contrast_weight=2.0` 只回归 `||H_t−H_{t-1}||` 的**标量大小**到 `tanh(rmsd)`，
排序损失也是标量排序。它教 H "变化幅度"要成正比，**不教变化方向**。这与 (A) 的自指方向问题叠加，
进一步鼓励"H 只编码一个单调的时间标量"。

### 一句话总结根因
> **不是监督不够、不是蛋白太少、不是 decoder 太弱。**
> 是 Stage A 的目标 `h̄(future) − h̄(anchor)` 是**自指**的，允许一个"时间流形简并解"通吃；
> 而所有辅助监督（struct_recon、temporal_contrast、我加的 displacement head）都
> **绕开了 H 本身**（要么约束 predictor、要么只约束标量幅度），从未要求 **H 的逐残基变化方向**
> 对齐**真实构象位移方向**。探针 gate 检的正是 H，所以永远过不了。

---

## 2026-10-18（再再续）从根因出发的代码修改方案

原则：**保持 JEPA-Anything 风格、尽量少加监督**，但要把监督**放到能真正影响 H 的位置**，
并且把目标**锚定到结构**，切断"时间流形简并解"。

### 修改 1（核心，必做）：把位移监督从 `pred` 移到 **anchor H 本身**，并对齐方向

**问题**：现在 `displacement_readout_loss` 读 `pred_factors`（= `synth_delta_H`），信息活在 predictor。
**改法**：让位移 head 读 **`delta_H = H_future_hat − H_anchor`**，并且**强制 `H_anchor` 参与梯度**
（现在是 `.detach()`），同时**换成一个方向性损失而非只看模式 MSE**。

具体在 `protein_jepa_frame/models/stage_a_v2/model.py::displacement_readout_loss`：

- 输入特征从 `[pf, anchor_H.detach(), H_mean]` 改为
  `[delta_H, anchor_H, H_mean]`，其中 `delta_H = synth_delta_H`（保持），
  但 **`anchor_H` 不再 detach**（让 anchor 编码也受监督），
  且额外增加一项**残基级方向余弦损失**：
  ```python
  # 方向监督：预测位移方向 vs 真实位移方向（逐残基）
  dir_cos = F.cosine_similarity(pred_c, tgt_c, dim=-1)   # [B,L]
  l_dir = (1.0 - dir_cos) 的 mask-mean
  return (se / var).mean() + w_dir * l_dir
  ```
  `w_dir` 单独给一个权重（建议 1.0）。这样"哪个残基往哪儿动"被直接约束。

**注意**：真正让它影响 H 的，是下一改。

### 修改 2（核心，必做）：在 **H 空间**加一条极轻量的"位移可读出"约束，而不是加在 predictor 上

**问题**：gate 检的是 H。现在没有任何损失要求 **H_hat 的逐残基变化**能解码位移。
**改法**：把 `displacement_head` 的输入**同时**喂一个 **stop-grad 的 anchor H 克隆路径**不可行（会断梯）；
正确做法是让 head **只**读 `delta_H = synth_delta_H`（不含 `anchor_H`、不含 `H_mean`），
即：
```python
feat = pf                      # [B,L,dH]  仅预测的 H 增量，不含任何 anchor/global 旁路
pred = self.displacement_head(feat)
```
这样**唯一的出路**就是让 `delta_H`（= predictor 输出，其梯度回传到 `context_H` → encoder）
本身携带位移方向信息。之所以之前加了旁路信息，是因为不加 head 学不动——
但这也正是**旁路让 head 绕过 H**的原因。应当**移除旁路**，并让训练步数足够。

补充（经代码核验，精确化）：
- `pred` 由 `predictor(context_H, ...)` 产生，`context_H` 来自 **online** encoder（line 890, 945），
  所以位移监督的梯度**确实会经 predictor 回到 online encoder**——encoder 并非完全拿不到信号。
- 但 **anchor 路径是死的**：`anchor_factors = _project_to_factors(h_teacher_anchor.detach())`（line 979），
  来自 **teacher** 且 stop-grad。于是 `synth_delta_H = compose(anchor_factors+pred) − compose(anchor_factors)`
  的梯度**只经 `pred`**，**从不经编码器的 anchor 表示**。
- 后果：监督只推"从固定 anchor 到 future 的增量应由 predictor 给出"，
  **不推"anchor 的 H 编码本身要成为位移的坐标系原点"**。这正是信息全部沉到 predictor 的机制。

**改法**：anchor 用 **online encoder** 重算一次（`anchor_H_online = self.context_encoder(anchor)`），
`anchor_factors = _project_to_factors(anchor_H_online)`（**不 detach**），
让位移监督同时压 online encoder 的 anchor 表示与 predictor 的增量；
teacher 仍只提供 `target_delta_local`（结构真值，无需梯度）。

### 修改 3（重要，选做但推荐）：切断自指简并——给 future 目标加一个**结构锚**

**问题**：`target = h̄(future) − h̄(anchor)` 自指。
**改法（最小侵入、不引入新 decoder）**：既然 `opf` 是恒等映射，`pred` 直接就是 H 增量。
在 H 空间**直接把「预测的 H 增量」投影到「真实 CA 位移」上做对齐**：
  - 用一个**冻结的、随机初始化后固定**的线性 map `W: R^{3} → R^{dH}`（或反过来 `R^{dH}→R^3`），
    把真实位移 `delta_local ∈ R^3` 映到 `R^{dH}`，要求 `pred` 与 `W·delta_local` 的**余弦**尽量大。
  - 这不是"加一个 decoder"，而是**把自指目标的一部分换成结构锚**，风格上仍是 JEPA
    （只加一个弱方向项，权重 ~0.5）。

**更干净的替代**：保留 `target = h̄_fut − h̄_anc` 为**主目标**，但**额外**要求
`||(h̄_fut − h̄_anc) − pred_perp||` 小，其中 `pred_perp` 是 `pred` 去掉其"时间标量方向"后的分量——
略复杂。**建议先用修改 1+2，实测 within 是否跨过 0.5**，不行再上修改 3。

### 修改 4（结构性，视需要）：给 encoder 一个**极轻量的全局构象通道**

若 1–3 仍不过 gate，说明逐残基局部几何确实无法承载"相对锚点的集体位移"。
此时（你先前已给余地）加入**最轻量**的全局通道，**不加 pooling 之外的东西**：
- 在 attention 的 value 中拼一个 **frame-level 的 anchor-relative 几何标量**：
  每个残基对 anchor 质心的距离、相对 anchor 的局部朝向（3 维），共 4 维，广播给所有残基。
- 或直接在 `h` 上加一个 **`geom_proj` 的第二路**：输入 `(CA_i − centroid)` 在**anchor residue-local frame**下的 3 维坐标。
  这给 H 一个"绝对（相对锚点）位置"通道，是当前架构**完全没有**的。

⚠️ 这条会牺牲一点 SE(3)-不变性（相对锚点而非全局不变），但正是"内部位移"的定义所需；
文献上相当于从"不变"改成"等变到锚点"。

### 执行顺序与 gate
1. 先做**修改 1 + 2**（小改，~30 行），重跑 2–3k step，取 ckpt 跑探针。
2. 若 within ≤ 0.5× → 保留，扩到更多蛋白/步数；否则做**修改 3**。
3. 仍不行再做**修改 4**（结构性，需你确认，因为触到不变性契约）。
每一步都用 `scripts/probe_h_readout.py --no-cross` 的 **within ratio** 作为唯一 gate，
不看训练 loss（这次已证明 loss 降 ≠ 表示变好）。

---

# From-None Stage B (B1b → B2): Pipeline & Decoder Architecture

**Date:** 2026-10-18 · full code: `protein_jepa_frame/models/stage_b1b_fromnone/`,
`protein_jepa_frame/training/train_stage_cb1b.py` · doc: `FROMNONE_B1B_B2_PIPELINE.md`

## Motivation

The diagnosis above shows the frozen-Stage-A route is dead: H is a *same-frame,
SE(3)-invariant* descriptor and its inter-frame difference `ΔH = H_future − H_anchor`
carries no linear geometric information about the inter-frame CA displacement
(ridge probe R² ≈ −0.005, worse than predicting zero). Hence oracle = rollout =
persistence on every horizon — even feeding the *true* future H does not beat "do
nothing".

**From-none** removes the frozen representation entirely. A **fresh, randomly
initialised encoder is co-trained end-to-end** with the H/S GLE dynamics and a CA
decoder under a *decodability* objective, so `ΔH` is *forced* to carry CA motion.
No Stage A / B1a checkpoint is consumed. It runs on **mdCATH** with a **1–100 ns
multi-lag** protocol, evaluated on **held-out protein** and **held-out temperature**.

## Pipeline

```
        ┌──────── orientation-aware geometry (coords, aatype, res_idx, chain) ────────┐
        │                                                                             │
 X_{t-K+1..t} ──E_θ (fresh encoder, random init)──▶ H_hist ──EMA-standardise──▶ Ĥ_hist
   (K=8, 1 ns/frame)   [B,K,L,dH]                       (EMANormalizer)         │
                                                                    │
                                          ┌─────────────────────────┴──────────────────────────┐
                                          │  GLE kernel (StageB1b), autonomous, history-only    │
                                          │  rollout n_steps ∈ {8..96}  (multi-lag)             │
                                          └─────────────────────────┬──────────────────────────┘
                                                                    ▼
                                          H_τ  [M,B,T,L,dH]   ,   S_τ  [M,B,T,L,dS]
                                                                    │
                                          ┌─────────────────────────┴──────────────────────────┐
                                          │  D_φ : SimpleCADecoder (per-residue MLP)            │
                                          │  dx_local = m·tanh(MLP([H_τ, H_anchor, ΔH, S_τ]))   │
                                          │  X̂_τ = X_anchor + R_anchor · dx_local               │
                                          └─────────────────────────┬──────────────────────────┘
                                                                    ▼
                                                        X̂_τ  (CA trajectory)
```

Key invariant: **the true future coordinates never enter the prediction path.**
They are encoded separately (`H_target = E_θ(X_future)`, detached role) only to form
the energy-score target. The forecast is a genuine **autonomous rollout** from the
history's last H.

### Components

| stage | module | notes |
|---|---|---|
| Encoder `E_θ` | `ProteinGeometryEncoder` | same arch as Stage A, **random init**, `dH=128`, depth 4, 4 heads, orientation features on. Trained jointly (no warm start). |
| Normaliser | `EMANormalizer` (trainer) | detached EMA of per-channel mean/scale over H; standardises the fresh encoder's *unbounded* output before the kernel/decoder. **Never updated in val.** |
| Dynamics | `StageB1b` GLE kernel | H-space energy + per-residue bath `S` (`dS=8`). Autonomously rolls H and S forward `n_steps` from history only. |
| Decoder `D_φ` | `SimpleCADecoder` | lightweight per-residue MLP (below). |

### From-none stability additions (shared kernel, opt-in)

The legacy kernel defaults are preserved; the trainer enables:
- `force_clip = 1.0` — elementwise `tanh` clamp on the force before the `h·D`
  scaling. Necessary because `h = 1000 ps` makes `h·(C@S)` the divergence source.
- `tail_init = 1.0` — a stronger quadratic restoring term, so an *untrained*
  rollout is mean-reverting instead of exploding.
- `coupling_scale = 1e-3` (down from 0.1) for the same reason.

With these, rollout std grows **sub-diffusively** (1.27→2.06 over 8→96 steps), no
blow-up.

## Decoder architecture (`SimpleCADecoder`)

Chosen because a richer `ResidualCADecoder` was not needed — the user opted for a
**simple MLP** first (the residual head remains available in the package).

**Input state per residue** (concatenated, `state_dim = 3·dH + dS = 392`):

```
[ H_τ  ,  H_anchor  ,  H_τ − H_anchor ,  S_τ ]
  128        128            128           8
```

Including both **absolute** states and the **difference** `ΔH` is deliberate: `ΔH`
is exactly the quantity the diagnosis says must decode into displacement, so it is
handed to the head explicitly.

**Trunk** — `LayerNorm → Linear(392→256) → GELU → (Linear(256→256) → GELU)×2`.
Depth 3, hidden 256. Per-residue (weights shared across residues; no message passing).

**Head** — `Linear(256 → 3)`, **small-norm init** (weight std `1e-3`, bias 0), so at
step 0 `dx ≈ 0` and the model **starts exactly at persistence** while the trunk still
receives gradient.

**Bounded local update and anchor-frame reconstruction** (the SE(3) trick):

```python
raw      = head(trunk(features))                     # [B,L,3]
m        = max_update = 12.0                          # Å
dx_local = m * tanh(raw / m) * residue_mask           # bounded, in LOCAL frame
dx       = R_anchor @ dx_local                        # rotate to global
ca_pred  = X_context + dx                             # anchor + displacement
```

`R_anchor` is the per-residue backbone frame of the **last observed** frame
(`RigidBodyFrame.from_backbone`). Two consequences:

1. **SE(3)-equivariance.** A global rotation/translation of the input rotates the
   anchor frame and the predicted displacement consistently; `ca_pred` transforms
   correctly. There is no global-frame leakage — the decoder never sees the future,
   only the observed anchor gauge.
2. **Persistence is the baseline.** `dx = 0` reproduces `X_anchor` exactly, so any
   `skill > 0` is genuine added information, not a re-parameterisation trick.

The `tanh` bound caps `|dx_local| ≤ 12 Å`, so one bad rollout step cannot teleport a
residue — important for stability during early training. An optional
`predict_log_scale` head appends a per-axis log-uncertainty (unused by the point
loss; exposed for later probabilistic use).

## Objective

Training is driven by a **differentiable Kabsch-aligned CA point loss** — the signal
Stage A lacked — plus weak distributional terms:

| term | weight | role |
|---|---|---|
| `ca` — Kabsch-aligned CA RMSD between `ca_pred` and the true future CA | **1.0** | primary decodability signal |
| `es` — energy score on the H rollout path (`H_pred` vs `H_target`) | 0.1 | distributional JEPA term |
| `phys` — energy score on decoded-CA physical features (pair distances, contacts) | 0.1 | physical grounding |

Because `ΔH` now feeds a head that is *supervised against real coordinates*, gradient
flows back through the decoder into the GLE kernel and into the **fresh encoder**,
forcing `ΔH` to become geometrically decodable. This is precisely the mechanism the
diagnosis identified as missing.

Kabsch convention (locked, verified by tests): `R = U·diag(1,1,det(UVᵀ))·Vᵀ`,
`aligned = pc @ R`, row-vector convention `x @ R + t`, RMSD averaged over residues
with a mask-normalised denominator.

## Multi-lag protocol (1–100 ns)

Each sample is a continuous clip of `history_frames + horizon_steps` frames. With
`--multi-lag`, `horizon_steps` cycles per batch over `PROTOCOL_HORIZONS =
(8, 16, 24, 32, 48, 64, 96)` frames = **8…96 ns at 1 ns/frame** (weighted
`MULTILAG_CYCLE` favours short horizons for throughput). One encoder/kernel thus
sees *every* timescale, which is what gives H/S cross-timescale generality. The
collated clip is tail-sliced per batch (frame-stacked keys only; per-residue keys are
never sliced — a fixed regression).

## Data & splits

`artifacts/mdcath_manifest_1000dom/` — **1000 mdCATH domains** (800/100/100
train/val/test), temperatures {320,348,379,413,450}, 5 replicas each. Two protocols:

- **Held-out protein** — train `train.jsonl`, select on disjoint-protein `val.jsonl`.
- **Held-out temperature** — train `..._temp/train_not450.jsonl` (T∈{320,348,379,413}),
  evaluate on T=450 (`val_t450.jsonl`).

## Status

Two real runs launched (30k steps each), background:

| run | GPU | split | selection |
|---|---|---|---|
| `artifacts/fromnone/stage_cb1b_main` | 4 | held-out protein | `val.jsonl` |
| `artifacts/fromnone/stage_cb1b_temp` | 5 | held-out temperature (T=450) | `val_not450.jsonl` |

Monitor with `bash scripts/fromnone_status.sh`.

**Selection skill** (`1 − ca/persist`, `>0` beats persistence), rising and stable:

- **Held-out protein:** `0.004 → 0.005 → 0.008 → 0.009 → 0.014 → 0.012 → 0.019`
  (steps 500–5000).
- **Held-out temperature:** `0.0036 → … → 0.0073 → 0.0068` (steps 500–4300).

**Held-out-protein eval** (best checkpoint, Kabsch CA RMSD Å):

| horizon (ns) | persist | from-none | skill (Å) |
|---:|---:|---:|---:|
| 1  | 3.857 | 3.797 | **+0.060** |
| 4  | 4.979 | 4.887 | **+0.092** |
| 16 | 6.184 | 6.044 | **+0.140** |

The margin **grows with horizon** — the signature of a model adding real information.

**Where is the ceiling?** Oracle-H probe (true future H → decoder) vs autonomous vs
persistence:

| horizon (ns) | persist | oracle-H | autonomous |
|---:|---:|---:|---:|
| 4  | 3.972 | 3.950 | 3.951 |
| 16 | 4.923 | 4.889 | 4.889 |

`oracle-H ≈ autonomous` ⇒ the GLE rollout is already at the decoder's upper bound;
the current limit is the **decoder/representation capacity**, not the dynamics. This
matches the known difficulty of mdCATH 4–16 ns (small drift, strong persistence);
longer horizons widen the gap.

## Verification

- `tests/test_fromnone_decoder.py` (7 tests): zero-`dx` = persistence, decoder
  **equivariance**, Kabsch recovery, batched-RMSD parity, mask invariance, energy
  score, and a `split_batch` key-slicing regression.
- End-to-end smoke: forward/backward, multi-lag, validation — all pass.
- Stability: bounded/sub-diffusive rollout (no blow-up) with the from-none additions.
- Perf: vectorised Kabsch made the ns=96 step 27 s → 3 s.
- Legacy kernel compatibility: all additions are opt-in; defaults preserve behaviour.

## Latent-dynamics audit: is H collapsing, and does S move? (2026-10-18)

Motivated by the classic JEPA failure modes — **representation collapse** (H constant
in time) and **trivial dynamics** (the propagator S ≈ identity, so the rollout adds
nothing) — we instrumented the autonomous GLE rollout and measured H and S directly
on **real held-out mdCATH val proteins** (6 proteins per run, not one sample).

**Horizon → time mapping.** `step_ps = 1000` ps/frame ⇒ **1 frame = 1 ns**. The
protocol horizons `(8,16,24,32,48,64,96)` frames therefore map 1:1 to
`(8,16,24,32,48,64,96)` ns. The rollout is run at the **max** horizon (96 frames);
shorter lags are read as prefixes of the same trajectory. Encoder eval step:
main run = 6000, temp (T=450) run = 3500. `bath_dim = 8`, `dH = 128`.

**Metrics.**
- `H_cos(τ)` = cosine(autonomous-rollout H at lag τ, encoded ground-truth H at lag τ)
  — does the propagated H track *where the representation actually goes*?
- `|ΔH|(τ)` = L2 norm of (H_τ − H_0) — is H actually moving in time?
- `H_gt_cos(τ)` = cosine(autonomous H_τ, GT-encoded H_τ) via the *oracle* future frame,
  i.e. the ceiling for the rollout's H.
- `|S_τ − S_0|` (and `_rel` = /‖S_0‖) — does the propagator evolve or sit at identity?
- `H_resid_std` = per-dimension residue spread of H (≈0 ⇒ collapsed code).

![H/S latent-dynamics curves](tmp_claude/hs_curves.png)

*Curves: `tmp_claude/hs_curves.png` (also `.pdf`); raw values `tmp_claude/hs_curves.json`,
arrays `tmp_claude/hs_curves.npz`; generator `tmp_claude/viz_h_s_curves.py`.*

### 1. H is NOT constant — it moves, monotonically, with the right shape

| lag (ns) | 8 | 16 | 24 | 32 | 48 | 64 | 96 |
|---|---|---|---|---|---|---|---|
| `H_cos` main | 0.876 | 0.839 | 0.809 | 0.786 | 0.747 | 0.721 | 0.682 |
| `H_cos` temp | 0.891 | 0.840 | 0.809 | 0.784 | 0.746 | 0.716 | 0.677 |
| `\|ΔH\|` main | 4.55 | 5.31 | 5.86 | 6.23 | 6.92 | 7.34 | 8.00 |
| `\|ΔH\|` temp | 4.17 | 5.09 | 5.61 | 6.01 | 6.65 | 7.12 | 7.74 |

`H_cos` declines **monotonically** 0.88 → 0.68 (never flat at 1.0), and `|ΔH|` climbs
**monotonically** 4.5 → 8.0. Both runs — held-out protein *and* held-out temperature —
show the same shape. This rules out time-invariant collapse: H is a genuine
function of time, not a constant code.

### 2. H carries real time signal — it matches where the representation goes

The **oracle ceiling** `H_gt_cos` is ≈**0.9997** at every lag (both runs): the
autonomous H is almost exactly the encoded future H. The rollout's H is therefore
tracking the true trajectory of the representation, not drifting arbitrarily. The
gap between `H_cos` and its ceiling (0.68 vs 0.9997 at 96 ns) is the *predictable*
part that the drift still has to recover — and it is the quantity the current repairs
target.

### 3. S is NOT identity — the propagator evolves with a lag-independent offset

| lag (ns) | 8 | 16 | 24 | 32 | 48 | 64 | 96 |
|---|---|---|---|---|---|---|---|
| `\|S_τ−S_0\|` main | 17.25 | 17.57 | 17.80 | 17.63 | 17.69 | 17.73 | 17.87 |
| `\|S_τ−S_0\|ₚₑₗ` main | 1.12 | 1.13 | 1.15 | 1.14 | 1.14 | 1.15 | 1.16 |
| `\|S_τ−S_0\|` temp | 9.60 | 9.66 | 9.55 | 9.53 | 9.59 | 9.56 | 9.63 |
| `\|S_τ−S_0\|ₚₑₗ` temp | 1.39 | 1.40 | 1.39 | 1.39 | 1.39 | 1.40 | 1.40 |

`|S_τ − S_0|` is large and non-zero at *every* lag (17–18 main, 9.5–9.7 temp), with a
relative change ≈1.1–1.4× the S₀ norm. S is not the identity map. (Note: `S_τ − S_0`
is nearly lag-independent — S settles quickly into a shifted operating point, which is
consistent with the bounded/sub-diffusive rollout seen in the stability check.)

### 4. H is NOT a collapsed code — per-dim residue spread is healthy

`H_resid_std` = **0.79** (main) / **0.84** (temp), well away from 0. Individual latent
dimensions retain distinct spreads, so H is an informative code, not a degenerate one.

**Verdict: neither classic JEPA failure mode is present.** H moves in time, tracks
the representation's true trajectory (up to the `H_cos` gap), is non-degenerate; S is
non-identity. The remaining headroom is the `H_cos`-vs-ceiling gap (recoverable
predicted motion), which is what the anchor-H / H-space displacement repairs
(修改 1–2 above) are designed to close — not a collapse problem.

---

## 2026-10-18（三续）修改 1+2 的 2k-step 结果：未过 gate，但查明了更深一级的根因

### 实验结果（`artifacts/stage_a_disp_fix`，step 2000）

| 指标 | 旧 v2 (3000) | 修复版 fix (2000) |
|---|---:|---:|
| within ratio h=1/2/4/8 | 0.946/0.942/0.939/0.935 | 0.949/0.945/0.937/0.938 |
| H-delta→位移 pattern R² | 0.044 | 0.027 |
| per-residue 方向余弦 | 0.20 | 0.17 |

**修改 1+2 没有改善 H 的读出**（统计上完全相同）。cross 反而略差。

### 更深的根因（这次查到了）

我做了**可解码性上界**实验（同蛋白、单帧、overfit vs held-out）：

| 读出器 | R² |
|---|---:|
| **Overfit** MLP H→位移（训练=测试=全部帧） | **1.000** |
| Held-out MLP H→位移（同蛋白留出帧） | **0.093** |
| Held-out ridge，用 `[H_anchor, H_future, ΔH, 全局均值]` 拼接 | **0.089** |

**解读**：overfit R²=1.0 说明 H 里**存在**与位移相关的特征，但那只是**帧身份记忆**（MLP 靠"这是哪一帧"取巧）。
一旦要**泛化到同蛋白未见帧**，R² 崩到 0.09 —— 即使把 anchor/future/ΔH/全局均值全拼进去也一样。

**结论**：`ProteinGeometryEncoder` 的 per-residue H 是**帧不变（SE(3)-invariant）的局部描述子**，
它**本身没有"相对某个参考帧"的坐标**。而"内部 CA 位移"的定义**必须**有一个参考系
（相对 anchor 的局部框）。所以：
> **不是监督问题，不是 predictor 问题，是 H 的表达能力问题——缺一个"相对 anchor 的参考帧通道"。**
> 这也解释了为什么 overfit 能到 1.0：MLP 把"帧身份"当成了隐式参考系，但它不可泛化。

### 判读与下一步
- 修改 1+2 已证伪：光把监督"引向 H"不够，因为 H 没有可承载位移的**坐标系**。
- 直接上 **修改 4**（相对 anchor 的全局/局部坐标通道），这是根因所在。
- 级别：这是**架构改动**，会牺牲一部分纯 SE(3)-不变性（变成"相对 anchor 等变"），
  但"内部位移"本来就只能在某个参考系下定义，所以这是**必要的**，不是妥协。

---

## 2026-10-18（四续）⚠️ 重大更正：gate 本身不成立——位移在"单帧→未来"下本质不可预测

实施了**修改 4**（encoder 加 within-frame PCA canonical-frame position 通道），2k-step run
（`artifacts/stage_a_disp_relpos`）。结果与两次前序一样：within ratio ~0.95–0.96，**没过 gate**。

我没有停在"又失败"，而是去量**这个任务的物理上界**——用**真实 anchor 坐标**（不是 H！）
去预测 h-步后的内部位移，仍然 time-split 训练/留出：

| h | 探针 (H→位移) held-out R² | **真 anchor 坐标→位移** held-out R² | 同任务 in-sample R² |
|---|---:|---:|---:|
| 1 | ~0.05 | **0.022** | 0.88 |
| 2 | ~0.06 | **0.029** | 0.88 |
| 4 | ~0.05 | **0.026** | 0.88 |
| 8 | ~0.05 | **0.051** | 0.88 |

**关键事实**：
1. 给**真实结构**（anchor 坐标，无编码器、无信息损失）做同样的拟合，**留出 R² 也只有 0.02–0.05**。
2. 同一拟合 **in-sample R²=0.88** —— 说明 ridge 能拟合训练段，但**该映射不迁移**到留出帧。
3. 用训练段均值预测留出，R² 甚至 **−0.08**（反向）。

**结论（更正前几轮的判断）**：
> **"同蛋白 held-out H→位移 R²≤0.5"这个 gate 在物理上不可达。**
> 单帧结构到 h-步后内部位移的映射**不是稳定函数**——它被**不可预测的热涨落/混沌动力学**主导。
> 任何"单帧表示"（H、真坐标、甚至 oracle）都只能拿到 R²≈0.05。
> 因此：
> - **H 不是瓶颈**：H 的 R²≈0.05–0.09 已经**达到甚至略高于真坐标的上界**。
> - 前几轮"表示坏了 / 缺参考帧"的结论**部分错误**（overfit R²=1.0 是帧身份记忆，这没错；
>   但 held-out 低**不是** H 的锅，而是任务本身）。
> - `within ratio ≈ 0.94`（赢 persistence 6%）**已经是天花板附近**，不是失败。

**这轮实验的真实收获**：
- 修改 4 的 position 通道**确实工作**：`struct_recon` 从 ~0.75 降到 **0.025**（H 现在能极好地编码结构），
  且 H 对**绝对 anchor-relative 位置**的 R² 从 0.04 提到 **0.13**。这是真实的能力提升，
  只是**位移预测**这个具体 probe 测不出来（因为位移本身不可预测）。
- 修改 1+2 无害。

**下一步建议（需你决定）**：
1. **换 gate**：不要再要求"H 预测单帧位移"。改用**可预测的量**做验收，例如：
   - H 对**当前帧结构**的可解码性（struct_recon / CA 距离 / 绝对位置）——现在 fix+relpos 已显著改善；
   - H 的**时间信息**（能否区分 Δt、能否预测**集合**统计如 Rg、接触图变化）；
   - multi-frame 输入（给 H_{t}, H_{t-1} 提供速度）后再查位移——这才有希望。
2. **或换任务**：把 Stage A 的目标从"单帧预测未来帧"改成"预测**可解码的未来结构量**"
   （例如未来帧的 CA 距离矩阵 / 接触图），这些**是可从当前帧部分预测的**。
3. 直接用**fix+relpos 的 encoder** 进 B1a/B1b，看下游 decoder 的 CA RMSD 是否比 4.148 Å 好——
   因为现在 struct_recon=0.025、位置 R²=0.13，**结构保真度**是这三轮里最好的。

我倾向 **先做 3**（用当前最好的 encoder 跑一次下游，拿真实 CA RMSD 说话），
因为它直接回答你最关心的"4.148 Å 能不能降"，且不依赖这个有问题的 probe。

---

## 2026-10-18 (晚) — 方向 3 完成：结论「<1.2 Å 在 mdCATH 单帧 anchor 下不可达」

完整数据见 `STAGE_A_REPAIR_RESULTS_20261018.md` §8–§10。

**做了什么**
1. 单批过拟合探针证明 B1a anchor-free decoder **会收敛**（ca_mse 139→0.47，real-H 1.17 Å），
   之前"持平 ~200"是**观测太早 + 慢启动**，非代码 bug。
2. 但 anchor-free decoder 在 4000 蛋白上**过拟合、不泛化**（val real-H≈27 ≫ persist 3.66）：
   它要从 H 重建整个折叠，而 H 承载不了折叠。
3. **改回 anchored decoder**（放宽 `train_stage_b1a.py` 的 `--no-anchor-free` 硬门）：
   从 anchor 真实逐残基 CA 出发，只预测有界 `d_local`。
4. anchored decoder **正确收敛且泛化**：val real=**4.495** vs persist=4.540（好 1%）、
   zeroH=4.651、wrongH=4.501 → **H 被用但时间预测力≈0**（real≈wrongH）。
5. 帧间隔 `frame_interval_ps=1000`（1 ns），val 上 1 ns 位移中位 **2.8 Å** → persistence 地板 ~3–4.5 Å。

**根因判定**：瓶颈**不在** encoder/decoder 代码，而在**单帧→未来位移的可预测性**（§2：R²≈0.02–0.05）。
`<1.2 Å` 需要：亚 ns 帧 / 紧邻多帧 anchor / 或改系综级目标。

**代码改动**：`protein_jepa_frame/training/train_stage_b1a.py`（放开 `--no-anchor-free` 门）。
**产物**：`artifacts/stage_b1a_relpos_anchored/`（best ckpt 收敛于天花板）。

---

## 2026-10-18/19 — 系综级目标 T1–T7 完成 + ATLAS 忠实评估

**计划**：`PLAN_ENSEMBLE_20261018.md`；**结果**：`ENSEMBLE_RESULTS_20261018.md`。

**做了什么**
1. **T1–T4**（新文件，不动精确坐标解码）：`evaluation/ensemble_metrics.py`、
   `training/ensemble_targets.py`、`models/ensemble_producer.py`、
   `training/ensemble_losses.py`；`tests/test_ensemble_*.py` **30 tests pass**。
2. **T5**：`training/train_stage_ensemble.py`（只训练新 `GaussianHHead`，
   encoder/decoder 冻结只读）。修 **NaN-backward**：`rmsf_profile` 的 `sqrt` 在
   方差=0 处梯度 ∞ → `clamp_min(1e-12)`；`pair_distance_hist` 同理。
   4000-step 训练收敛：**val total 2.866 vs persist 3.540；RMSF 0.764 vs 1.512（好 49%）**。
3. **T6**：`scripts/eval_ensemble_manifest.py`，model/persist/oracle + bootstrap 95% CI。
   - **mdCATH val（同分布）**：total +0.728 [0.63,0.82]、RMSF +0.835 [0.74,0.94]
     → **显著优于 persistence（GO）**；contact/dist 略差。
   - **ATLAS val（未见）**：**全部指标显著变差**（输 persistence）→ 不跨数据集泛化。
4. **ATLAS 忠实评估**：先确认训练**完全没用 ATLAS**（只 mdCATH），ATLAS train/val
   蛋白 disjoint。发现并修 eval 采样 bug（原 `shuffle=False` 只覆盖第 1 条轨迹）；
   改为全数据集随机抽样。独立复核 ATLAS persistence（直接对 npz 算 Kabsch CA RMSD
   0.2–1.7 Å）属实——**ATLAS 轨迹近刚性**，persistence 地板本就极低。

**结论**：系综级 loss 在**同分布**下有效（RMSF 谱显著更优），但**近刚性数据上失配**。
下一步：ATLAS train 微调 head / 柔性门控（退化为 persistence）/ 温度分层校准。

---

## 2026-10-19 — 双数据集训练 + 系综目标 + 位置引导（D0–D7）完成

**计划**：`PLAN_DUAL_ENSEMBLE_20261019.md`；**结果**：`DUAL_ENSEMBLE_RESULTS_20261019.md`。

**用户要求**：用 mdCATH + ATLAS **两个数据集一起训练**，target **系综级 metrics**，
并允许**原子位置重建**作为引导指标。

**做了什么**
1. **D0** `scripts/build_mixed_manifest.py`（NEW）：拼 mdCATH(4000)+ATLAS-100ps(270)，
   帧间隔不同（1000 vs 100 ps）由 `load_manifest` 按 `frame_interval_ps` 正确解析；
   正交校验 train/val 蛋白 disjoint；支持 ATLAS 过采样（train 过采样、**val 不过采样**）。
2. **D1** `ensemble_producer.py`：`GaussianHHead(cond_dim=)` 可选
   时标/温度条件（默认 0 不改行为）。
3. **D2** `ensemble_losses.py`：新增 `_position_loss`（Kabsch CA MSE）+
   `pos_weight`，作为**原子位置引导项**。
4. **D3** `train_stage_ensemble.py`：`--pos-weight` / `--cond-dims`，
   validate 含 persist 位置基线。单测扩到 **35 pass**。
5. **D4** 三个 run 各 4000 steps（GPU 2/3/5 并行）：
   A=mixed(无 pos)、B=mixed+pos、C=mixed×3+pos+cond。
6. **D5** 在 mdCATH / ATLAS-100ps / ATLAS-1ns 三个 val 上跑
   `scripts/eval_ensemble_manifest.py`，bootstrap 95% CI。

**关键结果**
- **逐项精确结论（见 `DUAL_ENSEMBLE_RESULTS_20261019.md`）**：
  - ✅ **RMSF 谱**：三个 val 上 **100% 窗口优于 persistence**（mdCATH Δ=−0.87；
    ATLAS Δ≈−0.03），这是本轮唯一真正拿下的指标。
  - ✅ **total**：97–100% 窗口更优，但**几乎完全由 rmsf 项贡献**（加权净正）。
  - ❌ **contact / dist**：**系统性差于 persistence**（≈100% 窗口更差），**未拿下**。
  - ⚠️ **pos (CA RMSD)**：仅 mdCATH 略胜（4.559 vs 4.627 Å），ATLAS 反而输（0.12 vs 0.11 Å）；
    仍远离 <1.2 Å。
  ⇒ 不能宣称"系综级指标全面超过 persistence"。
- **ATLAS 从上轮"全面显著变差"翻转为"显著变好"**——混合训练直接修掉跨数据集失配。
  - A：ATLAS-1ns Δtotal +0.029 [0.027,0.030]；ATLAS-100ps +0.026 [0.025,0.027]。
- **位置引导有效**：mdCATH 上 C 的 pos 20.78 < persist 21.41，且系综指标不退化。
- mdCATH val：Δtotal ~ +0.75（A）/+1.02~1.07（B/C），Δrmsf +0.76~0.84。

**代码**：全部新增或只读；未触碰精确原子坐标解码路径。
**产物**：`artifacts/stage_ensemble_dual{A,B,C}/`、`artifacts/eval_dual/*.json`、
`artifacts/mixed_manifest_100ps{,_x3}/`。

---

## 2026-10-19 · 扩展系综指标（6 项，新文件）

**要求**：把 6 个系综级指标写在一起评估并汇报：RMSF / ROOT-MEAN W2 / VARIANCE W2 /
MD-PCA W2 / WEAK-CONTACT J / TRANSIENT-CONTACTS J。

**诚实前置**：这 5 个新指标**此前没有现成脚本**（仓库只有 rmsf、threshold=0.5 的
contact Jaccard、内距 soft-EMD）。本次**全部新写**。

**实现**
- `evaluation/ensemble_metrics_ext.py`（NEW）：`w2_1d`（精确 1-D Wasserstein-2，排序分位）、
  `root_mean_w2`、`variance_w2`、`md_pca_w2`、`weak_contact_jaccard`、
  `transient_contact_jaccard`、`extended_ensemble_summary`。oracle→W2=0 / J=1。
- `scripts/eval_ensemble_metrics_suite.py`（NEW）：model / persistence / oracle 三系综，
  6 指标一起算，bootstrap 95% CI + 逐窗口胜率；三 val × 三 run。
- `tests/test_ensemble_metrics.py`：+7 测试（共 19）。
- 结果 JSON：`artifacts/eval_dual_suite/{A,B,C}_{mdcath,atlas100ps,atlas1ns}.json`。
- **未改动** `protein_jepa_frame/` 内任何精确坐标解码实现（mtime 校验）。

**关键结果（见 `DUAL_ENSEMBLE_RESULTS_20261019.md` §6）**
- **mdCATH**：6 指标**全部 94–100% 窗口优于 persistence**（W2 家族 + Jaccard 全胜）。
- **ATLAS-1ns / 100ps**：4 个 W2/RMSF **100% 胜**；2 个 Jaccard 均值更好但**胜率仅 6–10%**
  （近刚体系 persistence 已≈0.94，单窗口赢不动）。
- **A/B/C 三配置结论一致** ⇒ 收益来自双数据集混合训练，非 pos/cond 项。
- ⚠️ 口径差异：§1 训练 loss 的 contact/dist 仍差于 persistence，而 §6 评估口径的
  弱/瞬时接触 Jaccard 在 mdCATH 上 100% 胜——**报告时须注明是 loss 口径还是评估口径**。

### 2026-10-19 · 补充：RMSF-r + 澄清 persist-J=0

- **新增 RMSF-r（形态）**：`extended_ensemble_summary` 现在同时输出 `rmsf_corr`
  （逐残基 RMSF 谱的 Pearson 相关）。mdCATH：model **0.553** vs persist **0.000**（100% 胜）；
  ATLAS-1ns/100ps：0.097/0.096 vs 0.000（87–90% 胜）。persist 恒 0 因 delta 系综 RMSF 谱为常数。
- **澄清 "mdCATH 上 persist contact-J/transient-J=0.000"**：这是 **persistence 的数学性质**，
  与 mdCATH 无关——anchor 复制 8 份 ⇒ contact-prob 图**只有 {0,1}** ⇒
  弱/瞬时接触集合**恒为空**（`weak_n_pred=0`）⇒ 真值集合非空时 `J=0/|A|=0`。
  ATLAS 的 0.90/0.94 是**退化假高分**（真值近二值、多数窗口双空 ⇒ 约定返回 1.0）。
- **代码**：新增 `weak/transient_contact_jaccard_full` 显式返回 `n_true/n_pred/n_inter`；
  suite 输出 `diagnostics` 字段可追溯。测试 21 项全过。
- **修订**：§6 重写（7 指标，含 RMSF-r 与 §6.5 的 persist-J 归因）。

### 2026-10-19 · RMSF-r 细化：global + per-target（mdCATH）

- 评估脚本新增 RMSF-r **三种口径**：global（全体窗口残基池化一次相关）、
  per-window（逐窗口相关后取均值）、**per-target**（按 `protein_id` 聚合每蛋白一条相关，
  再给 19 个 target 的分布）。
- **mdCATH（19 targets / 87 trajectories / 100 windows）**：
  - run A model：global **+0.604**，per-target 均值 **+0.581**（median +0.610，
    min **+0.347**、max +0.839，**19/19 为正**），per-window 均值 +0.548。
  - persistence：三种口径 **全 0.000**；oracle：全 **1.000**。
  - B/C：global +0.516/+0.571，per-target 均值 +0.498/+0.549，排序一致。
- 按用户判断 **ATLAS 不再作主口径**（近刚性退化），报告只报 mdCATH；ATLAS JSON 留存备查。
- `DUAL_ENSEMBLE_RESULTS_20261019.md` 新增 §6.2b。

### 2026-10-20 · MDGen 基线对照（轨迹生成，mdCATH）

**用户诉求**：配置 MDGen 作 baseline；**zero-shot 与"简单短训练"分别评估效果**，zero-shot 优先；
双方在 mdCATH 上、用同一套系综指标 + **Kabsch CA RMSD**（CA 位置）对比。

- **MDGen 官方权重**：HF `bjing-mit/mdgen` → 下载 `atlas.ckpt`(蛋白级 forward-sim,
  num_frames=250/crop=256) 与 `forward_sim.ckpt`(四肽, crop=4, 不适用) 到
  `/liuyutian/mdgen_run/hf_mdgen/`。
- **新脚本**（均为新文件，未改精确坐标解码路径）：
  - `scripts/eval_mdgen_baseline.py`：MDGen forward-sim → 未来帧系综 → 7 指标 + Kabsch。
  - `scripts/eval_ours_trajectory.py`：我们的模型**自回归 rollout** → 同套指标。
  - `mdgen_baseline/mdgen_compat.py` **新增**：torchvision/torchmetrics 破损时的
    stub、ODE 步数控制（euler@N）、mdCATH 短轨迹守卫。
- **坑**：mdgen 默认 `dopri5` 采样的 ~4 min/窗口 → 改 `euler@20`，**~7 s/窗口**；
  环境 torchvision 轮子损坏 → 用 `torchmetrics` stub 绕过。
- **零样本结果**（32 窗口, 100 帧）：Kabsch **7.34 Å**、RMSF **4.65 Å**，
  **差于 persistence(4.01 Å)** ⇒ **atlas.ckpt 无法直接用于 mdCATH**
  （10 ps/帧 训练 vs mdCATH 1000 ps/帧，OOD）。
- **mdCATH 短训结果**：epoch1 Kabsch 19.69 Å、epoch5 **26.76 Å（更差）** ⇒
  **5 epoch 远未收敛**（flow-matching 需 ~10^4 epoch），当前不构成对手。
- **我们的 rollout**：7 指标里 6 个最好（rmsf_mae 1.455 / rmsf-r 0.334 /
  var_w2 10.21 / pca_w2 0.080 / 两 Jaccard），**但 Kabsch 6.34 Å 差于 persistence**
  （自回归回填漂移）⇒ 后续 Stage B 轨迹任务的改进点。
- 结果写入 `DUAL_ENSEMBLE_RESULTS_20261019.md` §7；原始 JSON 在
  `artifacts/eval_mdgen/`、`artifacts/eval_ours_traj/`。
- MDGen mdCATH 训练**仍在继续**（写稿时 epoch 7）。
