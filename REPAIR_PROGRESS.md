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
