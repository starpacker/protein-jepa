# H+S 表征反馈：匹配小实验结果（2026-11-01）

**最初两臂 pilot（`pilot_v1`）没有取得一致收益，也没有全面超越 BioEmu2。** Stage B 对 encoder 的真实梯度和保存恢复路径已经打通；500 步配对实验显示，更新 H 带来的 MD-PCA W2 改善伴随位置分布、涨落幅度和接触指标退步，RMSF correlation 没有可靠改善。按训练前写定的继续规则，该轮不升级为正式候选，也不直接扩大为三 seed 长训。后续空间后验诊断和四臂读出保持实验分别记录在下文，使用各自协议，不覆盖这些历史结果。

## 实验范围与可复现性

[训练前协议](HS_JOINT_PILOT_20261101.md)与[启动脚本](run_scripts/run_hs_joint_pilot.sh)固定了两臂：frozen / joint，共同 seed42、500 个优化步、同一历史 checkpoint 的内嵌组件、相同 stateless 训练窗口流与全部超参，只有 joint 打开 encoder 更新。decoder 固定；joint 使用固定起始 encoder 作为未来 latent teacher，并在观测历史保持 H 及其差分。两臂均使用 collective mobility、exact S0、noise_mix=0；因此这也是相对历史 noise_mix=.8 配方的显式改变，不能把本轮与历史结果的差异全部归因于 H 更新。

两臂验证都选中 step400，另独立评测最终 step500。标准评测使用 **200 窗口 / 50 蛋白 / 12 样本 / 四个 horizons=1,4,16,64 ns / 五个温度各40窗**，评估 seed=0。同窗 keys 与保存的 BioEmu2 参考完全一致；两臂共用随机 draws，BioEmu2 只保证相同窗口和采样预算，不能称跨架构同噪声。

任务沿用历史 **endpoint draws 对 future-path mixture（含 anchor）**，没有偷偷改变七指标定义。它不证明时间匹配的条件转移分布或有序动力学正确。该验证集多次用于开发，所有区间均为探索诊断；没有打开 test 数据，也没有重跑或改写冻结 BioEmu2 / SOTA 基线。

原始产物在 [pilot_v1](results/hs_joint_20261101/pilot_v1/)；[源码和输入哈希](artifacts/hs_joint_20261101/pilot_v1/experiment.json)、[参数审计](results/hs_joint_20261101/pilot_v1/checkpoint_audit.json)、[环境版本](results/hs_joint_20261101/pilot_v1/environment_versions.json)均已保存。训练源码快照在启动时保存；之后仅调整了未使用的“无内嵌组件”加载分支。该差异不进入本轮训练/评估路径，评估输出另存实际源码哈希。

## 七项指标

下表为等蛋白均值。每个蛋白贡献四个 horizons 的各一窗；更细的各 horizon、各温度、配对区间、几何结果见 [完整统计](results/hs_joint_20261101/pilot_v1/paired_summary.json)。

| 指标 | frozen best | joint best | frozen latest | joint latest | BioEmu2 |
|---|---:|---:|---:|---:|---:|
| RMSF MAE ↓ | 1.0394 | 1.0579 | 1.0219 | 1.0537 | 1.9917 |
| RMSF correlation ↑ | 0.4981 | 0.4983 | 0.4990 | 0.4980 | 0.5797 |
| Root-Mean W2 ↓ | 1.4985 | 1.5130 | 1.4901 | 1.5122 | 2.2904 |
| Variance W2 ↓ | 7.9542 | 8.0337 | 7.8963 | 8.0297 | 18.5754 |
| MD-PCA W2 ↓ | 7.3730 | 7.3259 | 7.3939 | 7.3428 | 6.1143 |
| Weak-contact J ↑ | 0.1032 | 0.1016 | 0.1051 | 0.1029 | 0.1316 |
| Transient-contact J ↑ | 0.2519 | 0.2479 | 0.2552 | 0.2494 | 0.2653 |

联合臂 best 在平均 RMSF MAE、Root-Mean W2、Variance W2 上优于 BioEmu2，在 RMSF correlation、MD-PCA W2 和两项接触指标上落后。这个均值方向比较不构成所有时标获胜，更不能替代完整评测套件。其 MD-PCA W2 从冻结臂的7.3730降到7.3259，仍高于 BioEmu2 的6.1143。

下面列出 joint 相对 frozen 的改善量，已统一为**正值更好**。`*` 表示10,000次等蛋白配对 bootstrap 的边际 CI95 不跨0。全部56格均报告；区间不校正多重比较，不含训练 seed 方差或 best 选点不确定度，不能把星号称为确认性显著性。

### best

| 指标 | 1 ns | 4 ns | 16 ns | 64 ns |
|---|---:|---:|---:|---:|
| RMSF MAE ↓ | -0.00950* | -0.01795* | -0.01375 | -0.03262 |
| RMSF correlation ↑ | +0.00466 | +0.00144 | -0.00221 | -0.00310 |
| Root-Mean W2 ↓ | -0.00993* | -0.01373* | -0.01270* | -0.02184* |
| Variance W2 ↓ | +0.02046 | -0.06453* | -0.06939 | -0.20469* |
| MD-PCA W2 ↓ | +0.01592* | +0.04898* | +0.06134* | +0.06242* |
| Weak-contact J ↑ | +0.00000 | -0.00262* | -0.00221 | -0.00135 |
| Transient-contact J ↑ | -0.00409* | -0.00365* | -0.00259* | -0.00549* |

### latest

| 指标 | 1 ns | 4 ns | 16 ns | 64 ns |
|---|---:|---:|---:|---:|
| RMSF MAE ↓ | -0.01233* | -0.02644* | -0.02822* | -0.06027* |
| RMSF correlation ↑ | +0.00453 | +0.00151 | -0.00302 | -0.00706 |
| Root-Mean W2 ↓ | -0.01179* | -0.01798* | -0.02316* | -0.03562* |
| Variance W2 ↓ | +0.01777 | -0.08716* | -0.12165* | -0.34241* |
| MD-PCA W2 ↓ | +0.01706* | +0.05722* | +0.08236* | +0.04812 |
| Weak-contact J ↑ | +0.00000 | -0.00319* | -0.00317* | -0.00261 |
| Transient-contact J ↑ | -0.00516* | -0.00588* | -0.00571* | -0.00650* |

RMSF correlation 的四个时标、两个 checkpoint 版本均未给出可靠正增益。best 的 MD-PCA 四时标改善，latest 的64 ns区间跨0；Root-Mean W2 在两个版本的全部四时标均增大，transient contact J 均降低，两者都退步。1 ns 真值 weak-contact 集合全部为空：本轮两臂该 Jaccard 都为0，不支持弱接触恢复的结论。

## 结构读出与几何

本轮预测评测附带的 `ca_bond` 是相邻 CA 距离相对 observed anchor 的绝对差，`ca_angle_cos_mae` 是相对 anchor 的转角余弦差。它们不是相对真实未来结构的误差；角度差增加可以包含真实运动。`ca_clash_fraction` 是非局部 CA 对距离小于3 Å的比例，与权威评测中2.5 Å cutoff区分。

joint best 的四时标碰撞比例均有小幅增加；64 ns 从0.25695%升到0.26606%。latest 的64 ns碰撞增量区间跨0，其余时标为正。两臂仍需要规范的 CA / 全主链几何评估，不能由这三项诊断宣称结构全部有效。

真实未来 H 的读出 gate 在同样200窗上重跑。阈值为本轮诊断预算 oracle/persistence endpoint RMSD≤1/3，ratio是两个等蛋白RMSD均值之比，并非逐窗口阈值。这不是对历史结果追溯修改的 canonical 规则。三个被测组件均在1/4 ns通过、16/64 ns失败；两种 joint 的读出误差还比冻结组件略高。八个joint−frozen配对比较的边际CI95全部大于0，平均误差增加0.01728–0.03870 Å。

| Horizon | 冻结 H RMSD (Å) | joint best | joint latest | 冻结 ratio | joint best ratio | joint latest ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6169 | 0.6416 | 0.6404 | 0.2800 | 0.2912 | 0.2907 |
| 4 | 0.7981 | 0.8158 | 0.8154 | 0.2932 | 0.2996 | 0.2995 |
| 16 | 1.4226 | 1.4613 | 1.4607 | 0.3570 | 0.3667 | 0.3665 |
| 64 | 2.2081 | 2.2387 | 2.2382 | 0.4266 | 0.4325 | 0.4325 |

冻结臂 best/latest 的 encoder、decoder 与 calibration配置逐项相同，因此冻结组件 gate 同时适用两者；joint best/latest各自实际评估。复用依据与 joint−frozen 的配对读出差值见 [读出比较](results/hs_joint_20261101/pilot_v1/readout_comparison.json)。该诊断使用真实未来 H，是 oracle读出而非预测成绩。它定位的是encoder＋固定decoder的误差，不能单独认定H缺少信息；其退化仍足以否定本轮“更新H同时保持读出”的要求。


## 补充：指定时间 endpoint 诊断

在主pilot结果已知后，新增独立评分工具，并在执行前固定[补充范围](results/hs_joint_20261101/pilot_v1/endpoint_diagnostic_protocol.json)：只评估两臂原本已选中的best（均step400），仍用相同200窗、50蛋白、12样本、seed0；不重新选点、不训练、不替换历史结果。真实未来端点只在采样完成后用于评分，未作为模型输入。该结果属于**事后开发诊断**。

Coordinate ES复用除以√(3N)的U-statistic；Distance ES复用canonical含对角V-statistic、最多512个固定非局部距离对；Contact Brier使用全部非局部CA对的硬8 Å事件频率。V-score和经验Brier针对提交的经验分布/概率，跨窗口平均不会消除它们相对底层生成分布的有限12样本偏差；coordinate U-score在iid采样下才是底层生成分布ES的无偏估计。真值仅一个endpoint，因此proper-score意义是跨窗口期望，不是每个窗口估得了真实条件系综。坐标只把真实endpoint对齐到observed anchor，预测不单独对齐；坐标ES会惩罚预测的整体刚体漂移，不是shape-only误差。

下面全部指标越低越好；Δ为joint−frozen，负值改善。CI仍为10,000次等蛋白配对的探索性边际区间。

| Horizon | 指标 | frozen | joint | persistence | Δ [CI95] |
|---:|---|---:|---:|---:|---|
| 1 | Coordinate ES (U) | 0.975726 | 0.970981 | 1.271944 | -0.004745 [-0.008085, -0.002016] |
| 1 | Distance ES (V) | 1.269807 | 1.267651 | 1.624493 | -0.002156 [-0.008797, +0.003446] |
| 1 | Contact Brier | 0.010817 | 0.010929 | 0.012319 | +0.000112 [+0.000079, +0.000145] |
| 4 | Coordinate ES (U) | 1.175706 | 1.171594 | 1.571875 | -0.004112 [-0.007693, -0.000921] |
| 4 | Distance ES (V) | 1.622029 | 1.625220 | 2.072172 | +0.003191 [-0.003692, +0.010396] |
| 4 | Contact Brier | 0.014325 | 0.014544 | 0.014636 | +0.000220 [+0.000151, +0.000294] |
| 16 | Coordinate ES (U) | 1.768653 | 1.763625 | 2.300956 | -0.005027 [-0.010549, +0.000472] |
| 16 | Distance ES (V) | 2.455188 | 2.460259 | 3.086986 | +0.005071 [-0.005811, +0.016907] |
| 16 | Contact Brier | 0.019482 | 0.019655 | 0.019220 | +0.000172 [+0.000057, +0.000282] |
| 64 | Coordinate ES (U) | 2.274361 | 2.270760 | 2.988185 | -0.003601 [-0.014290, +0.006303] |
| 64 | Distance ES (V) | 3.064115 | 3.062985 | 3.977112 | -0.001130 [-0.025647, +0.020852] |
| 64 | Contact Brier | 0.024225 | 0.024353 | 0.023343 | +0.000128 [-0.000133, +0.000371] |

joint在1/4 ns坐标ES有小幅改善，16/64 ns区间跨0；distance ES四时标均不确定；Contact Brier在1/4/16 ns变差，64 ns不确定。总体坐标ES从1.54861降至1.54424（约0.28%）；这与长时标读出及接触退步并存，不改变本轮不扩大训练的决定。

两臂在四时标的坐标和距离ES都优于persistence；joint的Contact Brier仅1 ns对persistence显示明确改善，其余时标区间跨0。这支持部分条件预测能力，不能隔离S的增量贡献，也不能替代时间匹配的BioEmu2基线。

此补充评估还实际计算了canonical CA几何：总体非局部2.5 Å碰撞比例frozen=0.07261%、joint=0.07550%、真值=0；两臂差的边际CI为正。CA键长越界比例分别为0.19167%、0.19247%，真值0.14737%；两臂差不确定。平均键长和Rg、各时标几何、每个样本原始结果都保留在[完整endpoint统计](results/hs_joint_20261101/pilot_v1/endpoint_paired_summary.json)及两份原始endpoint文件中；这仍是CA-only证据。

## 实现证据和研究边界

- 修复 exact posterior 在训练中的力/历史梯度断点；有限差分和真实 encoder 检查覆盖自主 forecast 梯度。
- 实现 `--train-encoder`、固定 teacher、观测历史 retention、完整 H/S 反传、独立 encoder LR，以及包含 live encoder 的 checkpoint 保存/恢复；预测 mask 只由 context 决定。
- 参数审计确认 joint 的75/75个 encoder 张量改变，best/latest相对初始L2分别为0.001579/0.001591；frozen 的 encoder、两臂 decoder、joint teacher 均保持原值。非零梯度和参数改变证明训练路径工作，不证明 H 更充分。
- 嵌入组件优先加载避免重载时依赖可变外部组件；通用 loader 补上 relative-position配置与合同检查。
- 核心改动和新增endpoint工具的全量测试 **430 passed, 1 skipped**，见 [测试记录](results/hs_joint_20261101/pilot_v1/full_tests_with_endpoint.log)。真实4步训练与“2步后中断再恢复”比较中，teacher/RNG逐位相同；参数最大差1.94e-7、验证项最大差1.19e-6，属于数值复现而非CUDA逐位确定性。原始严格逐位检查曾失败，之后保留差值报告，没有将失败改写为逐位一致，见 [恢复结果](artifacts/hs_joint_20261101/resume_probe/result.json)。
- [条件历史探针](results/hs_joint_20261101/conditional_history_baseline.json)仅在训练蛋白选正则、在独立验证蛋白评分，未检出可靠正增量。它不构成信息上界；旧“2–4%记忆上限”解释已撤回。数据划分和方法详见[研究审计](docs/RESEARCH_AUDIT_20261101.md)。
- [空间相关 bath 后验](protein_jepa_frame/models/correlated_bath_posterior.py)在上述500步pilot时仍是独立原型，未参与其训练或评分。其后已以显式 `exact_spatial` 模式接入 sampler，使用完整Cholesky抽样和有效残基packing；新模式的诊断见下文，不追溯改变pilot语义或数值。

## 后续可证伪问题

已新增[时间匹配endpoint评估器](scripts/eval_ca_endpoint_scores.py)及[7项测试](tests/test_ca_endpoint_scores.py)，报告coordinate U-statistic energy score、canonical physical-distance V-statistic energy score与hard-contact Brier，保留各自有限样本定义；规范CA几何使用3.3–4.3 Å键长有效带和2.5 Å非局部碰撞阈值。单窗口一个未来样本只能在跨窗口期望上检验proper score。它与历史七指标分开报告。之后如开展新小实验，可比较 H frozen/joint × S0 prior/exact，固定预算和观察输入，并报告所有时标、读出与规范几何；prior/exact都含bath，仅检验 S0历史条件化，不等同于 H+S 对 Markov 的主要记忆因果比较。

全面目标仍需独立保留确认、全部温度、T1主链、T2A状态布居、T2B/MSM、有序轨迹、以及适用的192 ns等任务。当前 CA-only pilot没有完成这些任务，也没有达到“各指标超越BioEmu2”。

## 复现评估

```bash
export PYTHONPATH=/tmp/lasp_stability_deps:.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# arm=frozen/joint；tag=best/latest；评估seed必须为0（训练seed为42）。
python -m scripts.eval_lasp_ca_dynamics \
  --checkpoint artifacts/hs_joint_20261101/pilot_v1/joint/checkpoint_best.pt \
  --out results/hs_joint_20261101/pilot_v1/joint_best.ensemble.json \
  --max-windows 200 --n-samples 12 --seed 0 --prediction-kind endpoint
python -m scripts.gate_ca_representation \
  --component artifacts/hs_joint_20261101/pilot_v1/joint/checkpoint_best.pt \
  --out results/hs_joint_20261101/pilot_v1/joint_best.h_gate.json --max-windows 200
python results/hs_joint_20261101/pilot_v1/audit_checkpoints.py
python results/hs_joint_20261101/pilot_v1/summarize_pilot.py
# 补充endpoint诊断；另一臂同样只改checkpoint和out。
python -m scripts.eval_ca_endpoint_scores \
  --checkpoint artifacts/hs_joint_20261101/pilot_v1/joint/checkpoint_best.pt \
  --out results/hs_joint_20261101/pilot_v1/joint_best.endpoint.json \
  --max-windows 200 --n-samples 12 --seed 0 --device cuda:0
python results/hs_joint_20261101/pilot_v1/summarize_endpoints.py
```

以上重新执行会写指定的本轮结果文件；历史冻结基线不在输出路径中。

## 后续：训练集读出保持诊断

[独立脚本](scripts/diagnose_ca_readout_retention.py)在固定训练集前64窗上比较冻结组件与joint best：32个蛋白各含16/64 ns两窗，全部anchor=3，五温度的蛋白数为10/7/5/5/5。这不是独立验证或均衡温度评测。teacher仅使用四个观测帧，实际权重为absolute=1、delta=0.1；observed readout用前三个已观测帧的真实H相对第四帧anchor解码。未来H仅进入可微oracle16/64读出诊断，未进入teacher/observed损失，没有运行自主forecast或优化器。

| 平均读出RMSD（Å） | 冻结组件 | joint best | 误差增加的蛋白 |
|---|---:|---:|---:|
| Observed（三帧均值） | 0.92723 | 0.94883 | 29/32 |
| Oracle16 | 2.12998 | 2.17176 | 29/32 |
| Oracle64 | 3.48446 | 3.51339 | 25/32 |

joint处teacher与oracle16/64参数梯度的cosine为0.37366/0.29688，observed readout对应为0.65294/0.24700。两个负梯度方向在预设相对参数范数步长1e-5、3e-5下均降低observed和两个oracle MSE；但observed方向不全面优于teacher：在3e-5时，oracle64 MSE降幅分别为teacher 0.021005、observed 0.015592 Å²。teacher的加权梯度范数为0.0670，observed为4.6233；目标单位不同，该比例不能直接用于选权重。

全部双侧有限差分结果和模型哈希保存在[训练集诊断结果](results/hs_joint_20261101/readout_diagnosis/train64_frozen_vs_joint_best.json)。16个导数检查的最大相对差异为0.394%，每次扰动从同一原参数赋值，最终逐位恢复；[两项CPU测试](tests/test_readout_retention_diagnosis.py)验证解析二次函数及正常/异常恢复。独立审查确认32个蛋白均来自固定train manifest，输入/源码哈希匹配，未访问val/test数据。

这说明teacher存在局部修复作用，但当前训练保持程度没有保证固定decoder读出稳定。observed readout可作为下一轮训练损失消融候选；它不支持替换teacher或选定权重，更不能解释历史Adam更新的因果、证明H信息损失或自主长程预测收益。冻结起点的observed/oracle64梯度cosine为−0.0508，也排除了把该方向当成全局保护保证。

## 后续：空间后验的固定权重三臂诊断

新模式 `exact_spatial` 将完整跨残基bath协方差接入StageB及CA训练/评估路径；每个观测窗口只滤波一次，再为12个ensemble成员抽取独立完整协方差样本。旧 `exact` 的独立噪声要求和既有末端jitter保持原样；新模式不添加jitter。dense Kalman参考、实际sampler Monte Carlo、非连续padding、梯度有限差分及单次/重复滤波的RNG/梯度对照均已通过。[核心全量回归](results/hs_joint_20261101/spatial_probe_v1/full_tests.log)为 **440 passed、1 skipped**；随后新增的读出诊断另有2项CPU测试通过。有效noise kernel/cutoff现已保存进run/checkpoint args，[真实CLI检查](artifacts/hs_joint_20261101/spatial_loader_check/result.json)验证了elastic/7.5的继承、零步保存/恢复，以及不一致cutoff被拒绝。

执行前固定[三臂协议](results/hs_joint_20261101/spatial_probe_v1/protocol.json)：共同使用历史step5500 checkpoint的全部encoder、decoder和kernel权重，三臂分别为`exact + noise_mix=0`、`exact_spatial + noise_mix=.8`、`prior + noise_mix=.8`；不训练、不选权重。源权重本来就在noise_mix=.8下训练，独立噪声臂属于干预，不能由比较宣称.8普遍最优。三个新checkpoint的step=0明确表示本次零更新，历史训练step另存血缘；没有把旧validation/optimizer记成新结果。

三臂均实际完成200窗/50蛋白/12样本的七指标与matched-endpoint评分，六个文件keys完全配对；全部源码快照、tensor/input/output哈希和exit0记录保存在[产物目录](results/hs_joint_20261101/spatial_probe_v1/)。使用相同基础随机种子，但不同协方差作用后的物理噪声不同。每温度只有10个蛋白，且温度组之间蛋白不重叠；这些子组不是同蛋白配对温度响应。全部区间仍为10,000次等蛋白bootstrap的探索性边际CI95，不含多重比较、训练seed或自适应开发选择不确定度。

| 七指标 | 独立噪声exact | 空间exact | 空间prior | 保存的BioEmu2 |
|---|---:|---:|---:|---:|
| RMSF MAE ↓ | 1.5498 | 1.3187 | 1.3176 | 1.9917 |
| RMSF correlation ↑ | 0.4507 | 0.5028 | 0.5027 | 0.5797 |
| Root-Mean W2 ↓ | 1.8858 | 1.6573 | 1.6584 | 2.2904 |
| Variance W2 ↓ | 10.3058 | 9.6195 | 9.6233 | 18.5754 |
| MD-PCA W2 ↓ | 7.0869 | 6.6010 | 6.6112 | 6.1143 |
| Weak-contact J ↑ | 0.0604 | 0.1070 | 0.1064 | 0.1316 |
| Transient-contact J ↑ | 0.1714 | 0.2467 | 0.2458 | 0.2653 |

空间exact相对独立噪声的七项总体均值都改善，除Variance W2外六项CI不跨0；但**1 ns的RMSF MAE和Root-Mean W2退步**。450 K子组的MAE增加0.58169 [0.49246,0.66480]、Root W2增加0.12841 [0.06837,0.18803]、Variance W2增加4.39111 [2.01145,6.36389]，均退步。完整各horizon/温度结果见[七指标统计](results/hs_joint_20261101/spatial_probe_v1/paired_summary.json)，总体均值不能代替全组胜出。

| Matched endpoint，总体，越低越好 | 独立噪声exact | 空间exact | 空间prior | 空间exact−独立噪声 [CI95] |
|---|---:|---:|---:|---|
| Coordinate ES (U) | 1.651324 | 1.535685 | 1.539490 | −0.115639 [−0.165661,−0.059730] |
| Distance ES (V) | 2.543159 | 2.141910 | 2.146208 | −0.401249 [−0.503378,−0.292159] |
| Contact Brier | 0.024924 | 0.016530 | 0.016505 | −0.008393 [−0.009160,−0.007617] |

空间exact的coordinate/distance ES相对persistence在四个horizon均改善；Brier只在1 ns有不跨0的改善区间。相对空间prior，**七项指标和三项endpoint评分的总体CI全部跨0**；coordinate ES差为−0.003804 [−0.007936,+0.000667]。局部仍有权衡：4/16 ns coordinate ES改善，64 ns Brier退步。两个空间臂同时含bath；它们比较的是从最早历史帧unit prior滤波得到的S0与直接放在anchor的unit prior，既含历史条件化也含有限步prior传播/协方差差异，不是H+S对无bath Markov的主要因果对照。不能据此把总体收益归因于历史记忆，也不能宣称没有记忆。[完整endpoint统计](results/hs_joint_20261101/spatial_probe_v1/endpoint_paired_summary.json)保留全部结果和有限12样本分数解释。

规范CA几何有明显代价：独立噪声→空间exact的总体2.5 Å碰撞比例从0.182384%降至0.062627%，但3.3–4.3 Å之外的键长比例从0.185638%升至0.527759%，增加0.342121个百分点 [0.161120,0.560545]；64 ns的键长越界从0.256588%升至1.167176%。真实endpoint的总体键长越界为0.147368%、碰撞为0。空间exact的总体碰撞也略高于空间prior，增量0.001677个百分点 [0.000384,0.003507]。

相对保存的BioEmu2，空间exact只有MAE、Root W2、Variance W2三项总体改善区间不跨0；RMSF correlation低0.076945 [−0.129039,−0.025477]、weak J低0.024623 [−0.036891,−0.012652]，明确落后。MD-PCA和transient J均值仍差且区间跨0；64 ns没有任何七指标显示可靠领先，correlation与两项接触均落后。所有权重固定，因此本诊断也没有修复既有长时标H读出缺口。**本轮完成了匹配空间后验和两个可复查诊断，没有达到全面胜过BioEmu2，也不构成扩大失败joint配方的依据。**

## 后续：固定 step500 的四臂读出保持消融（retention_v1）

**四臂均已完成，当前 recipe 不推广。** Observed readout supervision 相对普通 teacher 减轻了 H 读出退化，并改善部分历史系综指标；但它的读出仍差于 frozen，coordinate/distance endpoint ES 也更差，16/64 ns 的读出预算仍失败。强 teacher 的总体读出误差又略低于 observed 臂，因此不能把已有改善归为结构保持项独有的作用，更不能据此宣称 S 记忆增益。

[执行前协议](results/hs_joint_20261101/retention_v1/protocol.json)和[runner](run_scripts/run_hs_readout_pilot.py)固定四臂共同从 `spatial_probe_v1/spatial_exact.pt` 出发，使用 `exact_spatial`、Gaussian spatial noise、noise_mix=.8、collective mobility，训练 seed42、相同 stateless 窗口流、各500个优化步。它与最初 `pilot_v1` 的独立噪声配方不同，不能直接将跨实验数值变化归因于 retention。四臂均评估 **step500 的 latest**，不按各自含不同正则项的 validation total 选择 best。

| Arm | Encoder | Teacher absolute / delta 权重 | Observed readout 权重 |
|---|---|---:|---:|
| frozen（F） | 固定 | 未启用 | 0 |
| joint_teacher（T） | 更新 | 1 / 0.1 | 0 |
| joint_observed（O） | 更新 | 1 / 0.1 | 0.1 |
| joint_strong_teacher（S） | 更新 | 10 / 1 | 0 |

Observed 项只用前四个观测帧：前三帧 H 相对第四帧 anchor H，经固定 CA decoder 及现有 projection 解码，与对齐到该 anchor 的已观测 CA 做 masked 向量 MSE（Å²）。它不读取 future coords/mask；teacher 保留。权重在执行前固定，observed=.1 依据既有 train-only 梯度尺度诊断，未在本轮验证结果上扫权重；强 teacher 是另一种保持对照，未做梯度范数匹配。后者初始 teacher 梯度为零，不能把10倍权重解释成与 observed 等强度。

标准面板仍为200窗、50蛋白、每窗12样本、评估 seed0，horizons 为1/4/16/64 ns；每温度40窗、10个不同蛋白。[正式审计汇总](results/hs_joint_20261101/retention_v1/summary.json)确认四臂16个 jobs 全部 exit0、四个 checkpoint 均500步、12份评分文件与保存的 BioEmu2 完整配对、冻结 decoder/teacher 及 frozen encoder 张量未变，并检查注册输入/源码快照/结果哈希。原 gate 的全部条件与阈值重算一致。BioEmu2 仅复用原 JSON，核对 metric version、12样本预算、200 keys 及 persistence reference 适配；未重新运行、改写或恢复它的模型。

### 七项历史系综指标

下面为等蛋白总体均值，仍按 endpoint draws 对 future-path mixture（含 anchor）评分。BioEmu2 列保留原输入适配和历史评分语义，不能移作 matched-endpoint 比较。

| 指标 | F | T | O | S | 保存的 BioEmu2 |
|---|---:|---:|---:|---:|---:|
| RMSF MAE ↓ | 1.17144 | 1.13057 | 1.08894 | 1.12020 | 1.99167 |
| RMSF correlation ↑ | .52633 | .52904 | .53291 | .53127 | .57974 |
| Root-Mean W2 ↓ | 1.58198 | 1.57418 | 1.55050 | 1.56395 | 2.29036 |
| Variance W2 ↓ | 8.69216 | 8.34352 | 8.21842 | 8.37340 | 18.57541 |
| MD-PCA W2 ↓ | 6.71284 | 6.74476 | 6.75955 | 6.73739 | 6.11426 |
| Weak-contact J ↑ | .10965 | .11021 | .11493 | .11216 | .13161 |
| Transient-contact J ↑ | .25360 | .25650 | .26503 | .25908 | .26526 |

O−F 的六项总体指标（除 MD-PCA W2）均有不跨0的改善区间；MD-PCA W2 增加 .046717 [−.022695,+.099067]，方向不确定。O 相对 BioEmu2 的 RMSF MAE、Root W2、Variance W2 改善区间不跨0，但 weak J 低 .016681 [−.028100,−.005084]；RMSF correlation、MD-PCA W2、transient J 的均值方向仍差且区间跨0。不能把总体部分优势表述为全面胜出，或把跨0区间表述为等效。

### Matched endpoint 与读出配对结果

真实未来 endpoint 只用于评分，未输入 sampler；训练仍使用 `path_mixture` 目标。下表全部越低越好。Coordinate ES 为除以√(3N)的 U-statistic，Distance ES 为含对角的 V-statistic，Brier 使用经验硬接触概率；共享12样本预算不消除模型相关的有限样本效应。单窗只有一个真实 endpoint，proper-score 解释针对跨窗口期望。

| 总体 endpoint score | F | T | O | S | Persistence |
|---|---:|---:|---:|---:|---:|
| Coordinate ES (U) | 1.532012 | 1.553330 | 1.560462 | 1.556752 | 2.033240 |
| Distance ES (V) | 2.097754 | 2.125062 | 2.134247 | 2.120513 | 2.690191 |
| Contact Brier | .0161541 | .0166531 | .0164273 | .0162636 | .0173794 |

全部五个注册 contrasts 的总体差值如下。**Δ=前者−后者，负值更低**；H readout 是真实未来 H 的 oracle，不能称为 forecast。CI 为10,000次等蛋白 paired bootstrap、seed20261101 的边际95%区间。

| Contrast | True-H endpoint RMSD Δ（Å）[CI95] | Coordinate ES Δ [CI95] |
|---|---|---|
| O−T | −.014334 [−.018991,−.010330] | +.007131 [−.004756,+.020747] |
| O−S | +.001648 [+.000323,+.002967] | +.003710 [−.002530,+.010221] |
| T−F | +.020223 [+.014169,+.027317] | +.021318 [+.000343,+.051182] |
| O−F | +.005890 [+.003185,+.008951] | +.028449 [+.008395,+.056403] |
| S−F | +.004242 [+.002286,+.006559] | +.024740 [+.006696,+.049699] |

O−T 的四个 horizon 读出 RMSD 差区间均小于0，但 O−F 的四个区间均大于0：保持项减少了退化，没有改善到原冻结组件之上。O−S 的总体读出差区间为正，但分别看四个 horizon，区间均跨0，不能声称逐时标都落后。O−F 的 Distance ES 也增加 .036493 [.004018,.078112]；Contact Brier 总体差为 +.000273 [−.000296,+.001329]，方向不确定，其中1/4 ns分别改善 −.000170 [−.000239,−.000098]、−.000317 [−.000519,−.000123]，16/64 ns区间跨0。O−T 的总体 Contact Brier 改善 −.000226 [−.000415,−.000070]，与其 coordinate/distance ES 的不确定变化并存；三项 endpoint score 不构成一致胜出。

原读出 gate 的 oracle/persistence endpoint RMSD 均值比为：

| Arm | 1 ns | 4 ns | 16 ns | 64 ns |
|---|---:|---:|---:|---:|
| F | .280025 | .293158 | .356958 | .426635 |
| T | .289475 | .299078 | .363030 | .430453 |
| O | .282044 | .295016 | .359153 | .427661 |
| S | .281952 | .294572 | .358417 | .427226 |

四臂均在1/4 ns通过、16/64 ns失败，失败项均为原有 **oracle ratio≤1/3** 预算，其余原 gate 条件通过。ratio 是等蛋白均值之比，不是逐窗比值均值；此门槛只按 horizon 使用。总体和温度 ratio 只作描述，没有新增 pass 或 noninferiority 标准。

### 几何、接触计数与可比 validation loss

规范 CA 几何与 legacy anchor 诊断分别保存，未混用阈值。F/T/O/S 的总体键长越界比例（3.3–4.3 Å之外）为 .36673%/.33276%/.34160%/.37681%，非局部2.5 Å碰撞比例为 .05051%/.05535%/.05089%/.04799%；真实 endpoint 分别为 .14737% 和0。O−T 的 legacy angle cosine MAE 为 −.010655 [−.013542,−.007529]，仅表示相对 observed anchor 的变化更小，不能据此称未来角度更准确。Legacy clash 使用3 Å cutoff，单独列在 `legacy_geometry_vs_anchor`。

`ensemble_contact_counts` 另报 weak/transient 的 n_true、n_pred，完整覆盖五 contrasts × 十分组。O 的预测集合总体均值为279.875/384.070，真值为93.215/191.225；这是每窗集合大小的等蛋白均值，不是 pooled counts，不将大小方向直接定义为好坏。

跨臂 validation 不能直接比较含不同正则的 raw total。共同 forecast loss 按实际权重计算：`raw total − teacher_weight×teacher − teacher_delta_weight×teacher_delta − observed_readout_weight×observed_readout_mse`，冻结臂无这些保持项。

| Arm | Raw validation total | Common forecast validation loss |
|---|---:|---:|
| F | 3.535442 | 3.535442 |
| T | 3.514777 | 3.514118 |
| O | 3.570385 | 3.475634 |
| S | 3.528438 | 3.527097 |

这是训练期间原16个 validation clips、每个 horizon 分别评估后的均值，**不是 standard200 面板**。日志没有该面板的逐蛋白簇身份，因此未给它附 bootstrap CI，也没有用它重新选 checkpoint。Common loss 改善不能替代 matched-endpoint 和读出结果。

### 解释限制与产物

完整汇总保留 overall、四个 horizons 和五个温度的全部注册 contrasts。先对每蛋白的窗口取均值，再对蛋白等权；温度组各10个不同蛋白，不能解释为同蛋白配对温度效应。这仍是 **单训练 seed、使用已多次自适应开发的 validation 集** 的小实验。边际 CI 不含多重比较、训练 seed 方差或开发选择不确定度；CI跨0不是等效证据。Test 保持封存，冻结基线和旧报告数值未改。

本轮只支持“observed supervision 与更强 teacher 均可减轻部分读出退化”，不支持 H 信息充分、S 记忆增益、完整条件动力学正确或全面超过 BioEmu2。长时标 gate 失败和 frozen 对照下 endpoint 退步仍在，因此**不推广这套 recipe，也不因历史七指标的部分改善直接扩大训练**。

独立的 T1 主链与有序轨迹诊断见 [Backbone and paths 报告](RESULTS_BACKBONE_AND_PATHS_20261101.md)，其任务定义和证据边界单独记录，不与这里的历史系综分数合并。

[汇总脚本](scripts/summarize_hs_readout_pilot.py)的10项 focused CPU tests、语法检查及拒绝覆盖验证通过；正式输出 SHA-256 为 `684d2592b1f100dc7d9eb682edc5ecc61647badcb661da088fbb745dca16d94a`。执行命令如下；正式文件已存在，再次执行会明确拒绝覆盖，不会重跑训练或 GPU 评估。

```bash
PYTHONPATH=/tmp/lasp_stability_deps:. CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
python -m scripts.summarize_hs_readout_pilot \
  --out results/hs_joint_20261101/retention_v1/summary.json
```

## 后续：raw H 数值范围与读出投影诊断

**生成 H 偏离 encoder 的均值/方差约束，但简单读出投影损害分布评分，暂不采用。** 该 CA 路径直接传递 encoder 末层非仿射 LayerNorm 的 raw H，没有另一套 H/Z 标准化或逆变换。[已有200窗缓存的事后分解](results/hs_joint_20261101/latent_manifold_v1/cached200_decomposition.json)发现，64 ns 的 forecast centroid 对真实 H 的逐维 MSE 均值33.7033，被一个 `1mmuA00` 窗口的1483.422显著影响；同组中位数4.330、p90为5.397。该结果是事后诊断，不是预注册验证。

随后固定训练集 stateless seed42 的前32步窗口、每窗12个成员，仅在既有生成 H 送入冻结 CA decoder 前执行同 eps=1e-5 的 LayerNorm；不反馈到 H/S 轨迹，不改变 anchor H，不从标签拟合投影。32个蛋白互不相同，每个 horizon 对应不同的8个蛋白，因此下表不能解释为同蛋白时间曲线。真实 H 的通道方差在四组均约为0.999997。

| Horizon | 原生成 H 通道方差 | 原 coordinate U-ES ↓ (Å) | 投影后 U-ES ↓ (Å) | Persistence U-ES (Å) |
|---|---:|---:|---:|---:|
| 1 ns | 1.17235 | 1.79605 | 1.78730 | 2.24871 |
| 4 ns | 1.80123 | 1.96351 | 2.20042 | 2.56240 |
| 16 ns | 4.08545 | 3.18791 | 3.69085 | 4.21883 |
| 64 ns | 9.42989 | 1.86553 | 2.58823 | 2.44791 |
| Overall | 4.12223 | 2.20325 | 2.56670 | 2.86947 |

64 ns 的原 H 逐维 MSE 为9.03847，分解为均值偏移0.07159、径向4.51391、角向4.45297。投影后降至1.46032，但中心化方向 cosine 仍为0.26984，剩余误差几乎全为角向。总体 CA 成员 RMSD 从5.74942降至5.61029 Å，同时 centroid RMSD 从5.00680升至5.34012 Å、U-ES变差。键长越界比例从0.51966%降至0.24501%，非局部碰撞比例从0.03629%升至0.06056%。单样本误差、系综评分和几何改善不能互相替代。

干预与固定选择在首轮前登记；首轮第30窗因附带距离/接触评分器不支持该残基 gap 而失败。**v2是已见前29窗结果后的评分器修复，不是新的未见数据预注册**：只保留约定的 coordinate U-ES 和 CA 几何，全部32窗保留，已有前29窗的 H MSE/CA RMSD 逐位复现。[成功结果](results/hs_joint_20261101/latent_manifold_v1/train32_manifold_v2.json)、[真实 exit0](results/hs_joint_20261101/latent_manifold_v1/train32_manifold_v2.execution.json)、[独立 NumPy 复核](results/hs_joint_20261101/latent_manifold_v1/train32_manifold_v2.independent_verification.json)与[登记说明](results/hs_joint_20261101/latent_manifold_v1/registration_and_interpretation_note.json)均已存档。没有修改模型、开启 test 或产生新的验证集候选；本诊断也不证明 S 记忆收益或稳定的 Langevin/GLE 动力学。

约束解释还需区分结构与数值误差：理想算术下，非仿射 LayerNorm 的通道方差为 `v/(v+eps)<1`，不是精确单位球。当前 collective 更新的条件协方差是 `2δM`，不会保持均值零超平面或有限半径；但条件范数增量 `2δ HᵀM(f−CS)+δ²‖M(f−CS)‖²+2δ tr(M)` 同时包含漂移和噪声，不能只看正噪声项就把净膨胀归因于 S 或粗 Euler。[旧微步诊断](scripts/diagnose_lasp_microsteps.py)直接使用 unit-prior S0，不能原样替当前 exact-spatial 历史后验定因。下面补充同 Brownian 路径的细化诊断；归一化后的转移是否满足原后验/FDT结构仍未验证。

### 同初态、同 Brownian 路径的微步长诊断

**步长细化明显缩小相邻离散路径的差异，但这32个训练窗的长时H方差偏移仍在，coordinate U-ES未改善，因此没有更换推理步长。** [协议](docs/HS_MICROSTEP_DIAGNOSTIC_V1_20261101.md)在本次采样前登记，固定原 `spatial_exact.pt`、同32训练窗和每窗12成员，不训练或选择新checkpoint。四帧真实观察历史只计算一次完整 exact-spatial 后验；三臂共用同一个S0抽样、H0、温度、空间噪声与D/C/Gamma。先抽8份细步Brownian增量，再按sum/√group组合成每1 ns的m=1/4/8步，均推进相同1/4/16/64 ns。原1000 ps历史后验不声称对细化转移仍精确；m8也不是解析真值。

[正式执行](results/hs_joint_20261101/microstep_train32_v1/execution.json)真实exit0、耗时624.06秒，32×3分支全部完成、数值失败0；13项focused tests通过。各horizon是不同8个蛋白，表中为等蛋白描述均值，不能作同蛋白时间效应或未见数据确认。

| Horizon | H通道方差 m1 / m4 / m8 | Coordinate U-ES ↓ m1 / m4 / m8 (Å) | Persistence (Å) |
|---|---|---|---:|
| 1 ns | 1.174327 / 1.170923 / 1.170515 | 1.803507 / 1.810699 / 1.811763 | 2.248714 |
| 4 ns | 1.794724 / 1.759872 / 1.757871 | 1.930738 / 1.937847 / 1.939428 | 2.562401 |
| 16 ns | 4.101615 / 3.991735 / 3.985046 | 3.127779 / 3.125953 / 3.128109 | 4.218834 |
| 64 ns | 9.373221 / 9.116761 / 9.092159 | 1.800049 / 1.801201 / 1.801518 | 2.447912 |
| Overall | 4.110972 / 4.009823 / 4.001398 | 2.165518 / 2.168925 / 2.170205 | 2.869465 |

同噪声的m1→m4、m4→m8端点差异，overall H逐维RMS分别为.130368/.025285，CA RMSD为.355561/.070020 Å；64 ns分别为H .240255/.047322、CA .477989/.095608 Å。当前噪声数组的抽取顺序与旧投影实验不同，即使整数seed相同也不是旧实验的逐位路径重放，不能直接将两表m1值相减解释为干预收益。

逐微步条件范数预算保留raw及逐残基通道中心化两个版本。以64 ns、除以有效残基数×128的raw累计预算为例：

| 每1 ns微步数 | force交叉项 | bath交叉项 | 漂移平方项 | 噪声trace项 | 实际范数平方增量 |
|---|---:|---:|---:|---:|---:|
| 1 | −.382486 | −2.401761 | 1.409891 | 9.824377 | 8.440503 |
| 4 | .283522 | −2.111881 | .188407 | 9.824377 | 8.180054 |
| 8 | .334720 | −2.092132 | .092121 | 9.824377 | 8.155202 |

实际量减四项条件期望的残差分别为−.009518/−.004372/−.003884；raw范数增量不等于中心化通道方差增量。后验初态不确定性项另列，未重复加入已条件于当前H/S的预算。当前collective分支并未执行配置中的`max_step_norm=10`，本诊断没有暗中加裁剪；原始更新与force均保持冻结。bath项在这些路径上为负，**不等于与无S模型相比的因果记忆收益**。

[完整结果](results/hs_joint_20261101/microstep_train32_v1/diagnostic.json)及每窗原始H/S/CA、S0和噪声张量均保留，结果SHA-256为 `0bce069c70ac415b88f261e7710e4d78ad26fd1f3bc3855969c6c2e7ad11eee7`。[独立复核](results/hs_joint_20261101/microstep_train32_v1/independent_verification.json)重算全部macro统计、耦合路径差异、endpoint/几何与聚合；宏观统计最大差1.78e−14，预算聚合最大差1.56e−13。固定首个64 ns窗口，另用实际冻结force重放三臂前两macro步共26微步，H/S宏观输出逐位一致，6240个预算值最大差5.53e−12；831,601,248个随机数组值逐位重现。[初态补充核验](results/hs_joint_20261101/microstep_train32_v1/initial_state_and_verifier_hash_check.json)确认96分支的共同H/S初态及验证器自身哈希未变。

这是已多次检查过的有限训练面板；不证明全局稳定或连续时间收敛，也不能排除前述验证集`1mmuA00`异常窗的数值问题。没有读取test、重跑BioEmu2或据此采用投影/细化推理，全面胜出目标仍未达成。

### 固定协方差，只改变S初始历史均值

在同32个训练窗上，[固定均值对照协议](docs/HS_HISTORY_MEAN_DIAGNOSTIC_V1_20261101.md)比较真实四帧历史、四次重复anchor H0，以及零均值三个初始化。三臂共享原真实历史后验的完整协方差、同一个Chol×epsilon、H0和全部Brownian H/S，均用原m1连续传播。重复anchor仍用实际模型推断均值，未假设它为零；零均值也保留原协方差，不能混同于旧单位协方差prior。

[正式执行](results/hs_joint_20261101/history_mean_train32_v1/execution.json)真实exit0、168.75秒；32×3分支全部完成，数值失败0。采样前全部32窗重复历史的Cholesky与原值逐位相同；真实历史臂的S0及完整H/S/CA逐位复现旧m1路径。13项新增合成测试与13项microstep回归通过，新增测试另经独立重跑。没有新读MD坐标、validation或test，也没有训练或修改权重。

| Horizon | 真实历史 Coordinate U-ES ↓ (Å) | 重复anchor历史 (Å) | 零均值 (Å) | 重复−真实 / 零−真实 (Å) |
|---|---:|---:|---:|---|
| 1 ns | 1.803507 | 1.807197 | 1.803664 | .003691 / .000158 |
| 4 ns | 1.930738 | 1.951655 | 1.948679 | .020918 / .017941 |
| 16 ns | 3.127779 | 3.201630 | 3.196719 | .073852 / .068940 |
| 64 ns | 1.800049 | 1.806877 | 1.805758 | .006828 / .005709 |
| Overall | 2.165518 | 2.191840 | 2.188705 | .026322 / .023187 |

当前面板中真实历史均值的各组U-ES均较低，但没有进行CI或显著性检验，也不是每个窗口或每项指标都改善。例如64 ns成员RMSD为真实5.311974、零均值5.306477 Å，方向与U-ES不同。总体成员RMSD依次为5.746723/5.774439/5.765048 Å，H通道方差依次为4.110972/4.118239/4.120888。重复历史和零均值相对真实均值的差，用原有效bath维度Cholesky白化后，每窗标准化RMS的等蛋白均值分别为.197296/.221108；这是初始化改变量，不是可识别记忆量。

[完整结果](results/hs_joint_20261101/history_mean_train32_v1/diagnostic.json)SHA-256为 `3e6070346ce593b1b55b4e39586d8821b835a54f95681edceb08658f3624dd71`，保留所有初态、宏观H/S/CA、实际Brownian、逐成员评分/几何、逐窗配对差与全horizon汇总。[独立NumPy复核](results/hs_joint_20261101/history_mean_train32_v1/independent_verification.json)也已exit0，15项核验器测试通过；110,960,364个真实臂H/S/CA元素逐位重现旧路径，全部白化均值差、179,520个macro统计值、endpoint/几何和聚合重算通过，除float32协方差抽样舍入外最大绝对差2.84e−14。按旧协议固定的step4另重推重复历史后验、重放三臂前两macro步，后验均值/Cholesky与H/S均逐位一致。核验器登记发生于生产结果已存在之后、独立核验读取结果之前，未据新分数选窗；结果SHA-256为 `35942dea1704861c151845a816004a13dae949c82400ff1992e01c9ba0b9cd65`。

该诊断只隔离**当前已训练模型的S初始化均值及观察历史作用**。重复anchor是人工历史，可能离开训练分布；窗口已被反复检查，各horizon仍对应不同8个蛋白。它没有将同预算独立训练的H+S与H-Markov模型比较，因此既不能宣布S整体增益，也不能以小差异宣布记忆无效。随后完成的独立训练对照如下，不能用本节推理干预代替。

## 独立训练H-Markov对照完成：主指标差异仍不确定

[训练对照的决策声明](results/hs_joint_20261101/markov_v1/decision.json)已于2026-11-01 23:54:40 UTC独占保存，早于本轮读取target目标臂分数。固定复用其endpoint500作为S训练臂，只新训同起点、同500步数据流配置的C=0控制，再做standard200与endpoint200；主比较为16/64 ns等蛋白coordinate U-ES，全部时标/温度及七指标同时报告。[完整运行协议](results/hs_joint_20261101/markov_v1/protocol.json)随后登记，SHA-256为 `5eca1e002b696e127cbb7c4954b82f39227363f41f3c0955c053a382a221bb1a`；2026-11-02 00:10:31 UTC在GPU1启动，00:26:50 UTC整个runner实际exit0。训练、standard200和endpoint200分别797.76、102.39、76.11秒，三个子任务均exit0，见[实际进程记录](results/hs_joint_20261101/markov_v1/execution.json)。统计实现的[后续登记](results/hs_joint_20261101/markov_v1/summary.protocol.json)发生于训练启动之后、作者读取S/C0分数之前；早期决策已固定复用和主指标，不能把后续实现登记倒写为训练前登记。

控制仍保留bath维度、历史推断和随机数组请求形状，以保持H噪声配对；C=0使有限状态下H更新对S0、S噪声及额外历史无函数依赖，称为“给定固定anchor条件的H-Markov”控制。它仍计算不用的bath路径，不是等有效参数量或推理效率比较；共同起点本身也来自先前H+S训练。17项启动契约测试已通过，覆盖实际rollout的S0/S噪声干预、H噪声及最终RNG、历史独立性和三个加载器对零耦合的恢复；不能只因CLI写了`--bath-dim 0`就认定已禁用bath。旧S训练未保存逐行训练坐标内容哈希，因此相同数据流配置、当前文件身份与新训练登记不能追溯证明旧500步坐标内容逐位相同。

新checkpoint SHA-256为 `27df16b76693e5fd5d0cddb5940459a3318cfec7114108abd2125784d356520c`。初始化全部权重、最终完整H/decoder组件、优化器状态支持及500步数核对通过；最终采样RNG与Python/NumPy/Torch/CUDA全局RNG与复用S臂一致，最终权重与优化器动量允许不同。实际运行参数仅coupling_scale与输出路径不同；cfg/runtime的C=0单独核验，不能仅靠tensor哈希判断耦合配置。

[正式汇总](results/hs_joint_20261101/markov_v1/summary.json)实际exit0，15项汇总测试通过，SHA-256为 `6f6b266e041355b36b0d08beb423a2fcceb0db4c7dfca1e671a2cd90e1b69d5d`。[独立NumPy验证](results/hs_joint_20261101/markov_v1/independent_verification_v1.json)也已exit0：341个配对区间、670个描述均值、3,495次数值比较最大差0，419个源/输入/快照核对通过；13项核验器测试通过。独立核验重算统计并检查checkpoint字节哈希及配置/执行证据，没有第二次重载tensor；tensor/RNG逐位审计由正式汇总负责。核验SHA-256为 `0dc92b0c672406e380a4cdc1145bb1dc5f01dcc92c16820b357af1d5cb059b5b`。

| Coordinate U-ES ↓ (Å) | H-Markov C=0 | H+S | C0−S差的95% CI |
|---|---:|---:|---|
| **主指标：16/64 ns平均** | **1.992596** | **2.002682** | **[−.050978,+.020547]** |
| Overall | 1.520780 | 1.523357 | [−.022356,+.012430] |
| 1 ns | .947409 | .948872 | [−.007425,+.004137] |
| 4 ns | 1.150519 | 1.139194 | [+.001455,+.019975] |
| 16 ns | 1.724230 | 1.722816 | [−.026434,+.020999] |
| 64 ns | 2.260962 | 2.282548 | [−.099020,+.031881] |

主指标差为−.010086 Å，区间跨0，**这轮预算没有建立H+S的主指标优势，也不能认定两者等效或S无用**。总体distance V-ES为2.111200/2.111006，Brier为.016517/.016435，差的区间均跨0。4 ns三项endpoint评分都支持S臂较好；distance与Brier的C0−S差分别+.034259 [+.013081,+.053743]、+.000437 [+.000267,+.000603]。所有温度分层仍完整保留，不以不同蛋白的温度组冒充配对温度响应。

| 历史七指标，overall | H-Markov C=0 | H+S | C0−S差的95% CI |
|---|---:|---:|---|
| RMSF MAE ↓ | 1.340498 | 1.297004 | [−.015300,+.083897] |
| RMSF correlation ↑ | .508165 | .510739 | [−.009540,+.004520] |
| Root mean W2 ↓ | 1.650417 | 1.634393 | [−.028789,+.043237] |
| Variance W2 ↓ | 9.103686 | 9.099548 | [−1.085870,+.672644] |
| MD PCA W2 ↓ | 6.490708 | 6.571190 | [−.196265,+.001279] |
| Weak-contact Jaccard ↑ | .097068 | .099754 | [−.004395,−.000939] |
| Transient-contact Jaccard ↑ | .229949 | .234930 | [−.007228,−.002548] |

H+S的两项接触指标较好，但其余五项总体差均跨0，PCA的区间上界也仍大于0。相对保存的BioEmu2，C0与S同样是MAE/Root/Variance较好、correlation/weak/transient较差、PCA不确定。CA无效键比例C0/S为.003771/.003780、clash为.000591/.000583，总体配对区间均跨0；全部规范几何及分层结果均见汇总。

这是从已训练H+S起点作500步续训的单seed、重复开发集比较，区间未覆盖训练seed、多重比较和开发选择不确定性；差异也不能唯一识别物理记忆或latent S语义。本轮未给C0新增fullsuite/matched8，standard200的温度分层不能替代T2A/T3。冻结H的16/64 ns gate和完整T1/T2B等缺口保持不变，不作总体方案升级或全指标胜出声明。

## 目标监督对照完成：主指标未显示可靠改善（target_v1）

已按 [固定协议](docs/HS_TARGET_PILOT_V1_20261101.md) 与 [统计登记](results/hs_joint_20261101/target_v1/summary.protocol.json) 完成两臂各500步训练，真实 exit0，耗时分别698.84／703.07秒。两臂从同一 `spatial_exact.pt` 出发，冻结 H/CA decoder，仅将监督目标切换为 endpoint 或 future-path mixture；seed42、12成员、相同 stateless 数据流及优化器，RMSF/slow-mode等涨落损失共同关闭，避免单端点无法估计涨落造成额外差异。两臂实际第一步 h1 的全部训练损失项和梯度范数逐位相同，step500 的 encoder/decoder/投影配置均保持不变。历史上已有 endpoint 实验，这次检验的是当前 exact_spatial 设置下的匹配对照。

每臂依次运行训练、标准200窗七指标、独立200窗 matched-endpoint评分、完整7200窗 correct/wrong/none 温度与T2A评测、800窗 matched8。主比较是各蛋白16/64 ns coordinate U-ES的均值差（endpoint减path_mixture），使用固定10,000次配对蛋白 bootstrap；其他时标、温度、全部七指标和规范CA几何完整报告。H/decoder冻结，原16/64 ns oracle gate失败保持不变。单训练seed和反复使用的开发验证集限制仍适用。

原runner的两臂standard200均已 exit0（115.39／111.98秒），随后endpoint调用因传入不支持的 `--max-residues 96` 而在argparse阶段exit2；未进行该阶段采样、未产生endpoint结果，原runner最终exit1。该工具内部已固定max_residues=96，错误来自启动器接口，不能将成功训练或前两项结果记为整个评测成功。原始源码、协议、失败日志及execution均保留；修复仅删除无效CLI参数并在新执行记录下继续剩余三项评测，不重训或重跑已成功的standard200。

[CLI修订协议](docs/HS_TARGET_PILOT_CLI_AMENDMENT_V2_20261101.md)及[独立续跑器](run_scripts/continue_hs_target_pilot_v2.py)完成剩余任务，21项契约测试通过，其中包含真实parser正反例。旧输入、失败日志、成功结果及检查点继续由哈希保护。[endpoint续跑记录](results/hs_joint_20261101/target_v1/continuation_v2/endpoint.execution.json)和[path-mixture续跑记录](results/hs_joint_20261101/target_v1/continuation_v2/path_mixture.execution.json)均为completed/exit0；旧execution继续如实标为failed，不能覆盖这两次历史CLI错误。

续跑的两臂200窗matched-endpoint评分均已exit0（81.20／78.63秒），结果位于[endpoint目标臂](results/hs_joint_20261101/target_v1/endpoint.endpoint.json)和[path-mixture目标臂](results/hs_joint_20261101/target_v1/path_mixture.endpoint.json)。完整7200窗suite分别耗时4046.14／4011.37秒，matched8的800窗分别438.58／452.39秒，均实际exit0。2026-11-02 00:09:23 UTC全部10个逻辑任务完成，共12次执行尝试、2次保留的失败；每臂full/matched8重叠的全部800行原始结果核对一致。

[v2汇总入口](scripts/summarize_hs_target_pilot_v2.py)仅适配执行记录，复用冻结v1的统计定义和数值函数；已实际exit0生成[完整汇总](results/hs_joint_20261101/target_v1/summary_v2.json)，SHA-256为 `1828cbd81bc2c43854db3f1aab459f251036106cdff96e9e46885b358aab7bc0`。v2的22项测试加v1的11项测试共33 passed，独立审查通过；曾在suite仍运行时实际确认其拒绝提前生成汇总。复现入口为：

```bash
PYTHONPATH=/tmp/lasp_stability_deps:. CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
python -m scripts.summarize_hs_target_pilot_v2
```

[独立NumPy统计核验](results/hs_joint_20261101/target_v1/independent_verification_v1.json)也已实际exit0：从原始窗口重算341个配对区间、670个standard描述性均值和1700个suite描述性均值；12,070次数值比较最大绝对差7.11e−15，并核对319个源/输入/快照文件及800行重叠。11类核验器自测通过；核验结果SHA-256为 `be61856f0a74fc6a1891861e7b38a358189e8dce20df6a4affa0831f79c077be`。旧suite的4000次seed17区间保持原值，仅独立重算其均值/计数；旧ensemble组件sidecar没有历史执行摘要，当前哈希不能充当历史来源证明。

预定主指标为16/64 ns coordinate U-ES：endpoint **2.002682 Å**、path-mixture **2.027654 Å**；差为 **−0.024972 Å，95% CI [−0.102371,+0.022296]**（50蛋白、100窗）。各单独horizon的coordinate差也全部跨0。该结果不支持升级endpoint方案，也不能解释为两方案等效。

| 指标 | Endpoint | Path-mixture | Endpoint−path-mixture的95% CI |
|---|---:|---:|---|
| Coordinate U-ES，overall ↓ (Å) | 1.523357 | 1.534888 | [−.050549,+.012826] |
| Coordinate U-ES，1 ns ↓ (Å) | .948872 | .949522 | [−.002037,+.000781] |
| Coordinate U-ES，4 ns ↓ (Å) | 1.139194 | 1.134722 | [−.000192,+.009299] |
| Coordinate U-ES，16 ns ↓ (Å) | 1.722816 | 1.723073 | [−.022391,+.018233] |
| Coordinate U-ES，64 ns ↓ (Å) | 2.282548 | 2.332234 | [−.182580,+.028755] |
| Distance V-ES，overall ↓（distance/1 Å，无量纲） | 2.111006 | 2.109962 | [−.035004,+.032230] |
| Hard-contact Brier，overall ↓ | .016435 | .016777 | [−.001304,+.000173] |

局部也有代价：endpoint在4 ns distance ES和4/16 ns Brier更差，差的区间分别为[+.004474,+.021876]、[+.000046,+.000195]和[+.000111,+.000648]。规范CA无效键比例为.003780对.003335，差[+.000145,+.000875]；非局部clash比例为.000583对.000592，总体差跨0，4/16 ns则增加。键长均值3.840333/3.839498 Å和Rg 14.968915/14.948908 Å只描述，不按单调方向判优。

| 历史七指标，overall | Endpoint | Path-mixture | 保存的BioEmu2 |
|---|---:|---:|---:|
| RMSF MAE ↓ | 1.297004 | 1.257186 | 1.991674 |
| RMSF correlation ↑ | .510739 | .516639 | .579744 |
| Root mean W2 ↓ | 1.634393 | 1.615263 | 2.290364 |
| Variance W2 ↓ | 9.099548 | 8.928817 | 18.575412 |
| MD PCA W2 ↓ | 6.571190 | 6.649675 | 6.114255 |
| Weak-contact Jaccard ↑ | .099754 | .101836 | .131606 |
| Transient-contact Jaccard ↑ | .234930 | .238880 | .265260 |

相对BioEmu2，两臂的MAE/Root/Variance改善区间不跨0，correlation/weak/transient退步区间也不跨0；PCA均值更差但区间跨0。Endpoint对path-mixture仅PCA有不跨0的改善，同时MAE、correlation、Root、weak和transient退步，Variance差跨0。**全面超越BioEmu2的目标仍未达到。**完整overall、四horizon、五温度、correct/wrong/none和T2A数据均保留于上述汇总，不以总体三项改善代替全分层胜出。

这些区间为10,000次、seed20261101、等蛋白配对bootstrap；单训练seed、重复开发验证集和未校正多重比较使其属于边际探索性证据。standard200各温度含不同蛋白，不可当作配对温度响应；五温度均曾用于训练，完整suite只检验输入温度机制，未完成未见温度T3。BioEmu2沿用保存的七指标结果；历史T2A缺mask/reference/状态中心指纹，不作严格配对胜负，不重跑或改写历史基线。本实验也没有修复冻结H的16/64 ns gate，或完成匹配时间的BioEmu2 endpoint/path、T1与T2B/MSM比较。

### 完整suite：位置误差、温度机制与T2A仍有明显缺口

每臂fullsuite共12蛋白、300轨迹、7200窗；每个温度模式2400窗，每温度每模式480窗。它包含历史七指标、common-anchor的逐成员平均CA RMSD、逐预测最佳拟合的aligned RMSD和persistence skill，不含matched-endpoint ES、联合路径/ACF/MSD或完整骨架/MSM评分。旧字段`kabsch_ca_rmsd`在此实际指common-anchor误差，不能误写为对每个预测独立拟合。

| 训练臂 / 输入温度 | RMSF MAE ↓ | RMSF corr ↑ | MD PCA W2 ↓ | 成员平均CA RMSD ↓ (Å) |
|---|---:|---:|---:|---:|
| Endpoint / correct | 1.423087 | .452132 | 9.825226 | 5.719902 |
| Endpoint / wrong | 1.573691 | .460866 | 10.337522 | 5.611665 |
| Endpoint / none | 1.392494 | .466175 | 10.199844 | 5.526744 |
| Path-mixture / correct | 1.390867 | .460173 | 9.875301 | 5.665476 |
| Path-mixture / wrong | 1.547177 | .467730 | 10.328169 | 5.574022 |
| Path-mixture / none | 1.393154 | .471719 | 10.215651 | 5.506645 |

Persistence的同口径RMSD为4.930655 Å，correct模式的position skill分别−.789248/−.734821 Å；两臂四个horizon的成员平均误差都高于persistence。这与standard200的coordinate/distance U/V-ES优于persistence并不矛盾：成员平均位置误差与含系综多样性项的分布评分衡量不同。不能只报后一组的正向结果。

correct温度的PCA均值较好，correlation、接触和成员RMSD却较差，不能声称正确温度在所有指标上有效。wrong在320 K输入450 K、其他温度输入320 K；none使用参考348 K，348 K的correct/none各480窗所有原始指标逐位相同，不能作为独立证据。这里只描述固定同窗口温度干预，不以单臂legacy CI推导新的配对显著性。

| 温度 (K) | Full12 coverage E / P ↑ | Matched8 coverage E / P / 历史BioEmu2 ↑ | Matched8 population TV E / P / 历史BioEmu2 ↓ |
|---|---|---|---|
| 320 | .493750 / .489583 | .406250 / .406250 / .368750 | .808296 / .804574 / .626386 |
| 348 | .402083 / .402083 | .271875 / .271875 / .246875 | .75749 / .75419 / .58899 |
| 379 | .227083 / .229167 | .209375 / .221875 / .178125 | .72934 / .73194 / .62206 |
| 413 | .110417 / .110417 | .093750 / .093750 / .081250 | .64043 / .64125 / .54468 |
| 450 | .016667 / .016667 | .015625 / .015625 / .015625 | .60427 / .60673 / .56891 |

E/P为endpoint/path-mixture。Full12每轨迹96成员，matched8为48成员、8蛋白/200轨迹；full12与matched8不是同一支持集。表中历史BioEmu2仅作描述性并列：800窗口键和200轨迹键相配，但缺mask/reference/中心指纹，严格配对差保持null。低温coverage均值略高不能掩盖population TV/JSD在五温度都高于历史BioEmu2均值；自由能误差也仅379 K的历史均值比较方向较好。k=8时population MAE=TV/4，二者不算独立证据。覆盖要求距状态中心小于2 Å且hit fraction≥.001；48/96成员下至少一次命中即可，不是简单最近状态标签覆盖。

补充的 [model-free T2A参考作业](results/hs_joint_20261101/target_anchors_v1/execution.json) 已完成两个阶段、均 exit0：[2400窗预检](results/hs_joint_20261101/target_anchors_v1/mask_reference_audit.json) 确认12蛋白、300轨迹的全部窗口 mask 与各自全参考 mask 逐位一致，并保存参考张量、窗口 mask/anchor 的内容哈希；然后用冻结脚本产生 [persistence和MD分半参考分数](results/hs_joint_20261101/target_anchors_v1/raw.json)。它没有加载模型或重跑BioEmu2，不修改主要比较。旧脚本的 `oracle` 实为单向随机 half-reference→complement 诊断，状态中心由半份参考拟合，不能作为严格 ceiling/lower bound，也不能假设其分数必然夹住模型；修正后的解释见 [独立说明](results/hs_joint_20261101/target_anchors_v1/interpretation.json)。外部HDF5用前后文件身份检查，未声称对完整HDF5作密码学哈希。

T2A的时间范围也需单独解释：[suite采样池](scripts/eval_lasp_ca_suite.py) 仅含 correct-T、固定早期anchor7/8和h1/4/16/64的endpoint混合（每轨迹96成员；matched8为anchor7的48成员），参考及状态中心却来自整条MD轨迹的经验样本。完美matched-time条件预测也不保证该混合等于全轨迹布居；T2A改善不能自动解释为动力学改善，全轨迹经验布居也不等于已验证的平衡分布。该限制不改变既定T2A评分或全指标目标；同窗口真实MD endpoint池对全参考的补充诊断已完成，结果如下。


### 同窗口真实MD端点池的T2A时间范围诊断

[固定协议](results/hs_joint_20261101/t2a_time_support_v1/protocol.json)在新坐标读取和评分前登记；沿用已经审过的开发验证集窗口，不是未见数据确认。CPU2线程作业真实exit0、耗时56.003秒，见[执行记录](results/hs_joint_20261101/t2a_time_support_v1/execution.json)。12项合成测试通过。主模型训练和完整suite未受干预；没有载入模型、读取新目标臂分数或重跑BioEmu2。

full12使用12蛋白、300条轨迹、2400窗，每轨迹8个真实endpoint，零起始frame IDs为8/9/11/12/23/24/71/72；matched8使用8蛋白、200条轨迹、800窗，每轨迹4个endpoint，frame IDs为8/11/23/71。Persistence保留相同窗口的anchor权重。每窗只有一次真实观测，不能把它们算作96/48个独立模型样本，也不能视为完美条件生成器。参考及k=8、seed0的状态中心与当前suite相同；coverage还要求与参考中心的内部距离特征RMS差小于2 Å，不能等同于最近状态标签命中率。

| 面板 | 样本来源 | Coverage ↑ | Free-energy MAE ↓ (kcal/mol) | Population MAE ↓ | Population TV ↓ | Population JSD ↓ |
|---|---|---:|---:|---:|---:|---:|
| full12 | 真实endpoint池 | 0.325833 | 0.852711 | 0.155012 | 0.620047 | 0.309358 |
| full12 | Persistence | 0.235833 | 0.960993 | 0.190709 | 0.762836 | 0.420389 |
| matched8 | 真实endpoint池 | 0.275625 | 0.903830 | 0.161244 | 0.644974 | 0.328264 |
| matched8 | Persistence | 0.190000 | 1.018498 | 0.197509 | 0.790037 | 0.444570 |

表中先对每蛋白的各温度/replica轨迹取均值，再对蛋白等权。全部五温度、五指标、逐轨迹分数及帧编号保留在[结果](results/hs_joint_20261101/t2a_time_support_v1/result.json)，SHA-256 `b0f58e3db1439d01a889c4d13c372d0c064cae4ea7076ec2e921fecc4a1305b7`。full12真实端点池在320/348/379/413/450 K的coverage依次为0.595833/0.489583/0.345833/0.172917/0.025000，TV依次为0.689494/0.665500/0.621667/0.575005/0.548567；这两种指标的温度方向并不相同。

全部旧参考坐标/mask及2400窗anchor指纹匹配；评分后重读参考与窗口、重算features/pairs/centroids/labels，内容哈希一致，full12 persistence的全部terms逐位复现旧结果。外部HDF5仍是文件身份加选中张量内容校验，不声称整文件密码学哈希。旧half-reference只复制保存的对应轨迹分数，继续单列其半参考聚类和不同预算。

[独立复核](results/hs_joint_20261101/t2a_time_support_v1/independent_verification.json)另行通过：1980个聚合数值最大绝对差4.44e−16，313项源码/输入/快照哈希及12个HDF5文件身份检查通过。固定抽查首条已运行的 `1a02F00_T320_R0`，直接重读其既有数据并重算full/matched的endpoint/persistence四组评分，全部字段与features/pairs/centroids/labels哈希完全一致。复核文件SHA-256为 `55d5cf44b509413c871c1812c8c79b5119cb5d77174a6b25794317b3d0709185`。

该差异同时涉及时间范围、有限样本和时间相关性，不能据此单独定量归因于时间范围，也不是T2A性能ceiling或放宽验收条件。全参考可能包含这些endpoint/anchor，整条MD经验分布也没有被证明达到平衡；这里不报显著性、CI或新通过门槛。主目标仍要求模型本身在既定全部指标上胜出。

## 连续路径联合监督对照完成（2026-11-02）

此前path-mixture把监督标签换成未来帧池，并未对同一成员路径的联合时间分布评分。新的[固定方案](docs/HS_ORDERED_PATH_PILOT_V1_20261102.md)比较两个新训练臂：`joint_path`使用.5 endpoint U-ES+.5整条未来路径U-ES，`marginal_path`使用.5 endpoint U-ES+.5逐时刻U-ES均值。两臂使用相同全部未来标签、连续采样程序和解码工作，只改变坐标损失中的联合/边际定义；旧path-mixture不能替代这个匹配控制。

两臂都从原spatial_exact起点训练500步，seed42、12成员、完整grad64、相同stateless窗口选择、lr1e−4和warmup50/cosine；H/CA decoder/32次投影冻结。latent .1、soft-contact Brier20、clash40保留终点定义，其余原零权重保持零。本轮不作训练期val、不写best、不选中间checkpoint，仅评latest500；因此旧endpoint训练只作历史背景，不声称与其全部历史RNG逐位相同。两臂已逐步保存实际训练张量哈希，并已核验两者完整500窗、初始化和最终采样/全局RNG相同。

新增[可微路径接口](protein_jepa_frame/training/ordered_path_objective_v1.py)通过只读代理复用冻结sampler：每窗一次exact_spatial后验、一次连续H/S rollout，先按原batch形状解码终点，再解码中间帧；不重置S、不重编码预测、不重配成员。中间确定性decoder使用non-reentrant checkpointing，随机动力学部分不改。真实时间戳和observed固定支持均显式核验；未来缺失该支持会失败，未来新增可见但原observed无效的位置被忽略。未来标签不进入采样器。

25项原语测试与20项训练/执行器测试共45 passed，另有16项统计测试通过并独立复跑；覆盖h1数值/梯度/RNG逐位等价、真实decoder投影的checkpoint梯度等价、一次连续采样、未来隔离、padding、实际CLI和失败留存。[统计协议](results/hs_joint_20261101/ordered_path_v1/summary.protocol.json)在训练前登记，SHA-256 `4facecde654e82b780a587bdb0a81307da6902c43f6ec50fc305d92e74f78ff4`。主指标固定为16/64 ns等蛋白联合路径coordinate U-ES的joint−marginal差；共402个边际探索性配对区间，其中旧long-endpoint指标明确降为secondary。ACF/MSD/pathlength/reverse/shuffle完整描述并报告缺失支持，不为所有诊断指定单调质量方向。

[固定h64可行性检查](results/hs_joint_20261101/ordered_path_v1/smoke_v1/job/smoke.json)于00:36:49 UTC实际exit0：训练流step4的`3aq5A00_T320_R4`、96残基、12成员，各臂一次完整前后向、零optimizer更新。Joint/marginal分别6.676/5.150秒，峰值分配显存5,357,602,304/5,367,187,456字节（约4.990/4.999 GiB），预裁剪梯度范数5.408/5.343；64组kernel参数有梯度，H/decoder无梯度，模型前后哈希及两臂最终采样RNG一致。该检查没有AdamW状态，第二臂复用已预热allocator，不能把耗时/保留显存差解释为目标效率差或完整训练显存上界。结果SHA-256为 `47871551fe6c4dff3fd5080bfcd21e9f30b70f2b397728948b6eca02b0d35624`。

首次正式登记因root沿用`CUDA_VISIBLE_DEVICES=''`，在两GPU检查处exit1，未创建正式protocol或artifacts、未启动训练；[失败记录](results/hs_joint_20261101/ordered_path_register_v1.execution.json)和日志保留。恢复GPU可见性后，相同冻结源码的[第二次登记](results/hs_joint_20261101/ordered_path_register_v2.execution.json)实际exit0，不属于训练重启。[运行协议](results/hs_joint_20261101/ordered_path_v1/protocol.json)于00:38:13 UTC创建，SHA-256 `a31b87128702cd796b25fbbef60c3bd1245304cfd369c668970e54c72c7235cb`，固定101个源文件、13项输入和500个训练窗元数据。

GPU0/1两臂的train500、standard200、endpoint200、ordered200、fullsuite7200和matched8的800窗共12个逻辑任务全部实际exit0，主执行器于02:13:23 UTC完成。两份实际训练窗口日志各500行且完整文件逐字节相等，SHA-256均为`cbcad706d02645312d3a8209f0b18210a375998bb3b532a2f78788f24b8a71cb`；最终checkpoint/optimizer/RNG的完整审计通过。实际记录见[joint执行记录](results/hs_joint_20261101/ordered_path_v1/joint_path.execution.json)与[marginal执行记录](results/hs_joint_20261101/ordered_path_v1/marginal_path.execution.json)。本轮没有新增192 ns、MSM或test/BioEmu2重跑。

静态审计发现首版汇总wrapper把从JSON读取的`horizons`列表与Python原始tuple直接比较，会误拒绝有效ordered结果。此时尚未运行正式汇总，不能计作一次真实失败任务。冻结v1实现与注册保留，另建[v2](scripts/summarize_hs_ordered_pilot_v2.py)，仅对expected panel作JSON规范化并增加实际序列化往返回归；两组训练、评估源文件和训练前登记的402个统计区间定义均不因此改变。19项v2测试通过并独立复跑，[新实现协议](results/hs_joint_20261101/ordered_path_summary_v2/protocol.json)于00:55:58 UTC登记，SHA-256 `6804838c2c655869b43cc409aa89105e335db1b3d66d28927b668ecee47f5d35`。登记如实记录部分评估文件已存在，但未解析其性能数值。正式输出改为独立`summary_v2.json`，随后在12项任务全部完成后执行。

配套[独立NumPy核验器v1](results/hs_joint_20261101/ordered_path_verification_v1/independent_verify_v1.py)的29项合成测试通过，并于01:00:27 UTC完成仅元数据登记；[协议](results/hs_joint_20261101/ordered_path_verification_v1/protocol.json)SHA-256 `39c5adfb5945d6bfff895bde2143973f29f8b3af6924ef5f6526d560fe61650f`。其04:51:43 UTC实际执行在元数据检查处exit1：ensemble记录使用同一评分源文件的绝对路径，而protocol使用相对路径，源码SHA相同；失败发生于数值重建之前。[真实失败](results/hs_joint_20261101/ordered_path_independent_run_v1.execution.json)与全部v1文件保留。

新[核验器v2](results/hs_joint_20261101/ordered_path_verification_v2/independent_verify_v2.py)只规范化仓库内路径并继续要求文件SHA完全相同，另固定上述真实失败；31项测试通过后登记，协议SHA-256 `eda1e7367471a32505939c35aa97ff2e39fbe2933bf1b5214b95008304ff337c`。04:59:51 UTC[实际exit0](results/hs_joint_20261101/ordered_path_independent_run_v2.execution.json)，独立重算402个区间、670项标准均值、1700项suite均值，以及1500项ordered标量和1200项lag支持记录。196456次数值比较最大差2.27e−13，不调用producer统计、不重复加载checkpoint张量。[核验结果](results/hs_joint_20261101/ordered_path_verification_v2/verification.json)SHA-256 `52eaa9dbdd1dc70e896cd58d1160571a32b0ff459dbce5ba5c63d59dc6502fb8`；[正式汇总](results/hs_joint_20261101/ordered_path_v1/summary_v2.json)于04:51:31 UTC实际完成，SHA-256 `1585193a7c1fad16bd7866ee42bffe20b9a47333c1bb03b5be85aca327e4d81d`。

### 连续路径对照的完整读数

预定16/64 ns联合路径coordinate U-ES为joint **1.633475 Å**、marginal **1.665881 Å**；joint−marginal为 **−0.032406 Å，95% CI [−0.084580,+0.002483]**。次要long-endpoint U-ES为1.980917/2.020983 Å，差−0.040066 [−0.099524,+0.001033]。两个区间均跨0，不支持升级joint方案，也不能证明等效。

| 历史七指标，overall | Joint | Marginal | 保存的BioEmu2 | Joint−marginal的95% CI |
|---|---:|---:|---:|---|
| RMSF MAE ↓ | 1.376511 | 1.302179 | 1.991674 | [0.026123, 0.122332] |
| RMSF correlation ↑ | 0.505090 | 0.512944 | 0.579744 | [-0.015158, -0.001316] |
| Root mean W2 ↓ | 1.671418 | 1.637023 | 2.290364 | [0.017017, 0.053233] |
| Variance W2 ↓ | 9.554133 | 9.042571 | 18.575412 | [-0.037322, 1.090823] |
| MD PCA W2 ↓ | 6.501442 | 6.559375 | 6.114255 | [-0.151590, 0.052440] |
| Weak-contact Jaccard ↑ | 0.098605 | 0.101267 | 0.131606 | [-0.004216, -0.001118] |
| Transient-contact Jaccard ↑ | 0.229054 | 0.234348 | 0.265260 | [-0.007733, -0.002936] |

Joint相对marginal在MAE、correlation、Root和两类contact上退步，以上五项差的区间均不跨0；Variance和PCA差跨0。相对保存的BioEmu2，两臂仍为MAE/Root/Variance改善、correlation/weak/transient退步，PCA差不确定。全部温度和horizon的原始及配对结果保留于正式汇总。

| 时间匹配评分，overall ↓ | Joint | Marginal | Joint−marginal的95% CI |
|---|---:|---:|---|
| Endpoint coordinate U-ES (Å) | 1.510834 | 1.530257 | [-0.049227, 0.001323] |
| Endpoint distance V-ES（distance/1 Å） | 2.103459 | 2.112942 | [-0.044425, 0.016076] |
| Endpoint hard-contact Brier | 0.016764 | 0.016822 | [-0.001061, 0.000577] |
| Joint coordinate-path U-ES (Å) | 1.310987 | 1.326943 | [-0.042124, 0.001669] |
| Joint distance-path U-ES（distance/1 Å） | 1.675277 | 1.686449 | [-0.041452, 0.009632] |

两臂overall的两种联合路径评分都优于persistence；joint的coordinate-path差为−.473053 [−.593605,−.363371]。这不等于时间相关性已正确：joint的64 ns路径长度67.912 Å，MD为130.227 Å；lag-1 ACF为.835742，MD为.521274。h4的joint distance-path评分还劣于marginal，差+.005774 [.000591,.011742]。

规范CA非局部clash比例joint/marginal为.00067228/.00056674，joint−marginal为+.00010554 [.00000206,.00027016]，也有退步；无效键比例为.00407499/.00371323，差区间跨0。键长均值与Rg仅描述，不设单调优劣方向。

h64的joint原顺序/反序/打乱路径coordinate U-ES为1.847587/1.867347/1.858246，仅为描述性时间控制，不附加未登记的优越性检验。完整面板中的walk-RMSF或ACF相关性不可定义时仍保留null与支持记录；不以可用子集均值取代完整面板。

| Fullsuite训练臂 / 温度输入 | RMSF MAE ↓ | RMSF corr ↑ | Root W2 ↓ | Variance W2 ↓ | PCA W2 ↓ | Weak J ↑ | Transient J ↑ | 成员CA RMSD ↓ (Å) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| joint_path / correct | 1.457529 | 0.447141 | 2.115954 | 14.439000 | 9.699229 | 0.109582 | 0.246630 | 5.798562 |
| joint_path / wrong | 1.586201 | 0.456149 | 2.149159 | 16.069662 | 10.223106 | 0.113239 | 0.253897 | 5.677882 |
| joint_path / none | 1.392526 | 0.462934 | 2.068775 | 14.592391 | 10.073416 | 0.116560 | 0.259944 | 5.585005 |
| marginal_path / correct | 1.437827 | 0.458087 | 2.101092 | 14.599152 | 9.841614 | 0.112657 | 0.252240 | 5.726117 |
| marginal_path / wrong | 1.600631 | 0.464067 | 2.149090 | 16.280776 | 10.330055 | 0.114799 | 0.256875 | 5.637855 |
| marginal_path / none | 1.410898 | 0.471996 | 2.068823 | 14.993137 | 10.222543 | 0.119338 | 0.265553 | 5.531505 |

每臂fullsuite为12蛋白、300轨迹、7200窗；correct/wrong/none各2400窗。Persistence成员CA RMSD为4.930655 Å，correct模式joint/marginal的position skill为−.867908/−.795462 Å；matched8也为−.838731/−.758390 Å。正确温度并未在全部指标上带来改善，不能把输入温度控制当作未见温度T3。这里的成员CA误差使用共同观察锚点；不是逐预测独立Kabsch拟合。

| 温度 (K) | Full12 coverage J / M ↑ | Matched8 coverage J / M ↑ | Matched8 population TV J / M ↓ |
|---|---|---|---|
| 320 | 0.489583 / 0.491667 | 0.403125 / 0.409375 | 0.797108 / 0.809401 |
| 348 | 0.393750 / 0.393750 | 0.262500 / 0.265625 | 0.746196 / 0.753768 |
| 379 | 0.227083 / 0.227083 | 0.221875 / 0.218750 | 0.719882 / 0.730873 |
| 413 | 0.106250 / 0.106250 | 0.090625 / 0.090625 | 0.626745 / 0.636216 |
| 450 | 0.016667 / 0.016667 | 0.015625 / 0.015625 | 0.600476 / 0.603080 |

J/M分别为joint/marginal。Matched8仅8蛋白、200轨迹、800窗，覆盖/布居使用每轨迹48成员；full12为96成员。历史BioEmu2的mask/reference/状态中心指纹缺失，严格配对T2A差继续为null；未据描述性均值作胜出声明，也未重跑基线。两臂matched8的population TV在五温度的描述均值均高于前表的历史BioEmu2。

这些仍是单seed、自适应开发验证集中的边际探索区间，未涵盖训练seed、开发选择或多重比较不确定性。H gate、完整T1、T2B/MSM、192 ns新候选和独立确认均未因本实验补齐。**全指标超越BioEmu2的目标仍未达到。**

## 固定H的局部残差读出对照完成（2026-11-02）

H oracle gate仅约束当前编码器、解码器和投影的组合误差，不能解释为任意读出器的不可约误差下界，也不能证明H没有所需信息。冻结H的路径训练仍不能声称修复这项工程门槛。另一个同H、同初始化解码器的嵌套读出对照已经完成：A续训原D0，B增加零初始化的384参数局部向量分支，两者保留相同外部P32；其模块25项测试及独立复跑通过，零分支时输出与旧参数/H梯度逐位一致。两组各500次optimizer调用（各499次非零学习率）及全部评分已完成，正式汇总和独立核验均实际exit0。

该读出对照可复用train500/val200缓存，700窗CPU审计通过且train/val蛋白不重叠，无需新HDF读取、编码或采样。已知7个训练窗的旧residue-index连接与PSF不同，两臂均保持旧P32以隔离容量干预，另报真实PSF几何。缓存没有实际`time_ps`张量，不能把按frame ID及记录间隔重建的时间当作新时间戳核验。缓存也只有终点标签，故不计算需要整条未来路径的历史七指标。主指标预定为val的16/64 ns oracle固定观察锚点CA-RMSD之B−A；forecast12仅诊断原spatial_exact采样H在新解码器中的兼容性，不是新模型反馈采样。anchor-H和固定跨horizon错配H只作描述性诊断；严格同protein、同horizon的非恒等时间置换因每格仅一窗不可用。

读出对照的[统计协议](results/hs_joint_20261101/ca_local_residual_v1/summary.protocol.json)已于00:56:40 UTC、首次smoke/训练之前登记，SHA-256 `6019841688e0e39fb5bf52b27d9f04d05771e7db96d69fb4091e68f8746d87cf`；21项数值测试通过，独立审查完成。共1501个预定区间格子（1500项次要比较加1项主比较），两阶段raw/P32、3个对照、全部10个val分组均保留；明确null使完整面板的次要区间为null并单列支持，不删窗。主指标要求完整有限支持，否则汇总失败。训练拟合及两类H控制只描述；全部区间为单seed、自适应开发中的边际探索结果，不作多重比较或模型选择调整。


### 实际执行与读出结果

运行协议SHA-256 `0586051c18dcccb717a875e8bbdca0c58936dcf3e77075d06e21906295f83cf7`，汇总实现协议SHA-256 `706b5e6f734048654d1f4f5e02cb2da932788adbf86ada3565675cbd82e3c02e`，独立核验协议SHA-256 `50856217df78489f613db69f89568bfb935748b73fa73bca01e062a1459d6ca5`，均在本轮训练前登记。第一4窗smoke零optimizer、三种初始读出raw/P32逐位相等；随后全700窗零更新检查通过。CPU2单一真实进程顺序训练两臂及评分，[正式运行](results/hs_joint_20261101/ca_local_residual_run_v1.execution.json)于05:01:12 UTC实际exit0，耗时127.502秒。最终optimizer状态全部500步，两臂训练批次文件逐字节相同、最终RNG相同，冻结输入/源码前后不变。

[汇总](results/hs_joint_20261101/ca_local_residual_v1/summary.json)实际exit0，SHA-256 `d539fc4c2c62434275f71a9a0ed349a47e231191fc7c9638f9731947120a080a`。[独立NumPy复核](results/hs_joint_20261101/ca_local_residual_verification_v1/verification.json)实际exit0，34项合成测试先行通过；其SHA-256 `110dca75a39cc9f51333e81712723a58818dbfbc5089ed292f354977aca57a24`。1501个区间全部有完整支持，3480条描述性支持记录保留；另从保存坐标独立重算700窗、7800个坐标叶、21000个成员及1200组预测系综评分，757332次数值比较最大差4.55e−13。独立程序仅加载坐标sidecar，没有再次加载模型/cache张量或运行decoder；初始化和optimizer状态核验由汇总器承担。

主指标16/64 ns projected oracle CA-RMSD为B **1.833334 Å**、A **1.834329 Å**，B−A为 **−0.000994 Å，95% CI [−0.003571,+0.001444]**，不支持新增分支的优势。以下D0为冻结基线，A为base_tuned，B为local_residual。

| 验证集oracle，CA-RMSD ↓ (Å) | D0 | A | B | B−A及95% CI |
|---|---:|---:|---:|---|
| raw | 1.282664 | 1.298225 | 1.297561 | -0.000664 [-0.002273, 0.000984] |
| projected | 1.261451 | 1.276133 | 1.275573 | -0.000560 [-0.001996, 0.000930] |

两种续训读出器在验证集oracle上均比D0更差：projected A−D0为+.014682 [.007660,.022012] Å，B−D0为+.014122 [.006861,.021499] Å。训练集projected oracle RMSD由D0的1.706282降至A的1.668349、B的1.667006 Å；训练拟合改善没有转化为此验证面板的改善。

| Horizon (ns) | Raw B / A (Å) | Raw B−A的95% CI | Projected B / A (Å) | Projected B−A的95% CI |
|---|---|---|---|---|
| 1 | 0.647267 / 0.647015 | [-0.001273, 0.001912] | 0.629226 / 0.628961 | [-0.001411, 0.001980] |
| 4 | 0.825546 / 0.826346 | [-0.002383, 0.000920] | 0.806396 / 0.806912 | [-0.002308, 0.001439] |
| 16 | 1.469550 / 1.468807 | [-0.001827, 0.003158] | 1.445953 / 1.445124 | [-0.001666, 0.003203] |
| 64 | 2.247880 / 2.250734 | [-0.007356, 0.001652] | 2.220716 / 2.223534 | [-0.006947, 0.001077] |

| 缓存预测兼容性，overall ↓ | D0 | A | B | B−A及95% CI |
|---|---:|---:|---:|---|
| raw CA-RMSD (Å) | 5.105383 | 5.119048 | 5.210060 | 0.091012 [-0.006538, 0.283787] |
| projected CA-RMSD (Å) | 4.612087 | 4.621844 | 4.640295 | 0.018451 [0.000355, 0.052631] |
| raw Coordinate U-ES (Å) | 1.557168 | 1.563211 | 1.574777 | 0.011566 [-0.002409, 0.038310] |
| projected Coordinate U-ES (Å) | 1.535685 | 1.541028 | 1.544651 | 0.003623 [-0.000445, 0.010697] |
| raw Distance V-ES（distance/1 Å） | 2.250054 | 2.266783 | 2.316483 | 0.049699 [-0.005248, 0.157527] |
| projected Distance V-ES（distance/1 Å） | 2.141909 | 2.151536 | 2.156773 | 0.005237 [-0.001575, 0.017455] |
| raw Hard-contact Brier | 0.016673 | 0.016658 | 0.016684 | 0.000025 [-0.000005, 0.000057] |
| projected Hard-contact Brier | 0.016530 | 0.016533 | 0.016545 | 0.000013 [-0.000022, 0.000040] |

B的projected预测RMSD比A增加+.018451 [.000355,.052631] Å，16 ns单独也变差+.004290 [.000596,.007926] Å。B相对D0的projected预测RMSD、coordinate U-ES和distance V-ES分别增加+.028208 [.006849,.067330]、+.008966 [.003835,.016707]、+.014864 [.006708,.025632]；Brier差仍跨0。Raw预测RMSD在h1/4/16的B−A区间小于0，但64 ns点估计增加.373883 Å且区间跨0；不能用短时域的raw改善替代最终P32预测表现。

Anchor-H控制三臂raw/projected均为3.521675 Å、逐窗persistence ratio为1；B的错误horizon H读出raw/projected RMSD为3.818044/3.809011 Å，远差于正确oracle的1.297561/1.275573 Å。B的projected预测RMSD为4.640295 Å、逐窗ratio均值1.553805，仍差于anchor identity。跨horizon控制说明当前读出依赖H的内容，但未来oracle H不能作为预测能力证据，也不证明JEPA预训练收益。

| PSF几何，overall | D0无效键比例 | A无效键比例 | B无效键比例 | D0 clash比例 | A clash比例 | B clash比例 |
|---|---:|---:|---:|---:|---:|---:|
| oracle / raw | 0.147705 | 0.153996 | 0.154852 | 0.000647 | 0.000676 | 0.000682 |
| oracle / projected | 0.001474 | 0.001474 | 0.001474 | 0.000593 | 0.000641 | 0.000618 |
| forecast / raw | 0.623104 | 0.627776 | 0.628750 | 0.000537 | 0.000526 | 0.000525 |
| forecast / projected | 0.005278 | 0.005461 | 0.006198 | 0.000626 | 0.000616 | 0.000603 |

B的projected oracle clash比例相对A降低−.00002337 [−.00004428,−.00000686]，但这不改变主指标与预测兼容性结论；projected预测无效键比例B−A为+.00073684 [−.00010088,.00236842]、clash差−.00001300 [−.00004087,.00000247]，均不确定。完整raw/P32、canonical/PSF、四horizon和五温度的结果均在汇总中保留。

**本预算不支持采用局部残差分支，也不证明任意扩容无效或H信息不足。** 这是增加参数并改变优化路径的干预，非参数量匹配；预测兼容性沿用原模型的缓存H，没有新动力学采样或反馈重编码。未运行历史七指标、完整T1/T2A/T3/MSM或BioEmu2比较，不能替代主目标验收。

当前全指标目标、未见温度T3、完整T1/T2B和独立确认的缺口仍在。

## 有限成员接触损失：固定训练32窗梯度诊断（2026-11-02）

当前训练先平均12个成员的soft contact，再计算平方误差。固定观察窗口、模型与目标q，并假设成员条件独立时，E[(mean(z)−q)²]=(E[z]−q)²+Var(z)/12；后者是额外的成员方差惩罚。新原语将其分成plug-in Brier、无偏sample variance/12和二者之差；差值等于不同成员残差乘积的平均，允许有限样本负值。有限系综的这种估计目标区别见[Ferro的原始论文](https://empslocal.ex.ac.uk/people/staff/ferro/Publications/ferro2013.pdf)；soft contact的扩展由本地代数推导与测试支持，仅针对条件均值，不唯一识别完整接触分布。历史评测中的经验Brier口径保持原样。

[固定诊断协议](results/hs_joint_20261101/contact_bias_train32_v1/protocol.json)SHA-256 `33d9477b15a90687d38517e836612f0d083096cdd0ef49f924af02874832edd4`，在新张量读取之前登记。54项原语测试和15项诊断测试通过并分别独立复跑；代码及科学审阅完成。沿用microstep诊断已固定的32个训练蛋白、每蛋白一窗、每horizon八窗，只读取one-microstep分支的12成员终点坐标和既有目标。没有实例化模型、解码、采样或更新参数。

[真实运行](results/hs_joint_20261101/contact_bias_run_v1.execution.json)于05:15:21 UTC实际exit0，16.359秒，32窗均完成。结果[diagnostic.json](results/hs_joint_20261101/contact_bias_train32_v1/diagnostic.json)SHA-256 `66ab40aa3fb4d956c24fec124a3124e240c8eb993a9cadf50f8921f013905349`。对保存float32坐标转float64后建立独立endpoint CA叶子，计算UES+20×Brier+40×clash的坐标梯度；latent .1项不是该独立坐标叶子的函数，因此排除，不能称为完整训练参数梯度。原始输入及源文件前后哈希相同。

| Horizon | Plug-in Brier | 方差惩罚V | 逐窗V/Brier均值 | 逐窗‖20∇V‖/‖∇总坐标损失‖均值 | 惩罚与坐标UES梯度的余弦均值 |
|---|---:|---:|---:|---:|---:|
| overall | 0.01327872 | 0.00089914 | 7.551% | 2.839% | -0.118719 |
| 1 | 0.00949263 | 0.00036311 | 5.311% | 2.852% | -0.169497 |
| 4 | 0.01294199 | 0.00084495 | 8.223% | 3.151% | -0.107041 |
| 16 | 0.01962852 | 0.00142142 | 7.678% | 2.556% | -0.129535 |
| 64 | 0.01105173 | 0.00096708 | 8.992% | 2.796% | -0.068803 |

总体V/Brier的逐窗比值均值为7.551%，而均值之比为6.771%；两者不可混用。总体坐标损失中20V的均值为.01798279，总损失均值2.47764596，均值之比仅.7258%。惩罚梯度与坐标UES梯度在32/32窗都反向（余弦平均−.118719），与合并坐标损失则为19/32窗反向。去掉惩罚前后的总梯度余弦平均.9995929、最低.9993136，表明当前坐标空间扰动较小，不能把它直接认定为主要瓶颈。

独立标准库只读复算全部350项聚合（70项×overall及四horizon）、2240个有限标量和支持记录，与保存结果一致；逐窗乘积式与原训练损失的值/梯度恒等关系也核对通过。最大contact旧实现值差2.59e−15、梯度L2差4.27e−16，独立U-ES值差1.33e−15、梯度L2差8.06e−18。没有缺失窗或null项，没有bootstrap、验证集胜出或新候选模型结论。

后续已实现独立版本的接触校正路径损失：保持marginal监督、单次连续rollout及H/D/P32不变，只在校正臂减去20V；plugin控制保留原损失张量及运算顺序。16项合成测试覆盖原控制数值/完整参数梯度/RNG逐位一致、真实几何decoder+P32、全部成员梯度和未来隔离。在这一实现检查时点，该代码尚未用于训练；随后先完成固定训练前四窗的实际kernel梯度检查，再于下文所记05:52 UTC启动配对训练。梯度检查用于判断经过解码器/投影/动力学及latent项后相对量级是否仍小。

参数梯度诊断v1于05:29:20 UTC实际exit1，见[失败记录](results/hs_joint_20261101/contact_kernel_gradients_run_v1.execution.json)。错误来自把旧smoke中获得梯度的64个参数张量当作kernel总参数数；只读核对实际初始模型为73个、均设置requires_grad。检查在载入模型后的数量断言处中止，尚未读取训练轨迹、采样或求损失梯度，零optimizer更新。v1源码、协议、测试、日志和失败execution全部冻结保留；另建v2修正参数清单契约，未修改损失、数值容差或选择窗口。

### 实际kernel梯度v2完成

v2的17项测试通过，包含直接构造生产kernel的CPU参数清单回归；独立代码审查确认仅修正数量契约并加入清单和既有失败证据。新[协议](results/hs_joint_20261101/contact_kernel_gradients_v2/protocol.json)SHA-256 `7b71a4511cbbab89b1bd35eb5a99910e719183d4ae4593edc59ba113268cb44a`，在实际采样前登记。GPU运行于05:38:37 UTC实际exit0，耗时25.093秒；[结果](results/hs_joint_20261101/contact_kernel_gradients_v2/result.json)SHA-256 `bd03074123092497c83d85ee7118934f79d10ee2c677a68c5f02c163948bf4fc`。四窗均使用同一未更新初始模型，按旧训练前四窗与连续seed42噪声流运行，不能当作旧训练第2至4步的已学习状态。

每窗单次forward包含完整latent项、解码器/P32及动力学梯度；分别对plugin总损失、20V和实际校正损失求导并保存全部73个张量，每个完整梯度向量含669429个元素。无optimizer，参数`.grad`仍为None，模型哈希不变；反向重算未消耗额外全局或采样RNG。

| Horizon | ‖原总梯度‖ | ‖20∇V‖ | 范数比 | 惩罚与原总梯度余弦 | 原总梯度与校正后余弦 |
|---|---:|---:|---:|---:|---:|
| 1 | 1.409888 | 0.021089 | 1.496% | -0.432957 | 0.999910 |
| 4 | 7.229771 | 0.514977 | 7.123% | -0.964492 | 0.999845 |
| 16 | 3.028904 | 0.049937 | 1.649% | -0.405589 | 0.999888 |
| 64 | 4.386305 | 0.134154 | 3.058% | 0.174458 | 0.999542 |

全部三种导数均有9个未连接张量：8个history_posterior参数及最终energy-head标量偏置。它们在完整清单中保留并仅在诊断sidecar中填零。第3窗另有两个已连接但数值全零的温度投影weight，因此64个已连接张量中只有62个非零；不能把“非零”“已连接”或“requires_grad”计数混为一谈。单位裁剪的结果也保留，但仅为float64向量公式，不是AdamW更新预测。

独立CPU/NumPy从12个保存梯度向量重算448项整体及module统计，含36个预期null，最大差1.075e−13；逐元素校正恒等式全部通过预登记容差，最大绝对残差9.686e−8、最大相对L2残差2.288e−7。108项源码及108项快照、14项输入、全部sidecar/协议/结果哈希和记录的模型/RNG不变量均核对一致。h64峰值GPU分配约4.993 GiB。只读复核未重新运行模型、trajectory或验证评分。

这四个固定TRAIN点说明校正经过完整kernel后仍属较小方向扰动，且与原总梯度的夹角符号并不统一；不支持主要瓶颈或预测收益结论。下一项为新版本的plugin/population_u配对500步训练，保持marginal监督、H/D/P32、初始化、批次和噪声流相同，保留长期joint-path U-ES主指标及全部既有评测面板。未用于正式训练前不得把校正臂称为改进模型。

### 接触估计器配对训练已登记并启动

新训练器17项、运行器22项、统计适配8项测试通过；实现和科学审查完成。运行器在读取最终checkpoint时核验完整73个参数的顺序、优化器单组73个ID、64个活跃状态及9个未使用参数逐位未变。训练采用原ordered学习率计划，第一步2e−6、500步均为正学习率；与另一项CPU读出实验的首步零学习率不同。

[统计协议](results/hs_joint_20261101/contact_estimator_v1/summary.protocol.json)于05:50:57 UTC前登记，SHA-256 `5c5cfc6689ebc7ad89c076c8e021736b8960d084e2439ccfa45725718adf0c26`。固定step4/h64/L96/M12 smoke于05:51:35 UTC实际exit0，总耗时32.983秒，零optimizer；两臂73个参数、64个梯度、冻结组件无梯度，模型哈希前后相同且采样RNG相同。峰值GPU分配plugin约4.990 GiB、population_u约4.999 GiB。该smoke重置每臂seed42，因此其h64噪声与连续四窗梯度诊断不同，不能直接比较两个h64梯度范数。

[正式协议](results/hs_joint_20261101/contact_estimator_v1/protocol.json)于05:51:59 UTC前登记，SHA-256 `72f18dc2a0b569ad12d55b16fa16b25803e6208208b11e03516950214f5d73ba`；两臂训练05:52 UTC启动，GPU0为population_u、GPU1为plugin。两组均从同一spatial初始化重新训练，沿用完全相同的旧500窗marginal路径监督；旧marginal结果不替代本轮控制臂。全部12个逻辑任务预定保留，不因中间结果取消不利面板。在该启动记录时点仍在执行、尚无预测改进结论；全部任务现已完成，实际结果见文末完成节。

汇总实现37项测试通过并独立审查，于05:56:22 UTC实际登记exit0，[实现协议](results/hs_joint_20261101/contact_estimator_summary_v1/protocol.json)SHA-256 `14d31a65af0e97dba8ce31ffaa87278a0c93de696898bb853f59508b61fe0697`。独立NumPy核验46项合成测试通过并审查，于06:01:42 UTC实际登记exit0，[核验协议](results/hs_joint_20261101/contact_estimator_verification_v1/protocol.json)SHA-256 `43c3c36c75fa4a9d46bd3e9704a931d3d7a3120870ad43058d2a855e90ebe6c6`。二者实现登记发生在训练过程中、读取本轮预测评分之前；科学统计定义已在训练前冻结。当时完整汇总与核验仍待全部12个实际任务完成；现已实际exit0并独立核验，结果与边界见文末完成节。合成测试通过本身不替代实际核验。

## 固定训练四窗全路径读出诊断完成（2026-11-02）

**这四个固定TRAIN窗口中，真实H经冻结读出器/P32保留了大部分MD路径运动；自主forecast路径的运动更小。** 这将当前排查重点指向生成H的路径及其与读出器的兼容性，但没有单独隔离kernel与分布外读出的因果贡献。它不证明H的信息上限、任意读出器的误差下界、验证集泛化或归一化修复有效。

可视化见[PNG](results/hs_joint_20261101/train_path_readout_figures_v1/train_path_readout.png) / [PDF](results/hs_joint_20261101/train_path_readout_figures_v1/train_path_readout.pdf)：上方四个独立窗口面板保留全部六角色的shape-aligned路径长度，各自纵轴尺度；下方为单个h64窗口的真实/forecast H中心化范数曲线。[绘图脚本](results/hs_joint_20261101/train_path_readout_figures_v1/plot.py)只读取已核验JSON，不产生新统计检验；[provenance](results/hs_joint_20261101/train_path_readout_figures_v1/provenance.json)记录输入、源码及两种图件的SHA-256，输出拒绝覆盖。

[设计说明](docs/TRAIN_PATH_READOUT_DIAGNOSTIC_V1_20261102.md)与[执行前协议](results/hs_joint_20261101/train_path_readout_v1/protocol.json)固定旧ordered训练流的前四个实际批次，step1/2/3/4分别为h1/4/16/64，实际batch张量哈希与历史记录相同。四窗均使用已完成的旧marginal-path latest500，checkpoint SHA-256 `0f45200e331df6dc50c75cd0aec870dc965948fbbd4a5b430bd5d7ef2147886f`；并非原训练各步的参数状态。CPU float32推理、两线程、连续seed42采样流，每窗12个forecast成员；CPU随机抽样不是旧GPU训练噪声的逐位回放。没有optimizer、验证/test读取或BioEmu2重跑，也没有读取正在运行的接触配对实验的新评分。

协议于06:18:14 UTC登记，SHA-256 `8f2b3c36a5342f0a6d3fc9736f504d2be3dee98def9a6db0e971a7665abb630d`。[实际外层运行](results/hs_joint_20261101/train_path_readout_run_v1.execution.json)于06:18:34–06:18:59 UTC完成，exit0，耗时25.017秒；[内部执行记录](results/hs_joint_20261101/train_path_readout_v1/execution.json)为PID 2111248、23.469秒。[完整结果](results/hs_joint_20261101/train_path_readout_v1/result.json)SHA-256 `1c4509f82a5aa1ac81271bfca3bef7d813bc0c620ed640e62368d776792295a5`，四窗、六角色及所有成员/时间均保留，无失败替换。

六角色依次是：真实MD对齐到观察anchor；该MD经生产float32 P32；真实未来backbone经当前冻结encoder取得H后解码的oracle raw/P32；一次observed-only连续H/S rollout的forecast raw/P32。这里的真实H来自结构encoder，**不是Stage-A teacher/predictor**。forecast完成后才编码未来结构，所有读出均使用原观察anchor的H、CA和frame，未来H不进入自主rollout。每条坐标路径的t0都精确等于观察anchor；MD/oracle各一条路径，forecast各12条。

下表为**shape-aligned路径长度**（Å）：每个预测帧独立对齐到观察anchor，再累计每步的有效残基位移范数均值；forecast格子为12个成员路径长度的均值。四行属于不同蛋白，不能当作同蛋白的时间曲线，没有跨horizon平均或CI。

| Horizon (ns) / 蛋白 / 温度 | MD | MD P32 | Oracle raw | Oracle P32 | Forecast raw | Forecast P32 |
|---|---:|---:|---:|---:|---:|---:|
| 1 / 3l41A01 / 450 K | 5.634518 | 5.633981 | 5.625705 | 5.596506 | 1.813089 | 1.706689 |
| 4 / 3tpqM02 / 320 K | 6.481496 | 6.481820 | 6.410013 | 6.342879 | 3.394403 | 3.086006 |
| 16 / 1efdN01 / 348 K | 19.938947 | 19.925746 | 19.675799 | 19.603652 | 12.805100 | 11.879914 |
| 64 / 3aq5A00 / 320 K | 62.725661 | 62.607501 | 63.432035 | 63.022606 | 38.912034 | 35.990271 |

另一个明确区分的量是**anchor-gauge路径长度**（Å）：预测坐标保持原观察anchor坐标系，不对预测重新拟合。MD本身已经逐帧对齐；其余角色不能将下表与上表混用。保存的每步increment RMSD为残基平方位移均值的平方根，路径长度则累计残基位移范数的均值，两者也不同。

| Horizon (ns) | MD | MD P32 | Oracle raw | Oracle P32 | Forecast raw | Forecast P32 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 5.634518 | 5.633985 | 5.554483 | 5.526373 | 1.817351 | 1.713044 |
| 4 | 6.481496 | 6.482104 | 6.420051 | 6.350394 | 3.478673 | 3.183232 |
| 16 | 19.938947 | 19.924844 | 19.760073 | 19.680101 | 13.199320 | 12.310927 |
| 64 | 62.725661 | 62.607366 | 63.713850 | 63.313126 | 40.945844 | 38.193358 |

### 单个h64窗口的时间、重建与几何证据

h64窗口包含65个时刻、96个有效残基。内部距离特征固定512个physical nonlocal CA对，包含t0；lag1/2/4/8/16/32每条路径分别有64/63/61/57/49/33个合法时间对，forecast分别为这些计数的12倍，不跨成员连接。下表ACF为各路径自身去时间均值后的有限路径估计；MSD为内部距离变化平方均值（Å²），不能解释为原子扩散系数或弛豫时间。重建/预测RMSD为原anchor坐标系中逐未来帧、逐成员CA-RMSD的均值，只对真实参考做刚体对齐。

| h64角色 | Lag-1 ACF | Lag-1距离MSD (Å²) | 未来CA-RMSD (Å) | 未来无效CA键比例 | 未来CA clash比例 |
|---|---:|---:|---:|---:|---:|
| MD | 0.477522 | 0.800513 | 0.000000 | 0.000000 | 0.000000 |
| MD P32 | 0.478162 | 0.798623 | 0.057629 | 0.000000 | 0.000000 |
| Oracle raw | 0.477161 | 0.855198 | 0.279642 | 0.011513 | 0.000000 |
| Oracle P32 | 0.476196 | 0.844668 | 0.265595 | 0.000000 | 0.000000 |
| Forecast raw | 0.870864 | 0.398966 | 3.083740 | 0.671820 | 0.000195 |
| Forecast P32 | 0.864991 | 0.344501 | 2.816933 | 0.000000 | 0.000192 |

几何列是此窗口所有未来帧/成员的等权描述均值，排除t0；原JSON另保留包括t0的每成员每帧结果。使用**canonical CA**连接：相同chain且residue-index相邻的键长须在3.3–4.3 Å，非局部CA clash阈值2.5 Å；没有读取PSF连接，不能称为PSF几何或完整backbone有效性。Raw forecast虽然运动略大，但无效键比例67.18%，不能将其当作改善；P32修复该CA键检查后仍有clash，且路径运动仍明显低于MD。真实MD经P32的路径长度与时间统计变化很小，oracle raw/P32也接近MD；这四窗不支持把P32单独视为主要运动压缩来源。

h64真实H的末帧通道均值为2.56e−9、通道方差0.999997、中心化L2范数11.313693；forecast相应为−0.067795、6.375229、27.545828，均先逐token计算再对有效残基/成员平均。末帧forecast token中心化范数范围16.976243–58.344139。整个64步的H增量范数均值，真实/forecast为3.282447/3.618011；H空间运动并不小，不能由它直接推导坐标运动或有效物理时间。真实H保持encoder末层非仿射LayerNorm的近零均值/近单位方差支持，而自主生成H离开该支持；这是分布兼容性的描述，不证明信息消失、范数饱和机制或必须归一化。

先前固定TRAIN32的读出前LayerNorm已经使coordinate U-ES由**2.2033恶化到2.5667 Å**。本次四窗结果没有推翻该负结果；后续若在保存H上做归一化等反事实，也只能另行回答其限定问题，不能抹去已有评分、更不能在未检验前采用为修复。当前没有新候选训练、验证集性能或推广结论。

### 独立只读复核

独立CPU/NumPy审计仅以`torch.load(..., map_location='cpu', weights_only=True)`读取四份保存sidecar，未导入producer数值函数、加载模型、重跑decoder/P32或读取新轨迹。重算全部24个角色：425308个H描述张量元素、2112个H曲线数值、3002个anchor运动数值、2492个成员/帧的9968个几何标量，以及全部shape路径、ordered曲线/标量和两种U-energy score。1104条lag记录的5388个数值字段、828个null，以及50个ordered scalar null均核对一致。全体最大绝对差**4.62e−14**，geometry和lag统计最大差均为3.55e−15；没有超容差失败。单路径MD/oracle的U-score及短路径不可定义的量仍保留null。

另核对104个源码及104个快照、7项输入文件哈希、3份历史metadata快照、4个HDF大小/mtime，以及全部sidecar文件/递归张量哈希和历史实际batch张量哈希；没有把HDF stat当作完整文件哈希。初始真实/forecast H相等、anchor/实际时间网格和forecast之后RNG未再变化均从sidecar重核。记录中的初末模型哈希、零更新、实际exit0一致；模型内部不变与forecast P32精确等于adapter输出仍是producer执行时的检查证据，独立审计没有重新执行模型来重复这些断言。

## 保存H的观察半径映射反事实完成：结果为负（2026-11-02）

**联合去均值、缩放到观察H0半径的离线读出映射没有修复这四窗的运动与分布评分，不采用。** Raw与P32的coordinate-path和distance-path U-score均在四窗全部变差，shape-aligned路径长度也全部减少。它是固定生成H上的读出干预，没有重新运行动力学或将映射后的H反馈给kernel；不能据此单独断言范数漂移导致运动变慢，也不能把均值与半径两项作用分开归因。

[设计说明](docs/SAVED_H_SUPPORT_READOUT_DIAGNOSTIC_V1_20261102.md)、[实现](run_scripts/diagnose_saved_h_support_readout_v1.py)及[17项测试](tests/test_saved_h_support_readout_v1.py)在执行前通过并冻结。[协议](results/hs_joint_20261101/saved_h_support_readout_v1/protocol.json)于06:39:49 UTC登记，SHA-256 `0e61fa5fe1f8e80b6800fbafdac9a43878d214967d6403f97212745d839e4ba8`。沿用上一节四窗的保存H、CA、观察anchor和时间网格，CPU两线程，仅加载冻结decoder，未构造kernel、未打开轨迹文件、零新rollout/optimizer，未读val/test或重跑BioEmu2。原样重解码的baseline raw/P32坐标与父实验保存坐标逐位相同。

[实际外层运行](results/hs_joint_20261101/saved_h_support_readout_run_v1.execution.json)于06:39:55–06:40:09 UTC完成，exit0，耗时13.855秒，该execution SHA-256为 `97cc67548d329fc44c34ba0143a760c80989507980b339ddd9120fe77871d125`。[内部执行](results/hs_joint_20261101/saved_h_support_readout_v1/execution.json)为PID 2117096、12.452秒，SHA-256 `4350a6120396ac2ff3929e28432e70ce0445932cc769f64ff371ef76ab14b3d9`。[完整结果](results/hs_joint_20261101/saved_h_support_readout_v1/result.json)SHA-256 `b18b87e60e30699319c0b2c3118e8627026f2a47f92f77407ec74ea70d31887b`，四窗均完成，全部baseline与mapped角色保留。

映射对每个未来有效token使用`(H − channel_mean(H)) × r0 / centered_norm(H)`，其中每残基的`r0`仅由同一观察H0计算；float64完成去均值/范数/缩放后转回float32，没有epsilon、下限或clamp。t0保持原始字节，padding也按契约保持原样。真实未来H只接受同一个观察H0半径映射作为H-only control，不用于估计映射系数，也不进入decoder目标或动力学。

下表为四个独立TRAIN窗口的完整raw/P32配对。每格为**原样baseline → 映射后**；两种U-score均越低越好，coordinate单位Å、distance为固定物理距离特征的归一化评分。路径长度为前节定义的shape-aligned长度（Å），forecast均为同一12成员路径的均值。未做跨horizon汇总、bootstrap或CI。

| Horizon (ns) / 蛋白 | 读出 | Coordinate-path U-score ↓ | Distance-path U-score ↓ | Shape路径长度 (Å) |
|---|---|---|---|---|
| 1 / 3l41A01 | raw | 3.027545 → 3.123330 | 3.340945 → 3.693421 | 1.813089 → 1.768739 |
| 1 / 3l41A01 | P32 | 3.056027 → 3.147176 | 3.413099 → 3.745487 | 1.706689 → 1.702698 |
| 4 / 3tpqM02 | raw | 1.001860 → 1.081433 | 1.183452 → 1.517323 | 3.394403 → 3.159063 |
| 4 / 3tpqM02 | P32 | 1.007937 → 1.092930 | 1.219478 → 1.552867 | 3.086006 → 2.967792 |
| 16 / 1efdN01 | raw | 0.992778 → 1.148228 | 1.299204 → 1.699096 | 12.805100 → 9.253410 |
| 16 / 1efdN01 | P32 | 0.982862 → 1.158661 | 1.245988 → 1.674778 | 11.879914 → 8.968103 |
| 64 / 3aq5A00 | raw | 0.794470 → 1.154227 | 1.097367 → 2.349189 | 38.912034 → 20.687704 |
| 64 / 3aq5A00 | P32 | 0.774119 → 1.137266 | 1.028159 → 2.284072 | 35.990271 → 20.758282 |

### h64的成员误差、分布与几何不能互相替代

h64的P32成员/未来帧CA-RMSD均值由2.816933降至2.761343 Å，但同一输出的两种路径U-score都变差。前者只汇总单个成员对参考的误差，不能替代评价整个系综的分布评分。该窗MD的shape路径长度为62.725661 Å，lag-1内部距离ACF为0.477522、MSD为0.800513 Å²；映射后路径更短、MSD更小，ACF反而更高，不能把更高ACF解释为更准确的时间行为。

| h64角色 | 成员/帧CA-RMSD (Å) | Lag-1 ACF | Lag-1距离MSD (Å²) | 无效CA键比例 | CA clash比例 | Rg (Å) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline raw | 3.083740 | 0.870864 | 0.398966 | 0.671820 | 0.000195 | 13.566521 |
| Baseline P32 | 2.816933 | 0.864991 | 0.344501 | 0.000000 | 0.000192 | 13.414165 |
| Mapped raw | 2.785811 | 0.892369 | 0.140525 | 0.568764 | 0.000127 | 11.522289 |
| Mapped P32 | 2.761343 | 0.891746 | 0.136279 | 0.000000 | 0.000207 | 11.601451 |

几何/Rg为全部未来成员/帧的等权描述均值，排除t0；canonical CA键与clash定义沿用前节，**不是PSF或完整backbone有效性**。h64映射降低raw无效键比例，但仍为56.88%；P32后该键检查为零，clash却略增，不能以单项几何改善宣称有效构象。h4/h16映射后P32无效键比例仍分别为0.035714/0.011111，全部有限异常保留。以上不是新的运动时间尺度、平衡态或扩散系数估计。

真实H control接近恒等：h1/4/16/64未来token的H变化范数均值分别为1.350031e−6、9.467244e−7、9.639094e−7、8.884573e−7；对应forecast为1.312749、1.436147、4.977927、10.314766。Forecast映射后的最大通道均值残差跨四窗不超过9.13e−9，中心化半径残差不超过2.13e−7，保留float32转换残差而不当作精确球面约束。近恒等真实H控制说明该具体映射对encoder输出改动很小，不能证明未映射预测H缺乏信息。

这次观察半径映射与旧TRAIN32的eps=1e−5 LayerNorm并不完全相同，但其负结果没有推翻旧结果：旧coordinate U-ES仍为**2.2033 → 2.5667 Å**。两次有限训练集诊断均不支持采用各自的离线归一化。它们也不能否定所有潜变量约束、识别范数漂移的单独因果效应，或推出模型在新蛋白/温度上的表现；未改变当前训练/推理配方。

### 独立CPU/NumPy审计

仅加载父实验与本次保存sidecar，独立以观察H0计算系数，在float64重算映射后转float32；forecast和真实H控制共8份映射、**13751296个H元素全部逐位匹配，零差异**。四窗H0均逐位不变；所有实际mask均全有效，因此padding不变检查在这些真实输入上是空集检查，不能当作非空padding回归证据。595008个原样重解码坐标元素与父实验逐位一致；固定metadata/anchor/time字节也一致。

重算全部8个mapped坐标角色：417148个映射detail数值及56个report值、850616个H描述张量元素及4224个曲线值、396672个位移张量值及12816个位移报告值、2306个anchor运动数值、2136个成员/帧的8544个几何标量，以及480个shape-path字段、510个ordered曲线值、114个ordered标量与6个null。另核对720条lag记录的3468个数值和540个null，包含正确的合法时间对支持；全部两种路径U-score也独立重算。映射H与位移张量精确一致，其余全体最大绝对差**4.62e−14**，没有超容差失败或删除null。

107个源码及107个快照、11项输入文件哈希、6份metadata快照、父/子sidecar文件与递归张量哈希、真实exit0记录均核对一致。保存RNG状态与其摘要在映射/读出前后及四窗之间一致；记录的decoder初末哈希、零optimizer/rollout也一致。独立审计未调用producer数值函数、加载模型/decoder、读取HDF或重新采样：baseline字节、映射与保存坐标指标是独立复核，decoder参数保持、无梯度、未构造kernel等运行期断言仍属于producer证据，不能宣称独立重新执行了这些操作。

## 解释勘误：Euclidean U-energy score的非负性（2026-11-02）

冻结的[路径评分实现](evaluation/trajectory_metrics_v2.py)第133行docstring及第291行conventions、[ordered-path设计文档](docs/HS_ORDERED_PATH_PILOT_V1_20261102.md)第40行附近所称“有限成员U-score可为负”，对这里采用的Euclidean距离U-energy score不成立。固定目标y、M≥2时，三角不等式给出`Σ(i<j) ‖xi−xj‖ ≤ (M−1) Σi ‖xi−y‖`，因此`UES = Σi ‖xi−y‖/M − Σ(i<j) ‖xi−xj‖/[M(M−1)] ≥ 0`。正的特征维数归一化（如除以√P）及对目标帧、时间或窗口取平均均保持非负性。微小负数可由浮点舍入产生，不应解释为合法的负分布评分。

这与[接触population-U Brier校正](docs/CONTACT_ENSEMBLE_BIAS_V1_20261102.md)不同：不同成员残差乘积均值的有限样本值确实可以为负。此勘误只更正两者的解释，未改动任何冻结源码、文档、评分或clamp行为，也不表示当前实际输出曾出现负Euclidean U-score。

## 物理距离路径目标：固定四窗kernel梯度诊断完成（2026-11-02）

**绝对距离与有符号相邻距离增量的辅助目标均有有限、可复核的完整kernel梯度，但尺度和方向不同；没有训练效果结论。** 两者固定权重λ=1，相同Å单位不代表相同梯度尺度。本次没有按四窗结果调整系数、选择目标、执行optimizer或改变推理；CPU运行也不证明GPU显存可行性。

[原语](protein_jepa_frame/training/physical_distance_path_objective_v1.py)35项、[单forward适配器](protein_jepa_frame/training/physical_distance_augmented_objective_v1.py)19项及[诊断运行器](run_scripts/diagnose_distance_path_kernel_gradients_v1.py)18项测试通过，源码与[科学协议说明](docs/DISTANCE_PATH_KERNEL_GRADIENT_DIAGNOSTIC_V1_20261102.md)完成审查。沿用旧训练前四个实际TRAIN批次，h1/4/16/64分别为3l41A01、3tpqM02、1efdN01、3aq5A00；四窗均使用原始spatial初始化，非逐步已学习状态，也不宣称回放旧GPU噪声。CPU两线程、连续seed42抽样流，每窗只作一次observed-only rollout并在同一计算图上求五种梯度：base、absolute auxiliary、increment auxiliary及两个base+auxiliary总损失。Base保留原marginal坐标监督、latent .1、plugin contact20、clash40及既有零权重项；没有根据尚在执行的接触试验改动base。

两种辅助项使用全部相同的有效physical nonlocal residue pairs、成员、目标与1000 ps网格，每帧联合距离向量按√pair-count归一化后等权平均时间。Absolute评分d(t)，increment评分有符号d(t)−d(t−1)，并对两个预测帧保留梯度；它们是不同变换后观察量的时间边际评分，不是整个路径联合分布或动力学验证。

[执行前协议](results/hs_joint_20261101/distance_path_kernel_gradients_v1/protocol.json)SHA-256 `692241dc1b6ce697a347f445c57863c6074a06b5be8da80b62fbd97d720f262f`。[实际外层运行](results/hs_joint_20261101/distance_path_kernel_gradients_run_v1.execution.json)于07:11:23–07:12:52 UTC完成，exit0、88.878秒，execution SHA-256 `ec13083bce19259a5e755c72a9060b44f4fd3241f9a2be1f14c2baa029a28459`；内部PID 2136733、86.770秒。[完整结果](results/hs_joint_20261101/distance_path_kernel_gradients_v1/result.json)SHA-256 `77304c19880c09f1c41b8cd3fc6d7670ba3e61443a7de3b8eeada99b060a1252`。四窗均完成，零optimizer、无参数`.grad`累积；冻结组件不参与参数求导，无val/test或BioEmu2重跑。

| Horizon | Base loss | Absolute辅助 | Increment辅助 | Base+absolute | Base+increment |
|---|---:|---:|---:|---:|---:|
| 1 | 3.367737 | 3.354624 | 3.354624 | 6.722361 | 6.722361 |
| 4 | 1.637346 | 1.168327 | 1.144751 | 2.805673 | 2.782097 |
| 16 | 1.561052 | 1.396223 | 0.911481 | 2.957275 | 2.472533 |
| 64 | 1.048129 | 0.953586 | 0.677491 | 2.001715 | 1.725620 |

下表A为absolute辅助、I为increment辅助、B为base；范数比为倍数，均为未裁剪的完整669429维参数梯度。四窗独立描述，不给跨horizon平均或CI。

| Horizon | ‖∇A‖/‖∇B‖ / ‖∇I‖/‖∇B‖ | cos(A,B) / cos(I,B) | cos(A,I) | cos(B,B+A) / cos(B,B+I) | cos(B+A,B+I) |
|---|---|---|---:|---|---:|
| 1 | 3.161353 / 3.161353 | 0.724969 / 0.724969 | 1.000000 | 0.834046 / 0.834046 | 1.000000 |
| 4 | 0.510583 / 0.274503 | 0.958615 / 0.979447 | 0.949215 | 0.995271 / 0.999049 | 0.995085 |
| 16 | 1.473511 / 0.583429 | 0.573784 / 0.089425 | 0.368644 | 0.836937 / 0.875373 | 0.835563 |
| 64 | 0.981982 / 0.149165 | 0.444671 / 0.310384 | 0.173487 | 0.852860 / 0.990941 | 0.848055 |

λ=1的absolute梯度在部分窗口与base同量级或更大；increment在h64小得多，在h16与base接近正交。h1固定观察初帧使两种目标数学等价；实际辅助/总loss值精确相等，但梯度仅数值相等：最大元素差分别2.98023e−8和4.47035e−8，不能把表中舍入的1.000000宣称为逐位梯度一致。

所有五种导数均保留完整73张量清单，实际None支持均为9个：8个history_posterior参数与`energy.energy_head.2.bias`。仅在审计向量中将None填零。h16另有已连接但全零的`energy.temperature_proj.0.weight`和`bath.temp_proj.0.weight`，故该窗62个非零张量，其他三窗64个；None、已连接和非零计数不混用。全参数/各module的反向比较也全部保留：四窗各参数六类比较的负余弦数为0/8/37/30；h64 bath模块increment/base余弦为−.009796，范数.121667/.759345。该窗`bath.temp_proj.0.bias`对应余弦−.995850，不代表整个模型方向。没有从这些局部导数推断AdamW、裁剪后更新、预测收益或物理时间尺度。

独立CPU/NumPy仅加载四份梯度sidecar，重算20个完整向量、292个逐参数统计块和12个module统计块：**7008个数值字段**全部匹配，最大绝对差1.79412e−13、最大相对差4.25682e−14；**507个null**完整保留（504个零范数比较null、3个非h1恒等检查null）。8个加法恒等式与h1的4个值/梯度恒等式均通过冻结的逐元素容差`2e−6 + 5e−4×abs(expected)`；最大元素残差1.82539e−7，最大相对L2残差2.96334e−7，没有放宽容差或删除不利字段。

另核验108项源码及108份快照、6项输入文件哈希、2份历史metadata快照、4个HDF大小/mtime、全部sidecar哈希、73个参数名称/顺序/形状/类型及历史批次metadata。20次求导后的保存RNG均等于各自post-forward状态，三个跨窗口显式采样generator连续性检查通过；记录的模型/组件初末哈希及零更新一致。独立审计没有导入producer数值函数、重跑模型/decoder、加载实际batch张量或读取HDF/test：保存梯度统计、恒等关系、RNG及字节哈希独立核对，实际None连接、计算图来源、batch内容指纹真实性和模型运行期不变仍依赖producer证据。Loss仅核对sidecar/JSON一致、加法及h1关系，未从轨迹独立重新求值；HDF stat也不是完整文件哈希。首次临时审计stdout的NumPy整数序列化失败已在只读审计中纠正并完整复跑，未修改实验工件，也不是实验运行失败。

## 接触估计器配对500步试验完成：不采用population-U替换（2026-11-02）

**本轮没有证明population-U校正比plugin控制具有可靠预测优势，保持plugin默认。** 预先固定的h16/64 joint-path coordinate U-ES为1.650007 / 1.665881 Å，U−P差−0.015874 Å、95%区间[−0.059656,+0.009153]跨0；七项历史指标的总体点估计均向不利方向变化，且有短期评分和几何代价。这不是等效性结论，也不是对有限成员校正代数或其他训练配方的普遍否定。下文U/P分别为本轮重新训练的population_u/plugin，两者均只评latest500。

### 实际执行、匹配契约与独立核验

两臂各train、ensemble200、endpoint200、ordered200、suite7200、matched8共六项任务，全部12项实际exit0；标准面板50蛋白×四horizon、每窗12成员，suite为12蛋白/300轨迹，matched8为8蛋白/200轨迹/800窗。正式协议与训练前统计协议沿用上文登记值；同一spatial初始化、500个实际加载batch指纹逐字节相同、最终全局与采样RNG相同。只有output/contact_estimator两个训练参数不同，H/CA decoder/P32与坐标marginal、latent .1、clash40等其余设置相同；73参数张量全清单保留，64个active optimizer状态各500步，9个未使用参数未衰减。首个h1共享项相同，但所选loss为3.240859 / 3.249248，差为20V；梯度范数1.419146 / 1.409888本来就允许不同，不能套用旧试验的整loss/梯度相等检查。

[实际主运行](results/hs_joint_20261101/contact_pilot_run_v1.execution.json)：05:52:04.395–07:28:35.973 UTC，5791.577秒，exit0；SHA-256 `ccc73ff1eb53daf1763e624d806704684661feeb2080b32366d1c584cdc2f1b6`。

[producer汇总](results/hs_joint_20261101/contact_summary_run_v1.execution.json)：07:29:16.775–07:29:28.413 UTC，11.633秒，exit0；SHA-256 `572b4dc3c974c0e69658cee9d95c0fe0e24ed40ba7172e6fd28ebe0738de0c02`。

[独立核验](results/hs_joint_20261101/contact_independent_run_v1.execution.json)：07:30:01.989–07:30:11.583 UTC，9.594秒，exit0；SHA-256 `65734270f5deda07a1baf678e5b72ef50f3e55531a5cc31f88183197dd16334b`。

[完整汇总](results/hs_joint_20261101/contact_estimator_v1/summary.json) SHA-256 `f4b147562647051cd508eca52968656598015316024d33fb17b92381953f39d0`。

[独立核验结果](results/hs_joint_20261101/contact_estimator_verification_v1/verification.json) SHA-256 `81da4afbcd0c008957be98b9864dd798549bd48edce1f640b5e34f4358812b0a`。

U/P latest500 checkpoint SHA-256分别为 `4554451b991cee4106df47b5f80d7f651318854b53a913d412396d509ced5504` / `8136edc5f02a2e5b5c11331f0865095c47be03ec954b0d4be292593aebc14db3`。汇总实现37项、独立核验46项合成测试及代码审查在实际读评分前完成；统计定义在训练前冻结，实现登记发生在训练中。实际独立NumPy核验重建**402个配对CI、670个standard均值、1700个suite均值、1500个ordered scalar支持记录、1200个lag支持记录**，比较196462个数值，最大绝对差2.27374e−13，全部通过，null和缺失支持保留。

独立核验读取保存的原始评分/训练日志和metadata，未调用producer聚合数值函数、未重新加载模型或坐标、未采样或打开test。它独立核验文件哈希、真实exit0、500行训练证据及统计重建；checkpoint内部tensor/optimizer/RNG相等仍由producer wrapper的加载检查提供，独立核验只重新验证checkpoint字节哈希，不冒称第二次张量级核验。没有重跑BioEmu2或修改任何冻结结果。

所有CI是seed20261101、10000次按蛋白配对bootstrap、蛋白内窗口等权后蛋白等权的95% percentile区间；未校正多重比较、训练seed或自适应开发选择。标准200窗的温度层含不同蛋白，不能当作同蛋白温度响应。

### 预声明主指标与完整horizon评分

| 面板 | U | P | U−P | 95% CI |
| --- | --- | --- | --- | --- |
| h16/64 joint-path coordinate U-ES（主指标） | 1.650007 | 1.665881 | -0.015874 | [-0.059656, 0.009153] |
| h16/64 endpoint coordinate U-ES（次要） | 2.008557 | 2.020983 | -0.012426 | [-0.056156, 0.016373] |

| Joint-path U-score ↓ | 类型 | U | P | Persistence | U−P 95% CI |
| --- | --- | --- | --- | --- | --- |
| overall | coordinate | 1.318819 | 1.326943 | 1.784040 | [-0.030225, 0.004542] |
| overall | distance | 1.686404 | 1.686449 | 2.332095 | [-0.015993, 0.011644] |
| h1 | coordinate | 0.943006 | 0.944179 | 1.271944 | [-0.002820, 0.000475] |
| h1 | distance | 1.185988 | 1.186193 | 1.625211 | [-0.004559, 0.004331] |
| h4 | coordinate | 1.032256 | 1.031830 | 1.426558 | [-0.002182, 0.003061] |
| h4 | distance | 1.349324 | 1.343077 | 1.871269 | [0.001312, 0.011856] |
| h16 | coordinate | 1.422583 | 1.425217 | 1.935563 | [-0.016965, 0.007842] |
| h16 | distance | 1.859191 | 1.856429 | 2.568151 | [-0.021244, 0.023280] |
| h64 | coordinate | 1.877431 | 1.906544 | 2.502094 | [-0.103081, 0.012761] |
| h64 | distance | 2.351113 | 2.360098 | 3.263751 | [-0.046961, 0.018569] |

两臂coordinate/distance joint-path分数在overall及四个horizon均优于persistence，配对差区间均低于0；这没有证明校正收益，因为plugin同样通过。U−P的h4 distance区间完全为正，是明确保留的不利次要结果。

| Endpoint ↓ | 类型 | U | P | Persistence | U−P 95% CI |
| --- | --- | --- | --- | --- | --- |
| overall | coordinate U-ES | 1.524493 | 1.530257 | 2.033240 | [-0.027516, 0.008395] |
| overall | distance V-ES | 2.114808 | 2.112942 | 2.690191 | [-0.022147, 0.023059] |
| overall | hard-contact Brier | 0.016540 | 0.016822 | 0.017379 | [-0.001153, 0.000208] |
| h1 | coordinate U-ES | 0.943006 | 0.944179 | 1.271944 | [-0.002820, 0.000475] |
| h1 | distance V-ES | 1.271777 | 1.270853 | 1.624493 | [-0.003536, 0.005443] |
| h1 | hard-contact Brier | 0.010854 | 0.010850 | 0.012319 | [-0.000026, 0.000033] |
| h4 | coordinate U-ES | 1.137854 | 1.134882 | 1.571875 | [-0.002085, 0.008007] |
| h4 | distance V-ES | 1.638982 | 1.623140 | 2.072172 | [0.005746, 0.027206] |
| h4 | hard-contact Brier | 0.014705 | 0.014393 | 0.014636 | [0.000210, 0.000424] |
| h16 | coordinate U-ES | 1.723541 | 1.718800 | 2.300956 | [-0.014806, 0.019810] |
| h16 | distance V-ES | 2.448921 | 2.418962 | 3.086986 | [-0.004229, 0.070114] |
| h16 | hard-contact Brier | 0.018908 | 0.018568 | 0.019220 | [-0.000033, 0.000923] |
| h64 | coordinate U-ES | 2.293573 | 2.323167 | 2.988185 | [-0.099011, 0.018864] |
| h64 | distance V-ES | 3.099552 | 3.138813 | 3.977112 | [-0.115303, 0.023907] |
| h64 | hard-contact Brier | 0.021691 | 0.023477 | 0.023343 | [-0.005173, 0.000010] |

Endpoint coordinate为off-diagonal U-statistic；endpoint distance仍是原有含对角V-statistic、最多512个固定物理非局部pair，不能与joint-path distance U-score当成同一估计器。Hard-contact Brier仍用12成员经验硬接触概率，没有随着训练改成population-U。h4 distance与Brier均恶化且区间不跨0；总体两臂的coordinate/distance均优于persistence，但Brier的对persistence区间跨0。不存在匹配时序的BioEmu2路径或endpoint评分比较。

### 历史七指标、canonical CA几何与时间行为

| 标准200窗总体 | U | P | 保存BioEmu2 | Persistence | U−P 95% CI |
| --- | --- | --- | --- | --- | --- |
| RMSF MAE ↓ | 1.332912 | 1.302179 | 1.991674 | 1.710638 | [-0.005874, 0.080618] |
| RMSF correlation ↑ | 0.508799 | 0.512944 | 0.579744 | 0.022824 | [-0.010306, 0.001557] |
| Root mean W2 ↓ | 1.648185 | 1.637023 | 2.290364 | 1.572276 | [-0.001862, 0.026016] |
| Variance W2 ↓ | 9.323220 | 9.042571 | 18.575412 | 10.042719 | [-0.077703, 0.843352] |
| MD-PCA W2 ↓ | 6.610454 | 6.559375 | 6.114255 | 9.129378 | [-0.029916, 0.165640] |
| Weak-contact J ↑ | 0.101107 | 0.101267 | 0.131606 | 0.250000 | [-0.001612, 0.001304] |
| Transient-contact J ↑ | 0.234063 | 0.234348 | 0.265260 | 0.000000 | [-0.002500, 0.001844] |

七项U−P总体区间全部跨0，点估计全部较差。对保存BioEmu2，两臂均为MAE/Root/Variance较好、correlation/weak/transient较差，相应配对区间均不跨0；PCA不确定。Persistence的weak-contact总体.25受h1空真值/空预测集合约定影响，不能称为动态接触恢复。分层也不是单向结果：h1的MAE/correlation/Root/Variance/transient及h4的MAE/Root/Variance/weak/transient恶化区间不跨0，而h1/h4 PCA改善；h16 correlation恶化，h64 transient J改善差+0.008694 [0.003634,0.013674]。全部温度/horizon记录与计数均在完整汇总，不能择取h64接触改善升级配方。

| Canonical CA总体 | U | P | U−P 95% CI |
| --- | --- | --- | --- |
| 键长均值 Å | 3.840982 | 3.840610 | [-0.000327, 0.001556] |
| 无效键比例 ↓ | 0.004105 | 0.003713 | [-0.000129, 0.001299] |
| Clash比例 ↓ | 0.000583 | 0.000567 | [-0.000027, 0.000066] |
| Rg Å | 14.971267 | 14.991278 | [-0.055318, 0.004265] |

均值按蛋白等权，不是pooled fraction。键区间3.3–4.3 Å、非局部clash阈值2.5 Å；这不是PSF或完整backbone有效性。总体几何差区间跨0，但h4和h16的clash差分别+0.00003586 [0.00001944,0.00005405]、+0.00011195 [0.00000693,0.00028803]，保留不利结果；Rg/平均键长本身没有单向优劣。

| h64 ordered描述量 | U | P | MD |
| --- | --- | --- | --- |
| shape路径长度 Å | 65.515056 | 67.676822 | 130.226965 |
| 成员/帧RMSD Å | 5.485855 | 5.570830 | 0.000000 |
| 距离lag-1 ACF | 0.828891 | 0.829851 | 0.521274 |
| 距离lag-1 MSD Å² | 4.422786 | 5.309691 | 4.532827 |
| ACF曲线correlation | 0.937576 | 0.934523 | 1.000000 |

路径长度仍约为MD一半，lag-1 ACF明显过高；U的MSD均值更接近MD不能消除其他偏差。两臂h64成员/帧RMSD均劣于persistence 4.198511 Å，而系综U-score较好，成员误差与分布评分不能互换。

| h64描述性时间对照 | U coord | U distance | U ACF1 | P coord | P distance | P ACF1 |
| --- | --- | --- | --- | --- | --- | --- |
| forecast | 1.877431 | 2.351113 | 0.828891 | 1.906544 | 2.360098 | 0.829851 |
| reversed_forecast | 1.884157 | 2.369061 | 0.812398 | 1.916655 | 2.386677 | 0.812507 |
| shuffled_forecast | 1.880908 | 2.359427 | -0.021996 | 1.912022 | 2.373572 | -0.020100 |

Reverse/shuffle是固定输出排列，改变时间边际/可能改变endpoint，非另训模型，也不单独证明物理记忆；shuffle后ACF接近零但评分变化较小。h1 walk-RMSF及ACF曲线correlation无支持，完整面板保持null；persistence的零方差ACF、tortuosity同样保持null，单路径MD不计算需要M≥2的U-score。

### Full12温度干预与matched8覆盖/布居

| Full12 | 臂 | MAE ↓ | Corr ↑ | Root ↓ | Variance ↓ | PCA ↓ | Weak ↑ | Transient ↑ | 成员RMSD ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| correct | U | 1.411735 | 0.453345 | 2.095609 | 14.306159 | 9.791183 | 0.111445 | 0.250467 | 5.741730 |
| correct | P | 1.437827 | 0.458087 | 2.101092 | 14.599152 | 9.841614 | 0.112657 | 0.252240 | 5.726117 |
| wrong | U | 1.587572 | 0.459503 | 2.147148 | 16.202574 | 10.343086 | 0.112870 | 0.253880 | 5.638374 |
| wrong | P | 1.600631 | 0.464067 | 2.149090 | 16.280776 | 10.330055 | 0.114799 | 0.256875 | 5.637855 |
| none | U | 1.378657 | 0.467323 | 2.059841 | 14.659292 | 10.187626 | 0.117380 | 0.262182 | 5.534002 |
| none | P | 1.410898 | 0.471996 | 2.068823 | 14.993137 | 10.222543 | 0.119338 | 0.265553 | 5.531505 |

各mode为2400窗、12蛋白/300轨迹的描述均值，不增加配对CI。U的correct对P改善MAE/Root/Variance/PCA，却降低correlation/两类contact并略增成员RMSD；对wrong/none也非所有指标都改善。两臂correct成员RMSD均高于persistence 4.930655 Å，不能用涨落评分覆盖位置误差。Wrong把320 K改450 K、其他改320 K，none用348 K；348 K的correct/none实际输出相同，不计为独立支持。五温度均在训练中出现，不是未见温度泛化。

| K | Full12 coverage U/P ↑ | Matched8 coverage U/P ↑ | Matched8 TV U/P ↓ | Matched8 F-MAE U/P ↓ | 历史Bio coverage | 历史Bio TV |
| --- | --- | --- | --- | --- | --- | --- |
| 320 | 0.491667 / 0.491667 | 0.409375 / 0.409375 | 0.809103 / 0.809401 | 0.997164 / 1.009951 | 0.368750 | 0.626386 |
| 348 | 0.393750 / 0.393750 | 0.259375 / 0.265625 | 0.761050 / 0.753768 | 1.010738 / 1.008638 | 0.246875 | 0.588990 |
| 379 | 0.225000 / 0.227083 | 0.215625 / 0.218750 | 0.731673 / 0.730873 | 1.084515 / 1.094778 | 0.178125 | 0.622056 |
| 413 | 0.106250 / 0.106250 | 0.090625 / 0.090625 | 0.633368 / 0.636216 | 0.990114 / 0.990519 | 0.081250 | 0.544678 |
| 450 | 0.016667 / 0.016667 | 0.015625 / 0.015625 | 0.599965 / 0.603080 | 1.025278 / 1.025555 | 0.015625 | 0.568909 |

Full12每轨迹96成员、matched8每轨迹48成员，支持集不同。U/P matched8的TV与JSD五温度描述均值都高于保存BioEmu2；F-MAE仅379 K方向较好。Coverage相近不能证明布居正确；k=8时population MAE=TV/4，不算两份证据。历史Bio虽800窗键/200轨迹及声明预算相配，仍缺mask/reference/状态中心的不可变指纹，严格配对T2A差保持null；上表历史均值仅作描述并列，无新Bio优越性声明。

本轮只检验固定训练配方中的接触均值估计器，未证明S记忆、冻结H信息上限、backbone/PSF修复、稳定GLE动力学或新蛋白/新温度泛化；原H读出gate长期失败保持不变。已经登记的distance-path下一试验仍用**plugin、λ=1**，该选择与科学统计协议在读取本轮接触性能前已固定，不能事后称为据此挑选的最佳配置。

## 距离路径辅助目标配对试验已登记并启动（2026-11-02）

**更新：两臂在step30均失败，未产生latest500或预测评测；下述启动记录作为历史保留。** [设计说明](docs/HS_DISTANCE_PATH_PILOT_V1_20261102.md)固定比较absolute物理非局部距离向量U-ES与相邻有符号distance-increment向量U-ES；两臂辅助权重均λ=1，base均保留marginal坐标监督与plugin contact。共享pairs、路径、目标、1000 ps时间网格和√pair-count归一化；单位相同不等于梯度尺度相同。目标与统计定义在读取已完成接触试验性能之前固定，没有根据接触结果改臂、调系数或挑选窗口。增量评分约束变换后的时间边际，不单独识别完整路径分布、chirality或物理动力学。

两臂从同一spatial初始化重新训练500步，seed42、12成员、同一旧500窗选择规则、H/CA decoder/P32冻结、全部73参数交给AdamW；不作训练期validation、best或early stopping，预定只评latest500。协议要求实际batch指纹与最终采样/全局RNG配对一致；目前不能把最终审计要求写成已完成事实。保持每臂train、ensemble、endpoint、ordered、suite、matched8共12逻辑任务及原评测预算，主指标为increment−absolute的h16/64 joint-path coordinate U-ES。

| 冻结登记 | 协议SHA-256 | 登记UTC |
|---|---|---|
| [科学统计](results/hs_joint_20261101/distance_path_v1/summary.protocol.json) | `b70776567f2423398e0485470d8e74f8a651badcb374e244df586b77b339f181` | 2026-11-02T07:24:03.431586+00:00 |
| [GPU smoke](results/hs_joint_20261101/distance_path_v1/smoke_v1/protocol.json) | `ce13638475974c6fd1cf65b12e50d99d1e5c553f2183583efce90e354028a8aa` | 2026-11-02T07:30:04.864449+00:00 |
| [正式试验](results/hs_joint_20261101/distance_path_v1/protocol.json) | `d061862fe61eeee4d861b53137a22f064f5ad3bc0edab7086cd5fb23c8be879b` | 2026-11-02T07:33:36.556909+00:00 |
| [汇总实现](results/hs_joint_20261101/distance_path_summary_v1/protocol.json) | `3c17a0f5c7a9ce0ba88df1ec38cf670a4e56ffb7f7bebfe35d92c22c7466fffd` | 2026-11-02T07:34:38.004469+00:00 |
| [独立核验](results/hs_joint_20261101/distance_path_verification_v1/protocol.json) | `44f5a34911efeb35653eb33872e296545f6ebf29437adcf75c166130b82d459e` | 2026-11-02T07:37:25.208545+00:00 |

固定TRAIN step4/h64/L96/M12的[GPU smoke实际记录](results/hs_joint_20261101/distance_path_v1/smoke_v1/execution.json)于07:30:22.808 UTC实际exit0（外层完成约07:30:23），内部17.774秒、零optimizer；absolute/increment峰值分配分别5623429632 / 5632915968 bytes。两臂73参数/64 connected gradients、冻结组件无梯度、模型哈希不变，post-forward/post-backward采样与全局RNG一致且两臂证据相配。注册额外核验预算、固定batch metadata/stat、梯度支持及RNG记录；这一检查只支持该固定GPU步骤的可执行性，不是训练效果。

训练器29项、运行器41项、统计12项、汇总44项及独立核验63项测试均通过并经复核；科学统计与实现协议均在本轮训练前冻结，尚未执行本轮性能汇总或实际独立统计核验。固定402 CI、670 standard均值、1700 suite均值以及1500 scalar/1200 lag支持记录仍全部预定保留。

[Increment实际执行记录](results/hs_joint_20261101/distance_path_v1/increment.execution.json)和[absolute实际执行记录](results/hs_joint_20261101/distance_path_v1/absolute.execution.json)分别于07:37:50.597 / 07:37:50.621 UTC启动训练子进程，PID 2165965 / 2165974，GPU0 / GPU1；runner PID 2165767。07:39:01 UTC只读核验时两个PID均存活、执行记录均为running，两臂latest500 checkpoint均尚未存在。此状态说明实际启动，不证明500步、最终batch/RNG匹配或12项评测完成；本次状态更新未读取新distance性能评分、模型或训练数据。

### 距离试验v1实际失败更新（2026-11-02，07:39 UTC）

两臂都在固定第30个TRAIN窗口抛出`ValueError: duplicate physical residue identity`，纯损失的唯一性检查拒绝重复的`(chain_id,residue_index)`；每臂仅有29行已完成更新记录、30行已选窗口记录。该窗为1ruyH02、1ruyH02_T413_R4、h4，context296–299、future300–303，双方实际batch SHA-256均为 `9fe2bcc32f7380d9d2818ce2326f43cb8a896c4e7a32f9f2c07cd57570fa22aa`。失败发生在第30次更新前，不能写成完成500步。

Absolute训练子进程于07:39:14.033 UTC实际exit1（83.413秒），increment于07:39:14.173 UTC实际exit1（83.576秒）；两份arm execution均为failed且只含train任务，latest500均不存在，没有开启后续评测或生成性能结果。此前单个h64 smoke通过并未覆盖这个残基身份情况。v1源码、协议、日志和失败全部保留，不就地放宽唯一性检查、不跳过固定窗口、不续跑或任意重编号；先只读核对该TRAIN窗口的PDB/HDF非坐标拓扑metadata与loader身份来源，再决定是否另立物理定义明确的v2。

## 距离辅助目标v2全窗口拓扑预审计完成（2026-11-02）

v1失败窗口包含合法的49/49A及133/133A插入码残基，旧loader保留不同原子行，却将author residue number作为embedding residue_index导出。因此重复embedding ID不能直接当作物理身份冲突。新[v2设计](docs/HS_DISTANCE_PATH_PILOT_V2_20261102.md)将完整source/domain/segment/author-chain/number/insertion-code UID与旧模型ID分开；坐标、mask、aatype、residue_index、chain_id及旧batch哈希均保持原样。仅新增辅助损失的pair集合使用完整PSF无向肽键图距离大于2（包括不连通pair），在裁剪或mask之前计算完整图的一/二跳排除，再限制pair端点。geometry_row_ids只标识身份，不计算数值序号差。

审阅还用合成PDB/PSF复现了冻结parser对真正重复完整UID的静默合并。新helper先对同一份原始文本检查UID块、原子和altloc唯一性，再调用冻结parser验证完整PDB/PSF对应和legacy原子行；完整重复、非连续重复UID、歧义blank/A骨架原子均失败。合法A/B altloc及不同插入码保留。55项metadata测试经独立复核通过，冻结parser和v1源码没有修改。

[元数据协议](results/hs_joint_20261101/distance_pair_support_v2/protocol.json)在实际扫描前注册；[实际执行](results/hs_joint_20261101/distance_pair_support_v2/execution.json)PID2209245于08:17:00.267 UTC启动，08:22:25.921 UTC完成exit0，内部325.594秒（外层333.439秒）。全部471个唯一结构记录成功，覆盖原定500个TRAIN窗口，errors为空；没有读取轨迹坐标、模型、validation或test，也未替换窗口。固定十份源码/测试/说明、四份输入、所有HDF size/mtime及源码快照均复验。随后根侧实际调用load_support，再验完整471图、500窗口、输入/源码/快照及started/execution绑定。

| 产物 | SHA-256 |
|---|---|
| metadata protocol | `68343665e5f6d1f601271ef909187f3d7d16c26fb67112b33a2606c8eb79a0b0` |
| [完整support](results/hs_joint_20261101/distance_pair_support_v2/support.json) | `947edbc5e6a1868d20c774e43e1fc827194e667dffb5a0a773b2683366bb946e` |
| metadata execution | `cccb08204481bbd838c1fe978a893e4b2710cc610a014088210b47885df681a4` |
| [v2统计协议](results/hs_joint_20261101/distance_path_v2/summary.protocol.json) | `ed8a738fc71002accac863879645b9f45054c03365ee41441e0effddfa9db4b5` |

在所选前缀中仅一个结构存在重复embedding IDs；六个结构的PSF配对集合与v1 author-number定义不同，合计增加6对、移除14对。其余465个结构集合相同。每个记录的候选pair数介于1128至4371，之后仍按固定observed mask限定端点；以下数量是mask前拓扑候选数。

| 结构 | v1 author-number pairs | v2 PSF pairs | 新增 | 移除 |
|---|---:|---:|---:|---:|
| 1ruyH02 | 4365 | 4371 | 6 | 0 |
| 2nogA01 | 2280 | 2278 | 0 | 2 |
| 4u2vA02 | 3006 | 3003 | 0 | 3 |
| 3f5hA00 | 1329 | 1326 | 0 | 3 |
| 1o7dC02 | 3744 | 3741 | 0 | 3 |
| 2rf4E02 | 3243 | 3240 | 0 | 3 |

v2是另行定义的辅助目标，不是放宽v1 guard或恢复29步。两臂必须从原始spatial初始化重新训练500步，保持λ=1、plugin contact及全部正式评估定义。核心/adapter57项、trainer30项、runner56项、统计12项、summary62项测试通过并复核；独立NumPy verifier先通过97项完整测试，补started/execution核验后新增metadata相关63项再次通过根侧独立复测。数值定义仍为402 CI、670 standard均值和1700 suite均值。本段为08:22时的metadata完成记录，后续实际执行见下节；当时尚无v2预测性能结论。完整PSF身份也未替换冻结encoder/decoder/P32或base损失中的旧几何约定，不能据此声称全模型已修正插入码几何。

### v2 smoke与500步训练完成，评测进行中（2026-11-02，09:24 UTC）

新的[GPU smoke](results/hs_joint_20261101/distance_path_v2/smoke_v2/execution.json)实际exit0，训练器于08:26:11.483–08:26:46.651 UTC执行固定step4/h64/L96/M12，零optimizer更新。absolute/increment最大分配显存分别为5623289856/5632776192 bytes，梯度范数为7.631734/5.476785；73参数中64个有连接梯度，冻结组件无梯度。模型前后哈希一致，两臂forward后的采样/全局RNG匹配，backward没有改变它们。根侧另调用冻结smoke审计，重验完整support及实际四字段物理配对身份。它只证明固定步骤可以执行。

| 冻结登记 | 协议SHA-256 | 登记UTC |
|---|---|---|
| [GPU smoke](results/hs_joint_20261101/distance_path_v2/smoke_v2/protocol.json) | `cc86d316413ccdfa9c75fb453aee8b11ff9969bcccb830c94382bdc3f0830785` | 08:26:09.580 |
| [正式试验](results/hs_joint_20261101/distance_path_v2/protocol.json) | `f5119c4a2f3adf3482a5a1bbb9f6f6b23b59ebfedbbc663d478f340f86baaa10` | 08:32:12.938 |
| [汇总实现](results/hs_joint_20261101/distance_path_summary_v2/protocol.json) | `a56d9d6f2f751cae4517e597f9199dfdb5a3100f0d8be8cb0e50ec6346b00cca` | 08:35:16.981 |
| [独立核验](results/hs_joint_20261101/distance_path_verification_v2/protocol.json) | `3c11d32c807d64944a28867ddc3d05947d77278b3b84d61c8933f6773df0f812` | 08:36:41.305 |

[Increment训练](artifacts/hs_joint_20261101/distance_path_v2/increment/execution.json)PID2230227于08:34:27.002–08:55:35.105 UTC完成，[absolute训练](artifacts/hs_joint_20261101/distance_path_v2/absolute/execution.json)PID2230228于08:34:27.002–08:55:35.052完成；两组均500次更新、保存latest500、子进程exit0，内部约1268秒。第30窗不再因合法插入码失败；其物理pair支持为4371对，两臂均有有限梯度。

截至09:24 UTC，两组ensemble、endpoint、ordered评测均已exit0，温度条件suite正在执行，matched8尚未开始。全部12逻辑任务完成、最终配对batch/RNG审计、注册汇总和独立统计核验仍须分别通过；当前没有读取新性能评分或作优越性结论。v1的29步失败及其全部产物继续保留。

随后进行的只读训练审计调用冻结runner的checkpoint/window/paired审计并再次验证输入、源码、快照和HDF stats，全部通过，未打开评测评分。两份500行training_windows逐字节相同，SHA-256为 `7faf31c66ced36bae97e2a80cb03bba7df2c94e0ff1aff832cdaca1b3bef38f1`；每行物理配对身份与完整support一致。每步均73参数/64有效梯度，64个活动Adam状态均更新500次，9个未使用参数及冻结组件不变。初始模型相同，最终Python/NumPy/Torch/CUDA全局RNG树及采样RNG完全一致；后者SHA为 `8dcbff5254c846ad180b65de2f84c4bebe65a80653af22b708d21ee447b644be`。最终checkpoint SHA为increment `7090bc1c9ab750b077cbc53267712ce99191fec4135126dfe7378047cf8b5b8f`、absolute `f586f6ee061fe34f213cd152e4460d036c11eacff0cc936ffb4b8b56f1d23dd9`，均与实际训练记录绑定。这完成训练配对核验；完整评测与最终runner记录仍待结束。

## 距离辅助目标v2完整结果：increment主指标较差（2026-11-02）

**已完成全部12项任务、注册汇总及独立核验。固定λ=1的signed-increment辅助目标没有优于absolute目标：长期joint-path主指标及endpoint次指标均更差。** 本节替代上节09:24时的进行中状态，保留v1双臂29步失败、v2拓扑定义变更及全部执行历史。I=increment，A=absolute；两臂仍保留plugin contact、同一spatial初始化、500次更新、固定500窗及原冻结H/CA/P32。最终batch/物理支持/采样与全局RNG配对审计通过，不代表训练后模型或Adam矩相同。

[完整汇总](results/hs_joint_20261101/distance_path_v2/summary.json) SHA-256为 `2d88b8a71180dd6519e35bbe3e82bbd1da07fba2a3209e8dc9fa3a204e9ce12f`；[独立NumPy核验](results/hs_joint_20261101/distance_path_verification_v2/verification.json)为 `a64e13ed775eeab0ecb0db20297363942a6b4bbd0060ade270ac92cbe552e7b1`。后者复算402个配对区间、670个standard均值、1700个suite均值，以及1500个ordered scalar和1200个lag支持记录；196548次数值比较最大差 `1.1368683772161603e-13`。核验不重新采样或再次反序列化模型tensor；内部tensor/optimizer/RNG结论来自冻结汇总器审计并绑定checkpoint字节。未打开test、未重跑BioEmu2。

| 臂 | 任务 | 实际PID | 开始UTC | 完成UTC | exit |
| --- | --- | --- | --- | --- | --- |
| increment | train | 2230227 | 08:34:24.731 | 08:55:36.424 | 0 |
| increment | ensemble | 2257762 | 08:56:25.118 | 08:59:04.254 | 0 |
| increment | endpoint | 2261155 | 08:59:05.032 | 09:01:35.514 | 0 |
| increment | ordered | 2265928 | 09:01:36.285 | 09:03:41.244 | 0 |
| increment | suite | 2266262 | 09:03:42.297 | 10:45:21.325 | 0 |
| increment | matched8 | 2409378 | 10:45:24.007 | 10:58:24.401 | 0 |
| absolute | train | 2230228 | 08:34:24.731 | 08:55:36.571 | 0 |
| absolute | ensemble | 2257699 | 08:56:24.257 | 08:59:04.169 | 0 |
| absolute | endpoint | 2261151 | 08:59:04.977 | 09:01:33.263 | 0 |
| absolute | ordered | 2265909 | 09:01:34.082 | 09:03:39.049 | 0 |
| absolute | suite | 2266247 | 09:03:40.146 | 10:45:21.426 | 0 |
| absolute | matched8 | 2409377 | 10:45:23.997 | 10:58:32.345 | 0 |

以上是runner记录的子进程时间，区别于此前训练器内部时间。注册[汇总执行](results/hs_joint_20261101/distance_path_summary_v2/execution.json)PID2440944于11:03:02.997–11:04:02.207 UTC完成exit0（外层完成11:04:08.306）；[独立核验执行](results/hs_joint_20261101/distance_path_verification_v2/execution.json)PID2446775于11:05:27.727完成exit0、耗时22.499秒（外层完成11:05:27.751）。冻结登记哈希见上节，不作事后重新登记。

### 主指标与解释边界

| 预定指标 | I | A | I−A [95% CI] | 窗 | 蛋白 |
| --- | --- | --- | --- | --- | --- |
| h16/64 joint-path coordinate U-ES Å ↓ | 1.677629 | 1.649875 | +0.027754 [+0.001592, +0.072649] | 100 | 50 |
| h16/64 endpoint coordinate U-ES Å ↓ | 2.041973 | 2.011339 | +0.030635 [+0.007865, +0.060597] | 100 | 50 |

CI按蛋白配对bootstrap 10000次、seed20261101：先蛋白内等窗平均，再蛋白等权；主指标覆盖h16/64共100窗、50蛋白。两个区间均高于零，方向均不利于I。全部CI为未校正多重比较、训练seed或自适应开发选择的边际探索区间；不是跨seed复现，也不能把跨零区间写成等效。standard200每h为同50蛋白、50窗；每温度为不同的10蛋白、40窗，不能当成蛋白内温度因果响应。下表数值统一保留六位小数，显示为0.000000不必然是精确零；完整精度和支持数以链接JSON为准。

这里只比较absolute与signed-increment两种辅助目标，没有同期无辅助臂，因而**不能归因absolute相对无辅助训练的收益，也不据此采用新配方**。相同Å单位/λ不等于相同梯度尺度。PSF图只改变新增辅助的pairs；冻结base、encoder、decoder与P32仍保留旧约定。时间边际距离及有符号距离增量均不单独识别完整路径分布、取向、chirality、平衡或动力学。

### Standard200全部七项与保存BioEmu2对照

| 指标 | I | A | BioEmu2 | I−A [95% CI] | I−Bio [95% CI] | A−Bio [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| RMSF MAE Å ↓ | 1.350026 | 1.381884 | 1.991674 | -0.031858 [-0.056577, -0.008888] | -0.641648 [-0.952288, -0.345328] | -0.609790 [-0.923879, -0.308888] |
| RMSF corr ↑ | 0.509777 | 0.510046 | 0.579744 | -0.000269 [-0.003602, +0.003016] | -0.069967 [-0.118594, -0.021902] | -0.069698 [-0.117635, -0.021704] |
| Root-W2 Å ↓ | 1.663875 | 1.673132 | 2.290364 | -0.009257 [-0.020939, +0.002223] | -0.626490 [-0.845765, -0.420970] | -0.617232 [-0.837260, -0.411077] |
| Variance-W2 Å² ↓ | 9.371996 | 9.616114 | 18.575412 | -0.244118 [-0.482164, -0.013245] | -9.203416 [-12.697688, -5.877137] | -8.959298 [-12.443887, -5.645053] |
| MD-PCA-W2 Å ↓ | 6.527175 | 6.468704 | 6.114255 | +0.058471 [+0.012423, +0.111851] | +0.412920 [-0.298841, +1.233172] | +0.354449 [-0.353471, +1.177084] |
| Weak-contact J ↑ | 0.099871 | 0.096443 | 0.131606 | +0.003428 [+0.001723, +0.005082] | -0.031735 [-0.042798, -0.020679] | -0.035163 [-0.046458, -0.023896] |
| Transient-contact J ↑ | 0.232035 | 0.228067 | 0.265260 | +0.003968 [+0.002019, +0.005893] | -0.033224 [-0.050790, -0.015635] | -0.037192 [-0.055212, -0.019007] |

I对A改善RMSF MAE、Variance-W2及两类contact，MD-PCA-W2更差，correlation/Root-W2区间跨零。两臂对保存BioEmu2均改善MAE/Root/Variance，却降低correlation及两类contact；PCA差区间跨零，不能写为全面胜出。此处是冻结future-path mixture（含anchor）的empirical-W2/PCA投影口径，不是matched-time endpoint或joint-path Bio比较，也不是新增Bio运行。完整十个分组和所有区间保留在汇总中。

### Endpoint与连续ordered-path

| 分组 | endpoint指标 ↓ | I | A | persistence | I−A [95% CI] |
| --- | --- | --- | --- | --- | --- |
| overall | coordinate U-ES Å | 1.540523 | 1.524408 | 2.033240 | +0.016115 [+0.004579, +0.031027] |
| overall | distance V-ES | 2.131046 | 2.107741 | 2.690191 | +0.023306 [+0.000272, +0.050080] |
| overall | contact Brier | 0.017232 | 0.017112 | 0.017379 | +0.000120 [-0.000245, +0.000767] |
| h1 | coordinate U-ES Å | 0.942533 | 0.942724 | 1.271944 | -0.000191 [-0.001086, +0.000756] |
| h1 | distance V-ES | 1.273204 | 1.273178 | 1.624493 | +0.000026 [-0.002315, +0.002809] |
| h1 | contact Brier | 0.010891 | 0.010872 | 0.012319 | +0.000019 [-0.000010, +0.000048] |
| h4 | coordinate U-ES Å | 1.135611 | 1.132230 | 1.571875 | +0.003381 [-0.001304, +0.008135] |
| h4 | distance V-ES | 1.638369 | 1.623808 | 2.072172 | +0.014562 [+0.003786, +0.025898] |
| h4 | contact Brier | 0.014724 | 0.014573 | 0.014636 | +0.000151 [+0.000012, +0.000299] |
| h16 | coordinate U-ES Å | 1.736540 | 1.715146 | 2.300956 | +0.021394 [+0.002768, +0.044384] |
| h16 | distance V-ES | 2.456791 | 2.425603 | 3.086986 | +0.031188 [-0.006175, +0.077746] |
| h16 | contact Brier | 0.019081 | 0.019222 | 0.019220 | -0.000141 [-0.000379, +0.000143] |
| h64 | coordinate U-ES Å | 2.347407 | 2.307531 | 2.988185 | +0.039876 [+0.007793, +0.081252] |
| h64 | distance V-ES | 3.155820 | 3.108374 | 3.977112 | +0.047447 [-0.018306, +0.119432] |
| h64 | contact Brier | 0.024230 | 0.023779 | 0.023343 | +0.000451 [-0.000927, +0.002943] |

Endpoint coordinate为off-diagonal U估计、按√(3N)归一化；distance为含对角项的V估计、最多512个确定性pair、距离除以1Å后按√P归一化；contact为有限M=12的经验Brier。只有真实endpoint拟合到observed anchor，预测不另行拟合。不能把distance V-ES称为U，亦不能把新增训练PSF全集pair定义写成冻结评估的pair定义。I的overall coordinate/distance endpoint均更差；Brier差区间跨零。

| 分组 | ordered指标 ↓ | I | A | persistence | I−A [95% CI] |
| --- | --- | --- | --- | --- | --- |
| overall | joint coordinate U-ES Å | 1.332281 | 1.318494 | 1.784040 | +0.013787 [+0.000577, +0.036534] |
| overall | joint distance U-ES Å | 1.694711 | 1.678679 | 2.332095 | +0.016032 [+0.001761, +0.034187] |
| h1 | joint coordinate U-ES Å | 0.942533 | 0.942724 | 1.271944 | -0.000191 [-0.001086, +0.000756] |
| h1 | joint distance U-ES Å | 1.186477 | 1.186540 | 1.625211 | -0.000064 [-0.002431, +0.002765] |
| h4 | joint coordinate U-ES Å | 1.031336 | 1.031504 | 1.426558 | -0.000168 [-0.002670, +0.002320] |
| h4 | joint distance U-ES Å | 1.348499 | 1.344475 | 1.871269 | +0.004024 [-0.001105, +0.009342] |
| h16 | joint coordinate U-ES Å | 1.433610 | 1.420611 | 1.935563 | +0.012999 [-0.000650, +0.032389] |
| h16 | joint distance U-ES Å | 1.871925 | 1.849764 | 2.568151 | +0.022161 [-0.001011, +0.052108] |
| h64 | joint coordinate U-ES Å | 1.921647 | 1.879139 | 2.502094 | +0.042508 [+0.002427, +0.112515] |
| h64 | joint distance U-ES Å | 2.371945 | 2.333937 | 3.263751 | +0.038008 [+0.003435, +0.082636] |

连续路径使用单次rollout完整未来帧，原始observed anchor保留；每窗路径vector的U-score按冻结时间/坐标或pair数归一化。I的h64 coordinate和distance均更差；h1/h4差区间跨零，h16也跨零，不能据此声称短期等效。以下描述量不增加CI或单调优劣方向，不能用某一更接近MD的lag替代预定主指标。

| h | 角色 | 成员RMSD Å | 路径长 Å | 端到端 Å | 曲折度 | walk-RMSF MAE Å | walk corr | distance ACF1 | distance MSD1 Å² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| h1 | I | 2.738292 | 1.344009 | 1.344009 | 1.000000 | null | null | -1.000000 | 2.666725 |
| h1 | A | 2.738849 | 1.347484 | 1.347484 | 1.000000 | null | null | -1.000000 | 2.662100 |
| h1 | MD | 0.000000 | 1.807536 | 1.807536 | 1.000000 | null | null | -1.000000 | 3.866796 |
| h1 | persistence | 2.203072 | 0.000000 | 0.000000 | null | null | null | null | 0.000000 |
| h4 | I | 3.327712 | 5.523101 | 2.323576 | 2.438759 | 1.068624 | 0.522389 | 0.084469 | 2.958827 |
| h4 | A | 3.304217 | 5.457987 | 2.275086 | 2.486938 | 1.041007 | 0.523406 | 0.082881 | 2.843796 |
| h4 | MD | 0.000000 | 6.974818 | 2.197672 | 3.374186 | 0.000000 | 1.000000 | -0.092812 | 3.345412 |
| h4 | persistence | 2.443948 | 0.000000 | 0.000000 | null | 1.204901 | null | null | 0.000000 |
| h16 | I | 4.468440 | 20.377256 | 2.984179 | 7.136702 | 1.471837 | 0.571735 | 0.593476 | 3.848452 |
| h16 | A | 4.439739 | 19.233393 | 3.057969 | 6.683450 | 1.459477 | 0.573439 | 0.615785 | 2.657137 |
| h16 | MD | 0.000000 | 30.179146 | 3.376116 | 11.103446 | 0.000000 | 1.000000 | 0.311995 | 4.388271 |
| h16 | persistence | 3.248404 | 0.000000 | 0.000000 | null | 1.968192 | null | null | 0.000000 |
| h64 | I | 5.658276 | 70.607886 | 3.280238 | 20.629166 | 1.685283 | 0.562041 | 0.814839 | 5.881634 |
| h64 | A | 5.660391 | 70.545587 | 3.573578 | 19.084515 | 1.780314 | 0.564805 | 0.829682 | 5.077737 |
| h64 | MD | 0.000000 | 130.226965 | 4.454419 | 37.124346 | 0.000000 | 1.000000 | 0.521274 | 4.532827 |
| h64 | persistence | 4.198511 | 0.000000 | 0.000000 | null | 2.650001 | null | null | 0.000000 |

h64路径长I/A=70.607886/70.545587 Å，仍明显短于MD130.226965 Å；ACF1=0.814839/0.829682而MD0.521274。I的ACF1稍低，但MSD1=5.881634比A5.077737更高，MD为4.532827；不能解释为统一时间尺度修正。h64端到端距离更短、曲折度更高与主指标恶化同时存在，需保留潜在局部抖动/几何风险，而不是由这些描述数证明机制。

| h | lag ns | ACF I | ACF A | ACF MD | MSD I Å² | MSD A Å² | MSD MD Å² |
| --- | --- | --- | --- | --- | --- | --- | --- |
| h1 | 1 | -1.000000 | -1.000000 | -1.000000 | 2.666725 | 2.662100 | 3.866796 |
| h1 | 2 | null | null | null | null | null | null |
| h1 | 4 | null | null | null | null | null | null |
| h1 | 8 | null | null | null | null | null | null |
| h1 | 16 | null | null | null | null | null | null |
| h1 | 32 | null | null | null | null | null | null |
| h4 | 1 | 0.084469 | 0.082881 | -0.092812 | 2.958827 | 2.843796 | 3.345412 |
| h4 | 2 | -0.332065 | -0.345217 | -0.274313 | 4.416227 | 4.391780 | 4.669113 |
| h4 | 4 | -0.634192 | -0.617072 | -0.474979 | 6.970126 | 6.926457 | 6.983535 |
| h4 | 8 | null | null | null | null | null | null |
| h4 | 16 | null | null | null | null | null | null |
| h4 | 32 | null | null | null | null | null | null |
| h16 | 1 | 0.593476 | 0.615785 | 0.311995 | 3.848452 | 2.657137 | 4.388271 |
| h16 | 2 | 0.412571 | 0.409649 | 0.158342 | 3.521205 | 3.440476 | 6.112195 |
| h16 | 4 | 0.037180 | 0.038331 | -0.024439 | 6.081685 | 5.878606 | 8.510134 |
| h16 | 8 | -0.386436 | -0.391413 | -0.218759 | 9.754243 | 9.446340 | 11.953077 |
| h16 | 16 | -0.225540 | -0.291756 | -0.257514 | 13.036559 | 13.827875 | 23.684313 |
| h16 | 32 | null | null | null | null | null | null |
| h64 | 1 | 0.814839 | 0.829682 | 0.521274 | 5.881634 | 5.077737 | 4.532827 |
| h64 | 2 | 0.753024 | 0.760838 | 0.412940 | 2.268973 | 2.374361 | 6.284630 |
| h64 | 4 | 0.579161 | 0.590183 | 0.277030 | 3.982178 | 4.175037 | 8.702975 |
| h64 | 8 | 0.324694 | 0.335147 | 0.116972 | 6.637485 | 7.112943 | 11.760826 |
| h64 | 16 | -0.007670 | -0.007155 | -0.042809 | 10.448207 | 11.304954 | 14.870928 |
| h64 | 32 | -0.344245 | -0.356550 | -0.164278 | 15.481472 | 16.647834 | 19.305544 |

| 臂 | h64排列控制 | coordinate U-ES | distance U-ES | ACF1 |
| --- | --- | --- | --- | --- |
| increment | forecast | 1.921647 | 2.371945 | 0.814839 |
| increment | reversed_forecast | 1.931233 | 2.398881 | 0.797642 |
| increment | shuffled_forecast | 1.926100 | 2.383425 | -0.021168 |
| absolute | forecast | 1.879139 | 2.333937 | 0.829682 |
| absolute | reversed_forecast | 1.895415 | 2.377574 | 0.810882 |
| absolute | shuffled_forecast | 1.887774 | 2.357116 | -0.020379 |

Reverse/shuffle仅为预定输出排列，可能同时改变时间边际和endpoint，不是另训对照，也不证明S记忆。h1的ACF1=-1是两帧中心化的代数结果，不能解释为物理反相关；h1 walk-RMSF/ACF曲线correlation、persistence零方差ACF及曲折度的无支持值仍为null。h64两臂的MSD从lag1到lag2反而降低，也不支持单一连续时间尺度解释。单条MD不计算M≥2所需U-score。完整支持身份、available均值与complete-panel均值均保留，未删窗重新配对。

### Canonical CA几何

| 分组 | 指标 | I | A | 真实target | persistence | I−A [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| overall | 平均CA键长 Å | 3.840638 | 3.841200 | 3.833213 | 3.832086 | -0.000562 [-0.001052, -0.000131] |
| overall | 无效键比例 ↓ | 0.003556 | 0.003797 | 0.001474 | 0.001474 | -0.000241 [-0.000561, +0.000064] |
| overall | 非局部clash比例 ↓ | 0.000633 | 0.000695 | 0.000000 | 0.000000 | -0.000061 [-0.000087, -0.000039] |
| overall | Rg Å | 14.997398 | 14.895488 | 14.802277 | 15.042242 | +0.101910 [+0.072286, +0.135255] |
| h1 | 平均CA键长 Å | 3.832915 | 3.832910 | 3.831870 | 3.832086 | +0.000005 [-0.000003, +0.000012] |
| h1 | 无效键比例 ↓ | 0.001474 | 0.001474 | 0.001474 | 0.001474 | +0.000000 [+0.000000, +0.000000] |
| h1 | 非局部clash比例 ↓ | 0.000106 | 0.000102 | 0.000000 | 0.000000 | +0.000004 [+0.000000, +0.000007] |
| h1 | Rg Å | 14.939466 | 14.934154 | 14.929093 | 15.042242 | +0.005312 [+0.000876, +0.010950] |
| h4 | 平均CA键长 Å | 3.837365 | 3.836941 | 3.833855 | 3.832086 | +0.000424 [+0.000168, +0.000753] |
| h4 | 无效键比例 ↓ | 0.002053 | 0.001930 | 0.001474 | 0.001474 | +0.000123 [+0.000000, +0.000368] |
| h4 | 非局部clash比例 ↓ | 0.000474 | 0.000484 | 0.000000 | 0.000000 | -0.000010 [-0.000028, +0.000007] |
| h4 | Rg Å | 14.987503 | 14.947953 | 14.877753 | 15.042242 | +0.039551 [+0.012816, +0.066686] |
| h16 | 平均CA键长 Å | 3.844458 | 3.844831 | 3.833599 | 3.832086 | -0.000373 [-0.001010, +0.000203] |
| h16 | 无效键比例 ↓ | 0.004140 | 0.004228 | 0.001474 | 0.001474 | -0.000088 [-0.000544, +0.000351] |
| h16 | 非局部clash比例 ↓ | 0.000859 | 0.000930 | 0.000000 | 0.000000 | -0.000071 [-0.000133, +0.000007] |
| h16 | Rg Å | 15.097723 | 14.958883 | 14.738472 | 15.042242 | +0.138840 [+0.087586, +0.198667] |
| h64 | 平均CA键长 Å | 3.847814 | 3.850118 | 3.833527 | 3.832086 | -0.002304 [-0.003672, -0.001117] |
| h64 | 无效键比例 ↓ | 0.006559 | 0.007557 | 0.001474 | 0.001474 | -0.000998 [-0.001920, -0.000246] |
| h64 | 非局部clash比例 ↓ | 0.001095 | 0.001263 | 0.000000 | 0.000000 | -0.000168 [-0.000252, -0.000091] |
| h64 | Rg Å | 14.964900 | 14.740962 | 14.663789 | 15.042242 | +0.223939 [+0.168763, +0.283449] |

Canonical键长有效区间[3.3,4.3] Å、非局部clash阈值2.5 Å；按每个生成构象计算后蛋白等权，非pool全部原子。I的overall clash较少，但无效键比例差区间跨零；平均键长和Rg没有预定“越低越好”方向。不能用低CA clash代替N/C/O、PSF连通、Rama或T1全骨架验证。

### Full12温度mode与matched8 T2A

| Full12 mode | 臂 | MAE ↓ | Corr ↑ | Root ↓ | Variance ↓ | PCA ↓ | Weak ↑ | Transient ↑ | 成员RMSD Å ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| correct | increment | 1.441050 | 0.454359 | 2.106028 | 14.478563 | 9.750970 | 0.110309 | 0.248360 | 5.769079 |
| correct | absolute | 1.484753 | 0.452562 | 2.121334 | 14.791842 | 9.748160 | 0.107476 | 0.245178 | 5.793162 |
| wrong | increment | 1.611489 | 0.461576 | 2.156724 | 16.361577 | 10.298387 | 0.112525 | 0.252497 | 5.663969 |
| wrong | absolute | 1.649744 | 0.458332 | 2.171116 | 16.634026 | 10.326501 | 0.111544 | 0.252301 | 5.685508 |
| none | increment | 1.396180 | 0.470032 | 2.066393 | 14.782941 | 10.165933 | 0.116534 | 0.260648 | 5.549188 |
| none | absolute | 1.410904 | 0.471108 | 2.066173 | 14.895482 | 10.193015 | 0.116202 | 0.261512 | 5.538133 |

各mode为2400窗、12蛋白/300轨迹的描述均值；每臂三mode共7200窗。correct的成员RMSD I/A=5.769079/5.793162 Å，均高于persistence4.930655 Å。none的MAE和成员RMSD反而更低，因此correct未全面主导；wrong与none也保留全部七项。Wrong将320 K改为450 K，其余改320 K；none用348 K，348 K的correct/none输出相同，不算两份支持。五温度均见于训练，这是输入干预，不是未见温度泛化。原suite的4000次bootstrap/seed17区间只作为历史输出保留，本表不混入新的10000次配对CI。

| 面板 | K | coverage I/A ↑ | TV I/A ↓ | JSD I/A ↓ | F-MAE kcal/mol I/A ↓ |
| --- | --- | --- | --- | --- | --- |
| suite | 320 | 0.487500 / 0.500000 | 0.740385 / 0.737638 | 0.403997 / 0.399338 | 1.028229 / 1.040276 |
| suite | 348 | 0.393750 / 0.393750 | 0.734218 / 0.730050 | 0.389567 / 0.385215 | 1.025709 / 1.024597 |
| suite | 379 | 0.227083 / 0.225000 | 0.695053 / 0.693036 | 0.354572 / 0.353236 | 1.107226 / 1.104770 |
| suite | 413 | 0.106250 / 0.108333 | 0.631510 / 0.624515 | 0.301068 / 0.295524 | 1.078364 / 1.059110 |
| suite | 450 | 0.016667 / 0.016667 | 0.573976 / 0.569759 | 0.262680 / 0.258773 | 1.050782 / 1.044000 |
| matched8 | 320 | 0.403125 / 0.406250 | 0.806904 / 0.802207 | 0.459521 / 0.455043 | 1.000874 / 1.018933 |
| matched8 | 348 | 0.265625 / 0.265625 | 0.756546 / 0.745813 | 0.417199 / 0.407750 | 1.000367 / 0.985376 |
| matched8 | 379 | 0.218750 / 0.215625 | 0.727048 / 0.723821 | 0.383031 / 0.381668 | 1.083241 / 1.074263 |
| matched8 | 413 | 0.090625 / 0.087500 | 0.631245 / 0.627331 | 0.303095 / 0.299454 | 0.997646 / 0.985439 |
| matched8 | 450 | 0.015625 / 0.015625 | 0.598923 / 0.596698 | 0.284601 / 0.281456 | 1.026952 / 1.025802 |

| 历史Bio matched8 K | coverage ↑ | TV ↓ | JSD ↓ | F-MAE kcal/mol ↓ |
| --- | --- | --- | --- | --- |
| 320 | 0.368750 | 0.626386 | 0.291156 | 0.885832 |
| 348 | 0.246875 | 0.588990 | 0.269480 | 0.835022 |
| 379 | 0.178125 | 0.622056 | 0.292708 | 1.112960 |
| 413 | 0.081250 | 0.544678 | 0.239706 | 0.964945 |
| 450 | 0.015625 | 0.568909 | 0.258076 | 0.990064 |

Full12每轨迹96成员；matched8为800窗、8蛋白/200轨迹、每轨迹48成员，支持集不可混用。matched8的I相对A五温度TV/JSD均更高；两臂TV/JSD五温度描述均值也均高于保存BioEmu2，F-MAE仅379 K方向较好。k=8时population MAE=TV/4，不算独立证据。Coverage接近不证明布居正确。历史Bio虽800窗口键、200轨迹及声明预算相配，仍缺不可变mask/reference坐标/帧/状态中心指纹，因此严格配对T2A差保持null；上表仅作描述并列，不能升级为新的配对胜负结论。

本轮完成了预定配对试验并得到不利于increment主指标的结果，没有完成“所有任务超过BioEmu2”的目标。冻结H/CA读出gate长期失败保持原样；没有新增matched-time Bio路径、连续192 ns、MSM/T2B、全骨架可行性或跨seed复现证据。不能将本轮absolute对increment的优势外推为无辅助对照收益、S记忆效应或下一配方的自动采用依据。

