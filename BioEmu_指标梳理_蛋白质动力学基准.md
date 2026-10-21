# BioEmu 论文中的指标梳理：蛋白质构象系综、轨迹与状态预测

**整理日期：2026-10-21**  
**用途：** 为蛋白质构象系综/动力学模型的对比评测提供指标清单，并澄清 BioEmu 原论文实际评估了什么。

> **名称说明与重要结论**：可核实的 Science 主论文是 Lewis 等人的 *Scalable emulation of protein equilibrium ensembles with generative deep learning*（BioEmu，2025），重点是从序列采样**平衡构象系综**，不是逐帧预测未来坐标的轨迹预测器。其论文/官方模型说明中没有把“MSM state prediction accuracy（MSM 状态预测准确率）”作为主要指标。MSM/HMM/TICA用于构建/分析平衡状态及计算状态自由能误差；这与预测下一状态的分类准确率不同。若你说的“BioEmu-2”是另一篇具体论文，欢迎发链接/DOI，我可以再按那篇逐项校准。本文件先按可核实的 BioEmu Science 主论文及官方评估说明整理。

## 1. BioEmu 原文评估任务概览

| 任务 | 比较对象 | 论文采用的核心指标 | 衡量什么 | 是否是未来轨迹预测 |
|---|---|---|---|---|
| 已知功能构象变化采样 | 生成样本 vs 实验/PDB参考状态 | **Reference-state coverage（参考状态覆盖率）** | 目标状态有没有被模型采到 | 否；无时间顺序要求 |
| MD 平衡系综仿真 | 模型采样 vs 参考MD | **Metastable-state free-energy MAE** | 各亚稳态平衡自由能/占比是否匹配 | 否；比较稳态分布，不检验转移时间序列 |
| 小蛋白稳定性 | 预测折叠自由能 vs 实验 | **MAE、Pearson r、Spearman ρ** | 热力学量与实验的一致性 | 否 |

Science 摘要概括的三项成果分别是已知构象变化采样成功率约55–90%、相对自由能误差低于约1 kcal/mol、稳定性预测误差低于约1 kcal/mol且相关系数大于0.6。[Science 主论文](https://www.science.org/doi/10.1126/science.adv9817)

## 2. 与蛋白质轨迹/未来演化预测最相关的原文指标

### 2.1 参考构象覆盖率（Coverage）

官方评估说明：对每个参考状态，若至少 **0.1%** 的生成样本落在预设距离阈值内，则认为该状态被覆盖；先对每条序列的参考态求平均，再对序列求平均。局部解折叠和隐蔽口袋任务再按不同状态类型分别报告。[BioEmu 官方模型卡](https://huggingface.co/microsoft/bioemu)

可写作：

\[
C_s = \mathbf{1}\left[\frac{1}{N}\sum_{i=1}^N\mathbf{1}(d(x_i,r_s)<\epsilon)\geq 0.001\right]
\]

再按状态、蛋白分别宏平均。这里 \(d\) 和阈值 \(\epsilon\) 要按任务指定的距离定义（例如域运动的结构距离、局部区域距离、口袋几何量），不应把一个统一阈值硬套所有任务。

BioEmu 报告值（模型卡口径）：域运动总体覆盖约 **85%**；局部解折叠任务中局部折叠/解折叠参考状态约 **72%/74%**；隐蔽口袋任务中 apo/holo 状态约 **49%/85%**。这些数值是“有没有采到状态”的 coverage，不是每帧准确率、状态概率校准或转移预测精度。

**对你工作的启发**：建议保留 coverage，但同时报告覆盖所需样本数/采样预算、状态命中率，以及状态内最小距离或分位数距离。只报 coverage 容易被大量采样抬高，也无法区分“偶然碰到一次”与“稳定、正确地重现该态”。

### 2.2 亚稳态自由能误差（MSM/HMM state free-energy MAE）

官方流程概述：对参考MD做 TICA 投影，在低维空间用 HMM 划分亚稳态；把模型样本投影到相同空间，并按同一 HMM 归类；比较模型与参考MD中各态的自由能，计算 MAE。[官方模型卡](https://huggingface.co/microsoft/bioemu)

若状态概率为 \(p_k\)，相对自由能可定义为：

\[
F_k=-k_BT\ln p_k + c
\]

在一致的归一化/参考态下：

\[
\mathrm{MAE}_{F}=\frac{1}{K}\sum_{k=1}^{K}|F_k^{model}-F_k^{ref}|
\]

原文报告：内部 MD 平衡分布基准约 **0.91 kcal/mol**；快速折叠蛋白基准约 **0.74 kcal/mol**。注意模型卡说的是“态自由能 MAE”，不可误读成逐帧能量误差或下一状态预测误差。

**公平比较要点**：固定并公开特征、TICA训练集、维数、滞后时间、HMM状态数/拟合数据、状态匹配规则、温度、参考态与概率平滑方式。否则自由能误差可能主要反映状态划分差异。建议补充 state-wise 概率散点图、population error、Jensen–Shannon divergence（JSD）或总变差距离（TV），尤其是低占比态。

### 2.3 重要的负结论：并没有 MSM state prediction accuracy

“MSM state prediction accuracy”通常意味着：由真实历史状态（或结构）出发，预测滞后时间 \(\tau\) 后的状态标签，和参考轨迹标签比较。BioEmu 的上述评估则是对平衡样本分布做状态归属并比较自由能；它没有因此验证状态转移顺序、速率或逐步状态预测准确率。

若你的方法确实预测未来态，推荐明确定义并报告：

- **Lagged state accuracy**：\(\Pr[\hat{s}(t+\tau)=s(t+\tau)]\)，至少在多个 \(\tau\) 上报告；同时给 macro-F1、balanced accuracy，避免主态占比造成虚高。
- **转移矩阵误差**：参考与预测 \(T(\tau)\) 的矩阵误差（如按行 JSD/TV 或 Frobenius norm），分开报告稀有转移。
- **状态人口误差**：state population MAE/JSD/TV；这测平衡分布，不替代动力学指标。
- **动力学指标**：各态 MFPT、转移速率、寿命/驻留时间误差；比较 implied timescales，并检查 Chapman–Kolmogorov 一致性。
- **路径指标**：主通路/通量、路径概率或路径分布差异；若声称预测 folding pathway，仅有末态命中不够。
- **概率预测质量**：若输出下一态分布，报告 log loss / cross-entropy、Brier score、校准曲线；不能只报 argmax accuracy。

> 评估协议应说明模型预测的是连续坐标、离散状态标签，还是概率分布；从何种起始条件预测；预测时间间隔是多少；测试蛋白/轨迹是否与训练集按同源性隔离。

## 3. 结构和功能态覆盖的距离/分类指标

BioEmu 的功能构象基准主要报告按指定结构距离阈值定义的 coverage，而不是一套通用于所有任务的 RMSD 单指标。若要复现或扩展，可结合以下量：

- **全局/域级 RMSD、TM-score**：测结构接近度；域运动应对齐相关域或用域间几何量，避免全局对齐掩盖运动。
- **局部区域 RMSD / 接触图差异**：适合局部解折叠、开关环等事件。
- **口袋状态量**：口袋体积、可及表面积、关键残基间距/接触、口袋开闭分类；报告 apo/holo 分态的 precision/recall 或 coverage。
- **参考态命中阈值敏感性**：对多个 \(\epsilon\) 画 coverage–threshold 曲线，而不是只挑一个阈值。
- **采样效率**：每个状态达到首次命中的样本数/时间、coverage 随样本数曲线、GPU 时间/采样成本。

这些是适合补充的建议，不应错误标成 BioEmu 原论文全部实际报告的指标。

## 4. 稳定性/热力学指标（相关但不是轨迹预测）

原文还用生成系综预测折叠稳定性：MAE 与实验折叠自由能比较，并报告 Pearson 和 Spearman 相关。官方模型卡报告实验稳定性预测约 **0.76 kcal/mol MAE**、Pearson **0.66**、Spearman **0.64**。另有论文摘要层面的概括为误差低于约1 kcal/mol、相关系数大于0.6。[官方模型卡](https://huggingface.co/microsoft/bioemu) [Science 主论文](https://www.science.org/doi/10.1126/science.adv9817)

此项是跨蛋白热力学预测；如果你的方法输出 mutant 影响，可另报 \(\Delta\Delta G\) 的 MAE、相关性和方向准确率，但需与轨迹未来状态预测分开呈现。

## 5. 针对“超过 BioEmu”的建议评测表

| 评测维度 | 建议主指标 | 建议附加指标 | 适用情形 |
|---|---|---|---|
| 构象状态找回 | Reference-state coverage | 距离阈值曲线、首次命中样本数、宏平均 recall | 生成构象系综 |
| 平衡占比/热力学 | State free-energy MAE | population MAE、JSD/TV、低占比态误差 | 平衡系综模拟 |
| 未来离散状态预测 | Lagged macro-F1 / balanced accuracy | log loss、Brier、校准、按滞后时间曲线 | 下一态/多步状态预测 |
| 状态转移动力学 | 转移矩阵 JSD/TV、MFPT 相对误差 | implied timescale、驻留时间、CK test | 轨迹预测/动力学模拟 |
| 连续结构预测 | 多步结构 RMSD/TM-score | 局部 RMSD、接触图误差、结构物理性 | 未来坐标/结构预测 |
| 功能转换路径 | 路径分布/通量误差 | 关键中间态召回、转移路径命中率 | folding/构象转换路径 |
| 计算成本 | 达到目标误差的 wall-clock/GPU 成本 | 每有效独立样本成本 | 效率比较 |

**强烈建议**：把“静态/平衡系综指标”和“时间动力学指标”分成两张结果表。BioEmu 的 coverage 和自由能 MAE 可作为公平的 ensemble baseline；你若宣称未来演化预测，则应新增 lagged-state、transition-matrix、MFPT/路径等真正时间敏感的指标，不能仅凭 ensemble coverage 声称超越轨迹预测能力。

## 6. 参考来源

1. Lewis et al., *Scalable emulation of protein equilibrium ensembles with generative deep learning*, Science (2025). https://www.science.org/doi/10.1126/science.adv9817
2. Microsoft BioEmu 官方模型卡（评估定义与数值）。https://huggingface.co/microsoft/bioemu
3. BioEmu 预印本及补充信息入口：https://www.biorxiv.org/content/10.1101/2024.12.05.626885

---

**一句话总结：** BioEmu 的动力学相关主指标是“是否覆盖已知构象态”和“平衡亚稳态自由能是否接近MD参考”；它不是用 MSM 下一状态准确率来证明未来轨迹预测能力。