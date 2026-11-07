# 主链与有序路径诊断 — 2026-11-01

本报告补充[四臂训练报告](RESULTS_HS_JOINT_20261101.md)之外的评测覆盖。这里使用固定的 `spatial_probe_v1/spatial_exact.pt` 初始权重，**不是**看过四臂结果后挑选的500步模型。开发验证集被反复用于研究；本报告不是独立确认，也没有新增 BioEmu2 比较。完整目标仍未达成。

## 1. 主链 v2：200 窗口实际结果

[完整结果](results/hs_joint_20261101/backbone_v2/spatial_exact_allres200_v2.json)、[运行前协议](results/hs_joint_20261101/backbone_v2/spatial_exact_allres200_v2.protocol.json)、[执行审计](results/hs_joint_20261101/backbone_v2/spatial_exact_allres200_v2.execution.json)及同名前缀的日志、源码快照均已保存。50 蛋白各贡献1/4/16/64 ns各一窗，预测每窗12样本，最多96残基。预测先由观察历史产生，之后才读取未来标签；未来 H 只进入明确标注的 oracle 分支。checkpoint、输入与源码哈希在运行前后相同。

真实未来 CA 对齐到观察 anchor 一次，同一个刚体变换应用于真实 N/CA/C/O。主表是此固定坐标系下四种主链原子的 RMSD：`sqrt(mean_atoms(sum_xyz(error²)))`，预测不再拟合真值；另存独立 Kabsch shape 诊断。对样本、蛋白内窗口、蛋白依次等权。原指标 v1 的平均欧氏距离、错位 phi/psi、仅 sample0 几何及遗漏同链碰撞问题已由独立 v2 指标修正；历史数值保留，不与新数值直接排序。

| 路径 | 1 ns RMSD Å | 4 ns | 16 ns | 64 ns |
|---|---:|---:|---:|---:|
| 预测 CA → ideal lift | 3.011979 | 3.898644 | 5.401398 | 6.873976 |
| 预测 H → 旧 learned B1a | 2.277520 | 2.792134 | 4.122798 | 5.330447 |
| 真实未来 H → 当前 CA → ideal lift（oracle） | 1.623693 | 1.745951 | 2.240419 | 2.932808 |
| 真实未来 CA → ideal lift（oracle） | 1.405898 | 1.407848 | 1.413552 | 1.418747 |
| 当前 encoder 的真实未来 H → 旧 B1a（oracle） | 2.228697 | 2.714542 | 3.955175 | 5.114204 |
| 原 Stage-A 的真实未来 H → 旧 B1a（oracle） | 2.187056 | 2.671279 | 3.897717 | 5.063494 |
| 观察真实主链 persistence | 2.200122 | 2.706524 | 3.950230 | 5.131952 |
| 真实未来主链自身（reference） | 0 | 0 | 0 | 0 |

旧 B1a 本来就有 learned frame 和 torsion head。已核验其原 Stage-A encoder 权重、表示合同及 decoder 血缘；宽度相同本身不算兼容性证据。它产生自己的 CA，因而是另一种读出，不是给当前 CA 添加缺失原子。原生 H oracle 也接近 persistence，因此不能把旧头表现全部归因于 H 基底变化。单个未来实例的 RMSD 可以偏好低方差预测；此表不证明旧 B1a 的条件分布优于当前模型。

### 跨残基几何仍然失败

| 路径 / 指标 | 1 ns | 4 ns | 16 ns | 64 ns |
|---|---:|---:|---:|---:|
| 真实 CA ideal lift：C–N 均值 Å | 2.608851 | 2.609578 | 2.609038 | 2.605239 |
| 真实 CA ideal lift：C–N legacy valid fraction | 0 | 0 | 0 | 0 |
| 预测 CA ideal lift：C–N 均值 Å | 2.602676 | 2.592560 | 2.586566 | 2.586754 |
| 预测 H → B1a：C–N 均值 Å | 1.297541 | 1.365243 | 1.654010 | 1.992173 |
| 预测 H → B1a：C–N legacy valid fraction | .204546 | .205229 | .124369 | .085039 |
| 真实 MD：C–N 均值 Å | 1.349613 | 1.348301 | 1.348407 | 1.349038 |
| 真实 MD：C–N legacy valid fraction | .782967 | .786168 | .802293 | .791490 |

即使用真实未来 CA，当前 ideal lift 仍有约1.4 Å主链误差和约2.61 Å肽键；其 ω_trans fraction=1 不能证明链连续。真实 MD 本身的 legacy validity 也不为1，所以这些历史窗口仅作诊断；另存 OpenFold 类型相关 z 误差及有效计数。需要能表达并约束跨残基 C–N、键角和碰撞的读出，再分别测真实 H 的可读出性与自主预测。当前结果还没有正式统计 Ramachandran 或全主链/全原子 clash，**T1 未完成**。

### 残基类型与覆盖

数据集/已训练 embedding 的编号顺序为 `ACDEFGHIKLMNPQRSTVWYX`，OpenFold 化学顺序为 `ARNDCQEGHILKMFPSTWYVX`。本诊断只在 ideal lift 和化学参考指标处转换；旧 B1a 保留历史输入和错误 placement 语义，明确标注。不能整体替换旧 checkpoint 的 embedding 输入编号。另有手写 atom14 表的11/20 canonical残基原子集合错误与项目/OpenFold atom37顺序差异，详见[残基类型审计](docs/RESIDUE_TYPE_AUDIT_20261101.md)。这些是不同问题；当前 CA 头没有直接使用上述化学表，不能用它们解释四臂之间的 CA 差异。

HSD/HSE/HSP 等旧 parser 未识别的别名成为 UNK。OpenFold adapter **已有** UNK→GLY 主链模板回退，不能说它因此缺失主链。200窗共424个重复计入的未知残基（50蛋白中43个蛋白、106个蛋白内唯一位置）；所有路径对全部观察有效残基评分，覆盖率100%，没有删除 UNK。仅类型相关化学 z 参考排除未知类型并报告计数。读取元数据才能恢复别名身份，不能从整数20猜测。

第一次4窗 smoke 使用旧化学编号语义，仅是历史工具运行记录；第一次完整面板在严格残基检查处失败，保存日志而没有最终结果。上述表格只来自成功的 `spatial_exact_allres200_v2.json`，SHA-256 为 `10c2bf304988bb21fdd6db777d0b0b65533029f4cf9a787f50fb6fe011eebadf`。

## 2. 有序路径接口与证据边界

新增 `evaluation/ca_trajectory_adapter.py` 在一次既有采样中保留连续 H/S 序列，沿同一观察 anchor 解码中间帧，不重新编码预测帧或重置 bath。真实 checkpoint 的64 ns / 3成员验证中，终点 CA、终点 H 和 RNG 状态与原 endpoint sampler **逐位一致**；[记录](results/hs_joint_20261101/ordered_paths/real_adapter_check.json)只证明接口一致性。

`evaluation/trajectory_metrics_v2.py` 新增整个未来路径的 U-energy score（时间0除外、固定尺度归一化），以及每条路径内部的 ACF/MSD、路径长度与形状诊断。成员或 replica 之间不拼接时间；真实 MD 每窗只是一条路径，不复制成12个独立真值。`walk_rmsf` 仍池化成员和时间，包含成员间差异且不识别顺序。短轨迹 ACF 不是平衡松弛时间，内部距离 MSD 不是原子扩散常数。

新 runner `scripts/eval_ca_ordered_paths.py` 固定 standard200 与独立192 ns/50窗面板；反转/乱序保留未来帧集合与t0，但会改变时间边际及可能的终点，所以只作时间顺序敏感性诊断，不是记忆因果对照。每次运行分别登记协议、快照和输入哈希。本接口本身没有补齐 MSM/T2B、同蛋白温度配对或 BioEmu2 动力学对照。

## 3. 有序路径实际面板

[standard200结果](results/hs_joint_20261101/ordered_paths/spatial_exact_standard200.json)已完成，200窗/50蛋白，每窗12条路径；[执行记录](results/hs_joint_20261101/ordered_paths/spatial_exact_standard200.execution.json)确认exit0和运行前后源码/输入一致，结果SHA-256为 `c519197a5d11a9399426d1d4700a6bbaac94232d9c1700ca3be625aede3a9aab`。下表是等蛋白均值，配对推断单独报告。坐标path ES按 `sqrt(3LT)` 归一化，不能直接与采用其它尺度的endpoint ES数值比较。

[独立192 ns结果](results/hs_joint_20261101/ordered_paths/spatial_exact_long192_separate50.json)也完成50窗/50蛋白/每窗12条路径，[执行记录](results/hs_joint_20261101/ordered_paths/spatial_exact_long192_separate50.execution.json)确认exit0及哈希一致，结果SHA-256为 `3d3d1d5d08c7eeeafc471a280cc72f18aa9237ee28e381514b0aeecd8dafbe5b`。它单独登记，不假定与较短horizons构成同一条路径，也不把它混入standard200。

| Horizon ns | Forecast coordinate path ES ↓ | Persistence | Forecast distance path ES ↓ | Persistence | Forecast逐帧RMSD Å | Persistence |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | .948808 | 1.271944 | 1.193993 | 1.625211 | 2.702129 | 2.203072 |
| 4 | 1.043510 | 1.426558 | 1.367417 | 1.871269 | 3.240314 | 2.443948 |
| 16 | 1.459909 | 1.935563 | 1.935913 | 2.568151 | 4.326938 | 3.248404 |
| 64 | 1.891368 | 2.502094 | 2.398990 | 3.263751 | 5.579438 | 4.198511 |
| 192（独立面板） | 2.512399 | 3.260517 | 3.141047 | 4.297896 | 7.077851 | 5.439343 |

| Horizon ns | Forecast内部距离ACF lag1 | MD | Forecast内部距离MSD lag1 Å² | MD | Forecast形状路径长度 Å | MD |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | -1 | -1 | 2.373508 | 3.866796 | 1.281361 | 1.807536 |
| 4 | .077471 | -.092812 | 2.536203 | 3.345412 | 5.203273 | 6.974818 |
| 16 | .607944 | .311995 | 1.966154 | 4.388271 | 17.571144 | 30.179146 |
| 64 | .825096 | .521274 | 4.202990 | 4.532827 | 64.464629 | 130.226965 |
| 192（独立面板） | .899312 | .663777 | 5.548696 | 5.359163 | 192.306044 | 433.219385 |

1 ns只有两个含anchor时刻，去均值ACF的−1是数学退化，不能解释为反相关动力学。64 ns的ACF曲线相关性虽达.935806，lag1的相关性水平却高于MD，形状路径长度约为MD的一半。相同future帧集合乱序后，该ACF曲线相关性均值变为−.064806、形状路径长度变为172.820526 Å，而池化walk RMSF不变；这说明两类指标实际识别不同内容，不能单看高ACF曲线相关性就宣称动力学准确。

192 ns路径全部有限，但形状路径长度约为MD的44%，lag1 ACF偏高；其ACF曲线相关性.947620也不能消除这种水平偏差。内部距离MSD均值接近MD，同时形状路径长度偏低，提示这些观测量/平均方式测的是不同性质；不能把单项均值接近当成全部动力学校准。所有horizons的平均逐帧位置误差都高于persistence，与path ES的均值改善并存。新面板没有重跑BioEmu2，也不能用于宣称超过其动力学性能。

### 等蛋白配对区间

[统计预登记](results/hs_joint_20261101/ordered_paths/summary.protocol.json)在正式面板结果被读取前保存，固定10,000次按蛋白配对bootstrap、seed20261101、逐horizon评分；[正式汇总](results/hs_joint_20261101/ordered_paths/summary.json)验证完整面板、每条路径的合法lag计数、单条MD reference、execution exit0和所有来源。192 ns独立汇总，不作跨horizon配对。这些是自适应开发集的边际探索性区间，没有多重比较或训练seed校正，不设新的通过门槛。

下表为 forecast−persistence，负值更好：

| Horizon ns | Coordinate path U-ES Δ [95% CI] | Distance path U-ES Δ [95% CI] |
|---:|---|---|
| 1 | −.323136 [−.387299, −.268409] | −.431218 [−.547315, −.335592] |
| 4 | −.383049 [−.463576, −.309628] | −.503851 [−.647320, −.377920] |
| 16 | −.475655 [−.606206, −.362454] | −.632239 [−.903644, −.421666] |
| 64 | −.610726 [−.822015, −.418109] | −.864761 [−1.288987, −.508840] |
| 192（独立面板） | −.748118 [−1.026797, −.465225] | −1.156849 [−1.671575, −.722407] |

汇总还完整保留 reverse−forecast 与 shuffle−forecast。4 ns四项顺序对照区间均跨0；16 ns reverse coordinate和192 ns两项distance顺序对照也跨0。1 ns只有一个未来帧，排列完全相同，差值与区间均为0。不能宣称所有顺序控制均成立，也不能从这个控制推出 S 的因果贡献。独立实现复算30组均值/差值/区间，均在1e−12内一致；正式summary SHA-256为 `52511830a4ad43cb633b370d1f1ff688cf295d5ed9e51abeaed8401a8311e5f9`。

### 常量 persistence 的数值勘误

冻结原始工件中，persistence 的 `walk_rmsf_corr` 在4/16/64 ns各46/50窗、192 ns全部50窗返回有限数值。独立复现表明，这是常量轨迹减浮点均值后约10⁻¹⁴ Å残差被Pearson归一化造成；它的物理RMSF为零，相关性应未定义。所有这些窗口的path length、end-to-end和MSD均精确为零。汇总须将这一**已知确定性控制**的corr标为null、有效计数0，并保存原值与原因；不修改冻结raw JSON或对其它模型按结果挑阈值。主joint U-energy score直接用坐标与成员距离，不经过此中心化计算，独立toy核验确认不受该问题影响。

评测代码冻结后的全量CPU回归为 **540 passed, 1 skipped**，见[日志](results/hs_joint_20261101/ordered_paths/full_tests_final.log)；后续汇总脚本新增11项测试通过。测试支持实现契约，不构成预测性能或总体目标达成证据。

## 4. 下一步主链读出的可训练约束

新增独立 [backbone_objectives_v2.py](protein_jepa_frame/models/stage_c/backbone_objectives_v2.py)，直接作用于实际 N/CA/C/O，提供可微 C–N、CA–C–N/C–N–CA 两角及主链跨残基碰撞损失。其化学ID、Pro/UNK、链边界、缺失原子、聚合方式和单位均显式记录，不把四项不同单位隐式相加；默认12SD/1.5 Å宽松容忍只用于软约束，不等同T1通过标准。旧B1a损失缺少跨残基约束，Stage C旧bond/clash诊断被detach；新模块不改变旧训练语义。

18项CPU测试与独立四项联合float64 gradcheck通过，包括固定CA时OpenFold实际重建输出仍向旋转参数传递有限非零梯度。模块SHA-256为 `3356cd0ed27280237e8f8277349f4ccff9fcab9b972a0f0f3e42c45934227223`。这还不是训练后的模型；最小新实验、原始公式来源、旧实现的角度标准差问题，以及固定异常CA可能无法闭合肽键的限制见[修复方案](docs/BACKBONE_REPAIR_PLAN_20261101.md)。先校准真实H的主链读出，再用生成H检验自主预测，不能把oracle成绩当forecast。

新损失与有序路径汇总冻结后合并运行的新增测试为 **29 passed**，见[日志](results/hs_joint_20261101/backbone_repair/new_objective_and_summary_tests.log)；它们在此前540项全量回归之后新增。

### 训练集实际坐标与拓扑检查

[train32检查](results/hs_joint_20261101/backbone_repair/train32_geometry_check_v4.json)固定首32个训练蛋白，每个只取最后观察主链、最多96残基，无模型载入、optimizer、验证/测试数据或选参。全部32条状态保留：**31个窗口的拓扑可辨识，1个拒绝**，不代表完整蛋白或全训练集均已审计。该例 `1kskA01` 原始PDB前三残基是同链resSeq=1、插入码A/B/C；parser保留三个残基，却在导出`residue_index`时丢弃插入码，变成三个相同的(chain0,index1)。新损失的拒绝是正确的；不能悄悄丢掉残基、猜测连接或修改旧encoder的输入编号后宣称修复。

下表只描述31个可评分蛋白，按蛋白等权。MD与lift来自同一观察CA，后者用原float32 lifter重建，损失和坐标梯度均用float64计算。

[原始拓扑证据](results/hs_joint_20261101/backbone_repair/1kskA01_insertion_topology_evidence.json)还逐原子核验了1028个protein PDB atoms与PSF前缀的身份一致，PSF明确包含1A→1B、1B→1C及1C→2的C–N连接。因此新几何合同可以保留完整UID与真实peptide edges；它们应与旧encoder的`residue_index`输入分开，不能只按数值编号差1猜连接。

| 软约束项 | 观察真实MD均值 | 同一MD CA的ideal lift均值 | lift非零梯度蛋白数 |
|---|---:|---:|---:|
| C–N（Å） | 0 | 1.106837 | 31/31 |
| CA–C–N（cosine） | 0 | .013888 | 31/31 |
| C–N–CA（cosine） | 0 | .568412 | 31/31 |
| 跨残基主链clash（Å） | 3.6613e−7 | 3.2653e−5 | 31/31 |

所有已评分梯度有限、masked坐标梯度精确为0。这说明新损失能识别实际lift的明显错误，并不保证参数化读出能沿任意坐标梯度达到解。真实MD前三项为0只反映默认12SD宽松flat bottom，不能当成严格立体化学验证。训练前须增加独立于旧embedding输入的完整残基UID/化学类型/物理连接合同；当前32个蛋白不能整体认证为可训练。前次接口失败及严格拓扑拒绝的源码、协议和日志均保留；只有v4输出是完整覆盖审计，且明确记录31/32的loss适用范围。

## 5. 固定 CA learned head：500 步开发实验

本节记录当日 **21:31 UTC之后**完成的新训练；§4“尚未训练”和31/32拓扑检查保留为此前阶段的状态。新增独立化学/PSF拓扑合同与固定CA读出头后，两臂均完成500步。结果支持小幅改善真实H的骨架读出；几何监督改善部分连接项，同时牺牲坐标误差并增加碰撞软损失，**没有完成T1或总体目标，也不支持推广该recipe**。

[训练预注册](docs/BACKBONE_READOUT_PILOT_V2_20261101.md)已随初始化、定标和训练保存快照；[统计登记](results/hs_joint_20261101/backbone_repair_v2/summary.protocol.json)于21:39:11 UTC保存，早于正式评估。训练为500个不同蛋白各一窗，seed42、1/4/16/64 ns轮转、每窗一次更新；AdamW学习率1e−4、weight decay 1e−4、clip1，25步warmup后cosine至1e−5。两臂使用同一corrected-chemical B1a复制初始化，仅训练新head，encoder、CA decoder及H/S均冻结；没有torsion CE，psi通过实际O坐标受监督。

coordinate臂仅用实际N/C/O平方欧氏误差。geometry臂另外使用显式PSF的C–N、CA–C–N、C–N–CA、clash，权重分别1/4/4/100；总系数只由[前32个训练窗的零更新梯度定标](results/hs_joint_20261101/backbone_repair_v2/gradient_scale.json)确定：平均坐标/几何梯度范数3.797473/1.994159，`λ=.952149112`，未触及截断边界。两臂都固定采用500步latest，无验证集选点或选权重；[coordinate执行](results/hs_joint_20261101/backbone_repair_v2/train_coordinate.execution.json)、[geometry执行](results/hs_joint_20261101/backbone_repair_v2/train_geometry.execution.json)均exit0。

[正式评估](results/hs_joint_20261101/backbone_repair_v2/evaluation200.json)及[执行记录](results/hs_joint_20261101/backbone_repair_v2/evaluation200_run.execution.json)覆盖标准200窗、50蛋白、每窗全部12个forecast，训练/验证蛋白交集为0。forecast缓存先由观察历史生成，随后才读标签；oracle使用真实未来H及其冻结CA读出，**不是**直接用真实未来CA放置。真实未来主链仅经一次观察anchor对齐用于评分。新主指标排除固定CA，先计算每个样本的N/C/O RMSD，再按样本、蛋白内窗口、蛋白等权；不能直接与§1的四原子RMSD排序。

| 路径 / NCO RMSD Å ↓ | 1 ns | 4 ns | 16 ns | 64 ns | Overall |
|---|---:|---:|---:|---:|---:|
| init / oracle | 1.059652 | 1.280473 | 1.881392 | 2.670401 | 1.722979 |
| coordinate / oracle | .984673 | 1.212069 | 1.827931 | 2.623174 | 1.661962 |
| geometry / oracle | .999405 | 1.224998 | 1.838440 | 2.631051 | 1.673474 |
| init / forecast | 2.726943 | 3.710798 | 5.290545 | 6.822392 | 4.637670 |
| coordinate / forecast | 2.708793 | 3.711108 | 5.287308 | 6.823374 | 4.632646 |
| geometry / forecast | 2.712211 | 3.712743 | 5.289001 | 6.821747 | 4.633925 |
| 真实主链 persistence（描述） | 2.198955 | 2.700986 | 3.938316 | 5.117159 | 3.488854 |

coordinate−init的overall NCO差值为oracle **−.061018 Å** [−.071610, −.051021]、forecast **−.005024 Å** [−.008050, −.002066]；geometry−init分别为 **−.049506 Å** [−.058528, −.040965]与 **−.003744 Å** [−.006956, −.000573]。geometry的oracle四个horizon区间均负，但其forecast在4/16/64 ns相对init的区间均跨0，不能把微小overall改善说成稳定长时迁移。

下面是主对照 **geometry−coordinate，overall Δ [95% CI]**，均为未乘训练权重的原单位损失，负值更好。区间按蛋白配对bootstrap 10,000次、seed20261101计算；所有条目均有50蛋白/200配对窗口。

| 指标 | Oracle | Forecast |
|---|---|---|
| NCO RMSD Å | +.011512 [.009121, .013985] | +.001280 [.000506, .002069] |
| C–N软损失 Å | −.037804 [−.041223, −.034325] | −.028631 [−.029749, −.027410] |
| CA–C–N cosine软损失 | +.000130 [−.000612, .000882] | +.000004 [−.000261, .000264] |
| C–N–CA cosine软损失 | −.006608 [−.007908, −.005381] | −.004338 [−.004736, −.003943] |
| 跨残基主链clash软损失 Å | +2.8172e−5 [2.5605e−5, 3.0752e−5] | +1.9922e−5 [1.7445e−5, 2.2634e−5] |

因此几何臂改善C–N和C–N–CA，CA–C–N变化仍不确定，clash反而增加；不能用加权总损失掩盖这组得失。其C–N软损失仍为oracle .514618 Å、forecast .571931 Å，远非“链已修好”。NCO相对coordinate的overall退化也不能概括每个horizon：64 ns forecast差值 **−.001627 Å** [−.002988, −.000299]，方向相反。所有逐horizon、geometry−init及coordinate−init区间均保存在[完整汇总](results/hs_joint_20261101/backbone_repair_v2/summary.json)。

H控制仅在init与geometry两头的oracle分支执行，保持同一oracle CA、观察帧、序列及mask，只把H替换为H0、全零或有效残基间置换。geometry的overall NCO分别增加 **.040833** [.035026, .046607]、**.237495** [.203714, .272369]、**.218926 Å** [.185383, .252821]；两头三种干预的各horizon及overall区间均正。这支持所选读出对H的依赖，不能推导S记忆增益、信息上界、重新训练CA-only模型的最优性能或自主forecast中的同等增量。

全部12个head/干预路由的有效CA在推理后及CPU double转换后都逐位不变；另有persistence与真实target，共14路。评分使用CPU float64，空几何支持保留raw零/count0、汇总记null。本次val面板PSF与legacy推断边集合差异为 **0/200窗**；这不消除§4已证实的插入码问题，物理连接仍以显式PSF为准。七项执行均exit0，源码/输入/快照及定标和500步记录全部复验通过。全量回归[670 passed、1 skipped](results/hs_joint_20261101/backbone_repair_v2/full_tests.log)支持实现契约；[独立统计核验](results/hs_joint_20261101/backbone_repair_v2/summary_independent_verification.json)复算300组CI和350组主描述统计，最大数值差3.55e−15。

这些区间是单训练seed、重复使用开发验证集上的**边际探索性区间，未校正多重比较**；跨0不是等效。NCO是逐点误差而非proper ensemble score；oracle改善与小幅forecast迁移都不能证明完整条件分布、Ramachandran、全主链/全原子质量或T1达标。本轮没有新增test或BioEmu2评估。评估SHA-256为 `1c2a2e99c54e1c16f0fc09fad34c9afe09b01e93825271223a41c930422fb762`；summary SHA-256为 `dc58e7bd2e5a2c7815a9f950129e4ffd7164a317eea7eab5a19e72dca8f5cf1a`。

最后，[独立固定CA能力下界诊断](results/hs_joint_20261101/backbone_repair_v2/fixed_ca_capacity.json)用官方模板半径和三角不等式检查“CA过远，任何局部旋转也无法满足C–N软容忍”的充分条件。forecast在1/4/16/64 ns的等蛋白不可行边比例为 **0/.036842/.171930/.465276%**，保守平均损失下界为 **0/.000158/.000434/.001035 Å**；oracle、观察MD与target MD此长距离下界均为0。它只检出少量过远CA边，不能据此解释其余大量读出误差；零下界也不证明可解，因为没有联合处理键角、近距离或steric约束。这是另行登记的描述性诊断，不修改上述主统计协议或通过门槛。

该下界使用预先固定的浮点工程裕量，没有进行形式化区间算术认证。新增12项focused tests单独通过，未计入此前670项全量回归；独立NumPy复算60个汇总值，最大差4.34e−19。[诊断执行与测试索引](results/hs_joint_20261101/backbone_repair_v2/fixed_ca_capacity.evidence.json)保留了首个源码收集错误的失败记录及修复后正式成功记录；首失败发生于协议保存和缓存反序列化之前。

## 6. 局部 frame 搬运：相同500步预算的新读出实验

本节记录22:06 UTC之后完成的实验，保持§5全部冻结工件。[新预注册](docs/BACKBONE_TRANSPORT_PILOT_V1_20261101.md)针对一个明确的输入缺口：当前 simplified IPA 忽略 `R`，当前CA只经两两距离进入网络；最终方向为 `observed_R @ learned_quaternion`。新增[无参数搬运模块](protein_jepa_frame/models/backbone_frame_transport_v1.py)在显式PSF无向图的完整≤2-hop邻域上，以中心残基CA为原点，用float64 proper Kabsch将观察CA方向搬至当前CA方向，再以 `Q @ observed_R` 作为相同head的起点。rank不足2时保留观察frame；没有数值编号推断边、标签依赖或邻域截断。CA和可训练参数量均不变。

先对固定前32个训练窗口执行[零更新诊断](results/hs_joint_20261101/backbone_transport_v1/train32_diagnostic.json)。同一个观察O解析psi、不经过learned residual的seed-only对照，NCO RMSD由2.135425降至1.949361 Å，C–N软损失由.635453降至.179263 Å；共享初始化的NCO则由2.158664降至1.949600 Å。32窗全部保留、无rank fallback。这支持方向先验本身的作用；把旧训练head直接换basis的结果只是迁移诊断，不能当成新训练臂。

随后仅训练一个transport-coordinate臂，使用§5相同初始化、500窗顺序、seed42、AdamW、学习率计划、clip、dropout和实际NCO坐标目标；geometry系数为0。encoder、CA及H/S保持冻结。固定500步latest已[实际完成](results/hs_joint_20261101/backbone_transport_v1/train_run.execution.json)，36.30秒、exit0，checkpoint SHA为 `f4fded7c56cdbea1e3ef327b94f9ea855448f72126adae2d92aa9a75a2b75954`。无验证选点、权重搜索或追加训练。

[正式开发评估](results/hs_joint_20261101/backbone_transport_v1/evaluation200_v2.json)覆盖相同200窗、50蛋白、每窗全部12个forecast；oracle/forecast分别使用自己的当前CA搬运，target atoms/mask仅用于评分。以下仍为逐成员NCO RMSD、蛋白内窗口及蛋白等权，单位Å；不能与§1四原子RMSD混用。

| 路径 / NCO RMSD ↓ | 1 ns | 4 ns | 16 ns | 64 ns | Overall |
|---|---:|---:|---:|---:|---:|
| observed-coordinate500 / oracle | .984673 | 1.212069 | 1.827931 | 2.623174 | 1.661962 |
| transport-init / oracle | .896601 | 1.085209 | 1.685806 | 2.449226 | 1.529211 |
| transport-coordinate500 / oracle | .790093 | .990751 | 1.608914 | 2.379299 | 1.442264 |
| observed-coordinate500 / forecast | 2.708793 | 3.711108 | 5.287308 | 6.823374 | 4.632646 |
| transport-init / forecast | 2.707269 | 3.669830 | 5.228691 | 6.734833 | 4.585156 |
| transport-coordinate500 / forecast | 2.690341 | 3.667400 | 5.217981 | 6.726691 | 4.575603 |
| 真实主链 persistence（描述） | 2.198955 | 2.700986 | 3.938316 | 5.117159 | 3.488854 |

主对照 **transport-coordinate500−observed-coordinate500，overall Δ [95% CI]**：

| 指标 | Oracle | Forecast |
|---|---|---|
| NCO RMSD Å | −.219698 [−.248020, −.192619] | −.057042 [−.064241, −.050141] |
| C–N软损失 Å | −.410199 [−.488172, −.337044] | −.340631 [−.399343, −.285181] |
| CA–C–N cosine软损失 | −.072288 [−.089186, −.056555] | −.062677 [−.075214, −.050863] |
| C–N–CA cosine软损失 | −.069518 [−.087401, −.052951] | −.064815 [−.079239, −.051358] |
| 跨残基主链clash软损失 Å | −3.0924e−5 [−5.2560e−5, −1.2065e−5] | +5.3143e−6 [−7.1580e−6, 1.7792e−5] |

主对照oracle和forecast的四个horizon NCO区间均负；forecast的1/4/16/64 ns差值分别为−.018452/−.043707/−.069327/−.096683 Å。预测C–N软损失从.600562降至.259931 Å，两个键角软损失也改善，但预测clash变化的区间跨0，不能称为碰撞改善或等效。新forecast NCO仍在每个horizon差于persistence，C–N也远未归零。

零更新transport-init相对observed-init的overall NCO改善为oracle−.193769 Å、forecast−.052514 Å，说明相当一部分收益来自几何先验。再训练500步相对transport-init额外改善oracle−.086947 [−.098667,−.075445]、forecast−.009552 [−.012784,−.006140] Å；同时C–N软损失分别增加.001292 [.000300,.002318]与.002545 [.001306,.003660] Å。不能把先验的全部收益归因于新学习，也不能用坐标改善掩盖训练后的连接项得失。

正式面板的18,612个oracle、223,344个forecast有效残基成员均无需fallback。最大旋转正交误差分别为1.22e−7与1.32e−7；CA在所有推理和评分路由逐位固定。当前局部CA相对刚性拟合仍有残差：按残基成员汇总的平均fit RMSD分别为.825476和1.183252 Å，故搬运并非精确链闭合。重算的800个旧observed control逐窗骨架哈希、坐标误差与几何指标全部严格一致。

首评估在首窗因live约定中的tuple与旧JSON的list直接比较而exit1，未写结果；[失败执行](results/hs_joint_20261101/backbone_transport_v1/evaluation_run.execution.json)与快照保留。新增[序列化修订](docs/BACKBONE_TRANSPORT_EVAL_SERIALIZATION_V2_20261101.md)和[独立评估入口](scripts/eval_backbone_transport_pilot.py)只在严格比较前统一JSON表示，没有放宽数值/哈希容忍或修改科学协议、已训练权重与旧源。[成功评估执行](results/hs_joint_20261101/backbone_transport_v1/evaluation_v2_run.execution.json)为95.60秒、exit0；结果SHA为 `bdc18113bbf35f9b888b3d527bcf255ee8a76b0f3fa0ce95d573858d1f3770ea`。

[独立Python/NumPy复核](results/hs_joint_20261101/backbone_transport_v1/independent_statistics_verification.json)重算150组CI、250项主描述均值/count、23,500个float比较，最大差3.55e−15；另核验800旧control、222项来源/输入/快照/执行文件及500步训练顺序与学习率。[全量回归](results/hs_joint_20261101/backbone_transport_v1/full_tests.log)实际为 **756 passed、1 skipped**。CI仍为重复开发集、单训练seed、10,000次蛋白配对bootstrap的边际探索性区间，未校正多重比较。

### 6.1 训练集上的frame约定与固定CA容量定位

[独立train32 round-trip](results/hs_joint_20261101/backbone_capacity_v3/frame_roundtrip.json)确认native→OpenFold桥接和观察O解析psi符号正确：真实观察R/CA/psi的pooled NCO RMSD约.0881 Å；native与直接OpenFold重建最大坐标差1.53e−5 Å。残余主要符合固定化学模板与MD键振动差异，不能要求精确零误差。这也不同于完整SO(3)/psi族的最优值。旧CA lifter即使改用正确化学身份，C–N均值仍约2.6009 Å；单纯换桥或改法向符号不能完成修复。

新的[独立几何容量模块](protein_jepa_frame/models/stage_c/backbone_capacity_v1.py)在同32训练窗、相同oracle CA下，允许每残基自由选择proper SO(3)与当前36-bin算术均值的psi区间（约±175°）。每个psi的旋转用无平移Procrustes求解；2048个内点给出构造上界，NC精确拟合加O径向松弛、以及O圆轨迹的弦误差界给出理想几何下界。实际OpenFold float32 witness单列；该理想族的界不等于浮点程序的严格界，也不保证一个共享神经网络同时输出各残基最优参数。

[容量结果](results/hs_joint_20261101/backbone_capacity_v3/train32_capacity.json)的等蛋白NCO RMSD最优区间为 **[1.400487,1.402901] Å**，实际float32 witness为1.402901 Å；旧coordinate500为2.099965 Å。[固定最终transport500的同窗补充](results/hs_joint_20261101/backbone_transport_v1/train32_final_capacity.json)为 **1.860089 Å**，相对理想族的误差差距为 **[.457188,.459602] Å**，旧差距为[.697064,.699478] Å。这是同一组训练样本上的定位，不能解释为验证增益或H信息不足。

逐残基psi网格最优构造witness仍有C–N软损失.226391 Å，说明这种独立坐标拟合不保证联合链几何；连续族最优值仍有非零区间。horizon分组各含不同8个训练蛋白，其理想误差差异不能当成纯时间效应。容量诊断15项测试、round-trip 7项测试及[独立容量复核](results/hs_joint_20261101/backbone_capacity_v3/train32_capacity.independent_verification.json)均通过，后者重算20,091值最大差9.09e−13；前者15项已包含上面的756项全量回归，round-trip测试另存于结果目录。

本轮取得了可复核的骨架读出改进，但没有改变七项CA级BioEmu2指标、长时H门槛或H/S动力学。此前已有endpoint/path proper-score及部分静态任务评估；完整T1立体化学、proper ensemble分数、T2A/MSM、T2B、温度泛化上的全面优势与独立确认仍未建立，匹配的S记忆对照仍缺失，本轮没有新增这些方面的证据。没有读取test或重跑BioEmu2，整体目标保持未达成。

## 7. Ramachandran评分完成：搬运改善，但模型仍远差于真实MD

新增[参考评分模块](evaluation/ramachandran_reference_v1.py)及[固定数据与来源记录](evaluation/reference_data/ramachandran_top8000_v1/PROVENANCE.json)，使用官方Top8000/cctbx的general、Gly、cis-Pro、trans-Pro、pre-Pro和Ile/Val六类表。采用cctbx实际编译表的精度与周期双线性插值顺序；原始文本更高精度不能替代官方编译数值。[上游资料](https://github.com/rlabduke/reference_data/tree/master/Top8000/Top8000_ramachandran_pct_contour_grids)与[官方分类实现](https://github.com/cctbx/cctbx_project/blob/master/mmtbx/validation/ramalyze.py)的版本、内容哈希和许可证均已随本地数据固定。

Favored阈值为.02；allowed下界分别为general .0005、cis-Pro .002、其余.001。`allowed_only`明确排除favored，`non_outlier=favored+allowed_only`，前者不能按“越高越好”解释。化学分类优先级为Gly、按前一肽键omega判定的Pro、pre-Pro、Ile/Val、general；只有−90°<omega<90°属于cis。缺失/退化的Pro omega在本模块排除并计数，与上游默认为trans的fallback不同，不能隐去这一实现差异。

输入使用真实化学aatype和显式PSF的局部有向C_i→N_j边，geometry row ID只标记残基身份，不用作边的下标。逐样本保存可评分分母、六类、互斥排除原因、重叠缺失诊断，以及favored/allowed-only/non-outlier/outlier的计数和比例。无支持时比例为null，不作零违例；终端或缺邻接导致的未评分也必须与结构质量同时报告。

30项模块合成测试加25项[官方C++一致性测试](tests/test_ramachandran_official_conformance_v1.py)共 **55 passed**。后者编译固定版本的原始三个上游头文件，仅提供数组存储及断言适配；194,400个表格中心逐位一致，另核验1,105,452组角度及96个阈值边界查询，覆盖周期接缝、nextafter边界及随机角度，分类均一致。独立Bio.PDB合成几何还核对了phi/psi/omega符号。[实际执行](results/hs_joint_20261101/t1_reference_audit_v1/official_conformance.execution.json)exit0，源文件执行前后哈希一致；模块SHA-256为 `a036f9451c9776c3c2fd2d613b0a445b1f70c8e23d5c1eca163888e2e8995bf1`。

以上参考实现测试之后，按[两阶段评估协议](results/hs_joint_20261101/t1_reference_eval200_v1/protocol.json)完成实际模型评分：沿用冻结的200窗缓存、三个读出checkpoint和10条已有路由，先在原GPU后端重建并严格核对全部2000项旧骨架/固定CA哈希，再退出GPU，由CPU评分和统计；不重采H/S、训练或读取新MD窗口。15项runner测试及独立代码审查通过，登记前已核对200窗缓存拓扑和600项既有张量指纹；[登记上下文](results/hs_joint_20261101/t1_reference_eval200_v1/registration_context.json)明确区分这次既有缓存预检与登记之后的新评分。

四个固定contrasts、oracle/forecast、overall/四horizon/五温度与四个推断指标共320个配对蛋白区间，仍为开发集上的边际探索性95%区间；allowed-only只描述，不按单调质量排序。缺失任一成员分母会使对应窗比例为null，缺窗又使完整面板均值/区间为null，可用子集另列；scoring coverage另报且不能当作质量通过率。动态cis/trans类别的残基成员可随预测改变，因此分类型结果不作固定成员的因果配对解释。

主实验完成后，[GPU重建阶段](results/hs_joint_20261101/t1_reference_eval200_v1/decode.execution.json)于2026-11-02 00:10:42–00:11:17 UTC真实exit0，耗时35.02秒，全部2000条路由的骨架及固定CA哈希一致。保存的10,800个样本骨架SHA-256为 `5494664bfba3c401b446c7c6a48944611900df174dba1f798d1cf3d68403d90a`。[CPU评分阶段](results/hs_joint_20261101/t1_reference_eval200_v1/score.execution.json)于00:18:09 UTC真实exit0，耗时399.45秒；[完整评分](results/hs_joint_20261101/t1_reference_eval200_v1/result.json)SHA-256为 `4159380fc84fec484a620c0582d741dcb5f09a1c72d9a5955271af650d0b7b9a`。

[独立NumPy核验](results/hs_joint_20261101/t1_reference_eval200_v1/independent_verification.json)随后真实exit0，从10,800个成员计数重新聚合全部比例、描述均值及320个配对区间；390,028个数值比较最大差0，242个源/输入/快照核对通过，旧几何字段逐值保留。核验没有重新解码或计算二面角；其几何输入身份依赖已通过的重建哈希与冻结sidecar。结果SHA-256为 `f54bd17fbd42c136e465501f70801578943b0a2ec15060b2f37f1eb58e7163c0`。

下表为先成员、后窗口、再等蛋白的比例均值（%），不是将所有残基混池。Oracle用真实未来H/CA；forecast用原冻结动力学的12个预测成员。`init`为训练前读出，`coordinate`为此前固定500步坐标训练；`observed`/`transport`分别表示原观察frame与局部frame搬运。

| 路由 | Favored ↑ (%) | Allowed-only (%) | Non-outlier ↑ (%) | Outlier ↓ (%) |
|---|---:|---:|---:|---:|
| init observed / oracle | 46.903 | 20.316 | 67.219 | 32.781 |
| init transport / oracle | 68.095 | 15.686 | 83.781 | 16.219 |
| coordinate observed / oracle | 45.959 | 20.316 | 66.275 | 33.725 |
| coordinate transport / oracle | 67.767 | 16.199 | 83.965 | 16.035 |
| init observed / forecast | 38.587 | 23.809 | 62.396 | 37.604 |
| init transport / forecast | 45.242 | 22.288 | 67.531 | 32.469 |
| coordinate observed / forecast | 37.323 | 24.120 | 61.443 | 38.557 |
| coordinate transport / forecast | 43.784 | 23.175 | 66.959 | 33.041 |
| Persistence真实MD骨架 | 90.031 | 8.476 | 98.507 | 1.493 |
| Target真实MD骨架 | 90.016 | 8.295 | 98.310 | 1.690 |

所有10条路由、全部200窗均有评分支持，完整面板的null区间为0；各路由等蛋白scoring coverage均为97.822%，差为0。每条oracle/MD路由共18,612个输入残基、18,212个可评分残基；每条forecast路由为223,344/218,544。排除项仅为每个骨架的两端缺邻接：每条oracle/MD路由400个、forecast路由4,800个；未知类型、缺失原子和退化二面角排除均为0。总计1,005,048个输入残基、983,448个可评分残基。这次改善没有来自减少评分支持；完整六类计数仍保留，cis/trans-Pro成员随预测变化，不能混作固定残基类别配对效应。

| 固定比较（候选−参考） | Oracle outlier差及95% CI（百分点） | Forecast outlier差及95% CI（百分点） |
|---|---|---|
| coordinate transport − coordinate observed | −17.690 [−19.834,−15.621] | −5.517 [−6.058,−4.945] |
| init transport − init observed | −16.563 [−18.621,−14.616] | −5.135 [−5.659,−4.587] |
| coordinate transport − init transport | −.184 [−.475,+.094] | +.571 [+.384,+.768] |
| coordinate observed − init observed | +.943 [+.535,+1.343] | +.953 [+.736,+1.179] |

搬运在训练前后均改善Ramachandran比例；但坐标训练对transport的oracle outlier改善区间跨0，对forecast反而增加.571个百分点，favored下降1.458个百分点（95% CI [−1.688,−1.234]）。原observed框架的坐标训练也增加oracle/forecast outlier。因此不能将此前NCO误差降低等同于立体化学整体改善，或只报搬运的正向比较。

| Horizon | coordinate observed forecast outlier (%) | coordinate transport forecast outlier (%) | Target MD (%) | Transport−observed，95% CI（百分点） |
|---|---:|---:|---:|---|
| 1 ns | 27.265 | 20.112 | 1.563 | −7.154 [−7.708,−6.583] |
| 4 ns | 39.215 | 33.411 | 1.623 | −5.804 [−6.468,−5.110] |
| 16 ns | 42.388 | 37.204 | 1.629 | −5.184 [−6.101,−4.233] |
| 64 ns | 45.362 | 41.435 | 1.943 | −3.926 [−4.947,−2.832] |

完整五温度、四horizon、favored/non-outlier/outlier/coverage和所有四contrasts均见评分JSON。统计仍是单seed、反复使用开发集、未校正多重比较的边际探索性结果；不据这次新指标另选checkpoint。当前forecast outlier虽降低，仍远高于真实MD，64 ns差距尤其大。已有跨残基N/CA/C/O clash软损失和计数继续沿用旧显式拓扑实现，其范围不等于全原子MolProbity评分；完整匹配BioEmu2的T1仍未完成。

## 8. 前瞻性代码审查：下一项T1机制问题（执行前记录）

当前最明确的剩余缺陷仍是实际backbone Ramachandran：transport-coordinate500的forecast outlier为**33.041%**，target MD为**1.690%**；h64分别为**41.435% / 1.943%**，评分覆盖一致。此前NCO和C–N软损失改善并不等同完整T1修复，坐标训练相对transport-init还增加forecast outlier **.571个百分点**；上述负结果保留。

代码审查指出，[固定CA读出头](protein_jepa_frame/models/backbone_readout_v2.py#L7)的psi输出只控制O；[实际phi/psi评分](evaluation/ramachandran_reference_v1.py#L273)使用PSF邻接的N/CA/C原子，不包含O。因此在frames固定时，仅改变psi logits（包括改变角度bin读出方式）不能改变这些被评分的backbone二面角。需要检查的是相邻残基frame的联合方向约束；[局部frame搬运](protein_jepa_frame/models/backbone_frame_transport_v1.py#L9)本身并不保证肽链闭合。这不排除训练共享trunk时psi监督间接影响frame，但不能把这种间接作用与O的直接几何自由度混同。

前述独立逐残基SO(3)/psi容量构造仍有C–N软损失.226391 Å；[容量族明确未约束跨残基闭合、键角或clash](protein_jepa_frame/models/stage_c/backbone_capacity_v1.py#L265)。而过远CA充分条件在h64 forecast仅检出.465276%的边、平均损失下界.001035 Å，不能解释其余全部误差；零下界也不证明联合几何可行。这些已有结果都没有证明当前CA使Rama缺陷不可避免，无需重复旧frame桥接、独立逐残基拟合或原observed-frame geometry500实验。

**下一项待检验问题**是：在同一固定TRAIN缓存上，保持CA字节、显式PSF边、mask及O的psi不变，仅对transport500的相邻残基frames作预先限定预算的联合旋转修正，能否降低实际Rama outlier和肽键/键角误差，同时不牺牲NCO误差或增加主链clash？应保留未修正对照，并加入使用真实MD CA、在修正过程中固定该CA的控制；全部支持、失败和几何分项都报告。若得到正向构造，只说明该输入下存在局部改进方向；局部求解失败不能认证不可行。本节保留执行前的问题与边界；随后登记并完成的TRAIN oracle见第9节。**没有采用预测修复，也没有新泛化或完整T1通过结论**。


## 9. TRAIN32联合frame oracle：存在局部改进，未完成T1修复

固定TRAIN缓存前32个蛋白（h1/4/16/64各8个）的目标条件联合旋转诊断已完成。解码oracle CA下，等蛋白NCO RMSD从 **1.860089降至1.618759 Å**，Rama outlier从 **17.9991%降至12.3968%**；真实MD CA控制下分别从 **.578069降至.282617 Å**、**14.1361%降至5.1182%**，真实MD为 **1.6059%**。两种条件的全部32窗NCO都改善，但仍有逐窗几何退化，h64解码CA的omega偏离还增加。这是利用真实目标坐标和phi/psi的局部构造，不是预测修复、共享新head、H信息上限或T1通过，也未新增BioEmu2比较。

[预登记协议](results/hs_joint_20261101/train_joint_frames_v1/protocol.json)与[实现说明](docs/TRAIN_JOINT_FRAME_ORACLE_V1_20261102.md)固定了五个路由：transport500在解码oracle CA和真实MD CA各自的原始/联合修正骨架，以及真实MD。两种CA条件使用同一缓存未来oracle H，但分别重新运行冻结head，因此条件之间的差异并非纯CA因果效应。每种条件从零旋转开始，用float64局部刚体原子族同时优化相邻frame；CA字节和初始psi参数固定，O仍随frame旋转。它不是重新调用OpenFold读出的逐位等价承诺。

预算预先固定为200步Adam、lr=.02、wd=0、梯度clip=1、每残基SO(3)旋转角不超过30°，只报告最终步，不挑最佳步、重启或调整权重。目标为NCO均方欧氏误差、参考phi/psi周期误差各权重1，以及冻结几何的C–N/两键角/clash权重1/4/4/100；Å量按固定1 Å单位归一化。评分支持固定为 `row.atom_mask & row.target_atom_mask & row.observed_mask[:,None]`，拓扑来自显式PSF边，损失的参考有效torsion支持在优化前固定，预测退化不缩小分母。

周期项使用预先固定的连续几何退化惩罚：充分非退化时为1−cos角差，confidence为0时为2；这里的confidence仅为几何归一化因子，不是模型置信度。严格无效阈值1e−10 Å只用于诊断/官方评分；非零但低于阈值的预测可略小于2，不能声称所有strict-invalid都恰好取最大惩罚。本次全部轨迹的固定支持上strict-invalid数为0。精确塌缩和反向角的平坦梯度仍限制局部求解；两种CA条件各有 **31/32** 窗最终最大旋转到达30°边界，不能据残余误差认证全局不可行。

下表均先在蛋白内聚合、再等权平均32个蛋白，不混池残基；NCO、各原子和四原子误差使用固定坐标gauge，没有为其另做Kabsch。另存的shape-only指标仍按冻结约定独立对齐。CA误差在各条件修正前后完全一致。

| 路由 | NCO RMSD Å | N Å | CA Å | C Å | O Å | 四原子 RMSD Å |
|---|---|---|---|---|---|---|
| 解码CA·原始 | 1.860089 | 1.746322 | 1.660435 | 1.780428 | 2.020676 | 1.818213 |
| 解码CA·联合修正 | 1.618759 | 1.644786 | 1.660435 | 1.567068 | 1.634349 | 1.630007 |
| 真实CA·原始 | 0.578069 | 0.426808 | 0.000000 | 0.452088 | 0.782597 | 0.500623 |
| 真实CA·联合修正 | 0.282617 | 0.229192 | 0.000000 | 0.213784 | 0.369843 | 0.244753 |
| 真实MD | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

PSF几何软损失使用冻结的宽平底容差（键长/键角12 SD、clash容差1.5 Å），其零值不代表完整立体化学通过。两角软损失为无量纲余弦差，实际角度另列，实际角度本身不按越低越好排序。omega列为每条有效肽键的 `180−abs(omega)` 后等蛋白平均；合法cis也会增大该数值，因此不能把它直接当作违例率，cis/trans计数同时保留。

| 路由 | C–N软损失 Å | CA–C–N软损失 | C–N–CA软损失 | clash软损失 Å | 实际C–N Å | 实际两角 ° | ω偏离trans ° |
|---|---|---|---|---|---|---|---|
| 解码CA·原始 | 0.177694 | 0.021250 | 0.015930 | 0.000202 | 1.511243 | 116.4013 / 121.4020 | 15.530153 |
| 解码CA·联合修正 | 0.084994 | 0.009283 | 0.007898 | 0.000164 | 1.439330 | 117.3777 / 122.7950 | 15.363483 |
| 真实CA·原始 | 0.133234 | 0.011898 | 0.011984 | 0.000050 | 1.461735 | 117.7823 / 122.3698 | 15.239496 |
| 真实CA·联合修正 | 0.015212 | 0.000535 | 0.001501 | 0.000007 | 1.349533 | 118.9576 / 124.3722 | 11.362863 |
| 真实MD | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.348300 | 117.7329 / 123.5344 | 9.139908 |

Rama比例采用同一官方六类表；A为allowed-only，不能按越高越好解释。比例为等蛋白均值，而F/A/O和cis/trans为实际计数，两种聚合不能互换。

| 路由 | Favored % | Allowed-only % | Non-outlier % | Outlier % | F/A/O计数 | cis/trans ω计数 |
|---|---|---|---|---|---|---|
| 解码CA·原始 | 66.5240 | 15.4769 | 82.0009 | 17.9991 | 1836/420/489 | 41/2736 |
| 解码CA·联合修正 | 75.4499 | 12.1533 | 87.6032 | 12.3968 | 2080/333/332 | 36/2741 |
| 真实CA·原始 | 71.6184 | 14.2455 | 85.8639 | 14.1361 | 1970/389/386 | 36/2741 |
| 真实CA·联合修正 | 84.7832 | 10.0986 | 94.8818 | 5.1182 | 2322/281/142 | 22/2755 |
| 真实MD | 89.5513 | 8.8428 | 98.3941 | 1.6059 | 2458/243/44 | 17/2760 |

所有五个路由各有2,809个输入残基、2,745个Rama可评分残基、64个排除（均为两端拓扑邻接不完整）；等蛋白coverage为97.6643%。masked、未知类型、缺原子、退化torsion等其他互斥排除均为0。每路由有2,777条显式PSF肽键，C–N、两角、omega均有效，masked/degenerate计数均为0。六类F/A/O计数完整如下；cis/trans-Pro类别成员会随预测omega改变，不能作为固定成员的类别因果效应。

| Rama类别（F/A/O） | 解码CA·原始 | 解码CA·联合修正 | 真实CA·原始 | 真实CA·联合修正 | 真实MD |
|---|---|---|---|---|---|
| general | 1348/295/339 | 1508/238/236 | 1436/272/274 | 1689/198/95 | 1788/168/26 |
| glycine | 132/35/38 | 147/31/27 | 141/36/28 | 166/21/18 | 183/17/5 |
| cis-proline | 1/1/3 | 4/1/3 | 2/0/3 | 4/1/1 | 4/1/2 |
| trans-proline | 53/24/38 | 70/21/21 | 63/22/30 | 85/17/12 | 80/26/7 |
| pre-proline | 63/19/25 | 80/15/12 | 73/16/18 | 85/17/5 | 95/10/2 |
| isoleucine or valine | 239/46/46 | 271/27/33 | 255/43/33 | 293/27/11 | 308/21/2 |

全部horizon及五个路由保留如下，每格8个不同蛋白。horizon之间不能作为同一蛋白随时间变化的因果曲线，也不生成跨horizon推断、CI或挑选阈值。

| Horizon ns | 路由 | NCO Å | Rama outlier % | C–N软损失 Å | clash软损失 Å | ω偏离trans ° |
|---|---|---|---|---|---|---|
| 1 | 解码CA·原始 | 0.595685 | 8.985204 | 0.064309 | 0.000026 | 11.702200 |
| 1 | 解码CA·联合修正 | 0.371809 | 2.505773 | 0.003110 | 0.000002 | 11.394313 |
| 1 | 真实CA·原始 | 0.445094 | 7.727216 | 0.061180 | 0.000025 | 11.739390 |
| 1 | 真实CA·联合修正 | 0.201851 | 1.577390 | 0.000798 | 0.000001 | 9.346253 |
| 1 | 真实MD | 0.000000 | 1.444411 | 0.000000 | 0.000000 | 8.929496 |
| 4 | 解码CA·原始 | 0.585622 | 8.991878 | 0.071671 | 0.000023 | 13.281587 |
| 4 | 解码CA·联合修正 | 0.384833 | 3.702188 | 0.005280 | 0.000001 | 12.980125 |
| 4 | 真实CA·原始 | 0.446736 | 8.387040 | 0.068627 | 0.000017 | 12.904611 |
| 4 | 真实CA·联合修正 | 0.202883 | 2.616444 | 0.003488 | 0.000002 | 11.615281 |
| 4 | 真实MD | 0.000000 | 1.093779 | 0.000000 | 0.000000 | 10.574198 |
| 16 | 解码CA·原始 | 1.013449 | 20.107223 | 0.172415 | 0.000091 | 14.755333 |
| 16 | 解码CA·联合修正 | 0.768787 | 13.134265 | 0.038149 | 0.000028 | 13.438785 |
| 16 | 真实CA·原始 | 0.602931 | 16.986061 | 0.145665 | 0.000064 | 15.255799 |
| 16 | 真实CA·联合修正 | 0.286687 | 6.288269 | 0.015136 | 0.000007 | 10.691283 |
| 16 | 真实MD | 0.000000 | 2.040061 | 0.000000 | 0.000000 | 8.489046 |
| 64 | 解码CA·原始 | 5.245602 | 33.911963 | 0.402381 | 0.000670 | 22.381489 |
| 64 | 解码CA·联合修正 | 4.949606 | 30.244985 | 0.293437 | 0.000623 | 23.640710 |
| 64 | 真实CA·原始 | 0.817517 | 23.444229 | 0.257463 | 0.000094 | 21.058186 |
| 64 | 真实CA·联合修正 | 0.439046 | 9.990672 | 0.041426 | 0.000018 | 13.798634 |
| 64 | 真实MD | 0.000000 | 1.845211 | 0.000000 | 0.000000 | 8.566894 |

逐窗负向结果也保留：解码CA条件的Rama outlier为27窗降低、2窗持平、3窗升高；C–N软损失有1窗升高，C–N–CA角软损失有1窗升高（26降、5平），clash有1窗升高（30降、1平），N原子误差有2窗升高。真实CA条件Rama为31降、1平，C–N和clash均无升高。omega偏离trans在解码CA条件有14窗升高、真实CA有2窗升高；尤其h64解码CA虽然NCO由5.245602降至4.949606 Å、Rama由33.911963%降至30.244985%，omega偏离仍由22.381489°升至23.640710°。因此不能将局部NCO或平均Rama改善等同于各项几何同时修复。

![全部32个训练窗口的配对结果；两列CA条件、两行NCO/Rama，颜色为horizon](results/hs_joint_20261101/train_joint_frames_report_v1/paired_scatter.png)

图中对角线以下为该指标降低，NCO使用log轴，Rama保留线性零点；所有32窗都显示。可导出[PDF](results/hs_joint_20261101/train_joint_frames_report_v1/paired_scatter.pdf)，并查看[逐点CSV](results/hs_joint_20261101/train_joint_frames_report_v1/paired_points.csv)、[完整描述聚合及逐窗方向计数](results/hs_joint_20261101/train_joint_frames_report_v1/descriptive_summary.json)、[仅从冻结JSON生成图表的脚本](results/hs_joint_20261101/train_joint_frames_report_v1/render_report.py)和[真实绘图执行与输入/输出哈希](results/hs_joint_20261101/train_joint_frames_report_v1/execution.json)。图为结果后的描述性汇总，没有额外优化或选择。

[实际执行](results/hs_joint_20261101/train_joint_frames_v1/execution.json)于2026-11-02 08:36:19–08:39:00 UTC完成，PID 2233860，161.07秒，exit0；32窗、64个条件优化和160个路由评分全部成功，head参数与RNG前后身份一致，head optimizer更新为0。协议SHA-256为 `b95f2e55b5889ea7b466cdfcc5b49038bdecd7173998500ef1bb4abec541a31f`；[完整结果](results/hs_joint_20261101/train_joint_frames_v1/result.json)SHA-256为 `9602489b45ce378c284c25b35e616b8beaadd0cac1b1262136dd4d6b597c149d`。

结果完成后的[独立机械核验](results/hs_joint_20261101/train_joint_frames_verification_v1/audit_output.json)读取全部64个保存sidecar，用独立NumPy/标准库公式复算初始/最终原子族、旋转、NCO/atom误差、PSF几何及目标项，并检查12,864个标量trace条目的代数。它核对160组NCO/坐标分数、640个atom RMSD、13,885条边的C–N/两角/omega等；最大键角差8.53e−14°、原子族差2.84e−14 Å，未发现不一致。中间坐标未保存，因此201项trace是标量记录，不是逐步坐标重算。该审计不独立重建head或RNG，官方Rama表也未由它重写。审计输出SHA-256为 `3b56331c5cc63a090eec2259e6f71f7486351eca280a6c043c8979031ee7e696`。

另一次[冻结官方Rama scorer重算](results/hs_joint_20261101/train_joint_frames_official_rama_recheck_v1/audit_output.json)从保存骨架重算全部160个路由、比较47,563个浮点值，最大差0。它是同一官方实现的重放，**不是独立Rama表实现**。这两项都是结果后的核验，不能改称前瞻登记的独立实验；各自[sidecar审计来源](results/hs_joint_20261101/train_joint_frames_verification_v1/provenance.json)和[Rama重算来源](results/hs_joint_20261101/train_joint_frames_official_rama_recheck_v1/provenance.json)明确原始tool session及原PID/起止时间未保存为null，后来的归档时间另列，没有补造执行记录。Rama重算输出SHA-256为 `1d707440963fc15b807eaaf10d074d9ef750c638dd2c254d614744f671b937c2`。

这个有真实未来目标参与的构造说明：在限定局部刚体族和预算内，同一固定CA下存在NCO与平均Rama的联合改进；仍不足以证明一个共享读出头能学到它，也不能据两CA条件差异推断CA的单独作用。它不改变预测动力学、现有评估或BioEmu2比较，完整匹配T1和整体目标仍未达成。

## 10. 冻结H距离读出：metadata审计与首次probe失败记录

新诊断预定用相同的缓存observed/aligned-target骨架重新编码原始Stage A和当前冻结H，再各拟合相同容量、相同初始化的距离读出。固定TRAIN500缓存前400蛋白用于拟合、后100只对新读出留出，500步顺序为前400加前100；各horizon留出25蛋白。比较未来H输入、同一head的observed-H替换以及observed-CA距离持久性。它测量有限模型和预算下的结构可读性，编码器可能见过全部500蛋白，不是JEPA因果效应、H信息上限或真实未来预测。

在任何缓存张量或编码器加载前，必须根据这500个蛋白自己的完整PSF肽键图验证pair支持，不能复用距离试验的另一组471个结构。[v1预审计](results/hs_joint_20261101/frozen_h_distance_support_v1/protocol.json)于09:23:31.778 UTC登记，SHA-256为 `3d5ee0296fec987587a9c51f1598f69b3444473cb1fb3316721e43454d412748`。44项合成测试和两次独立审阅当时均通过，但实际数据暴露了测试未覆盖的序列化表示差异。

[实际执行](results/hs_joint_20261101/frozen_h_distance_support_v1/execution.json)PID2295492于09:25:26.135–09:29:29.938 UTC运行243.795秒，exit1。全部500条记录均被访问、均报 `full metadata differs from historical cache topology`，成功plan为0；[失败support](results/hs_joint_20261101/frozen_h_distance_support_v1/support.json)完整保留500条错误，SHA-256为 `db270c9de07db9c8fdeb00a917989e8aa0c52c30c17ab1397d31fc71b5cc8cef`，execution SHA为 `82ba454b619eccbf7ac41c0df8016b9f0882b237b99ea608d3832507d27211c8`。本次没有加载轨迹坐标、缓存张量、模型、validation或test，也没有训练读出或登记正式probe。

源码和冻结JSON复核定位到 `CHEMICAL_ALIAS_POLICY.histidine_aliases`：现场拓扑保留Python tuple，历史JSON载入后是list。两者规范化JSON哈希相同，但直接Python dict相等比较为false，因此全部窗口在比较处失败；随后缓存topology与JSON plan直接比较也会遇到同一问题。旧distance support中与本panel重叠的39个结构，在JSON表示下的身份（只剥离新raw-identity审计字段）和所有metadata hashes均相同。这个证据支持序列化表示错误，不能把失败预审计改称成功。

v1源码、协议、快照和失败产物保持冻结。另立v2统一元数据的规范化JSON表示，要求全部字段、顺序、值及历史拓扑SHA仍精确一致，并加入真实alias常量、JSON往返和缓存tuple/plan-list的回归测试；修正不得更改UID、pair政策、旧embedding IDs、窗口或数据。只有新版本完整500记录通过后，才可登记和运行正式H读出诊断。

### v2 metadata成功，正式读出已登记（09:45 UTC）

[v2源码与说明](docs/FROZEN_H_DISTANCE_SUPPORT_V2_20261102.md)修复两个序列化比较边界，保留所有原始字段和值、序列顺序及精确历史topology SHA；真实值变化和不支持的类型仍失败。55项v2合成测试经作者、独立审阅和根侧复测通过，另55项冻结物理support测试由作者复测通过。v1全部11份注册源文件哈希再次核对未变，v2将v1失败protocol/support/execution作为固定输入保存。

[v2元数据协议](results/hs_joint_20261101/frozen_h_distance_support_v2/protocol.json)于09:39:18.692 UTC登记，SHA-256为 `97043b675941b2020e00123f185d1225ce914fb8d006405b7480cfc1f3f8f138`。[实际运行](results/hs_joint_20261101/frozen_h_distance_support_v2/execution.json)PID2316279于09:39:47.130–09:42:20.687 UTC完成153.465秒、exit0；500个不同蛋白全部成功，错误为0，没有替换窗口。与v1相同的窗口序列SHA为 `9faa8ab701250356eee3fa8459a53771edfc78a5e20e4204af4017c5cbc9f5cb`。

[完整support](results/hs_joint_20261101/frozen_h_distance_support_v2/support.json)为203536354 bytes，SHA-256 `262a9de49b9de46a514f165201c52681db6e56b936a6cd8cc20f4fb4a5fad1a3`；execution SHA为 `6bc9218fd7566eabe9d0b82946a3340cfa741863b238ecc34673188a89b02732`。mask前每蛋白候选pair为1128–4371对。根侧另外实际调用load_support，复验全部500个plan、历史拓扑哈希、输入/源码/快照/HDF stats及执行绑定后才固定该support SHA。此时仍未反序列化缓存或编码器，坐标有限性及实际observed支持留待正式runtime检查。

[正式读出协议](results/hs_joint_20261101/frozen_h_distance_probe_v1/protocol.json)随后于09:45:10.296 UTC登记成功，SHA-256为 `c64575df00410f8c4d2386438d5d13212597d6578219545b0ebf2c0c01b2e257`，固定57份源码和9份输入。纯encoder/readout40项、runner22项测试均通过独立/根侧复测。维持预定500步、相同seed42 fresh256-hidden head、全部PSF pairs、无H归一化和latest-only规则；以逐蛋白log-distance MSE为主描述量，ÅRMSE在等蛋白均方误差聚合后开根，原始expm1预测不裁剪并单列负值计数。协议登记不代表训练或评分已成功，尚无本诊断的可读性结论。

### 首次probe在当前编码器配置读取处失败（09:48 UTC）

[独立复算协议](results/hs_joint_20261101/frozen_h_distance_probe_verification_v1/protocol.json)在probe执行前登记，SHA-256为 `8c8a58488c589fb0675cf74df7b41a368723960d0275cca6652192dd5227e472`，43项NumPy/标准库合成测试经独立复测通过。它预定复算100个NPZ、500组逐蛋白分数及25组聚合，不重建encoder或训练梯度。由于下面的实际probe失败，该verifier没有对实际性能运行。

[实际probe](results/hs_joint_20261101/frozen_h_distance_probe_v1/execution.json)PID2329279于09:47:39.826–09:48:31.443 UTC运行51.617秒、exit1；execution SHA为 `86b245407699ce17284285d09ef224afb6b18262f586cfb4e3834a03b90aa200`。全部500行缓存的固定mask、坐标有限性及物理支持检查已通过并保存在all500_support_audit中。原始Stage A组完成500更新并保留latest500；当前组在encoding阶段抛出 `KeyError: 'stage_a_v2_config'`，更新为0，没有进入100个留出蛋白的评分，也没有result.json或性能结论。

原实现错误地假设当前component内嵌Stage A配置，合成fixture也复制了这个错误假设。实际checkpoint的component只有encoder权重及 `calibration_args.stage_a_ckpt` 等来源信息；冻结CAComponentBundle/Calibration代码从该路径的原始Stage A checkpoint构造encoder，再严格加载当前component.encoder。这个来源恰好是已固定SHA的原始Stage A文件，并非新选checkpoint。只读key/config检查没有生成预测。

v1的pure/runner/verifier源码、协议、原始组500步及失败全部保留。新v2将明确绑定上游配置路径与文件SHA，继续只加载当前component的encoder权重，保留预定的权重形状feature-flag规则；不添加默认配置或候选fallback。两组都须在新目录从相同初始head重新拟合，不能续用v1已完成的原始组。H读出的两组可比评分仍未完成。


## 11. 冻结H距离读出完成：当前H更易读出真实结构，未建立未来预测能力

[v2正式结果](results/hs_joint_20261101/frozen_h_distance_probe_v2/result.json)与[已登记的独立复算](results/hs_joint_20261101/frozen_h_distance_probe_verification_v2/verification.json)均实际exit0。固定预算下，当前冻结encoder的target-H总体log-distance MSE为 **0.002610**，原始Stage A为 **0.014400**，四个horizon的组均值都更低；总体MAE/RMSE为 **0.768433/1.667921 Å**，原始组为 **2.016940/3.299109 Å**。这是用**真实未来aligned-target骨架重新编码**后的结构可读性结果，不能将target-H当作kernel预测的未来H。

按预定TRAIN500的不同蛋白顺序，前400拟合、后100只对新readout留出，各horizon留出25蛋白；编码器可能见过全部500蛋白。两臂均先完成observed及aligned-target重新编码，再各自从相同seed42初始化拟合fresh symmetric `[Hi,Hj,masked_H_mean]` 384→256→256→1 head，前400一次加前100第二次，共500次Adam更新，lr=0.001、wd=0、clip=1，CPU float32/2线程、无AMP/TF32、latest500 only。未使用缓存raw-target oracle_H或历史head，没有归一化H、筛pair方差、选择最佳步或额外validation。支持为完整PSF肽键图上距离大于2的全部observed-valid pairs，先完整图排除再crop/mask；五路使用同一未来标签和pair支持。

每蛋白先在其全部pairs上计算log-distance MSE、Å MAE和Å MSE，再对蛋白等权平均；表中RMSE为 **sqrt(等蛋白平均Å MSE)**，不是逐蛋白RMSE均值，也不是所有pairs混合加权。原始float32预测log转float64后expm1，不裁剪负距离；真实标签保留原始float32物理距离转float64，不用expm1(log-target)替换。总体每路386,092 pairs，每蛋白1,128–4,371 pairs；五路负距离数均为0。pair总数是支持计数，不是独立样本数。

| 分组 / 蛋白数 | 输入与读出 | log-distance MSE | MAE Å | RMSE Å | pair总数 | 负距离数 |
|---|---|---:|---:|---:|---:|---:|
| 总体 / 100 | 原始Stage A · target-H | 0.014400 | 2.016940 | 3.299109 | 386092 | 0 |
| 总体 / 100 | 原始Stage A · observed-H替换 | 0.060033 | 3.688484 | 6.745690 | 386092 | 0 |
| 总体 / 100 | 当前冻结encoder · target-H | 0.002610 | 0.768433 | 1.667921 | 386092 | 0 |
| 总体 / 100 | 当前冻结encoder · observed-H替换 | 0.048102 | 2.794299 | 5.571410 | 386092 | 0 |
| 总体 / 100 | observed-CA距离持久性 | 0.047814 | 2.665988 | 5.637213 | 386092 | 0 |

完整horizon分组如下。各行均为25个不同蛋白，horizon之间不是同一批蛋白的时间曲线；未计算CI、显著性或选取获胜horizon。

| 分组 / 蛋白数 | 输入与读出 | log-distance MSE | MAE Å | RMSE Å | pair总数 | 负距离数 |
|---|---|---:|---:|---:|---:|---:|
| h1 ns / 25 | 原始Stage A · target-H | 0.013581 | 1.884844 | 3.245683 | 90335 | 0 |
| h1 ns / 25 | 原始Stage A · observed-H替换 | 0.028778 | 2.565855 | 4.368240 | 90335 | 0 |
| h1 ns / 25 | 当前冻结encoder · target-H | 0.002675 | 0.707948 | 1.922890 | 90335 | 0 |
| h1 ns / 25 | 当前冻结encoder · observed-H替换 | 0.017546 | 1.645724 | 3.254126 | 90335 | 0 |
| h1 ns / 25 | observed-CA距离持久性 | 0.015445 | 1.433705 | 2.804794 | 90335 | 0 |
| h4 ns / 25 | 原始Stage A · target-H | 0.015062 | 2.123148 | 3.455355 | 101458 | 0 |
| h4 ns / 25 | 原始Stage A · observed-H替换 | 0.031627 | 2.903450 | 4.489874 | 101458 | 0 |
| h4 ns / 25 | 当前冻结encoder · target-H | 0.002370 | 0.786992 | 1.331256 | 101458 | 0 |
| h4 ns / 25 | 当前冻结encoder · observed-H替换 | 0.022538 | 1.978274 | 3.323766 | 101458 | 0 |
| h4 ns / 25 | observed-CA距离持久性 | 0.020644 | 1.737693 | 3.093729 | 101458 | 0 |
| h16 ns / 25 | 原始Stage A · target-H | 0.015199 | 2.114005 | 3.446084 | 97981 | 0 |
| h16 ns / 25 | 原始Stage A · observed-H替换 | 0.068410 | 4.044977 | 7.602437 | 97981 | 0 |
| h16 ns / 25 | 当前冻结encoder · target-H | 0.003162 | 0.892819 | 1.913914 | 97981 | 0 |
| h16 ns / 25 | 当前冻结encoder · observed-H替换 | 0.056768 | 3.311902 | 6.549047 | 97981 | 0 |
| h16 ns / 25 | observed-CA距离持久性 | 0.057764 | 3.176301 | 6.725489 | 97981 | 0 |
| h64 ns / 25 | 原始Stage A · target-H | 0.013758 | 1.945761 | 3.031014 | 96318 | 0 |
| h64 ns / 25 | 原始Stage A · observed-H替换 | 0.111317 | 5.239651 | 9.218449 | 96318 | 0 |
| h64 ns / 25 | 当前冻结encoder · target-H | 0.002235 | 0.685972 | 1.412453 | 96318 | 0 |
| h64 ns / 25 | 当前冻结encoder · observed-H替换 | 0.095556 | 4.241295 | 7.722412 | 96318 | 0 |
| h64 ns / 25 | observed-CA距离持久性 | 0.097405 | 4.316254 | 8.027607 | 96318 | 0 |

负向与控制结果不能省略。当前target-H相比原始target-H，逐蛋白log-MSE为98降、2升（1wh4A01/h1：0.037358→0.040274；1vwxT01/h64：0.020309→0.024550），MAE为97降、3升，Å MSE为98降、2升。当前target-H相比同一head的observed-H替换，log-MSE在100/100蛋白降低，但MAE和Å MSE各有1个蛋白升高。上述方向计数是结果后的描述，不能替代不确定性估计。

同一个当前head换成observed-H后，总体log-MSE **0.048102**、MAE **2.794299 Å**，均略差于observed-CA持久性的 **0.047814 / 2.665988 Å**；RMSE则略低（5.571410 vs 5.637213 Å）。逐蛋白log-MSE有82/100差于持久性，MAE有83/100更差，Å MSE有77/100更差。h1/h4三项均差于持久性；h16 log-MSE/RMSE略低但MAE更高；h64三项组均值较低。因此不能从总体RMSE或长horizon均值选出预测胜者。observed-H是用目标H训练的同一head的输入替换控制，不是另行训练的history-only预测器；observed-CA持久性不训练，静态折叠信号可能主导两者。target-H额外获得真实未来坐标，误差差异不识别JEPA的因果贡献，也不是H信息上限或可泛化动力学能力。

![100个readout留出蛋白的两组配对log-distance MSE；颜色为horizon，全部点保留](results/hs_joint_20261101/frozen_h_distance_probe_report_v2/paired_scatter.png)

左图横轴为原始Stage A target-H，纵轴为当前target-H；右图横轴为当前head的observed-H替换，纵轴为当前target-H。两轴均为log刻度，虚线为等误差，每面板显示全部100点。可查看[PDF](results/hs_joint_20261101/frozen_h_distance_probe_report_v2/paired_scatter.pdf)、[500条逐蛋白×角色CSV](results/hs_joint_20261101/frozen_h_distance_probe_report_v2/paired_points.csv)、[完整聚合/方向计数/身份JSON](results/hs_joint_20261101/frozen_h_distance_probe_report_v2/descriptive_summary.json)、[JSON-only生成脚本](results/hs_joint_20261101/frozen_h_distance_probe_report_v2/render_report.py)和[真实绘图执行](results/hs_joint_20261101/frozen_h_distance_probe_report_v2/execution.json)。绘图PID2369735于10:11:38.969–10:11:40.479 UTC完成1.511秒、exit0，仅加载10份已完成的producer/verifier JSON；依赖来自 `/tmp/lasp_readout_plot_deps`，没有读取tensor、checkpoint、cache或HDF。它是结果后的呈现，不是新注册统计，JSON逐行重查聚合最大差1.42e−14；图已视觉检查。report JSON SHA为 `082af4fdd72e6e1ae1ca034c8acb3a5928c856440acb723aa5dee6fab6a6ac90`，CSV SHA为 `39370ef7a107de05d09d69eb60e10850e3aca7c62be5b4a6072c6f8e95fe9d0d`，其余输入、脚本及输出SHA见执行记录。

来源与执行身份如下。原始权重严格取 `model/context_encoder.*`，当前权重严格取 `component_checkpoint.encoder`；两者构造配置均来自已固定原始Stage A文件，当前组件的 `calibration_args.stage_a_ckpt` 和 `component_checkpoint.stage_a_ckpt` 均记录为同一个该路径。原始checkpoint SHA为 `b2de4fed497c62cb196d908aa68a3a0d64a59e8c8179b9a91558797384b78689`，当前checkpoint SHA为 `396e6e4df35019140324484bf8d560a292cf2688a11946f2e0ccde5cd6ea8dda`；这是文件身份，不是同一权重。两个实际128维encoder均为4层/4 heads，orientation与relative-position特征开启，anchor-conditioned关闭，冻结且eval；保存flags和权重结构推断flags一致。

| 身份 | SHA-256 |
|---|---|
| 原始encoder张量，编码前后相同 | `b7af1c028aae96e72e0311c54454d4c21582b1cac2e6b4ef3bda7fc1a5ecf4cd` |
| 当前encoder张量，编码前后相同 | `741d8fbf712981d8aefd4a4841d75ea201663c247dca92396e69f4bc43bbcc27` |
| 两臂共同fresh head初始化 | `3385e66cf009811328d72675885311575fc062a9aa60f7b923eeaf3d0cd43a71` |
| 原始head，500更新后 | `c42abc91c68016d9a2417d9f00add6db0602b21a833e0927c946cfef39577d78` |
| 当前head，500更新后 | `8101d7db6586d89f386674530c5c43bbd39c3cd81142957f793fee082f876850` |

producer记录全程CPU RNG哈希前后相等、评分前后head不变。第10节的v1原始臂500步和当前臂失败仍完整保留并被v2输入哈希固定；v2两臂均fresh重新训练，没有续训、复用旧head或改动科学预算。pure+runner共79项合成测试与verifier74项测试通过根侧及独立复核；测试数不能代替下面的实际执行证据。

另一次结果后的[原始臂v1/v2机械一致性审计](results/hs_joint_20261101/frozen_h_distance_probe_original_parity_v1/audit_output.json)实际PID2367826于10:10:14.289–10:10:14.362 UTC完成、exit0：500行支持、500个observed/target联合feature哈希及500条训练日志全部相同；保存head的6个初始化张量、6个最终张量与18个optimizer张量共30个逐位相等。checkpoint容器字节不同（版本metadata不同），不影响该张量结论。该审计只反序列化两份已保存的原始臂readout checkpoint，没有重放encoder、加载cache或做预注册新统计。审计输出SHA为 `14303e510d55641d37161c2ccfa9f26958d856b3fae332e21b771b022293a795`。v2两encoder虽构造与flags相同，权重SHA不同，500行联合feature哈希也全部不同；联合哈希不能区分observed与target各自是否变化。

[v2协议](results/hs_joint_20261101/frozen_h_distance_probe_v2/protocol.json)登记执行于10:02:34.618 UTC完成，SHA `2a615e12f9b1d1ac656c73c8bf9e9d71dfc1e59ae78dc7b99fca1a455370eaca`。[probe实际执行](results/hs_joint_20261101/frozen_h_distance_probe_v2/execution.json)PID2354935于2026-11-02 10:04:19.846–10:07:03.108 UTC完成163.262秒、exit0；execution SHA `3130bec918d4666d5be3cf570e1ae5b3441e6a7ac8b70a006b50ba3313e9ac03`，result SHA `0d97e1e1a1c0f7e99e1ccb94955105882251f78ee75096426021eece097bd7ac`。

[独立verifier协议](results/hs_joint_20261101/frozen_h_distance_probe_verification_v2/protocol.json)在probe开始前于10:03:22.103 UTC完成登记，SHA `dbfe401dfdfa7300a6cc672626d614357a0ba85a1ab23c4beb534ad129cfcd1b`。[实际独立执行](results/hs_joint_20261101/frozen_h_distance_probe_verification_v2/execution.json)PID2361812于10:07:26.631–10:07:40.307 UTC完成13.676秒、exit0，execution SHA `039671c4382a8c74b263639e41199e7e82ae28642edda9d50ab3fd7dbcaeb022`；verification SHA `910667fb67a07f6ec6fc76d2762128d2cf01d479a4cb5c972e82ca689a566670`。

独立NumPy/标准库实现无producer数值函数导入，重算100份保存预测NPZ、500组逐蛋白分数、25组聚合，核对1,000条训练日志身份与顺序、368份字节哈希和明确的state/config来源；4,329个浮点比较最大差 **1.14e−13**（atol=1e−12、rtol=2e−12，float32 log1p标签对应容许2 ULP）。它不重建encoder、不重放优化器或训练梯度，也不从原始坐标重算pair距离；checkpoint/cache仅哈希，HDF只验登记stat而非全内容哈希。冻结/RNG/同初始化的结论因此是producer保存身份记录及其一致性核验，不能写成独立模型重放。此次成功限定于固定head容量、步数和readout留出panel的真实结构可读性，不构成T1完整骨架修复、BioEmu2全指标胜出或部署采用依据；距离辅助目标试验的性能判断仍须其自己的完整登记评估。

## 12. TRAIN四窗一步H矩诊断：范数预算失配，尚未确定修复方案

固定的四个TRAIN窗口全部完成，独立复算通过。当前核在真实编码H上给出的**一步原始及去通道均值后的平方范数变化均为正**，真实H路径相应变化接近零；同时，四个窗口的预测切向增量二阶矩均低于各自的观测时间均值。这两种差异需要同时保留：不能把结果简化成“噪声整体太大”，也不能据此采用减噪、归一化或删除bath。85个相关transition不是85个独立蛋白；以下只报告四窗各自描述量，无跨窗总体量、CI、阈值或模型采用结论。

使用旧 `ordered_path_v1/marginal_path/checkpoint_latest.pt` 第500步核（SHA `0f45200e331df6dc50c75cd0aec870dc965948fbbd4a5b430bd5d7ef2147886f`），以及先前路径诊断保存的原生坐标编码 `true_H` 和四帧observed history。没有新rollout、优化、读出、HDF读取或test/validation读取。当前步H的force由冻结CPU float32模型计算，随后H、force及系数提升到float64做解析矩和因果Gaussian filter；这不是原float32采样器的逐位重放。H为128维，实际bath为每残基8维，四窗有效残基数依次为96、57、91、96，全部具有定义良好的切向投影。

完整空间噪声协方差先计算 `K=F Fᵀ`，再取固定valid支持的主子矩阵。D/C/Gamma/K在原始observed anchor固定；force随当前真实H变化，已含实际温度条件，不再额外乘温度。四帧history采用原采样器的批量编码，独立anchor与保存H0逐位一致；history末帧与单独anchor最大差依次为7.15e−7、9.54e−7、7.15e−7、9.54e−7，按预定容差报告而不强行替换。初始S为最早observed帧的N(0,I)，先同化3个history增量；每个未来transition先保存预测矩，再同化下一真实H，共12次history及85次未来更新，没有重新锚定。因而t>0条件包含已见真实路径历史，不能称为只凭最初anchor的forecast能力。

记 `M=√D (K⊗I) √D`、当前条件bath为N(m,P)、h=1000 ps，则一步均值增量和协方差为 `b=hM(f−Cm)`、`Σ=2hM+h²MCPCᵀM`；保留完整跨残基P相关性，不构造全H维协方差。raw平方范数变化预算是 `2H·b + ||b||² + trΣ`，centered预算对H与增量施加去通道均值投影Q。切向投影再去除当前QH径向分量，统计的是有限1 ns增量二阶矩，不是连续时间瞬时扩散系数。下表每项先对固定有效残基等权，再在本窗全部transition等权；单位均为latent单位平方/一次1 ns步，非Å指标。

| TRAIN窗口 / 温度 / 步数 | raw期望变化 | raw观测变化 | centered期望变化 | centered观测变化 | 切向期望二阶矩 | 切向观测二阶矩 |
|---|---:|---:|---:|---:|---:|---:|
| 3l41A01 / 450 K / 1 | 28.802289 | −2.77926e−5 | 28.552939 | −2.77926e−5 | 31.509704 | 50.894019 |
| 3tpqM02 / 320 K / 4 | 11.661392 | 9.64873e−7 | 11.558645 | 9.64873e−7 | 12.805704 | 29.176259 |
| 1efdN01 / 348 K / 16 | 14.531761 | 6.70009e−7 | 14.409341 | 6.70009e−7 | 15.520539 | 29.056532 |
| 3aq5A00 / 320 K / 64 | 13.173267 | 1.14953e−7 | 13.065232 | 1.14953e−7 | 14.195266 | 18.173436 |

四窗raw及centered的实际当前平方范数均值约127.9996，这是此次保存H的测量，不是假设所有训练后encoder输出严格零均值或精确固定半径。raw期望变化在各窗全部transition保持正；观测raw变化绝对值最大分别为2.78e−5、9.10e−6、1.96e−5、1.01e−5。切向的逐步方向并不一致：预测低于观测仅为1/1、3/4、9/16、25/64步，不能把窗口均值差写成每一步都运动不足。原始观测增量平方范数均值为63.104550、34.984270、36.614079、21.942874，径向增量二阶矩分别为12.210531、5.808012、7.557547、3.769438；近恒半径上的有限弦位移本来就可含向内径向分量，不能把所有径向运动视为错误。

| 窗口步数 | raw漂移交叉项 | raw漂移平方 | 热噪声trace | bath条件不确定性trace | C=0 raw期望变化 | C=0 centered期望变化 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | −3.234457 | 0.585489 | 31.259415 | 0.191841 | 29.191552 | 28.945303 |
| 4 | −1.351610 | 0.359575 | 12.588714 | 0.064714 | 12.750390 | 12.647640 |
| 16 | −1.233414 | 0.701463 | 14.999239 | 0.064473 | 15.843734 | 15.717479 |
| 64 | −1.239761 | 1.112472 | 13.254871 | 0.045685 | 14.949262 | 14.833977 |

在这些teacher状态上，负漂移交叉项没有抵消正热噪声trace及其余正项，故期望平方范数增长。`C=0`仅在同一H、force、M上代数删除耦合，不另行过滤或生成轨迹；其窗均增长反而更大，不能归因为bath导致此次范数失配，也不证明bath改善自主预测。bath不确定性项与给定一个已实现S的条件噪声不同，后者仅保留2hM。

核原本已有正二次尾项，不能说缺少恢复力。按保存的 `softplus(log_tail)`、提升后的H和总force，报告脚本在float64分解 `force_tail=−H*tail_coeff`、`force_residual=force_total−force_tail`，再施加同一M。centered交叉项的tail/residual/bath窗均分别为：h1 **−2.689969 / +0.305006 / −0.849494**；h4 **−1.192603 / +0.948379 / −1.107386**；h16 **−1.726161 / +1.688627 / −1.195879**；h64 **−1.530543 / +1.650596 / −1.359813**。三项和与保存的总交叉预算最大差6.66e−16。这里是既有force的代数分解，不是各项独立动力学的因果效果，也不是float32运算级梯度重放。channel预算另外保留：helper的 `channel_expected_change` 是 **128×E[Δ(mean(H)²)]**；观测侧原字段未乘128，图中明确乘128后比较，不能把两种量直接相减。

四窗六面板图保留全部transition、线性坐标及C=0控制，没有挑选峰或时间段；空支持应显示null/空格/断线而不补零，此次切向支持完整。PNG已逐张视觉检查，PDF与逐步CSV/JSON如下：

| 窗口 | PNG | PDF | CSV | JSON |
|---|---|---|---|---|
| h1 | [图](results/hs_joint_20261101/train_latent_moments_report_v1/rendered/step001.png) | [PDF](results/hs_joint_20261101/train_latent_moments_report_v1/rendered/step001.pdf) | [CSV](results/hs_joint_20261101/train_latent_moments_report_v1/rendered/step001.csv) | [JSON](results/hs_joint_20261101/train_latent_moments_report_v1/rendered/step001.json) |
| h4 | [图](results/hs_joint_20261101/train_latent_moments_report_v1/rendered/step002.png) | [PDF](results/hs_joint_20261101/train_latent_moments_report_v1/rendered/step002.pdf) | [CSV](results/hs_joint_20261101/train_latent_moments_report_v1/rendered/step002.csv) | [JSON](results/hs_joint_20261101/train_latent_moments_report_v1/rendered/step002.json) |
| h16 | [图](results/hs_joint_20261101/train_latent_moments_report_v1/rendered/step003.png) | [PDF](results/hs_joint_20261101/train_latent_moments_report_v1/rendered/step003.pdf) | [CSV](results/hs_joint_20261101/train_latent_moments_report_v1/rendered/step003.csv) | [JSON](results/hs_joint_20261101/train_latent_moments_report_v1/rendered/step003.json) |
| h64 | [图](results/hs_joint_20261101/train_latent_moments_report_v1/rendered/step004.png) | [PDF](results/hs_joint_20261101/train_latent_moments_report_v1/rendered/step004.pdf) | [CSV](results/hs_joint_20261101/train_latent_moments_report_v1/rendered/step004.csv) | [JSON](results/hs_joint_20261101/train_latent_moments_report_v1/rendered/step004.json) |

执行与审计证据：

- [主协议](results/hs_joint_20261101/train_latent_moments_v1/protocol.json) SHA `29d37b53f6c17b5f7dbcc7242a3869b0b3c5dc6b3747a7ffac584e53af2272f2`；[实际producer](results/hs_joint_20261101/train_latent_moments_v1/execution.json) PID2424126，2026-11-02 10:53:38.308–10:56:42.418 UTC，184.109秒、exit0；[结果](results/hs_joint_20261101/train_latent_moments_v1/result.json) SHA `fac5e892603d8a0914ba98eaf6cbaacf6f316f83a9ff9677bbd7fc99a3bcc0cb`。保存的核/encoder及CPU RNG前后身份相同；峰值RSS 1,363,623,936字节，结果检查时目录882,158,053字节。
- [独立协议](results/hs_joint_20261101/latent_moment_verification_v1/protocol.json) SHA `185000ba968952d29e05e9aef7a31b791dbb66227c2a907a30c2001a4ca9f4c3`；[实际verifier](results/hs_joint_20261101/latent_moment_verification_v1/execution.json) PID2441100，11:03:04.970–11:05:06.118 UTC，121.148秒、exit0；[核验结果](results/hs_joint_20261101/latent_moment_verification_v1/verification.json) SHA `f8647a25c5e9e464c10c93006c676e4692225e737622d63dd344a252fb6d23c2`。NumPy独立重建局部128×128协方差、85步解析/观测预算和全部97步bath过滤；105,382,755个数值比较最大差8.53e−14，53,474,304个filter比较最大差2.22e−14。未导入producer聚合函数，没有独立重放encoder、force模型或采样；模型/RNG不变及先预测后同化的结论仍依赖冻结源码和身份记录。
- [结果后绘图脚本](results/hs_joint_20261101/train_latent_moments_report_v1/report_v1.py)只加载已核验JSON/NumPy sidecars，不加载模型、checkpoint或PyTorch sidecars；3项合成分解/单位/导出测试通过。它没有另行注册科学统计。[实际绘图](results/hs_joint_20261101/train_latent_moments_report_v1/execution.json) PID2454396，11:08:54.835–11:09:14.401 UTC，19.567秒、exit0；[报告清单](results/hs_joint_20261101/train_latent_moments_report_v1/report.json) SHA `8b4f851c042f8e3618d80dccf26b53d705346dad5eef0f20d5bb5a6b8bb8c384`，记录全部源文件/输入前后哈希及输出PNG/PDF/CSV/JSON哈希。

本结果定位了这四个teacher路径上当前核的条件一步矩失配，未检验自主长程路径、时间相关结构、Mori闭合、热力学FDT或BioEmu2性能。既有事后H半径映射使路径更差的负结果继续有效；这里的解析预算不能逆转该负结果，也不等价于证明训练时几何约束有效。后续若提出修改，仍须独立固定干预、对照和实际评估，不能由此次描述直接采用。

## 13. TRAIN四窗PCA连续性：canonical特征跳变与H跳变同时出现，未做模型干预

相同四个TRAIN窗口的89帧、85个相邻1 ns transition全部完成，独立核验通过。完整逐步图显示，保存H的较大增量与原始canonical坐标特征的较大增量同时出现，而物理CA形状变化呈现不同的幅度模式。几何最优的轴符号/次序重新标记可以使若干canonical增量大幅缩小；**保存H从未改变，也没有重新编码**。这是表示与几何的描述性关联，尚未证明PCA通道导致H跳变、改善动力学或构成可部署修复。

这次只反序列化既有四份TRAIN path sidecars，使用原生坐标与已保存 `true_H`，没有加载模型/checkpoint、运行encoder、生成轨迹、更新参数或读取HDF/test/validation。所有89帧在CPU float32/B1下重现当前canonical helper，并与冻结的 `_within_frame_canonical_positions` 实际函数逐帧比较：89/89完全相等，最大差0。该函数实际使用中心化CA scatter的降序PCA轴和投影三阶矩的非负符号规则；旧docstring的largest-component说明不作为实现依据，冻结源码未修改。

使用proper Kabsch先把下一帧CA对齐到当前帧。行坐标满足 `Y R≈X`，所以最终signed basis比较采用 `B0ᵀ Rᵀ B1`。原始eigensolver eigenvector及其skew符号可能同时翻转并抵消，不能单独按其符号计事件。PCA基允许det=−1，不能把其差异直接解释为SO(3)旋转角。每步均枚举全部48个signed permutation，按固定词典序选择几何basis成本最小者；不使用H、模型分数、阈值或事后筛选。每步保留原始输出、全部候选成本、完整overlap和最佳重标记，不将最佳候选替换进模型。

以下先按原始顺序列出**每个窗口的第一个transition**，不是挑峰或根据效果选例。H列为有效残基等权的增量平方范数，单位为latent单位²/一次1 ns步；其余RMS列单位Å。canonical RMS与物理Kabsch RMSD是不同量，最佳重标记也不必等于物理RMSD。

| 窗口 / 第一步0→1 ns | H raw增量² | H切向增量² | CA Kabsch RMSD | 全pair距离变化RMS | 原始canonical增量RMS | 最佳重标记analysis增量RMS |
|---|---:|---:|---:|---:|---:|---:|
| 3l41A01 / h1 | 63.104550 | 50.894019 | 6.191149 | 3.761485 | 14.900543 | 6.406347 |
| 3tpqM02 / h4 | 2.965337 | 2.937824 | 1.868580 | 1.418412 | 1.953270 | 1.953270 |
| 1efdN01 / h16 | 1.682140 | 1.673864 | 0.967143 | 0.785210 | 1.343871 | 1.343871 |
| 3aq5A00 / h64 | 54.112511 | 44.018648 | 0.904766 | 0.666013 | 14.685613 | 0.922656 |

第一个窗口确有6.19 Å的物理形状变化，不能把所有较大H增量概括为“物理结构几乎不变”。h4/h16的第一步最优候选就是identity，没有canonical RMS降低。为展示同一窗口的相邻变化，下表进一步给出h64窗口**连续最初四步**，仍不采用任何跳变阈值或分组统计；其余60步完整保留在图及CSV中。

| h64时间区间ns | H raw增量² | H切向增量² | CA Kabsch RMSD Å | 原始canonical RMS Å | 最佳重标记RMS Å | 原始basis成本 | 最佳basis成本 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0→1 | 54.112511 | 44.018648 | 0.904766 | 14.685613 | 0.922656 | 4.000327 | 0.000505 |
| 1→2 | 54.0788 | 44.0420 | 0.954545 | 14.5738 | 1.01922 | 4.00068 | 0.001844 |
| 2→3 | 1.56288 | 1.55235 | 1.00790 | 1.24702 | 1.24702 | 0.010130 | 0.010130 |
| 3→4 | 1.52703 | 1.51529 | 0.861770 | 0.880643 | 0.880643 | 0.000895 | 0.000895 |

h64第一步的proper-Kabsch后basis overlap对角约为 `(0.999875, −0.999956, 0.999917)`，选择的P为 `diag(1,−1,1)`；这是最终signed basis的实际轴2方向不一致，不是仅观察raw skew符号。两帧normalized eigengap分别为 `(0.427258,0.333403)` 与 `(0.451677,0.303776)`，此例不需要诉诸重复特征值才出现较大的canonical变化。轴2绝对标准化三阶矩从0.00362447变为0.000102244；全部三轴与全部帧曲线均提供，没有从中拟合“接近零”的事件阈值。各步全部48候选、ties、Kabsch奇异值及反射修正诊断保存在原始NPZ/JSON；signed permutation无法解决一般重复子空间中的连续旋转，因此不将其最优残差当作普适修复界限。

分析严格区分两种数值：保留原函数float32 canonical输出；将保存的float32 centered CA与signed basis提升到float64，重算analysis Z后才做恒等分解，没有重做float64 eigendecomposition。按固定prior-basis约定，`ΔZ=(Y R−X)B0 + (Z1−Y R B0)`；两个平方项还必须加上带符号交叉项。h64第一步的shape²、basis²、交叉项分别为 **0.818602、211.024175、+3.824446 Å²**，合计215.667223 Å²。该分配依赖所选代数约定，不是唯一的因果分摊，不能报告为有界解释百分比。四窗分解向量最大闭合误差分别3.61e−15、2.11e−15、5.22e−15、3.97e−15 Å；原始float32与重构analysis坐标的逐分量最大差分别1.06e−6、1.15e−6、1.62e−6、1.84e−6 Å，均单独保留，未把roundoff伪称为精确原始特征分解。

物理Kabsch RMSD从native CA重新以float64中心化并拟合proper rotation/translation；该R再用于source-centered特征分解，所以shape项保留很小的float32中心化残差。全pair距离变化使用相同valid CA支持的全部 `i<j`，各窗pair数为4560、1596、4095、4560；这是一般形状描述，**不是**冻结评估的PSF nonlocal physical-pair指标，没有改动正式评估定义。H切向支持在全部85步均完整，但H本身是既有encoder表示，不是Å单位的物理速度。第12节切向增量目标的异质性因此需要和本节特征连续性一起理解，不能直接据其窗均差扩大动力学噪声。

所有图已逐张视觉检查。七个数据面板依次为H raw/切向增量²、物理Kabsch/all-pair变化、原始/analysis/重标记canonical RMS、原始/最佳basis成本、两条eigengap、三条绝对标准化skew、shape²/basis²/带符号cross分解。增量面板x为下一帧时间，frame几何面板x为该帧自身时间；h1保留单点marker。CSV合计包含全部85条transition与89条frame记录，JSON保留相同全部字段及null语义，无零填补、事件筛选、相关系数、跨窗总体值或CI。

| 窗口 | PNG / PDF | 全transition CSV | 全frame CSV | JSON |
|---|---|---|---|---|
| h1 | [PNG](results/hs_joint_20261101/train_pca_continuity_report_v1/rendered/step001.png) / [PDF](results/hs_joint_20261101/train_pca_continuity_report_v1/rendered/step001.pdf) | [1步](results/hs_joint_20261101/train_pca_continuity_report_v1/rendered/step001.transitions.csv) | [2帧](results/hs_joint_20261101/train_pca_continuity_report_v1/rendered/step001.frames.csv) | [JSON](results/hs_joint_20261101/train_pca_continuity_report_v1/rendered/step001.json) |
| h4 | [PNG](results/hs_joint_20261101/train_pca_continuity_report_v1/rendered/step002.png) / [PDF](results/hs_joint_20261101/train_pca_continuity_report_v1/rendered/step002.pdf) | [4步](results/hs_joint_20261101/train_pca_continuity_report_v1/rendered/step002.transitions.csv) | [5帧](results/hs_joint_20261101/train_pca_continuity_report_v1/rendered/step002.frames.csv) | [JSON](results/hs_joint_20261101/train_pca_continuity_report_v1/rendered/step002.json) |
| h16 | [PNG](results/hs_joint_20261101/train_pca_continuity_report_v1/rendered/step003.png) / [PDF](results/hs_joint_20261101/train_pca_continuity_report_v1/rendered/step003.pdf) | [16步](results/hs_joint_20261101/train_pca_continuity_report_v1/rendered/step003.transitions.csv) | [17帧](results/hs_joint_20261101/train_pca_continuity_report_v1/rendered/step003.frames.csv) | [JSON](results/hs_joint_20261101/train_pca_continuity_report_v1/rendered/step003.json) |
| h64 | [PNG](results/hs_joint_20261101/train_pca_continuity_report_v1/rendered/step004.png) / [PDF](results/hs_joint_20261101/train_pca_continuity_report_v1/rendered/step004.pdf) | [64步](results/hs_joint_20261101/train_pca_continuity_report_v1/rendered/step004.transitions.csv) | [65帧](results/hs_joint_20261101/train_pca_continuity_report_v1/rendered/step004.frames.csv) | [JSON](results/hs_joint_20261101/train_pca_continuity_report_v1/rendered/step004.json) |

可复核身份与真实执行：

- [主协议](results/hs_joint_20261101/train_pca_continuity_v1/protocol.json) SHA `7074ab4e3583261027f2d8411be2c3d9511d7c072acb31a11bb69cf60bc7ea6a`；[producer执行](results/hs_joint_20261101/train_pca_continuity_v1/execution.json) PID2498920，2026-11-02 11:47:38.326–11:47:40.701 UTC，2.374353秒、exit0；[结果](results/hs_joint_20261101/train_pca_continuity_v1/result.json) SHA `4e8341d120a1d2fdf695dc96b9f14bf7ea5877485484c1963fdf9a16d7655f2f`。输入tensor哈希及CPU RNG前后不变；峰值RSS 666,193,920字节，结果检查时目录35,774,791字节。
- [独立协议](results/hs_joint_20261101/pca_continuity_verification_v1/protocol.json) SHA `47d3fb4d5b77f6ad722375714220626798afd7182d7e855edb613bec89d326d1`；[verifier执行](results/hs_joint_20261101/pca_continuity_verification_v1/execution.json) PID2499053，11:48:39.622–11:48:41.131 UTC，1.508574秒、exit0；[核验结果](results/hs_joint_20261101/pca_continuity_verification_v1/verification.json) SHA `27a57b733a3edd30ebb00c9dc83be344b174ee79cbfaeb6bd1d8627c07f509f9`。NumPy独立保存数组复算覆盖89帧、85步，2,810,000个比较最大差5.46e−12；eigensystem残差4.58e−7、basis正交误差5.92e−7与source-skew残差7.53e−8按预定float32边界核验。它没有独立重放float32 eigh、encoder、H反事实或源轨迹；逐帧source函数完全相等仍是producer实际记录，不应混同独立模型重放。
- [JSON-only绘图脚本](results/hs_joint_20261101/train_pca_continuity_report_v1/report_v1.py)及3项小型合成测试通过独立复核，仅在producer/verifier完成后执行；没有新增统计登记。[实际绘图](results/hs_joint_20261101/train_pca_continuity_report_v1/execution.json) PID2500707，11:50:58.984–11:51:05.978 UTC，6.994167秒、exit0；[报告清单](results/hs_joint_20261101/train_pca_continuity_report_v1/report.json) SHA `b11a55797d43fea8430b157a229377f68ae920bff5b617dbc783fd6627bf6f20`。所有输入/源文件前后SHA及PNG/PDF/CSV/JSON输出SHA完整保留。

下一步若要识别PCA通道对H的影响，需要另行固定并验证通道级反事实；本节并未实施。几何重标记看到了相邻两个真实帧，不能直接用于只给observed history的预测；对每一步各自选择的P也不自动组成一个全局一致的新轨迹。其他encoder通道、非线性交互、真实构象变化及数值退化仍可能贡献H变化。此次没有改善任何实际H/CA预测，没有更改训练/评估目标，也不推翻既有归一化负结果或建立JEPA因果收益、Mori/FDT闭合与BioEmu2全指标胜出。

## 14. TRAIN四窗relpos通道反事实：改变H增量，尚未验证读出或预测收益

固定的89帧及85个相邻transition已完成437次冻结encoder forward，独立核验通过。这次真正改变了encoder的内部relpos输入：保持原生坐标、mask、embedding IDs和其他通道不变，将第13节几何预先选定的signed permutation P施加到下一帧canonical特征。h64第一步的有效残基平均H增量平方范数从 **54.112511** 变为 **1.019987**；把同一个P同时施加到当前帧和下一帧的控制则为 **54.021170**。这是该固定内部通道干预对表示输出的实际影响，比上一节的保存H关联更进一步；仍未检验物理因果、结构读出、Stage B自主路径或任何正式评估收益。

本次只从已核验的continuity NumPy sidecars读取四窗原生坐标、固定metadata、89帧H/canonical与85个P，没有重读原始Torch路径sidecar、HDF、test或validation。反序列化两份完整checkpoint字典以获得当前encoder权重及上游Stage A配置，字典中可能包含其他模块权重；只构造和执行encoder，Stage B/decoder构造及forward均为零。当前encoder state SHA为 `741d8fbf712981d8aefd4a4841d75ea201663c247dca92396e69f4bc43bbcc27`，与前述路径诊断一致。CPU两线程、B1、float32、eval/no-grad；使用encoder原本显式matmul/softmax attention，没有SDPA backend替换。

执行严格分三阶段：先完成全部89帧的original prepass与identity-hook sham共178次，所有valid H字节都与保存H相同、masked输出为零，sham与original全字节相同，才写入prepass gate并允许反事实；随后每个pair各运行next-P及current-P，共170次，shared-P控制重用同一个next-P输出；最后全部89帧original postpass与prepass全字节相同。hook只作用于 `relpos_proj` 的一次输入，原始canonical与保存float32特征逐位一致，替换采用signed column gather，identity保留原输入对象，每次均移除hook。全阶段模型参数和包括非持久 `rbf_centers` 在内的全部buffer、输入tensor及CPU RNG身份不变。

下表按每个窗口的**第一个时间区间0→1 ns**报告，未按干预效果选例。raw及centered列都是有效残基等权的H增量平方范数，单位latent单位²/一次1 ns步；centered仅去除通道均值，不作半径归一化。此次这些首步raw/centered值在下列六位小数精度相同，完整双精度数值分别保留。

| TRAIN窗口 / horizon | 原始raw增量² | next-only raw增量² | shared-P raw增量² | 原始centered增量² | next-only centered增量² | shared-P centered增量² | 首步P |
|---|---:|---:|---:|---:|---:|---:|---|
| 3l41A01 / h1 | 63.104550 | 17.459423 | 62.442893 | 63.104550 | 17.459423 | 62.442893 | 非identity |
| 3tpqM02 / h4 | 2.965337 | 2.965337 | 2.965337 | 2.965337 | 2.965337 | 2.965337 | identity |
| 1efdN01 / h16 | 1.682140 | 1.682140 | 1.682140 | 1.682140 | 1.682140 | 1.682140 | identity |
| 3aq5A00 / h64 | 54.112511 | 1.019987 | 54.021170 | 54.112511 | 1.019987 | 54.021170 | 非identity |

h1的首步物理Kabsch变化为上一节的6.191 Å，next-only后的H增量仍明显非零，不能说此干预删除了真实构象运动。h4/h16首步是精确identity控制，三个输出完全相同。继续沿h64窗口原时间顺序展示最初四步，不构造任何跳变阈值、事件类别或总体效果统计：

| h64区间ns | 原始raw增量² | next-only raw增量² | shared-P raw增量² | 原始切向增量² | next-only切向增量² | shared-P切向增量² |
|---|---:|---:|---:|---:|---:|---:|
| 0→1 | 54.112511 | 1.019987 | 54.021170 | 44.018648 | 1.016641 | 43.975949 |
| 1→2 | 54.078831 | 1.275317 | 53.772123 | 44.041983 | 1.268255 | 43.782009 |
| 2→3 | 1.562884 | 1.562884 | 1.562884 | 1.552354 | 1.552354 | 1.552354 |
| 3→4 | 1.527035 | 1.527035 | 1.527035 | 1.515285 | 1.515285 | 1.515285 |

切向列须保留不同参考的限定：original及next-only投影使用原始centered H0方向，shared-P使用改变后的centered H0P方向。因此shared-P切向值不是同一投影下的修复效果。所有逐残基切向valid/null支持和参考标签均保存，没有以零代替未定义量。

记原始增量d=H1−H0、下一帧效应e1=H1P−H1、当前帧效应e0=H0P−H0，则next-only增量为d+e1，shared-P增量为d+e1−e0。每种变化均保留 `effect²+2〈d,effect〉` 的平方项和带符号交叉项，不能只看正的effect²或把两项写成有界解释百分比。例如h64首步next-only的effect²为 **52.233158**、cross为 **−105.325681**，其全精度恒等式给出1.0199873359；shared-P相应为 **212.441312** 与 **−212.532652**，给出54.0211700546。这里展示值已四舍五入，不要求逐个截断后再求和仍严格闭合。shared-P控制显示共同轴重标记仍会改变表示，模型不被假设为轴标签等变；在该首步它保留了较大两帧差异，而next-only相对重标记使差异缩小。

这里固定使用了全部85个geometry-only P。协议保留identity候选，且不因候选ties或数值退化而筛除pair；这不表示实际数据中必然出现了ties或退化。P只由两帧CA/basis决定，没有根据H搜索或调参；但仍使用了未来真实结构，且每个pair独立选P，相邻pair不自动拼成一条一致的新H轨迹。干预后的canonical输入可能离开模型训练时的规范约定。此次没有运行任何读出、核或预测评估，不能从更小H增量推导更准确结构、合理物理时间尺度或更好forecast。第12节的一步矩失配必须结合这种表示依赖理解，但本次没有重新估计其矩、改变噪声/恢复力或实施训练修复；既有归一化负结果、JEPA因果证据不足及Mori/FDT/BioEmu2目标缺口均继续有效。

真实执行及独立验证：

- [主协议](results/hs_joint_20261101/train_relpos_counterfactual_v1/protocol.json) SHA `43cd6e7e951ce8976bfb30e9639050dc154eb4962ff0881fdc7f32de2984033f`；[producer执行](results/hs_joint_20261101/train_relpos_counterfactual_v1/execution.json) PID2553149，2026-11-02 12:27:36.570–12:27:52.568 UTC，15.998777秒、exit0；[结果](results/hs_joint_20261101/train_relpos_counterfactual_v1/result.json) SHA `f50b9b94c0cd5e59c61f484f385059cd560554277402ac37c2d08fe22dbb7ec0`。记录437次forward及178/348/437的阶段边界；峰值RSS 663,224,320字节，最终结果前检查的目录210,165,157字节。
- [独立协议](results/hs_joint_20261101/relpos_counterfactual_verification_v1/protocol.json) SHA `bc42f8df2e72911f0b14158106cd97edfda55786b5cc1cb39470ef7c7dbe9eca`；[verifier执行](results/hs_joint_20261101/relpos_counterfactual_verification_v1/execution.json) PID2553892，12:28:43.919–12:28:47.428 UTC，3.508584秒、exit0；[核验结果](results/hs_joint_20261101/relpos_counterfactual_verification_v1/verification.json) SHA `b65166d4256047dd751923b3b3d5019e74a33a5d28487438b9e6f0f3531cd58c`。NumPy独立重算所有pair效应/交叉/切向量并核对89帧pre/sham/post、全部437调用顺序、P及hook输入输出字节；20,699,561个数值比较最大差6.82e−13。它不加载模型或独立执行encoder/hook，模型/RNG不变和实际分支执行仍由冻结源码及producer身份记录支持。

四窗的全部85个pair均导出为CSV/JSON并绘图，PNG已经逐张视觉检查。每窗七个数据面板依次保留raw/centered三种H增量²、next-only/shared-P各自raw/centered的effect²与带符号cross，以及明确区分当前参考方向的切向增量²。所有增量面板横轴为下一帧时间，h1使用单点marker；没有截取局部区间、池化窗口、补零、阈值或相关分析。完整曲线显示next-only与shared-P两种控制的差异，shared-P并非严格等于原始且部分步骤略增，next-only也保留非零残差；不能把它描述成所有H动态已被消除。

| 窗口 | PNG | PDF | 全pair CSV | JSON |
|---|---|---|---|---|
| h1 | [图](results/hs_joint_20261101/train_relpos_counterfactual_report_v1/rendered/step001.png) | [PDF](results/hs_joint_20261101/train_relpos_counterfactual_report_v1/rendered/step001.pdf) | [1步](results/hs_joint_20261101/train_relpos_counterfactual_report_v1/rendered/step001.pairs.csv) | [JSON](results/hs_joint_20261101/train_relpos_counterfactual_report_v1/rendered/step001.json) |
| h4 | [图](results/hs_joint_20261101/train_relpos_counterfactual_report_v1/rendered/step002.png) | [PDF](results/hs_joint_20261101/train_relpos_counterfactual_report_v1/rendered/step002.pdf) | [4步](results/hs_joint_20261101/train_relpos_counterfactual_report_v1/rendered/step002.pairs.csv) | [JSON](results/hs_joint_20261101/train_relpos_counterfactual_report_v1/rendered/step002.json) |
| h16 | [图](results/hs_joint_20261101/train_relpos_counterfactual_report_v1/rendered/step003.png) | [PDF](results/hs_joint_20261101/train_relpos_counterfactual_report_v1/rendered/step003.pdf) | [16步](results/hs_joint_20261101/train_relpos_counterfactual_report_v1/rendered/step003.pairs.csv) | [JSON](results/hs_joint_20261101/train_relpos_counterfactual_report_v1/rendered/step003.json) |
| h64 | [图](results/hs_joint_20261101/train_relpos_counterfactual_report_v1/rendered/step004.png) | [PDF](results/hs_joint_20261101/train_relpos_counterfactual_report_v1/rendered/step004.pdf) | [64步](results/hs_joint_20261101/train_relpos_counterfactual_report_v1/rendered/step004.pairs.csv) | [JSON](results/hs_joint_20261101/train_relpos_counterfactual_report_v1/rendered/step004.json) |

[JSON-only绘图脚本](results/hs_joint_20261101/train_relpos_counterfactual_report_v1/report_v1.py)及3项小型合成测试经独立复核后，只执行一次；没有重跑encoder、反序列化checkpoint或添加新统计登记；已登记输入仍做字节哈希读取。[实际绘图](results/hs_joint_20261101/train_relpos_counterfactual_report_v1/execution.json) PID2568589，2026-11-02 12:39:42.878–12:40:02.887 UTC，20.008468秒、exit0；execution SHA `03e1dcb485ae9d99f66bcb9cb04a5267ddcd62a6b8dc90d95d1e19a71f4bb9ae`。[报告清单](results/hs_joint_20261101/train_relpos_counterfactual_report_v1/report.json) SHA `ff7ef1cbc8332ef7df7a3c0bb40fa5cac8bc784b526583d5c709775c41b8ccc9`，保存全部源文件/输入前后SHA及PNG/PDF/CSV/JSON输出SHA；最终renderer源SHA `fbbdb1e152f9982c9e0d8cd426d40d4423df50b2fc82f8431c1ba2c552d6b67c`。没有覆盖任何已登记producer、verifier或先前报告产物。


## 15. TRAIN四窗局部坐标原始特征：部分跳变减小，局部取向变化仍可放大增量

同一组已核验的TRAIN四窗、89帧/85个相邻pair上，局部骨架坐标特征并未一致降低原始PCA特征的时间增量。此次只比较原始几何特征：保存的PCA canonical位置、恒零位置，以及每个残基的局部N–CA–C坐标架作用于CA相对可见残基质心的位置。输入沿用第13节的native坐标、固定mask/身份/时间和已保存的proper Kabsch R；没有物化H数组、反序列化checkpoint、运行encoder、重新拟合PCA/Kabsch、采样、更新参数或读取新轨迹。历史输入仍做字节哈希核验。[协议与定义](docs/TRAIN_LOCAL_RELPOS_CONTINUITY_V1_20261102.md)保留全部支持，不按结果筛除pair，也没有选择阈值、相关分析、窗口池化或置信区间。

下表为每窗按时间顺序的第一个0→1 ns增量，数值是有效残基平均平方特征增量的平方根，单位Å。四窗分别为3l41A01/450 K、3tpqM02/320 K、1efdN01/348 K、3aq5A00/320 K；有效残基数为96/57/91/96。

| 窗口 | PCA source RMS | local source RMS | zero RMS |
|---|---:|---:|---:|
| h1 | 14.900543 | 13.021871 | 0 |
| h4 | 1.953270 | 3.958564 | 0 |
| h16 | 1.343871 | 3.090634 | 0 |
| h64 | 14.685613 | 2.996405 | 0 |

h4和h16的首步local增量反而更大。h64前四个连续pair进一步显示两种特征增量的大小关系随时间步改变，而非统一的运动缩减：

| h64时间/ns | PCA source RMS/Å | local source RMS/Å | local shape²/Å² | local basis²/Å² | signed cross/Å² |
|---|---:|---:|---:|---:|---:|
| 0→1 | 14.685613 | 2.996405 | 0.818602 | 10.053425 | −1.893581 |
| 1→2 | 14.573778 | 2.720545 | 0.911157 | 8.609827 | −2.119617 |
| 2→3 | 1.247016 | 3.185423 | 1.015863 | 10.593378 | −1.462322 |
| 3→4 | 0.880643 | 3.289753 | 0.742648 | 11.365760 | −1.285932 |

最后三列来自记录的float32坐标/局部架提升至float64后的分析量，不是重新计算的float32 source增量。用行坐标X、Y和Y R≈X的保存旋转，固定分解为shape=(Y R−X)F₀ᵀ、basis=Z₁−(Y R)F₀ᵀ，因此完整精度下ΔZ=shape+basis，平方均值等于shape²+basis²+2〈shape,basis〉；source/analysis舍入差及分解残差另存。该分配不是唯一因果归因，signed cross不是有界“解释比例”。h64首步local分析增量²为8.978446 Å²，而basis项10.053425 Å²、shape项0.818602 Å²、cross为−1.893581 Å²。该步当前质心杠杆臂长度的残基均值12.685795 Å，局部架运输后变化的Frobenius范数均值0.316966；局部取向变化乘以较长杠杆臂仍可产生显著增量。

全部89帧的有效支持中，exact degeneracy、未定义骨架夹角及四种记录的≤1e−8归一化epsilon计数均为零。这里local跳变并不需要局部架坍缩；这些数值标记也不等价于完整化学几何合格。PCA和local特征幅值近乎相等，是非退化正交坐标变换保留单点范数的预期结果，不表示不同残基共享同一坐标架、保留相同成对距离或具有相同信息可读性。zero分支增量恒零，同时删除位置输入，不能把这一退化控制的平滑性当作表示质量。

本次没有H、读出、JEPA训练或forecast结果。更小原始特征增量不能推出更好信息、物理时间尺度或动力学；更大增量也不直接证明local表示不可用。第14节的分支干预证据与此处局部架敏感性共同限制了“替换PCA即可修复”的推断。任何候选encoder仍须另行检验表示可读性、训练效果及完整评估；既有负结果和原始BioEmu2目标缺口保持不变。

执行、失败保留与独立核验：

- [producer协议](results/hs_joint_20261101/train_local_relpos_continuity_v1/protocol.json) SHA `4629b28ab2a81340eda6762fc0dee6eb2ee413faddee315cf8072aa7a457e07c`；[实际执行](results/hs_joint_20261101/train_local_relpos_continuity_v1/execution.json) PID2596888，2026-11-02 13:08:41.772–13:08:53.092 UTC，11.319655秒、exit0。[结果及全部frame/pair NPZ索引](results/hs_joint_20261101/train_local_relpos_continuity_v1/result.json) SHA `d7a35e37a781350951958ca4f84f03f75f683e48d107f194f531b219218f5049`；输入张量和CPU RNG前后身份一致。最终结果前资源检查为峰值RSS 575,275,008字节、目录12,329,708字节。
- 首版独立checker在producer之前[登记](results/hs_joint_20261101/local_relpos_continuity_verification_v1/protocol.json)，但[实际执行](results/hs_joint_20261101/local_relpos_continuity_verification_v1/execution.json) PID2597376于13:09:41.525–13:09:42.195 UTC以exit1结束，错误为`source numeric mismatch: centered`；[失败日志](results/hs_joint_20261101/local_relpos_continuity_verification_v1_run.log)及全部v1源码/协议/失败产物保留。checker使用了另一种NumPy float32归约顺序计算质心，再相减，在较大平移坐标/接近零的中心化分量上产生差异。首失败帧的两种float32质心最大差1.144409e−5 Å，7个中心化分量越过旧检查容差；保存质心相对float64数学均值最大差仅9.536743e−7 Å。使用保存的float32质心逐元素重放中心化则字节完全一致。这是checker数值复现边界的问题，不是producer数据修复。
- [v2协议](results/hs_joint_20261101/local_relpos_continuity_verification_v2/protocol.json) SHA `6369069073774af60d4d044d6274603266304e685bb4b2e73f54596f56b4cbd7`明确是**producer完成后的checker修订**：固定v1失败证据和未变producer结果，要求原v1登记早于producer，不伪称v2预登记早于结果。修订以L个总行数（含padding）和n个有效残基推导float32求和/除法误差界γ_L(u)Σ|CA|/n，γ_L=Lu/(1−Lu)，同时计入float64参考归约误差、向外取整的绝对和上界以及显式subnormal舍入项；不是按实际失败幅度调阈值。之后以记录的float32质心重放中心化，要求字节相等。其余几何/float64代数定义和容差不变；平移、相消、近零、subnormal、错误质心、中心化篡改和失败链时间顺序均有合成拒绝测试。没有重跑producer。
- [v2实际核验](results/hs_joint_20261101/local_relpos_continuity_verification_v2/execution.json) PID2610369，13:24:34.657–13:24:37.017 UTC，2.359982秒、exit0；[verification](results/hs_joint_20261101/local_relpos_continuity_verification_v2/verification.json) SHA `c4ae4a545be97545d1f37ae59c7722c1c2410c8d9f7aa0323ac24b81bc135223`，89帧/85pair、845,103个数值比较，最大差2.273737e−13。独立NumPy重建局部架并按预定float32边界核验，再以记录数组重算全部float64代数；它没有重放encoder/PCA/Kabsch，也不验证H。实际质心参考误差最大1.475575e−5 Å、解析界最大6.506121e−4 Å分别保留，二者可能来自不同帧/通道，不构成同一样本的误差比。

全部89帧/85pair的CSV、JSON和四窗完整曲线如下；各PNG已逐张视觉检查，h1保留单点marker。面板涵盖三分支source增量与幅值、local分解/有符号cross、局部架变化、夹角支持/epsilon计数及source-analysis舍入差。全残基数组通过上述producer结果索引保留；没有仅输出挑选的步骤。

| 窗口 | PNG | PDF | 全frame CSV | 全pair CSV | JSON |
|---|---|---|---|---|---|
| h1 | [图](results/hs_joint_20261101/train_local_relpos_continuity_report_v1/rendered/step001.png) | [PDF](results/hs_joint_20261101/train_local_relpos_continuity_report_v1/rendered/step001.pdf) | [2帧](results/hs_joint_20261101/train_local_relpos_continuity_report_v1/rendered/step001.frames.csv) | [1步](results/hs_joint_20261101/train_local_relpos_continuity_report_v1/rendered/step001.transitions.csv) | [JSON](results/hs_joint_20261101/train_local_relpos_continuity_report_v1/rendered/step001.json) |
| h4 | [图](results/hs_joint_20261101/train_local_relpos_continuity_report_v1/rendered/step002.png) | [PDF](results/hs_joint_20261101/train_local_relpos_continuity_report_v1/rendered/step002.pdf) | [5帧](results/hs_joint_20261101/train_local_relpos_continuity_report_v1/rendered/step002.frames.csv) | [4步](results/hs_joint_20261101/train_local_relpos_continuity_report_v1/rendered/step002.transitions.csv) | [JSON](results/hs_joint_20261101/train_local_relpos_continuity_report_v1/rendered/step002.json) |
| h16 | [图](results/hs_joint_20261101/train_local_relpos_continuity_report_v1/rendered/step003.png) | [PDF](results/hs_joint_20261101/train_local_relpos_continuity_report_v1/rendered/step003.pdf) | [17帧](results/hs_joint_20261101/train_local_relpos_continuity_report_v1/rendered/step003.frames.csv) | [16步](results/hs_joint_20261101/train_local_relpos_continuity_report_v1/rendered/step003.transitions.csv) | [JSON](results/hs_joint_20261101/train_local_relpos_continuity_report_v1/rendered/step003.json) |
| h64 | [图](results/hs_joint_20261101/train_local_relpos_continuity_report_v1/rendered/step004.png) | [PDF](results/hs_joint_20261101/train_local_relpos_continuity_report_v1/rendered/step004.pdf) | [65帧](results/hs_joint_20261101/train_local_relpos_continuity_report_v1/rendered/step004.frames.csv) | [64步](results/hs_joint_20261101/train_local_relpos_continuity_report_v1/rendered/step004.transitions.csv) | [JSON](results/hs_joint_20261101/train_local_relpos_continuity_report_v1/rendered/step004.json) |

[绘图脚本](results/hs_joint_20261101/train_local_relpos_continuity_report_v1/report_v1.py)源SHA `d2ada12e30a5bcee3753fef289ab02b22f7a90f2bfaf9dd4440fea1ad6d2b013`，只在v2成功后执行一次；[实际执行](results/hs_joint_20261101/train_local_relpos_continuity_report_v1/execution.json) PID2611242，13:25:48.077–13:25:54.537 UTC，6.459616秒、exit0，execution SHA `53bd61627bbba893841ba0d042213cbb23fe9b97946f8b87567af12dfbf58121`。[报告清单](results/hs_joint_20261101/train_local_relpos_continuity_report_v1/report.json) SHA `1b234cef0ea4d0365194010c57f9ae19d19fcd09825e07cf01d99ffa252d6f1d`固定输入/源/输出身份；renderer不反序列化checkpoint或执行模型，哈希读取保留。所有既有冻结产物保持原样。


## 16. Stage A三臂适配：zero/local未改善端点读出，三臂预测均弱于坐标持久性

固定500步的PCA/zero/local配对适配及各自500步新读出训练均完成，独立核验通过。本次不采用zero或local替换：主指标真实future_H的物理距离log-MSE分别为PCA 0.009628、zero 0.205728、local 0.140957。仅凭PCA在这一预算内较好不能推出其内在表示最优；预训练的projection、predictor和辅助heads对PCA有起点优势。这也不是“平滑原始特征即改善可读性”的正结果。其余几何通道仍存在，不能把zero/local较高误差说成所有几何信息消失或H坍缩。

[事前试验定义](docs/STAGE_A_RELPOS_ADAPTATION_TRIAL_V1_20261102.md)固定原先TRAIN500面板：前400个不同蛋白用于适配及读出拟合，后100个仅在本次适配/拟合之外，每个h1/4/16/64各25个。后100个已经用于先前TRAIN诊断，且可能在预训练中出现，并非盲测或未见蛋白评估。三臂从相同current encoder及原Stage A predictor/OPF/辅助heads出发，各自同步teacher并重置EMA。原配置不变，native八帧历史加一个严格未来端点，B4按固定400行顺序循环五次；GPU0 FP32、seed42、同一AdamW/学习率日程及历史loss定义，无早停/重选。实际随机状态、空间mask、结构pair抽样和ranking支持逐步保存；各臂teacher随自身更新，latent训练loss只作臂内诊断，不能据此排列表示质量。

终点online encoder冻结后，三臂各用同初始化width256 symmetric head在前400个对齐的缓存端点几何重新编码H上训练，顺序为400行一次加前100行再次。future_H是**真实未来几何的编码**，不是forecast；observed_H_substitution是同一个head的输入替换控制；predicted_H是仅使用native历史、已知温度和horizon的Stage A端点predictor输出，没有未来坐标/teacher/目标对齐输入。它没有调用Stage B随机动力学。三臂训练和三个head拟合全部完成之后才进行评分及读取新的heldout八帧历史；此前整个历史cache已反序列化并校验，其中包括heldout单帧几何及未用于计算的旧H，不能称为从未读取heldout坐标。predicted_H也只检验通过真实target_H上拟合的固定head进行迁移，不是另训predicted_H专用decoder，更不是预测可恢复性的上限。

全部100蛋白共386,092个固定有效full-PSF非局部pair一致保留。先对每个蛋白的pair误差取均值，再对蛋白等权；RMSE为等权MSE的平方根，不是平均各蛋白RMSE。负距离预测不截断，实际所有角色/臂的负预测计数均为0。

| 输入角色 | PCA log-MSE | zero log-MSE | local log-MSE | PCA RMSE/Å | zero RMSE/Å | local RMSE/Å |
|---|---:|---:|---:|---:|---:|---:|
| future_H（主指标） | 0.009628 | 0.205728 | 0.140957 | 1.965673 | 10.453598 | 8.619644 |
| observed_H替换 | 0.054046 | 0.206257 | 0.159039 | 5.929396 | 10.456051 | 9.160820 |
| predicted_H（固定次指标） | 0.181512 | 0.223018 | 0.251984 | 9.088464 | 11.009984 | 11.219255 |
| 原始observed-CA距离持久性（同一基线） | 0.047814 | 0.047814 | 0.047814 | 5.637213 | 5.637213 | 5.637213 |

固定10,000次按horizon分层、蛋白配对bootstrap（seed20261102、线性percentile）复算结果如下，差值方向均为替换臂减PCA；只重复已登记统计，不增加比较或选择。区间条件于本次单seed训练，不包含训练seed不确定性，也没有多重比较校正或架构优越性结论。

| 对比 | future_H log-MSE差值及95%区间（主） | predicted_H log-MSE差值及95%区间（固定次） |
|---|---|---|
| zero−PCA | +0.196100 [0.178326, 0.217235] | +0.041506 [0.011893, 0.071983] |
| local−PCA | +0.131329 [0.120904, 0.143538] | +0.070472 [0.044494, 0.098251] |

三臂predicted_H在每个horizon的log-MSE、MAE和RMSE均差于原始CA持久性。与此同时，**h64的zero预测优于PCA**，不能将总体差值扩写为每个horizon都输。下表保留全部horizon的预测log-MSE，括号内为RMSE/Å：

| horizon/ns | PCA predicted_H | zero predicted_H | local predicted_H | CA persistence |
|---|---:|---:|---:|---:|
| 1 | 0.103313 (7.040266) | 0.204220 (10.037774) | 0.193293 (9.874407) | 0.015445 (2.804794) |
| 4 | 0.146937 (7.576036) | 0.233276 (11.443539) | 0.228996 (10.860227) | 0.020644 (3.093729) |
| 16 | 0.204986 (10.091435) | 0.245164 (12.109995) | 0.287805 (12.445530) | 0.057764 (6.725489) |
| 64 | 0.270810 (11.027328) | 0.209410 (10.320635) | 0.297842 (11.538939) | 0.097405 (8.027607) |

PCA真实future_H读出较好而predicted_H误差较大，说明此试验的表示可读性与观测驱动预测仍须分开判断，不能用oracle输入指标替代预测质量。先前未做此次适配的current encoder读出log-MSE约0.002610、RMSE1.66792 Å，而此次PCA为0.009628、1.965673 Å；该历史数字仅为上下文，未构成本试验中配对的未适配控制，不能据此估计适配收益或损害的因果效应。本次既没有证明H信息上限/JEPA因果收益，也没有检验表示时间稳定性、完整backbone、Stage B/Mori/FDT、旧decoder兼容性或超过BioEmu2。下一步的可检验方向是保护已可读的冻结H，再比较观测驱动predictor的latent目标与物理读出辅助目标；这只是后续设计，不是本次获得的改进结果。

实际执行与失败保留：

- [producer协议](results/hs_joint_20261101/stage_a_relpos_adaptation_v1/protocol.json) SHA `148c197e6198f149e16bad205a6c68172d7231404fd112d766ea838c0a1aeb49`。[score实际执行](results/hs_joint_20261101/stage_a_relpos_adaptation_v1/score/execution.json) PID2673659，2026-11-02 15:12:15.842–15:18:05.330 UTC，349.486685秒、exit0；[完整结果](results/hs_joint_20261101/stage_a_relpos_adaptation_v1/score/result.json) SHA `bd15f570f9142ad49cde031828d6a50a98d49643a65c9ed83d5d2abb95948840`。PCA/zero/local适配分别134.548323/152.329654/166.208141秒，单进程峰值GPU allocation均约0.6877 GiB；对应head拟合58.490316/76.845803/54.092159秒。全部9阶段PID、时间及内存见[resources.csv](results/hs_joint_20261101/stage_a_relpos_adaptation_report_v1/resources.csv)，阶段耗时含各自IO/审计，内存峰值不相加。
- 首版独立checker在评分前[登记失败](results/hs_joint_20261101/stage_a_relpos_verification_v1/registration.execution.json)，原因是NumPy float32无法直接JSON序列化；所有v1源和失败记录保留。v2只将该标量转换为Python `float(...)`，预定的FP32容差界与数值公式相同。[v2协议](results/hs_joint_20261101/stage_a_relpos_verification_v2/protocol.json) SHA `2d0db20611853c9793e23d4717b65bbc5d63475b876c1c8ee79ef49688bd8d9a`登记于训练之后、SCORE之前，不称为训练前checker登记。
- [v2核验执行](results/hs_joint_20261101/stage_a_relpos_verification_v2/execution.json) PID2677266，15:19:22.127–15:19:33.088 UTC、exit0，记录seconds10.890746；[verification](results/hs_joint_20261101/stage_a_relpos_verification_v2/verification.json) SHA `3ec9468842cfedd91404171e87ca5664d73379403f8712779491e6848de66ce5`，100行/5组/3臂/3读出角色、1,500适配及1,500读出更新，7,322个比较，最大差4.547474e−13。独立核验覆盖保存预测、聚合/bootstrap和执行/抽样证据，不是encoder训练的数值重演。
- [renderer实际执行](results/hs_joint_20261101/stage_a_relpos_adaptation_report_v1/execution.json) PID2677406，15:20:46.530–15:20:54.998 UTC，8.467852秒、exit0；execution SHA `f8bf14c368d9c7cb8959e02f0ee065cfa53e2a588a650618f843459dc09a31ae`。[报告清单](results/hs_joint_20261101/stage_a_relpos_adaptation_report_v1/report.json) SHA `090c3fae99fd0b66605f76115aa8a9e0a25eca6e13c76c1445a0f0c9bd8ccc86`，固定源/输入/输出哈希。绘图只读取核验通过的JSON/JSONL，没有重跑模型或bootstrap，所有冻结产物保持原样。

完整[逐蛋白100行CSV](results/hs_joint_20261101/stage_a_relpos_adaptation_report_v1/per_protein.csv)、[全部50组/角色指标](results/hs_joint_20261101/stage_a_relpos_adaptation_report_v1/aggregates.csv)、[全部预定统计JSON](results/hs_joint_20261101/stage_a_relpos_adaptation_report_v1/metrics.json)和[生成报告](results/hs_joint_20261101/stage_a_relpos_adaptation_report_v1/REPORT.md)均保留。以下三图已逐张视觉检查，显示全部蛋白/全部horizon/全部500步，不挑选事件或平滑曲线；ranking计数保留全部七个历史间隔，读出拟合曲线为in-sample，不能当作heldout效果。

| 内容 | PNG | PDF | 数值记录 |
|---|---|---|---|
| 100蛋白的三角色配对log-MSE | [图](results/hs_joint_20261101/stage_a_relpos_adaptation_report_v1/paired_log_mse.png) | [PDF](results/hs_joint_20261101/stage_a_relpos_adaptation_report_v1/paired_log_mse.pdf) | [CSV](results/hs_joint_20261101/stage_a_relpos_adaptation_report_v1/per_protein.csv) |
| 各horizon物理MAE/RMSE及持久性 | [图](results/hs_joint_20261101/stage_a_relpos_adaptation_report_v1/physical_errors_by_horizon.png) | [PDF](results/hs_joint_20261101/stage_a_relpos_adaptation_report_v1/physical_errors_by_horizon.pdf) | [CSV](results/hs_joint_20261101/stage_a_relpos_adaptation_report_v1/aggregates.csv) |
| 臂内训练loss、梯度、ranking支持及读出拟合 | [图](results/hs_joint_20261101/stage_a_relpos_adaptation_report_v1/learning_diagnostics.png) | [PDF](results/hs_joint_20261101/stage_a_relpos_adaptation_report_v1/learning_diagnostics.pdf) | [适配1,500步](results/hs_joint_20261101/stage_a_relpos_adaptation_report_v1/adaptation_updates.csv)、[读出1,500步](results/hs_joint_20261101/stage_a_relpos_adaptation_report_v1/readout_updates.csv) |
