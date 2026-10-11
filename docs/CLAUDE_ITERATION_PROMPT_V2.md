# Claude Code Prompt: 持续优化 Stage C，直到严格 gate 通过

下面的内容请完整交给 Claude Code 执行。

---

你现在负责继续优化当前 Protein-JEPA-Frame / LSP-WM ABC framework。

当前 v5b 不是最终成功版本。它完成了一次真实的部分改善，但仍未通过我们定义的严格 quality gate，而且 decoder 仍然几乎不依赖 Q information。你的任务是继续自主调研、修改代码、训练、评估和迭代，直到最终模型真正达到成功条件。

不要只追求比上一版稍微好一点。不要在小幅改善后停止。除非所有最终 gate 都满足，否则任务仍然是未完成状态。

具体如何定位问题、选择模型结构、修改 loss、调整数据流程、改变训练策略或重写某些组件，由你自行调研和决定。本文只规定目标、实验纪律、有效性约束和停止条件，不提供固定的技术修复方案。

---

## 1. 当前已知状态

当前 v5b checkpoint：

```text
artifacts/full_abc_stage_c_quse_v5/step_400.pt
```

当前 full evaluation：

```text
tmp_claude/eval_v5b_step400_full/full_report.json
tmp_claude/eval_v5b_step400_full/bootstrap_report.json
tmp_claude/eval_v5b_step400_full/per_protein_results.jsonl
```

当前 v5b 的 evaluation protocol 本身已经完整通过：

- 500 held-out samples；
- 1500 records；
- horizons: 32 ns、64 ns、96 ns；
- seeds: 42、123、456；
- oracle/direct/autonomous/persistence/qshuffle 全部存在；
- failed samples = 0；
- metrics 全部 finite。

但是，v5b 仍然未达到最终 quality gate：

- Contact J 相对 persistence 的提升约为 `+0.012 ~ +0.018`，低于要求的 `+0.03`；
- Weak Contact J 相对 persistence 的提升约为 `+0.014 ~ +0.017`，低于要求的 `+0.03`；
- Transient Contact J 相对 persistence 的提升约为 `+0.015 ~ +0.017`，低于要求的 `+0.03`；
- 96 ns RMSD relative improvement 约为 `9.77% ~ 9.98%`，低于要求的 `10%`；
- qshuffle 与 oracle 几乎完全相同，说明 decoder 仍然没有实质性使用 residue-specific Q。

因此，v5b 只能被标记为：

```text
partial improvement, not final success
```

不要把它描述为已解决、已收敛或最终 SOTA。

---

## 2. 最终目标

最终目标是获得一个在 held-out protein-disjoint evaluation 上，具有显著、稳定、可复现改善的 Stage C model。

必须同时评估：

- direct mode；
- autonomous mode；
- oracle mode；
- persistence baseline；
- qshuffle diagnostic。

最终 quality gate 至少要求 direct 和 autonomous 两个 mode 同时通过。oracle 不能替代 direct 或 autonomous。

---

## 3. Persistence-relative 严格 quality gate

由于 long-horizon 的绝对 RMSD 受到数据本身可预测性的限制，不使用统一的 `RMSD < 2 Å` 作为 long-horizon 硬条件。

对每个正式 horizon，使用同一批 held-out samples 计算 model 和 persistence 的 metric。

设：

```text
RMSD_p          = persistence RMSD
ContactJ_p      = persistence contact Jaccard
WeakJ_p         = persistence weak contact Jaccard
TransientJ_p    = persistence transient contact Jaccard
```

定义：

```text
RMSD_skill = 1 - RMSD_model / RMSD_p

Contact_skill =
    (ContactJ_model - ContactJ_p) / max(1 - ContactJ_p, epsilon)

Weak_skill =
    (WeakJ_model - WeakJ_p) / max(1 - WeakJ_p, epsilon)

Transient_skill =
    (TransientJ_model - TransientJ_p) /
    max(1 - TransientJ_p, epsilon)
```

### RMSD gate

对 direct 和 autonomous 的每一个正式 horizon，必须同时满足：

```text
RMSD_skill >= 0.10
RMSD_p - RMSD_model >= 0.15 Å
```

### Contact J gate

对每一个正式 horizon：

```text
ContactJ_model - ContactJ_p >= 0.03
Contact_skill >= 0.10
```

### Weak Contact J gate

```text
WeakJ_model - WeakJ_p >= 0.03
Weak_skill >= 0.10
```

### Transient Contact J gate

```text
TransientJ_model - TransientJ_p >= 0.03
Transient_skill >= 0.10
```

这些条件必须对每一个正式 horizon 成立，不能只报告平均 horizon 结果，也不能只挑选表现最好的 horizon。

---

## 4. 统计和稳定性 gate

所有最终 aggregate 必须以 protein 为 bootstrap cluster，而不是把同一 protein 的多个 seed 或多个窗口当成独立 protein。

最终候选模型必须满足：

1. 全部正式 samples 成功处理；
2. failed sample 数为 0；
3. metric 中没有 NaN/Inf；
4. RMSD improvement 的 protein-clustered bootstrap 95% CI 下界大于 0；
5. Contact J improvement 的 bootstrap 95% CI 下界大于 0；
6. Weak J improvement 的 bootstrap 95% CI 下界大于 0；
7. Transient J improvement 的 bootstrap 95% CI 下界大于 0；
8. 每个主要 metric 的 protein-level win rate 至少为 60%；
9. autonomous rollout 不得有不可接受的 divergence；
10. direct 和 autonomous 都必须通过所有 horizon 的 gate。

如果 point estimate 达标，但 confidence interval 或 win rate 不达标，不能宣布成功。

---

## 5. Q-dependence 是独立硬条件

最终模型不仅要表现变好，还必须证明 decoder 真的使用了 Q。

当前 v5b 的主要问题是：

```text
oracle ≈ qshuffle ≈ direct
```

这表示 decoder 输出几乎不随 residue-specific Q 改变。仅仅提高 contact 或 RMSD，而 qshuffle 仍与 oracle 几乎相同，不能视为完整成功。

最终 candidate 必须通过 Q-dependence integrity check：

1. qshuffle 必须在结构 metric 上显著劣于正确 Q；
2. qshuffle 的退化不能只是浮点噪声；
3. 至少两个正式 horizon 必须显示出 qshuffle degradation；
4. degradation 必须有 protein-clustered bootstrap 统计支持；
5. 至少满足以下之一：

```text
qshuffle RMSD - correct-Q RMSD >= 0.10 Å
```

或：

```text
correct-Q Contact J - qshuffle Contact J >= 0.02
```

或对应的 Weak/Transient Contact J 出现同等有意义的统计退化。

如果 qshuffle 与 oracle 仍然几乎一样，即使 persistence-relative quality gate 暂时通过，也不能宣布最终完成。

---

## 6. Evaluation integrity rules

必须使用 canonical evaluation 实现和当前官方 geometry path：

```text
scripts/eval_world_model.py
protein_jepa_frame/eval/structure_metrics.py
protein_jepa_frame/models/stage_c/openfold_atom37.py
protein_jepa_frame/eval/stage_c_checkpoint.py
```

不得修改 metric 定义、persistence 定义、mask、horizon、split 或 failure protocol 来制造成功。

model 和 persistence 必须使用完全相同的：

- protein samples；
- horizons；
- residue mask；
- chain-aware pair mask；
- Kabsch protocol；
- target structures；
- atom ordering。

不允许：

- 静默跳过失败 sample；
- 将 NaN/Inf 转换成 0；
- 只汇报成功 sample；
- 使用未来坐标训练 direct/autonomous prediction；
- 使用 future H 或 future Q 作为 direct/autonomous 输入；
- 用 oracle 结果替代 direct/autonomous；
- 通过改变 metric 统计方式降低阈值；
- 通过挑选 checkpoint 或 horizon 伪造稳定性。

必须确认 context Q、future Q、oracle Q 和 rollout Q 的状态空间协议一致，并且 training/evaluation 使用一致的 H path。

---

## 7. 不得无脑延长训练

训练时间和训练步数不能成为默认解决方案。

必须遵守：

1. 不得因为指标未达标就无限追加 steps；
2. 不得重复运行相同配置直到随机碰巧达标；
3. 不得只保存偶然表现最好的 seed；
4. 不得使用 test split 选择模型；
5. 不得把更长训练时间本身当成方法改进；
6. 不得在没有明确实验假设的情况下扩大预算；
7. 不得因为出现小幅改善就停止。

每次修改必须有明确的实验记录、独立 output directory 和可复现 checkpoint。

如果确实需要改变训练预算，必须先给出证据说明当前模型处于 underfitting，而不是平台期或结构性失败。训练预算增加必须是受控实验，不得无限制扩大。

当前已有 step 50、100、150、200、250、300、350、400、450、500、550、600 等 checkpoint。不要假设 step 400 自动是最优，也不要假设 step 600 一定更好。需要使用统一 evaluation protocol 做可比验证。

---

## 8. 自主研究和修改要求

不要按照预先写好的固定修复清单机械修改。

你需要自主判断：

- 当前瓶颈究竟是 decoder、Q representation、H shortcut、Stage B rollout、数据协议、loss 对齐、frame geometry、evaluation mismatch 还是其他问题；
- 当前观测到的改善是否具有因果意义；
- 哪些修改能够让 decoder 真正利用 Q；
- 哪些修改只是让模型更接近 context-only 或 persistence-like predictor；
- 哪些修改会引入 future leakage 或 evaluation/training mismatch。

每轮实验只需要提交：

1. 本轮假设；
2. 修改内容；
3. 训练结果；
4. evaluation 结果；
5. 与上一轮的差异；
6. 是否通过当前 gate；
7. 下一轮计划。

具体采用什么架构、loss、训练阶段、数据策略或数值方法，由你自行调研和决定。

---

## 9. 推荐的迭代纪律

每轮按以下顺序执行：

1. 检查当前 workspace、进程和 checkpoint；
2. 运行相关 unit tests；
3. 运行小规模 smoke evaluation；
4. 读取 raw per-protein records；
5. 分析 RMSD、四类 contact metrics、persistence、qshuffle 和 failure；
6. 提出一个明确假设；
7. 进行一次受控修改；
8. 用独立目录训练；
9. 运行 validation evaluation；
10. 若有稳定改善，再进行完整 validation；
11. 对候选模型运行统一 bootstrap；
12. 只有最终候选才运行正式 held-out test。

不要在没有分析上一轮的情况下同时启动大量不同实验。

建议每轮保存：

```text
artifacts/iter_000_baseline/
artifacts/iter_001_*/
artifacts/iter_002_*/
...
```

禁止覆盖历史实验。

---

## 10. 最终停止条件

只有满足下面全部条件时，才允许停止：

1. direct 在每个正式 horizon 通过 RMSD gate；
2. direct 在每个正式 horizon 通过 Contact J gate；
3. direct 在每个正式 horizon 通过 Weak J gate；
4. direct 在每个正式 horizon 通过 Transient J gate；
5. autonomous 在每个正式 horizon 通过 RMSD gate；
6. autonomous 在每个正式 horizon 通过 Contact J gate；
7. autonomous 在每个正式 horizon 通过 Weak J gate；
8. autonomous 在每个正式 horizon 通过 Transient J gate；
9. 所有正式 samples 无失败；
10. 所有关键改善 bootstrap CI 下界大于 0；
11. protein-level win rate 至少为 60%；
12. autonomous rollout 无不可接受 divergence；
13. qshuffle 明显劣于正确 Q；
14. final checkpoint、config、code revision 和 raw records 全部可追溯；
15. 结果不是通过无脑增加训练 steps 得到的；
16. 另一个 agent 可以用相同命令复现结果。

如果没有满足全部条件，必须明确写：

```text
当前仍未达标，继续迭代。
```

不得因为相对上一版有小幅改善就写“完成”“成功”“SOTA”或“可以停止”。

不要设置任意的自动停止步数。只要资源和外部指令允许，就继续围绕最终 gate 进行有记录、有假设、有验证的迭代优化。

