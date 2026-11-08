# H+S 表征反馈诊断实验（2026-11-01）

本协议在两臂训练前写定。目标是检验 Stage B 的自主预测梯度能否改进当前 H，
并保持结构读出；这不是全面超越 BioEmu2 的完成判据。

## 已有证据与假设

当前 CA 主线原本冻结 encoder/decoder，仅训练 kernel。新增路径使观测历史的
encoder 接收完整 H/S rollout 与历史后验的梯度，同时用固定起始 encoder 作
detached 未来 latent 目标，并约束观测 H 的绝对值和时间差。decoder 固定。
精确 S0 的能量/历史导数已用有限差分修复；系数在一个 rollout 内仍固定于锚点。

初始组件的 standard200 oracle readout gate 在 16/64 ns 未达到本轮指定的
oracle/persistence RMSD <= 1/3 预算（实测 0.357/0.427）。因此本轮是诊断探索，
不能将训练完成或短期 loss 降低称为正式模型通过。

## 对照与预算

- 两臂 `frozen` / `joint`，仅后者打开 `--train-encoder`，seed=42。
- 同一起点：`artifacts/lasp_s0_arms_20261031/exact_seed42/checkpoint_best.pt`。
  恢复它内嵌的 encoder/decoder 和 kernel；该历史 checkpoint 使用相关噪声。
  本轮两臂同时采用 collective mobility、`noise_mix=0`、exact S0，属于显式
  配方改变，不重命名或覆盖历史结果。
- 500 个新增优化步，horizons=1/4/16/64，12 samples，完整64步反传，
  同一 stateless train 窗口流；训练/验证 manifest 均沿当前 full 划分。
- kernel LR=1e-4，encoder LR=1e-5，50步 warmup＋cosine；teacher absolute=1，
  teacher delta=.1。其余 loss 权重保持启动脚本所列的当前配方。
- 验证每100步、每h16窗，保存 val-selected best 与最终 latest；两者都评估，
  避免把 step0 被选为best误报成训练收益。
- 该验证集已经多次参与历史开发，所有结果只能是开发诊断。test 保持封存。

## 评测与继续规则

1. 必须完成两个相同预算的训练；检查参数实际变化、encoder梯度、无非有限值、
   teacher和decoder保持原值、checkpoint重载与继续训练一致。
2. 对两臂 best/latest 做固定 standard200、12样本、同窗口与噪声的七项指标；
   继承历史 endpoint 对 future-path-mixture 口径并明确标记。按h报告全部结果，
   不把它当作时间条件端点或动力学证明。
3. `joint` 对 `frozen` 的优先关注项为 rmsf_corr 和 md_pca_w2；其余五指标及
   CA几何全部报告。蛋白等权配对bootstrap只作探索区间，不宣称全家族显著性。
   任一主要指标改善伴随其他指标恶化，必须明确记为权衡。
4. 重跑两个组件的 H readout gate；若退化则该轮不能作为最终候选，优先诊断
   表征漂移与decoder误差。达到本轮预算仍不证明H包含全部关键信息。
5. 只有出现一致的开发收益、无显著几何/读出退化，才扩大到三seed及完整
   温度/T2A/轨迹/T1套件。单seed失败不推出框架无效或不存在记忆。
6. 最终目标仍包括权威评测文档中的所有适用指标、独立保留评估与公平BioEmu2
   对照；本轮的通过不替代该目标。

启动：`bash run_scripts/run_hs_joint_pilot.sh`。
产物：`artifacts/hs_joint_20261101/pilot_v1/{frozen,joint}`。
