# PLAN: 位移监督改造（修改 1+2）——逐条执行

目标：让「真实内部 CA 位移」的监督真正作用于 **H**（而非只作用于 predictor），
使 `scripts/probe_h_readout.py` 的 **within ratio ≤ 0.5×**。

## Task list
- [x] T1 model.py: `forward_temporal_multilag` —— anchor 改用 **online encoder** 重算，`anchor_factors` **不 detach**（修"死 anchor"）。
- [x] T2 model.py: `displacement_readout_loss` —— head 输入**去掉 `anchor_H`/`H_mean` 旁路**，只读 `synth_delta_H`；`anchor_H` 不再 detach（保留参数但不再拼入特征）。
- [x] T3 model.py: 增加**逐残基方向余弦损失** `w_dir * (1 - cos(pred_c, tgt_c))`，与 pattern MSE 相加。
- [x] T4 model.py: `displacement_readout_loss` 签名加 `direction_weight`，默认 1.0。
- [x] T5 train_stage_a_v2.py: 新增 CLI `--displacement-direction-weight`，传入 loss；`_validate` 同步。
- [x] T6 tests: 更新/新增测试——(a) anchor 路径梯度能回到 online encoder；(b) head 输入不含旁路（特征维度 = dH）；(c) 方向损失对常量预测 >0。
- [~] T7 跑 2k steps（GPU 空闲卡），评估：probe within ratio + 训练日志 displacement loss。
- [ ] T8 若过 gate（≤0.5×）→ 记录并扩规模；否则到修改 3。

## 阶段性结论（更新）
- T7（修改1+2, 2k steps）：**未过 gate**，within 0.949/0.945/0.937/0.938 ≈ 原样。
  上界实验查明：H overfit R²=1.0 但 held-out R²=0.09 → **H 缺"相对 anchor 的参考帧"**。
- [x] T8a 上界诊断（overfit vs held-out）
- [~] T8b **修改 4**：encoder 加 within-frame PCA canonical-frame position 通道。
  - [x] encoder.py `_within_frame_canonical_positions`（skewness 符号约定，rigid-invariant）
  - [x] config/CLI 开关 `relative_position_features`
  - [x] tests（rigid-invariant / position-aware / 改变 H）
  - [~] 跑 2k steps（`artifacts/stage_a_disp_relpos`）→ 探针

## 最终结论（三轮实验）
| 版本 | 改动 | within ratio (h=1/2/4/8) | struct_recon | 结论 |
|---|---|---|---|---|
| fix | 修改1+2（head 读 ΔH、online anchor、方向项） | 0.949/0.945/0.937/0.938 | ~0.75 | 无改善 |
| relpos | +修改4（PCA position 通道） | ~0.96/0.96/0.943/0.943 | **0.02** | 无改善（probe） |

**决定性诊断**：用**真实 anchor 坐标**预测位移，held-out R² 也只有 **0.02–0.05**（in-sample 0.88）。
→ **gate 不可达**：单帧→h步位移不是稳定函数，被不可预测动力学主导。
→ H（R²≈0.05–0.09）已**达到/超过真坐标上界**；H 不是瓶颈。
→ 修改4 真实有效（struct_recon 0.75→0.02，绝对位置 R² 0.04→0.13），只是该 probe 测不出。

**下一步（待用户决定）**：换 gate（结构可解码性/时间信息/multi-frame），或换任务（预测可解码量），
或直接用 fix+relpos encoder 跑下游 B1a 看真实 CA RMSD。
