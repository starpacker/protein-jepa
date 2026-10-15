# Stage B1a 诊断报告

日期：2026-10-15  
当前状态：**oracle_ca_mse 未收敛（frozen/unfrozen），same-frame oracle 已通过**

---

## 1. 已完成的训练轮次汇总

| 运行目录 | steps | unfreeze | same_frame | oracle_ca_mse@0 | oracle_ca_mse@end | 结论 |
|---|---|---|---|---|---|---|
| stage_b1a_sameframe | 1000 | 0 | ✓ | 0.00 | ~0.00 | **通过**：decoder 功能正常 |
| stage_b1a_frozen | 5000 | 0 | ✗ | ~125 | ~142–176 | 未收敛 |
| stage_b1a_fix4_frozen | 3000 | 0 | ✗ | ~125 | ~142–176 | 未收敛 |
| stage_b1a_fix4_unfrozen | 3000 | 2 | ✗ | ~125 | ~142–176 | 未收敛 |

---

## 2. 关键发现：same-frame oracle 完全通过

`stage_b1a_sameframe` 的日志显示 `oracle_ca_mse` 在 step 0 就是 **0.000**，之后保持在 0.000–0.0004。
这证明：
- Decoder（IPA + direct_h_position head）结构正确，能从 H 重建坐标
- H normalizer 没有阻断梯度
- 损失计算、坐标对齐逻辑均正常

same-frame 通过意味着 **解码器没有 bug**，问题在别处。

---

## 3. 真正的问题：future-frame H 不含足够的绝对位置信息

same-frame 和 future-frame 的唯一区别在于提供给 decoder 的 H 是哪一帧编码的：

- **same-frame**：`H = encoder(target_coords)` — decoder 拿到的 H 正好描述 target 结构，ca_mse → 0
- **future-frame**：`H = encoder(future_coords)` — decoder 拿到的 H 描述未来帧，但 anchor 是当前帧 `R_anchor, t_anchor = _extract_backbone_frames(ctx_coords)`

`oracle_ca_mse` 衡量的是：**以当前帧为锚点出发，用 future H 引导，能否重建出 future 绝对坐标**。

ca_mse 停在 ~125 Å²（对应 ~11 Å RMSD），这正是蛋白质跨帧刚体位移的典型量级（涉及平移+旋转）。
这说明 **decoder 没有从 future H 中提取到足够的绝对位移信号，来弥合 anchor 和 target 之间的刚体偏移**。

---

## 4. 根因分析

### 4a. Stage A 的 H 是 SE(3) 不变表示，不含全局位置（最可能的根本原因）

Stage A 训练时使用的特征（`STAGE_AB_REDESIGN_PLAN.md §1.2`）是严格 SE(3) 不变的：
- backbone 二面角、CA-CA 距离、相对局部框架方向

这意味着 `encoder(frame_t)` 和 `encoder(frame_t+τ)` 编码的 H 只反映**局部几何构象**，
不含跨帧的全局平移/旋转信息。

IPA decoder 以 `R_anchor, t_anchor`（当前帧）为起点，依赖 H 提供的信号做迭代更新。
但如果 H_future 相对 H_current 的差异主要是刚体运动，而不是内部构象变化，
decoder 无法从中提取出 ~11 Å 的平移量，ca_mse 就会停在刚体偏移水平。

### 4b. direct_h_position head 零初始化，初始增量为零

`structure_decoder.py:605-606`：
```python
nn.init.zeros_(self.direct_position_head[-1].weight)
nn.init.zeros_(self.direct_position_head[-1].bias)
```
Step 0 的 direct position update 为零，依赖梯度从零开始学。
如果 H 本身不含位移信号（4a），梯度会很小，head 的收敛极慢或停滞。

### 4c. H normalizer 压缩了 H 的动态范围（次要因素）

normalizer 做 per-channel z-score，将 H 的每个通道压缩到均值 0 标准差 1。
跨帧的细微构象差异在 H 空间本就很小，归一化后可能进一步被压缩，降低 decoder 的可用信号。

---

## 5. 建议的修复方向（供 agent 决策）

### 方案 A：在 Stage A H 之外，显式附加帧间位移/旋转特征（推荐）

B1a 的 oracle 任务本质上是要从 H_future 恢复 **future 帧的绝对坐标**，
这需要 H 携带相对当前帧的刚体变换信息。

改法：在喂给 decoder 之前，拼接一个显式的帧间特征：
```python
# 伪代码
delta_t = t_future_com - t_current_com     # [B, 3] 质心位移
H_aug = cat([H_future, delta_t.unsqueeze(1).expand_as(H_future[..., :3])], dim=-1)
```
或者用更完整的相对刚体表示（四元数 + 平移）。

### 方案 B：修改 ca_mse 的参考系，改为 Kabsch 对齐后测量（降低目标难度）

当前 `oracle_ca_mse` 测的是**绝对坐标**误差，包含刚体偏移。
如果改为测 Kabsch 对齐后的 RMSD（即内部构象误差），Stage A 的 H 应该已经足够。
这样 ca_mse 会立即下降，但代价是 decoder 不再学全局位置——需要确认这是否符合 B1b 的需求。

### 方案 C：Stage A 补充训练，让 H 包含帧间信息

在 Stage A 的 OPF 目标之外，加入一个辅助预测头：用 `H_t - H_{t+τ}` 预测 CA 位移向量。
这样 Stage A 的 H 本身就会编码局部运动信息，B1a 的 decoder 就有信号可用。

### 方案 D（快速验证）：将 anchor 也换成 future 帧（zero-displacement oracle）

临时把 `R_anchor, t_anchor = _extract_backbone_frames(tgt_coords)`（而非 ctx_coords）。
如果 ca_mse 立即收敛（和 same-frame 一样），确认根因是刚体位移问题，而非 H 构象编码问题。

---

## 6. 文件位置

| 文件 | 说明 |
|---|---|
| `protein_jepa_frame/training/train_stage_b1a.py` | 训练主循环，`_step()` 函数 |
| `protein_jepa_frame/models/stage_b1a/model.py` | `StageBOracleDecoder`，`HNormalizer` |
| `protein_jepa_frame/models/stage_b1a/objectives.py` | `compute_stage_b1a_loss` |
| `protein_jepa_frame/models/stage_c/structure_decoder.py:597-610` | `direct_h_position` head 初始化 |
| `protein_jepa_frame/models/stage_a_v2/model.py` | Stage A encoder（SE(3) 不变） |
| `run_b1a_sameframe.sh` | same-frame smoke test（已通过） |
| `run_b1a_frozen.sh` / `run_b1a_unfrozen.sh` | future-frame 训练（未收敛） |
| `artifacts/abc_v3/stage_a_fix4_frozen/checkpoint_step6450.pt` | 当前最佳 Stage A ckpt |
| `artifacts/abc_v3/stage_b1a_fix4_unfrozen/checkpoint_latest.pt` | 最新 B1a ckpt（step 3000） |
