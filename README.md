# Protein-JEPA-Frame

这是一个从零整理的 protein dynamics 版 V-JEPA 框架。目标不是复刻 `/liuyutian/protein_jepa` 中混杂的 MSM-label 训练路线，而是保留 V-JEPA 的核心机制：

- context encoder 只看可见的时空 residue token；
- target encoder 是 context encoder 的 EMA 拷贝，只在 no-grad 下编码完整结构；
- predictor 用 visible latent、时间/序列位置和 mask token 预测被遮蔽 token 的 target latent；
- loss 只在 latent space 上计算，不重建坐标，不把 MSM state 当训练标签。

## 设计方案

蛋白质轨迹样本统一成：

```python
{
    "coords": FloatTensor[T, L, 4, 3],      # N, CA, C, O
    "atom_mask": BoolTensor[T, L, 4],
    "residue_mask": BoolTensor[T, L],
    "aatype": LongTensor[L],
    "residue_index": LongTensor[L],
    "chain_id": LongTensor[L],
    "time_ps": FloatTensor[T],
}
```

V-JEPA 的视频 patch `[t, h, w]` 被替换为 protein token `[t, residue]`。mask 支持连续时间块、连续 residue 块和 spatiotemporal block。MSM label 可以存在于 manifest/npz 里，但只能用于采样、QC 和 evaluation，不进入 encoder/predictor。

## 快速验证

```bash
cd /liuyutian/protein_jepa_frame
python -m pytest
python -m protein_jepa_frame.cli smoke --output-dir artifacts/smoke --steps 5
```

## 训练真实轨迹

准备 `manifest.jsonl`，每行一个 trajectory：

```json
{"npz_path":"/abs/path/traj.npz","protein_id":"1abc_A","trajectory_id":"1abc_A_R1","num_frames":10000,"frame_interval_ps":10.0}
```

`traj.npz` 至少包含 `coords`，推荐同时包含 `atom_mask/residue_mask/aatype/residue_index/chain_id/time_ps`。

```bash
python -m protein_jepa_frame.cli train --config configs/atlas_baseline.yaml
```

## 当前实现边界

默认 encoder 是轻量 SE(3)-invariant backbone geometry encoder，已经完整支持 V-JEPA 式 EMA target、visible-only 构图、spatiotemporal predictor 和 checkpoint。ProteinMPNN/ESM3 可以后续接入 `ProteinJEPA` 的 encoder 接口，但第一版先保证训练目标干净、无 MSM supervision、无 masked-coordinate leakage。
