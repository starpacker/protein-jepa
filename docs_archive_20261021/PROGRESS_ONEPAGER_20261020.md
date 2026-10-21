# Protein-JEPA-Frame — 一页汇报（2026-10-20）

## 做什么
给定 mdCATH 蛋白的 **anchor 帧**（+ 短历史），预测**未来构象系综 / 未来帧轨迹**（CA）。
衡量"预测的动力学分布"与真实 MD 的差距，对手是最诚实的 **persistence**（anchor 复制）。

## 两套模型（两条技术路线）

| | 模型 1 · Stage-A ensemble | 模型 2 · from-none GLE |
|---|---|---|
| 表征 | JEPA 式**预训练** encoder（冻结） | **随机**初始化 encoder（无预训练） |
| 动力学 | Gaussian H **系综头**采样 | **GLE**（广义 Langevin）随机 latent 演化 |
| 可训练量 | 只训练轻量系综头 | 端到端全量 |
| 训练数据 | mdCATH-1000dom + ATLAS（200dom 为旧版） | mdCATH-1000dom |
| 训练目标 | RMSF/contact/内距 分布 + 位置项 | CA RMSD + 系综分布项（RMSF-r / 方差 / spread） |
| 一句话 | 强表征 + 轻头 | 从零端到端 + 显式随机动力学 |

## 三套 baseline
1. **persistence**：anchor 复制（两个文档都用的"必须打赢"的对手）
2. **MDGen zero-shot**（`atlas.ckpt`，官方权重，无微调）
3. **bioemu2 zero-shot**（`bioemu2-preview`，微软 diffusion 系综，无微调）

## 核心结果

### 模型 vs persistence（mdCATH，同一窗口）
| 指标 | 模型 1（1000dom val，新） | 模型 2 v4e（1000dom val） |
|---|---|---|
| RMSF 幅度 | ✅ 1.154 vs pers 1.772 | ✅ 100% 窗口胜 |
| **RMSF-r（形态）** | ✅ **0.538**（global）vs pers 0.000 | ✅ **0.51 / 0.50 / 0.50**（4/16/64 ns）vs ~0 |
| 3×W2（root/var/pca） | ✅ 全部优于 pers | ✅ var/pca 100% 胜；root 短时标 ✗ |
| 接触类 Jaccard | ✅ 优于 pers | ✅ transient 100% 胜 |
| **CA 位置（Kabsch）** | ⚠️ 7.58 Å vs pers 3.31 Å（rollout 漂移） | ⚠️ root_W2 短时标失败 |

> **口径已统一**：模型 1 现已在 **1000dom** 重训并评估（附录 A），与模型 2 / MDGen / bioemu2 同 val 集。

### MDGen zero-shot 在 mdCATH 上（用户重点）
| | Kabsch CA RMSD | RMSF |
|---|---|---|
| persistence | **4.01 Å** | **1.84 Å** |
| MDGen zero-shot | 7.34 Å（seed7: 7.84） | 4.65 Å |
| 我们的 rollout | 6.34 Å | 1.46 Å |

**MDGen zero-shot 劣于 persistence**，主因 **10 ps→1000 ps/帧 步长 OOD**（+ 蛋白族不同）；
MDGen mdCATH 短训远未收敛（Kabsch 19.7→26.8 Å，震荡）。

## 产物
- 整合报告：`CURRENT_PROGRESS_20261020.md`
- 轨迹 PDB（肉眼检查）：`artifacts/pdbs_20261020/`（6 条轨迹 × 3 窗口 × 16 帧，CA-only）
- 指标代码：`evaluation/ensemble_metrics{,_ext}.py`；**精确坐标解码路径未改动**；单测 26 项通过

## 诚实结论
- ✅ **系综分布指标（RMSF 谱/形态 + W2 + 接触类）稳定优于 persistence** —— 这是目前最扎实的成果。
- ⚠️ **CA 绝对位置精度仍是短板**（rollout 漂移 / 短时标 root_W2 失败）。
- ⚠️ 模型 1 已统一到 1000dom；MDGen/bioemu2 为 **zero-shot**（MDGen 还是 OOD 步长），不代表其正常水平。
- ✅ bioemu2 已完成：**长时标（16–64 ns）显著优于 persistence**（RMSF-r ~0.6，接触 Jaccard 更高）；
  **短时标（1 ns）差于 persistence**（扩散系综 ≠ 短时确定性推进）。
