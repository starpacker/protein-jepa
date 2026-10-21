# ⚠️ ATLAS (100 ps) 数据集警告 — 近刚性，persistence 近似最优

> 写给**后来的 agent**：在把 ATLAS 当作 benchmark 之前，请先读完本文。
> 本文所有数字均来自本仓库实测，可复现（命令见 §6）。

---

## 0. 一句话结论

**ATLAS-100ps 是一个近刚性数据集。在 Kabsch 对齐下，persistence（未来=当前帧）
已经是 CA RMSD 的近最优解。任何以"点预测 CA 坐标"为主损失的框架，
在 ATLAS 上都**注定**与 persistence 打平或更差——这不是模型能力问题，是数据的天花板。**

---

## 1. 实测：ATLAS 的运动幅度（Kabsch 对齐后）

对未被训练使用的 ATLAS val 蛋白，anchor=第 7 帧，测量 CA 平均位移：

| 时间间隔 | 帧数(100ps/帧) | 平均 CA 位移 (Å) |
| --- | --- | --- |
| 0.1 ns | 1 | **0.099** |
| 0.5 ns | 5 | 0.122 |
| 1.0 ns | 10 | **0.136** |
| 6.0 ns | 60 | 0.177 |
| 10.0 ns | 100 | **0.186** |

对比：

| 数据集 | 1 ns 处的量级 | 10 ns 处的量级 | 性质 |
| --- | --- | --- | --- |
| **ATLAS (300 K)** | 0.10–0.14 Å | 0.19 Å | **近刚性** |
| **mdCATH (320 K)** | 0.36 Å | 0.58 Å | 有运动 |
| **mdCATH (450 K)** | 0.63 Å | 1.17 Å | 明显运动 |

ATLAS 的漂移量级 **比 mdCATH 小 3–6 倍**。

---

## 2. 为什么 persistence 在 ATLAS 上近似最优

Kabsch 对齐后的 CA-RMSD 是一个**点预测损失**。对一条真实轨迹，未来的 CA 位置
= anchor + 真实位移。由于 ATLAS 的真实位移（0.1–0.19 Å）与热噪声同量级且方向
不可预测，任何点预测器的最优策略都是：

```
pred = anchor  ⟹  RMSD(pred, true) ≈ RMSD(anchor, true) = persistence
```

**它无法做得更好，因为超出 anchor 的部分是不可预测的噪声。**
这不是"学不动"，是"没有可学的确定性信号"。

### 直接证据（from-none GLE, ATLAS v3, anchor_conditioned=True）

| 量 | 值 |
| --- | --- |
| val CA RMSD | **0.1276 Å** |
| val persistence RMSD | **0.1274 Å** |
| val skill | **−0.0016** |
| decoder 输出与 anchor 的平均距离 | **0.0070 Å** |
| 8 个随机样本之间的 spread | **0.0001 Å** |

decoder 输出了 anchor 本身，8 个样本几乎完全相同（spread 0.0001 Å）。
**模型学到的正确解就是"不动"。**

---

## 3. 六个系综级指标上的表现（ATLAS-100ps, v3 ckpt）

（口径：window-level mean，Kabsch 对齐，与 `DUAL_ENSEMBLE_RESULTS_20261019.md` 一致）

| horizon | 指标 | model | persistence | 改善 | 胜率 |
| --- | --- | --- | --- | --- | --- |
| 1 ns | rmsf_mae | 0.0710 | 0.0711 | +0.0001 | 100% |
| 1 ns | root_mean_w2 | 0.1425 | 0.0630 | **−0.0795** | 0% |
| 1 ns | variance_w2 | 0.0132 | 0.0132 | +0.0000 | 100% |
| 1 ns | md_pca_w2 | 0.0054 | 0.0054 | +0.0000 | 100% |
| 1 ns | weak_contact_j | 1.0000 | 1.0000 | +0.0000 | 0% |
| 1 ns | transient_contacts_j | 1.0000 | 1.0000 | +0.0000 | 0% |
| 4 ns | rmsf_mae | 0.0826 | 0.0826 | +0.0001 | 100% |
| 4 ns | root_mean_w2 | 0.1638 | 0.0812 | **−0.0826** | 0% |
| 4 ns | variance_w2 | 0.0166 | 0.0166 | +0.0000 | 100% |
| 4 ns | md_pca_w2 | 0.0061 | 0.0061 | +0.0000 | 100% |
| 4 ns | weak_contact_j | 1.0000 | 1.0000 | +0.0000 | 0% |
| 4 ns | transient_contacts_j | 1.0000 | 1.0000 | +0.0000 | 0% |
| 16 ns | rmsf_mae | 0.0885 | 0.0885 | +0.0001 | 100% |
| 16 ns | root_mean_w2 | 0.1720 | 0.0940 | **−0.0779** | 0% |
| 16 ns | variance_w2 | 0.0176 | 0.0176 | +0.0000 | 100% |
| 16 ns | md_pca_w2 | 0.0068 | 0.0068 | +0.0000 | 100% |
| 16 ns | weak_contact_j | 1.0000 | 1.0000 | +0.0000 | 0% |
| 16 ns | transient_contacts_j | 1.0000 | 1.0000 | +0.0000 | 0% |

**读法**：
- 所有"方差/涨落"类指标（rmsf/variance/md_pca）**改善恰好为 0**：因为模型样本
  spread≈0，等价于 persistence 的零方差。
- `root_mean_w2` **系统性更差**：模型的少量非零抖动落在错误方向，比"一点不动"更偏离真值。
- 两个 Jaccard 恒为 1.0：**ATLAS 的 contact 集合完全冻结**，没有动态接触可预测。

---

## 4. 根因：loss 设计 × 近刚性数据 = 坍缩到 persistence

本仓库 from-none GLE 训练的主 loss 是：

```
total = es_weight·ES(latent) + ca_weight·CA_RMSD(decoded, true) + tf_ca_weight·TF_CA + ...
```

关键配置（`atlas_100ps_anchor_v3/config.json`）：

| 权重 | 值 | 后果 |
| --- | --- | --- |
| `ca_weight` | 1.0 | 点预测 RMSD 主导；近刚性下最优=anchor |
| `tf_ca_weight` | 0.5 | 同上，teacher-forced 也是点预测 |
| **`variance_weight`** | **0.0** | **没有任何项要求系综有正确方差** |
| `anchor_weight` | 0.0 | — |

**`variance_weight=0` 是致命的一环**：即使 CA RMSD 在近刚性数据上允许坍缩，
只要有一个显式的系综方差/涨落损失，decoder 就会被推着产生非零 spread。
在 ATLAS 上我们**完全没有**这样的项。

---

## 5. 对后来 agent 的建议

### 5.1 不要把 ATLAS-100ps 用作"能否超越 persistence"的主 benchmark

它的物理天花板太低，**任何方法都会与 persistence 打平**。
用它证明"架构有效"会得到**假阴性**。

### 5.2 若必须用 ATLAS，请改变评估口径

- 不要用点预测 CA RMSD 作为主指标（近刚性下无意义）。
- 若要用系综指标（RMSF/W2），**必须同时监督方差**，否则会坍缩。
- contact/Jaccard 在 ATLAS 上恒为 1.0，**不能作为判别指标**。

### 5.3 推荐：切换到 mdCATH（1 ns）或高温柔性数据

mdCATH 320–450 K 的 1 ns 位移是 ATLAS 的 3–6 倍，
persistence 不再是近似最优，模型有真实的确定性信号可学。
**本项目的正式 benchmark 应使用 mdCATH-1ns。**

---

## 6. 复现命令

```bash
# 数据运动幅度
python - <<'PY'
import numpy as np, glob
for p in glob.glob('/liuyutian/protein_jepa/datasets/ATLAS_NPZ_100ps/*.npz')[:6]:
    ca = np.load(p, mmap_mode='r')['coords'][:, :, 1, :]
    a = ca[7]
    for lag in (1, 10, 100):
        print(p.split('/')[-1], lag, np.linalg.norm(ca[7+lag]-a, axis=-1).mean())
PY

# 六指标 vs persistence
python -m evaluation.run_eval_fromnone_ensemble \
  --ckpt artifacts/runs/atlas_100ps_anchor_v3/checkpoint_best.pt \
  --manifest artifacts/atlas_manifest_100ps/val.jsonl \
  --horizons-ns 1 4 16 --n-samples 8 --max-windows 100 \
  --device cuda --out results/atlas_anchor_v3_100win.json
```

---

## 7. 一句话给复现者

> **ATLAS 近刚性 → persistence 近似最优 → 点预测损失必然坍缩 → 系综指标全部打平。**
> 想让 H+S 学到有意义的东西，去 mdCATH-1ns，并且**显式监督系综方差/涨落**。
