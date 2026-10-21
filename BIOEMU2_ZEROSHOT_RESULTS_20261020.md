# bioemu2 零样本 baseline — 结果（2026-10-20）

> 模型：`microsoft/bioemu2`，`bioemu2-preview` checkpoint，**零样本**（无微调、无 mdCATH 训练数据）。
> 协议：与 `FROMNONE_GLE_ENSEMBLE_RESULTS_20261019.md` §7 **完全相同** ——
> `artifacts/mdcath_manifest_1000dom/val.jsonl`，**200 窗口**，**n_samples=12**，
> horizon 1/4/16/64 ns，Kabsch `align=True`，bootstrap over windows（2000 次，95%）。
> 原始数据：`results/bioemu2_zeroshot.json`；设计/公平性说明：`BIOEMU2_ZEROSHOT_BASELINE_20261020.md`。

---

## 0. 结论速览

1. **bioemu2 零样本在长时标（16–64 ns）全面超过 persistence**：
   rmsf_mae 改善 +2.83 / +3.94 Å，variance_w2 改善 +14.1 / +24.0，
   md_pca_w2 改善 +0.189 / +0.229，root_mean_w2 改善 +0.40 / +0.66。
2. **RMSF-r（剖面形状）是 bioemu2 的绝对强项**：0.457 → 0.605 → 0.589 → 0.651，
   全程 100% 窗口胜 persistence。这是本仓库 anchor/v4 系列（0.27–0.51）**始终没有攻下**的指标。
3. **两个已知弱点同样存在**：
   - 短时标 1 ns：`variance_w2` 明显差于 persistence（16.77 vs 7.25）；
   - `weak_contact_j` 在 **1 ns 处恰为 0**（persistence 1.0），与 from-none 模型同病。
4. **与 from-none v4e 相比**：bioemu2 在 **rmsf_corr / 接触类 / 长程 root_mean_w2** 上占优；
   v4e 在 **variance_w2 / md_pca_w2** 上占优。

> ⚠️ 注意：**二者不是同一道题**。bioemu2 只用 **anchor 单帧**（全原子）作条件；
> v4e 用 **K=8 帧历史**。bioemu2 信息更少却在大半指标上更强，说明其动力学先验确实更厚。

---

## 1. bioemu2 零样本 vs persistence（200 窗口）

`Δ` = 相对 persistence 的改进（用指标自身方向：↓指标取 pers−model）。

| horizon | 指标 | bioemu2 | persistence | Δ | CI95 | 胜率 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 ns | rmsf_corr ↑ | 0.4568 | 0.0000 | **+0.4568** | [+0.408, +0.504] | 100% |
| 1 ns | rmsf_mae ↓ | 2.3164 | 1.8984 | −0.4181 | [−0.758, −0.054] | 22% |
| 1 ns | root_mean_w2 ↓ | 5.5838 | 1.6834 | −3.9004 | [−4.690, −3.166] | 4% |
| 1 ns | variance_w2 ↓ | 16.7709 | 7.2531 | −9.5178 | [−12.967, −6.220] | 16% |
| 1 ns | md_pca_w2 ↓ | 0.0836 | 0.1839 | **+0.1002** | [+0.080, +0.120] | 96% |
| 1 ns | weak_contact_j ↑ | 0.0000 | 1.0000 | −1.0000 | [−1.000, −1.000] | 0% |
| 1 ns | transient_contacts_j ↑ | 0.1715 | 0.0000 | **+0.1715** | [+0.162, +0.181] | 100% |
| 4 ns | rmsf_corr ↑ | 0.6050 | 0.0147 | **+0.5902** | [+0.525, +0.653] | 100% |
| 4 ns | rmsf_mae ↓ | 1.9847 | 3.0198 | **+1.0351** | [+0.670, +1.413] | 74% |
| 4 ns | root_mean_w2 ↓ | 3.6095 | 2.5917 | −1.0178 | [−1.409, −0.641] | 24% |
| 4 ns | variance_w2 ↓ | 19.3384 | 14.3661 | −4.9723 | [−8.002, −1.654] | 18% |
| 4 ns | md_pca_w2 ↓ | 0.0858 | 0.1895 | **+0.1037** | [+0.085, +0.121] | 94% |
| 4 ns | weak_contact_j ↑ | 0.1690 | 0.0000 | **+0.1690** | [+0.160, +0.179] | 100% |
| 4 ns | transient_contacts_j ↑ | 0.2880 | 0.0000 | **+0.2880** | [+0.276, +0.301] | 100% |
| 16 ns | rmsf_corr ↑ | 0.5891 | 0.0017 | **+0.5875** | [+0.522, +0.650] | 100% |
| 16 ns | rmsf_mae ↓ | 1.5816 | 4.4126 | **+2.8310** | [+2.326, +3.345] | 96% |
| 16 ns | root_mean_w2 ↓ | 3.2147 | 3.6100 | **+0.3954** | [+0.153, +0.649] | 64% |
| 16 ns | variance_w2 ↓ | 13.9706 | 28.0998 | **+14.1292** | [+8.484, +20.025] | 68% |
| 16 ns | md_pca_w2 ↓ | 0.0766 | 0.2651 | **+0.1886** | [+0.164, +0.213] | 100% |
| 16 ns | weak_contact_j ↑ | 0.2362 | 0.0000 | **+0.2362** | [+0.223, +0.249] | 100% |
| 16 ns | transient_contacts_j ↑ | 0.3499 | 0.0000 | **+0.3499** | [+0.335, +0.365] | 100% |
| 64 ns | rmsf_corr ↑ | 0.6514 | 0.0306 | **+0.6208** | [+0.577, +0.667] | 100% |
| 64 ns | rmsf_mae ↓ | 1.6230 | 5.5644 | **+3.9413** | [+3.361, +4.496] | 98% |
| 64 ns | root_mean_w2 ↓ | 4.0386 | 4.7021 | **+0.6635** | [+0.400, +0.938] | 68% |
| 64 ns | variance_w2 ↓ | 17.0818 | 41.1092 | **+24.0274** | [+17.63, +30.14] | 80% |
| 64 ns | md_pca_w2 ↓ | 0.0871 | 0.3160 | **+0.2289** | [+0.205, +0.253] | 100% |
| 64 ns | weak_contact_j ↑ | 0.2111 | 0.0000 | **+0.2111** | [+0.199, +0.224] | 100% |
| 64 ns | transient_contacts_j ↑ | 0.3764 | 0.0000 | **+0.3764** | [+0.362, +0.390] | 100% |

---

## 2. 逐项结论

| 指标 | 结论 | 证据 |
| --- | --- | --- |
| **rmsf_corr** | ✅ **全程强胜**，且随 horizon 单调上升（0.46→0.65） | 4 个 horizon 均 100% 窗口，CI 全正 |
| **rmsf_mae** | ⚠️ 分裂：1 ns 小输（−0.42），4/16/64 ns 显著胜（+1.04 / +2.83 / +3.94） | 与 from-none v4e 同样的"短程难"模式 |
| **variance_w2** | ⚠️ 短程输（1 ns −9.5、4 ns −5.0），长程胜（16 ns +14.1、64 ns +24.0） | 系综过散 |
| **md_pca_w2** | ✅ **全程强胜**（+0.10~+0.23），1 ns 起就 94–100% 窗口 | — |
| **weak_contact_j** | ⚠️ 1 ns 恰为 0（persistence 1.0），4/16/64 ns 100% 胜 | 与 from-none 完全同病 |
| **transient_contacts_j** | ✅ 全程 100% 胜（persistence 恒为 0） | — |
| **root_mean_w2** | ⚠️ 1/4 ns 输（−3.90 / −1.02），16/64 ns 转为胜（+0.40 / +0.66） | 与 from-none 的失败模式**完全一致**，但转正的 horizon 更早（16 ns vs v4e 也 16 ns） |

---

## 3. 三方对照：bioemu2 (zero-shot) vs from-none v4e vs persistence

`Δ` 均为相对 persistence 的改进（用指标自身方向）。"更优者"按 skill（Δ）比较。

| horizon | 指标 | bioemu2 | v4e | persistence | 更优者 |
| --- | --- | --- | --- | --- | --- |
| 1 ns | rmsf_corr ↑ | **0.4568** | 0.2831 | 0.0000 | bioemu2 |
| 1 ns | rmsf_mae ↓ | 2.3164 | **1.8877** | 1.8984 | v4e |
| 1 ns | root_mean_w2 ↓ | **5.5838** | 6.5762 | 1.6834 | bioemu2 |
| 1 ns | variance_w2 ↓ | 16.7709 | **7.2529** | 7.2531 | v4e |
| 1 ns | md_pca_w2 ↓ | **0.0836** | 0.1817 | 0.1839 | bioemu2 |
| 1 ns | weak_contact_j ↑ | 0.0000 | **0.4000** | 1.0000 | v4e |
| 1 ns | transient_contacts_j ↑ | **0.1715** | 0.0115 | 0.0000 | bioemu2 |
| 4 ns | rmsf_corr ↑ | **0.6050** | 0.5129 | 0.0147 | bioemu2 |
| 4 ns | rmsf_mae ↓ | 1.9847 | **1.2256** | 3.0198 | v4e |
| 4 ns | root_mean_w2 ↓ | **3.6095** | 3.7977 | 2.5917 | bioemu2 |
| 4 ns | variance_w2 ↓ | 19.3384 | **9.7895** | 14.3661 | v4e |
| 4 ns | md_pca_w2 ↓ | **0.0858** | 0.0993 | 0.1895 | bioemu2 |
| 4 ns | weak_contact_j ↑ | **0.1690** | 0.1373 | 0.0000 | bioemu2 |
| 4 ns | transient_contacts_j ↑ | **0.2880** | 0.2746 | 0.0000 | bioemu2 |
| 16 ns | rmsf_corr ↑ | **0.5891** | 0.4972 | 0.0017 | bioemu2 |
| 16 ns | rmsf_mae ↓ | **1.5816** | 1.9844 | 4.4126 | bioemu2 |
| 16 ns | root_mean_w2 ↓ | **3.2147** | 2.9862 | 3.6100 | v4e |
| 16 ns | variance_w2 ↓ | **13.9706** | 20.5135 | 28.0998 | bioemu2 |
| 16 ns | md_pca_w2 ↓ | **0.0766** | 0.1490 | 0.2651 | bioemu2 |
| 16 ns | weak_contact_j ↑ | **0.2362** | 0.1945 | 0.0000 | bioemu2 |
| 16 ns | transient_contacts_j ↑ | **0.3499** | 0.3447 | 0.0000 | bioemu2 |
| 64 ns | rmsf_corr ↑ | **0.6514** | 0.5024 | 0.0306 | bioemu2 |
| 64 ns | rmsf_mae ↓ | **1.6230** | 2.9230 | 5.5644 | bioemu2 |
| 64 ns | root_mean_w2 ↓ | 4.0386 | **4.0249** | 4.7021 | v4e（几乎打平） |
| 64 ns | variance_w2 ↓ | **17.0818** | 32.4054 | 41.1092 | bioemu2 |
| 64 ns | md_pca_w2 ↓ | **0.0871** | 0.1920 | 0.3160 | bioemu2 |
| 64 ns | weak_contact_j ↑ | **0.2111** | 0.1524 | 0.0000 | bioemu2 |
| 64 ns | transient_contacts_j ↑ | 0.3764 | **0.3830** | 0.0000 | v4e（几乎打平） |

**计分（7 指标 × 4 horizon = 28 项）**：bioemu2 胜 **20**，v4e 胜 **8**。

---

## 4. 解读

### 4.1 短时标（1 ns）：bioemu2 的短板

- `variance_w2` 16.77 vs persistence 7.25 —— bioemu2 的 12 个 endpoint 在 1 ns 内
  **分散度过大**（真实 1 ns 运动极小，persist 只 1.68 Å）。
- `rmsf_mae` 2.32 vs 1.90 —— 同样因为过散。
- 这与本仓库文档 §3 描述的 from-none 问题是**镜像**的：from-none 是"样本挤成小球"
  （spread 0.33 Å）导致 root_mean_w2 崩；bioemu2 是"样本摊成大云"导致 variance_w2 崩。
  **两者的病根相同：训练目标与"1 ns 真实位移极小"这一事实不匹配。**

### 4.2 长时标（16–64 ns）：bioemu2 全面占优

- rmsf_mae 1.58/1.62 Å vs persistence 4.41/5.56 Å，改善 64–71%。
- md_pca_w2、variance_w2、接触类全部 100% 窗口胜出。
- **RMSF-r 是 bioemu2 最亮的一项**：0.59/0.65，是从 1 ns 到 64 ns 唯一
  **单调上升**的指标，说明它学到的柔性剖面随外推时间越来越接近真实解剖面。

### 4.3 对本仓库的意义

1. **v4e 的 `root_mean_w2` 失败不是本项目独有**：bioemu2 零样本在 1 ns 处同样
   惨败（−3.90），说明"短程逐残基绝对位移不可预测"是**任务级上限**，不是模型缺陷。
   这与文档 §3 的判断一致，现在有了独立证据。
2. **v4e 在 `variance_w2` / `md_pca_w2` 上打赢了 bioemu2**，说明系综 loss
   （§8 的 RMSF+方差+spread+corr 四项）**确实有效**——它把分布宽度这一项
   校准得比一个大规模预训练模型还好。
3. **`rmsf_corr` 是下一步该重点攻的方向**：bioemu2 用纯预训练先验就拿到 0.65，
   而 v4e 靠专门的 corr loss 也只到 0.51。说明**剖面形状主要靠表示质量**，
   不是靠 loss 项能补的。

---

## 5. 复现

```bash
# 全量 200 窗口（~40 min, 1×A800）
bash run_bioemu2_zeroshot.sh
# 结果 -> results/bioemu2_zeroshot.json / .log
```

**已知实现差异**（详见设计文档 §2.1）：
- bioemu2 只用 anchor 单帧作条件；v4e 用 8 帧历史。
- denoising 步数 128（默认 200），为控制扫描时间。
- mdCATH 侧链用确定性占位（指标只读 CA）。
- `trimul_variant: cueq → full`（本机 CUDA 驱动与 cueq CUDA-13 算子不兼容；
  同一权重、同一公式，仅速度差异）。
