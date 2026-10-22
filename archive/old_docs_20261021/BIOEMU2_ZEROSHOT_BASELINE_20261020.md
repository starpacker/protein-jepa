# bioemu2 零样本（zero-shot）baseline — 设计说明（2026-10-20）

> 目的：把 **bioemu2（microsoft/bioemu2, `bioemu2-preview` checkpoint）** 当作
> **完全零样本**的外部基线，用与本仓库 from-none GLE 模型**完全相同**的系综评测口径
> 打分，为 `FROMNONE_GLE_ENSEMBLE_RESULTS_20261019.md` 的结论提供外部参照。
>
> **没有任何微调**：使用官方发布的权重，不训练、不调参、不喂 mdCATH 数据。

---

## 1. 为什么这是一个有效的对照

对照文档（`FROMNONE_GLE_ENSEMBLE_RESULTS_20261019.md`）§7 的核心结论是：
from-none GLE 在 RMSF / variance-W2 / MD-PCA-W2 / 瞬态接触上稳定胜 persistence，
但 `root_mean_w2` 在短时标系统性失败。要判断这些数字"是好是坏"，需要一个
**领域内公认的、独立训练的动态模型**作为标尺 —— bioemu2 正是这样一个模型
（Microsoft Research，蛋白质+核酸+配体的动力学基础模型，原生支持
`propagate(lag_time_ns=...)` 时间前推采样）。

## 2. 对照的公平性设计（apples-to-apples）

| 项 | from-none GLE | bioemu2 baseline |
| --- | --- | --- |
| manifest | `artifacts/mdcath_manifest_1000dom/val.jsonl` | **同一个** |
| 窗口生成 | `evaluation.data.iter_windows` + `EvalProtocol` | **同一个函数、同一协议** |
| horizon | 1 / 4 / 16 / 64 ns（mdCATH 1 ns/帧） | **同一组** |
| 真值系综 | `w.future_path` 的未来 CA 帧 | **同一份数据** |
| persistence | anchor CA 复制 T 次 | **同一实现** |
| 指标代码 | `evaluation.ensemble_metrics_ext.extended_ensemble_summary` | **逐字复用** |
| 聚合 | 窗口均值 + bootstrap over windows (2000×, 95%) | **同一实现** |
| 对齐 | 各自 Kabsch 对齐到自身首帧 (`align=True`) | **同一口径** |

**唯一的差别是被测模型本身**：bioemu2 每个窗口从 anchor 抽取 N=12 条
`lag_time_ns = horizon` 的前推 endpoint（endpoint 由 bioemu2 自己 rigid-align
到输入帧），取 CA 与原子的等价指标。

### 2.1 需要显式声明的差异（诚实披露）

1. **条件信息**：bioemu2 只用 **anchor 单帧**（全原子）作为条件；from-none GLE
   用 **K=8 帧历史**。bioemu2 这一侧的信息更少，若它仍能赢，说明其动力学先验更强。
2. **只比 CA**：bioemu2 输出全原子，但本评测只取 CA 骨架，与 from-none 模型口径
   一致。
3. **anchor 构建**：mdCATH 的 HDF5 只存 backbone（N, CA, C, O）。侧链重原子用
   确定性占位坐标补齐（见 §3），因为**指标只读 CA**，且 bioemu2 的 featurization
   只要求坐标有限、非退化。N/CA/C/O 全部来自真实 mdCATH 坐标。
4. **denoising 步数**：bioemu2 默认 200 步；本 baseline 用 **128 步**（`--num-steps`）
   以把 200 窗口 × 4 horizon × 12 样本的扫描压到可控时间。这是唯一的推理超参改动。
5. **MSA**：每个（蛋白, 温度）序列用 ColabFold 在线搜索一次并缓存，全流程复用。

## 3. 实现

| 文件 | 作用 |
| --- | --- |
| `evaluation/baselines/run_bioemu2_zeroshot.py` | 主评测驱动（复用本项目全部评测代码） |
| `evaluation/baselines/bioemu2_io.py` | mdCATH backbone → bioemu2 System / anchor 桥接 |
| `run_scripts/`run_bioemu2_zeroshot.sh` | 启动脚本（200 窗口，n_samples=12，1/4/16/64 ns） |
| `baselines/bioemu2/` | 上游 bioemu2 源码副本 + 本地 checkpoint/config（可追溯） |

**关键实现点**：
- 每个（蛋白, 温度）只做 **一次** `get_msa` + `model.embed`，随后所有窗口/horizon
  复用 trunk embeddings（`propagate_with_embeddings`）。
- 每个窗口的 N 条 rollout 在**一次 denoiser 调用**内完成（等价于官方
  `propagate(num_samples=N)`）。
- `config.yaml` 中 `trimul_variant: cueq → full`：本机 CUDA 驱动 12.9 与 cueq
  CUDA-13 算子不匹配（`libnvrtc.so.13` 缺失），改用等价的非融合实现。
  **这不改变数值**（同一权重、同一公式），只影响速度。

## 4. 运行方式

```bash
# 全量（200 窗口）
bash run_scripts/ run_bioemu2_zeroshot.sh

# 冒烟测试
python -m evaluation.baselines.run_bioemu2_zeroshot --smoke

# 自定义
python -m evaluation.baselines.run_bioemu2_zeroshot \
    --manifest artifacts/mdcath_manifest_1000dom/val.jsonl \
    --horizons-ns 1 4 16 64 --n-samples 12 --max-windows 200 \
    --num-steps 128 --device cuda --out results/bioemu2_zeroshot.json
```

结果写入 `results/bioemu2_zeroshot.json`，格式与
`results/{anchor,v4c,v4d,v4e}_ensemble_v2.json` **完全一致**，可直接并表。
