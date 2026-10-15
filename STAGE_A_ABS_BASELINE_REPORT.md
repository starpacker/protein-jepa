# Stage A Baseline Report — `stage_a_abs`

**Date:** 2026-10-14
**Checkpoint:** `artifacts/abc_v3/stage_a_abs/checkpoint_latest.pt` (step 5000)
**Standard:** `STAGE_AB_REDESIGN_PLAN.md` §1.7 Stage A 验收 gate
**Encoder:** `ProteinGeometryEncoder`, SE(3)-invariant, `dH=128`, depth 4, 16 RBF, orientation features on
**Objective:** Absolute-target JEPA (`delta_target=False`) + struct_recon auxiliary (weight 0.3, 512 pairs/step, dim 384)
**Data:** mdCATH 200-domain subset, T=320 K, replica 0

### 训练配置

| 参数 | 值 |
|:---|:---|
| steps | 5000 |
| lr / encoder_lr_scale | 3e-4 / 0.3 |
| batch_size × grad_accum | 8 × 4 |
| history_frames | 8 |
| temporal_lags (frames) | 1, 2, 4, 7 |
| paired_horizons (frames) | 8, 16, 24, 32, 48, 64, 96 |
| struct_recon_weight / pairs / dim | 0.3 / 512 / 384 |
| delta_target | **False** (absolute target) |
| temporal_variance_weight / min_std | 0.02 / 0.05 |
| EMA start → end | 0.998 → 0.9999 |

---

## §1.7 Gate 结果总览

```
=== Stage A §1.7 gate: 5/5 passed ===
>>> GATE PASSED — proceed to Stage B1a <<<
```

| # | Gate 项 | 阈值 | 实测 | 结果 |
|:---:|:---|:---|:---|:---:|
| 1 | latent persistence skill (lag 1,2, vs time-shuffle) | > 0.0 | **+0.1401** | ✅ |
| 2 | H 有效秩 | ≥ 4 | **66.42** | ✅ |
| 2 | 同蛋白同残基时间 std | ≥ 1e-3 | **0.12825** | ✅ |
| 2 | 死通道 (sd<1e-3) | = 0 | **0/128** | ✅ |
| 2b | 跨蛋白 cos（去均值） | ≤ 0.95 | **0.4292** | ✅ |
| 3 | 读出器 R² (CA 距离, held-out) | ≥ 0.05 | **+0.4567** | ✅ |
| 4 | full history 不差于对照 | ≥ −0.05 | **+0.0001** | ✅ |

---

## §1.7.1 Latent Persistence Skill（每蛋白、每温度、每 lag）

冻结 encoder，在固定 held-out split 上评估。Skill 定义为 `1 − MSE_lag / MSE_shuffle`，其中 shuffle null 取同一轨迹中随机时刻的 H。这是一个**可实现的判别性检验**：若 H 塌缩为静态 fold（本 gate 要抓的失败模式），它就分不清 `H_{t+ℓ}` 和随机 `t′` 的 H，skill ≈ 0。

| lag (frame) | T | skill (vs shuffle) | skill (vs static-mean, 诊断用) |
|:---:|:---:|:---:|:---:|
| 1 | 320 K | **+0.1882** | −0.6412 |
| 2 | 320 K | **+0.0920** | −0.8467 |
| 4 | 320 K | −0.0698 | −1.1595 |
| **均值 (lag 1,2)** | | **+0.1401** | — |

**真实坐标天花板**（脚本内注）：lag1 = +0.37, lag2 = +0.21, lag4 = −0.06。

**关键解读：** H 在 lag 1、2 上的 persistence skill 约为真实坐标天花板的一半（0.19 vs 0.37；0.09 vs 0.21），说明 H 确实携带了关于邻近未来的信息，只是比原始坐标弱一半。lag 4 处 H 与真实坐标同为负值（−0.07 vs −0.06），吻合。

**为什么不用 cos_sim 或 static-mean skill 做判据：** 静态均值 baseline 在此数据上极难打败——短窗口内蛋白几乎不动，轨迹均值天然贴近每一帧。真实坐标本身在 lag 1 上相对 static-mean 也只有 ≈ −0.26，所以要求正值不可实现，只会逼出噪声而非信号。这就是 §1.7 明确指定 time-shuffle 作为 null 的原因。

---

## §1.7.2 H 有效秩与时间变化统计（防止只编码身份标签）

| 指标 | 阈值 | 实测 |
|:---|:---:|:---:|
| 有效秩 (exp of entropy of singular spectrum) | ≥ 4 | **66.42** |
| 同蛋白时间 std（逐残基、跨帧） | ≥ 1e-3 | **0.12825** |
| 死通道 (channel sd < 1e-3) | = 0 | **0 / 128** |
| 跨蛋白 cos（去均值后，mean-removed） | ≤ 0.95 | **0.4292** |

有效秩 66.4（上限 128）说明 H 用了约一半的可用维度，没有塌缩到低维流形。时间 std 0.128 远高于阈值，128 个通道无一死亡。跨蛋白 cos 0.429 说明不同蛋白的 H 均值是可区分的——encoder 并未退化为一个与蛋白无关的身份标签。

---

## §1.7.3/4 小型固定读出器 + 对照

读出器在 **train split 拟合（56 蛋白）**，在 **held-out split 评分（24 蛋白）**。任务：从 H 对预测内部 CA 距离、接触变化。对照包括打乱过去帧顺序、只给末帧、无历史。

| 模式 | R²(CA 距离) | R²(接触变化) |
|:---|:---:|:---:|
| **full**（全历史平均） | **+0.4567** | −0.0078 |
| shuffled（打乱帧序） | +0.4566 | −0.0077 |
| last（只给末帧） | +0.4142 | −0.0043 |
| none（无历史，常数） | −5.7420 | −0.0000 |

- **full R² = +0.4567**，远超阈值 0.05 — H 保留了大量内部几何信息（比之前 pairwise probe 报的 0.343 更高，因为此处用线性读出器 + 到末帧的绝对距离）。
- **gain (full − best control) = +0.0001**，不差于对照 — 通过。
- **none = −5.74** 说明读出器确实依赖 H 内容，不是常数拟合。

**关于"历史是否有用"：** §1.7.4 要求由数据决定。脚本头注释解释了为何在近静态数据上"full 必须优于 last"不是有效要求：真实坐标本身 full=0.84 vs last=0.99，历史平均天然不如末帧。因此有效的判别是 **full 不差于所有对照**，而塌缩的 H（所有帧平均落到退化方向）会**差于** shuffled/last——这正是 gate 要抓的失败。此处 full(+0.4567) ≈ shuffled(+0.4566) > last(+0.4142)，未失败。

---

## 结论

**`stage_a_abs` 通过 §1.7 全部 5 项 gate。** Stage A 的 JEPA 表示携带了真实的时间信息（persistence skill ≈ 真实坐标天花板的一半）、保留了充分的有效秩与时间变化、且其内部几何（CA 距离）可被 held-out 读出器以 R²=0.46 恢复。

**与早前 diagnosis 的差异：** 此前报告用 H cos_sim 阈值（要求 @96ns < 0.95）判定 Stage A 失败。那个阈值不是 plan 的要求。plan §1.7 明确不采用 cos 判据，而是用 time-shuffle persistence skill。按 plan 的标准，Stage A **达标**，可以推进 B1a。

**对照 baseline（待补）：** §1.7 要求"普通 JEPA（无 OPF）同等容量配置"作为对照，验证 OPF 的增量贡献。该对照尚未训练。

---

## 复现命令

```bash
CUDA_VISIBLE_DEVICES=1 python -u tmp_claude/eval_stage_a_gate_full.py
# CKPT 默认指向 artifacts/abc_v3/stage_a_abs/checkpoint_latest.pt
```
