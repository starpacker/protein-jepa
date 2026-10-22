# 进度汇报整理计划 — 2026-10-20

> 目的：明天向老师汇报 current project progress。整理脚本 + 文档，整合两份结果文档，
> 对齐 3 套 baseline（persistence / mdgen-zero-shot / bioemu2），
> 说明两套模型的训练方法，整理评估指标与实现（核对正确性），
> 并**保存轨迹 PDB 文件**（MDGen 轨迹 / 我们两套模型 decode 的原子 position / ground-truth）。

## 验收清单（做完一项打勾）

> **状态：全部完成（T1–T7 ✅）+ bioemu2 结果已产出**（`results/bioemu2_zeroshot.json`，已入 `CURRENT_PROGRESS_20261020.md` §3.5），
> 跑完把结果填入 `CURRENT_PROGRESS_20261020.md` §3.5。

- [x] **T1** 写本 plan 文件（当前）— `PROGRESS_REPORT_PLAN_20261020.md`
- [x] **T2** 核对评估指标实现正确性（读 `ensemble_metrics.py` / `ensemble_metrics_ext.py`，
      跑单元测试，逐指标写"定义 + 公式 + 对齐口径 + 退化情况"）
      — 26 单测通过；oracle=0/1 锚点正确；persistence rmsf_corr=0 及 delta 系综退化已复现
- [x] **T3** 写**轨迹 PDB 导出脚本**（新文件 `scripts/export_eval_trajectories_pdb.py`）：
      - [x] T3a MDGen 轨迹（zero-shot atlas.ckpt + mdCATH 短训 ckpt）
      - [x] T3b 我们模型 1（Stage-A ensemble dualA）decode 的 CA position
      - [x] T3c 我们模型 2（from-none GLE v4e）decode 的 CA position
      - [x] T3d ground-truth 未来帧
      - [x] T3e persistence（anchor 复制）
- [x] **T4** 跑导出脚本，产出 PDB 到 `artifacts/pdbs_20261020/`，附 README 说明目录结构
      — 3 窗口 × 6 轨迹 × 16 帧，CA-only；README 含口径/局限
- [x] **T5** 确认 3 套 baseline 的**同一窗口口径**（persistence / mdgen-zero-shot / bioemu2）
      是否已有可比的数字；缺的补跑
      — **重要发现**：两份文档用的 val 蛋白集**不相交**（DUAL=200dom/20 蛋白，
      FROMNONE=1000dom/100 蛋白），且 DUAL 的 20 个 val 有 17 个在 1000dom **train** 里
      ⇒ 跨文档数值不可直接横比；整合文档时必须显式声明（见 T6f）。
      bioemu2 缺 `libnvrtc.so.13`（已用 `LD_LIBRARY_PATH` 修复），✅ **已跑完**（200 窗口 × 4 horizon，结果 `results/bioemu2_zeroshot.json`）。
- [x] **T6** 整合两份文档 → 新文件 `CURRENT_PROGRESS_20261020.md`：
      - [x] T6a 项目总览 + 两套模型是什么（一图/一表）
      - [x] T6b 两套模型的**训练方法**详解（Stage-A ensemble 路线 vs from-none GLE 路线）
      - [x] T6c **评估指标与实现**（含正确性核对结论）
      - [x] T6d **3 baseline 对比表**（persistence / mdgen-zero-shot / bioemu2 ✅）
      - [x] T6e 两套模型 × 3 baseline 的汇总对比
      - [x] T6f 局限与诚实声明 + 下一步
- [x] **T7** 汇报用一页小结（slide-ready）— `PROGRESS_ONEPAGER_20261020.md`

---
## 关键事实（已核实）
- 模型 1：Stage-A ensemble 路线 → `artifacts/stage_ensemble_dualA/checkpoint_best.pt`
  （+ 冻结 Stage-A encoder `stage_a_disp_relpos` + B1a decoder）。文档 = DUAL。
- 模型 2：from-none GLE 路线 → `artifacts/fromnone/stage_cb1b_v4e/checkpoint_latest.pt`
  （v4f 正在训练）。文档 = FROMNONE。
- Baseline：persistence（两文档都有）；mdgen zero-shot = `atlas.ckpt`；
  bioemu2 = `bioemu2-preview`（`run_bioemu2_zeroshot.sh`，✅ 已完成，`results/bioemu2_zeroshot.json`）。
- mdCATH 帧间隔 1000 ps；MDGen atlas.ckpt 原生 10 ps/帧（OOD）。

---

## 追加任务（2026-10-20 晚）：1000dom 统一重训 + 轨迹导出

> 目标：把**默认数据集统一为 `mdcath_manifest_1000dom`**，并**在 1000dom 上重训 Model 1**
> （Stage-A encoder + 锚定解码器 + Gaussian 系综头），使 **MDGen / from-none GLE / Model 1
> 三者数据口径一致**，可做直接对比。完成后**重新导出轨迹 PDB**。

- [x] **R1** 默认数据集切到 1000dom（`train_stage_a_v2.py` / `train_stage_b1a.py` /
      `train_stage_ensemble.py` 的 `--manifest` 默认值）
- [x] **R2** 在 1000dom 上重训 **B1a 锚定解码器** → `artifacts/stage_b1a_relpos_anchored_1000dom/`
      （`run_b1a_relpos_anchored_1000dom.sh`，6000 steps，GPU 4；首个 best ckpt 已在 step250 产出）
- [x] **R3** 在 1000dom 上重训 **Gaussian 系综头** → `artifacts/stage_ensemble_1000dom/`
      （`run_ensemble_1000dom.sh`，依赖 R2 的 best ckpt，GPU 7）
      - 修了一个真 bug：`train_stage_ensemble.py` 用 `torch.linspace(0,len-1,N)` 取 N 个
        **等间距**索引（末位四舍五入到 len → IndexError），已改为均匀 stride 子采样（保证 in-range）。
- [x] **R4** 在 1000dom val 上评估 Model 1（7 系综指标 + Kabsch），与 MDGen/from-none 同表
- [x] **R5** 重新导出轨迹 PDB（`artifacts/pdbs_1000dom_20261020/`，3 窗口 × 6 轨迹 × 16 帧）
- [x] **R6** 更新 `CURRENT_PROGRESS_20261020.md`（加入 1000dom Model 1 结果）

> Stage-A encoder `artifacts/stage_a_disp_relpos/` **已经是 1000dom 训练的**（config 已核实），无需重训。

- [x] **R7** B1a 跑完后（6000 steps），用**最终 B1a best ckpt** 重跑 Gaussian 系综头
      （重训 4000 步），并刷新 R4/R5 的评估与 PDB，确保 Model 1 内部自洽（B1a = 最终版）。
      — ✅ best val 3.8130（旧 3.8461）；RMSF-r global 0.538→0.673；PDB 重导至
        `artifacts/pdbs_1000dom_20261021/`。
      - 原因：R3 的系综头是在 B1a **step-250** 时的 `checkpoint_best.pt` 上训的，
        而 B1a 到 6000 步还在持续下降；最终应让系综头见到最终 B1a。
