# Protein JEPA Frame

A compact research framework for protein-dynamics representation learning and
latent-state prediction on **mdCATH** (5 temperatures: 320 / 348 / 379 / 413 / 450 K).

---

## 📌 Start here — the two canonical documents (evaluation reference)

| # | Document | What it is |
|---|---|---|
| 1 | **[`EVALUATION_METHODS_AND_COMMANDS_20261021.md`](EVALUATION_METHODS_AND_COMMANDS_20261021.md)** | **The authoritative evaluation methods + reproduction commands + full results reference.** §1.2 lists **every evaluated method** (Model 1 / Model 2 / Lasp-final / H+S / MDGen / bioemu2 + persistence & oracle), §2 defines *every* metric (CA-RMSD / Kabsch, RMSF & RMSF-r, Root-Mean / Variance / MD-PCA W2, contact Jaccard, energy scores, CA geometry validity, backbone N/CA/C/O stereochemistry, BioEmu-style static + MSM dynamics, order-aware trajectory metrics, temperature generalization) with units/direction/code, §9 gives **15 verified runnable commands** (one per result table), and §3–§7 carry all result tables. Every number carries its `n` + sampling budget; static vs. dynamics metrics are separated. |
| 2 | **[`PROJECT_REPORT_20261021.md`](PROJECT_REPORT_20261021.md)** | **Full self-contained project report** — models, training recipes, results vs. persistence / MDGen / bioemu2, and limitations. |

> Its plan / companion protocol lives in `archive/docs_20261031/`.

Everything else in the repo is either **code** (below) or **archived history** (`archive/`, see [Repository layout](#repository-layout)).

---

## ⭐ Current framework — LASP correct-framework rebuild (2026-10-31)

This is the **baseline for further work**, replacing the frozen SOTA as the
reference implementation. It keeps the SOTA dynamics and fixes the bath so the
memory variable `S` actually enters the trajectory.

| Document | What it is |
|---|---|
| [`LASP_CORRECT_FRAMEWORK_PLAN_20261031.md`](LASP_CORRECT_FRAMEWORK_PLAN_20261031.md) | Design, pre-registered criteria, tooling. |
| [`RESULTS_LASP_CORRECT_FRAMEWORK_20261031.md`](RESULTS_LASP_CORRECT_FRAMEWORK_20261031.md) | Kalman sign fix, exact `S₀` posterior, exploratory memory probe, 9-arm ablation, verdict. |
| [`docs/RESEARCH_AUDIT_20261101.md`](docs/RESEARCH_AUDIT_20261101.md) | Representation lineage, frozen BioEmu2 comparison, interpretation corrections and next experiments. |
| [`RESULTS_HS_JOINT_20261101.md`](RESULTS_HS_JOINT_20261101.md) | Completed paired H-feedback pilot, spatial-posterior probe, and four-arm readout-retention experiment; partial improvements, persistent long-horizon readout failure, no recipe promoted. |
| [`RESULTS_BACKBONE_AND_PATHS_20261101.md`](RESULTS_BACKBONE_AND_PATHS_20261101.md) | Separate backbone and ordered-trajectory diagnostics, with their own task definitions and evidence limits. |
| [`HS_JOINT_PILOT_20261101.md`](HS_JOINT_PILOT_20261101.md) | Protocol fixed before that pilot, including best/latest evaluation and the rule for expanding training. |

State of the art it must beat: the frozen SOTA ensemble in
`results/sota_frozen_20261024_backup/` (never re-evaluated; kept as the yardstick).

What is fixed relative to the old SOTA:

- the Kalman gain sign (`K = −P·B_hᵀ·V⁻¹`; the covariance was fine, the mean was not);
- `S₀` is a full Kalman posterior over the history (`s0_mode=exact`), not an
  untrained amortised GRU, and not a "drive-only" recursion that is 76–95% off;
- the bath is trained at a scale where `S` moves the trajectory
  (`--coupling-scale 0.1 --gamma-scale 2e-5 --gamma-floor 2e-6`);
- the exact `S₀` posterior assumes the collective shared-displacement transition
  and diagonal H innovations. New runs default to `--collective-mobility`;
  exact mode requires `--history-conditioning` and `--noise-mix 0`.
- The opt-in `exact_spatial` mode now conditions the full cross-residue bath
  covariance for spatial collective noise. It requires observed history and
  collective mobility, and samples the full bath Cholesky factor.

The 2026-10-31 active-scale arms used `--noise-mix .8`, so their exact-mode
posterior is not an exact posterior for that spatially correlated-noise sampler.
They remain historical mechanism/ablation results, not evidence that the exact
posterior is calibrated under correlated H noise. Existing checkpoint settings
are preserved at evaluation. The canonical arm launcher now uses diagonal H
noise; the established training target and evaluation protocol are unchanged.

The historical probe reported **2–4%** error reductions for selected horizons.
**Audit correction (2026-11-01):** these numbers are not an upper bound on
memory and do not measure predictive information conditional on `H₀`.
The probe's anchor-only control uses an intercept rather than an `H₀` predictor,
and its regularization is selected on the same held-out proteins it scores.
An inconclusive `exact` versus `prior` comparison does not establish absence
of memory. The retrained arms did **not** beat the frozen SOTA overall;
their `md_pca_w2` gains coexist with losses on other metrics.
See the results document and research audit for the historical numbers and
the scope of the conclusions they support.

The 2026-11-01 pilot completed both frozen-H and trainable-H arms and evaluated
their best and final checkpoints on the same 200 windows, with 12 samples each.
Updating H slightly improved MD-PCA W2, while several other metrics and oracle
H-to-CA readout worsened. RMSF correlation showed no reliable improvement.
This does not satisfy the registered rule for expanding training, and the goal
of beating BioEmu2 across the full evaluation remains unmet.

A subsequent fixed-weight, three-arm probe completed the same 200-window panel.
Restoring spatial noise with its matching posterior improved overall scores
relative to diagonal noise, but increased invalid CA bonds and worsened several
450 K subgroup metrics. Against a prior with the same spatial noise, all ten
overall score intervals included zero. The source weights were trained with
spatial noise, so this is a sampler intervention, not evidence that this noise
setting is generally optimal or that history memory caused the gains. Full
results and a separate train-only readout-gradient diagnosis are in the
[2026-11-01 report](RESULTS_HS_JOINT_20261101.md).

A subsequent four-arm retention experiment completed **500 steps per arm** and
evaluated every arm's fixed latest checkpoint on the same 200 windows, with
12 samples each. Observed-frame readout supervision improved several historical
ensemble scores and reduced oracle readout degradation relative to the ordinary
teacher. Readout remained worse than frozen H, however, and coordinate/distance
endpoint energy scores also worsened against frozen. Stronger teacher retention
gave slightly lower overall readout error than observed supervision. All four
arms still failed the 16/64 ns oracle-ratio budget. This recipe is **not promoted**.
The [four-arm summary](results/hs_joint_20261101/retention_v1/summary.json) includes
all five registered contrasts, canonical and legacy geometry separately, and
validation loss after subtracting the different retention penalties. This is a
single-training-seed experiment on an adaptively reused development validation
set; marginal paired intervals do not account for that selection or establish
equivalence when they include zero. Test data remain sealed.

The fixed spatial-exact starting checkpoint also completed a **200-window
backbone diagnostic**, a **200-window continuous-path panel**, and a separate
**192 ns / 50-protein panel**. Even real future CA coordinates produce roughly
2.61 Å peptide C–N distances under the current ideal lift, so full-backbone
readout remains a concrete failure. Continuous forecasts improve whole-path
energy scores over persistence, while average per-frame position errors remain
higher and lag-one correlations exceed MD. At 192 ns the mean shape-path length
is about 44% of MD's. These results do not establish MSM kinetics or superiority
to BioEmu2. See the [backbone and path report](RESULTS_BACKBONE_AND_PATHS_20261101.md)
for protocols, numerical caveats, and the separate residue-type boundary audit.

A subsequent fixed-CA backbone pilot completed two matched **500-step** heads
on 500 training proteins and evaluated their fixed final checkpoints on the
same 200 development windows. PSF topology now handles insertion codes and
explicit histidine aliases while preserving the old learned embedding IDs.
Coordinate supervision reduced overall oracle N/C/O RMSD by 0.0610 Å against
the corrected zero-update head; forecast improvement was only 0.0050 Å.
Adding geometry supervision improved C–N and C–N–CA soft losses, but increased
the backbone clash soft loss and slightly worsened overall N/C/O RMSD against
coordinate-only training. CA remained bitwise fixed throughout. These results
show a measured tradeoff, not a completed T1 repair or a new BioEmu2 win.
The [registered pilot](docs/BACKBONE_READOUT_PILOT_V2_20261101.md) and
[paired results](results/hs_joint_20261101/backbone_repair_v2/summary.json)
retain all horizons, controls and exploratory confidence intervals. Full
regression at this stage passed **670 tests, with one skipped**.

A later matched **500-step local-frame transport** experiment supplies the
current CA directions that the simplified IPA does not consume. On the same
200-window development panel, overall forecast N/C/O RMSD decreased from
**4.6326 to 4.5756 Å**, and C–N soft loss from **0.6006 to 0.2599 Å**; forecast
clash change remains uncertain. Oracle N/C/O RMSD decreased from 1.6620 to
1.4423 Å. All 800 recomputed historical controls matched exactly, and an
independent NumPy audit reproduced all 150 paired intervals. The regression run
at that stage passed **756 tests, with one skipped**. This improves backbone readout;
forecast N/C/O error remains worse than persistence and CA-level BioEmu2
scores are unchanged. See the [registered protocol](docs/BACKBONE_TRANSPORT_PILOT_V1_20261101.md),
[results and train-only capacity diagnosis](RESULTS_BACKBONE_AND_PATHS_20261101.md#6-局部-frame-搬运相同500步预算的新读出实验),
and [frozen evaluation](results/hs_joint_20261101/backbone_transport_v1/evaluation200_v2.json).

A fixed 32-window training diagnostic found raw-H mean/variance drift. Applying
LayerNorm only before CA readout lowered latent error but worsened coordinate
U-energy score from **2.2033 to 2.5667 Å**; this projection is not adopted.
A separate [coupled Brownian microstep diagnostic](results/hs_joint_20261101/microstep_train32_v1/diagnostic.json)
completed all 32 training windows and passed independent replay and statistics
checks. Refining each 1 ns step from 1 to 8 substeps left 64 ns H variance at
**9.0922** (from **9.3732**) and did not improve endpoint U-energy score.
These finite training diagnostics do not establish global numerical stability
or a causal benefit from S memory; the inference step remains unchanged.
A subsequent [fixed-covariance S-initialization control](results/hs_joint_20261101/history_mean_train32_v1/diagnostic.json)
completed all 96 branches; the real-history arm exactly replayed the original paths. Overall
endpoint U-energy score was **2.1655** with the real history mean, **2.1918**
with a repeated-anchor history mean, and **2.1887** with a zero mean.
This isolates initialization in the current trained model. A separately
registered [H-Markov training control](docs/HS_MARKOV_PILOT_V1_20261101.md)
completed 500 updates and both 200-window evaluations with coupling set to zero,
the same warm start and paired H-noise draws; it retains unused bath computation.
Its [verified comparison](results/hs_joint_20261101/markov_v1/summary.json) gives
C0−H+S long-horizon U-energy difference **−0.0101 Å [−0.0510, +0.0205]**.
H+S has better weak/transient contact scores, but its primary advantage is not
established in this budget. No overall recipe promotion follows.
A [matched endpoint-versus-path-target experiment](docs/HS_TARGET_PILOT_V1_20261101.md)
has completed all 10 logical jobs: 500 updates, standard200, endpoint200,
the full 7200-window temperature/T2A suite, and matched8 for each arm.
A tested [CLI-only continuation](docs/HS_TARGET_PILOT_CLI_AMENDMENT_V2_20261101.md)
preserves two earlier argument-parser failures. The
[complete summary](results/hs_joint_20261101/target_v1/summary_v2.json) and
[independent verification](results/hs_joint_20261101/target_v1/independent_verification_v1.json)
agree on all 341 paired intervals and 2,370 descriptive means.
The primary 16/64 ns coordinate U-energy score is **2.0027 / 2.0277 Å**
(endpoint / path-mixture), with difference **−0.0250 Å [−0.1024, +0.0223]**.
This does not support promoting endpoint training. Both arms improve overall
RMSF MAE, Root-W2 and Variance-W2 against saved BioEmu2, but worsen RMSF
correlation, weak contacts and transient contacts; PCA differences remain
uncertain. The [H+S report](RESULTS_HS_JOINT_20261101.md) includes all outcomes
and the limits of these exploratory development-set comparisons.

A [joint-path versus time-marginal training experiment](docs/HS_ORDERED_PATH_PILOT_V1_20261102.md)
completed all 12 jobs with identical future labels and continuous sampling, 500 updates per
arm, frozen H/CA readout, and a preregistered long-horizon joint-path U-score.
The matched marginal control separates temporal joint supervision from adding
intermediate labels. All 45 implementation tests and 16 statistics tests passed;
a fixed h64/L96/M12 zero-update forward/backward used about 5 GiB and left
weights unchanged. Their 500 actual training-window logs match byte for byte.
The [verified summary](results/hs_joint_20261101/ordered_path_v1/summary_v2.json)
gives a long-horizon joint-path U-score difference of **−0.0324 Å [−0.0846, +0.0025]**.
Joint supervision worsens overall RMSF MAE, RMSF correlation, root mean W2 and
both contact scores relative to marginal supervision; it does not support a recipe
upgrade. At 64 ns its path length remains **67.91 Å versus MD 130.23 Å**, and
lag-1 ACF is **0.836 versus 0.521**. All 402 paired intervals were independently
recomputed. Versioned metadata-check repairs and the actual first verifier
failure are preserved; training and statistical definitions were unchanged.

A separate fixed-cache decoder capacity probe completed both 500-step training stages
and independent verification of all 1,501 intervals. It compares
continued training of D0 with the same decoder plus a zero-initialized 384-parameter
local-vector residual, using frozen H and unchanged P32. The [statistical protocol](results/hs_joint_20261101/ca_local_residual_v1/summary.protocol.json)
was registered before training; its primary is paired 16/64 ns oracle CA-RMSD.
The existing caches contain endpoint labels only. Cached forecast-H diagnostics
therefore cannot establish improvement on the historical seven metrics or on
new dynamics rollouts. The [verified result](results/hs_joint_20261101/ca_local_residual_v1/summary.json)
is inconclusive for the added branch: long-horizon projected oracle RMSD difference
**−0.0010 Å [−0.0036, +0.0014]**. Both tuned decoders worsen validation oracle
RMSD relative to D0. The added branch also worsens projected cached-forecast
RMSD versus the tuned control by **+0.0185 Å [+0.0004, +0.0526]**.
This probe does not support adopting the residual branch.

A [training-only contact-loss diagnostic](results/hs_joint_20261101/contact_bias_train32_v1/diagnostic.json)
identified a finite-member variance penalty in the 12-member plug-in training
Brier. On all 32 fixed training windows its coordinate gradient opposes the
coordinate U-score gradient, but averages only **2.84%** of the combined
coordinate-loss gradient norm. A completed, independently checked
[kernel-gradient audit](results/hs_joint_20261101/contact_kernel_gradients_v2/result.json)
found correction norms of **1.50%–7.12%** across four fixed initial-weight
training windows; total-gradient directions remain close (cosine at least
**0.99954**). These are local derivative results, without optimizer updates or
forecast evidence. A [matched 500-step contact-estimator comparison](docs/HS_CONTACT_ESTIMATOR_PILOT_V1_20261102.md)
has completed all 12 jobs and independent verification of 402 paired intervals.
The [verified result](results/hs_joint_20261101/contact_estimator_v1/summary.json)
gives a primary long-horizon path U-score difference of **−0.0159 Å
[−0.0597, +0.0092]**. All seven overall historical metric point estimates worsen,
with intervals spanning zero; short-horizon distance/contact and CA-clash
tradeoffs remain. This pilot does not support replacing plugin with population-U.
The existing evaluation estimators remain unchanged; the [full report](RESULTS_HS_JOINT_20261101.md#接触估计器配对500步试验完成不采用population-u替换2026-11-02)
retains the persistence/BioEmu2 comparisons, temperature results and limits.

A completed [four-window TRAIN path-readout diagnostic](RESULTS_HS_JOINT_20261101.md#固定训练四窗全路径读出诊断完成2026-11-02)
separates true-MD, true-H readout and forecast-H paths before/after P32. In its
single h64 window, shape-aligned path lengths are **62.73 / 63.02 / 35.99 Å**
(MD / oracle P32 / forecast P32). Independent CPU/NumPy recomputation of all
four windows agrees within **4.62e−14**. These fixed training examples implicate
the forecast path more than the frozen true-H readout, without establishing an
H information bound, generalization or a normalization remedy; the earlier
32-window LayerNorm score deterioration remains unchanged.

A subsequent [saved-H counterfactual](RESULTS_HS_JOINT_20261101.md#保存h的观察半径映射反事实完成结果为负2026-11-02)
centers future H and scales it to each residue's observed-H0 radius before frozen
readout. Coordinate and distance path U-scores worsen in all four TRAIN windows,
both raw and P32. At h64, P32 shape-path length falls **35.99→20.76 Å** and
coordinate U-score rises **0.7741→1.1373 Å**. Independent checks match every
mapped H value bitwise. This offline map is not adopted; it does not isolate
norm drift as a cause of slow dynamics or overturn the earlier LayerNorm result.

A completed [distance-target gradient audit](RESULTS_HS_JOINT_20261101.md#物理距离路径目标固定四窗kernel梯度诊断完成2026-11-02)
compares absolute-distance and signed distance-increment U-scores on four fixed
TRAIN windows, with auxiliary weight one and no updates. Their h64 gradient
norms are **98.2% / 14.9%** of the base gradient, with mutual cosine **0.1735**.
Independent CPU/NumPy checks reproduced all gradient statistics within
**1.80e−13**. These differ in scale and direction; the audit establishes neither
training efficacy nor GPU feasibility and does not select a coefficient.

A completed [matched distance-target pilot](RESULTS_HS_JOINT_20261101.md#距离辅助目标v2完整结果increment主指标较差2026-11-02)
compares fixed-weight signed increments (I) with absolute distances (A), both
retaining plugin contact. After the preserved v1 step-30 topology failures,
the separately defined PSF-based v2 passed metadata checks for all 500 windows and
completed both fresh 500-update arms and all 12 jobs. Registered summaries and
independent NumPy checks passed. The h16/64 joint-path coordinate U-score is
**1.677629 vs 1.649875 Å**, I−A **+0.027754** (95% CI **[+0.001592, +0.072649]**);
the endpoint secondary is also worse. Some fluctuation/contact scores improve,
but neither arm beats saved BioEmu2 across all seven metrics; matched8 population
TV/JSD remain descriptively worse at all five temperatures. This compares two
auxiliary targets, supplies no matched no-auxiliary causal result, and does not
adopt a new recipe. Full negative results, geometry, lag controls and limitations
are retained in the linked report.

A completed, independently checked [T2A time-support diagnostic](results/hs_joint_20261101/t2a_time_support_v1/result.json)
compares the same windows' true MD endpoints with the whole-trajectory reference.
For 12 proteins, endpoint-pool coverage is **0.3258** and population TV **0.6200**
(persistence: **0.2358 / 0.7628**). The 8 actual endpoints per trajectory have a
different budget from 96 model draws; these descriptive scores are not a model
performance ceiling and do not change the registered evaluation.

A [Ramachandran reference scorer](evaluation/ramachandran_reference_v1.py) now
uses pinned official Top8000/cctbx tables and six residue classes. Its 55 focused
tests include direct conformance with the original C++ scorer. All 2,000 frozen
route hashes matched, and scoring of 10,800 sample backbones plus independent
verification of all 320 intervals completed. Local frame transport lowers
forecast outliers from **38.56% to 33.04%**, while target MD is **1.69%**.
Coordinate training itself slightly worsens forecast outliers. The
[complete Rama results](RESULTS_BACKBONE_AND_PATHS_20261101.md#7-ramachandran评分完成搬运改善但模型仍远差于真实md)
document the remaining geometry gap; full matched T1 is still incomplete.

A separate [TRAIN32 joint-frame reference oracle](RESULTS_BACKBONE_AND_PATHS_20261101.md#9-train32联合frame-oracle存在局部改进未完成t1修复)
now holds CA fixed and fits bounded rotations using the true target backbone.
With decoded oracle CA, NCO error falls **1.860→1.619 Å** and Rama outliers
**18.00%→12.40%**; with true MD CA, they fall **.578→.283 Å** and
**14.14%→5.12%** (target MD: **1.61%**). NCO improves in all 32 windows,
but decoded-CA Rama worsens in 3, individual geometry losses can worsen, and
h64 omega deviation increases. Both conditions reach the 30° rotation bound
in 31/32 windows. All five roles, six Rama classes, support, negative outcomes,
and post-result verification limits are reported. This target-informed local
construction is not a forecast repair, a new trained head, or a T1/BioEmu2 win.

A completed [frozen-H distance-readout probe](RESULTS_BACKBONE_AND_PATHS_20261101.md#11-冻结h距离读出完成当前h更易读出真实结构未建立未来预测能力)
fits identical fresh heads on 400 proteins and scores 100 readout-heldout proteins.
With H re-encoded from **true future coordinates**, current versus original Stage A
log-distance MSE is **0.002610 vs 0.014400**, with RMSE **1.668 vs 3.299 Å**.
Current target-H improves log-MSE in 98/100 proteins, but its observed-H substitution
is worse than distance persistence in 82/100 (overall **0.048102 vs 0.047814**).
Both fresh 500-step arms and registered independent replay completed; the failed
v1 remains preserved. Encoders may have seen all 500 proteins. This is bounded
structural readability, not forecasting skill, an H information bound, or a T1 win.

A verified [four-window TRAIN one-step H-moment diagnostic](RESULTS_BACKBONE_AND_PATHS_20261101.md#12-train四窗一步h矩诊断范数预算失配尚未确定修复方案)
completed 85 teacher-forced transitions without new rollouts or updates. Expected
raw squared-norm changes are **28.802 / 11.661 / 14.532 / 13.173**, versus nearly
zero observed changes; centered budgets show the same mismatch. Predicted tangent
second moments are below each window's observed mean, but individual transitions
vary widely. Existing quadratic restoration is present, and algebraically removing
coupling increases mean norm growth. Four complete plots and independent NumPy
moment/filter replay are linked. These findings establish neither a normalization
repair nor autonomous forecast skill, Mori/FDT validity, or a BioEmu2 improvement.

A verified [PCA-continuity diagnostic](RESULTS_BACKBONE_AND_PATHS_20261101.md#13-train四窗pca连续性canonical特征跳变与h跳变同时出现未做模型干预)
retains all 89 frames and 85 transitions from those same TRAIN windows. In the
h64 window's first transition, CA Kabsch RMSD is **0.905 Å**, while the original
canonical-feature increment is **14.686 Å**, or **0.923 Å** after geometry-only
axis relabelling; saved H's mean per-residue squared increment is **54.113**.
All four complete plots and frame/transition tables are linked, including cases
with substantial physical motion and identity relabellings. No H was changed or
re-encoded. This is a representation association, not a causal channel test,
deployable continuity repair, or forecasting improvement.

A completed [relpos branch counterfactual](RESULTS_BACKBONE_AND_PATHS_20261101.md#14-train四窗relpos通道反事实改变h增量尚未验证读出或预测收益)
then re-encoded all 89 frames with a frozen encoder and retained all 85 pair controls.
All original/identity/post replays matched exactly. In the h64 first transition,
changing only the next frame's geometry-selected axis labels changes mean
per-residue H increment² from **54.113 to 1.020**; applying the same labels to both
frames gives **54.021**. First-step identity controls in h4/h16 remain unchanged.
The 437 forwards and saved-array audit passed with model parameters, all buffers
and RNG unchanged. These controlled internal-branch effects may depart from the
learned canonical convention; future-aware independent pairs are not a consistent
trajectory. Four complete plots and all 85 pair CSV/JSON records are linked.
Readout quality, physical kinetics and forecast benefits remain untested.

A completed [raw local-coordinate comparison](RESULTS_BACKBONE_AND_PATHS_20261101.md#15-train四窗局部坐标原始特征部分跳变减小局部取向变化仍可放大增量)
retains all 89 frames and 85 pairs without loading H or running an encoder.
Local N–CA–C coordinates reduce the first two h64 PCA jumps, but increase the
next two and the first h4/h16 jumps; local orientation changes acting on the
centroid lever arm remain substantial even though all recorded valid frames
are nondegenerate. Zero features are trivially continuous and discard position.
The first NumPy checker failed a float32 centroid-reduction comparison; a
preserved, explicitly post-producer v2 repair uses an analytical rounding bound
and exact centering replay from the recorded centroid, and passes. The producer
was not rerun. All curves/arrays and failure evidence are linked; these raw
feature results establish no H readability, training or forecast improvement.

A completed [matched Stage A relpos adaptation trial](RESULTS_BACKBONE_AND_PATHS_20261101.md#16-stage-a三臂适配zerolocal未改善端点读出三臂预测均弱于坐标持久性)
retains the PCA control: after 500 adaptation updates and fresh matched readouts,
true-future-H log-MSE was 0.009628/0.205728/0.140957 for PCA/zero/local.
Neither substitution is adopted. Observed-only predicted-H RMSE was
9.088464/11.009984/11.219255 Å versus raw CA persistence at 5.637213 Å;
all three arms lost to persistence at every horizon, although zero beat PCA's
predicted-H score at h64. The single-seed PCA-pretrained warm start and the
already-inspected TRAIN100 panel limit interpretation: these are endpoint
readout results, not architecture superiority, a matched adaptation gain,
Stage B dynamics, or progress past BioEmu2. The report retains all 100 proteins,
all horizon groups, training traces, and the checker's pre-score serialization
failure followed by v2 verification; existing negative findings remain.

Reproduce:

```bash
export PYTHONPATH=/tmp/lasp_stability_deps:.
bash run_scripts/drive_lasp_s0_arms.sh        # 9-arm training grid
bash run_scripts/run_lasp_s0_arms_eval.sh     # standard-200 ensemble metrics
bash run_scripts/analyze_lasp_s0_arms.sh      # paired + 3-seed verdicts
```

## Secondary documents

The 2026-10 Stage-A / GLE / Stage-B reports referenced by earlier revisions of
this file (`STAGE_A_AND_GLE.md`, `STAGE_B_HS_RESULTS_20261021.md`,
`EVALUATION_PLAN.md`, `REPAIR_PROGRESS.md`,
`NEXT_STEP_EVALUATION_3TASKS_20261021.md`) are kept in
`archive/docs_20261031/` — they document the superseded pipeline and are not
part of the live code.

Historical evaluation reports do **not** establish forecasting skill. New B1b training
uses history-only autonomous H+S predictions; future H is a detached latent target.
Its physical loss compares decoded predictions with real future coordinates.
Formal B1b runs require a passed, matching [H readout gate](evaluation/run_h_gate.py).

## The canonical evaluation contract

```text
X_{1:K} [B,K,L,4,3]
  -> frame-wise encoder
H_context [B,K,L,dH]
  -> predictor(H_context, tau, temperature)
H_future [B,L,dH]
  -> shared structure decoder
X_hat_future [B,L,3]
```

For the current Stage-A checkpoints, the predictor emits OPF factor-coordinate
deltas `[B,L,J,r]`. Evaluation reconstructs the absolute future latent as:

```text
H_future = OPF.compose(OPF.decompose(H_anchor) + pred_factors)
```

The autonomous H+S path instead initializes a bath state from the full history
and rolls out the learned GLE transition from `H_anchor`.

## Repository layout

| Path | Contents |
|---|---|
| `protein_jepa_frame/` | Model, data, training, and vendored OpenFold geometry code. |
| `evaluation/` | Leakage-audited CA/backbone-RMSD evaluation, MSM & BioEmu metrics, checkpoint adapters. |
| `tests/` | Unit and protocol tests (`python -m pytest -q`). |
| `scripts/` | Dataset preparation, training and evaluation entry points (Python). |
| `run_scripts/` | The **current** shell recipes (`run_*.sh`) referenced by the canonical docs. |
| `results/` · `artifacts/` | Evaluation JSON results and derived products (large; git-ignored). |
| `archive/` | Frozen historical reports, scripts and scratch (`old_docs_20261021/`, `old_scripts_20261021/`, …). *Not part of the live code.* |

Checkpoints, trajectories, generated reports, caches, and scratch experiments
are intentionally not part of this repository. Their canonical paths and hashes
are recorded in `evaluation/MODELS.md` when an evaluation run uses them.

## Install and test

```bash
pip install -e ".[dev]"
python -m pytest -q
```

The evaluation package has an additional smoke path:

```bash
python -m pytest evaluation/tests -q
python -m evaluation.run_eval --config evaluation/configs/smoke.yaml \
  --max-proteins 2 --horizons 1 4
```

See `evaluation/README.md` and `evaluation/RUNBOOK.md` for the full held-out
protein and held-out-temperature protocols.

---

## Reproducing the 3-task evaluation

Each task's exact command is in the "Repro" section of
[`EVALUATION_METHODS_AND_COMMANDS_20261021.md`](EVALUATION_METHODS_AND_COMMANDS_20261021.md).
The corresponding shell recipes live in [`run_scripts/`](run_scripts/):

| Task | Recipe |
|---|---|
| T1 backbone suite | `scripts/eval_backbone_suite.py` |
| T2 static + MSM | `scripts/eval_states_and_msm.py` |
| T3 temperature (Model 1) | `run_scripts/run_ensemble_temp_1000dom.sh` |
| T3 temperature (Model 2) | `scripts/eval_temperature_generalization_model2.py` |
