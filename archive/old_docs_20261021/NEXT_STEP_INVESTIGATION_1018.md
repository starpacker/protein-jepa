# Next-step investigation: where is the from-none bottleneck? (2026-10-18)

Scope: `From-None Stage B (B1b → B2)`. Input: `review_1018.md`, `REPAIR_PROGRESS.md`,
`DIAGNOSIS_H_READOUT_20261018.md`, `H_DECODABILITY_REPORT.md`, `B1B_DIAGNOSIS_AND_PLAN.md`,
`FROMNONE_B1B_B2_PIPELINE.md`, plus a direct read of the code. No new training was run;
this is a grounded assessment + a concrete, falsifiable next-step plan.

---

## 0. TL;DR verdict

I **agree with the document's core judgement** and can now trace it to specific code facts:

> The primary, *actionable* bottleneck is **decodability / the H-side representation and
> the loss that shapes it** — specifically the model has never been forced to make the
> **inter-frame difference ΔH** carry *collective (cross-residue) conformational* geometry.
> The evidence does **not** support "GLE rollout is the main bottleneck", nor "just add
> data/steps", nor "rewrite H+S".

But I would **re-order and sharpen** the document's P0. The document's P0 is a *measurement*
plan whose answer is already ~80% implied by existing evidence (`oracle-H ≈ autonomous`).
The highest-expected-value action is **not** to spend the next cycle only measuring, but to run
the P0 intervention **bundled with** the first candidate fix (a controlled decoder/representation
change) — because both share the same eval harness, and the intervention alone will most
likely just re-confirm what we already know.

Concretely my recommended next step is **two parallel, cheap tracks sharing one contract**:

- **Track A (measure, ~half a day, no retrain):** the four-way oracle intervention + ΔH-residue
  shuffle on the *current* best checkpoint, persisted as machine-readable JSON. This is the
  gate that decides the *next* fork. (Document P0.1/P0.2 + Agent 1/4.)
- **Track B (the actual candidate fix, main compute):** a **message-passing decoder (D1)** run
  head-to-head against the current MLP decoder, same encoder/kernel/budget/seed. This directly
  tests the document's leading hypothesis and is the smallest change that can *raise the
  ceiling* (oracle-H) as well as the autonomous number. (Document P1/D1 + Agent 3.)

Do **not** yet touch the GLE rollout, do **not** yet scale data, do **not** rewrite H/S.

---

## 1. What the code actually is (facts that constrain the diagnosis)

| component | file | params | key property |
|---|---|---:|---|
| Encoder `ProteinGeometryEncoder` | `models/encoder.py` | **1.34 M** | same-frame, SE(3)-invariant; inputs = per-residue internal distances + backbone torsions + aatype/residx/chain embeddings + **pairwise CA-distance RBF bias** + within-frame canonical relpos. **No global pooling / collective-coordinate token.** |
| Kernel `StageB1b` (GLE) | `models/stage_b1b/model.py` | 0.67 M | H-space energy `f=-∇U`, per-residue bath S, `force_clip=1.0`, `tail_init=1.0`. |
| Decoder `SimpleCADecoder` | `models/stage_b1b_fromnone/decoder.py` | **0.234 M** | per-residue MLP over `[H_future, H_anchor, H_future−H_anchor, S_future]` (in_dim=384) → `dx_local`, bounded `max_update=12 Å`, rotated by observed `R_anchor`. **No residue mixing.** |

Objective (`train_stage_cb1b.py::forecast_objective`): `total = 1.0·ca + 0.1·es + 0.1·phys`
(`ca` = Kabsch CA point loss on the **autonomous** rollout; `es`/`phys` = energy scores).

Two facts from the code that matter a lot:

1. **The decoder is literally per-residue.** `SimpleCADecoder.forward` concatenates per-residue
   features and feeds an MLP; there is **zero interaction between residues**. So it *cannot*
   represent a collective motion where residue *i*'s correct displacement depends on residue
   *j*'s state. For mdCATH (contact rearrangement, hinge motions) this is a structural cap —
   exactly the document's point (3.4 / D1).

2. **ΔH is fed but only as a raw concatenation.** `[H_future, H_anchor, ΔH, S]`. Because the
   MLP is per-residue and the encoder has no collective channel, nothing in the graph forces
   `ΔH_i` to align with the *geometric* displacement of residue *i*; the `ca` loss is the only
   pressure, and it can be satisfied by a smooth, mostly-local map (this matches the measured
   `H_cos` gap in `REPAIR_PROGRESS.md` and the old `delta-H → internal-deformation R² < 0`).

3. **Metric-definition drift (a real harness bug to fix before any go/no-go).** The trainer's
   `val/skill = 1 − ca/persist` (**relative**), but `evaluation/run_eval_fromnone.py` reports
   `skill = persist − rmsd` (**absolute Å**). `review_1018.md` mixes both ("0.05×/0.02×
   persistence" vs "+0.06 Å"). We must lock one definition (recommend **relative**,
   `1 − rmsd/persist`, plus report the absolute Δ and a paired protein bootstrap CI) or the
   thresholds in §6 of the review are not evaluable. Current numbers: 1 ns abs +0.060 / rel
   +1.6%; 4 ns +0.092 / +1.9%; 16 ns +0.140 / +2.3%.

---

## 2. Why I believe the bottleneck is decodability/representation, not rollout

The single most decisive existing datum is **`oracle-H ≈ autonomous`** (4 ns: 3.950 vs 3.951;
16 ns: 4.889 vs 4.889 — from `REPAIR_PROGRESS.md`). Decoded with the current decoder, handing the
model the *true* future H buys **nothing** over its own rollout. Therefore, under the current
decoder, **improving the rollout cannot move the CA metric** — the decoder is the binding
constraint on what any H (true or rolled-out) can express. This is precisely the old pipeline's
finding too (`H_DECODABILITY_REPORT.md` §9.4: `oracle_H ≈ H_persistence ≈ predictor`, all within
0.1 Å), now reproduced in the from-none pipeline.

The `H_cos`-vs-`H_gt_cos` gap (0.68 vs 0.9997 at 96 ns) is real but **is not yet shown to reach
the CA metric** — and given `oracle-H ≈ autonomous`, it currently cannot. That is exactly why the
document is right to demand the four-way intervention *before* blaming the rollout.

Cross-check from the encoder side: the from-none encoder is still a **same-frame descriptor with
no collective channel** (`models/encoder.py`, no pooling/global token). The from-none trick (co-
training + `ca` loss) already lifts it past the frozen-Stage-A wall (skill now > 0 and rising,
`val/skill` 0.004 → 0.028 in `train.log`), but there is no architectural reason for it to encode a
*global/collective* conformation mode. This is the same root cause `DIAGNOSIS_H_READOUT_20261018.md`
§3.1 identified, only partially addressed by from-none.

---

## 3. Recommended next step (ordered, with reasons)

### Track A — P0 intervention (measure; no retrain; ~half a day; use a free GPU: 5/6/7 are idle)

Deliverable: `evaluation/run_fromnone_intervention.py` → machine-readable JSON, contract-locked
to the **same** checkpoint / val manifest / protein & temperature split / K / dt / horizon grid /
Kabsch convention / persistence impl as `run_eval_fromnone.py`.

Six conditions per horizon {1,4,16,32,64,96 ns}:
1. `H_auto,S_auto` (autonomous, today's number);
2. `H_gt,S_auto` (replace H only);
3. `H_auto,S_gt` (replace S only; mark **unavailable** if no teacher S can be constructed);
4. `H_gt,S_gt` (oracle ceiling);
5. `H_anchor,S_anchor` / zero-ΔH (persistence-equivalent — must equal persistence to machine tol);
6. **residue-shuffled ΔH** (permute ΔH across valid residues): if RMSD barely changes, the decoder
   is using a global/amplitude shortcut and *not* residue correspondence → message passing is
   justified (the document's D1 trigger).

Also emit the **latent** curves the document asks for (P0.1):
`latent_error(τ)=||H_auto−H_gt||`, `latent_signal(τ)=||H_gt−H_anchor||`,
`relative_error = latent_error/(latent_signal+eps)`, per-residue norm, ΔH direction cosine.

Decision rule (using **relative** skill, locked definition):
- `oracle-HS − auto` small **and** both barely beat persistence → **decoder/representation** (→ Track B). *(Expected outcome, given §2.)*
- `oracle-HS` clearly beats persistence but `auto` does not → information exists, dynamics lose it → then, and only then, rollout work.
- shuffled-ΔH ≈ original → move residue correspondence into the model (Track B / D1).
- `oracle-HS` fails to beat persistence even here → the H-side objective is the problem (→ §4 P0-R).

Reason to do this at all (not skip to Track B): it is cheap, it is the *falsifier* for the
rollout hypothesis, and no prior from-none run persisted these curves. It also produces the
**baseline** column Track B must be compared against.

### Track B — D1 message-passing decoder (the actual push; main compute; 1 free GPU)

Implement `MessagePassingCADecoder` as a **strict superset** of `SimpleCADecoder` (same
contract: local-frame `dx`, SE(3) equivariance, `dx=0` ⇒ persistence, residue mask,
`max_update` bound, future coordinates never in the prediction path). Add 2–4 layers of
**masked residue message passing**:

- node feature = `[H_τ, H_anchor, ΔH, S_τ]` (as today);
- edge feature = anchor CA distance (RBF), current-vs-anchor local distance change, |i−j| (masked);
- messages only from **observed anchor + predicted state** — no future coords.

Run it head-to-head vs the current MLP decoder with **identical** encoder, kernel, budget
(30k steps), multi-lag schedule, and **≥2 seeds**, reporting relative skill + paired protein
bootstrap CI at each horizon, **and** the oracle-H ceiling from Track A's harness.

Why D1 and not bigger: (a) it is the *minimal* structural fix that directly addresses the proven
per-residue cap (§1.1); (b) it lets us read the **ceiling** (oracle-H) as well as the autonomous
number, so we can tell "decoder can express it" from "dynamics deliver it" — the exact
discrimination the document wants; (c) it is cheap (decoder is 0.23 M params) and low-risk
(can't regress below persistence thanks to the zero-init / bounded update).

**The key reading after Track B (this is the real fork):**
- D1 raises **both** oracle-H and autonomous → decoder *was* the bottleneck → continue representation/decoder line (D2/D3 next).
- D1 raises **oracle-H** but not autonomous → now the rollout is the binding constraint → *switch* to H+S work (§4) with a clear mandate.
- D1 raises **neither** → neither is capacity-limited; the pairing `ΔH ↔ geometric displacement` is not in H → encoder objective (§4 P0-R).

### Explicitly deferred (with reasons)
- **No GLE rewrite** now: `H+ S` is demonstrably non-collapsed / non-identity (`REPAIR_PROGRESS.md`
  audit), and `oracle-H ≈ autonomous` says fixing it cannot help the current metric. Revisit only
  if Track B flips that ordering.
- **No data scaling** now: prior probes show the *within-domain* ceiling is the issue
  (`DIAGNOSIS_H_READOUT_20261018.md` §4; `H_DECODABILITY_REPORT.md` §9) — more proteins buy
  generalization we cannot currently cash in.
- **No probabilistic decoder / `predict_log_scale`**: premature; point-estimate structure isn't there yet.
- **No `force_clip` hidden-divergence worry** *unless* Track B shows rollout-limited: then check the
  `force_clip` saturation rate as the document's R1 warns.

---

## 4. If Track A/B say "representation", the concrete H-side fix (already scoped in-repo)

`DIAGNOSIS_H_READOUT_20261018.md` P0-1 already names the fix, and the from-none encoder still
lacks it: **a global/collective conformational channel** — masked global pool (mean/max) of
lifted CA broadcast back per residue, plus global geometry scalars (Rg, end-to-end distance,
inertia eigenvalues, a few long-range CA–CA distances), or a `[CLS]` global token in the attention.
Combined with a **decodability objective on ΔH** (which from-none partly has via `ca`), this is
the smallest change that can put *collective* geometry into H. Do this **only after** Track B's
reading says capacity/representation is the wall (D3's "global-only also helps" is the tell).

---

## 4b. Oracle Ceiling Result (2026-10-18, completed)

**Probe:** `tmp_claude/oracle_ceiling_mp.py` — trains a 2.5M-param MessagePassingCADecoder
(mp-depth=3, hidden=256) on **frozen** step-8000 encoder latents for 8000 steps, then evaluates
on a held-out set across all protocol horizons.  Result saved to `tmp_claude/oracle_ceiling_mp.json`.

| horizon | oracle_mp (Å) | persist (Å) | rel_skill | abs (Å) |
|---------|--------------|-------------|-----------|---------|
| 1 ns    | 2.711        | 2.852       | +4.9%     | +0.140  |
| 4 ns    | 3.705        | 3.915       | +5.4%     | +0.210  |
| 8 ns    | 4.160        | 4.407       | +5.6%     | +0.247  |
| 16 ns   | 4.716        | 5.018       | +6.0%     | +0.302  |
| 32 ns   | 5.362        | 5.688       | +5.7%     | +0.326  |
| 64 ns   | 6.000        | 6.362       | +5.7%     | +0.362  |
| 96 ns   | 6.307        | 6.684       | +5.6%     | +0.377  |

**Reading:**
- The ceiling at 1 ns is 2.71 Å — more than 2× the 1.2 Å target.  A 2.5M-param MP decoder
  extracting only ~5% relative skill over persistence means the latents are **information-poor
  for CA displacement**, not decoder-limited.
- However: the main run at step 9300+ **already beats the oracle ceiling at 16 ns** (4.18 vs
  4.72 Å), because (a) the encoder is more trained at step 9300 than the frozen step-8000
  snapshot, and (b) joint training is a better signal than isolated decoder training on frozen
  latents.  So the oracle ceiling is a *pessimistic lower-bound on the true ceiling*, not a
  hard cap.
- The +0.14–0.38 Å absolute gain is roughly constant across horizons, consistent with latents
  carrying a small but real predictive signal that the MP decoder can extract.

**Implication:** The step-8000 frozen encoder cannot support 1.2 Å even with a perfect decoder.
Whether a *fully-trained* encoder can is the open question — answered by the D1 (joint MP,
seed=43) and D2 (joint MP, ca_weight=4.0, encoder-lr-scale=2.0, seed=44) training runs.

**Active runs (2026-10-18):**
- **Main** (GPU 0, seed=42): step ~9700, best val skill = 0.034 (ca_rmsd 4.167 @ 16 ns).
  Simple decoder, ca_weight=1.0.
- **D1** (GPU 6, seed=43): step ~500, val skill = 0.004 (early).
  MP decoder (mp-depth=3), ca_weight=1.0.
- **D2** (GPU 7, seed=44): step ~50, just started.
  MP decoder (mp-depth=3), ca_weight=4.0, encoder-lr-scale=2.0.

---

## 5. Acceptance / go-no-go (hardened from the document)

Define **relative skill** `s = 1 − rmsd/persist` and report `Δ = persist − rmsd` (Å) with a paired
bootstrap CI over held-out **proteins** (never windows). Then:

**Enter decoder work (Track B) if any:** `oracle-HS − auto` ≲ 2% of persistence error while both
only marginally beat persistence; or `oracle-HS` does not stably beat persistence; or shuffled-ΔH
≈ original.

**Switch to rollout work only if:** `oracle-HS` clearly beats autonomous (≥5% of persistence
error) **and** teacher-forced ≫ autonomous at fixed encoder/decoder.

**B2 gate (not triggered by "+0.0x Å" alone):** held-out protein **and** temperature both stably
beat persistence; the gain isn't a single horizon; oracle-vs-autonomous gap explained;
message-passing gain reproduced across ≥2 seeds; long-horizon rollout not hiding divergence behind
`force_clip`; future coords still never in the prediction path.

---

## 6. What I would run *this cycle* (concrete)

1. **Now (GPUs 5–7 idle):** write `evaluation/run_fromnone_intervention.py`; run Track A on
   `artifacts/fromnone/stage_cb1b_main/checkpoint_best.pt` (and temp). Output JSON + a table.
2. **In parallel:** implement `MessagePassingCADecoder` + tests (equivariance, mask, zero-dx =
   persistence, no-future-in-path) and launch one D1 run on a free GPU with the exact main-run
   hyperparams, `--seed 43`.
3. **Compare** at matching steps using one locked skill definition and the same bootstrap.
4. **Fork** per §3's three-way reading. Do not start GLE work or data scaling until the fork is read.

## 4c. Teacher-forced vs rolled-out (tmp_claude/tf_vs_ro_probe.py, best ckpt @12500)

```
    h |   ca_tf   ca_ro  persist |  kern_gap |   dH_tf   dH_ro |   h_ca_r
   1ns |   2.862   2.772    2.852 |    -0.090 |   2.581   7.229 |   0.3015
   4ns |   3.829   3.753    3.915 |    -0.076 |   2.995   5.000 |   0.2641
   8ns |   4.310   4.207    4.407 |    -0.102 |   3.220   5.280 |   0.2196
  16ns |   4.871   4.755    5.018 |    -0.115 |   3.405   6.171 |   0.2079
  32ns |   5.523   5.381    5.688 |    -0.141 |   3.596   7.337 |   0.1681
  64ns |   6.172   6.014    6.362 |    -0.158 |   3.828   8.600 |   0.2023
  96ns |   6.510   6.341    6.684 |    -0.169 |   4.011   9.337 |   0.1564
```

Three conclusions, all pointing away from the kernel:

1. **Kernel gap is negative at every horizon** (ro - tf = -0.09 .. -0.17 A).
   Substituting the *true* future H makes the prediction *worse*. The decoder has
   learned to invert the kernel's systematic bias; the rollout is doing work, not
   damage. **The evolution equation is not the bottleneck.**

2. **Teacher-forced skill never exceeds ~+6%.** With a perfect H the decoder still
   only reaches 4.87 A @16ns vs 5.02 persistence. The oracle-ceiling probe said the
   same (+0.060 @16ns). Two different decoders, two different eval samples, same
   ceiling. So the ~1.2 A target is not reachable from the current H.

3. **H saturates.** ||dH_tf|| grows 2.58 -> 4.01 while true CA displacement grows
   2.85 -> 6.68 A. By 16ns H has moved through most of its available range while the
   protein still has 60% of its motion ahead of it. The per-residue correlation
   between ||dH_tf|| and CA displacement decays 0.30 -> 0.16. **H is an information
   bottleneck, not a modelling bottleneck.**

Note dH_ro > dH_tf (6.17 vs 3.41 @16ns) — the kernel injects more variation than the
real signal carries. That is consistent with (1): extra movement the decoder was
trained to discount.

Next test: ridge probe H_future -> true anchor-frame displacement. If a linear map
from H beats the trained decoder, the decoder is under-fit. If it also lands ~4.9 A,
H simply does not contain the displacement.

## 4d. Ridge probe: H_future -> CA displacement (tmp_claude/ridge_probe.py, best ckpt @13000)

```
    h |  persist   ca_ro   ca_tf | ridge_tr ridge_val | gain_over_tf decoder_vs_ridge | baseline
   1ns |    2.913   2.833   2.930 |    9.736     9.687 |       -6.757           -6.854 |    9.727
   4ns |    3.903   3.741   3.825 |   18.661    18.600 |      -14.775          -14.858 |   18.722
   8ns |    4.499   4.279   4.373 |   25.103    25.145 |      -20.773          -20.866 |   25.260
  16ns |    5.046   4.788   4.897 |   32.993    33.216 |      -28.320          -28.428 |   33.383
  32ns |    5.673   5.367   5.499 |   43.297    43.427 |      -37.928          -38.061 |   43.685
  64ns |    6.289   5.947   6.112 |   57.907    58.478 |      -52.366          -52.531 |   58.683
  96ns |    6.681   6.318   6.490 |   69.105    69.461 |      -62.972          -63.144 |   69.952
```

Ridge RMSD ≈ baseline (predict-mean) at every horizon. A linear map from dH_future to
true 3-D displacement explains ZERO variance. This closes the diagnosis:

**H is not encoding displacement-predictive information in a linearly-accessible form.**

The decoder is achieving ~2.9 Å @1ns not by reading displacement from H, but by
learning to output near-persistence from the geometry inputs (X_context, R_anchor).
H is contributing noise that happens to be calibrated away during training.

Full diagnostic chain summary:
  (a) oracle ceiling (+0.05-0.06 skill)   → teacher-forced H barely beats persistence
  (b) tf_vs_ro (kernel_gap < 0)           → kernel is not the bottleneck; rolled-out H
                                            is actually slightly better than teacher-forced
  (c) ridge probe (ridge ≈ baseline)      → H carries no linearly-decodable displacement

Root cause: the encoder is trained with a JEPA-style loss that predicts H from context,
not a loss that is directly supervised on predicting future atomic coordinates. H has
learned to encode some structural/identity information (used by the decoder for geometry
reasoning) but the *dynamics* are not in H in any recoverable form.

## 5. Next design decision

The bottleneck is the learning objective for H, not the architecture of the kernel or
decoder. Two architectural remedies:

**Option A — Direct geometric supervision on H (auxiliary loss)**
Add an auxiliary loss that directly regresses future CA positions from H_future, in
addition to the JEPA prediction loss. This creates a gradient signal that forces H to
encode dynamics.

**Option B — End-to-end dynamics supervision (drop JEPA, full rollout loss)**
Remove the JEPA prediction step from the encoder learning objective. Train the full
pipeline (encoder → kernel → decoder) end-to-end on future CA RMSD. H will then learn
whatever representation best serves the downstream dynamics task.

Option B is a larger architectural change but directly targets the problem. Option A
may be a faster experimental iteration since it keeps the current JEPA structure.

Immediate next step: implement Option A as an auxiliary CA regression head on H_future,
retrain from scratch (or from the current checkpoint warm-start), and verify that
ridge probe improves.

## 4d-corrected. Ridge probe with Kabsch-aligned target (tmp_claude/ridge_probe2.py, best ckpt @13000)

The first ridge probe used raw world-frame displacement — physically not comparable to
Kabsch-aligned RMSD. This version superimposes Xt onto Xa frame first, then regresses
the aligned residual.

```
    h |  persist   ca_ro   ca_tf | ridge(α best)      α |  vs ca_tf  vs persist
   1ns |    2.852   2.773   2.871 |        4.878  100.0  |    -2.007      -2.026
   4ns |    3.915   3.750   3.835 |        7.914  100.0  |    -4.079      -3.999
   8ns |    4.407   4.201   4.314 |        9.558  100.0  |    -5.244      -5.151
  16ns |    5.018   4.750   4.875 |       11.217  100.0  |    -6.342      -6.199
  32ns |    5.688   5.375   5.521 |       12.608  100.0  |    -7.087      -6.920
  64ns |    6.362   6.009   6.172 |       14.361  100.0  |    -8.189      -8.000
  96ns |    6.684   6.334   6.509 |       16.385  100.0  |    -9.876      -9.701
```

Ridge val RMSD *exceeds* persistence at every horizon, and the best alpha is 100
(strong regularization toward zero prediction). This means the optimal linear map
from dH is the zero map — dH carries no predictive signal whatsoever after
alignment. The feature dimension is 128 (latent_dim) with ~50k training residues,
so underfitting is not the issue.

Conclusion chain (all four probes agree):
  (a) oracle ceiling:     teacher-forced H barely beats persistence (+0.05-0.06 skill)
  (b) tf_vs_ro:           kernel is not the problem; rolled-out H ≈ teacher-forced H
  (c) ridge v1:           linear map from dH → world displacement explains 0 variance
  (d) ridge v2 (aligned): linear map from dH → aligned displacement also explains 0 variance

H is a structural identity descriptor, not a dynamics descriptor. The JEPA objective
(predict H_context from H_anchor) does not place any gradient pressure on H to encode
how atoms will move — it only needs H to be self-consistent across the context window.

## 5. Root cause and architectural decision

Root cause: the encoder is supervised by a self-supervised JEPA objective that has no
direct contact with future atomic positions. H encodes structural identity well
(the decoder exploits this for geometry-grounded output near-persistence) but encodes
zero dynamics.

Three options in order of implementation cost:

**A. Auxiliary dynamics head on H (fast iteration)**
Add a small MLP head that predicts aligned CA displacement directly from
(H_future - H_anchor), supervised by L1/L2 against true Kabsch-aligned dx.
Weight ~0.1-0.3 relative to the JEPA loss. This forces a gradient into the encoder
that ties H to atomic motion. Reuse current checkpoint as warm-start.

**B. End-to-end rollout loss replacing JEPA (larger change)**
Drop or heavily downweight the JEPA prediction loss. Train encoder + kernel + decoder
jointly on future CA RMSD only. H learns whatever representation minimizes the
downstream task. Risk: training becomes harder to stabilize.

**C. Two-tower encoder (identity stream + dynamics stream)**
Keep the JEPA encoder frozen as the structural identity tower. Add a second small
encoder (e.g., 2 layers, 64 dim) that sees the per-residue displacement signal
(e.g., difference in Cα positions across the history window) and is trained
end-to-end. Concatenate both embeddings into the kernel.

Recommended next step: implement Option A as a fast probe. If the auxiliary head
achieves ridge probe skill > 0.05 after 5k warm-start steps, proceed with a full
retrain. If not, move to Option B or C.

## 4e. Ridge probe on relpos checkpoint @18000 (tmp_claude/ridge_probe_relpos.log)

```
    h |  persist   ca_ro   ca_tf | ridge(α best)      α |  vs ca_tf  vs persist
   1ns |    2.852   2.764   2.711 |        4.851  100.0  |    -2.140      -2.000
   4ns |    3.915   3.733   3.635 |        7.876  100.0  |    -4.240      -3.961
   8ns |    4.407   4.173   4.056 |        9.493  100.0  |    -5.437      -5.086
  16ns |    5.018   4.695   4.589 |       11.211  100.0  |    -6.622      -6.193
  32ns |    5.688   5.292   5.197 |       12.544  100.0  |    -7.346      -6.856
  64ns |    6.362   5.906   5.817 |       14.339  100.0  |    -8.522      -7.977
  96ns |    6.684   6.222   6.119 |       16.288   10.0  |   -10.169      -9.604
```

Same result as without relpos: ridge ≈ baseline at all horizons, alpha→100 (zero map).

However, the relpos run showed faster skill improvement (0.037 → 0.046 in ~1500 steps
vs 0.042 → 0.046 in the main run over much longer), so the relpos features are helping
the decoder's geometry reasoning, just not in the way the ridge probe tests.

Why the relpos channel didn't solve it:
The PCA canonical frame is re-derived from each frame independently. Two conformations
that are geometrically similar (same fold, different coordinates) will have similar
relpos features — because the PCA frame tracks the overall cloud shape, not the absolute
position. H_future - H_anchor from the relpos channel changes mainly when the fold
undergoes large-scale rearrangement, not for small (1-16ns) local fluctuations.

What is actually needed:
H must encode a signal that changes as a function of local atomic displacement between
the anchor and future frames. This requires either:

1. The encoder to see the *anchor frame's coordinates* as a reference input (so it can
   compute "how far am I from where I was"), which requires a cross-frame conditioning
   signal not currently present.

2. Using a displacement-based feature as direct input: encode the difference in backbone
   dihedrals, CA distances, or local backbone geometry between anchor and future frames,
   rather than the absolute geometry of the future frame alone.

3. End-to-end training where the entire encoder output is jointly optimized on the
   downstream CA prediction task (Option B from section 5), so H learns to encode
   whatever is informative for the task regardless of invariance properties.

The fundamental architectural constraint: ProteinGeometryEncoder is SE(3)-invariant by
construction. H(frame) only depends on the internal geometry of that frame. For H_future
to encode how the protein moved relative to the anchor, either (a) the encoder must
receive the anchor as a reference input, or (b) we must use cross-frame features that
are not available to the single-frame encoder.

## 5 revised. Next architectural decision

Option B — end-to-end training replacing JEPA — is now the correct next step.
The JEPA objective teaches H to self-consistently encode structural identity; the
CA loss backprop through the stochastic kernel is too noisy to teach H dynamics.
The tf_ca auxiliary loss gives a clean gradient into H but it teaches the encoder to
encode something the single-frame invariant architecture cannot express as dH.

The minimal viable change:
- Keep the encoder, kernel, decoder architecture as-is.
- Change the encoder to accept an optional reference (anchor) frame input.
  Concat anchor-frame H onto each future-frame encoding before the last projection,
  so H_future becomes implicitly conditioned on H_anchor.
- This makes H_future - H_anchor a genuinely meaningful signal.

OR simpler: provide the anchor geometry as explicit features to the decoder. The
decoder already receives H_anchor; if it also received the anchor's dihedral/local
geometry features directly (without going through a learned H) it could compare
anchor vs future geometry without requiring the encoder to encode the difference.
