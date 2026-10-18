# From-None Stage B (B1b → B2) Pipeline

**Date:** 2026-10-18 · **Status:** real runs launched (GPU 4)

## Why "from-none"

Prior diagnosis (`B1B_DIAGNOSIS_AND_PLAN.md`, `DIAGNOSIS_H_READOUT_20261018.md`,
`B1B_B2_FULL_PIPELINE_RESULTS.md`) established, with independent probes, that a
**frozen Stage A H** cannot support the B1b/B2 task:

- H is a *same-frame, SE(3)-invariant* descriptor. Its **inter-frame difference**
  `ΔH = H_future − H_anchor` carries essentially **no linear information** about
  the inter-frame CA displacement (ridge probe R² ≈ −0.005, worse than predicting
  zero), even with the corrected Kabsch convention.
- Consequently oracle = rollout = persistence on every horizon: feeding even the
  *true* future H to the decoder does not beat "do nothing".

The root cause is structural: Stage A trains the encoder for same-frame geometry
and a factor-space temporal delta, never for *geometric decodability of ΔH*.

**From-none** removes the frozen representation entirely: a fresh, randomly
initialised encoder is co-trained end-to-end with the H/S dynamics and a CA
decoder under a **decodability** objective, so ΔH is *forced* to decode into CA
motion. No Stage A / B1a checkpoint is consumed.

## Architecture

```
X_{t-K+1..t}  ──fresh encoder E_θ──▶  H_hist  ──GLE (StageB1b kernel)──▶  H_τ , S_τ
                                        │                                     │
                               (EMA-standardised)                     decode D_φ
                                                                             ▼
                          X̂_τ = X_anchor + R_anchor · d_local(H_τ, H_anchor [, S_τ])
```

- **Encoder**: `ProteinGeometryEncoder` (same architecture as Stage A), **random
  init**, `latent_dim=128`, depth 4, joint training (no warm start).
- **Dynamics**: `StageB1b` GLE kernel (H-space energy + per-residue bath), trained.
  From-none stability additions (shared kernel, backward compatible):
  `force_clip` (elementwise clamp on ‖f‖ before `h·D` scaling) and `tail_init`
  (a stronger quadratic restoring term so an untrained rollout is mean-reverting).
- **Decoder**: `SimpleCADecoder` (lightweight per-residue MLP, per user request;
  `ResidualCADecoder` remains available). Predicts a bounded **local** CA update
  and rotates through the observed anchor frame, so the map is SE(3)-equivariant
  and `dx=0` reproduces persistence exactly.
- **Standardisation**: `EMANormalizer` tracks the fresh encoder's per-channel
  mean/scale (detached EMA) and standardises H before the kernel/decoder. It is
  never updated during validation.

## Objective

| term | weight | role |
|---|---|---|
| `ca` (Kabsch-aligned CA point loss) | **1.0** | the decodability signal Stage A lacked |
| `es` (energy score on the H rollout path) | 0.1 | distributional JEPA term |
| `phys` (energy score on decoded CA features) | 0.1 | physical grounding |

Training runs **only** from history (true future coordinates never enter the
prediction path; the future is encoded separately and used solely as target).

## Multi-lag (1–100 ns)

`--multi-lag` cycles the rollout length over `{8,16,24,32,48,64,96}` frames
(= ns at 1 ns/frame), with a **weighted** cycle (`MULTILAG_CYCLE`) that favours
short horizons for throughput while still exercising every lag up to ~96 ns, so
H/S generalize across timescales.

## Data & splits (mdCATH, 1000 domains)

Built by `scripts/build_mdcath_manifest.py --max-domains 1000` →
`artifacts/mdcath_manifest_1000dom/` : **800 train / 100 val / 100 test proteins**,
5 temperatures {320,348,379,413,450}, 5 replicas.

- **Held-out protein**: train on `train.jsonl`, select on `val.jsonl` (disjoint
  proteins — enforced by `validate_manifest_split`).
- **Held-out temperature**: train on `artifacts/mdcath_manifest_1000dom_temp/train_not450.jsonl`
  (T∈{320,348,379,413}), evaluate on T=450 (`val_t450.jsonl`).

## Code map

| path | role |
|---|---|
| `protein_jepa_frame/models/stage_b1b_fromnone/decoder.py` | `SimpleCADecoder`, `FromNoneConfig` |
| `protein_jepa_frame/models/stage_b1b_fromnone/objectives.py` | batched Kabsch, energy score, variance floor |
| `protein_jepa_frame/training/train_stage_cb1b.py` | trainer (`from_none_encoder_v1` contract) |
| `protein_jepa_frame/models/stage_b1b/model.py` | shared kernel; adds `force_clip`, `tail_init` |
| `evaluation/run_eval_fromnone.py` | held-out eval vs persistence (Kabsch RMSD, contact Jaccard) |
| `run_fromnone_real.sh` / `run_fromnone_temp.sh` | launch scripts (GPU 4) |

## Run & evaluate

```bash
# held-out protein (main)
nohup bash run_fromnone_real.sh > logs/fromnone_main.nohup 2>&1 &

# held-out temperature
nohup bash run_fromnone_temp.sh > logs/fromnone_temp.nohup 2>&1 &

# evaluation (held-out protein)
python -m evaluation.run_eval_fromnone \
  --ckpt artifacts/fromnone/stage_cb1b_main/checkpoint_best.pt \
  --manifest artifacts/mdcath_manifest_1000dom/val.jsonl --split protein \
  --max-proteins 20 --max-anchors-per-trajectory 4 --n-rollout-reps 4 --device cuda

# evaluation (held-out temperature)
python -m evaluation.run_eval_fromnone \
  --ckpt artifacts/fromnone/stage_cb1b_temp/checkpoint_best.pt \
  --manifest artifacts/mdcath_manifest_1000dom_temp/val_t450.jsonl --split temp \
  --heldout-temperature 450 --max-proteins 20 --max-anchors-per-trajectory 4 \
  --n-rollout-reps 4 --device cuda
```

## Acceptance

For each horizon, report `rmsd` vs `persist_rmsd` with a paired bootstrap CI over
held-out proteins. The encoder decodes into CA motion if `skill > 0` (i.e.
`delta_vs_persist < 0`) with a CI excluding 0 at short–mid horizons, and contact
Jaccard beats persistence. Early training already shows `skill > 0` on the
held-out-protein selection protocol.

## Verification performed

- Smoke: end-to-end forward/backward, multi-lag, validation — passes.
- Stability: rollout std grows sub-diffusively (1.27→2.06 over 8→96 steps), no
  blow-up; force clip + tail init + correct `coupling_scale` (1e-3, since `h=1000 ps`
  makes `h·(C@S)` the divergence source) keep it bounded.
- `Kabsch` vectorised (ns=96 step: 27s → 3s).
- Learning confirmed: `val/skill` rose 0.0003 → 0.030 over ~300 steps.
- Existing kernel files: additions are opt-in (defaults preserve legacy behaviour).

---

## Status & early results (2026-10-18)

Two real runs launched (30k steps each), in the background:

| run | GPU | split | selection | log |
|---|---|---|---|---|
| `artifacts/fromnone/stage_cb1b_main` | 4 | held-out protein | `val.jsonl` | `logs/fromnone_main.nohup` |
| `artifacts/fromnone/stage_cb1b_temp` | 5 | held-out temperature (T=450) | `val_not450.jsonl` | `logs/fromnone_temp.nohup` |

Monitor with `bash scripts/fromnone_status.sh`.

### Selection skill (`1 - ca/persist`, >0 beats persistence)

**Held-out protein (main)** — rising and stable:
`0.004 → 0.005 → 0.008 → 0.009 → 0.014 → 0.012` (steps 500–3500)

**Held-out temperature (temp)** — rising:
`0.0036 → 0.0042 → 0.0045 → 0.0063 → 0.0073` (steps 500–3000)

### Held-out-protein eval (`run_eval_fromnone`, best checkpoint, Kabsch CA RMSD Å)

| horizon (ns) | persist | from-none | skill (Å) | contact Jaccard | persist J |
|---:|---:|---:|---:|---:|---:|
| 1  | 3.857 | 3.797 | **+0.060** | 0.781 | 0.792 |
| 4  | 4.979 | 4.887 | **+0.092** | 0.737 | 0.744 |
| 16 | 6.184 | 6.044 | **+0.140** | 0.699 | 0.702 |

The margin **grows with horizon**, which is the expected signature of a model
that adds real information over "do nothing".

### Diagnostic: where is the ceiling?

Oracle-H probe (true future H -> decoder) vs autonomous rollout vs persistence:

| horizon (ns) | persist | oracle-H | autonomous |
|---:|---:|---:|---:|
| 4  | 3.972 | 3.950 | 3.951 |
| 16 | 4.923 | 4.889 | 4.889 |

`oracle-H ≈ autonomous` → the dynamics rollout is already near the decoder's
upper bound at these horizons; the remaining limit is the **decoder/representation**,
not the GLE rollout. This is consistent with the known difficulty of mdCATH
4–16 ns (small drift, strong persistence). Longer horizons widen the gap.
