# Canonical checkpoints for evaluation (locked 2026-10-17)

Every evaluation run records these paths + SHA256 in `meta.json`. If a file's hash
changes, results are **not** comparable across runs.

## SHA256 (full)
| checkpoint | sha256 |
|---|---|
| `artifacts/abc_v3/stage_a_fix4/checkpoint_latest.pt` | `144fe92a1656a97b73186dfc284f4d764d982d9b026784d28856561f9bddeb6b` |
| `artifacts/abc_v3/stage_a_mixed/checkpoint_latest.pt` | `fa78d675a8d94da6aeace1e48cd25d9e838f36579de1cee9704fc64530fa1a56` |
| `artifacts/abc_v3/stage_a_fix4_frozen/checkpoint_step6450.pt` | `3db3ed9cca4554ab224a0c94a5a48bce052d8e9fbbe8513b31cc1c3320192e04` |
| `artifacts/abc_v3/stage_b1a_fix4_unfrozen/checkpoint_latest.pt` | `0b163398c4ef5e3f8b1d046da5ad515d6032b3b90bd00712268dcae99b5ba568` |
| `artifacts/abc_v3/stage_b1a_frozen/checkpoint_latest.pt` | `08e78ff7401b186b629961b3ccb66e1ffe24197c65cf924399ce2cf45fd37e08` |
| `artifacts/abc_v3/stage_b1b_mdcath/checkpoint_latest.pt` | `2651ad98c3c8acefd55bc7b417c89d23d05b41076898728149d48f0faa63019c` |
| `artifacts/abc_v3/stage_b1b_mixed/checkpoint_latest.pt` | `e7662bd8f40cb9d1d302f1ec5aa384efff317c6e51ab271de56bff618c843851` |

## Variant → module mapping

| Variant | encoder source | predictor/OPF | decoder | dynamics |
|---|---|---|---|---|
| `h_only` | `stage_a_fix4/...pt` key `model` (step 14000) | predictor + OPF from explicit `stage_a_predictor_ckpt` | B1a oracle decoder trained on this H | direct predictor |
| `h_ft` | `stage_b1a_fix4_unfrozen/...pt` key `encoder` (step 3000, top-2 blocks unfrozen) | predictor + OPF from explicit Stage-A source checkpoint | B1a oracle decoder | direct predictor |
| `h_plus_s` | same as `h_ft` (frozen after B1a) | not used for transition; history initializes `S0` | `ResidualCADecoder` from B1b ckpt key `decoder` | `StageB1b` key `model`, GLE rollout |
| `persist` | — | — | identity | — |

For `h_only` and `h_ft`, the predictor receives the complete
`H_context [B,K,L,dH]`.  Its checkpoint output is
`pred_factors [B,L,J,r]`; evaluation constructs
`H_future = OPF.compose(OPF.decompose(H_anchor) + pred_factors)` before calling
the decoder.  The factor output is not itself treated as an absolute `H_future`.

## ⚠️ Lineage inconsistency found (2026-10-17) — DECIDE BEFORE TRUSTING CROSS-VARIANT NUMBERS

The H lineage is **not a single chain**. Concretely:

1. `stage_b1a_fix4_unfrozen` was trained with `stage_a_ckpt =
   artifacts/abc_v3/stage_a_fix4_frozen/checkpoint_step6450.pt`
   (recorded `step=6000` inside; config says 6450) — i.e. its fine-tuned `encoder`
   descends from **`stage_a_fix4_frozen`**.
2. Both B1b runs (`stage_b1b_mdcath`, `stage_b1b_mixed`) were trained with
   `stage_a_ckpt = artifacts/abc_v3/stage_a_mixed/checkpoint_latest.pt`
   and `b1a_ckpt = null` — i.e. the GLE dynamics were **never** trained on the
   fine-tuned encoder. They sit on `stage_a_mixed`.

Consequences for a fair 4-way comparison:
- `h_only` (fix4) vs `h_ft` (fix4_frozen→unfrozen) vs `h_plus_s` (mixed) mixes **three**
  different Stage-A bases. Any conclusion like "H+S helps" would be confounded by the
  encoder swap, not by the dynamics equation.
- All Stage-A checkpoints in this lineage have `delta_target=False` (the known-bad target;
  see `H_quality_step5000.md` §3 and `B1B_DIAGNOSIS_AND_PLAN.md`).

**Resolution (T0.2 decision, to be honoured by the runner):**
- The runner takes an explicit `--encoder-base` and `--b1b-ckpt`; it does **not** silently
  mix. Two headline configurations are supported and both are reported:
  - **CONFIG-A ("as-trained", for diagnosing current artifacts):** encoders exactly as each
    run trained them. Reported with a loud caveat that the base differs.
  - **CONFIG-B ("aligned base", the scientifically valid comparison):** one shared Stage-A
    encoder for all four variants; the B1b/H-ft projections must be **retrained on that same
    base** before this config is meaningful. Until such a retrain exists, CONFIG-B is a
    placeholder and the runner refuses to emit skill numbers for it (fails fast).
- `meta.json` always records which config was used and the encoder hash per variant.
