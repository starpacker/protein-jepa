# OPF for Stage-A: Orthogonal Predictive Factorization of the JEPA Paired Target

Source: JEPA-Anything (arXiv 2609.20800) — `paper/jepa-anything.md`.

## 1. What OPF is (from the paper)

Standard Stage-A JEPA predicts a **single monolithic latent target**. OPF partitions the
target space into `K` learned projectors `P_k ∈ R^(d×r)` (with `K·r = d`), predicts each
factor through a **dedicated branch**, and synthesizes a complete latent state via the
Moore–Penrose pseudoinverse of the analysis map.

Key mechanism (Sec 2.2–2.3):

- **Factorization**: `z_t^(k) = P_k^T · sg(z_t)`, each factor is an `r`-dim subspace.
- **Dedicated predictors** `q_k(context) -> r`-dim.
- **Synthesis**: `z_hat = (P^T)† · [q_1, ..., q_K]` (transpose synthesis when `P` is orthogonal).
- **Orthogonality loss** (Eq 7): within-factor `||P_k^T P_k − I_r||_F²` + cross-factor `Σ ||P_i^T P_j||_F²`.
- **Factor activity loss** (Eq 9, `L_fac`): keeps every projected coordinate active (`σ >= γ_fac`).
- **Encoder activity loss** (`L_enc`): anti-collapse gradient on the online encoder state.
- **Total**: `L_train = L_base + λ_orth L_orth + λ_fac L_fac + λ_enc L_enc`.

The key benefit (Prop 1 + Cor 1): non-overlapping + unit-conditioned coordinates →
`κ₂(P)=1`, `σ_min(P)=1`, stable long-horizon rollout. The matched ablation "1 shared target
vs 4 orthogonal factors" is the cleanest evidence it works.

## 2. Current Stage-A structure in our code

- Model: `ProteinJEPA` (`protein_jepa_frame/models/jepa.py`), Stage-A via `forward_paired`.
- Predictor: `PairedDeltaPredictor` / `HistoryConditionedTransition` (`predictor.py`) — a
  **single** trunk predicting a **full `d`-dim** `delta_target = future_endpoint − delta_anchor`.
- Config: `ModelConfig` (`config.py`), we add OPF flags there.
- Loss: `paired_forecast_terms(outputs)` (`jepa.py:2393`) computes `delta_loss` on the single
  `delta_prediction`; weights are in `OptimizationConfig.paired_delta_weight` etc.

**Identified gap**: our existing "orthogonality" lives in the LSP-WM Mori bath / memory
layer (Stage-B, `v51_mode_orthogonality`, `GaugeFixedBath`), **not** in the Stage-A predictive
target. OPF is exactly the missing piece.

## 3. Implementation plan

### 3.1 New module: `OPFPredictor` (in `models/predictor.py` or a new `models/opf.py`)

A factored predictor that mirrors the paper's mechanism inside the existing paired-path:

```
class OPFPredictor(nn.Module):
    """K orthogonal factors, each with a dedicated residual head."""
    def __init__(latent_dim, num_factors=4, factor_activity_gamma=0.5,
                 depth=2, dropout=0.1, thermodynamic_conditioning=True):
        self.projectors = nn.ParameterList(  # P_k ∈ R^(d×r)
            [nn.Parameter(std_orth_(torch.zeros(d, d//K))) for k in range(K)]
        )
        # dedicated branch per factor, each output latent_dim//K
        self.heads = nn.ModuleList([PairedDeltaPredictor(r, depth, ...) for k in range(K)])

    def forward(endpoint_state, horizon_time_ps, temperature_k, dataset_index) -> dict:
        # factorize the (stop-grad) target into K components
        # predict each factor; concatenate; synthesize; return dict
```

- `num_factors = 1` must **exactly reproduce** the current single-head behavior (the matched
  ablation baseline). When `K==1`, `r == d`, the orthogonality term vanishes, and synthesis is
  identity — this is the clean control.
- Optionally a **shared trunk + K branch heads** (as the paper permits) to keep parameters
  matched against the unconstrained multi-head control.

### 3.2 Wire into `forward_paired` (`jepa.py`)

- After computing `delta_target`, also compute factor projections:
  `factor_targets = [P_k^T @ delta_target]` (with stop-grad on target as the paper does).
- Add `self.opf_predictor`: if `config.opf_factor_count > 1`, build OPF; else None.
- When enabled, replace the single `delta_prediction` with the OPF output:
  `outputs["delta_prediction"] = synthesize([q_k(...)])`.
- Keep `state_prediction = prediction_anchor + delta_prediction` unchanged so downstream
  (M1/R1/DetSkill, LSP-WM) reads the **synthesized full state** as before.
- Emit diagnostic tensors in `outputs`: `factor_coords` (per-factor predictions),
  `factor_targets`, `projectors` (for the audit metrics).

### 3.3 Add OPF loss terms

New standalone `opf_terms(outputs)` (mirroring `v51_mode_orthogonality_loss` pattern):

- `pred_loss`: MSE over all factors `Σ_k ||q_k − P_k^T delta_target||²` (this *replaces*/mixes
  with the scalar `delta_loss`).
- `orth_loss`: within-factor `||P_k^T P_k − I||_F²` + cross `||P_i^T P_j||_F²`.
- `factor_activity_loss`: `Σ max(0, γ − σ(coord_k,j))`.
- `encoder_activity_loss`: `Σ max(0, γ_enc − σ(online context state_j))`.

### 3.4 Config (`config.py`, both ModelConfig and OptimizationConfig)

```
# ModelConfig
opf_enabled: bool = False
opf_factor_count: int = 4          # 1 == matched baseline (single head)
opf_shared_trunk: bool = True      # true -> shared trunk + K heads (parameter-matched)
opf_predictor_depth: int = 2
opf_activity_gamma: float = 0.5
opf_encoder_activity_gamma: float = 0.5

# OptimizationConfig
opf_pred_weight: float = 1.0       # scales factor prediction loss
opf_orth_weight: float = 0.1       # λ_orth
opf_factor_activity_weight: float = 0.05   # λ_fac
opf_encoder_activity_weight: float = 0.05  # λ_enc
```

### 3.5 Train-loop wiring (`train.py`)

- At the `total_loss` assembly (near the `paired_delta_weight` block, ~line 1033), add the
  four OPF terms when `opf_enabled`.
- Log `opf_*` scalars alongside existing metrics (like line 1624 logs v51 terms).

### 3.6 Matched ablation (the paper's cleanest experiment)

`stage_a_factor_count ∈ {1, 4}`, everything else identical (data / encoder / backbone /
optimization budget / eval split). This is the JEPA-Anything "1 shared target vs K factors"
control, summarized in a report. `K=1` is the implementation-time regression guarantee.

### 3.7 Orthogonality mechanism audit (mirror Table 5)

Add to `protein_jepa_frame/eval/m1k1r1_benchmark.py` a new function, evaluated on the
trained Stage-A model (extract from `outputs["projectors"]` / `factor_coords`):

- **Cross-factor overlap**: `Σ_{i<j} ||P_i^T P_j||_F² / (K(K-1)/2)`.
- **σ_min(P)**: min singular value of concatenated `P = [P_1, ..., P_K]`.
- **κ₂(P)**: condition number `σ_max(P)/σ_min(P)`.
- **Transpose-synthesis NMSE**: `||P·(P^T z) − z||² / ||z||²` for exact factors.
- **Factor activity**: fraction of coordinates with `σ >= γ`.

These join the existing M1 `force_source/diffusion_source/kernel_norm` fingerprint checks.

## 4. Boundary / do-not-copy

We keep Stage-B thermodynamics (GENERIC, PMF, Mori-Zwanzig memory, temperature) intact.
OPF is a **Stage-A predictive prior**; the paper's "orthogonality → numerical stability →
stable rollout" complements (not replaces) our "thermodynamic consistency → correct dynamics".
Result: a combined "representation (OPF) + structure (thermo)" story.

## 5. File touch list

| File | Change | Status |
|------|--------|--------|
| `protein_jepa_frame/models/opf.py` (new) | `OPFPredictor` + `opf_terms()` | ✅ Implemented |
| `protein_jepa_frame/models/jepa.py` | Import `opf`; validate + build `opf_predictor`; wire into `forward_paired`; export audit tensors | ✅ Implemented |
| `protein_jepa_frame/config.py` | Add ModelConfig + OptimizationConfig OPF fields + state_dict recovery | ✅ Implemented |
| `protein_jepa_frame/training/train.py` | Import `opf_terms`; compute `opf_terms_dict`; add to `total_loss`; log scalars | ✅ Implemented |
| `configs/lspwm_v44_stage_a_opf_k1_control.yaml` | Matched single-head control (K=1) | ✅ Created |
| `configs/lspwm_v44_stage_a_opf_k4.yaml` | OPF 4-factor config | ✅ Created |
| `protein_jepa_frame/eval/m1k1r1_benchmark.py` | Add `eval_opf_orthogonality_audit()` (Table-5 metrics) | ✅ Implemented |
| `docs/OPF_STAGE_A_REPORT.md` | Matched-ablation results + audit metrics | ⏳ To run |

### Implementation status (verified)

- `OPFPredictor` with orthogonal-initialized projectors, shared-trunk or independent heads,
  pseudoinverse synthesis `(P^T)^+`, and the four losses (pred/orth/fac/enc) — smoke tested.
- `K==1` reduces cleanly (single factor, no cross-term, identity-overlap) — the matched baseline.
- End-to-end `forward_paired` + `opf_terms` + backward gradient flow to projectors — verified.
- Config loading + `model_config_from_dict` recovery of `opf_factor_count` — verified.
- State-dict save/load roundtrip (0 missing / 0 unexpected) — verified.
- YAML duplicate-key merge for appended `model:`/`optimization:` blocks — verified OK.

### Remaining / next steps

~~1. Launch the matched ablation (K=1 vs K=4) on the Stage-A manifest.~~
~~2. Run `eval_opf_orthogonality_audit()` to capture the Table-5 metrics.~~
~~3. Summarize in `docs/OPF_STAGE_A_REPORT.md`: endpoint skill + rollout stability + geometry audit.~~

### Status (completed 2026-09-27)

The matched ablation and full evaluation are done — see `docs/OPF_STAGE_A_REPORT.md`.
The first implementation was diagnosed (against the official `jepa_anything` repo) and
fixed for three bugs: a dead-saddle predictor init, detached projector targets killing
`L_pred`/`L_fac` projector gradients, and an impossible activity floor. After the fixes,
the OPF geometry converges to exact Prop-1 (κ₂=1.0001, cross-overlap ≈ 1e-17,
synthesis NMSE ≈ 7e-14) and the K4 4-factor split trains to 72.8% folding-kinetics
skill vs the K1 single-factor control's 80.5%.