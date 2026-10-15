# M01: Temporal Interface Specification

> Date: 2026-09-28 | Status: Verified | Version: 1.0

## 1. Notation

| Symbol | Shape | Description |
|--------|-------|-------------|
| X_t | [B, L, 4, 3] | Current structure frame (N, CA, C, O) |
| h_t | [B, L, 128] | Per-residue spatial encoding |
| z_t | [B, L, 128] | History-fused representation (from temporal encoder) |
| q_t | [B, L, 8] | Stage B dynamics variables (per-residue) |
| s_t | [B, L, 16] | Per-residue bath state |
| G | [B, 8] | Global state = masked_mean(q_t) |
| c | [B, L, D_c] | Static protein/environment condition |
| tau | [B] | Prediction horizon (ps) |
| T | [B] | Simulation temperature (K) |

## 2. Encoding Pipeline

```
X_{t-L:t}  →  E_spatial  →  h_{t-L:t}  →  (optional E_temporal)  →  H_last
                                              ↓
                                         project_to_q  →  q_t

batch["context_coords"]  [B, T_ctx, L, 4, 3]
    ↓ StageBLASP.forward_spatial()
H                         [B, T_ctx, L, 128]
    ↓ [:, -1]
H_last                    [B, L, 128]
    ↓ project_to_q()
q_t                       [B, L, 8]
    ↓ compute_g()
G                         [B, 8]
```

## 3. Dynamics Interface

```python
# Forces and diffusion
forces    = model.compute_energy_and_forces(q, ca_dists, mask)  # [B, L, 8]
diffusion = model.compute_per_residue_diffusion(q, G)           # [B, L]

# Bath (initialized once from H_last)
s_0       = model.initialize_bath(H_last, mask)                 # [B, L, 16]
gamma, C  = model.compute_bath_params(q, mask)                  # [B, L, 16], [B, L, 8, 16]

# Integration step
q_next, s_next = model.step(q, s, ca_distances=ca_dists, 
                              residue_mask=mask, dt_ps=100.0)
```

## 4. Rollout Interface (M02-compatible)

```python
# Initialize once
state = {"Q": q_t, "S": s_0, "step": 0, "h_frozen": H_last}
ca_dists = torch.cdist(CA, CA)  # frozen from anchor frame

# Roll forward (no future data access)
for step in range(n_steps):
    state["Q"], state["S"] = model.step(
        state["Q"], state["S"], 
        ca_distances=ca_dists,
        residue_mask=mask,
        dt_ps=dt, stochastic=True
    )
    state["step"] += 1
```

## 5. Decoding Interface

```python
# StructureDecoder
outputs = decoder(aatype, H_last, q_t, R_anchor, t_anchor, mask)
# → atom37_coords [B, L, 37, 3], atom37_mask [B, L, 37]
# → backbone_coords [B, L, 4, 3], torsion_logits [B, L, 7, 36]
```

## 6. Causality Guarantee

- `H_last` computed ONLY from context window `X_{t-L:t}`
- `q_t` derived ONLY from `H_last` (no future leakage)
- `ca_distances` frozen from last context frame
- Bath initialized once per rollout; never refreshed from future
- Decoder uses only frozen context; never sees target coordinates
