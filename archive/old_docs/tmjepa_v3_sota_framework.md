# Thermodynamic Mori-JEPA V3: SOTA Framework and Evaluation Methodology

> **Last updated**: 2026-09-03
> **Status**: Experimental candidate; Mori memory has not passed the promotion gates
> **Architecture**: Temperature-explicit free energy with heat capacity, Eyring-Einstein fluctuation-dissipation clock, generalized bath generator (Γ = S + A), continuous-discrete Bayesian observer, coupled Van Loan transition

---

## 1. Executive Summary

Thermodynamic Mori-JEPA (TM-JEPA) V3 is a self-supervised learning framework for protein molecular dynamics that models backbone trajectories as a **thermodynamic generalized Langevin equation (GLE)** with **explicit temperature dependence**, **colored noise from a generalized bath generator**, and **Bayesian latent state estimation**.

The framework is built on four physical principles:

1. **Temperature-explicit free energy**: The equilibrium measure is a Boltzmann distribution with learned heat capacity, enabling quantitative prediction across temperatures (320–450 K on mdCATH).
2. **Fluctuation-dissipation theorem**: Diffusion, coupling, and relaxation rates share a common Eyring-Einstein temperature clock, ensuring thermodynamic consistency.
3. **Generalized Mori bath**: The hidden memory variables evolve under a generator Γ = **S** + **A**, where **S** is positive semidefinite (dissipation) and **A** is skew-symmetric (rotation/oscillation). This captures damped oscillatory modes — the key generalization over the diagonal OU bath.
4. **Continuous-discrete Bayesian observer**: The Kalman filter operates on latent auxiliary variables and observation noise simultaneously, separating frame-level measurement noise from true dynamical innovations.

The implementation provides explicit promotion gates, but the retained validation
results do not promote Mori memory. See `tmjepa_v3_results_2026-09-03.md` and
`tmjepa_v3_logic_audit_2026-09-04.md` for the current evidence and known defects.

---

## 2. Architecture Overview

### 2.1 Data Flow

```
Raw coordinates (backbone CA, N, C, O)
  │
  ▼
FixedBackboneCollectiveVariables (16D SE(3)-invariant CVs)
  │
  ▼
Standardize (cv_mean, cv_scale)
  │
  ├───► ProteinConditionEncoder (sequence embedding)
  │
  ▼
ThermodynamicMoriDynamicsV3
  ├── HeatCapacityFreeEnergy (temperature-explicit F)
  └── EyringEinsteinKinetics (diffusion, coupling, relaxation, bath generator)
       │
       ▼
  Kalman filter (filter / _filter_with_observation_noise)
       │
       ▼
  Rollout (SDE integration with Van Loan joint covariance)
       │
       ▼
  Energy score, path NLL, innovation whiteness, etc.
```

### 2.2 Collective Variables (CVs)

The `FixedBackboneCollectiveVariables` module (`protein_jepa_frame/models/thermodynamic.py`) computes 16 differentiable, SE(3)-invariant descriptors from backbone atom coordinates:

| Component | Description |
|-----------|-------------|
| Virtual Cα distances | 4 inter-segment distances |
| Backbone dihedral angles (sin/cos) | 8 features from N–Cα–C–N dihedrals |
| Radius of gyration | 1 log-scale feature |
| End-to-end distance | 1 log-scale feature |
| Contact features | 2 pairwise contacts with sigmoid cutoff |

These are **standardized** using per-dimension mean and scale buffers computed from the training data.

### 2.3 ProteinConditionEncoder

A graph neural network that processes sequence features (aatype, residue index, chain ID, mask) into a per-protein condition vector of dimension `condition_dim` (default 64). The condition vector modulates the free energy surface, diffusion tensor, coupling vectors, and relaxation rates.

---

## 3. ThermodynamicMoriDynamicsV3

### 3.1 Free Energy: `HeatCapacityFreeEnergy`

**File**: `protein_jepa_frame/models/thermodynamic_v3.py`, class `HeatCapacityFreeEnergy`

The free energy is modeled as a **temperature-dependent learned potential**:

$$F(q, \theta, T) = \beta F(q, \theta, T) / k_B T$$

where:

$$\beta F(q, \theta, T) = \frac{U_0(q, \theta)}{k_B T} - S_0(q, \theta)/k_B + C(q, \theta) \left[ \frac{T - T_0}{T} - \ln\frac{T}{T_0} \right]$$

- $U_0$: reference enthalpy at $T_0 = 348$ K (learned MLP)
- $S_0$: reference entropy at $T_0$ (learned MLP)
- $C$: heat capacity in units of $k_B$ (learned MLP, softplus+tanh-activated)
- The gradient $-\nabla F$ gives the thermodynamic force (drift)

**Key equations**:
- Free energy: $F = U_0 - T \cdot S_0 + C \cdot k_B T \left[ \frac{T - T_0}{T} - \ln\frac{T}{T_0} \right]$
- Entropy: $S = S_0 + C \cdot k_B \ln(T/T_0)$
- Heat capacity: $C$ (constant with respect to $T$)
- Energy gradient: $\nabla F = \nabla U_0 - T \cdot \nabla S_0 + \nabla C \cdot k_B T \left[ \frac{T - T_0}{T} - \ln\frac{T}{T_0} \right]$

### 3.2 Eyring-Einstein Kinetics: `EyringEinsteinKinetics`

**File**: `protein_jepa_frame/models/thermodynamic_v3.py`, class `EyringEinsteinKinetics`

This module computes all kinetic coefficients from the condition embedding and temperature:

#### Diffusion Tensor

Low-rank (rank `diffusion_rank`) PSD diffusion:

$$D(q, \theta, T) = D_0 \cdot \exp\left(-\frac{E_a}{k_B T}\right) \cdot \sum_{r=1}^{R} \sigma_r(q, \theta)^2 \cdot v_r(q, \theta) \otimes v_r(q, \theta)$$

- $D_0$: `diffusion_scale` (learnable prefactor)
- $E_a$: `activation_energy_raw` (learnable per-mode activation energy)
- $\sigma_r$: learned mode amplitudes (from `mobility_head`)
- $v_r$: learned mode directions (from `mobility_head`)
- The **exact Itô divergence** is computed analytically: $\nabla \cdot D = \sum_r \nabla \cdot (\sigma_r^2 v_r \otimes v_r)$

#### Coupling Vectors

$$C(q, \theta, T) = C_0 \cdot \exp\left(-\frac{E_a^{(m)}}{k_B T}\right) \cdot \sum_{a=1}^{A} c_a(q, \theta) \cdot u_a(q, \theta) \otimes e_a$$

- $C_0$: `coupling_scale`
- $E_a^{(m)}$: memory activation energy (separate from diffusion)
- $c_a$: learned coupling amplitudes
- $u_a$: learned coupling directions
- $e_a$: standard basis for auxiliary dimension $A$

#### Generalized Bath Generator

The key architectural innovation:

$$\Gamma = \mathbf{S} + \mathbf{A}$$

where:
- $\mathbf{S} = \text{diag}(\lambda_1, \dots, \lambda_A)$ is the **dissipation** matrix (diagonal relaxation rates)
- $\mathbf{A}$ is a **skew-symmetric rotation** matrix learned as `memory_rotation_raw`

The rotation matrix captures **damped oscillatory/rotational hidden modes** — the diagonal OU bath ($\Gamma = \mathbf{S}$) is a special case when $\mathbf{A} = 0$, which is insufficient for protein dynamics (as confirmed by V3 residual analysis).

**When `coupled_bath_enabled=False`**: Rotation is forced to zero, collapsing to diagonal OU. This path is **deprecated** — the diagonal OU cannot capture the antisymmetric residual structure identified in the V3 analysis.

#### Eyring-Einstein Clock

All rates (diffusion, coupling, relaxation) share the same temperature clock:

$$k(T) = k(T_0) \cdot \exp\left(-\frac{E_a}{k_B}\left(\frac{1}{T} - \frac{1}{T_0}\right)\right)$$

Each rate has a learnable activation energy. The memory clock is independent of the diffusion clock, allowing different temperature sensitivity for memory vs. diffusive modes.

### 3.3 Local Transition: `_local_transition`

**File**: `protein_jepa_frame/models/thermodynamic_v3.py`, lines 541–629

The local transition dispatches based on `coupled_bath_enabled`:

#### Coupled Bath Path (default, SOTA)

Dispatches to `_coupled_bath_transition` (line 561–578). This is the **only active path** when `coupled_bath_enabled=True` (the new default).

#### Diagonal OU Path (deprecated)

Dispatches to `_bath_coefficients` (line 580–628). Analytic closed-form for diagonal OU. **Never called** when `coupled_bath_enabled=True`.

### 3.4 Coupled Bath Transition: `_coupled_bath_transition`

**File**: `protein_jepa_frame/models/thermodynamic_v3.py`, lines 481–539

This implements the **Van Loan matrix exponential** for the exact affine-Gaussian transition of the joint $(q, a)$ system:

$$\begin{pmatrix} q \\ a \end{pmatrix}_{t+dt} = \exp\left(\begin{pmatrix} 0 & C \\ 0 & -\Gamma \end{pmatrix} dt\right) \begin{pmatrix} q \\ a \end{pmatrix}_t + \text{affine drift} + \text{noise}$$

The Van Loan construction augments the state to $2 \times \text{dim} + 1$ and computes a single matrix exponential that simultaneously yields:
- The transition matrix for $(q, a)$
- The integrated process covariance
- The affine drift offset

The block structure:

```
drift_matrix = [[0,     C  ],
                [0,   -Γ   ]]

noise_rate = [[2D,    0  ],
              [0,   2S   ]]
```

The Van Loan matrix is:

$$\begin{pmatrix} -M & N & d \\ 0 & M^T & 0 \\ 0 & 0 & 0 \end{pmatrix}$$

where $M = \text{drift\_matrix}$, $N = \text{noise\_rate}$, $d = \text{affine\_drift}$.

### 3.5 Kalman Filter: `filter` / `_filter_with_observation_noise`

**File**: `protein_jepa_frame/models/thermodynamic_v3.py`, lines 631–917

The filter dispatches based on `observation_noise_enabled`:

#### With Observation Noise (default, SOTA)

`_filter_with_observation_noise` (lines 753–917): An **augmented-state Kalman filter** that jointly estimates:
- Auxiliary (memory) variables $a_t$
- Observation error $e_t$ (the difference between true latent $q_t$ and observed sequence $x_t$)

The state is $[a_t, e_t]$ with dimension $A + Q$. The observation model is:

$$x_{t+1} - x_t = -D\nabla F \cdot dt + C \cdot a_t \cdot dt - e_t + \text{noise}$$

The filter tracks:
- `mean_sequence`: posterior mean of auxiliary variables
- `covariance_sequence`: posterior covariance of auxiliary variables
- `observation_error_mean_sequence`: posterior mean of observation error
- `observation_error_covariance_sequence`: posterior covariance of observation error
- `innovation`: whitened residuals (standardized by Cholesky factor)
- `path_nll`: per-step negative log-likelihood

#### Without Observation Noise (deprecated)

`filter` (lines 631–751): A standard Kalman filter on auxiliary variables only. **Never called** when `observation_noise_enabled=True`.

### 3.6 Rollout: `rollout`

**File**: `protein_jepa_frame/models/thermodynamic_v3.py`, lines 919–1018

Stochastic SDE integration for forecast:

1. Draw initial ensemble of $a$ from posterior $N(\mu_a, \Sigma_a)$
2. For each step up to `max_rollout_steps`:
   - Compute `_local_transition` for each ensemble member
   - Form joint covariance matrix for $(q, a)$
   - Cholesky sample joint noise
   - Update $q$ and $a$ via affine-Gaussian transition
3. Return ensemble of $q$ trajectories and $a$ trajectories

The rollout uses **adaptive step size** (minimum of `base_step_ps` and remaining time), ensuring accurate integration even for non-uniform time grids.

---

## 4. Training Objectives

### 4.1 Loss Terms

**File**: `protein_jepa_frame/models/jepa.py`, function `tmjepa_terms` (line 3212)

The training loss is a weighted sum of objectives:

| Weight Config Key | Default | Description |
|-------------------|---------|-------------|
| `tmjepa_energy_score_weight` | 0.0 | Energy score of ensemble forecast vs. target |
| `tmjepa_transition_nll_weight` | 0.0 | Gaussian NLL of ensemble mean prediction |
| `tmjepa_path_nll_weight` | 1.0 | Per-step path NLL from Kalman filter |
| `tmjepa_equilibrium_score_weight` | 0.0 | Denoising score matching loss |
| `tmjepa_innovation_mean_weight` | 1.0 | Innovation mean zero constraint |
| `tmjepa_innovation_covariance_weight` | 1.0 | Innovation covariance = identity constraint |
| `tmjepa_innovation_autocorrelation_weight` | 1.0 | Innovation lag-1 whiteness constraint |
| `tmjepa_memory_intervention_weight` | 0.0 | Memory filter advantage |
| `tmjepa_auxiliary_kl_weight` | 0.0 | KL divergence on auxiliary posterior |
| `tmjepa_auxiliary_transition_kl_weight` | 0.0 | KL on auxiliary transition |

### 4.2 Innovation Whiteness Regularization

The innovation (whitened residual from the Kalman filter) should be **spatiotemporally white** if the model is correctly specified:

- **Mean zero**: $\mathbb{E}[z_t] = 0$ — enforced by `innovation_mean_loss`
- **Covariance identity**: $\mathbb{E}[z_t z_t^T] = I$ — enforced by `innovation_covariance_loss` (separate diagonal and off-diagonal penalties)
- **Lag-1 uncorrelated**: $\mathbb{E}[z_t z_{t-1}^T] = 0$ — enforced by `innovation_autocorrelation_loss`

### 4.3 Energy Score

For an ensemble forecast $\{q^{(k)}_T\}_{k=1}^K$:

$$\text{ES} = \frac{1}{K} \sum_k \|q^{(k)}_T - q_T^*\| - \frac{1}{2K^2} \sum_{k,l} \|q^{(k)}_T - q^{(l)}_T\|$$

The **energy skill** is $1 - \text{ES} / \|\text{persistence}\|$.

### 4.4 Path NLL

The sum of per-step negative log-likelihoods from the Kalman filter, normalized by CV dimension:

$$\text{NLL} = \frac{1}{T \cdot D} \sum_t \left[ \frac{1}{2} \|z_t\|^2 + \frac{1}{2} \log|\Sigma_t| + \frac{D}{2} \log(2\pi) \right]$$

---

## 5. Evaluation Protocol

### 5.1 Promotion Gates

**File**: `protein_jepa_frame/scripts/evaluate_tmjepa_v3.py`, function `evaluate`

All eight implemented gates must pass for promotion. Two of these gates require
complete protein coverage and a complete temperature-by-horizon grid; they are
coverage guards, not claims of model quality.

| Gate | Description | Required Condition |
|------|-------------|-------------------|
| `positive_skill_every_temperature_and_horizon` | Energy skill > 0 | All temperature × horizon bins |
| `coverage_calibrated_every_temperature` | 90% coverage within 0.85–0.95 | All temperature bins |
| `covariance_closed_every_temperature` | Innovation covariance RMSE < 0.15 | All temperature bins |
| `multi_lag_white_every_temperature` | Lagged correlation RMS < 0.05 | Lags 1, 2, 4, 8 |
| `diffusion_psd` | Diffusion PSD | Min eigenvalue > 0 |
| `mori_filter_predictive` | Memory filter advantage > 0.005 | When memory enabled |

### 5.2 Evaluation Metrics

**Horizon-level**:
- `energy_score`: Ensemble forecast error
- `persistence_energy_score`: Persistence forecast error
- `energy_skill`: 1 - ES / persistence
- `memory_reset_energy_score`: Error without memory initialization
- `memory_filter_advantage`: Reset - filtered energy score
- `coverage_90`: Fraction of targets within 1.645σ of ensemble
- `numerical_ck_energy_distance`: Cramér–von Mises energy distance

**Temperature-resolved**:
- `closure`: Innovation statistics (mean, covariance, lagged covariance, path NLL)
- `horizons`: Per-horizon energy skill and coverage at each temperature

**Physics**:
- `minimum_diffusion_eigenvalue`: PSD check
- `mean_activation_energy_kcal_per_mol`: Eyring-Einstein clock
- `mean_memory_activation_energy_kcal_per_mol`: Memory clock
- `mean_absolute_ito_divergence_per_ps`: Itô consistency
- `mean_integrated_memory_to_diffusion_ratio`: Memory kernel integral
- `mean_memory_drift_rms_per_ps`: Auxiliary drift magnitude
- `mean_bath_information_fraction`: Posterior information from Kalman filter
- `mean_auxiliary_rotation_rate_per_ps`: Bath rotation rate
- `maximum_fdt_generator_residual`: FDT generator residual (should be near zero)
- `mean_observation_noise_variance`: Estimated frame noise

### 5.3 Memory Intervention Test

When `memory_enabled=True` and `memory_intervention_enabled=True`, the evaluation compares:

1. **Filtered forecast**: Rollout initialized from Kalman-filtered auxiliary posterior
2. **Reset forecast**: Rollout initialized from zero-mean, identity-covariance auxiliary state

The **memory filter advantage** is the gap between reset and filtered energy scores. The **memory intervention loss** is $\text{ReLU}(\text{filtered} - \text{reset} + \text{margin})$.

---

## 6. Key Configuration Flags

### 6.1 Model Config (`ModelConfig`)

| Flag | Default | Description |
|------|---------|-------------|
| `tmjepa_enabled` | `False` | Enable TM-JEPA architecture |
| `tmjepa_version` | `2` | must be `3` for V3 |
| `tmjepa_coupled_bath_enabled` | **`True`** | Van Loan coupled bath with rotation (was `False` — **changed in 2026-09-03 fix**) |
| `tmjepa_observation_noise_enabled` | **`True`** | Observation-noise-aware Kalman filter (was `False` — **changed in 2026-09-03 fix**) |
| `tmjepa_memory_enabled` | `True` | Enable auxiliary memory variables |
| `tmjepa_auxiliary_dim` | `8` | Dimension of bath (number of Mori modes) |
| `tmjepa_diffusion_rank` | `4` | Rank of PSD diffusion tensor |
| `tmjepa_memory_times_ps` | `(1000, ...)` | Per-mode relaxation times |
| `tmjepa_observation_noise_scale` | `0.1` | Initial observation noise std |
| `tmjepa_freeze_potential` | `False` | Freeze free energy and condition encoder |
| `tmjepa_freeze_kinetics` | `False` | Freeze kinetics module |
| `tmjepa_freeze_diffusion` | `False` | Freeze diffusion tensor only |
| `tmjepa_freeze_memory` | `False` | Freeze memory module |

### 6.2 Optimization Config (`OptimizationConfig`)

| Flag | Default | Description |
|------|---------|-------------|
| `lr` | `3e-4` | Learning rate |
| `encoder_lr_scale` | `0.2` | Encoder LR multiplier |
| `tmjepa_path_nll_weight` | `1.0` | Path NLL weight |
| `tmjepa_innovation_mean_weight` | `1.0` | Innovation mean zero |
| `tmjepa_innovation_covariance_weight` | `1.0` | Innovation covariance identity |
| `tmjepa_innovation_autocorrelation_weight` | `1.0` | Innovation lag-1 whiteness |

### 6.3 Data Config (`DataConfig`)

| Flag | Default | Description |
|------|---------|-------------|
| `num_frames` | `64` | Context window length |
| `frame_stride` | `1` | Frame skip |
| `paired_horizons_frames` | `(1, 2, 4, 8, 16)` | Forecast horizons |
| `sampling_mode` | `hierarchical` | Temperature-balanced sampling |

---

## 7. Code Architecture

### 7.1 Module Structure

```
protein_jepa_frame/
├── models/
│   ├── jepa.py                          # ProteinJEPA, loss functions
│   ├── thermodynamic.py                 # V2 dynamics, FixedBackboneCVs
│   └── thermodynamic_v3.py              # V3 dynamics (core SOTA model)
├── data/
│   ├── dataset.py                       # Dataset builders
│   ├── collate.py                       # Collation functions
│   └── sampler.py                       # HierarchicalClipSampler
├── training/
│   └── train.py                         # Training loop
├── scripts/
│   ├── evaluate_tmjepa.py               # V2 evaluation
│   └── evaluate_tmjepa_v3.py            # V3 evaluation (SOTA)
├── config.py                            # Dataclass configs
└── configs/
    └── phase7_tmjepa_v3_mori_pilot.yaml  # V3 Mori training config
```

### 7.2 Key Code Paths

#### Forward Pass (V3)

```
ProteinJEPA.forward_tmjepa()
  └─► ThermodynamicMoriJEPAV3.forward()
        ├─► FixedBackboneCollectiveVariables()     # CV computation
        ├─► ProteinConditionEncoder()               # Condition embedding
        ├─► ThermodynamicMoriDynamicsV3.filter()
        │     └─► _filter_with_observation_noise()  # Kalman filter (default)
        │           └─► _local_transition()          # Per-step transition
        │                 └─► _coupled_bath_transition()  # Van Loan (default)
        └─► ThermodynamicMoriDynamicsV3.rollout()
              └─► _local_transition()               # Per-step rollout
                    └─► _coupled_bath_transition()   # Van Loan (default)
```

#### Training Loss

```
train_paired()
  └─► tmjepa_terms()                              # Distributional objectives
        ├─► energy_score, transition_nll, path_nll
        ├─► innovation_mean_loss, covariance_loss, autocorrelation_loss
        └─► memory_intervention_loss, auxiliary_kl_loss
```

---

## 8. Defects Found and Fixed (2026-09-03)

### 8.1 Default `coupled_bath_enabled=False`

**Problem**: The diagonal OU bath (`_bath_coefficients` path) could not capture rotational/oscillatory hidden modes. The V3 residual analysis showed clear antisymmetric structure in the innovation residual, which requires a generalized bath generator $\Gamma = \mathbf{S} + \mathbf{A}$.

**Fix**: Changed default from `False` to `True` in `config.py` line 183.

**Impact**: `_local_transition` now dispatches to `_coupled_bath_transition` (Van Loan matrix exponential) by default. The `_bath_coefficients` (diagonal OU) path is never called when `coupled_bath_enabled=True`.

### 8.2 Default `observation_noise_enabled=False`

**Problem**: The Kalman filter without observation noise (`filter`) could not separate frame-level measurement noise from true dynamics. The 1 ns observation interval introduces significant frame noise from the CV discretization.

**Fix**: Changed default from `False` to `True` in `config.py` line 184.

**Impact**: `filter` now dispatches to `_filter_with_observation_noise` by default, which jointly estimates auxiliary variables and observation error in an augmented-state Kalman filter.

### 8.3 Joint Training Essential

**Finding**: The V3 evaluation protocol requires **joint optimization** of all kinetic parameters. Freezing diffusion while training the coupled bath is invalid because the coupled bath adds both colored drift and one-step variance — the instantaneous diffusion must be re-estimated jointly.

**Current practice**: The V3 Mori config (`phase7_tmjepa_v3_mori_pilot.yaml`) correctly sets `tmjepa_freeze_diffusion: false` and `tmjepa_freeze_potential: true` (potential is pretrained). The PMF config freezes kinetics for initial potential learning, then the Mori config joint trains everything.

---

## 9. Training Protocol

### 9.1 Phase 7 Training Pipeline

1. **PMF pretraining**: Train `HeatCapacityFreeEnergy` with `tmjepa_freeze_kinetics: true`. Only the potential and condition encoder are trained. Uses `tmjepa_equilibrium_score_weight: 0.001` (denoising score matching).

2. **Mori joint training**: Initialize from PMF checkpoint. Train all components with `tmjepa_freeze_potential: true` (freeze the well-calibrated free energy surface). Optimize innovation whiteness, path NLL, and energy score.

### 9.2 Recommended Config (SOTA)

```yaml
model:
  tmjepa_enabled: true
  tmjepa_version: 3
  tmjepa_coupled_bath_enabled: true        # Generalized bath Γ = S + A
  tmjepa_observation_noise_enabled: true   # Augmented Kalman filter
  tmjepa_memory_enabled: true              # Auxiliary memory variables
  tmjepa_auxiliary_dim: 8                 # 8 Mori modes
  tmjepa_diffusion_rank: 4                # Rank-4 diffusion
  tmjepa_memory_times_ps: [1074, 1075, 1099, 1165, 14060, 14786, 15618, 16000]
  tmjepa_freeze_potential: true           # Freeze pretrained free energy
  tmjepa_freeze_diffusion: false          # Joint train diffusion
  tmjepa_freeze_memory: false             # Joint train memory

optimization:
  tmjepa_path_nll_weight: 1.0             # Per-step path NLL
  tmjepa_innovation_mean_weight: 1.0      # Innovation mean zero
  tmjepa_innovation_covariance_weight: 1.0 # Innovation covariance identity
  tmjepa_innovation_autocorrelation_weight: 1.0  # Innovation whiteness
  tmjepa_energy_score_weight: 0.2         # Ensemble forecast quality
```

---

## 10. Extending and Improving the Framework

### 10.1 Generalized Bath Beyond Linear

The current bath generator $\Gamma = \mathbf{S} + \mathbf{A}$ is linear. A natural extension is **state-dependent rotation** $\mathbf{A}(q, \theta)$, which would allow the oscillatory modes of the bath to depend on the protein conformation.

### 10.2 Multi-Scale Memory

Currently, each auxiliary variable has a fixed relaxation time. An extension would allow **hierarchical memory** where fast modes modulate slow modes (e.g., through a learned coupling between auxiliary variables beyond the skew-symmetric rotation).

### 10.3 Learned CVs

The `tmjepa_learned_cv_dim` flag is reserved for learned CVs but currently raises `ValueError` in V3. The extensibility layer is designed so that learned CVs can be added via the `v53_multi_scale_history_enabled` and related flags, though this path is not yet validated.

### 10.4 Evaluation Extensions

The evaluation protocol (`evaluate_tmjepa_v3.py`) can be extended with:
- **Equilibrium sampling**: Compute the stationary distribution of the learned SDE via long rollouts
- **Transition path analysis**: Compute committor probabilities and transition paths
- **Spectral analysis**: Decompose the auxiliary generator into modes with relaxation times and oscillation frequencies

---

## 11. References

- **Codebase**: `/liuyutian/protein_jepa_frame/`
- **V3 model**: `protein_jepa_frame/models/thermodynamic_v3.py`
- **Evaluation**: `protein_jepa_frame/scripts/evaluate_tmjepa_v3.py`
- **Config**: `protein_jepa_frame/configs/phase7_tmjepa_v3_mori_pilot.yaml`
- **Results**: `protein_jepa_frame/docs/tmjepa_v3_results_2026-09-03.md`
- **Proposal**: `protein_jepa_frame/docs/first_principles_thermodynamic_mori_jepa_proposal_2026-09-02.md`
- **Test suite**: `protein_jepa_frame/tests/test_tmjepa_v3.py` (8 tests, all passing)

---

## 12. Appendix: Code Path Verification

The following verification confirms that the corrected defaults route through the correct code paths:

```python
# Verify: coupled_bath_enabled=True → _coupled_bath_transition is called
model = ThermodynamicMoriDynamicsV3(coupled_bath_enabled=True, ...)
result = model._local_transition(q, aux, cond, temp, dt, create_graph=False)
assert 'history_matrix' in result  # from _coupled_bath_transition
assert 'auxiliary_transition_matrix' in result  # from _coupled_bath_transition

# Verify: _bath_coefficients is NEVER called
# (proven by monkey-patching both methods — only the coupled path fires)

# Verify: observation_noise_enabled=True → _filter_with_observation_noise
filt = model.filter(seq, mask, time_ps, cond, temp)
assert 'observation_error_mean_sequence' in filt  # from _filter_with_observation_noise

# Verify: non-zero rotation (generalized bath)
kin = model.kinetics(q, cond, temp)
assert not torch.allclose(kin['auxiliary_rotation'], 0)  # Γ = S + A with A ≠ 0
```

All 8 unit tests pass. The `_bath_coefficients` (diagonal OU) path is never reached when `coupled_bath_enabled=True`.
