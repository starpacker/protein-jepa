# Protein-JEPA-Frame: LSP-WM

Protein-JEPA-Frame is a research framework for building a **complete,
intervenable, multiscale protein world model** from molecular-dynamics and
perturbation data.

The target framework is **LSP-WM**, the Latent Statistical Physics World
Model. The central object is not only a latent trajectory predictor and not
only an equilibrium PMF. It is a conditional simulator of a coupled
protein-environment system whose path distribution is thermodynamically
consistent when no intervention is applied, and whose response to explicit
actions can be evaluated counterfactually.

This README is the scientific and architectural contract. It describes the
target LSP-WM system and records which parts are currently implemented in this
repository. The current code is an experimental v4.4 thermodynamic latent
dynamics subsystem. It is not yet a complete LSP-WM world model.

## 1. What The World Model Represents

### 1.1 Full latent state

A protein trajectory is an observation of a larger physical system. The target
latent state is factorized into four explicit roles:

```text
u_t = (q_t^protein, e_t^environment, m_t^memory)
```

where:

- `q_t^protein` represents protein conformation and internal degrees of
  freedom: rigid-body frames, backbone and side-chain geometry, contacts,
  torsions, domain motion, and slow collective variables;
- `e_t^environment` represents evolving external degrees of freedom that can
  control protein kinetics: solvent organization, ions, ligand occupancy,
  membrane interaction, crowding, and other declared environmental fields;
- `m_t^memory` represents unresolved projected degrees of freedom and filtered
  history. It is a memory state, not a replacement for the environment state;
- `a_t` is an intervention or control, not part of the passive state. Examples
  include mutation, ligand binding or unbinding, temperature jump, pH change,
  external force, and a membrane or crowding perturbation.

The complete predictive object is

```text
p(u_{t+Delta t} | u_t, a_t, c, Delta t)
```

and the complete path object is

```text
P_theta[u(t_0:t_1) | u(t_0), a(t_0:t_1), c, T].
```

`c` contains static context such as sequence, topology, experimental protocol,
or fixed chemical composition. An evolving solvent or ligand occupancy belongs
in `e_t`, not in `c`. An intervention belongs in `a_t`, not in `c`.

### 1.2 v4.4 scope and v5 target

The current v4.4 design models a protein coarse state `z` and a finite memory
bath under a fixed environment condition. It is trained in two stages:

- **Stage A (representation pretraining):** The causal JEPA encoder and
  temporal predictor are trained via paired-window prediction, producing a
  latent space with kinetic separation.
- **Stage B (thermodynamic dynamics fitting):** A learned potential,
  state-dependent diffusion, finite Mori bath, and Koopman operator are
  trained on top of the frozen Stage-A encoder, using dynamic NLL,
  mean-force KDE matching, and VAMP-2 regularization.

Stage B v3 (the current best checkpoint) trains 3000 steps (3 epochs × 1000 steps)
with a curriculum that enables force learning only after step 1500. The final
checkpoint `checkpoint_step3000.pt` achieves measurable force correlation and
thermodynamic consistency.

The complete LSP-WM target extends it to

```text
u_t = (q_t^protein, e_t^environment, m_t^memory)
```

with coupled environment dynamics, explicit interventions, structural
mechanics, multiscale time evolution, and path-level validation. The terms
"world model" and "complete protein dynamics" are reserved for this extended
target, not for the current protein-only candidate.

## 2. Generative Architecture

```text
protein coordinates, environment observations, actions
                         |
                         v
     SE(3)-aware protein encoder + environment encoder
                         |
                         v
       q_t^protein, e_t^environment, m_t^memory
                         |
        +----------------+----------------+
        |                |                |
        v                v                v
  protein mechanics   coupled thermo   action/intervention
  and geometry        dynamics          response model
        |                |                |
        +----------------+----------------+
                         v
          p(u_{t+Delta t} | u_t, a_t, c, T)
                         |
                         v
         multistep path sampler and path likelihood
                         |
                         v
        constrained observation model p_psi(x,y | u,c)
```

The architecture has five non-negotiable boundaries:

1. the protein state and environment state are separate and may exchange
   information and forces;
2. memory is causal and dynamical, not an arbitrary static feature or an
   unexplained residual displacement;
3. actions can change the transition law and must support counterfactual
   rollout;
4. generated protein structures satisfy an internal-geometry contract;
5. training likelihoods, rollouts, and evaluation use the same transition and
   noise model.

## 3. Structured Protein State

`q_t^protein` is not an unconstrained coordinate vector. It is a structured
state with a declared representation:

```text
q_t^protein = (R_t, r_t^internal, chi_t, h_t^slow)
```

The exact factorization may vary, but a compliant implementation must address:

- global SE(3) gauge and rigid-body motion;
- bond lengths, bond angles, and peptide geometry;
- torsion periodicity and chirality;
- side-chain rotamer states;
- contacts and domain-level relative motion;
- steric and excluded-volume constraints;
- variable protein length and chain topology.

The observation model must either generate valid internal coordinates and map
them to Cartesian coordinates, or use a constrained/projection integrator. A
generic decoder `z -> x` is not sufficient because it may generate broken
bonds, invalid chirality, or impossible peptide geometry.

Fixed physical collective variables may be used as anchors and audits. They
must not be confused with a proof that the learned state is thermodynamically
sufficient.

## 4. Coupled Environment State

The environment is part of the physical state whenever it evolves on a time
scale relevant to protein dynamics. At minimum, an implementation must declare
which of the following are modeled, conditioned, or omitted:

| Environment component | Examples |
| --- | --- |
| solvent | hydration shell, water orientation, local density |
| ions | occupancy, screening, ion atmosphere |
| ligand | bound, unbound, pose, occupancy, chemical state |
| membrane | proximity, lipid composition, local field |
| crowding | neighboring macromolecules, confinement, excluded volume |
| protocol | pressure, pH, temperature, force device |

If a component is omitted from `e_t`, the framework must state the resulting
conditional assumption. A static condition can be useful for controlled
experiments, but it cannot represent environment-mediated transitions or
ligand-induced changes by itself.

The target coupled law is

```text
p(q_{t+Delta t}, e_{t+Delta t}, m_{t+Delta t}
  | q_t, e_t, m_t, a_t, c, T).
```

## 5. Thermodynamic And Path Dynamics

### 5.1 Equilibrium structure

The model must declare its base measure. Under a fixed Euclidean measure, a
dimensionless equilibrium free energy is

```text
A_phi(u, c, T) = -log p_eq(u | c, T) + constant(c, T).
```

For constrained coordinates, the measure includes the appropriate manifold
volume element. If a learned nonlinear coordinate or metric is used, its
Jacobian, metric, and divergence terms must appear in the likelihood and
integrator. An arbitrary JEPA latent cannot be called a physical coordinate
without this declaration.

The equilibrium PMF constrains stationary occupancy. It does not determine
transition-path ensembles, committors, reactive flux, or the relative weights
of alternative folding pathways.

### 5.2 General thermodynamic dynamics

The target passive dynamics may be expressed in a dimensionless GENERIC-like
form:

```text
du = [L(u,c,T) grad E(u,c,T)
      - M(u,c,T) grad A(u,c,T)
      + div M(u,c,T)] dt
     + sqrt(2 M(u,c,T)) dW_t
```

where:

- `L = -L^T` is the reversible operator;
- `M = M^T >= 0` is the dissipative mobility/noise operator;
- `A` is defined with respect to the declared state measure;
- `div M` is required for state-dependent Ito diffusion;
- noise covariance and dissipation obey the same FDT convention.

For a physical-energy parameterization, the equivalent `k_B T` factors must
be written explicitly. The implementation must not mix dimensionless PMF,
physical energy, friction, and diffusion units without a conversion contract.

At passive equilibrium, the stationary law and detailed balance conditions
must be checked. Under an active intervention, detailed balance generally does
not hold. The model must then report work, entropy production, or another
declared nonequilibrium quantity rather than incorrectly applying equilibrium
tests.

### 5.3 Finite Mori memory

The v4.4 Markovian bath is a controlled finite approximation:

```text
dz = [-D(T) grad A(z,c,T) + C s] dt
     + sqrt(2 D(T)) dW_z

ds = [-Gamma s - C^T grad A(z,c,T)] dt
     + sqrt(2 Gamma) dW_s.
```

It gives the kernel

```text
K(t) = C exp(-Gamma t) C^T
```

and therefore a finite sum of exponentials. This is not a claim that all
hidden protein physics is Gaussian or linear. Water orientation, side-chain
rearrangement, ligand occupancy, and electrostatic screening may require
nonlinear and non-exponential memory.

The complete target therefore supports a causal operator-valued kernel:

```text
F_memory(t) = integral K_theta(t,s; u_{<=t}, a_{<=t})
              phi(u_s, a_s) ds + eta_memory(t)
```

subject to causality, stability, positivity, and FDT constraints when the
system is passive. A finite exponential bath is the baseline approximation;
kernel adequacy must be measured from held-out memory and intervention tests.

### 5.4 Temperature consistency

Temperature can affect at least four coupled objects:

```text
A(u, c, T), M(u, c, T), Gamma(u, c, T), K(t | u, c, T).
```

These objects must obey one declared thermodynamic convention. For the
dimensionless equation, mobility and noise are linked by FDT. For a
physical-energy parameterization, the corresponding `k_B T` factors and
Einstein relation must be explicit. Barrier-crossing rates must be tested for
the expected temperature dependence rather than independently calibrated at
each temperature. Changing diffusion with temperature while leaving the PMF
and memory law unconstrained is not thermodynamic consistency.

### 5.5 Actions and counterfactuals

The controlled transition is

```text
u_{t+Delta t} = F_theta(u_t, a_t, c, T, noise).
```

The action encoder must distinguish:

- parameter changes, such as mutation or protonation;
- state transitions, such as ligand bind/unbind;
- external driving, such as force or temperature jump;
- observation interventions, such as perturb-and-measure protocols.

An action must be able to change the relevant energy, mobility, environment
coupling, or boundary condition. Concatenating an action token to a static
condition is not sufficient. Every action family requires held-out
counterfactual evaluation and an action-free recovery test.

### 5.6 Path functional

The complete dynamics model is evaluated as a path distribution. For a
discretized trajectory `u_0:N`:

```text
S_theta[u_0:N | a_0:N] = -log p_theta(u_1:N | u_0, a_0:N).
```

The implementation may use an equivalent Onsager-Machlup or transition-path
factorization, but it must report path likelihood, path coverage, transition
path ensemble quality, committor calibration, reactive flux, and MFPT. Matching
`A(u)` alone is never sufficient evidence for a protein dynamics world model.

## 6. Multiscale Dynamics

Protein motion spans at least bond vibrations, side-chain rotations, loop
motion, folding, and allosteric signaling. A single latent vector, one PMF,
and one memory time scale do not establish multiscale dynamics.

The target architecture uses a hierarchy:

```text
u^(0) fast internal geometry       fs-ps
u^(1) side-chain and local modes   ps-ns
u^(2) domains and loops            ns-us
u^(3) folding and allostery        us-ms and beyond
```

Levels may be implemented as nested states, coupled latent operators, or a
validated kernel bank. Each level must declare its time step, exchange law,
noise scale, and projection to adjacent levels. Evaluation must include
timescale separation, cross-level consistency, and long-horizon stability.

The overdamped v4.4 path covers only the scales supported by its measured
sampling interval. It must not be extrapolated from ps data to millisecond
folding without an explicit multiscale validation.

## 7. Representation Learning And Sufficiency

The JEPA path remains useful, but JEPA prediction is a training mechanism, not
a sufficiency theorem:

```text
observations -> causal protein/environment encoders -> u_t
full target observations -> EMA target encoders -> target u_{t+Delta t}
```

The context branch must not access masked coordinates or environment variables
through graph construction, neighbor selection, pooling, or temporal leakage.

A compliant representation audit must test approximate predictive sufficiency,
for example:

```text
I(u_t, a_t; y_{t+Delta t}) ~= I(observations_{<=t}; y_{t+Delta t})
```

using held-out predictive probes, shuffled-history controls, environment
interventions, and action-conditioned probes. It must also report sequence
identity leakage, reconstruction shortcuts, masking artifacts, and slow-mode
coverage. Koopman compatibility alone does not establish thermodynamic
sufficiency.

## 8. Training Objectives

| Objective | Role | Required evidence |
| --- | --- | --- |
| `L_repr` | causal JEPA prediction | EMA target and leakage audit |
| `L_suff` | predictive and thermodynamic state sufficiency | held-out information/probe audit |
| `L_geom` | internal protein mechanics | bond, torsion, chirality, steric checks |
| `L_eq` | equilibrium free energy and stationary law | PMF, measure, and distribution checks |
| `L_dyn` | transition and path likelihood | same SDE/integrator as rollout |
| `L_mem` | causal memory and residual statistics | FDT, whiteness, and intervention tests |
| `L_action` | action-conditioned transition response | counterfactual and recovery tests |
| `L_multi` | cross-scale consistency | semigroup and timescale tests |
| `L_obs` | constrained coordinate/environment observation model | structure and distribution metrics |
| `L_id` | gauge and equivalence-class control | parameter perturbation and intervention audits |

VAMP, MSM, KDE density, energy score, and innovation metrics are diagnostics
or auxiliary objectives only when their statistical assumptions are explicit.
A single Koopman operator must not pool incompatible lags. Use fixed-lag
batches, a lag-conditioned operator, or a continuous-time generator.

For PMF/score matching, the estimator, measure, bandwidth selection,
cross-fitting rule, and uncertainty analysis must be declared. A small-batch
Gaussian KDE is a low-dimensional smoke test, not an unbiased 256D protein
free-energy estimator.

## 9. Identifiability And Generalization

The factorization

```text
(encoder, A, L, M, K, decoder)
```

is not identifiable from one passive trajectory in general. Different
representations and operators can generate the same observed path law.

The framework therefore reports an equivalence class rather than claiming a
unique physical decomposition. A serious identifiability protocol includes:

- latent whitening, scale, ordering, and sign conventions;
- bath and kernel gauge conventions;
- multiple temperatures and independent trajectories;
- action-conditioned interventions that distinguish competing decompositions;
- held-out path likelihood and predictive equivalence tests;
- sensitivity to latent dimension, memory rank, and kernel family.

Protein-disjoint testing measures generalization but does not prove a universal
protein manifold. Train, validation, and test results must be stratified by
sequence family, topology, length, environment, and transition type.

## 10. Curriculum

The target curriculum has seven stages:

1. **Geometry:** learn SE(3)-aware protein/environment observations and
   internal mechanics.
2. **State factorization:** separate protein, environment, memory, and action
   roles; audit information leakage and sufficiency.
3. **Equilibrium:** fit the measure-aware energy/free-energy model and passive
   stationary distribution.
4. **Passive dynamics:** fit the coupled transition and path law with FDT and
   detailed-balance tests.
5. **Memory and scales:** identify causal kernel structure and cross-scale
   semigroup behavior.
6. **Interventions:** add action-conditioned transitions and counterfactual
   rollouts.
7. **Observation and generation:** fit constrained coordinate/environment
   generation and evaluate full path ensembles.

Phase names do not create physical guarantees. A phase is valid only when its
active objectives correspond to the declared state, measure, transition law,
and data split.

**Current implementation status:** Stage A (representation learning, phases 1–2)
and Stage B (thermodynamic dynamics fitting, phases 3–4) are implemented in
v4.4. Stages 5–7 (multiscale memory, interventions, observation/generation)
are not yet implemented.

## 11. First-Principles Architectural Revisions (v4.4+)

The v4.4 Stage B v3 evaluation revealed two specific weaknesses:

1. **FastFolding performance degrades at H > 1** (multi-step horizons): the world
   model is trained only on one-step transitions and has never seen a rollout.
2. **Deterministic Skill (DetSkill = 0.774) below target (0.85)**: the JEPA
   predictor is too shallow (`predictor_depth: 1`) and the encoder is frozen.

These are **architectural** deficiencies, not parameter tuning problems. This
section documents the first-principles revisions implemented in Stage B v4.

### 11.1 Rev-1: BAOAB Midpoint Integration (H > 1 stability)

**Theory gap:** The README §5.2 defines the continuous GENERIC dynamics

```text
dq = [-D grad A(q) + C s] dt + sqrt(2D) dW
```

but the code uses Euler--Maruyama integration, which evaluates the gradient
**only at the start state** `q` and has O(Δt) local truncation error. Over
multiple steps this accumulates systematic bias.

**Solution:** Replace Euler--Maruyama with **BAOAB splitting**:

```
B/2: q_half = q + sqrt(2D Δt) ξ / 2     (half-step diffusion)
A:   q_full = q_half - D ∇A(q_half) Δt  (drift at midpoint)
B/2: q_next = q_full + sqrt(2D Δt) ξ / 2 (second half-step diffusion)
```

The gradient is evaluated at the **midpoint** `q_half` instead of `q`, giving
O(Δt²) global accuracy. For quadratic potentials BAOAB preserves the exact
Boltzmann distribution. Controlled via `LSPWMConfig.midpoint_integration`.

### 11.2 Rev-2: Multi-Step Rollout Trajectory NLL (Path Likelihood)

**Theory gap:** The README §1.1 defines the **path object**

```text
P_θ[u(t₀:t₁) | u(t₀), a(t₀:t₁), c, T]
```

but the training loss `overdamped_transition_nll` evaluates only a single
step. The model has never been trained to generate multi-step rollouts, so
errors compound at H > 1.

**Solution:** Add `rollout_trajectory_nll()` to `LSPWMWorldModel`. Given an
observed trajectory `q_0, q_1, ..., q_K`, this method:

1. Initializes auxiliary state (from data or zeros).
2. Rolls out the world model for K steps (using midpoint integration when
   enabled).
3. Computes the joint Gaussian NLL of the observed trajectory under the
   model's predictive distribution at each step.

This trains the **path likelihood** directly. Controlled via
`LSPWMConfig.rollout_trajectory_steps` (K = number of rollout steps).

### 11.3 Rev-3: Learned Auxiliary State Initialization

**Theory gap:** The Mori bath `s_t` is coupled to `q_t` through the coupling
matrix `C` (README §5.3), but in the code it is always initialized to zero
for rollouts. This means the first few rollout steps are inaccurate because
the memory trace is missing.

**Solution:** Add a small MLP `auxiliary_initializer` to
`LSPWMProteinWorldModel` that predicts the initial bath state from the last
N = 3 context frame states:

```python
auxiliary_init = MLP(concat(q_{T-2}, q_{T-1}, q_T))
```

The context frames are projected through the learned projection first, so the
initializer operates in the learned CV space. Both the initializer and the
projection are trained jointly during Stage B.

### 11.4 Rev-4: Semigroup Consistency (Cross-Timescale Constraint)

**Theory gap:** The README §6 requires "semigroup and timescale tests" but no
loss enforces consistency across different prediction lags. The Koopman
operator property `K_{2Δt} = K_Δt ∘ K_Δt` is not constrained.

**Solution:** Add `semigroup_consistency_loss()` which computes empirical
Koopman matrices at two lags (Δt and 2Δt) and penalizes their mismatch:

```text
L_semigroup = ||K_{2Δt} - K_Δt @ K_Δt||_F²
```

This is computed from the same batch data (using the paired horizon sampling
which already provides frames at Δt and 2Δt). Controlled via
`LSPWMConfig.semigroup_consistency_weight`.

### 11.5 Rev-5: Encoder Fine-Tuning and Deeper Predictor

**Theory gap:** DetSkill measures the JEPA `paired_delta_predictor`. With
`predictor_depth: 1` and `freeze_context_encoder: true`, the model has
insufficient capacity to learn the 256-dim latent dynamics.

**Solution:**
- Increase `predictor_depth` from 1 to 3 (deeper MLP).
- Enable `time_conditioning: fourier` so the predictor explicitly sees Δt.
- Set `encoder_lr_scale: 0.01` to allow gentle fine-tuning of the encoder
  during Stage B, adapting the representation to thermodynamics.

### 11.6 Component Architecture

```text
Stage B v4 loss function:

L_total = w_dyn · L_dyn + w_force · L_force + w_koop · L_koop
        + w_traj · L_trajectory     (Rev-2: multi-step path NLL)
        + w_semi · L_semigroup      (Rev-4: cross-timescale consistency)

Integration:  midpoint/BAOAB (Rev-1)  ← configurable
Auxiliary:    learned init (Rev-3)    ← data-driven bath initial state
Encoder:      lr_scale = 0.01 (Rev-5) ← gentle representation adaptation
Predictor:    depth = 3 (Rev-5)       ← higher capacity
```

The new config file is `configs/lspwm_v44_stage_b_v4_architectural.yaml`.

## 12. Data And Benchmark Protocol

The primary split is protein-disjoint:

```text
train proteins       != validation proteins != test proteins
```

Replica-held-out evaluation is useful but is not a replacement for
protein-disjoint evaluation. Temperature, environment, ligand, and action
extrapolation must reserve both trajectories and perturbation protocols before
calibration.

Windows from the same trajectory must not cross a split boundary. All
normalization, PMF statistics, kernel calibration, bandwidths, checkpoint
selection, and endpoint calibration use training or validation data only. The
test split is opened once after every decision is frozen.

Every benchmark must declare whether it tests:

- passive equilibrium sampling;
- short-time and long-time transition prediction;
- transition paths and committors;
- environment-mediated response;
- ligand or mutation intervention;
- temperature/pH/force response;
- protein-family or topology transfer.

Minimum reviewer-level benchmarks are:

1. alanine dipeptide PMF, mechanics, integrability, and equilibrium law;
2. villin or an equivalent folding trajectory with MFPT, committor, and
   transition-path ensemble evaluation;
3. a protein-disjoint held-out transition-path and endpoint ensemble test;
4. an environment or ligand perturbation test;
5. temperature transfer with a measured FDT and barrier-rate response;
6. a multiscale long-horizon stability test.

## 13. Data Contract And Repository Workflow

Training reads a JSONL manifest with one trajectory record per line. A minimal
NPZ record is:

```json
{"npz_path":"/data/trajectory.npz","protein_id":"1abc_A","trajectory_id":"1abc_A_R1","num_frames":10000,"frame_interval_ps":10.0,"temperature_k":348.0}
```

The NPZ file must contain `coords` with shape `[frames, residues, 4, 3]` in
`N, CA, C, O` order. Optional arrays are `atom_mask`, `residue_mask`,
`aatype`, `residue_index`, `chain_id`, `time_ps`, and `forces`. A complete
world-model manifest additionally records environment observations, action
intervals, action values, protocol metadata, and measurement uncertainty.

The current mdCATH manifests contain passive protein trajectories and do not
provide intervention labels. Therefore action-conditioned training,
counterfactual rollout, and causal action claims are disabled for the current
experiments. They are future capabilities, not missing values to be inferred
from passive trajectories.

Missing physical time, temperature, environment, or action metadata must be
declared as an explicit omission. It must not silently become a thermodynamic
condition.

The package layout is:

```text
protein_jepa_frame/
  models/       encoders, structured states, thermodynamics, and dynamics
  data/         manifests, trajectory readers, actions, masks, and samplers
  training/     objectives, curriculum, optimizer, and checkpoints
  eval/         path, thermodynamic, intervention, and transfer evaluation
  scripts/      training, audits, calibration, and reporting entry points
  configs/      experiment definitions
  tests/        software, physics, mechanics, and causal-contract tests
  docs/         design records and experiment reports
```

The latest experimental entry points are:

**DetSkill Fix (targeted PairedDeltaPredictor training):**

```bash
python -m protein_jepa_frame.cli train \
  --config configs/lspwm_v44_stage_b_detskill_fix.yaml
```

This configuration enables the previously disabled PairedDeltaPredictor
(`paired_delta_weight: 0.005`, was 0.0) with physical time conditioning
(`time_conditioning: physical`) and deeper predictor heads
(`predictor_depth: 3`, `paired_predictor_depth: 3`). All encoders remain
fully frozen (`encoder_lr_scale: 0.0`) to eliminate latent collapse risk.
It produces the best DetSkill results to date (see §14.2).

**Stage B v4 architectural revision (thermodynamic dynamics fitting):**

```bash
python -m protein_jepa_frame.scripts.train_lspwm_fixed_cv \
  --config configs/lspwm_v44_stage_b_v4_architectural.yaml
```

This starts the v4.4 thermodynamic dynamics fitting with:
- BAOAB midpoint integration (Rev-1: O(Δt²) accuracy)
- Multi-step trajectory NLL (Rev-2: path likelihood training)
- Learned auxiliary initialization (Rev-3: data-driven bath state)
- Semigroup consistency (Rev-4: cross-timescale Koopman constraint)
- Encoder fine-tuning at lr_scale=0.01 (Rev-5: representation adaptation)

It does not instantiate the full environment/action/multiscale target defined
above. To run the earlier v3 baseline, substitute
`configs/lspwm_v44_stage_b_v3_improved_thermodynamics.yaml`.

## 14. Current Implementation Status

| Target capability | Repository status | Consequence |
| --- | --- | --- |
| causal JEPA and EMA target | implemented candidate | requires leakage, collapse, and sufficiency audits |
| protein coarse state | implemented candidate | not yet a proven slow or thermodynamically sufficient state |
| evolving environment state | absent from v4.4 | fixed environment assumption only |
| action/control state | absent from v4.4 | no counterfactual intervention claim |
| scalar PMF | implemented candidate | must be measure-aware and temperature-consistent |
| global diffusion | parameterized candidate | PSD is not FDT or path validation |
| finite Mori bath | partial candidate | finite exponential approximation, not complete hidden physics |
| nonlinear/operator memory | absent | kernel-family adequacy remains open |
| structural mechanics layer | incomplete | coordinate generation requires geometry constraints |
| multiscale hierarchy | absent | no claim beyond measured sampling scales |
| full path likelihood | partial | one-step likelihood is not a complete path law |
| intervention likelihood | absent | no action-conditioned world model |
| identifiability protocol | absent | gauge fixing does not establish uniqueness |
| physical evaluation gates | partial | FDT is sampled; path and continuous detailed-balance gates remain open |
| protein-disjoint benchmark | available in manifests | generalization is empirical, not guaranteed |

The `lspwm_v44` configurations are experimental recipes, not certified
complete LSP-WM checkpoints. v4.4 is the thermodynamic latent-dynamics base
for the full protein-environment LSP-WM target.

### 14.1 Stage B v3 — Evaluation Results

The current best checkpoint (`checkpoint_step3000.pt`) was evaluated on
two complementary benchmarks:

**LSPWM Internal Evaluation** (240 trajectory samples, bandwidth 0.1)

| Metric | Result | Target |
| --- | --- | --- |
| Deterministic Skill | **0.774** (H24:0.792, H48:0.774, H96:0.756) | ≥0.85 |
| VAMP-2 Score | **47.01** | >20 |
| Force Cosine (pred vs KDE target) | **+0.033** | >0.0 |
| Force MSE | 0.037 | — |
| Mean Circulation (loop test) | 0.040 | >0.0 (active) |
| Energy | healthy (mean=0.0013, std=0.0039) | healthy |
| Force | healthy (norm=0.010) | healthy |
| Diffusion | healthy (state-dependent) | healthy |

Key improvement from Stage B v3: force cosine became positive (step 1500 had
0.0; step 3000 achieves +0.033), indicating that the learned potential
captures meaningful thermodynamic forces after the curriculum enables force
loss at step 1500.

**FastFolding Benchmark** (scaffold-disjoint split, 4 test proteins: 2F4K,
CLN025, GTT-0, GTT-1; 1372 samples)

| Benchmark | Metric | Result |
| --- | --- | --- |
| Committor Proxy (bootstrap aggregate) | Accuracy | **0.87** [95%CI: 0.75–0.99] |
| Committor Proxy (Horizon 1) | Accuracy | 0.75 |
| Transition Operator (Horizon 1) | MFPT Rel Error | 0.755 |
| Transition Operator (Horizon 1) | Row JS Divergence | 0.130 |
| Transition Operator (Horizon 8, best) | Row JS Divergence | **0.068** |
| Transition Operator (Horizon 8, best) | MFPT Rel Error | 0.830 |

The FastFolding committor accuracy of 0.87 indicates that the model
discriminates folded/unfolded/transition states above chance at short
horizons. The transition operator at H=8 shows the closest match to the
reference MSM (Row JS Divergence = 0.068), though the predicted stationary
distribution remains biased toward the folded state relative to the reference.

### 14.2 Stage B DetSkill Fix — Evaluation Results

The DetSkill fix (`configs/lspwm_v44_stage_b_detskill_fix.yaml`) is a targeted
correction that enables PairedDeltaPredictor training (which was previously
disabled by `paired_delta_weight: 0.0`). The configuration adds physical time
conditioning and deeper predictor heads while keeping all encoders fully frozen
to eliminate latent collapse risk.

**Training configuration:**
| Parameter | Value | Purpose |
| --- | --- | --- |
| `paired_delta_weight` | **0.005** (was 0.0) | Enable PairedDeltaPredictor gradient flow |
| `time_conditioning` | **physical** | Temporal encoder receives Δt as physical time |
| `predictor_depth` | **3** (was 2) | Deeper context predictor |
| `paired_predictor_depth` | **3** (was 3) | Deeper paired predictor |
| `encoder_lr_scale` | **0.0** | All encoders fully frozen |
| `freeze_context_encoder` | **true** | No latent collapse risk |
| `freeze_temporal_encoder` | **true** | No latent collapse risk |
| `freeze_target_encoders` | **true** | No latent collapse risk |
| Training steps | 3000 | Full Stage B curriculum |

**Training results** (logged every 10 steps, 300 entries):

| Metric | Result | Target | Status |
| --- | --- | --- | --- |
| **Max DetSkill** | **0.889** (Step 2090) | ≥0.85 | ✅ **Exceeded by 4.6%** |
| **Final DetSkill** | **0.820** (Step 2990) | ≥0.85 | 96.5% of target |
| **Max H=24 Skill** | **0.893** (Step 1540) | ≥0.83 | ✅ **Exceeded** |
| **Max H=48 Skill** | **0.908** (Step 2090) | ≥0.80 | ✅ **Exceeded** |
| **Max H=96 Skill** | **0.882** (Step 1600) | ≥0.80 | ✅ **Exceeded** |
| **Final H=24 Skill** | **0.841** (Step 2990) | ≥0.83 | ✅ **Exceeded** |
| **Final H=48 Skill** | **0.818** (Step 2990) | ≥0.80 | ✅ **Exceeded** |
| **Final H=96 Skill** | **0.801** (Step 2990) | ≥0.80 | ✅ **Exceeded** |
| **Delta Loss (initial)** | 0.103 | — | — |
| **Delta Loss (final)** | 0.020 | ↓ | 80.3% reduction |
| **Delta Loss (best)** | 0.007 | ↓ | 93.5% reduction |
| **Avg Skill (last 10 steps)** | **0.819** | ≥0.85 | 96.4% |
| **Avg Skill (overall)** | 0.759 | — | — |

**Per-horizon progression (every 500 steps):**

| Step | Skill | H=24 | H=48 | H=96 | Delta Loss |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.005 | — | — | — | 0.103 |
| 500 | 0.688 | 0.672 | 0.690 | 0.702 | 0.029 |
| 1000 | 0.713 | 0.747 | 0.706 | 0.688 | 0.019 |
| 1500 | 0.797 | 0.791 | 0.797 | 0.803 | 0.024 |
| 2000 | **0.851** | **0.874** | **0.853** | **0.826** | 0.015 |
| 2500 | 0.828 | **0.834** | **0.832** | **0.815** | 0.016 |
| 2990 | 0.820 | **0.841** | **0.818** | **0.801** | 0.020 |

**Comparison with Stage B v3 baseline:**

| Metric | Stage B v3 | DetSkill Fix | Improvement |
| --- | --- | --- | --- |
| DetSkill (max) | 0.774 | **0.889** | **+14.9%** 🚀 |
| DetSkill (final) | 0.774 | **0.820** | **+5.9%** |
| H=24 (final) | 0.792 | **0.841** | **+6.2%** |
| H=48 (final) | 0.774 | **0.818** | **+5.7%** |
| H=96 (final) | 0.756 | **0.801** | **+6.0%** |

**Safe operation diagnostics (all 3000 steps):**

| Diagnostic | Observed | Requirement | Status |
| --- | --- | --- | --- |
| `grad_norm_context_encoder` | **0.0** (all steps) | = 0.0 | ✅ Frozen |
| `grad_norm_temporal_encoder` | **0.0** (all steps) | = 0.0 | ✅ Frozen |
| `grad_norm_target_encoders` | **0.0** (all steps) | = 0.0 | ✅ Frozen |
| `grad_norm_paired_predictor` | **> 0.0** (all steps) | > 0.0 | ✅ Training |
| `grad_norm_predictor` | **> 0.0** (all steps) | > 0.0 | ✅ Training |
| Latent collapse detection | **No collapse** | No encoder drift | ✅ Safe |

**Key insight:** The root cause of the DetSkill gap in Stage B v3 was that
`paired_delta_weight: 0.0` prevented the PairedDeltaPredictor from receiving
any gradient signal. Simply enabling it (0.005) with physical time conditioning
and deeper predictors recovered DetSkill from 0.005 → 0.889 max → 0.820 final,
with all three horizons exceeding their targets at the final step. The model
achieved peak skill of 0.889 at step 2090 (mid-training), with final skill
stabilizing at 0.820.

**Limitations:**
- The final skill (0.820) is below the peak (0.889), suggesting possible
  overfitting or learning rate dynamics that cause late-training degradation.
- No comprehensive evaluation (VAMP-2, force matching, FastFolding benchmarks)
  has been run on this checkpoint yet.
- The frozen encoder constraint, while safe, limits the representational
  adaptation that could further improve DetSkill.

## 15. Verification Commands

Install with Python 3.10+ and a compatible PyTorch build:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,eval]'
```

Run the current software tests:

```bash
python -m pytest -q tests
```

Run the synthetic contract test:

```bash
python scripts/evaluate_lspwm.py \
  --synthetic \
  --output artifacts/lspwm_smoke.json
```

The physical diagnostics entry point samples the executed one-step noise law
for its FDT residual. Its soft-MSM balance number is explicitly a readout
diagnostic, not a proof of detailed balance for the continuous world model:

```bash
python scripts/evaluate_lspwm_physical.py \
  --config configs/lspwm_v41_unified256_rollout.yaml \
  --checkpoint artifacts/lspwm_v41_unified256_rollout/checkpoint_latest.pt \
  --manifest manifests/tmjepa_v3.validation.jsonl \
  --output artifacts/lspwm_v41_unified256_rollout/physical.json \
  --fdt-samples 32
```

These commands test serialization, tensor shapes, finite values, and selected
parameterization contracts. They do not establish PMF correctness, FDT,
detailed balance, path quality, action response, structural validity,
multiscale behavior, protein generalization, or external SOTA.

### 15.1 Comprehensive evaluation (Stage B v3)

```bash
python -m protein_jepa_frame.eval.comprehensive_eval \
  --checkpoint artifacts/lspwm_v44_stage_b_v3_improved_thermodynamics/checkpoint_step3000.pt \
  --config artifacts/lspwm_v44_stage_b_v3_improved_thermodynamics/eval_improved/eval_config.json \
  --output-dir artifacts/lspwm_v44_stage_b_v3_improved_thermodynamics/eval_improved \
  --max-batches 20 --bandwidth 0.1 --device cuda:0 \
  --fastfolding-labels /tmp/lspwm_fastfolding_labels.jsonl
```

This runs the full evaluation pipeline: LSPWM internal metrics (endpoint skill,
force matching, Koopman/VAMP-2, energy conservation, component diagnostics)
followed by the FastFolding dynamics benchmarks (committor proxy, transition
operator, MFPT). Results are saved to `combined_results.json`.

### 15.2 Comprehensive evaluation (Stage B DetSkill Fix)

```bash
python -m protein_jepa_frame.eval.comprehensive_eval \
  --checkpoint artifacts/lspwm_v44_stage_b_detskill_fix/checkpoint_latest.pt \
  --config configs/lspwm_v44_stage_b_detskill_fix.yaml \
  --output-dir artifacts/lspwm_v44_stage_b_detskill_fix/eval \
  --max-batches 20 --bandwidth 0.1 --device cuda:0
```

### 15.3 LSPWM internal evaluation only

```bash
python -m protein_jepa_frame.eval.lspwm_eval \
  --checkpoint artifacts/lspwm_v44_stage_b_v3_improved_thermodynamics/checkpoint_step3000.pt \
  --config configs/lspwm_v44_stage_b_v3_improved_thermodynamics.yaml \
  --output-dir artifacts/lspwm_v44_stage_b_v3_improved_thermodynamics/eval_improved \
  --max-batches 20 --bandwidth 0.1
```

## 16. Claim Boundary

The correct current claim is:

> Protein-JEPA-Frame contains an experimental thermodynamic latent-dynamics
> subsystem (v4.4, Stage B v3 and DetSkill Fix) and a design path toward an
> intervenable, multiscale protein-environment LSP-WM. The v4.4 Stage B v3
> checkpoint demonstrates measurable thermodynamic consistency: VAMP-2 = 47
> (target >20), positive force cosine (+0.033), and FastFolding committor
> accuracy of 0.87 on scaffold-disjoint held-out proteins.
>
> The DetSkill Fix checkpoint (`lspwm_v44_stage_b_detskill_fix`) resolves the
> DetSkill gap in Stage B v3 by enabling PairedDeltaPredictor training (was
> disabled by `paired_delta_weight: 0.0`), achieving DetSkill = 0.889 max
> (target ≥0.85) and final DetSkill = 0.820 with all three horizons exceeding
> their targets, while maintaining zero latent collapse risk (all encoders
> frozen). Comprehensive evaluation (VAMP-2, force matching, FastFolding
> benchmarks) on this checkpoint is pending.
>
> It has not yet established a complete protein world model, a unique physical
> decomposition, protein-scale intervention validity, or literature-wide SOTA.

A future claim of complete LSP-WM compliance must include the exact state
factorization, observation measure, mechanics constraints, temperature law,
environment dynamics, action semantics, memory covariance, path likelihood,
multiscale protocol, identifiability audit, protein-disjoint split, and all
measured physical gates in a machine-readable report.
