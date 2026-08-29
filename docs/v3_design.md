# V3 Protected Hierarchical Dynamics Model

Date: 2026-08-26

## Objective

V3 is a redesign around the evidence from Phase 1 and V2, not another loss
reweighting pass. It has two explicit products:

1. a protected representation path for VAMP and MSM;
2. a history-conditioned transition path for latent and physical forecast MSE.

The two products share a frozen Phase-1 geometry/state encoder at the first
gate, but they have separate readouts and promotion criteria. This avoids
forcing one pooled endpoint to be simultaneously invariant, metastable and
sensitive to fast local motion.

## Lessons From V0-V2.3.1

1. The formal Phase-1 scores (`0.5588 / 0.5646 / 0.5381`) and honest
   non-overlap scores are different protocols. Both must be reported.
2. Direct endpoint regression can show high apparent paired skill while losing
   VAMP/MSM information. Lower training loss is not a promotion signal.
3. Online-context versus EMA-future comparisons inflated V2.3 skill. All V3
   deltas, predictions and persistence baselines use one frozen teacher space.
4. A global endpoint lacks velocity and local deformation information. V2.3.1
   therefore recovered only `1-4%` canonical persistence-relative skill.
5. Small-batch contrastive slow heads collapse or learn protein identity. V3
   estimates its kinetic objective from all within-window pairs and keeps the
   champion state available unchanged.
6. Semigroup consistency can become small for an uninformative operator. It is
   excluded from the first V3 gate and may return only after one-step skill is
   established.
7. A Gaussian head cannot repair a biased deterministic mean. Probabilistic
   training remains blocked until deterministic forecast baselines pass.

## V3.0 Architecture

### Protected champion state

The Phase-1 encoder and bidirectional temporal tower are loaded exactly and
frozen. The original `state`/`endpoint` readout remains bit-identical to the
champion, so V3 cannot silently destroy the baseline while learning dynamics.

### Multiscale history transition

The transition receives the full observed champion-state history. For strides
`1/2/4/8`, it computes masked mean and standard deviation of state increments,
then combines these with the endpoint and window mean. A time/temperature
conditioned residual MLP predicts the canonical champion-state delta.

The output is zero-initialized, making persistence the exact initial model.
This tests whether velocity and multiscale fluctuations predict a future
champion state better than the last state alone.

### Reversible metastable readout

An 8-state soft assignment head is trained on frozen champion frame states.
Its VAMP-2 objective pools valid time-lagged pairs from both context and future
windows, giving an effective covariance sample much larger than batch size.
Uniform occupancy regularization prevents one-state collapse; a small entropy
term encourages useful, non-diffuse assignments.

The `metastable` readout is evaluated independently for VAMP, MSM and linear
dynamics. The original champion readout is always reported beside it.

### Physical transition vector

V3 replaces the four ambiguous aggregate observables with 12 normalized,
SE(3)-invariant targets:

- retained, gained and lost nonlocal-contact fractions;
- signed circular changes for phi, psi and omega;
- absolute circular changes for phi, psi and omega;
- signed radius-of-gyration change;
- Kabsch-aligned CA RMSD;
- signed local CA-distance change.

Evaluation reports MAE, MSE and skill relative to a physical persistence vector
(`retained=1`, every change channel zero).

## Training Stages

### Gate A: anchored deterministic, 2k

Config: `configs/phase3_v3_0_history_metastable_2k.yaml`

- mdCATH only;
- non-overlap horizons `48/96/192`;
- frozen champion geometry and offline temporal tower;
- history transition, metastable head and physical head train;
- checkpoint at 2k;
- no Gaussian, semigroup or rollout loss.

### Gate B: local/domain state

Gate B is implemented only if Gate A has positive held-out forecast skill. It
adds residue/domain motion tokens and future-mask residue JEPA while retaining
the exact champion branch as a control. It is not justified if history alone
cannot beat persistence.

### Gate C: stochastic rollout

A mixture or diffusion residual is introduced only after the deterministic
mean passes. Selection then uses NLL/CRPS/coverage, two-step rollout error,
basin populations and implied-timescale error.

## Promotion Rules

Gate A continues beyond 2k only when all of the following hold on held-out
proteins:

1. canonical latent skill is positive at all `48/96/192` horizons and at least
   `0.10` at horizon 48;
2. physical-observable MSE skill over persistence is positive at every horizon;
3. `metastable` beats the honest champion on at least two of VAMP/MSM/linear
   R2 without a material regression on the third;
4. the original frozen champion readout remains numerically unchanged;
5. shuffled-horizon and shuffled-future controls are worse than the real pair.

Final promotion additionally requires the aligned Phase-1 scoreboard, the
honest non-overlap scoreboard, protein bootstrap confidence intervals and three
seeds. A task-specific readout improvement is reported as such; it is not
misrepresented as a universal representation improvement.

## Expected Failure Signals

- VAMP improves but MSM falls: assignments capture continuous fast variance,
  not metastable basins; adjust lag curriculum or state count.
- MSM improves but latent MSE stays near persistence: history is insufficient;
  proceed to local/domain tokens only if physical MSE has signal.
- Observable MSE improves but latent MSE does not: the champion state is not a
  closed dynamical coordinate; forecast physical state directly in Gate B.
- Training VAMP improves while held-out VAMP does not: protein identity or
  occupancy shortcut; require grouped batches and same-protein shuffle tests.

## Gate A Result

Gate A completed at 2k and did not pass. Canonical latent skill at horizons
`48/96/192` was `0.0078/0.0102/-0.0015`. The metastable readout reached honest
quick VAMP/MSM/linear R2 of `0.4104/0.4098/-0.7561`, compared with champion
`0.4049/0.4154/-0.8434`. These are small readout gains, not a closed world
state. The physical head requires mean and shuffle controls before its high
aggregate persistence-relative skill can be interpreted. No 10k continuation
or stochastic transition is authorized from this result.

## V3.1 Local Physical State Gate

The post-V3.0 control audit found that the global physical head beats a
held-out horizon mean (`0.021-0.027` MSE versus `0.033-0.044`) and becomes much
worse when context history is shuffled (`0.042-0.062`). The signal is therefore
not only a target-mean shortcut. Shuffling horizon has little effect, however,
and the champion latent delta is insensitive to shuffled history.

V3.1 keeps all V3.0 weights fixed in practice and trains four sequence-local
dynamics tokens. Each token pools frozen residue latents from one contiguous
sequence segment and predicts that segment's 12-channel physical transition.
The prediction history is detached, so this gate cannot erase the champion or
retroactively optimize the failed global transition. The new
`local_dynamics` evaluation readout concatenates the champion endpoint with the
four learned local tokens.

Config: `configs/phase3_v3_1_local_dynamics_2k.yaml`. Continuation requires
local physical MSE to beat persistence and shuffled-history controls at all
three horizons, plus an improvement on at least two honest representation
metrics without a material third-metric regression.

### V3.1 result

V3.1 passed the local physical gate: persistence-relative MSE skill was
`0.8230/0.8376/0.8237`, and shuffled-history MSE was substantially worse than
the real-context prediction. Its local plus metastable readout reached honest
quick VAMP/MSM/linear R2 `0.3855/0.4337/-0.4159`. This improves MSM and R2 over
the champion but does not retain champion VAMP (`0.4049`). V3.1 is therefore a
multi-head candidate, not yet a single-readout promotion. Horizon shuffling
still has little effect and is the primary target for V3.2.
