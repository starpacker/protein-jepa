# TM-JEPA V3 logic audit (2026-09-04)

## Scope and actual execution graph

The current fixed-CV run is not the encoder-to-dynamics pipeline shown in the
concept diagram. With `tmjepa_learned_cv_dim=0`, `forward_tmjepa` bypasses the
geometry and temporal Transformers. The active graph is:

`coordinates -> 16 fixed CVs -> Kalman filter -> V3 dynamics -> rollout`.

Sequence features enter separately through `ProteinConditionEncoder`. The
checkpoint optimizer confirms that only the TM-JEPA kinetics group receives
Adam state in the observed-long run.

## Confirmed logic defects

| ID | Layer | Defect | Consequence | Status |
| --- | --- | --- | --- | --- |
| C1 | Training | `init_exclude_prefixes: ["tmjepa."]` discards the pretrained potential and condition encoder while `tmjepa_freeze_potential=true` freezes both. | A seeded random energy landscape is frozen and kinetics must compensate for it. | Guard added; current run remains invalid. |
| C2 | Training | Fixed-CV mode still added unused context, temporal, and paired-predictor groups to AdamW. | Misleading LR/freeze interpretation and dead checkpoint/optimizer state. | Fixed. |
| C3 | Evaluation | Filtered samples included endpoint observation noise; reset samples did not. | `memory_filter_advantage` compared different predictive distributions and was biased. | Fixed. |
| C4 | Evaluation | Persistence used the inferred latent `current`, not the last observed CV. | Observation-error inference could inflate or depress the baseline and therefore skill. | Fixed. |
| C5 | Evaluation | `paired_all_horizons` duplicates each context five times and closure statistics counted all copies. | Transition counts were inflated fivefold; uncertainty estimates would be false. | Fixed for closure and physics aggregation. |
| C6 | Data | CV statistics were fitted sequentially with `shuffle=False` and an early batch cap. The saved 2,560 frames cover only the beginning of a protein-sorted manifest. | The affine gauge is biased toward the first proteins and can create heavy standardized outliers. | Sampler fixed; statistics and all dependent checkpoints must be regenerated. |
| C7 | Training | Paired mode bypassed the mutual-exclusion check for `init_checkpoint` and `resume_checkpoint`. | Both could be loaded in sequence with ambiguous semantics. | Fixed. |
| C8 | Reporting | One document claimed eight gates passed, while the old evaluator implemented six gates and retained results explicitly said Mori was not promoted. | The reported scientific status was internally contradictory. | Claim corrected; two explicit coverage gates were added. |
| C9 | Training | PMF score matching used the Kalman-denoised state, which depended on the untrained potential and frozen kinetics. | The equilibrium training distribution moved with the model and was not the observed CV distribution. | Fixed to use the last observed CV. |

## Framework risks requiring controlled ablation

These are not safe to patch by choosing an arbitrary constant. Each needs an
identifiability or held-out experiment.

1. **It is currently not JEPA.** `latent_dim`, the geometry Transformer, the
   causal Transformer, target EMA, and paired predictor do not affect V3 when
   fixed CVs are used. Setting `encoder_lr_scale>0` cannot change this. V3 also
   rejects any nonzero learned-CV dimension.
2. **The 16-CV closure is a strong bottleneck.** Global shape, contiguous-third
   segment summaries, contacts, and averaged torsions omit nonlocal hinges,
   side-chain packing, and transition-state coordinates. A Mori bath cannot
   recover state information that never enters the resolved variables.
3. **Equilibrium and kinetics share one frozen condition representation.** If
   it is unfrozen, the supposedly frozen potential changes through its input;
   if it stays frozen, kinetics cannot learn a separate protein representation.
   Separate equilibrium and kinetic adapters are needed before end-to-end use.
4. **Observation noise has no clear physical source.** mdCATH coordinates are
   simulated coordinates, not noisy measurements. Treating unresolved fast CV
   motion as iid measurement error changes the state being forecast and lets
   observation variance absorb process variance.
5. **White diffusion, observation noise, and Mori memory are weakly identifiable
   at one 1 ns sampling interval.** Current observation variance contributes
   about `0.0194` to increment variance, while the minimum diffusion direction
   in the latest checkpoint contributes only about `0.0017` over 1 ns. This is
   consistent with diffusion being displaced by the observation model.
6. **The transition is only locally exact.** Van Loan is exact after freezing
   force and kinetic coefficients at the start of a step. The filter takes one
   such 1 ns step, so nonlinear free-energy and state-dependent mobility error
   are not controlled by `base_step_ps`; that setting only subdivides rollout.
7. **The potential is not yet validated as a PMF.** Denoising score matching on
   trajectory frames assumes equilibrated samples and sufficient temperature
   overlap. Replica stationarity, burn-in, held-out score, and the three-term
   enthalpy/entropy/heat-capacity identifiability have not been established.
8. **The kinetic parameterization is restrictive in uneven ways.** Diffusion
   has a global low-rank basis plus coordinatewise state dependence; coupling
   directions depend on protein but not conformation; bath rotation is global.
   Capacity should only be expanded after the simpler model is identifiable.
9. **The memory grid is not aligned with the identified poles.** Residual
   analyses concentrate modes near 1.0-1.6 ns and 10.5-16 ns, while the current
   log-spaced grid spends modes in unsupported middle scales and allows only a
   limited relaxation adjustment.

## Training defects and optimization risks

1. Batch size 2 makes the 16x16 innovation covariance and lag-correlation
   penalties high variance. The observed covariance loss has spikes up to
   `66.3`; its late median is near `0.31`, so the mean loss is dominated by
   rare clips.
2. Path NLL already scores innovation mean and covariance. Adding three
   moment penalties at weight 1 overweights noisy minibatch estimates and
   changes the likelihood optimum. Closure penalties need accumulated or
   shrinkage statistics and a lower second-stage weight.
3. The latest diffusion minimum fell from roughly `4.0e-5` to below `1.0e-6`.
   Strict positivity is guaranteed by construction, so the current PSD gate
   cannot detect this practical collapse.
4. The intervention objective has an absolute margin of `0.001` and weight
   `0.1`. Turning coupling toward zero makes filtered and reset forecasts nearly
   equal at negligible cost. Use a persistence-normalized margin and report a
   held-out likelihood/energy advantage.
5. An eight-member ensemble gives a noisy 16-dimensional energy-score gradient.
   Use a larger ensemble or common/antithetic random numbers and gradient
   accumulation before changing model capacity.
6. All five horizons share a context but are sampled and filtered separately.
   Filtering once and rolling out all horizons would reduce compute and make
   horizon comparisons paired.
7. There is no in-training protein-disjoint validation or checkpoint selection.
   A 3,000-step latest checkpoint can be worse than an earlier checkpoint.
8. Logs omit module-level gradient norms, diffusion trace/condition number,
   observation/process variance ratio, coupling norm, integrated kernel ratio,
   and posterior bath information. Failures currently appear only after a full
   evaluation.

## Evaluation defects and residual limitations

1. Training `memory_intervention_advantage` is deterministic MSE, whereas
   evaluation `memory_filter_advantage` is stochastic energy score. They must
   be named separately and must not be compared numerically.
2. Validation samples 100 contexts with replacement. Temperature-by-horizon
   bins are small, some proteins may be absent, and a single stochastic seed is
   used. Promotion needs exhaustive or explicitly stratified contexts plus
   protein bootstrap intervals and repeated rollout seeds.
3. Training logs report mean squared lag correlation; the gate reports its RMS.
   A logged value of `0.0135` means RMS about `0.116`, not `0.0135`, and fails
   the `<0.05` target.
4. `transition_nll` uses diagonal variance estimated from eight rollout samples
   and ignores cross-CV covariance. It is not a reliable multivariate NLL.
5. Coverage is marginal per CV, not joint 90% coverage, and its variance is
   estimated from only eight samples. Both marginal and multivariate rank or
   Mahalanobis calibration should be reported.
6. `diffusion_psd` is nearly tautological because the parameterization adds a
   positive diagonal floor. Add condition number, floor-hit fraction, and
   empirical increment-covariance agreement.
7. `base_step_ps` sensitivity in evaluation probes rollout discretization but
   not filter transition-density error. These must be reported separately.
8. Training metrics are not promotion evidence. The current run has no held-out
   result, and historical V3 results show negative 16 ns skill and no predictive
   Mori-filter advantage.

## Repair order

1. Regenerate balanced CV statistics and retrain/validate the PMF checkpoint.
2. Fit a no-memory, no-observation-noise overdamped baseline with path NLL.
3. Identify observation variance from multi-stride likelihood (1/2/4 ns) or
   keep it disabled; promote it only on held-out likelihood and calibration.
4. Add Mori memory from the same PMF/overdamped initialization, jointly refit
   diffusion, and require positive protein-bootstrap filter advantage.
5. Only after fixed-CV closure passes, pretrain a slow JEPA representation,
   freeze its gauge, and add it as a residual closure coordinate. Do not simply
   unfreeze the currently bypassed encoder.
