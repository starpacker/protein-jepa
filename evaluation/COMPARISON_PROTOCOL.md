# Frozen-representation comparison protocol

The question is whether a trained memory bath adds predictive information beyond
an independently trained no-bath kernel on the same frozen H representation.
The required rows are `h_markov` (bath dimension 0), `h_plus_s` (bath dimension >0),
`direct_jepa` (history-to-endpoint prediction), and coordinate `persist`.
The old `h_only`/`h_ft` names retain their direct-predictor compatibility meaning;
they must not be relabelled as a no-bath dynamics ablation.

Two scopes must be reported separately:

1. **Within-protein chronological:** training, development and final-test time
   intervals are disjoint on each physical trajectory. Every complete history
   and future endpoint/path belongs to its declared interval. Fit the encoder,
   normalizer, decoder and all predictors using the training interval only.
2. **Cross-protein:** complete protein/domain/source identities are disjoint
   across training, development and final test. All learned stages respect this
   split; an upstream encoder trained on test proteins invalidates this claim.

Development data selects checkpoints and diagnoses the readout. Final test is
opened only after the protocol and selection rule are locked. It cannot select
training duration, seed, horizon, oracle threshold, crop or sample count.

Use `python -m evaluation.preregister --config <experiment.yaml> --output
<new_registration.json>` before the final comparison. The experiment YAML must
declare both scopes, each train/development/test manifest, all protocol fields,
training seeds and exact training budget. The output is immutable. Evaluation
configurations set `preregistration`, `comparison_scope`, and
`dataset_role: final_test`; a changed manifest or endpoint protocol fails.

## Fairness and causal contract

- All learned variants have the same content-verified encoder, fitted train-only
  normalizer and frozen raw-H decoder. Oracle and anchor-H controls are identical
  functions on this representation. Main predictions decode predicted H only.
- Train H Markov and H+S independently with the same windows, supervision,
  optimizer family, number of updates, sample budget, loss weights and checkpoint
  selection metric. Report both parameter counts and compute; adding a bath adds
  parameters, so a win alone does not isolate memory from capacity.
- The H Markov kernel reads the last history H. The direct JEPA predictor and
  bath posterior read the same complete history. This isolates the proposed
  state augmentation, while history information differs by model design.
- Keep all finite and nonfinite outcomes. Nonfinite predictions fail the run;
  no sample rejection, best-of-M, target-driven alignment or window selection.
- Use identical exact scoring windows and deterministic per-window sampling
  seeds. The same scalar seed is not automatically common H noise: a bath can
  consume additional random draws. Report independent-stream support explicitly.
- Gamma bounds only constrain the isolated bath Euler factor. Log D, C,
  h*Gamma, normalized-H rollout norm/max and latent displacement. A finite pilot
  does not establish coupled stability or equilibrium correctness.

## Measurements and decision rule

The primary comparison is H+S minus H Markov, separately at every registered
horizon, for (1) empirical nonlocal CA-distance energy score and (2) common-anchor
CA RMSD of the coordinate ensemble mean. Lower is better. Also compare each
model with persistence and H+S with direct JEPA. Report effect sizes and paired
95% protein-bootstrap intervals without selecting favorable horizons. With one
protein there is no population significance claim. Multiple horizons and seeds
remain exploratory unless a correction/combined decision rule is preregistered.

Average paired window differences within each protein, then give proteins equal
weight. Output each sample's primary/shape RMSD, displacement and CA validity;
keep ensemble-mean geometry separate because an average need not be a valid
conformation. Report internal distance, contact and signed CA pseudo-torsion
errors. CA bond/clash diagnostics cannot certify full backbone stereochemistry.

The distance energy score scores the actual finite empirical forecast using the
V-statistic diagonal; deterministic baselines are point-mass forecasts. It is a
proper score in observable space, not evidence of full trajectory kinetics.
RMSF, ACF and transition times require additional path-level experiments.

## Failed gate policy

Do not change oracle thresholds to authorize comparison training. A deliberately
diagnostic checkpoint uses `run_scope: diagnostic_comparison`; evaluation requires
`diagnostic_comparison: true`, writes `diagnostic_aggregate`, and cannot produce
headline evidence. It still checks representation identity, split lineage,
causality, all exact windows, finite samples and paired statistics. State a
failed readout gate alongside every numerical comparison, including negative or
inconclusive results. Continue readout diagnosis independently.
