# V3.5 Velocity-Conditioned Robust Probabilistic Dynamics

Date: 2026-08-27

## Why V3.4 saturated

V3.4 improved normalized persistence-relative state skill from the V3.2
baseline to `1.70%` at step 10000, but its trained horizon-pair skill remained
`0.30%` and horizon shuffling changed MSE by only `0.16%`. The model predicted
less than one percent of the observed target horizon-difference energy.

The oscillating `closed_local_target_difference_mse` is not a model loss. It
is the standardized energy of the realized target difference in each batch.
Across V3.4 steps 2000-10000 its coefficient of variation was `1.45`, and its
correlation with the optimized difference MSE was effectively `1.0`. No one
channel dominated; a small set of high-motion context windows dominated the
squared objective. The deterministic endpoint-only model therefore learned a
weak conditional mean while the objective followed thermal trajectory noise.

## Mechanism

V3.5 changes the closed local transition only. Frozen representation weights
are inherited from V3.4 step 10000 and the closed operator is reinitialized.

1. The context state includes standardized finite-difference velocities at
   strides `1/2/4/8`, computed strictly from observed context frames.
2. Continuous drift/relaxation coefficients predict normalized state deltas;
   the known train-only channel scales map the result back to physical state
   units. This makes output conditioning consistent with the loss.
3. A Student-t scale head separates predictable mean motion from heavy-tailed
   realized fluctuations. It reports NLL and scale coverage without replacing
   deterministic MSE evaluation.
4. The new horizon-difference objective normalizes each context/horizon pair
   by its detached target RMS (with a `0.5` floor) and uses Huber loss. Raw
   difference MSE remains a logged diagnostic but has zero optimization weight.

## 2K gate

The run uses mdCATH only, five same-context horizons, batch size two, and saves
at steps `500/1000/1500/2000`. Held-out evaluation uses the same filtered
validation manifest, seed, and raw/normalized controls as V3.4.

Promotion requires evidence in all three categories:

- normalized mean and worst-horizon persistence skill must improve over the
  comparable V3.4 checkpoints, with the 10K V3.4 result as the final target;
- trained horizon-pair skill and horizon-shuffle sensitivity must materially
  exceed V3.4 (`0.30%` and `0.16%` respectively);
- Student-t NLL must beat persistence under the same predicted scale, and
  uncertainty coverage must be non-degenerate.

If velocity improves shuffled-history sensitivity but not pairwise horizon
skill, the next change should model distributions or transition ensembles over
rollouts rather than extending deterministic mean training.
