# V2.3 Dual-Timescale Anchored World Model

V2.3 is designed to cross the V2.2 failure mode: V2.2 can predict paired
endpoints beyond persistence, but the shared representation loses the
Phase-1 VAMP/MSM/linear-dynamics quality after continued optimization. V2.3
therefore separates the offline representation contract from the causal
forecast contract and adds a calibrated transition path in stages.

## Audit Findings That Shape V2.3

1. The formal Phase-1 champion scores `0.5588 / 0.5646 / 0.5381` on the
   aligned contextual quick protocol, whose selected VAMP and linear metrics
   use overlapping windows. The honest Phase-2 E0 scores `0.4065 / 0.4328 /
   -0.8443` at non-overlapping lag 6. These are different scoreboards and both
   must be reported.
2. V2.1 fixed the direct paired data contract and obtained held-out horizon-48
   skill `0.3675`. V2.2 reached `0.6054` at 2k on the same paired diagnostic,
   but its endpoint and slow skill fell sharply by 4k. Lower training loss is
   not a sufficient selection signal.
3. The current encoder had a leakage path: orientation-aware torsion features
   used the full residue mask instead of the visible token mask. This has been
   fixed and covered by a regression test.
4. The current paired horizons `8/16/24/48` include overlapping windows for
   horizons below the 47-raw-frame context span. V2.3 uses `48/96/192` as its
   primary non-overlap forecast buckets and keeps short horizons only as a
   separately labeled auxiliary diagnostic.

## Architecture

### Shared geometry trunk

The spatial encoder remains SE(3)-invariant and uses visible-only graph
construction, signed backbone torsions, local-frame orientation features,
sequence separation and missing-atom masks. The fixed masked-neighbor contract
is tested before any long run.

### Offline representation tower

`h_offline(t)` is a bidirectional temporal state tower used only for offline
representation probes. It is trained with the proven future-only JEPA/state
objective, low-weight state-delta preservation, and a slow-state variance/
covariance control. Its readout is explicit and remains comparable to the
Phase-1 `state` readout.

### Causal dynamics tower

`h_causal(t)` is a separate causal temporal tower. It receives the same spatial
features but never sees future frames. The forecast state is its final valid
endpoint, and its primary training pairs are non-overlapping raw offsets
`48/96/192` with physical `tau`, temperature and simulation-domain metadata.

The two towers do not share temporal parameters. The spatial trunk receives a
small, separately scaled causal gradient; this prevents paired regression from
erasing the representation learned by the offline objective. A residual
adapter can be enabled after the 2k gate if the shared trunk still shows
negative transfer.

## Transition Model

The first V2.3 transition is a diagonal Gaussian residual operator:

```text
delta ~ Normal(mu(h_causal, log(tau), temperature, domain),
               sigma(h_causal, log(tau), temperature, domain))
z_future = z_context + delta
```

`mu` is zero-initialized, `log_sigma` is bounded, and the objective is a
proper Gaussian NLL plus a small Huber mean diagnostic. The deterministic
persistence model is scored with the same NLL convention. This is the minimum
probabilistic step needed to test calibration; a mixture or diffusion model is
not introduced until Gaussian calibration fails for a documented reason.

## Representation Preservation

Each paired training step has two controlled branches:

```text
L = L_offline_JEPA
  + 0.25 * L_causal_forecast
  + 0.10 * L_observable
  + regularization
```

The exact weights are selected on a 2k gate, not by the final test set. The
offline branch preserves the original residue/state JEPA signal. The causal
branch uses endpoint delta/state NLL. The optional observable head predicts
contact persistence, torsion change, radius-of-gyration change and local
distance statistics from the causal endpoint. These targets are computed
directly from coordinates and are not MSM labels.

MSM labels remain evaluation-only for the primary self-supervised run.

## Experiment Order

### V2.3-S: scratch fair track

1. 2k deterministic joint gate with dual temporal towers and no stochastic
   head; verify no leakage, finite gradients and nonzero pairs for `48/96/192`.
2. 2k Gaussian transition gate; compare NLL, calibration, coverage and paired
   skill against Gaussian persistence.
3. Continue only the best checkpoint selected by protein-stratified validation
   to 4k, then 6k. Stop on two consecutive validation regressions.
4. Run the full aligned champion comparison and the honest non-overlap
   scoreboard at the selected checkpoint.

### V2.3-A: champion-anchored engineering track

Initialize the offline tower from the Phase-1 champion, freeze or very slowly
update its representation for the first gate, and train the causal tower and
observable head. This track answers whether a useful world-model transition
can be added without destroying the strongest existing representation. It is
explicitly labeled warm-started and cannot replace the scratch promotion claim.

The anchored track is the fastest route to a practical model that can exceed
the formal baseline on forecast metrics, while V2.3-S remains the fair claim.

## Evaluation and Promotion Gates

Every selected checkpoint must report both protocols:

| Gate | Required result |
| --- | --- |
| Aligned representation | beat Phase-1 champion on at least two of VAMP, MSM balanced accuracy/accuracy, and linear R2 without a large third-metric regression |
| Honest representation | beat E0 and centered-CA/geometry references where applicable |
| Causal forecast | positive non-overlap R2 or positive skill against persistence at raw horizon 48, with protein-level bootstrap CI |
| Probabilistic forecast | NLL, CRPS/coverage and calibration better than Gaussian persistence |
| Rollout | no exploding variance, stable basin populations and improved transition/ITS error at 2-step rollout |
| Generalization | three seeds, unseen proteins, trajectory-heldout probes, no final test-set selection |

No V2.3 checkpoint is called a promoted world model from paired skill alone.
The first operational stopping rule is based on validation metrics: retain the
best 2k/4k/6k checkpoint and stop when representation or forecast gates regress
twice. This directly addresses the V2.2 2k-to-4k failure.

## Implementation Checklist

- [x] fix orientation/torsion masked-neighbor leakage;
- [x] add explicit endpoint and slow-endpoint evaluation readouts;
- [x] add dual offline/causal temporal towers and independent EMA teachers;
- [x] add non-overlap-only primary horizon sampler with physical-tau buckets;
- [x] add Gaussian residual transition and calibration metrics;
- [x] add coordinate-derived observable targets;
- [ ] run 2k gate before any 10k continuation;
- [ ] compare scratch and champion-anchored tracks under identical samples.

## Implemented V2.3 Entry Points

- Scratch deterministic gate: `configs/phase2_v2_3_s_deterministic_2k.yaml`
- Scratch Gaussian gate: `configs/phase2_v2_3_s_gaussian_2k.yaml`
- Champion-anchored Gaussian gate: `configs/phase2_v2_3_a_champion_anchored_2k.yaml`

The shared spatial trunk scales causal forecast gradients by `0.10`; the
offline representation branch receives its own optimizer scale and EMA
teacher. The anchored track remaps the Phase-1 temporal tower into the offline
tower while leaving the causal tower random, and runs the offline tower at
`0.05x` learning rate. This preserves the distinction between a scratch claim
and an engineering warm start.

The paired evaluator now reports deterministic persistence skill plus Gaussian
NLL, Gaussian CRPS, one- and two-sigma coverage, observable MAE, and the same
NLL/CRPS under a persistence mean with identical predicted variance. Paper
evaluation exposes `endpoint` as the offline tower and `causal_endpoint` as the
forecast tower, so the two contracts cannot be silently mixed.

## Post-2k Correction

The deterministic 2k audit exposed a state-space error in the paired contract.
The old persistence baseline compared the online context endpoint against an
EMA future endpoint. This rewards online/teacher coordinate alignment in
addition to physical prediction and inflated held-out skill from about
`0.10` canonical skill to about `0.93` reported skill.

The next implementation must define all deltas in one frozen or EMA teacher
space:

```text
delta_target = z_teacher(future) - z_teacher(context)
z_prediction = z_teacher(context) + delta_prediction(z_online(context), tau)
persistence = z_teacher(context)
```

It must also restore future-mask State-JEPA supervision to the offline tower.
Paired endpoint regression alone produced near-random VAMP and MSM accuracy
below the majority baseline. No Gaussian run is allowed on the old contract.

The first corrective experiment is V2.3.1-A. It freezes an exact copy of the
Phase-1 champion representation and trains a random causal tower to predict
canonical champion-state deltas. This guarantees that adding dynamics cannot
erase the representation baseline. A scratch claim still requires the later
joint future-mask State-JEPA track; V2.3.1-A remains explicitly anchored.
