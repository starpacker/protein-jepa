# Protein dynamics evaluation

The primary endpoint error applies **one real anchor→target Kabsch transform** to
all predictions. Independent prediction→target Kabsch RMSD is a secondary shape
metric. The target never enters a forecast producer; real future H is encoded
separately for oracle controls.

Before scoring, a canonical decoder pose is calibrated once using only the
decoded current H and the observed last history frame. That history-only rigid
transform is then fixed for all predictions, oracle controls and horizons. It
removes arbitrary decoder gauge without fitting any prediction to a future target.

New results require a self-contained schema-v2 representation: encoder, raw-H
oracle decoder, fitted train-only normalizer, encoder/decoder configuration, and
content hashes. H+S uses normalized H internally and inverts normalization once
before the shared raw-H decoder. Direct predictors consume raw history H and must
have a matching encoder hash and a verified `delta_factors` output contract.
Changing the encoder invalidates an old predictor.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m evaluation.run_eval \
  --config evaluation/configs/smoke.yaml \
  --b1a-ckpt path/to/schema_v2_b1a.pt \
  --b1b-ckpt path/to/schema_v2_b1b.pt \
  --stage-a-ckpt path/to/matched_stage_a.pt \
  --stage-a-predictor-ckpt path/to/matched_stage_a.pt \
  --train-manifest path/to/train.jsonl \
  --manifest-override path/to/validation.jsonl \
  --outdir artifacts/eval/new_smoke
```

Config checkpoint paths are explicit examples for the repaired pilot; these files
are created by training. Missing or incompatible checkpoints fail. A contract
smoke checkpoint is always diagnostic, even if its tensor schema is valid. Formal
autonomous checkpoints must carry their passed oracle gate. Validation remains a
pilot/model-selection set; a final test set must remain untouched.

Window budgets cap **distinct proteins**, optionally trajectories per protein and
anchors per trajectory. Raw-frame horizons and physical ps come from the shared
data protocol. Duplicate physical horizons from lag aliases are scored once.
Reader errors, invalid masks, incompatible physical dt and nonfinite forecasts
fail rather than silently removing difficult windows.

`--split protein_holdout` checks distinct-protein disjointness against every
`--train-manifest`. Filtering to 450 K is `temperature_subset`; it is not a
`temperature_holdout` if any declared training manifest includes 450 K. Actual
temperature exclusion must hold for every learned module from Stage A onward.
Declared manifests and checkpoint/source hashes are recorded for audit; declarations
must represent the actual training lineage.

Outputs:

- `per_window.jsonl`: complete protein/T/trajectory/anchor/target/horizon keys,
  ensemble-mean primary and shape errors, each variant's own oracle and anchor-H
  controls, sample scores, latent motion and physical distance energy score.
- `samples.jsonl`: one row per exact window/variant/model replicate. No best-of-M
  selection. Ensemble coordinate means are scored separately and are not claimed
  to be physical conformations.
- `per_protein.jsonl`: equal-window means within each protein/T/horizon/variant.
- `results.json` and `results.md`: equal-protein summaries, SEM and paired protein
  bootstrap. Single-protein runs cannot establish significance. Persistence
  compared with itself has identically zero paired differences.
- `meta.json`: effective config, source tree hash and git dirty status, checkpoint
  hashes/contracts, manifest/split audit, seed, sample count, physical-score scale,
  library/device versions and metric definition.

The physical endpoint score is the energy score of the **empirical nonlocal CA
pair-distance distribution**, scaled by a fixed Angstrom constant. It includes
its V-statistic diagonal and does not normalize using target values. This is a
proper observable-space forecast score, not a proof of full path dynamics. Report
RMSF, ACF, path distributions and geometry validity separately for long rollouts.
An oracle-normalized skill is absent/undefined when the oracle does not beat
persistence; paired errors remain meaningful.

Legacy checkpoints are rejected by default. `--allow-legacy` is an explicit
**diagnostic** adapter: legacy H+S additionally requires `--legacy-encoder-ckpt`
matching its training source, and direct legacy prediction requires an explicit
`--legacy-predictor-output absolute_factors|delta_factors`. It never infers old
semantics from a mutable flag. Legacy/mock results use `diagnostic_aggregate`
instead of `aggregate` and cannot become headline evidence.

Kalman likelihood is a separate diagnostic, not the autonomous training objective
or a scientific forecast gate:

```bash
python -m evaluation.run_gate --ckpt path/to/schema_v2_b1b.pt \
  --manifest path/to/validation.jsonl --train-manifest path/to/train.jsonl \
  --split protein_holdout --horizon 4 --device cpu \
  --out artifacts/eval/kalman_diagnostic.json
```

It scores the observed **continuous** anchor→1→2→3→4 path, checks every real dt
against the kernel, and normalizes every ablation by actual transitions×valid
residues×latent dimensions. It explicitly labels its teacher-forced conditional
likelihood as diagnostic only.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  python -m pytest evaluation/tests -q
```

The tests include independent geometry controls, deliberate target-leak rejection,
exact pairing, protein weighting, NaN-pair handling, actual tiny schema bundles,
legacy rejection, complete-history producer contracts, and byte-identical CPU
mock end-to-end reruns. Mock output tests the instrument, not model quality.
