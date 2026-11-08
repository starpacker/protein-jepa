"""Register unchanged forecast estimands for the physical-distance target pair.

Registration reads source files and checks output existence only. Training,
checkpoint loads, score reads and evaluation are separate authorized actions.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts import hs_distance_pilot_statistics_v1 as numeric
from scripts import register_hs_ordered_pilot_statistics_v1 as ordered_registration

old = ordered_registration.old
BASE = Path('results/hs_joint_20261101/distance_path_v1')
ARTIFACTS = Path('artifacts/hs_joint_20261101/distance_path_v1')
VERSION = 'hs_distance_pilot_statistics_v1'
SOURCES = tuple(sorted(set((*ordered_registration.SOURCES,
    'scripts/hs_distance_pilot_statistics_v1.py',
    'scripts/register_hs_distance_pilot_statistics_v1.py',
    'tests/test_hs_distance_pilot_statistics_v1.py'))))


def contract():
    result = ordered_registration.contract()
    result.update(schema_version=VERSION, arms=list(numeric.ARMS),
        contrast='increment minus absolute',
        ordered_comparisons=[numeric.CONTRAST, 'increment_vs_persistence', 'absolute_vs_persistence'],
        suite_descriptive_means=1700,
        implementation='Explicit local arm slots call frozen ordered numerical helpers and rename only exact labels. '
            'All 402 intervals, 670 standard descriptive means, 1700 suite descriptive means, metric definitions, '
            'bootstrap draws and support/null policies are unchanged; no global monkeypatch.',
        training=dict(optimizer_steps=500, seed=42, n_samples=12, grad_steps=64,
            initial_checkpoint='artifacts/hs_joint_20261101/spatial_probe_v1/spatial_exact.pt',
            initial_checkpoint_sha256='396e6e4df35019140324484bf8d560a292cf2688a11946f2e0ccde5cd6ea8dda',
            coordinate='Both arms: 0.5 endpoint U-ES + 0.5 mean-time marginal U-ES',
            identical_future_labels_and_sampling_contract=True,
            only_arm_arguments=['distance_path_objective','output'],
            distance_path_weight=1., contact_estimator='plugin', step_ps=1000.,
            distance_targets=dict(absolute='d(t)', increment='signed d(t)-d(t-1)'),
            distance_score='At each future time, mean member-to-target Euclidean distance minus half the mean ordered off-diagonal member distance; divide by sqrt(pair count). Equal future-time means within window, then equal windows.',
            distance_pairs='All unique valid physical nonlocal CA pairs: different chains or same-chain absolute residue-index separation at least three. Identical fixed observed support and pair ordering in both arms; no target filtering, cap or fitted scaling.',
            distance_paths='One continuous observed-only M12 rollout shared by base and both auxiliaries. Same paths, targets and uniform 1ns time grid; both adjacent predicted frames receive increment gradients. Observed t0 is fixed.',
            remaining_terms='Endpoint latent .1 + plugin endpoint soft-contact Brier20 + endpoint clash40; all prior zero-weight terms stay zero.',
            frozen_encoder_decoder=True, frozen_projection_steps=32,
            optimizer='AdamW lr1e-4, weight_decay0.01, warmup50/cosine, clip_grad_norm1; unused parameters retain grad=None',
            kernel_parameter_tensors=73,
            selection='latest exactly500; no training-time validation, best selection, resume or restart',
            control='Fresh matched absolute-distance arm; historical no-auxiliary marginal_path results are supplementary, not an additional matched control.'),
        distance_target_limits=[
            'The auxiliary U-energy scores are proper for their respective conditional transformed marginal laws; their mean over times does not identify the full joint coordinate-path law.',
            'Internal distances do not identify orientation or chirality; signed distance increments are not Cartesian displacement vectors. Neither auxiliary alone establishes kinetics, equilibrium or Mori closure.',
            'Equal Angstrom units and equal fixed weight do not imply equal gradient scales or effective update sizes. No weight tuning, normalization fitting or extra comparison is introduced.',
            'This paired intervention tests absolute versus signed-increment targets, not the benefit of adding any auxiliary relative to no auxiliary.',
            'At h1 a common fixed observed anchor translates every member and target equally: scores and parameter gradients agree up to floating arithmetic. This does not apply to an independently trainable initial frame.',
            'Euclidean metric U-energy scores are nonnegative in exact arithmetic by the triangle inequality; no clamp is added. This differs from the potentially negative finite corrected contact Brier estimator.',
            'Increment supervision may increase jitter or harm endpoint distribution and geometry; retain all predeclared outcomes without adding diagnostic CIs or selecting favorable horizons.',
            'Finite nonzero gradients establish an operative intervention, not forecast benefit. The total loss includes geometry penalties and is not a pure proper scoring rule.',
            'Single-seed adaptive development comparison; all temperatures were seen in training, so correct/wrong/none is an input-mechanism test, not held-out-temperature generalization.'])
    result['scientific_limits'] = [
        text.replace('Target intervention', 'Distance-target intervention')
        for text in result['scientific_limits']
        if not text.startswith('Joint versus marginal training compares')]
    result['scientific_limits'].extend(result['distance_target_limits'])
    return result


def register():
    output = old.local(BASE/'summary.protocol.json')
    old.require(not output.exists(), 'refusing registration overwrite')
    for arm in numeric.ARMS:
        old.require(not old.local(BASE/f'{arm}.execution.json').exists()
                    and not old.local(ARTIFACTS/arm/'checkpoint_latest.pt').exists()
                    and not old.local(ARTIFACTS/arm/'run.json').exists(),
                    'training already started before statistics registration')
        for kind in ('ensemble','endpoint','ordered','suite','matched8'):
            old.require(not old.local(BASE/f'{arm}.{kind}.json').exists(),
                        'formal model score exists before statistics registration')
    inputs = old.Inputs()
    for source in SOURCES:
        inputs.track(source)
    inputs.verify()
    record = dict(contract(), registered_utc=datetime.now(timezone.utc).isoformat(),
        registration_stage='Before both new-arm training and formal evaluation scores. Prior development and TRAIN-only gradient diagnostics informed this adaptive intervention; it is not unseen confirmation.',
        source_hashes={old.label(p):digest for p,digest in inputs.hashes.items()},
        output=str(BASE/'summary.json'))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(path=str(output),sha256=old.sha(output))))
    return record


if __name__ == '__main__':
    register()
