"""Register unchanged forecast estimands for the contact-estimator training pair.

Registration reads source files and checks output existence only. Training,
checkpoint loads, score reads and evaluation are separate authorized actions.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts import hs_contact_pilot_statistics_v1 as numeric
from scripts import register_hs_ordered_pilot_statistics_v1 as ordered_registration

old = ordered_registration.old
BASE = Path('results/hs_joint_20261101/contact_estimator_v1')
ARTIFACTS = Path('artifacts/hs_joint_20261101/contact_estimator_v1')
VERSION = 'hs_contact_pilot_statistics_v1'
SOURCES = tuple(sorted(set((*ordered_registration.SOURCES,
    'scripts/hs_contact_pilot_statistics_v1.py',
    'scripts/register_hs_contact_pilot_statistics_v1.py',
    'tests/test_hs_contact_pilot_statistics_v1.py'))))


def contract():
    result = ordered_registration.contract()
    result.update(schema_version=VERSION, arms=list(numeric.ARMS),
        contrast='population_u minus plugin',
        ordered_comparisons=[numeric.CONTRAST, 'population_u_vs_persistence', 'plugin_vs_persistence'],
        suite_descriptive_means=1700,
        implementation='Explicit local arm slots call frozen ordered numerical helpers and rename only exact labels. '
            'All 402 intervals, 670 standard descriptive means, 1700 suite descriptive means, metric definitions, '
            'bootstrap draws and support/null policies are unchanged; no global monkeypatch.',
        training=dict(optimizer_steps=500, seed=42, n_samples=12, grad_steps=64,
            coordinate='Both arms: 0.5 endpoint U-ES + 0.5 mean-time marginal U-ES',
            identical_future_labels_and_sampling_contract=True,
            only_arm_arguments=['contact_estimator','output'],
            contact_estimators=dict(plugin='mean_pairs((mean_members(z)-q)^2)',
                population_u='plugin minus mean_pairs(sample_variance(z,unbiased=True))/12'),
            contact_soft_transform='z=sigmoid((8-distance_angstrom)/0.5); same endpoint trainer pair mask',
            remaining_terms='Endpoint latent .1 + selected endpoint soft-contact estimator20 + endpoint clash40; all prior zero-weight terms stay zero.',
            frozen_encoder_decoder=True, frozen_projection_steps=32,
            optimizer='AdamW lr1e-4, weight_decay0.01, warmup50/cosine, clip_grad_norm1; unused parameters retain grad=None',
            kernel_parameter_tensors=73,
            selection='latest exactly500; no training-time validation, best selection, resume or restart',
            control='Fresh matched plugin arm; historical marginal_path results are supplementary, not the new control.'),
        contact_estimator_limits=[
            'The variance correction targets squared error of each conditional mean soft contact under iid ensemble members; contacts within a member may be correlated.',
            'Distinct soft-contact laws with the same mean have the same expected corrected contact term; coordinate U-energy supplies additional distributional supervision.',
            'Finite corrected losses can be negative and are not clamped. Empirical Brier evaluation remains unchanged and is not an implementation error.',
            'The total objective includes a clash penalty and is not a pure proper scoring rule.',
            'Small nonzero parameter-gradient differences establish an operative intervention, not forecast improvement or its principal cause.',
            'Single-seed adaptive development comparison; all temperatures were seen in training, so correct/wrong/none is an input-mechanism test, not held-out-temperature generalization.'])
    result['scientific_limits'] = [
        text.replace('Target intervention', 'Contact-estimator intervention')
        for text in result['scientific_limits']
        if not text.startswith('Joint versus marginal training compares')]
    result['scientific_limits'].extend(result['contact_estimator_limits'])
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
