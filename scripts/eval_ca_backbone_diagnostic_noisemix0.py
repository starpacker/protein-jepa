"""noise_mix=0 reconstruction wrapper for the CA-only backbone diagnostic (E3.1).

The exact/prior/gru S_0-arm checkpoints record ``args.noise_mix=0.8`` (stale
launch metadata; the launch scripts actually pass ``--noise-mix 0``), and the
``s0_mode='exact'`` Kalman posterior requires identity spatial noise
covariance. ``scripts/eval_ca_backbone_diagnostic.py`` reconstructs
``CADynamics`` through ``eval_ca_endpoint_scores.load_model``, which re-reads
the stale ``args.noise_mix=0.8`` and crashes on the identity guard in
``model.py``.

This wrapper imports the diagnostic module verbatim and substitutes a
``load_model`` that reconstructs ``CADynamics`` with ``noise_mix=0.0``, matching
the actual training launch. All scoring, lifting, geometry, provenance and
versioning paths are unchanged. Only the val manifest is read; test.jsonl is
never opened.
"""
from __future__ import annotations

import torch

import scripts.eval_ca_backbone_diagnostic as D
from protein_jepa_frame.training.train_lasp_ca_dynamics import CADynamics
from protein_jepa_frame.utils.contracts import state_sha256


def _load_model_noisemix0(path, checkpoint, device):
    if checkpoint.get('checkpoint_kind') != 'lasp_ca_dynamics' or 'component_checkpoint' not in checkpoint:
        raise ValueError('requires lasp_ca_dynamics with an embedded CA component')
    args = checkpoint['args']
    model = CADynamics(path, device, path,
        history_conditioning=args.get('history_conditioning', False),
        noise_mix=0.0, noise_length=args.get('noise_length', 8.),
        collective_mobility=args.get('collective_mobility', False),
        noise_kernel=args.get('noise_kernel'), noise_cutoff=args.get('noise_cutoff'),
        s0_mode=args.get('s0_mode', 'gru')).eval()
    model.prediction_kind = 'endpoint'
    if state_sha256(model.kernel) != state_sha256(checkpoint['kernel']):
        raise ValueError('loaded kernel differs from checkpoint')
    if state_sha256(model.bundle.encoder) != state_sha256(checkpoint['component_checkpoint']['encoder']):
        raise ValueError('loaded encoder differs from checkpoint')
    component = checkpoint['component_checkpoint']
    decoder_state = component.get('delta_ca_decoder', component.get('decoder'))
    if decoder_state is None or state_sha256(model.bundle.decoder) != state_sha256(decoder_state):
        raise ValueError('loaded decoder differs from embedded component')
    return model


# Substitute the stale-metadata loader for the noise_mix=0 reconstruction.
D.load_model = _load_model_noisemix0


def main():
    D.main()


if __name__ == '__main__':
    main()
