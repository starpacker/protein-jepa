"""CPU contracts for the separate fixed-CA, corrected-chemical backbone head."""
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from protein_jepa_frame.models.backbone_readout_v2 import (
    BackboneReadoutConfig, BackboneReadoutV2, HEAD_VERSION,
)
from protein_jepa_frame.models.stage_b1a.model import StageBOracleConfig, StageBOracleDecoder
from protein_jepa_frame.models.stage_c.geometry import rotation_matrix_from_quaternion
from protein_jepa_frame.models.stage_c.structure_decoder import (
    DecoderConfig, StructureDecoder, _torsion_logits_to_angles, reconstruct_atom37_from_frames,
)


LEGACY = Path(__file__).resolve().parents[1] / "artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt"
LEGACY_SHA256 = "5c02ad80b155edd853d788798d8512e757764f09f2fb9e8e97db06ccc2d88b0a"


def inputs(dim=8, length=6, batch=2, dtype=torch.float32):
    g = torch.Generator().manual_seed(241)
    H = torch.randn(batch, length, dim, generator=g, dtype=dtype)
    R = rotation_matrix_from_quaternion(torch.randn(batch, length, 4, generator=g, dtype=dtype))
    CA = 3.0 * torch.randn(batch, length, 3, generator=g, dtype=dtype)
    embedding = torch.arange(length).repeat(batch, 1) % 21
    # Explicit historical dataset -> OpenFold order; no inferred His from UNK.
    lookup = torch.tensor([0, 4, 3, 6, 13, 7, 8, 9, 11, 10, 12, 2, 14, 5, 1, 15, 16, 19, 17, 18, 20])
    chemical = lookup[embedding]
    mask = torch.ones(batch, length, dtype=torch.bool)
    return H, R, CA, embedding, chemical, mask


@pytest.fixture
def legacy():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(72)
        result = StructureDecoder(DecoderConfig(q_dim=8, c_hidden=16, n_heads=4,
            n_ipa_blocks=2, torsion_bins=12, dropout=0, global_h_context=True, anchor_free=False))
        # Nonzero learned IPA outputs, analogous to a trained legacy checkpoint.
        for block in result.ipa_blocks:
            torch.nn.init.normal_(block.linear_out.weight, std=.05)
        torch.nn.init.normal_(result.global_h_proj[-1].weight, std=.05)
        return result.eval()


def manual_corrected_fixed_ca(source, args):
    """Independent old-module composition with corrected chemistry/fixed CA."""
    H, R0, CA, embedding, chemical, mask = args
    s = source.embed_inputs(embedding, H, mask)
    for block, norm, transition in zip(source.ipa_blocks, source.layer_norms, source.transitions):
        residual, _ = block(s, R0, CA, mask)
        s = norm(s + source.dropout(residual))
        s = s + transition(s)
    R = R0 @ rotation_matrix_from_quaternion(source.frame_update(s))
    logits = source.torsion_head(s)
    torsions = _torsion_logits_to_angles(logits)
    atoms, atom_mask = reconstruct_atom37_from_frames(R, CA, chemical, torsions)
    backbone = torch.stack([atoms[..., 0, :], CA, atoms[..., 2, :], atoms[..., 4, :]], -2)
    return backbone, R, logits[..., 1, :], atom_mask[..., [0, 1, 2, 4]]


def test_initialization_matches_corrected_fixed_ca_control_and_does_not_share(legacy):
    before = {k: v.clone() for k, v in legacy.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    head = BackboneReadoutV2.from_legacy_decoder(legacy)
    assert torch.equal(rng, torch.random.get_rng_state())
    args = inputs()
    expected, R, psi_logits, atom_mask = manual_corrected_fixed_ca(legacy, args)
    result = head(*args)
    assert torch.allclose(result['backbone'], expected, atol=2e-6, rtol=1e-6)
    assert torch.equal(result['R'], R)
    assert torch.allclose(result['psi_logits'], psi_logits, atol=1e-7, rtol=1e-6)
    assert torch.equal(result['atom_mask'], atom_mask)
    assert torch.count_nonzero(result['torsions'][..., [0, 2, 3, 4, 5, 6]]) == 0
    for name, value in legacy.state_dict().items():
        assert torch.equal(value, before[name])
    assert head.aatype_embed.weight.data_ptr() != legacy.aatype_embed.weight.data_ptr()
    assert torch.equal(head.aatype_embed.weight, legacy.aatype_embed.weight)
    assert not any(any(word in name for word in ('position_head', 'normalizer', 'probe', 'torsion_head'))
                   for name, _ in head.named_parameters())
    frozen = {name for name, p in head.named_parameters() if not p.requires_grad}
    assert frozen == {f'ipa_blocks.{i}.{part}.bias' for i in range(2) for part in ('linear_k', 'linear_b')}


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_fixed_ca_and_rigid_equivariance(legacy, dtype):
    head = BackboneReadoutV2.from_legacy_decoder(legacy).to(dtype=dtype)
    args = inputs(dtype=dtype)
    output = head(*args)
    assert torch.equal(output['backbone'][..., 1, :], args[2])
    assert torch.equal(output['t'], args[2])
    G = rotation_matrix_from_quaternion(torch.tensor([.31, -.2, .53, .8], dtype=dtype))
    shift = torch.tensor([7.35, -1.29, 4.36], dtype=dtype)
    transformed = list(args)
    transformed[1] = G @ args[1]
    transformed[2] = args[2] @ G.T + shift
    moved = head(*transformed)
    assert torch.equal(moved['backbone'][..., 1, :], transformed[2])
    assert torch.allclose(moved['backbone'], output['backbone'] @ G.T + shift, atol=1e-5, rtol=1e-6)
    assert torch.allclose(moved['R'], G @ output['R'], atol=2e-6, rtol=1e-6)
    assert torch.allclose(moved['psi_logits'], output['psi_logits'], atol=2e-6, rtol=1e-6)


def test_padding_is_ignored_including_nan_and_empty_batch_member(legacy):
    head = BackboneReadoutV2.from_legacy_decoder(legacy)
    plain = inputs(batch=1, length=4)
    expected = head(*plain)
    padded = list(inputs(batch=2, length=7))
    padded[5].zero_()
    padded[5][0, :4] = True
    for index in range(5):
        padded[index].fill_(float('nan') if index < 3 else -99)
        padded[index][0, :4] = plain[index][0]
    padded[0].requires_grad_()
    output = head(*padded)
    for key in output:
        assert torch.isfinite(output[key]).all()
        assert torch.allclose(output[key][0, :4], expected[key][0], atol=2e-6, rtol=1e-6)
    invalid = ~padded[5]
    for key in ('backbone', 't', 'torsions', 'psi_logits', 'atom_mask'):
        assert torch.count_nonzero(output[key][invalid]) == 0
    assert torch.equal(output['R'][invalid], torch.eye(3).expand(int(invalid.sum()), 3, 3))
    output['backbone'].square().sum().backward()
    assert torch.isfinite(padded[0].grad).all()
    assert torch.count_nonzero(padded[0].grad[invalid]) == 0


def test_chemical_ids_affect_only_atom_producer_and_unknown_is_backbone_safe(legacy):
    head = BackboneReadoutV2.from_legacy_decoder(legacy)
    args = list(inputs())
    args[4] = torch.zeros_like(args[4])  # ALA
    alanine = head(*args)
    args[4] = torch.full_like(args[4], 14)  # PRO, not dataset row 14=R
    proline = head(*args)
    assert torch.equal(alanine['R'], proline['R'])
    assert torch.equal(alanine['psi_logits'], proline['psi_logits'])
    assert not torch.allclose(alanine['backbone'], proline['backbone'])
    args[4] = torch.full_like(args[4], 7)
    glycine = head(*args)
    args[4] = torch.full_like(args[4], 20)
    unknown = head(*args)
    assert torch.equal(unknown['backbone'], glycine['backbone'])
    assert unknown['atom_mask'].all()


def test_actual_atom_loss_reaches_every_trainable_parameter_and_ca_cannot_train_head(legacy):
    head = BackboneReadoutV2.from_legacy_decoder(legacy)
    args = inputs()
    output = head(*args)
    target = torch.randn(output['backbone'].shape, generator=torch.Generator().manual_seed(19))
    (output['backbone'] - target).square().mean().backward()
    for name, parameter in head.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
        else:
            assert parameter.grad is None, name
    assert head.psi_head[2].out_features == head.config.torsion_bins
    head.zero_grad(set_to_none=True)
    output = head(*args)
    # The tensor retains the common stack graph, but its CA slice has zero
    # derivative with respect to all model parameters.
    output['backbone'][..., 1, :].square().sum().backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in head.parameters())


def test_head_only_zero_update_checkpoint_roundtrip(legacy, tmp_path):
    head = BackboneReadoutV2.from_legacy_decoder(legacy)
    payload = head.checkpoint_dict()
    payload['step'] = 0
    payload['control'] = 'corrected_chemical_fixed_ca_zero_update'
    path = tmp_path / 'corrected_chemical_zero_update.pt'
    with path.open('xb') as stream:
        torch.save(payload, stream)
    loaded = torch.load(path, weights_only=False, map_location='cpu')
    assert loaded['head_version'] == HEAD_VERSION
    restored = BackboneReadoutV2(BackboneReadoutConfig(**loaded['head_config'])).eval()
    restored.load_state_dict(loaded['head'], strict=True)
    for name, value in head(*inputs()).items():
        assert torch.equal(value, restored(*inputs())[name])


def test_strict_trusted_checkpoint_loading_and_hash_failure(tmp_path):
    config = StageBOracleConfig(latent_dim=8, aux_dim=8, decoder_c_hidden=16,
        decoder_n_heads=4, decoder_n_ipa_blocks=2, decoder_torsion_bins=12,
        decoder_dropout=0, decoder_anchor_free=False)
    oracle = StageBOracleDecoder(config)
    path = tmp_path / 'legacy.pt'
    payload = {'oracle': oracle.state_dict(), 'stage_b1a_config': asdict(config), 'step': 23}
    torch.save(payload, path)
    head = BackboneReadoutV2.from_legacy_checkpoint(path)
    assert head.initialization['legacy_step'] == 23
    assert head.initialization['legacy_state_key'] == 'oracle'
    with pytest.raises(ValueError, match='SHA256'):
        BackboneReadoutV2.from_legacy_checkpoint(path, expected_sha256='0' * 64)
    probe_key = next(k for k in payload['oracle'] if k.startswith('probe.'))
    payload['oracle'].pop(probe_key)
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match='Missing key'):
        BackboneReadoutV2.from_legacy_checkpoint(path)


def test_reject_incompatible_source_and_invalid_valid_inputs(legacy):
    source = StructureDecoder(DecoderConfig(q_dim=8, c_hidden=16, n_ipa_blocks=1, anchor_free=True))
    with pytest.raises(ValueError, match='anchored'):
        BackboneReadoutV2.from_legacy_decoder(source)
    head = BackboneReadoutV2.from_legacy_decoder(legacy)
    bad = list(inputs())
    bad[4][0, 0] = 21
    with pytest.raises(ValueError, match='chemical_aatype outside'):
        head(*bad)
    bad = list(inputs())
    bad[0][0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite valid raw_H'):
        head(*bad)
    bad = list(inputs())
    bad[5] = bad[5].float()
    with pytest.raises(ValueError, match='residue_mask must be bool'):
        head(*bad)


@pytest.mark.skipif(not LEGACY.exists(), reason='optional local frozen B1a checkpoint unavailable')
def test_real_legacy_checkpoint_init_and_actual_atom_gradients():
    head = BackboneReadoutV2.from_legacy_checkpoint(LEGACY, expected_sha256=LEGACY_SHA256)
    assert head.initialization['legacy_step'] == 6000
    assert head.config.raw_h_dim == head.config.c_hidden == 128
    assert head.config.n_ipa_blocks == 4
    args = inputs(dim=128, length=21)
    output = head(*args)
    assert torch.equal(output['backbone'][..., 1, :], args[2])
    target = torch.randn(output['backbone'].shape, generator=torch.Generator().manual_seed(74))
    (output['backbone'] - target).square().mean().backward()
    for name, parameter in head.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
    legacy_payload = torch.load(LEGACY, weights_only=False, map_location='cpu')
    for name in ('aatype_embed.weight', 'state_proj.weight', 'frame_update.quat_net.2.weight'):
        assert torch.equal(head.state_dict()[name], legacy_payload['oracle']['decoder.' + name])
    n = head.config.torsion_bins
    assert torch.equal(head.psi_head[2].weight, legacy_payload['oracle']['decoder.torsion_head.net.2.weight'][n:2*n])
