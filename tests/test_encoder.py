import torch

from protein_jepa_frame.models.encoder import ProteinGeometryEncoder


def _batch():
    B, T, L = 2, 3, 12
    coords = torch.randn(B, T, L, 4, 3)
    coords[..., 1, :] += torch.arange(L).view(1, 1, L, 1)
    residue_mask = torch.ones(B, T, L, dtype=torch.bool)
    aatype = torch.full((B, L), 20, dtype=torch.long)
    residue_index = torch.arange(L).repeat(B, 1)
    chain_id = torch.zeros(B, L, dtype=torch.long)
    return coords, residue_mask, aatype, residue_index, chain_id


def test_encoder_is_rotation_translation_invariant():
    encoder = ProteinGeometryEncoder(latent_dim=32, depth=2, num_heads=4, num_rbf=8, dropout=0.0)
    encoder.eval()
    coords, residue_mask, aatype, residue_index, chain_id = _batch()
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    shifted = coords @ q + torch.tensor([10.0, -2.0, 4.0])

    with torch.no_grad():
        out_a = encoder(coords, residue_mask, aatype, residue_index, chain_id)
        out_b = encoder(shifted, residue_mask, aatype, residue_index, chain_id)

    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_context_encoder_does_not_depend_on_masked_coordinates():
    encoder = ProteinGeometryEncoder(latent_dim=32, depth=2, num_heads=4, num_rbf=8, dropout=0.0)
    encoder.eval()
    coords, residue_mask, aatype, residue_index, chain_id = _batch()
    token_mask = residue_mask.clone()
    token_mask[:, :, 4:8] = False
    perturbed = coords.clone()
    perturbed[:, :, 4:8] += 1000.0 * torch.randn_like(perturbed[:, :, 4:8])

    with torch.no_grad():
        out_a = encoder(coords, residue_mask, aatype, residue_index, chain_id, token_mask=token_mask)
        out_b = encoder(perturbed, residue_mask, aatype, residue_index, chain_id, token_mask=token_mask)

    assert torch.allclose(out_a[token_mask], out_b[token_mask], atol=1e-5)


def test_orientation_encoder_distinguishes_mirror_images():
    encoder = ProteinGeometryEncoder(
        latent_dim=32, depth=2, num_heads=4, num_rbf=8, dropout=0.0, orientation_features=True
    )
    encoder.eval()
    coords, residue_mask, aatype, residue_index, chain_id = _batch()
    mirrored = coords.clone()
    mirrored[..., 0] *= -1.0

    with torch.no_grad():
        original = encoder(coords, residue_mask, aatype, residue_index, chain_id)
        reflected = encoder(mirrored, residue_mask, aatype, residue_index, chain_id)

    assert not torch.allclose(original, reflected, atol=1e-5)


def test_orientation_context_encoder_does_not_read_masked_neighbors():
    encoder = ProteinGeometryEncoder(
        latent_dim=32, depth=2, num_heads=4, num_rbf=8, dropout=0.0, orientation_features=True
    )
    encoder.eval()
    coords, residue_mask, aatype, residue_index, chain_id = _batch()
    token_mask = residue_mask.clone()
    token_mask[:, :, 4:8] = False
    perturbed = coords.clone()
    perturbed[:, :, 4:8] += 1000.0 * torch.randn_like(perturbed[:, :, 4:8])

    with torch.no_grad():
        out_a = encoder(coords, residue_mask, aatype, residue_index, chain_id, token_mask=token_mask)
        out_b = encoder(perturbed, residue_mask, aatype, residue_index, chain_id, token_mask=token_mask)

    assert torch.allclose(out_a[token_mask], out_b[token_mask], atol=1e-5)
