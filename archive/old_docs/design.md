# Protein Dynamics V-JEPA Migration

## Mapping From V-JEPA To Protein Dynamics

| V-JEPA | Protein dynamics |
| --- | --- |
| video clip `[T, H, W]` | MD clip `[T, L]`, where `L` is residue count |
| image/video patch | residue at one trajectory frame |
| 3D multiblock mask | contiguous time/residue block mask |
| ViT encoder | protein geometry encoder or ProteinMPNN/ESM adapter |
| target encoder EMA | same mechanism, no-grad full-structure branch |
| predictor mask tokens | learned query tokens for masked `(time, residue)` positions |
| feature prediction loss | latent loss on masked residue tokens |

## Non-Negotiable Leakage Rule

Masking happens before context featurization. For the context branch, masked residues are removed from the valid graph for that frame, so visible nodes cannot use masked coordinates through KNN selection, edge features, or message passing. The target branch may encode the full valid structure because it is no-grad and only supplies latent targets.

## Training Objective

1. sample a trajectory clip;
2. sample target mask `M` over `[T, L]`;
3. encode full clip with EMA target encoder and collect normalized target latent on `M`;
4. encode visible-only clip with context encoder;
5. predictor receives visible latent plus learned target mask queries and predicts latent at `M`;
6. optimize Smooth L1/L1 latent prediction loss and optional variance regularization;
7. update target encoder with EMA.

## Role Of MSM Labels

MSM labels are not training labels in this framework. They can be used for:

- choosing hard transition clips;
- QC and per-protein split auditing;
- frozen linear probes after representation learning;
- VAMP/TICA/transition evaluation.

Keeping MSM out of the training input prevents the model from becoming a supervised state classifier and makes the learned representation testable.
