# Baselines for Protein-JEPA-Frame

Three classical baseline methods for protein dynamics state decomposition,
compared against the V4-D model.

## Methods

### 1. TICA + MSM (deeptime)

Standard Markov State Model pipeline on aligned CA coordinates:
- TICA projection (lag=1, dim=10, kinetic_map scaling)
- KMeans clustering (48 microstates)
- Reversible Maximum Likelihood MSM (lag=1)
- PCCA+ coarse-graining to 6 metastable states

**Reference**: Pérez-Hernández et al. (2013) "Identification of slow molecular
order parameters for Markov model construction"

### 2. VAMPnets

Small MLP (256→128→64→6) mapping CA features to 6-state softmax membership
probabilities, trained to maximize the VAMP-2 score.

**Reference**: Mardt et al. (2018) "VAMPnets for deep learning of molecular
kinetics"

### 3. SPIB (State Predictive Information Bottleneck)

Same architecture as VAMPnets, but with a modified loss:
- VAMP-2 score (surrogate for mutual information between current and future
  state assignments)
- Entropy regularization (bottleneck, controlled by β)
- Loss = -VAMP-2 + β · H(z)

**Reference**: Wang et al. (2020) "State Predictive Information Bottleneck"

## Usage

```bash
# Run on subset for quick testing (5 proteins)
python -m baselines.run_all --subset 5

# Run full validation set
python -m baselines.run_all

# Run specific methods with custom parameters
python -m baselines.run_all --methods tica_msm vampnet
python -m baselines.run_all --beta 0.5 --max-epochs 300
```

## Data

All baselines use the same mdCATH CA features:
- Extract CA coordinates from HDF5
- Kabsch-align to first frame's CA
- Flatten to (F, R×3) per trajectory
- Per-protein: find common valid CA residues across all trajectories

## Metrics

| Metric | Description |
|--------|-------------|
| VAMP-2 score | Per-trajectory VAMP-2 at lags 1, 2, 5, 10, 20 |
| Implied timescales | TICA timescales at lags 2, 5, 10, 20 |
| State accuracy | Top-1, top-2, balanced accuracy vs MSM labels |
| Persistence accuracy | Baseline: predict current state = previous state |
| Brier skill score | 1 - (Brier_model / Brier_persistence) |
| MSM probe accuracy | Linear probe from features to MSM labels |

## Results

Results are saved to `baselines/results/`:
- `tica_msm/tica_msm_results.json`
- `vampnet/vampnet_results.json`
- `spib/spib_results.json`
- `combined_results.json` (all methods in one file)

## Comparison with V4-D

V4-D results are documented in `docs/results/`. The key comparison points:
- V4-D uses learned latent representations; baselines use raw CA coordinates
- V4-D produces soft state probabilities; TICA+MSM produces hard labels
- VAMPnets and SPIB produce soft probabilities like V4-D