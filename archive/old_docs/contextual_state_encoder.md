# Contextual State Encoder Requirement

Protein conformations are not Markov states when observed as isolated frames.
The same geometry can appear on folding and unfolding paths, but the future
distribution is different. The learned state must therefore encode temporal
context, not only the instantaneous structure.

## Required Semantics

For a frame `t`, the state latent should be interpreted as:

```text
z_t = state(geometry around t, temporal context around t)
```

not:

```text
z_t = state(single-frame geometry at t)
```

This applies to both branches:

- input/context states: online temporal encoder over the visible context window.
- target states: EMA temporal target encoder over the target/reference window.

The current state-JEPA path follows this direction: residue latents are first
computed per frame, then `TemporalStateEncoder` produces frame-level contextual
states, and JEPA losses are applied to these state tokens.

## FastFolding Diagnostic

FastFolding should include direction-aware probes:

- folding vs unfolding direction from `dQ_native/dt`.
- committor proxy from future folded/unfolded basin arrival.
- retrieval where neighbors must agree in dynamics context, not only RMSD/Q.

If two frames have similar RMSD and Q but opposite direction labels, a useful
world-model state encoder should separate them.

## Next Model Extensions

- use longer windows for state readout on FastFolding.
- keep long-horizon and multi-lag state prediction enabled.
- add hard negatives matched by geometry but opposite direction labels for
  downstream analysis, not as supervised pretraining labels by default.
