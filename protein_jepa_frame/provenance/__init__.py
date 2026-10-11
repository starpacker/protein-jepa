"""Provenance tracking for reproducible experiments.

Gate 0/Gate 11 canonical implementation:
- RunManifest: single source of truth for experiment provenance
- ManifestManager: create, save, validate run manifests
- Audit trail: full chain from data to checkpoint to results
"""

from protein_jepa_frame.provenance.run_manifest import (
    RunManifest,
    ManifestManager,
    build_run_lineage_chain,
)

__all__ = [
    "RunManifest",
    "ManifestManager",
    "build_run_lineage_chain",
]