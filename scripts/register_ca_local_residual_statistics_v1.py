"""Freeze fixed-cache decoder-probe estimands before its smoke or training."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from scripts import ca_local_residual_statistics_v1 as numeric

REPO = Path(__file__).resolve().parents[1]
BASE = REPO/'results/hs_joint_20261101/ca_local_residual_v1'
ARTIFACTS = REPO/'artifacts/hs_joint_20261101/ca_local_residual_v1'
SOURCES = ('scripts/ca_local_residual_statistics_v1.py',
           'scripts/register_ca_local_residual_statistics_v1.py',
           'tests/test_ca_local_residual_statistics_v1.py')
CACHES = {
    'results/hs_joint_20261101/backbone_repair/readout_train500_v2.pt':
        'd48faeb6e7385fa0b37a1daf440bf4708ed3472d83e2073b8a44d9073a216718',
    'results/hs_joint_20261101/backbone_repair/readout_val200_v2.pt':
        'd1a04f5861badc004f39d9496e3f245b6c32739a8d2deed5bb74840eaf605fdc',
}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def register():
    numeric.require(not ARTIFACTS.exists(), 'probe artifacts already exist before statistics registration')
    numeric.require(not BASE.exists() or not any(BASE.iterdir()), 'probe outputs already exist')
    for name, expected in CACHES.items():
        numeric.require(sha(REPO/name) == expected, 'fixed cache hash mismatch')
    value = dict(numeric.contract(), registered_utc=datetime.now(timezone.utc).isoformat(),
        registration_stage='Before first4 smoke, any optimization, or probe score output. '
                           'Historical validation diagnostics were already inspected; adaptive development, not unseen confirmation.',
        source_hashes={name: sha(REPO/name) for name in SOURCES}, input_hashes=CACHES,
        implementation='Only numerical helpers frozen here. Separate runner and provenance wrapper '
                       'will be registered before their use without changing these estimands.')
    BASE.mkdir(parents=True, exist_ok=True)
    output = BASE/'summary.protocol.json'
    with output.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(path=str(output.relative_to(REPO)), sha256=sha(output))))


if __name__ == '__main__': register()
