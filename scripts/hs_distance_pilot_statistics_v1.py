"""Distance-target labels around frozen ordered-pilot numerical estimands.

All inputs are already audited scalar records. Explicit local positional slots
preserve bootstrap, weighting, support and null policies; frozen module globals
are never modified. No data, model, score file or checkpoint is read here.
"""
from scripts import hs_ordered_pilot_statistics_v1 as ordered

ARMS = ('increment', 'absolute')
CONTRAST = 'increment_vs_absolute'
SCORES = ordered.SCORES
paths, require = ordered.paths, ordered.require
descriptive, scalar = ordered.descriptive, ordered.scalar
_SLOTS = dict(zip(ARMS, ordered.ARMS))
_LABELS = {value:key for key,value in _SLOTS.items()}
_LABELS.update({ordered.CONTRAST:CONTRAST,
               **{old+'_vs_persistence':new+'_vs_persistence' for new,old in _SLOTS.items()}})


def ordered_slots(values):
    require(set(values) == set(ARMS), 'distance objective arms differ')
    return {_SLOTS[arm]:values[arm] for arm in ARMS}


def rename_ordered(value):
    """Rename exact labels only, preserving numerical values and input objects."""
    if isinstance(value, dict):
        renamed = {_LABELS.get(key,key):rename_ordered(item) for key,item in value.items()}
        require(len(renamed) == len(value), 'label mapping collision')
        return renamed
    if isinstance(value, list):
        return [rename_ordered(item) for item in value]
    if isinstance(value, tuple):
        return tuple(rename_ordered(item) for item in value)
    return _LABELS.get(value,value) if isinstance(value,str) else value


def standard_analysis(ensemble, endpoints, baseline):
    return rename_ordered(ordered.standard_analysis(
        ordered_slots(ensemble), ordered_slots(endpoints), baseline))


def observed_reference_pairing(rows):
    return ordered.observed_reference_pairing(ordered_slots(rows))


def ordered_analysis(rows):
    return rename_ordered(ordered.ordered_analysis(ordered_slots(rows)))
