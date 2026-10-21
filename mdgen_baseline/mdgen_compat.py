"""Compatibility shims to run the released MDGen code under our modern stack
(numpy 2.x, torch 2.8, pytorch_lightning 2.6).  NEW file; MDGen source is
imported read-only and never edited.

Import this module *before* importing mdgen.* to apply the patches:

    import mdgen_baseline.mdgen_compat  # noqa: F401

Patches:
  1. ``mdgen.tensor_utils.batched_gather`` -- numpy>=1.24 rejects ``data[list]``
     with a mixed list of tensors/slices (ragged).  Wrap the index in a tuple.
  2. Allow ``argparse.Namespace`` when torch.load'ing the .ckpt (torch>=2.6
     defaults ``weights_only=True``), and force ``weights_only=False`` for the
     Lightning checkpoint load.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch


def _patch_batched_gather():
    import mdgen.tensor_utils as tu

    def batched_gather(data, inds, dim=0, no_batch_dims=0):
        ranges = []
        for i, s in enumerate(data.shape[:no_batch_dims]):
            r = torch.arange(s)
            r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
            ranges.append(r)
        remaining_dims = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
        remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
        ranges.extend(remaining_dims)
        return data[tuple(ranges)]          # <- tuple() is the fix

    tu.batched_gather = batched_gather
    # geometry.py did ``from .tensor_utils import batched_gather`` -> the name is
    # bound in its own namespace, so patch it there explicitly.
    import mdgen.geometry as geo
    geo.batched_gather = batched_gather


def _patch_dataset_crop():
    """Pad/crop variable-length mdCATH sequences to ``args.crop``.

    MDGen only pads/crops when ``args.atlas`` is set (and then also does random
    replica selection, which we do NOT want).  We wrap ``MDGenDataset.__getitem__``
    so that for our fixed-length-crop runs the returned tensors are cropped (or
    zero-padded) to ``crop`` residues, matching the frozen ``pos_embed`` size.
    """
    import numpy as np
    import torch
    from mdgen.dataset import MDGenDataset
    from mdgen.rigid_utils import Rigid

    _orig = MDGenDataset.__getitem__

    def __getitem__(self, idx):
        # Guard: trajectories shorter than num_frames would make the original
        # code call np.random.choice on an empty range.  Retry with a different
        # index (the dataset repeats/samples at random anyway).
        for _ in range(8):
            try:
                out = _orig(self, idx)
                break
            except ValueError:
                idx = int(np.random.randint(0, len(self.df)))
        else:
            raise RuntimeError("MDGenDataset: too many short trajectories")
        # normalise numpy -> tensor for the fields the collate path expects
        if not isinstance(out["seqres"], torch.Tensor):
            out["seqres"] = torch.as_tensor(np.asarray(out["seqres"]))
        if not isinstance(out["mask"], torch.Tensor):
            out["mask"] = torch.as_tensor(np.asarray(out["mask"], dtype=np.float32))
        crop = getattr(self.args, "crop", None)
        if crop is None or self.args.atlas:
            return out
        L = out["mask"].shape[0]
        if L == crop:
            return out
        if L > crop:
            start = np.random.randint(0, L - crop + 1)
            for k in ("torsions", "trans", "rots"):
                out[k] = out[k][:, start:start + crop]
            for k in ("seqres", "mask", "torsion_mask"):
                out[k] = out[k][start:start + crop]
        else:
            pad = crop - L
            out["rots"] = torch.cat([out["rots"],
                                     torch.eye(3).expand(out["rots"].shape[0], pad, -1, -1)], 1)
            out["trans"] = torch.cat([out["trans"],
                                      torch.zeros(out["trans"].shape[0], pad, 3)], 1)
            out["torsions"] = torch.cat([out["torsions"],
                torch.zeros(out["torsions"].shape[0], pad, out["torsions"].shape[-2], 2)], 1)
            out["torsion_mask"] = torch.cat([out["torsion_mask"],
                                             torch.zeros(pad, out["torsion_mask"].shape[-1])], 0)
            out["mask"] = torch.cat([out["mask"], torch.zeros(pad, dtype=torch.float32)], 0)
            out["seqres"] = torch.cat([out["seqres"], torch.zeros(pad, dtype=torch.long)], 0)
        return out

    MDGenDataset.__getitem__ = __getitem__


def _patch_torch_load():
    torch.serialization.add_safe_globals([argparse.Namespace])
    _orig = torch.load

    def torch_load(f, *a, **k):
        k.setdefault("weights_only", False)
        return _orig(f, *a, **k)

    torch.load = torch_load


def _patch_torchvision_import_order():
    """Make ``pytorch_lightning`` importable when ``torchvision`` is broken.

    In this environment ``import torchvision`` fails ("operator
    torchvision::nms does not exist" -- the wheel was built against a different
    torch), and ``pytorch_lightning`` -> ``torchmetrics`` -> ``torchvision``
    therefore raises before we can reach MDGen.  MDGen never uses torchmetrics,
    so if the real import chain is unusable we install a minimal stub of
    ``torchmetrics`` (only the names PL touches at import time).  No-op when the
    real stack imports cleanly.
    """
    import importlib
    import sys
    import types

    try:
        importlib.import_module("torchmetrics")
        return
    except Exception:
        pass

    tm = types.ModuleType("torchmetrics")
    tm.__path__ = []

    class _Metric:
        def __init__(self, *a, **k):
            pass

    tm.Metric = _Metric
    tm.MetricCollection = object
    tm.Metric = _Metric
    fn = types.ModuleType("torchmetrics.functional")
    fn.__path__ = []
    tm.functional = fn
    sys.modules.setdefault("torchmetrics", tm)
    sys.modules.setdefault("torchmetrics.functional", fn)


def _patch_linalg_backend():
    """cuSOLVER's batched syevd can fail on some CUDA/ptxas combos; MAGMA is a
    drop-in fallback for the tiny 3x3 Kabsch-style eigendecompositions MDGen uses.
    """
    try:
        torch.backends.cuda.preferred_linalg_library("magma")
    except Exception:
        pass


_ODE_STEPS = {"n": 50}
_ODE_PATCHED = {"done": False}


def _patch_ode_steps():
    """Let ``args.inference_steps`` control the number of ODE steps.

    ``wrapper.inference`` calls ``sample_ode(sampling_method=...)`` without a
    ``num_steps`` (the released code left the wired-through line commented out),
    so a fixed-step solver like euler always runs its default 50 steps.  We wrap
    ``Sampler.sample_ode`` to inject ``num_steps`` from a module-global that the
    evaluation script can set via :func:`set_ode_steps`.

    Imported lazily (by :func:`set_ode_steps`) to avoid pulling the transport
    stack -- and its torchmetrics/torchvision dependency -- at import time.
    """
    if _ODE_PATCHED["done"]:
        return
    from mdgen.transport.transport import Sampler
    _orig = Sampler.sample_ode

    def sample_ode(self, *, sampling_method="dopri5", num_steps=None, **kw):
        if num_steps is None:
            num_steps = _ODE_STEPS.get("n", 50)
        return _orig(self, sampling_method=sampling_method,
                     num_steps=num_steps, **kw)

    Sampler.sample_ode = sample_ode
    _ODE_PATCHED["done"] = True


def set_ode_steps(n: int):
    """Set the fixed-step ODE step count used by the euler sampler."""
    _ODE_STEPS["n"] = int(n)
    _patch_ode_steps()


def apply():
    _patch_torchvision_import_order()
    _patch_linalg_backend()
    _patch_batched_gather()
    _patch_torch_load()
    _patch_dataset_crop()


apply()
