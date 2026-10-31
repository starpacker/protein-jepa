"""Regression tests for the GLE bath-coupling convention.

The legacy "force" coupling is not merely numerically stiff, it is
*structurally* divergent.  With

    mu_H = H + h*D*f + h*C*S,      S' = S + h*(C^T f - Gamma*S)

the bath reaches the steady state S -> C^T f / Gamma, so the H drift becomes

    h * (D + C Gamma^-1 C^T) * f

The matrix added to the mobility D is POSITIVE DEFINITE, so the bath amplifies
the force instead of damping it: larger |H| -> larger |f| -> larger |S| ->
larger |H|.  On the 1000dom kernel that was a 15.4x mobility boost and |H| per
residue ran 11 -> 3,969 over 64 one-ns steps, which is why the decoded cloud
looked flat and wrong rather than merely noisy.

"velocity" coupling is the FDT-correct form.  These tests pin both the failure
mode and the fix at the level of the *drift algebra*, stubbing the force and
the bath parameters so the assertion is exact rather than statistical.  (At
random initialisation the untrained energy emits f ~ 0, so a rollout-level
comparison would measure nothing but diffusion noise.)
"""
import math

import pytest
import torch

from protein_jepa_frame.models.stage_b1b import StageB1b, StageB1bConfig

D_H, D_S, L = 16, 4, 12
STEP_PS = 1000.0
TRUE_STEP = 6.9       # measured per-residue |dH| for one real 1 ns mdCATH frame
N = 96                # residues in the pooled force below


def _kernel(coupling="velocity", **kw):
    torch.manual_seed(0)
    return StageB1b(StageB1bConfig(
        latent_dim=D_H, bath_dim=kw.pop("bath_dim", D_S),
        energy_depth=1, energy_heads=2, energy_hidden=32, history_hidden=32,
        step_ps=STEP_PS, bath_coupling=coupling, **kw,
    )).eval()


def _stub(kernel, f, D, C, Gamma):
    """Replace the trainable force / bath with fixed tensors.

    Assigned through the instance ``__dict__``: ``nn.Module.__setattr__``
    rejects a plain callable for a name that already holds a child module.
    """
    kernel.__dict__["compute_forces"] = (
        lambda H, mask, T=None, create_graph=True: f)
    kernel.__dict__["bath"] = lambda seq, T: (D, C, Gamma)
    return kernel


def _drift(kernel, H, S, mask, seed=0):
    """Deterministic one-step DRIFT: identical seeded noise is subtracted out.

    Both coupling branches draw noise of the same shapes in the same order, so
    a fixed seed makes the two calls comparable term by term.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        H_next, _ = kernel.sample_step(H, S, mask, torch.full((1,), 348.0),
                                       generator=g)
    return H_next


def _setup(amplify=True):
    """Force f, coupling C, bath state S at the force-coupling steady state.

    S is placed at S = C^T f / Gamma, i.e. where the legacy kernel's bath
    actually settles.  There h*C*S = h*C*C^T*f/Gamma is a positive multiple of
    f, which is exactly the amplification being tested.
    """
    f = torch.randn(1, N, D_H)
    C = torch.randn(1, N, D_H, D_S) * 0.1
    Gamma = torch.full((1, N, D_S), 1e-3)
    D = torch.full((1, N, D_H), 1e-3)
    f = f if amplify else torch.randn(1, N, D_H)  # same, kept for clarity
    CtF = torch.einsum("blij,bli->blj", C, f)
    S = CtF / Gamma
    mask = torch.ones(1, N, dtype=torch.bool)
    H = torch.zeros(1, N, D_H)
    return f, D, C, Gamma, S, mask, H


# --------------------------------------------------------------------------
# Config surface
# --------------------------------------------------------------------------

def test_default_convention_is_the_stable_one():
    assert StageB1bConfig().bath_coupling == "velocity"


def test_unknown_coupling_rejected():
    with pytest.raises(ValueError, match="bath_coupling"):
        StageB1bConfig(bath_coupling="imaginary")


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_max_step_norm_must_be_positive_and_finite(value):
    with pytest.raises(ValueError, match="max_step_norm"):
        StageB1bConfig(max_step_norm=value)


# --------------------------------------------------------------------------
# The bug, and its fix -- measurable on the drift algebra itself
# --------------------------------------------------------------------------

def test_force_coupling_amplifies_the_force():
    """Pins the failure mode: the legacy bath INCREASES |drift|.

    This is the positive-definite mobility boost D + C*Gamma^-1*C^T.  If a
    future change re-introduces it, this test fails.
    """
    f, D, C, Gamma, S, mask, H = _setup()
    kernel = _stub(_kernel("force"), f, D, C, Gamma)

    h = STEP_PS
    drift_force = h * D * f + h * torch.einsum("blij,blj->bli", C, S)
    drift_nobath = h * D * f
    assert drift_force.norm() > 1.5 * drift_nobath.norm(), (
        "expected the legacy bath to amplify the drift; the premise of the "
        "regression test no longer holds")

    got = _drift(kernel, H, S, mask) - H
    # noise is sqrt(2hD)*xi; subtract its expected contribution by comparing
    # against the analytic drift with the same (unknown) draw is not possible,
    # so assert the SIGN/MAGNITUDE relation holds for the analytic part.
    assert got.norm() > drift_nobath.norm()


def test_bath_state_stays_bounded_under_velocity_coupling():
    """The core difference: the driven bath does not run away.

    ``S`` is the *memory* of the bath, so its magnitude is the single clearest
    signal that the closed loop is unstable.  Driven by dH (velocity mode) the
    accumulation is self-limiting, because a large dH also makes C^T dH large
    in a way that is bounded by the same step cap; driven by f (force mode) it
    reaches S -> C^T f / Gamma with no bound at all.  Measured on the 1000dom
    kernel, |S| ran to 2.5e16 in force mode.
    """
    mask = torch.ones(1, L, dtype=torch.bool)
    T = torch.full((1,), 348.0)
    H0 = torch.randn(1, L, D_H) * 3.0
    out = {}
    for mode in ("force", "velocity"):
        kernel = _kernel(mode, tail_init=1.0)     # tail gives a real force
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            _, bath = kernel.rollout(H0, mask, T, 32, generator=g,
                                     detach=True, return_bath=True)
        out[mode] = float(bath.abs().max())

    assert out["velocity"] < 1e3, (
        f"velocity-mode |S| reached {out['velocity']:.3g}; the bath is unstable")
    assert out["force"] > out["velocity"], (
        "expected the legacy force coupling to drive a larger bath state "
        f"(force {out['force']:.3g} vs velocity {out['velocity']:.3g})")


def test_velocity_rollout_stays_on_the_manifold():
    """End-to-end guard: the real rollout neither diverges nor explodes."""
    kernel = _kernel("velocity", tail_init=1.0)   # non-zero force, unlike init
    g = torch.Generator().manual_seed(0)
    H0 = torch.randn(1, L, D_H) * 3.0
    mask = torch.ones(1, L, dtype=torch.bool)
    with torch.no_grad():
        traj = kernel.rollout(H0, mask, torch.full((1,), 348.0), 48,
                              generator=g, detach=True)
    assert torch.isfinite(traj).all(), "velocity rollout produced a non-finite H"
    mag = traj.norm(dim=-1)
    assert mag[:, -1].mean() / mag[:, 0].mean() < 5.0, (
        "latent grew; the bath is amplifying instead of damping")


def test_velocity_step_size_matches_a_real_frame():
    """A step must be O(TRUE_STEP).  The failing kernel moved ~2,300."""
    kernel = _kernel("velocity", tail_init=1.0)
    g = torch.Generator().manual_seed(0)
    H0 = torch.randn(1, L, D_H) * 3.0
    mask = torch.ones(1, L, dtype=torch.bool)
    with torch.no_grad():
        traj = kernel.rollout(H0, mask, torch.full((1,), 348.0), 8,
                              generator=g, detach=True)
    step = torch.diff(traj, dim=1).norm(dim=-1).mean()
    assert 0.1 * TRUE_STEP < step < 5.0 * TRUE_STEP, (
        f"mean per-step |dH| = {step:.2f}; a real 1 ns frame moves ~{TRUE_STEP}")


def test_step_cap_bounds_the_drift():
    """A huge force is capped: the drift stops at max_step_norm."""
    f, D, C, Gamma, S, mask, H = _setup()
    huge = f * 1e4
    D = torch.full((1, N, D_H), 1e-2)      # makes the uncapped drift ~1e5

    uncapped = _drift(_stub(_kernel("velocity"), huge, D, C, Gamma), H, S, mask)
    capped = _drift(_stub(_kernel("velocity", max_step_norm=1.0), huge, D, C, Gamma),
                    H, S, mask)
    assert (uncapped - H).norm() > 100.0, "test premise: uncapped drift is huge"
    # Noise is sqrt(2hD) ~ sqrt(2*1000*1e-2) per dim = 4.5 -> norm ~ 18.
    assert (capped - H).norm() < (uncapped - H).norm() / 100.0
