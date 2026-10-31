"""Exact scalar harmonic covariance audit of two distinct sampler branches.

The default retains the historical non-collective velocity diagnostic.
Current SOTA uses collective_mobility=True: S receives the complete realised
H displacement, including shared noise. This distinction changes the continuous
limit, not merely the finite-step discretization error. No checkpoint is evaluated.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def audit(step, diffusion=1.0, coupling=1.0, gamma=1.0, stiffness=1.0,
          collective_mobility=False, noise_factor=1.0):
    h, d, c, g, k = step, diffusion, coupling, gamma, stiffness
    if min(h, d, g, k) <= 0:
        raise ValueError('step, diffusion, gamma and stiffness must be positive')
    if not collective_mobility and noise_factor != 1.0:
        raise ValueError('legacy diagnostic assumes unit spatial factor')
    m = d * noise_factor**2 if collective_mobility else d
    transition = np.array([[1-h*m*k, -h*m*c],
                           [-h*c*m*k, 1-h*g-h*m*c*c]])
    if collective_mobility:
        # Shared H noise also enters S through C^T * displacement.
        joint_mobility = np.array([[m, m*c], [m*c, g+m*c*c]])
        noise = 2*h*joint_mobility
    else:
        joint_mobility = None
        noise = np.diag([2*h*d, 2*h*g])
    radius = float(max(abs(np.linalg.eigvals(transition))))
    if radius >= 1:
        raise ValueError(f'Unstable transition at step={h}: radius={radius}')
    covariance = np.linalg.solve(
        np.eye(4)-np.kron(transition, transition), noise.reshape(-1)
    ).reshape(2, 2)
    residual = covariance-transition@covariance@transition.T-noise
    target = np.diag([1/k, 1.])
    drift = (transition-np.eye(2))/h
    continuous_residual = drift@target+target@drift.T+noise/h
    return dict(step=h, diffusion=d, coupling=c, gamma=g, stiffness=k,
                branch='collective_shared_displacement' if collective_mobility
                       else 'legacy_noncollective_independent_bath_noise',
                noise_factor=noise_factor, spectral_radius=radius,
                covariance=covariance.tolist(), noise_covariance=noise.tolist(),
                joint_mobility=None if joint_mobility is None else joint_mobility.tolist(),
                target_H_variance=1/k, H_variance_ratio=float(k*covariance[0, 0]),
                continuous_target_lyapunov_residual=float(abs(continuous_residual).max()),
                lyapunov_residual=float(np.max(np.abs(residual))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = dict(
        scope='Scalar constant-parameter harmonic special cases; not checkpoint accuracy',
        correction='Historical coupled results describe the non-collective branch, NOT current SOTA.',
        coupled=[audit(h) for h in (0.1, 0.01, 0.001)],
        uncoupled_control=[audit(h, coupling=0) for h in (0.1, 0.01, 0.001)],
        current_sota_collective=[audit(h, collective_mobility=True) for h in (0.1, 0.01, 0.001)],
    )
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text+'\n')
    print(text)


if __name__ == '__main__':
    main()
