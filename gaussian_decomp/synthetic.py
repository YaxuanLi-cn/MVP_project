"""Synthetic Gaussian clouds for smoke tests and quick ablations."""

from __future__ import annotations

import numpy as np

from .data import GaussianCloud, matrix_to_quaternion


def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def make_synthetic_cloud(n_per_part: int = 180, seed: int = 11) -> tuple[GaussianCloud, np.ndarray]:
    """Create a bent two-part object with anisotropic, rotated splats."""
    rng = np.random.default_rng(seed)
    specs = [
        (np.array([-0.55, 0.0, 0.0]), np.array([0.38, 0.18, 0.22]), 0.2),
        (np.array([0.45, 0.12, 0.10]), np.array([0.18, 0.48, 0.16]), -0.55),
        (np.array([0.0, 0.0, 0.55]), np.array([0.22, 0.20, 0.35]), 0.1),
    ]
    positions, scales, rots, opacity, source = [], [], [], [], []
    for center, radii, angle in specs:
        r = _rot_z(angle)
        local = rng.normal(size=(n_per_part, 3)) * radii
        p = local @ r.T + center
        # Splats follow the local part direction, making covariance useful.
        local_std = np.maximum(radii * rng.uniform(0.15, 0.32, size=(n_per_part, 3)), 0.015)
        positions.append(p)
        scales.append(np.log(local_std))
        rots.append(np.repeat(matrix_to_quaternion(r[None]), n_per_part, axis=0))
        opacity.append(rng.uniform(0.55, 1.0, n_per_part))
        source.append(p + rng.normal(scale=0.012, size=p.shape))
    cloud = GaussianCloud(np.concatenate(positions), np.concatenate(scales), np.concatenate(rots), np.concatenate(opacity))
    return cloud, np.concatenate(source)
