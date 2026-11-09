"""Baseline and Gaussian-aware primitive decomposition.

The implementation follows an EM-like alternating scheme:

1. soft assignment of Gaussians to primitives;
2. weighted moment updates for centers and orientations;
3. Gaussian-aware covariance regularization and shape refinement.

The baseline sets every Gaussian covariance to zero and therefore reduces to
fitting primitives to Gaussian centers.  The improved method propagates each
Gaussian covariance through the moment update, so elongated or rotated splats
affect primitive extent and orientation even when their centers are sparse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .data import GaussianCloud
from .primitives import Primitive, SuperFrustum, Superquadric


@dataclass
class FitConfig:
    n_primitives: int = 8
    primitive_type: Literal["superquadric", "superfrustum"] = "superquadric"
    iterations: int = 35
    temperature: float = 0.08
    min_scale: float = 0.01
    max_scale: float = 2.0
    covariance_weight: float = 1.0
    opacity_weight: float = 1.0
    compactness_weight: float = 0.02
    overlap_weight: float = 0.02
    feature_weight: float = 0.05
    seed: int = 7
    use_gaussian_attributes: bool = True


@dataclass
class FitReport:
    method: str
    iterations: int
    final_loss: float
    losses: list[dict] = field(default_factory=list)
    assignment_entropy: float = 0.0
    attribute_ablation: dict = field(default_factory=dict)


def _safe_cov(x: np.ndarray, weights: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    w = weights / (weights.sum() + eps)
    mu = np.sum(x * w[:, None], axis=0)
    d = x - mu
    cov = (d * w[:, None]).T @ d
    cov += np.eye(x.shape[1]) * eps
    return cov


def _eigh_sorted(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh((cov + cov.T) * 0.5)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    # Make the frame right-handed and deterministic.
    if np.linalg.det(vecs) < 0:
        vecs[:, -1] *= -1
    return vals, vecs


def _weighted_kmeans(points: np.ndarray, weights: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(points)
    centers = np.empty((k, 3), dtype=np.float64)
    centers[0] = points[rng.choice(n, p=weights / weights.sum())]
    distances = np.full(n, np.inf)
    for i in range(1, k):
        distances = np.minimum(distances, np.sum((points - centers[i - 1]) ** 2, axis=1))
        prob = weights * (distances + 1e-8)
        prob /= prob.sum()
        centers[i] = points[rng.choice(n, p=prob)]
    for _ in range(20):
        d2 = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = d2.argmin(axis=1)
        new = centers.copy()
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                new[j] = np.average(points[mask], axis=0, weights=weights[mask])
        if np.max(np.abs(new - centers)) < 1e-5:
            break
        centers = new
    return centers


def _init_primitives(cloud: GaussianCloud, config: FitConfig) -> list[Primitive]:
    p = cloud.positions
    w = cloud.weights ** config.opacity_weight
    centers = _weighted_kmeans(p, w, config.n_primitives, config.seed)
    # Voronoi assignment gives a sensible initial local covariance.
    labels = np.sum((p[:, None, :] - centers[None, :, :]) ** 2, axis=2).argmin(axis=1)
    result: list[Primitive] = []
    eye = np.eye(3)
    for j, c in enumerate(centers):
        mask = labels == j
        ww = w[mask]
        if not np.any(mask):
            cov, frame = np.eye(3) * 0.04, eye
        else:
            cov = _safe_cov(p[mask], ww)
            if config.use_gaussian_attributes:
                cov += np.average(cloud.covariance[mask], axis=0, weights=ww) * config.covariance_weight
            vals, frame = _eigh_sorted(cov)
            cov = frame @ np.diag(vals) @ frame.T
        vals, frame = _eigh_sorted(cov)
        radii = np.sqrt(np.maximum(vals * 5.0, config.min_scale**2))
        radii = np.clip(radii, config.min_scale, config.max_scale)
        if config.primitive_type == "superfrustum":
            axis = frame[:, 0]
            local = (p[mask] - c) @ frame if np.any(mask) else np.zeros((1, 3))
            half = max(float(np.percentile(np.abs(local[:, 0]), 90)) if len(local) else radii[0], radii[0])
            low = local[:, 0] < 0 if len(local) else np.array([True])
            high = ~low if len(local) else np.array([True])
            r0 = np.maximum(np.sqrt(np.diag(_safe_cov(local[low, 1:3], ww[low] if len(local) and np.any(low) else np.ones(1)))) * 2, config.min_scale) if len(local) else radii[1:]
            r1 = np.maximum(np.sqrt(np.diag(_safe_cov(local[high, 1:3], ww[high] if len(local) and np.any(high) else np.ones(1)))) * 2, config.min_scale) if len(local) else radii[1:]
            feature_mean = np.average(cloud.features[mask], axis=0, weights=ww) if cloud.features is not None and np.any(mask) else None
            result.append(SuperFrustum(c, axis, 2 * half, r0, r1, weight=float(ww.sum() if len(ww) else 1.0), feature_mean=feature_mean))
        else:
            feature_mean = np.average(cloud.features[mask], axis=0, weights=ww) if cloud.features is not None and np.any(mask) else None
            result.append(Superquadric(c, frame, radii, weight=float(ww.sum() if len(ww) else 1.0), feature_mean=feature_mean))
    return result


def _primitive_covariance(p: Primitive) -> np.ndarray:
    if isinstance(p, Superquadric):
        return p.rotation @ np.diag((p.radii / 2.2) ** 2) @ p.rotation.T
    frame = p._frame()
    r = (p.radius0 + p.radius1) / 2
    return frame @ np.diag([r[0] ** 2, r[1] ** 2, (p.length / 2.2) ** 2]) @ frame.T


def _distances(cloud: GaussianCloud, primitives: list[Primitive], gaussian_aware: bool, feature_weight: float = 0.0) -> np.ndarray:
    p = cloud.positions
    covs = cloud.covariance if gaussian_aware else np.zeros((len(p), 3, 3))
    out = np.empty((len(p), len(primitives)), dtype=np.float64)
    for j, primitive in enumerate(primitives):
        pcov = _primitive_covariance(primitive)
        inv = np.linalg.inv(pcov + np.eye(3) * 1e-5)
        d = p - primitive.center if isinstance(primitive, Superquadric) else p - primitive.center
        # Center distance is the assignment term. The trace term rewards a
        # primitive whose extent explains each splat without overfitting it.
        center_d = np.einsum("ni,ij,nj->n", d, inv, d)
        if gaussian_aware:
            tr = np.einsum("ij,nji->n", inv, covs)
            out[:, j] = center_d / (1.0 + tr) + 0.15 * np.log1p(tr)
        else:
            out[:, j] = center_d
        if feature_weight > 0 and cloud.features is not None and getattr(primitive, "feature_mean", None) is not None:
            f = cloud.features.astype(np.float64)
            f = (f - f.mean(axis=0)) / (f.std(axis=0) + 1e-6)
            pm = (np.asarray(primitive.feature_mean) - cloud.features.mean(axis=0)) / (cloud.features.std(axis=0) + 1e-6)
            out[:, j] += feature_weight * np.mean((f - pm[None]) ** 2, axis=1)
    return out


def _update_primitive(
    primitive: Primitive,
    points: np.ndarray,
    weights: np.ndarray,
    gaussian_cov: np.ndarray,
    features: np.ndarray | None,
    config: FitConfig,
) -> Primitive:
    total = float(weights.sum())
    if total < 1e-7:
        return primitive
    center = np.sum(points * weights[:, None], axis=0) / total
    cov = _safe_cov(points, weights)
    if config.use_gaussian_attributes:
        cov += np.einsum("n,nij->ij", weights, gaussian_cov) / total * config.covariance_weight
    vals, frame = _eigh_sorted(cov)
    radii = np.clip(np.sqrt(np.maximum(vals * 5.0, config.min_scale**2)), config.min_scale, config.max_scale)
    if isinstance(primitive, Superquadric):
        # Exponents are kept interpretable and stable; the data-driven part is
        # the pose and scale, while exponents can be learned later with a
        # differentiable renderer if desired.
        feature_mean = np.average(features, axis=0, weights=weights) if features is not None else primitive.feature_mean
        return Superquadric(center, frame, radii, primitive.e1, primitive.e2, primitive.support, total, feature_mean)
    axis = frame[:, 0]
    local = (points - center) @ frame
    coord = local[:, 0]
    lo = coord <= np.median(coord)
    hi = ~lo
    def radial(mask: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        if np.count_nonzero(mask) < 2:
            return fallback
        rcov = _safe_cov(local[mask, 1:], weights[mask])
        return np.clip(np.sqrt(np.maximum(np.diag(rcov) * 5.0, config.min_scale**2)), config.min_scale, config.max_scale)
    r0, r1 = radial(lo, radii[1:]), radial(hi, radii[1:])
    length = np.clip(np.percentile(np.abs(coord), 90) * 2.0, config.min_scale * 2, config.max_scale * 2)
    feature_mean = np.average(features, axis=0, weights=weights) if features is not None else primitive.feature_mean
    return SuperFrustum(center, axis, length, r0, r1, primitive.e1, primitive.e2, primitive.support, total, feature_mean)


def _losses(cloud: GaussianCloud, primitives: list[Primitive], resp: np.ndarray, config: FitConfig) -> dict:
    d = _distances(cloud, primitives, config.use_gaussian_attributes, config.feature_weight)
    data = float(np.sum(resp * d) / len(cloud.positions))
    entropy = float(-np.sum(resp * np.log(resp.clip(1e-9))) / len(resp))
    compact = float(np.mean([np.prod(p.radii) if isinstance(p, Superquadric) else p.length * np.prod((p.radius0 + p.radius1) / 2) for p in primitives]))
    overlap = 0.0
    for i in range(len(primitives)):
        for j in range(i):
            ci = primitives[i].center
            cj = primitives[j].center
            overlap += np.exp(-np.sum((ci - cj) ** 2) / 0.03)
    total = data + config.compactness_weight * compact + config.overlap_weight * overlap
    return {"data": data, "compactness": compact, "overlap": overlap, "entropy": entropy, "total": float(total)}


def fit_primitives(cloud: GaussianCloud, config: FitConfig | None = None) -> tuple[list[Primitive], FitReport]:
    config = config or FitConfig()
    if cloud.size < config.n_primitives:
        raise ValueError(f"Need at least n_primitives={config.n_primitives} Gaussians, got {cloud.size}")
    work, center, scale = cloud.normalized()
    primitives = _init_primitives(work, config)
    losses: list[dict] = []
    for _ in range(config.iterations):
        d = _distances(work, primitives, config.use_gaussian_attributes, config.feature_weight)
        logits = -d / max(config.temperature, 1e-4)
        logits -= logits.max(axis=1, keepdims=True)
        resp = np.exp(logits)
        resp /= resp.sum(axis=1, keepdims=True).clip(1e-12)
        # Opacity gives visible/high-confidence Gaussians greater influence,
        # while responsibilities prevent hard cluster discontinuities.
        sample_w = work.weights ** config.opacity_weight
        for j, p in enumerate(primitives):
            primitives[j] = _update_primitive(p, work.positions, resp[:, j] * sample_w, work.covariance, work.features, config)
        losses.append(_losses(work, primitives, resp, config))
    # Transform the normalized solution back to the source coordinate system.
    s = float(scale[0])
    for p in primitives:
        if isinstance(p, Superquadric):
            p.center = p.center * s + center
            p.radii = p.radii * s
        else:
            p.center = p.center * s + center
            p.length *= s
            p.radius0 *= s
            p.radius1 *= s
    final = losses[-1] if losses else {"total": 0.0, "entropy": 0.0}
    report = FitReport(
        method="gaussian-aware" if config.use_gaussian_attributes else "point-cloud-baseline",
        iterations=len(losses), final_loss=float(final["total"]), losses=losses,
        assignment_entropy=float(final.get("entropy", 0.0)),
        attribute_ablation={"covariance": config.use_gaussian_attributes, "orientation": config.use_gaussian_attributes, "opacity": config.opacity_weight > 0},
    )
    return primitives, report


def fit_baseline(cloud: GaussianCloud, n_primitives: int = 8, **kwargs) -> tuple[list[Primitive], FitReport]:
    kwargs.setdefault("opacity_weight", 0.0)
    kwargs.setdefault("feature_weight", 0.0)
    return fit_primitives(cloud, FitConfig(n_primitives=n_primitives, use_gaussian_attributes=False, **kwargs))


def fit_gaussian_aware(cloud: GaussianCloud, n_primitives: int = 8, **kwargs) -> tuple[list[Primitive], FitReport]:
    return fit_primitives(cloud, FitConfig(n_primitives=n_primitives, use_gaussian_attributes=True, **kwargs))
