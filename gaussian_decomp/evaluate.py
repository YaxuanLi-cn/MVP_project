"""Geometry metrics for primitive assemblies."""

from __future__ import annotations

import numpy as np

from .primitives import Primitive


def sample_primitives(primitives: list[Primitive], points_per_primitive: int = 1024) -> np.ndarray:
    chunks = [p.sample(points_per_primitive, seed=i) for i, p in enumerate(primitives)]
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 3))


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    # Chunking keeps evaluation usable for large ShapeSplat objects.
    def one_way(x, y):
        vals = []
        for start in range(0, len(x), 2048):
            d = x[start:start + 2048, None, :] - y[None, :, :]
            vals.append(np.min(np.sum(d * d, axis=2), axis=1))
        return float(np.mean(np.concatenate(vals)))
    return 0.5 * (one_way(a, b) + one_way(b, a))


def coverage(points: np.ndarray, primitives: list[Primitive], threshold: float = 0.05) -> float:
    if len(points) == 0:
        return 0.0
    d = np.stack([np.abs(p.signed_distance_approx(points)) for p in primitives], axis=1)
    return float(np.mean(np.min(d, axis=1) < threshold))


def assembly_metrics(points: np.ndarray, primitives: list[Primitive], points_per_primitive: int = 512) -> dict:
    sampled = sample_primitives(primitives, points_per_primitive)
    return {
        "chamfer": chamfer_distance(points, sampled),
        "coverage@0.05": coverage(points, primitives, 0.05),
        "coverage@0.1": coverage(points, primitives, 0.1),
        "n_primitives": len(primitives),
        "compression_ratio": float(len(points) / max(len(primitives), 1)),
    }


def primitive_assignments(points: np.ndarray, primitives: list[Primitive]) -> np.ndarray:
    if len(primitives) == 0:
        return np.full(len(points), -1, dtype=np.int64)
    distances = np.stack([np.abs(p.signed_distance_approx(points)) for p in primitives], axis=1)
    return distances.argmin(axis=1)


def part_consistency(points: np.ndarray, labels: np.ndarray, primitives: list[Primitive]) -> dict:
    """Evaluate whether fitted primitives agree with per-point part labels.

    Labels may be integer IDs or strings. Each primitive receives its majority
    part label; purity and boundary leakage are reported without requiring a
    learned semantic classifier.
    """
    labels = np.asarray(labels)
    if len(points) != len(labels):
        raise ValueError("labels must have one entry per point/Gaussian")
    assignment = primitive_assignments(points, primitives)
    mapped = np.empty(len(labels), dtype=object)
    total = 0.0
    weighted_purity = 0.0
    leakage = 0.0
    primitive_label = {}
    for j in range(len(primitives)):
        mask = assignment == j
        if not np.any(mask):
            primitive_label[j] = None
            continue
        values, counts = np.unique(labels[mask], return_counts=True)
        winner = values[np.argmax(counts)]
        primitive_label[j] = winner.item() if hasattr(winner, "item") else winner
        majority = float(counts.max() / counts.sum())
        weighted_purity += counts.sum() * majority
        leakage += counts.sum() * (1.0 - majority)
        total += counts.sum()
        mapped[mask] = winner
    purity = weighted_purity / max(total, 1.0)
    # Pairwise label agreement, useful when labels are sparse/non-contiguous.
    nmi = _normalized_mutual_information(labels, assignment)
    return {
        "purity": float(purity),
        "boundary_leakage": float(leakage / max(total, 1.0)),
        "nmi": float(nmi),
        "n_parts": int(len(np.unique(labels))),
        "n_primitives": len(primitives),
        "primitive_majority_labels": primitive_label,
    }


def _normalized_mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    av, ai = np.unique(a, return_inverse=True)
    bv, bi = np.unique(b, return_inverse=True)
    table = np.zeros((len(av), len(bv)), dtype=np.float64)
    np.add.at(table, (ai, bi), 1.0)
    pxy = table / max(len(a), 1)
    px, py = pxy.sum(1), pxy.sum(0)
    nz = pxy > 0
    mi = float(np.sum(pxy[nz] * np.log(pxy[nz] / (px[:, None] * py[None, :])[nz])))
    hx = float(-np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = float(-np.sum(py[py > 0] * np.log(py[py > 0])))
    return mi / max(np.sqrt(hx * hy), 1e-12)


def scene_metrics(cloud, primitives: list[Primitive]) -> dict:
    """Scene-level structural metrics that do not require RGB rendering."""
    points = cloud.positions
    assignments = primitive_assignments(points, primitives)
    used = np.bincount(assignments, minlength=len(primitives)) if len(assignments) else np.zeros(len(primitives))
    bbox = points.max(0) - points.min(0)
    occupied = float(np.prod(bbox))
    centers = np.stack([p.center for p in primitives]) if primitives else np.empty((0, 3))
    pairwise = []
    if len(centers) > 1:
        pairwise = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)[np.triu_indices(len(centers), 1)]
    return {
        "n_gaussians": int(cloud.size),
        "n_primitives": len(primitives),
        "compression_ratio": float(cloud.size / max(len(primitives), 1)),
        "scene_bbox": bbox.tolist(),
        "scene_bbox_volume": occupied,
        "primitive_occupancy_fraction": float(np.mean(used > 0)) if len(used) else 0.0,
        "primitive_center_separation_mean": float(np.mean(pairwise)) if len(pairwise) else 0.0,
        "primitive_center_separation_min": float(np.min(pairwise)) if len(pairwise) else 0.0,
        "assignment_entropy": float(-np.sum((used / max(used.sum(), 1)).clip(1e-12) * np.log((used / max(used.sum(), 1)).clip(1e-12)))),
    }
