"""Small dependency-free OFF mesh reader and surface sampler."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_off(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip()
        if header != "OFF":
            raise ValueError(f"Expected OFF header in {path}, got {header!r}")
        counts = f.readline().split()
        n_vertices, n_faces = int(counts[0]), int(counts[1])
        vertices = np.asarray([[float(v) for v in f.readline().split()[:3]] for _ in range(n_vertices)], dtype=np.float64)
        faces = []
        for _ in range(n_faces):
            vals = [int(v) for v in f.readline().split()]
            if vals[0] >= 3:
                faces.append(vals[1:])
    return vertices, np.asarray(faces, dtype=object)


def sample_off(path: str | Path, n: int = 10000, seed: int = 0, normalize: bool = True) -> np.ndarray:
    vertices, faces = load_off(path)
    triangles, areas = [], []
    for face in faces:
        # Fan triangulation supports polygons as well as standard triangles.
        for i in range(1, len(face) - 1):
            tri = vertices[[face[0], face[i], face[i + 1]]]
            triangles.append(tri)
            areas.append(np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])) * 0.5)
    triangles = np.asarray(triangles)
    areas = np.asarray(areas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    probs = areas / areas.sum().clip(1e-12)
    chosen = rng.choice(len(triangles), size=n, p=probs)
    tri = triangles[chosen]
    uv = rng.random((n, 2))
    flip = uv.sum(axis=1) > 1
    uv[flip] = 1 - uv[flip]
    points = tri[:, 0] + uv[:, :1] * (tri[:, 1] - tri[:, 0]) + uv[:, 1:] * (tri[:, 2] - tri[:, 0])
    if normalize:
        lo, hi = points.min(axis=0), points.max(axis=0)
        center = (lo + hi) / 2
        scale = max(float(np.max(hi - lo)), 1e-8)
        points = (points - center) / scale
    return points
