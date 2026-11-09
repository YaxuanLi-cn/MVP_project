"""Analytic primitive representations used by the decomposition model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json

import numpy as np

from .data import matrix_to_quaternion


@dataclass
class Superquadric:
    center: np.ndarray
    rotation: np.ndarray
    radii: np.ndarray
    e1: float = 1.0
    e2: float = 1.0
    support: float = 1.0
    weight: float = 1.0
    feature_mean: np.ndarray | None = None

    kind: str = "superquadric"

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64)
        self.rotation = np.asarray(self.rotation, dtype=np.float64).reshape(3, 3)
        self.radii = np.maximum(np.asarray(self.radii, dtype=np.float64), 1e-6)

    def implicit(self, points: np.ndarray) -> np.ndarray:
        q = (np.asarray(points) - self.center) @ self.rotation
        a, b, c = self.radii
        # A numerically stable superquadric implicit function. e1/e2 are
        # clamped in the fitter, but clipping here protects external callers.
        e1, e2 = max(self.e1, 0.1), max(self.e2, 0.1)
        term_xy = (np.abs(q[:, 0] / a) ** (2 / e2) + np.abs(q[:, 1] / b) ** (2 / e2)) ** (e2 / e1)
        return term_xy + np.abs(q[:, 2] / c) ** (2 / e1) - 1.0

    def signed_distance_approx(self, points: np.ndarray) -> np.ndarray:
        q = (np.asarray(points) - self.center) @ self.rotation
        scale = np.linalg.norm(q / self.radii, axis=1).clip(1e-8)
        return (scale - 1.0) * float(np.min(self.radii))

    def sample(self, n: int = 1024, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n * 4, 3))
        x /= np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)
        # radial projection onto the implicit surface; exact parameterization
        # is unnecessary for evaluation and this works for all exponents.
        q = x * self.radii
        for _ in range(3):
            # ``implicit`` expects world coordinates; q is expressed in the
            # primitive's local frame during radial projection.
            v = self.implicit(q @ self.rotation.T + self.center)
            q /= np.maximum(1.0 + v[:, None], 1e-3)
        q = q[:n]
        return q @ self.rotation.T + self.center


@dataclass
class SuperFrustum:
    center: np.ndarray
    axis: np.ndarray
    length: float
    radius0: np.ndarray
    radius1: np.ndarray
    e1: float = 1.0
    e2: float = 1.0
    support: float = 1.0
    weight: float = 1.0
    feature_mean: np.ndarray | None = None

    kind: str = "superfrustum"

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64)
        self.axis = np.asarray(self.axis, dtype=np.float64)
        self.axis /= np.linalg.norm(self.axis).clip(1e-8)
        self.length = max(float(self.length), 1e-6)
        self.radius0 = np.maximum(np.asarray(self.radius0, dtype=np.float64), 1e-6)
        self.radius1 = np.maximum(np.asarray(self.radius1, dtype=np.float64), 1e-6)

    def _frame(self) -> np.ndarray:
        z = self.axis
        ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.85 else np.array([0.0, 1.0, 0.0])
        x = np.cross(ref, z)
        x /= np.linalg.norm(x).clip(1e-8)
        y = np.cross(z, x)
        return np.stack([x, y, z], axis=1)

    def implicit(self, points: np.ndarray) -> np.ndarray:
        f = self._frame()
        q = (np.asarray(points) - self.center) @ f
        t = np.clip(q[:, 2] / self.length + 0.5, 0.0, 1.0)
        r = (1.0 - t[:, None]) * self.radius0 + t[:, None] * self.radius1
        radial = (np.abs(q[:, :2] / r) ** (2 / max(self.e2, 0.1))).sum(axis=1) ** (max(self.e2, 0.1) / max(self.e1, 0.1))
        cap = np.maximum(np.abs(q[:, 2]) / (self.length / 2), 1.0)
        return radial + np.maximum(cap - 1.0, 0.0) - 1.0

    def signed_distance_approx(self, points: np.ndarray) -> np.ndarray:
        f = self._frame()
        q = (np.asarray(points) - self.center) @ f
        t = np.clip(q[:, 2] / self.length + 0.5, 0.0, 1.0)
        r = (1.0 - t[:, None]) * self.radius0 + t[:, None] * self.radius1
        radial = np.linalg.norm(q[:, :2] / r, axis=1)
        return (np.maximum(radial, np.abs(q[:, 2]) / (self.length / 2)) - 1.0) * float(np.min(r))

    def sample(self, n: int = 1024, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        f = self._frame()
        t = rng.uniform(-0.5, 0.5, n)
        theta = rng.uniform(0, 2 * np.pi, n)
        u = (t + 0.5)[:, None]
        r = (1 - u) * self.radius0 + u * self.radius1
        q = np.column_stack([r[:, 0] * np.cos(theta), r[:, 1] * np.sin(theta), t * self.length])
        return q @ f.T + self.center


Primitive = Superquadric | SuperFrustum


def primitive_to_dict(p: Primitive) -> dict:
    d = asdict(p)
    for key in ("center", "rotation", "radii", "axis", "radius0", "radius1", "feature_mean"):
        if key in d:
            d[key] = np.asarray(d[key]).tolist()
    return d


def save_primitives(primitives: list[Primitive], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"primitives": [primitive_to_dict(p) for p in primitives]}, f, indent=2)


def load_primitives(path: str) -> list[Primitive]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)["primitives"]
    result: list[Primitive] = []
    for d in raw:
        kind = d.pop("kind", "superquadric")
        if kind == "superfrustum":
            result.append(SuperFrustum(**d))
        else:
            result.append(Superquadric(**d))
    return result
