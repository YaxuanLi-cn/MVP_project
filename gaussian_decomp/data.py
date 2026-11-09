"""Data structures and loaders for pretrained 3D Gaussian Splatting clouds.

The loader intentionally accepts the property names emitted by the reference
3DGS implementation (``x/y/z``, ``scale_0..2``, ``rot_0..3``, ``opacity``),
plus a compact NPZ format useful for experiments and unit tests.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class GaussianCloud:
    """A collection of 3D anisotropic Gaussians.

    ``scales`` are stored in log-scale, matching the 3DGS parameterization.
    ``rotations`` are quaternions in ``wxyz`` order.  The class exposes a
    covariance matrix view so fitting code can remain agnostic to the source
    file format.
    """

    positions: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    opacity: np.ndarray
    features: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=np.float64)
        self.scales = np.asarray(self.scales, dtype=np.float64)
        self.rotations = np.asarray(self.rotations, dtype=np.float64)
        self.opacity = np.asarray(self.opacity, dtype=np.float64).reshape(-1)
        n = len(self.positions)
        if self.positions.shape != (n, 3):
            raise ValueError("positions must have shape (N, 3)")
        if self.scales.shape != (n, 3):
            raise ValueError("scales must have shape (N, 3)")
        if self.rotations.shape != (n, 4):
            raise ValueError("rotations must have shape (N, 4) in wxyz order")
        if len(self.opacity) != n:
            raise ValueError("opacity must have shape (N,)")
        if self.features is not None:
            self.features = np.asarray(self.features, dtype=np.float64)
            if len(self.features) != n:
                raise ValueError("features must have the same number of rows as positions")

    @property
    def size(self) -> int:
        return len(self.positions)

    @property
    def covariance(self) -> np.ndarray:
        """Return per-Gaussian 3x3 covariance matrices."""
        rotations = quaternion_to_matrix(self.rotations)
        std = np.exp(np.clip(self.scales, -14.0, 8.0))
        return rotations @ np.einsum("ni,nij->nij", std**2, np.eye(3)[None]) @ np.transpose(rotations, (0, 2, 1))

    @property
    def orientation(self) -> np.ndarray:
        return quaternion_to_matrix(self.rotations)

    @property
    def weights(self) -> np.ndarray:
        # 3DGS stores opacity in logit space.  Accept already-normalized input
        # as well, which makes hand-authored NPZ files convenient.
        if np.nanmin(self.opacity) < 0.0 or np.nanmax(self.opacity) > 1.0:
            return 1.0 / (1.0 + np.exp(-np.clip(self.opacity, -30.0, 30.0)))
        return np.clip(self.opacity, 1e-5, 1.0)

    def subset(self, indices: np.ndarray) -> "GaussianCloud":
        return GaussianCloud(
            self.positions[indices], self.scales[indices], self.rotations[indices],
            self.opacity[indices], None if self.features is None else self.features[indices]
        )

    def normalized(self) -> tuple["GaussianCloud", np.ndarray, np.ndarray]:
        """Center and scale the cloud for numerically stable fitting."""
        lo = self.positions.min(axis=0)
        hi = self.positions.max(axis=0)
        center = (lo + hi) / 2.0
        scale = np.max(hi - lo)
        scale = max(float(scale), 1e-8)
        p = (self.positions - center) / scale
        s = self.scales - np.log(scale)
        return GaussianCloud(p, s, self.rotations.copy(), self.opacity.copy(), self.features), center, np.array([scale])


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=1, keepdims=True).clip(1e-12)
    w, x, y, z = (q / norm).T
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], axis=1).reshape(-1, 3, 3)


def matrix_to_quaternion(m: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to stable wxyz quaternions."""
    out = np.zeros((len(m), 4), dtype=np.float64)
    for i, r in enumerate(m):
        t = np.trace(r)
        if t > 0:
            s = np.sqrt(t + 1.0) * 2
            out[i] = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
        else:
            j = int(np.argmax(np.diag(r)))
            k, l = (j + 1) % 3, (j + 2) % 3
            s = np.sqrt(max(1.0 + r[j, j] - r[k, k] - r[l, l], 1e-12)) * 2
            q = np.zeros(4)
            q[j + 1] = 0.25 * s
            q[0] = (r[l, k] - r[k, l]) / s
            q[k + 1] = (r[k, j] + r[j, k]) / s
            q[l + 1] = (r[l, j] + r[j, l]) / s
            out[i] = q
    out /= np.linalg.norm(out, axis=1, keepdims=True).clip(1e-12)
    return out


def load_gaussian_cloud(path: str | Path) -> GaussianCloud:
    path = Path(path)
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        required = ["positions", "scales", "rotations", "opacity"]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"NPZ is missing required arrays: {missing}")
        return GaussianCloud(*(data[k] for k in required), data["features"] if "features" in data else None)
    if path.suffix.lower() == ".ply":
        return load_ply(path)
    raise ValueError(f"Unsupported Gaussian file: {path} (use .ply or .npz)")


def _parse_ply_header(f) -> tuple[str, int, list[tuple[str, str]], int]:
    if f.readline().strip() != b"ply":
        raise ValueError("Not a PLY file")
    fmt, count, props = None, None, []
    while True:
        line = f.readline()
        if not line:
            raise ValueError("Truncated PLY header")
        text = line.decode("ascii", errors="replace").strip().split()
        if not text:
            continue
        if text[0] == "format":
            fmt = text[1]
        elif text[0] == "element" and text[1] == "vertex":
            count = int(text[2])
        elif text[0] == "property" and len(text) >= 3 and text[1] != "list":
            props.append((text[1], text[2]))
        elif text[0] == "end_header":
            break
    if fmt is None or count is None:
        raise ValueError("PLY has no vertex element or format")
    return fmt, count, props, f.tell()


def load_ply(path: str | Path) -> GaussianCloud:
    with open(path, "rb") as f:
        fmt, count, props, _ = _parse_ply_header(f)
        names = [p[1] for p in props]
        if fmt == "ascii":
            rows = [f.readline().decode("ascii").split() for _ in range(count)]
            arr = np.asarray(rows, dtype=np.float64)
        elif fmt == "binary_little_endian":
            code = {"char": "b", "uchar": "B", "short": "h", "ushort": "H", "int": "i", "uint": "I", "float": "f", "double": "d"}
            fmt_struct = "<" + "".join(code[t] for t, _ in props)
            size = struct.calcsize(fmt_struct)
            arr = np.asarray([struct.unpack(fmt_struct, f.read(size)) for _ in range(count)], dtype=np.float64)
        else:
            raise ValueError(f"Unsupported PLY format: {fmt}")
    def col(name: str, default: float = 0.0) -> np.ndarray:
        return arr[:, names.index(name)] if name in names else np.full(count, default)
    pos = np.column_stack([col("x"), col("y"), col("z")])
    scales = np.column_stack([col(f"scale_{i}") for i in range(3)])
    rotations = np.column_stack([col(f"rot_{i}", 1.0 if i == 0 else 0.0) for i in range(4)])
    opacity = col("opacity", 1.0)
    feature_names = [n for n in names if n.startswith("f_dc_") or n.startswith("features_")]
    features = arr[:, [names.index(n) for n in feature_names]] if feature_names else None
    return GaussianCloud(pos, scales, rotations, opacity, features)


def save_npz(cloud: GaussianCloud, path: str | Path) -> None:
    np.savez_compressed(path, positions=cloud.positions, scales=cloud.scales, rotations=cloud.rotations, opacity=cloud.opacity, **({"features": cloud.features} if cloud.features is not None else {}))


def load_shapesplat_directory(path: str | Path, scale_mode: str = "auto", quat_order: str = "wxyz") -> GaussianCloud:
    """Load one object from ShapeSplats/modelnet_3dgs_preprocessed.

    Expected files are ``coord.npy``, ``scale.npy``, ``quat.npy`` and
    ``opacity.npy``.  The public preprocessed release has used both positive
    standard deviations and log-scales in different revisions, so ``auto``
    detects the representation conservatively; it can be overridden with
    ``scale_mode=std`` or ``scale_mode=log``.
    """
    root = Path(path)
    def read(name: str) -> np.ndarray:
        candidate = root / name
        if not candidate.exists():
            raise FileNotFoundError(f"ShapeSplat object is missing {candidate}")
        return np.asarray(np.load(candidate), dtype=np.float64)
    positions = read("coord.npy").reshape(-1, 3)
    scales_raw = read("scale.npy").reshape(-1, 3)
    if scale_mode == "auto":
        # Log-scales contain negative values in standard 3DGS exports. A
        # strictly positive, small-valued array is treated as standard dev.
        scale_mode = "std" if np.all(scales_raw > 0) and np.percentile(scales_raw, 99) < 5 else "log"
    scales = np.log(np.maximum(scales_raw, 1e-8)) if scale_mode == "std" else scales_raw
    quat = read("quat.npy").reshape(-1, 4)
    if quat_order.lower() == "xyzw":
        quat = quat[:, [3, 0, 1, 2]]
    opacity = read("opacity.npy").reshape(-1)
    color_path = root / "color.npy"
    features = np.asarray(np.load(color_path), dtype=np.float64).reshape(len(positions), -1) if color_path.exists() else None
    return GaussianCloud(positions, scales, quat, opacity, features)


def convert_shapesplat_directory(path: str | Path, output: str | Path, scale_mode: str = "auto", quat_order: str = "wxyz") -> GaussianCloud:
    cloud = load_shapesplat_directory(path, scale_mode=scale_mode, quat_order=quat_order)
    save_npz(cloud, output)
    return cloud
