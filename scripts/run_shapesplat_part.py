#!/usr/bin/env python3
"""Evaluate primitive structural consistency on ShapeSplat-Part annotations.

The ShapeSplat-Part release contains normalized ShapeNet Part point clouds,
normals and per-point part IDs rather than a matching 3DGS PLY. We construct a
GaussianCloud from each annotated point set; covariance is anisotropic in the
surface-normal frame, which provides a deterministic Gaussian-aware input for
the consistency experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gaussian_decomp.data import GaussianCloud, matrix_to_quaternion
from gaussian_decomp.evaluate import part_consistency
from gaussian_decomp.fitting import FitConfig, fit_primitives
from gaussian_decomp.primitives import save_primitives


CATEGORIES = {
    "airplane": "02691156", "bag": "02773838", "cap": "02946225",
    "car": "02958343", "chair": "03001627", "earphone": "03261776",
    "guitar": "03467517", "knife": "03624134", "lamp": "03636649",
    "laptop": "03642806", "motorbike": "03790512", "mug": "03797390",
    "pistol": "03948459", "rocket": "04099429", "table": "04379243",
}


def load_part_file(path: Path) -> tuple[GaussianCloud, np.ndarray]:
    arr = np.loadtxt(path, dtype=np.float64)
    points, normals, labels = arr[:, :3], arr[:, 3:6], arr[:, 6].astype(np.int64)
    # The normal frame gives each point a thin extent along the surface normal.
    # Quaternion orientation is not needed by this benchmark because the
    # Gaussian-aware fitter consumes the covariance matrices derived here.
    frames = []
    for normal in normals:
        z = normal / max(np.linalg.norm(normal), 1e-8)
        ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.85 else np.array([0.0, 1.0, 0.0])
        x = np.cross(ref, z); x /= max(np.linalg.norm(x), 1e-8)
        y = np.cross(z, x)
        frames.append(np.stack([x, y, z], axis=1))
    rotations = matrix_to_quaternion(np.asarray(frames))
    scales = np.log(np.column_stack([np.full(len(points), 0.012), np.full(len(points), 0.012), np.full(len(points), 0.004)]))
    cloud = GaussianCloud(points, scales, rotations, np.ones(len(points)), normals)
    return cloud, labels


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/shapesplat_part/extracted/shapenetcore_partanno_segmentation_benchmark_v0_normal")
    p.add_argument("--categories", default="airplane,car,chair,lamp,table")
    p.add_argument("--objects-per-category", type=int, default=10)
    p.add_argument("-k", "--n-primitives", type=int, default=8)
    p.add_argument("--iterations", type=int, default=12)
    p.add_argument("--output", default="artifacts/shapesplat_part_consistency.json")
    args = p.parse_args()
    root = Path(args.root)
    jobs = []
    for name in [x.strip() for x in args.categories.split(",") if x.strip()]:
        category_dir = root / CATEGORIES[name]
        jobs.extend((name, path) for path in sorted(category_dir.glob("*.txt"))[:args.objects_per_category])
    rows = []
    for index, (category, path) in enumerate(jobs, 1):
        cloud, labels = load_part_file(path)
        row = {"category": category, "object": path.stem, "n_points": cloud.size, "n_parts": int(len(np.unique(labels)))}
        for method, aware in [("baseline_superquadric", False), ("gaussian_aware_superquadric", True)]:
            config = FitConfig(n_primitives=args.n_primitives, iterations=args.iterations, primitive_type="superquadric", use_gaussian_attributes=aware, covariance_weight=1.0 if aware else 0.0, opacity_weight=0.0 if not aware else 1.0, feature_weight=0.0, seed=7)
            primitives, report = fit_primitives(cloud, config)
            metrics = part_consistency(cloud.positions, labels, primitives)
            primitive_path = Path(args.output).with_name(f"{Path(args.output).stem}_{category}_{path.stem}_{method}.json")
            save_primitives(primitives, str(primitive_path))
            row[method] = {"final_loss": report.final_loss, "metrics": metrics, "primitive_file": str(primitive_path)}
        rows.append(row)
        print(f"[{index}/{len(jobs)}] {category}/{path.stem}", flush=True)
    result = {"categories": list(CATEGORIES), "settings": vars(args), "n_objects": len(rows), "rows": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
