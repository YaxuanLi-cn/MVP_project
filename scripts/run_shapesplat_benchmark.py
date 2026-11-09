#!/usr/bin/env python3
"""Run the three primitive methods over a multi-category ShapeSplat subset."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gaussian_decomp.data import convert_shapesplat_directory, load_gaussian_cloud
from gaussian_decomp.evaluate import assembly_metrics
from gaussian_decomp.fitting import FitConfig, fit_primitives
from gaussian_decomp.geometry import sample_off
from gaussian_decomp.primitives import save_primitives


METHODS = [
    ("baseline_superquadric", False, "superquadric"),
    ("gaussian_aware_superquadric", True, "superquadric"),
    ("gaussian_aware_superfrustum", True, "superfrustum"),
]


def run_one(category: str, object_id: str, args) -> dict:
    object_dir = Path(args.data_root) / category / "test" / object_id
    npz = Path(args.work_root) / f"{category}_{object_id}.npz"
    npz.parent.mkdir(parents=True, exist_ok=True)
    if not npz.exists():
        convert_shapesplat_directory(object_dir, npz)
    cloud = load_gaussian_cloud(npz)
    mesh_path = Path(args.mesh_root) / category / "test" / f"{object_id}.off"
    source = sample_off(mesh_path, n=args.source_points, seed=7) if mesh_path.exists() else None
    row = {"category": category, "object": object_id, "n_gaussians": cloud.size, "feature_dim": None if cloud.features is None else int(cloud.features.shape[1]), "source_mesh": str(mesh_path) if source is not None else None}
    for label, aware, primitive_type in METHODS:
        config = FitConfig(n_primitives=args.n_primitives, iterations=args.iterations, primitive_type=primitive_type, use_gaussian_attributes=aware, opacity_weight=1.0 if aware else 0.0, feature_weight=args.feature_weight if aware else 0.0, seed=args.seed)
        primitives, report = fit_primitives(cloud, config)
        primitive_path = Path(args.output).with_name(f"{Path(args.output).stem}_{category}_{object_id}_{label}.json")
        save_primitives(primitives, str(primitive_path))
        result = {"final_loss": report.final_loss, "assignment_entropy": report.assignment_entropy, "primitive_file": str(primitive_path)}
        if source is not None:
            result["source_mesh_metrics"] = assembly_metrics(source, primitives, args.samples)
        row[label] = result
    sidecar = Path(args.output).with_name(f"{Path(args.output).stem}_{category}_{object_id}.row.json")
    sidecar.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/shapesplat")
    p.add_argument("--mesh-root", default="data/modelnet40")
    p.add_argument("--work-root", default="artifacts/benchmark_npz")
    p.add_argument("--categories", default="airplane,bottle,bowl,car,chair,cone,cup,desk,lamp,table")
    p.add_argument("--objects-per-category", type=int, default=8)
    p.add_argument("-k", "--n-primitives", type=int, default=16)
    p.add_argument("--iterations", type=int, default=12)
    p.add_argument("--feature-weight", type=float, default=0.05)
    p.add_argument("--source-points", type=int, default=3000)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--workers", type=int, default=1, help="parallel objects; keep at 1 for stable memory use")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", default="artifacts/shapesplat_benchmark.json")
    args = p.parse_args()
    jobs = []
    for category in [x.strip() for x in args.categories.split(",") if x.strip()]:
        ids = sorted([p.name for p in (Path(args.data_root) / category / "test").iterdir() if p.is_dir()])
        valid = [x for x in ids if all((Path(args.data_root) / category / "test" / x / name).exists() for name in ["coord.npy", "scale.npy", "quat.npy", "opacity.npy"])]
        jobs.extend((category, x) for x in valid[:args.objects_per_category])
    sidecars = {}
    for path in Path(args.output).parent.glob(f"{Path(args.output).stem}_*.row.json"):
        try:
            row = json.loads(path.read_text())
            sidecars[(row["category"], row["object"])] = row
        except Exception:
            pass
    pending = [job for job in jobs if job not in sidecars]
    print(f"Running {len(pending)} pending objects ({len(sidecars)} already complete), {len(METHODS)} methods, {args.iterations} iterations", flush=True)
    rows = list(sidecars.values())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, c, o, args) for c, o in pending]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"Finished {row['category']}/{row['object']}", flush=True)
    rows.sort(key=lambda x: (x["category"], x["object"]))
    result = {"settings": vars(args), "n_objects": len(rows), "methods": [x[0] for x in METHODS], "rows": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
