#!/usr/bin/env python3
"""Download and evaluate a small real ShapeSplat subset.

This creates a reproducible table for the project report. The optional
``opacity_surface_proxy`` metric is explicitly a proxy built from the real
Gaussian representation; it is not an independent mesh ground truth.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from gaussian_decomp.data import convert_shapesplat_directory, load_gaussian_cloud
from gaussian_decomp.evaluate import assembly_metrics
from gaussian_decomp.fitting import FitConfig, fit_primitives
from gaussian_decomp.geometry import sample_off
from gaussian_decomp.primitives import save_primitives
from download_shapesplat import API, RAW, download, request_json


def ensure_object(repo: str, category: str, split: str, object_id: str, root: Path, token: str) -> Path:
    prefix = f"{category}/{split}/{object_id}"
    target_root = root / category / split / object_id
    if (target_root / "coord.npy").exists() and (target_root / "opacity.npy").exists():
        return target_root
    entries = request_json(API.format(repo=repo, prefix=prefix), token)
    for entry in entries:
        if entry.get("type") != "file" or not entry.get("path", "").endswith(".npy"):
            continue
        rel = entry["path"].split(prefix + "/", 1)[-1]
        download(RAW.format(repo=repo, path=entry["path"]), target_root / rel, token)
    return target_root


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--objects", default="bowl_0065,bowl_0066,bowl_0067,bowl_0068")
    p.add_argument("--category", default="bowl")
    p.add_argument("--split", default="test")
    p.add_argument("--repo", default="ShapeSplats/modelnet_3dgs_preprocessed")
    p.add_argument("--token", default=None)
    p.add_argument("--data-root", default="data/shapesplat")
    p.add_argument("--output", default="artifacts/shapesplat_subset.json")
    p.add_argument("--source-root", default="data/modelnet40", help="directory containing category/split/object.off")
    p.add_argument("-k", "--n-primitives", type=int, default=16)
    p.add_argument("--iterations", type=int, default=20)
    args = p.parse_args()
    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or ""
    if not token:
        raise SystemExit("Set HF_TOKEN after accepting the ShapeSplats dataset terms.")
    rows = []
    for object_id in [x.strip() for x in args.objects.split(",") if x.strip()]:
        print(f"Processing {args.category}/{args.split}/{object_id}", flush=True)
        object_dir = ensure_object(args.repo, args.category, args.split, object_id, Path(args.data_root), token)
        npz = Path(args.data_root) / f"{args.category}_{args.split}_{object_id}.npz"
        convert_shapesplat_directory(object_dir, npz)
        cloud = load_gaussian_cloud(npz)
        proxy = cloud.positions[cloud.weights >= np.percentile(cloud.weights, 75)]
        source_path = Path(args.source_root) / args.category / args.split / f"{object_id}.off"
        source = sample_off(source_path, n=6000, seed=7) if source_path.exists() else None
        row = {"object": object_id, "n_gaussians": cloud.size, "feature_dim": None if cloud.features is None else int(cloud.features.shape[1])}
        row["source_mesh"] = str(source_path) if source is not None else None
        for label, aware, primitive_type in [("baseline_superquadric", False, "superquadric"), ("gaussian_aware_superquadric", True, "superquadric"), ("gaussian_aware_superfrustum", True, "superfrustum")]:
            config = FitConfig(n_primitives=args.n_primitives, iterations=args.iterations, primitive_type=primitive_type, use_gaussian_attributes=aware, opacity_weight=1.0 if aware else 0.0, feature_weight=0.05 if aware else 0.0)
            primitives, report = fit_primitives(cloud, config)
            out = Path(args.output).with_name(f"{Path(args.output).stem}_{object_id}_{label}.json")
            save_primitives(primitives, str(out))
            row[label] = {"final_loss": report.final_loss, "assignment_entropy": report.assignment_entropy, "proxy_metrics": assembly_metrics(proxy, primitives, 256)}
            if source is not None:
                row[label]["source_mesh_metrics"] = assembly_metrics(source, primitives, 256)
        rows.append(row)
    result = {"category": args.category, "split": args.split, "n_objects": len(rows), "rows": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
