#!/usr/bin/env python3
"""Fit a primitive assembly to a full-scene 3DGS PLY."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gaussian_decomp.data import load_gaussian_cloud
from gaussian_decomp.evaluate import scene_metrics
from gaussian_decomp.fitting import FitConfig, fit_primitives
from gaussian_decomp.primitives import save_primitives


def downsample_cloud(cloud, max_gaussians: int, seed: int):
    if max_gaussians <= 0 or cloud.size <= max_gaussians:
        return cloud
    rng = __import__('numpy').random.default_rng(seed)
    # Importance sample visible splats while retaining some low-opacity
    # structure so scene boundaries are not erased.
    probs = cloud.weights + 0.15
    probs /= probs.sum()
    idx = rng.choice(cloud.size, size=max_gaussians, replace=False, p=probs)
    return cloud.subset(idx)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cloud", help="full-scene 3DGS PLY/NPZ")
    p.add_argument("-k", "--n-primitives", type=int, default=64)
    p.add_argument("--primitive-type", choices=["superquadric", "superfrustum"], default="superquadric")
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--output", default="artifacts/scene_primitives.json")
    p.add_argument("--report", default="artifacts/scene_report.json")
    p.add_argument("--max-gaussians", type=int, default=120000)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    full_cloud = load_gaussian_cloud(args.cloud)
    input_gaussians = full_cloud.size
    cloud = downsample_cloud(full_cloud, args.max_gaussians, args.seed)
    config = FitConfig(n_primitives=args.n_primitives, primitive_type=args.primitive_type, iterations=args.iterations, use_gaussian_attributes=True, feature_weight=0.05)
    primitives, report = fit_primitives(cloud, config)
    save_primitives(primitives, args.output)
    result = {"cloud": args.cloud, "input_gaussians": input_gaussians, "fit_gaussians": cloud.size, "sampling": {"max_gaussians": args.max_gaussians, "seed": args.seed}, "primitive_file": args.output, "fit_report": report.__dict__, "scene_metrics": scene_metrics(cloud, primitives)}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
