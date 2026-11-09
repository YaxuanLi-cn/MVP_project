#!/usr/bin/env python3
"""Evaluate structural consistency from per-Gaussian/point part labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gaussian_decomp.data import load_gaussian_cloud
from gaussian_decomp.evaluate import part_consistency, scene_metrics
from gaussian_decomp.primitives import load_primitives


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cloud", help="3DGS .ply/.npz")
    p.add_argument("primitives", help="primitive JSON")
    p.add_argument("labels", help=".npy labels, one per Gaussian")
    p.add_argument("-o", "--output", default="artifacts/part_consistency.json")
    args = p.parse_args()
    cloud = load_gaussian_cloud(args.cloud)
    labels = np.load(args.labels, allow_pickle=True)
    primitives = load_primitives(args.primitives)
    result = {"cloud": args.cloud, "primitives": args.primitives, "part_consistency": part_consistency(cloud.positions, labels, primitives), "scene_metrics": scene_metrics(cloud, primitives)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
