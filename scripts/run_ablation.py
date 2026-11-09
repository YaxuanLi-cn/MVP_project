#!/usr/bin/env python3
"""Run the baseline/attribute ablation on one Gaussian file.

Example:
  python scripts/run_ablation.py artifacts/synthetic.npz --source artifacts/synthetic.source.npy -k 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gaussian_decomp.cli import _read_xyz
from gaussian_decomp.data import load_gaussian_cloud
from gaussian_decomp.evaluate import assembly_metrics
from gaussian_decomp.fitting import FitConfig, fit_primitives


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--source")
    p.add_argument("-k", "--n-primitives", type=int, default=8)
    p.add_argument("--iterations", type=int, default=35)
    p.add_argument("--primitive-type", choices=["superquadric", "superfrustum"], default="superquadric")
    p.add_argument("--output", default="artifacts/ablation.json")
    args = p.parse_args()
    cloud = load_gaussian_cloud(args.input)
    source = _read_xyz(args.source)
    rows = []
    for label, use_attributes, covariance_weight, opacity_weight in [
        ("positions-only", False, 0.0, 0.0),
        ("opacity-only", False, 0.0, 1.0),
        ("covariance+orientation", True, 1.0, 0.0),
        ("all-gaussian-attributes", True, 1.0, 1.0),
    ]:
        config = FitConfig(
            n_primitives=args.n_primitives,
            primitive_type=args.primitive_type,
            iterations=args.iterations,
            use_gaussian_attributes=use_attributes,
            covariance_weight=covariance_weight,
            opacity_weight=opacity_weight,
        )
        primitives, report = fit_primitives(cloud, config)
        row = {"setting": label, "report": report.__dict__}
        if source is not None:
            row["metrics"] = assembly_metrics(source, primitives)
        rows.append(row)
    result = {"input": str(Path(args.input)), "rows": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
