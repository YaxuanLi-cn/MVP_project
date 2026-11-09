"""Command line interface for fitting and ablation experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import GaussianCloud, convert_shapesplat_directory, load_gaussian_cloud, save_npz
from .evaluate import assembly_metrics
from .fitting import FitConfig, fit_primitives
from .primitives import save_primitives
from .synthetic import make_synthetic_cloud


def _write_xyz(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, points)


def _read_xyz(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    p = Path(path)
    if p.suffix == ".npy":
        return np.asarray(np.load(p), dtype=np.float64)
    if p.suffix == ".npz":
        d = np.load(p)
        for key in ("points", "vertices", "positions"):
            if key in d:
                return np.asarray(d[key], dtype=np.float64)
        raise ValueError("source NPZ must contain points, vertices, or positions")
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            vals = line.split()
            if len(vals) >= 3:
                try:
                    rows.append([float(x) for x in vals[:3]])
                except ValueError:
                    pass
    return np.asarray(rows, dtype=np.float64)


def cmd_generate(args: argparse.Namespace) -> None:
    cloud, source = make_synthetic_cloud(args.n_per_part, args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_npz(cloud, out)
    _write_xyz(out.with_suffix(".source.npy"), source)
    print(json.dumps({"gaussians": cloud.size, "output": str(out), "source": str(out.with_suffix('.source.npy'))}, indent=2))


def cmd_fit(args: argparse.Namespace) -> None:
    cloud = load_gaussian_cloud(args.input)
    config = FitConfig(
        n_primitives=args.n_primitives, primitive_type=args.primitive_type,
        iterations=args.iterations, temperature=args.temperature,
        covariance_weight=args.covariance_weight,
        opacity_weight=args.opacity_weight if args.method == "gaussian-aware" else 0.0,
        feature_weight=args.feature_weight if args.method == "gaussian-aware" else 0.0,
        compactness_weight=args.compactness_weight, seed=args.seed,
        use_gaussian_attributes=args.method == "gaussian-aware",
    )
    primitives, report = fit_primitives(cloud, config)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_primitives(primitives, str(out))
    result = {"report": report.__dict__, "primitive_file": str(out)}
    source = _read_xyz(args.source)
    if source is not None:
        result["metrics"] = assembly_metrics(source, primitives, args.samples)
    if args.report:
        Path(args.report).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def cmd_analyze(args: argparse.Namespace) -> None:
    cloud = load_gaussian_cloud(args.input)
    cov = cloud.covariance
    eigvals = np.linalg.eigvalsh(cov)
    orientation = cloud.orientation
    anisotropy = eigvals[:, -1] / eigvals[:, 0].clip(1e-10)
    result = {
        "n_gaussians": cloud.size,
        "position_bounds": [cloud.positions.min(axis=0).tolist(), cloud.positions.max(axis=0).tolist()],
        "opacity": {"mean": float(np.mean(cloud.weights)), "p10": float(np.percentile(cloud.weights, 10)), "p90": float(np.percentile(cloud.weights, 90))},
        "covariance": {"mean_eigenvalues": np.mean(eigvals, axis=0).tolist(), "median_anisotropy": float(np.median(anisotropy))},
        "orientation_determinant_error": float(np.mean(np.abs(np.linalg.det(orientation) - 1.0))),
        "features": None if cloud.features is None else list(cloud.features.shape),
    }
    print(json.dumps(result, indent=2))


def cmd_convert_shapesplat(args: argparse.Namespace) -> None:
    cloud = convert_shapesplat_directory(args.input, args.output, scale_mode=args.scale_mode, quat_order=args.quat_order)
    print(json.dumps({"gaussians": cloud.size, "input": args.input, "output": args.output, "scale_mode": args.scale_mode, "quat_order": args.quat_order}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gaussian-decomp", description="Gaussian-aware primitive decomposition from pretrained 3DGS")
    sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate-synthetic", help="create a synthetic Gaussian cloud and source point cloud")
    g.add_argument("-o", "--output", default="artifacts/synthetic.npz")
    g.add_argument("--n-per-part", type=int, default=180)
    g.add_argument("--seed", type=int, default=11)
    g.set_defaults(func=cmd_generate)

    f = sub.add_parser("fit", help="fit primitive assembly to a 3DGS PLY/NPZ")
    f.add_argument("input")
    f.add_argument("-o", "--output", default="artifacts/primitives.json")
    f.add_argument("--source", help="optional .npy/.npz/XYZ source geometry for evaluation")
    f.add_argument("--report", help="write JSON report")
    f.add_argument("--method", choices=["baseline", "gaussian-aware"], default="gaussian-aware")
    f.add_argument("--primitive-type", choices=["superquadric", "superfrustum"], default="superquadric")
    f.add_argument("-k", "--n-primitives", type=int, default=8)
    f.add_argument("--iterations", type=int, default=35)
    f.add_argument("--temperature", type=float, default=0.08)
    f.add_argument("--covariance-weight", type=float, default=1.0)
    f.add_argument("--opacity-weight", type=float, default=1.0)
    f.add_argument("--compactness-weight", type=float, default=0.02)
    f.add_argument("--feature-weight", type=float, default=0.05)
    f.add_argument("--samples", type=int, default=512)
    f.add_argument("--seed", type=int, default=7)
    f.set_defaults(func=cmd_fit)

    a = sub.add_parser("analyze", help="summarize geometric information in Gaussian attributes")
    a.add_argument("input")
    a.set_defaults(func=cmd_analyze)

    c = sub.add_parser("convert-shapesplat", help="convert one real ShapeSplat object directory to NPZ")
    c.add_argument("input", help="directory containing coord.npy, scale.npy, quat.npy, opacity.npy")
    c.add_argument("-o", "--output", default="artifacts/shapesplat_object.npz")
    c.add_argument("--scale-mode", choices=["auto", "std", "log"], default="auto")
    c.add_argument("--quat-order", choices=["wxyz", "xyzw"], default="wxyz")
    c.set_defaults(func=cmd_convert_shapesplat)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
