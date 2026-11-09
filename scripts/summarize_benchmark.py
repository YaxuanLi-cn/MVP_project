#!/usr/bin/env python3
"""Summarize a shapesplat_benchmark JSON into report-friendly statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def stats(rows, method):
    loss = np.asarray([r[method]["final_loss"] for r in rows], dtype=float)
    metrics = [r[method]["source_mesh_metrics"] for r in rows]
    chamfer = np.asarray([m["chamfer"] for m in metrics], dtype=float)
    cov05 = np.asarray([m["coverage@0.05"] for m in metrics], dtype=float)
    cov10 = np.asarray([m["coverage@0.1"] for m in metrics], dtype=float)
    return {"n": len(rows), "loss_mean": float(loss.mean()), "loss_std": float(loss.std(ddof=1)), "chamfer_mean": float(chamfer.mean()), "chamfer_std": float(chamfer.std(ddof=1)), "coverage05_mean": float(cov05.mean()), "coverage05_std": float(cov05.std(ddof=1)), "coverage10_mean": float(cov10.mean()), "coverage10_std": float(cov10.std(ddof=1))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", default="artifacts/shapesplat_benchmark.json", nargs="?")
    p.add_argument("-o", "--output", default="artifacts/shapesplat_benchmark_summary.json")
    args = p.parse_args()
    data = json.loads(Path(args.input).read_text())
    methods = data["methods"]
    overall = {m: stats(data["rows"], m) for m in methods}
    by_category = {}
    for category in sorted({r["category"] for r in data["rows"]}):
        rows = [r for r in data["rows"] if r["category"] == category]
        by_category[category] = {"n_objects": len(rows), **{m: stats(rows, m) for m in methods}}
    result = {"input": args.input, "n_objects": data["n_objects"], "methods": methods, "overall": overall, "by_category": by_category}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
