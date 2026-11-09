# Large ShapeSplat benchmark

## Dataset and protocol

The extended benchmark uses the real `ShapeSplats/modelnet_3dgs_preprocessed`
repository with matching `ModelNet40_Auto_aligned` source meshes. It contains
67 valid objects from nine categories: airplane (7), bottle (8), bowl (8), car
(7), cone (8), cup (7), desk (8), lamp (7), and table (7). Three entries listed
by the dataset index were excluded because they contained only normals and did
not provide the required `coord.npy`, `scale.npy`, `quat.npy`, and
`opacity.npy` files.

Each method fits 16 primitives for 20 refinement iterations. The Gaussian-aware
method uses covariance, orientation, opacity, and 3-channel color features.
Source meshes are sampled at 3,000 points after axis-aligned bounding-box
normalization. Primitive surfaces are sampled at 256 points per primitive.

## Aggregate results

| Method | Objects | Final loss | Source Chamfer ↓ | Coverage@0.05 ↑ | Coverage@0.1 ↑ |
|---|---:|---:|---:|---:|---:|
| Center-only / Superquadric | 67 | 2.8989 ± 0.1455 | 0.01529 ± 0.02014 | 0.7096 ± 0.2882 | 0.8728 ± 0.2235 |
| Gaussian-aware / Superquadric | 67 | 2.0993 ± 0.1991 | **0.01434 ± 0.01904** | 0.7209 ± 0.3053 | 0.8570 ± 0.2497 |
| Gaussian-aware / SuperFrustum | 67 | **0.8450 ± 0.1188** | 0.02083 ± 0.02147 | **0.7542 ± 0.2626** | **0.8993 ± 0.1718** |

Compared with the center-only baseline, Gaussian-aware Superquadric reduces
mean source Chamfer by 6.2% and improves coverage@0.05 by 1.1 percentage
points. SuperFrustum obtains the lowest internal fitting loss and the highest
coverage at both thresholds, but has higher average Chamfer. This indicates that
its assignment objective is easier to optimize on elongated structures while
its current surface parameterization is less accurate in Euclidean geometry.

## Category-level interpretation

Gaussian-aware Superquadric has lower Chamfer than the baseline on airplane,
bowl, car, cone, cup, lamp, and table; the difference is small or neutral on
bottle and desk. The largest gains occur on airplane and table in fitting loss,
while desk and table remain difficult in surface metrics because 16 primitives
are insufficient for their articulated or highly elongated geometry.

The complete per-object results are in `artifacts/shapesplat_benchmark.json`;
`artifacts/shapesplat_benchmark_summary.json` contains machine-readable overall
and category-level aggregates. Per-object primitive assemblies are stored as
`artifacts/shapesplat_benchmark_*_*.json`.

## Reproduction

```bash
export HF_TOKEN=hf_...
python scripts/download_shapesplat_batch.py \
  --categories airplane,bottle,bowl,car,cone,cup,desk,lamp,table \
  --objects-per-category 8 --workers 8

python scripts/run_shapesplat_benchmark.py \
  --categories airplane,bottle,bowl,car,cone,cup,desk,lamp,table \
  --objects-per-category 8 --n-primitives 16 --iterations 20 \
  --source-points 3000 --samples 256 --workers 1

python scripts/summarize_benchmark.py
```
