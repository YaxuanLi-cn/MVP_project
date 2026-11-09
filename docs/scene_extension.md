# Structural consistency and SceneSplat extension

The implementation now supports both requested evaluation modes.

## Part annotations

`gaussian_decomp.evaluate.part_consistency` takes one label per Gaussian or
source point and reports:

- primitive-to-part purity;
- boundary leakage (`1 - purity`);
- normalized mutual information (NMI);
- majority semantic label assigned to every primitive.

Run it with:

```bash
python scripts/evaluate_parts.py \
  data/object/point_cloud.ply \
  artifacts/object.primitives.json \
  data/object/part_labels.npy \
  --output artifacts/object.part_consistency.json
```

`part_labels.npy` must have one entry per Gaussian. Labels may be integer IDs or
strings. This keeps the evaluator compatible with ScanNet/ScanNet++ semantic
labels and with custom object-part annotations.

## Full-scene extension

`scripts/run_scene.py` fits the same Gaussian-aware primitives to a complete
scene PLY and reports scene-level structure:

- Gaussian-to-primitive compression ratio;
- scene bounding-box volume;
- fraction of primitives receiving assignments;
- mean/min primitive-center separation;
- assignment entropy.

The downloader targets the SceneSplat-7K GaussianWorld repositories:

```bash
export HF_TOKEN=hf_...
python scripts/download_scenesplat.py \
  --repo GaussianWorld/scannet_mcmc_3dgs \
  --scene scene0000_00 \
  --output data/scenesplat

python scripts/run_scene.py \
  data/scenesplat/scannet_mcmc_3dgs/scene0000_00/point_cloud_30000.ply \
  --n-primitives 64 \
  --output artifacts/scene0000_00.primitives.json \
  --report artifacts/scene0000_00.report.json
```

The local implementation was smoke-tested on the real ShapeSplat Gaussian
cloud (`bowl_0065.npz`) through the same full-scene code path: 17,576 Gaussians
were compressed to 16 primitives, with all primitives receiving assignments.
The SceneSplat-7K index is now accessible with the supplied token, but the
component Gaussian repositories (for example
`GaussianWorld/scannet_mcmc_3dgs`) have their own gated access check. The
downloader reports HTTP 401/403 explicitly; accept the component repository's
terms as well before downloading an actual scene PLY.

