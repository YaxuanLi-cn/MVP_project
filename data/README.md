# Local data

Dataset files are intentionally excluded from Git. Download the exact data
used by the report with:

```bash
export HF_TOKEN=hf_...
python scripts/download_data.py
```

For the larger benchmark subset used in the extended report:

```bash
python scripts/download_shapesplat_batch.py \
  --categories airplane,bottle,bowl,car,chair,cone,cup,desk,lamp,table \
  --objects-per-category 8
```

The script downloads the four ShapeSplat bowl objects, matching ModelNet40 OFF
meshes, one SceneSplat-7K ScanNet Gaussian scene, and the ShapeSplat-Part
annotation archive into this directory. Use `--skip-part` to omit the 675 MB
part annotation archive or `--skip-scene` to omit the scene PLY.
