# Local data

Dataset files are intentionally excluded from Git. Download the exact data
used by the report with:

```bash
export HF_TOKEN=hf_...
python scripts/download_data.py
```

The script downloads the four ShapeSplat bowl objects, matching ModelNet40 OFF
meshes, and one SceneSplat-7K ScanNet Gaussian scene into this directory.
