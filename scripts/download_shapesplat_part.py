#!/usr/bin/env python3
"""Download and extract the ShapeSplat-Part/ShapeNet Part annotations."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen
import shutil
import zipfile


MIRROR = "https://omnomnom.vision.rwth-aachen.de/data/point2vec/data/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/shapesplat_part")
    p.add_argument("--url", default=MIRROR)
    args = p.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "partanno.zip"
    if not archive.exists():
        print(f"Downloading ShapeSplat-Part annotations to {archive}", flush=True)
        with urlopen(args.url) as src, open(archive, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
    extracted = root / "extracted"
    marker = extracted / "shapenetcore_partanno_segmentation_benchmark_v0_normal"
    if not marker.exists():
        print(f"Extracting {archive}", flush=True)
        extracted.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extracted)
    print(f"ShapeSplat-Part data is available under {marker}")


if __name__ == "__main__":
    main()
