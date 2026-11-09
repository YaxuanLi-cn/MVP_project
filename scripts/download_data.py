#!/usr/bin/env python3
"""Download the datasets used by the report into ``data/``.

No credentials are stored in the repository. Set ``HF_TOKEN`` or pass
``--token``. The script is resumable and skips files that already exist.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_shapesplat import API, RAW, download, request_json


def download_shape_objects(token: str, root: Path, category: str, objects: list[str]) -> None:
    repo = "ShapeSplats/modelnet_3dgs_preprocessed"
    for object_id in objects:
        prefix = f"{category}/test/{object_id}"
        target_root = root / "shapesplat" / category / "test" / object_id
        entries = request_json(API.format(repo=repo, prefix=prefix), token)
        for entry in entries:
            if entry.get("type") != "file" or not entry.get("path", "").endswith(".npy"):
                continue
            rel = entry["path"].split(prefix + "/", 1)[-1]
            target = target_root / rel
            if target.exists():
                continue
            print(f"Downloading {entry['path']}")
            download(RAW.format(repo=repo, path=entry["path"]), target, token)


def download_modelnet_meshes(root: Path, category: str, objects: list[str]) -> None:
    repo = "naderalfares/ModelNet40_Auto_aligned"
    for object_id in objects:
        target = root / "modelnet40" / category / "test" / f"{object_id}.off"
        if target.exists():
            continue
        url = RAW.format(repo=repo, path=f"{category}/test/{object_id}.off")
        print(f"Downloading {repo}:{category}/test/{object_id}.off")
        download(url, target, "")


def download_scene(token: str, root: Path, scene: str) -> None:
    repo = "GaussianWorld/scannet_mcmc_3dgs"
    target = root / "scenesplat" / "scannet_mcmc_3dgs" / scene / "point_cloud_30000.ply"
    if target.exists():
        return
    rel = f"{scene}/ckpts/point_cloud_30000.ply"
    print(f"Downloading {repo}:{rel}")
    download(RAW.format(repo=repo, path=rel), target, token)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--token", default=None)
    p.add_argument("--data-root", default="data")
    p.add_argument("--category", default="bowl")
    p.add_argument("--objects", default="bowl_0065,bowl_0066,bowl_0067,bowl_0068")
    p.add_argument("--scene", default="scene0000_00")
    p.add_argument("--skip-scene", action="store_true")
    args = p.parse_args()
    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or ""
    if not token:
        raise SystemExit("Set HF_TOKEN after accepting the gated dataset terms.")
    root = Path(args.data_root)
    objects = [x.strip() for x in args.objects.split(",") if x.strip()]
    download_shape_objects(token, root, args.category, objects)
    download_modelnet_meshes(root, args.category, objects)
    if not args.skip_scene:
        download_scene(token, root, args.scene)
    print(f"Data is available under {root}")


if __name__ == "__main__":
    main()
