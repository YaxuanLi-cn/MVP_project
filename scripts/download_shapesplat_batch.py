#!/usr/bin/env python3
"""Download a multi-category ShapeSplat benchmark subset."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import sys
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_shapesplat import API, RAW, download, request_json


REPO = "ShapeSplats/modelnet_3dgs_preprocessed"
MESH_REPO = "naderalfares/ModelNet40_Auto_aligned"


def list_test_objects(category: str, limit: int) -> list[str]:
    try:
        entries = request_json(API.format(repo=REPO, prefix=f"{category}/test"), "")
    except HTTPError as exc:
        if exc.code == 404:
            print(f"Skipping {category}: no test split", flush=True)
            return []
        raise
    ids = [e["path"].split("/")[-1] for e in entries if e.get("type") == "directory"]
    return sorted(ids)[:limit]


def download_object(category: str, object_id: str, root: Path, token: str) -> str:
    prefix = f"{category}/test/{object_id}"
    target_root = root / category / "test" / object_id
    entries = request_json(API.format(repo=REPO, prefix=prefix), token)
    for entry in entries:
        if entry.get("type") != "file" or not entry.get("path", "").endswith(".npy"):
            continue
        rel = entry["path"].split(prefix + "/", 1)[-1]
        target = target_root / rel
        if not target.exists():
            download(RAW.format(repo=REPO, path=entry["path"]), target, token)
    return f"{category}/{object_id}"


def download_mesh(category: str, object_id: str, root: Path) -> str:
    target = root / category / "test" / f"{object_id}.off"
    if not target.exists():
        download(RAW.format(repo=MESH_REPO, path=f"{category}/test/{object_id}.off"), target, "")
    return f"{category}/{object_id}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--categories", default="airplane,bottle,bowl,car,chair,cone,cup,desk,lamp,table")
    p.add_argument("--objects-per-category", type=int, default=8)
    p.add_argument("--data-root", default="data/shapesplat")
    p.add_argument("--mesh-root", default="data/modelnet40")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--token", default=None)
    args = p.parse_args()
    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or ""
    if not token:
        raise SystemExit("Set HF_TOKEN after accepting the ShapeSplats dataset terms.")
    root = Path(args.data_root)
    jobs = []
    for category in [x.strip() for x in args.categories.split(",") if x.strip()]:
        for object_id in list_test_objects(category, args.objects_per_category):
            jobs.append((category, object_id))
    print(f"Downloading {len(jobs)} ShapeSplat objects", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_object, c, o, root, token) for c, o in jobs]
        for future in as_completed(futures):
            print(f"Downloaded {future.result()}", flush=True)
        mesh_futures = [pool.submit(download_mesh, c, o, Path(args.mesh_root)) for c, o in jobs]
        for future in as_completed(mesh_futures):
            print(f"Downloaded mesh {future.result()}", flush=True)
    print(f"Data is available under {root}")


if __name__ == "__main__":
    main()
