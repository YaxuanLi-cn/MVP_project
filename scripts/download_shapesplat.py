#!/usr/bin/env python3
"""Download a small real ShapeSplat object from Hugging Face.

The ShapeSplats repositories are gated. Set ``HF_TOKEN`` (or
``HUGGINGFACE_HUB_TOKEN``) after accepting the dataset terms on Hugging Face.
Only the selected object's files are downloaded, rather than a multi-GB class
archive.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


API = "https://huggingface.co/api/datasets/{repo}/tree/main/{prefix}?recursive=true&expand=false"
RAW = "https://huggingface.co/datasets/{repo}/resolve/main/{path}?download=true"


def request_json(url: str, token: str) -> list[dict]:
    req = Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    try:
        with urlopen(req) as response:
            return json.load(response)
    except HTTPError as exc:
        if exc.code == 401:
            raise RuntimeError("Hugging Face rejected the token. Accept the ShapeSplats dataset terms and check HF_TOKEN.") from exc
        raise


def download(url: str, target: Path, token: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    try:
        with urlopen(req) as response, open(target, "wb") as out:
            total = response.headers.get("Content-Length")
            done = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
                done += len(block)
                if total and done % (32 * 1024 * 1024) < len(block):
                    print(f"  {target.name}: {done / int(total):.0%}", flush=True)
    except HTTPError as exc:
        if exc.code == 401:
            raise RuntimeError("Hugging Face rejected the token while downloading. Check dataset access and HF_TOKEN.") from exc
        raise


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="ShapeSplats/modelnet_3dgs_preprocessed")
    p.add_argument("--category", default="bowl")
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--object", dest="object_id", default="bowl_0065")
    p.add_argument("--output", default="data/shapesplat")
    p.add_argument("--token", default=None, help="defaults to HF_TOKEN/HUGGINGFACE_HUB_TOKEN")
    args = p.parse_args()
    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or ""
    if not token:
        raise SystemExit("Real ShapeSplat files are gated. Set HF_TOKEN after accepting the dataset terms on Hugging Face.")
    prefix = f"{args.category}/{args.split}/{args.object_id}"
    files = request_json(API.format(repo=args.repo, prefix=prefix), token)
    files = [x for x in files if x.get("type") == "file" and x.get("path", "").endswith(".npy")]
    if not files:
        raise SystemExit(f"No .npy files found under {prefix}; check category/split/object.")
    root = Path(args.output) / args.category / args.split / args.object_id
    for entry in files:
        rel = entry["path"].split(prefix + "/", 1)[-1]
        target = root / rel
        print(f"Downloading {entry['path']} ({entry.get('size', '?')} bytes)")
        download(RAW.format(repo=args.repo, path=entry["path"]), target, token)
    print(f"Downloaded {len(files)} files to {root}")


if __name__ == "__main__":
    main()
