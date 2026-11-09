#!/usr/bin/env python3
"""Download one SceneSplat-7K scene Gaussian PLY.

The SceneSplat-7K index is published through GaussianWorld repositories. The
repository may require a separate Hugging Face dataset approval even when the
ShapeSplats token works.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_shapesplat import RAW, download
from urllib.error import HTTPError


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="GaussianWorld/scannet_mcmc_3dgs")
    p.add_argument("--scene", default="scene0000_00")
    p.add_argument("--output", default="data/scenesplat")
    p.add_argument("--token", default=None)
    args = p.parse_args()
    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or ""
    if not token:
        raise SystemExit("Set HF_TOKEN after accepting the SceneSplat-7K repository terms.")
    rel = f"{args.scene}/ckpts/point_cloud_30000.ply"
    target = Path(args.output) / args.repo.split("/")[-1] / args.scene / "point_cloud_30000.ply"
    print(f"Downloading {args.repo}:{rel}")
    try:
        download(RAW.format(repo=args.repo, path=rel), target, token)
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise SystemExit(
                f"Access to {args.repo} is still restricted (HTTP {exc.code}). "
                "Accept the SceneSplat-7K terms on the main page and the "
                "component repository's own gated terms, then retry."
            ) from exc
        raise
    print(target)


if __name__ == "__main__":
    main()
