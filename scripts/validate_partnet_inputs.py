#!/usr/bin/env python3
"""Validate prepared PartNet-Mobility images, meshes, metadata, and GT."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh


GT_KEYS = {
    "points",
    "part_ids",
    "motion_hierarchy",
    "is_part_revolute",
    "is_part_prismatic",
    "revolute_plucker",
    "revolute_range",
    "prismatic_axis",
    "prismatic_range",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--expected-count", type=int, default=77)
    args = parser.parse_args()

    sample_dirs = sorted(path for path in (args.root / args.split).iterdir() if path.is_dir())
    errors = []
    categories = {}
    fallbacks = []
    for sample_dir in sample_dirs:
        paths = {
            "mesh": sample_dir / "whole.obj",
            "image": sample_dir / "whole.png",
            "info": sample_dir / "info.json",
            "gt": args.root / "gt" / f"{sample_dir.name}.npz",
        }
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            errors.append({"id": sample_dir.name, "missing": missing})
            continue
        info = json.loads(paths["info"].read_text(encoding="utf-8"))
        categories[info["category"]] = categories.get(info["category"], 0) + 1
        if info["source_render_count"] == 0:
            fallbacks.append(sample_dir.name)
        with Image.open(paths["image"]) as image:
            image.verify()
        mesh = trimesh.load(paths["mesh"], force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
            errors.append({"id": sample_dir.name, "error": "empty mesh"})
        elif not np.isclose(mesh.extents.max(), 2.0, atol=1e-5):
            errors.append(
                {"id": sample_dir.name, "max_extent": float(mesh.extents.max())}
            )
        with np.load(paths["gt"]) as ground_truth:
            missing_gt = sorted(GT_KEYS - set(ground_truth.files))
            if missing_gt:
                errors.append({"id": sample_dir.name, "missing_gt": missing_gt})

    manifest = json.loads((args.root / "manifest.json").read_text(encoding="utf-8"))
    summary = {
        "samples": len(sample_dirs),
        "gt_files": len(list((args.root / "gt").glob("*.npz"))),
        "manifest_requested": manifest["requested"],
        "manifest_succeeded": manifest["succeeded"],
        "manifest_failed": manifest["failed"],
        "categories": categories,
        "fallback_images": fallbacks,
        "errors": errors,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if (
        errors
        or len(sample_dirs) != args.expected_count
        or manifest["succeeded"] != args.expected_count
        or manifest["failed"] != 0
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
