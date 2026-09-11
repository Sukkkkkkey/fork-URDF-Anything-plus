#!/usr/bin/env python3
"""Prepare PartNet meshes in the URDF-Anything+ model coordinate frame.

PartNet-Mobility uses +Z as up and -X as the canonical front direction.
URDF-Anything+ expects +Y as up and +Z as front.  Its generated URDF applies
the inverse conversion as the fixed base-joint rotation
``rpy="pi/2 0 -pi/2"``.  This probe applies the corresponding source-to-model
conversion before inference:

    (x, y, z)_partnet -> (-y, z, -x)_model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import trimesh


PARTNET_TO_MODEL = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def transform_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Return a copy of ``mesh`` rotated from PartNet to model coordinates."""
    result = mesh.copy()
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = PARTNET_TO_MODEL
    result.apply_transform(transform)
    return result


def prepare_sample(input_root: Path, output_root: Path, sample_id: str) -> dict:
    source_dir = input_root / sample_id
    source_mesh = source_dir / "whole.obj"
    source_image = source_dir / "whole.png"
    if not source_mesh.is_file():
        raise FileNotFoundError(source_mesh)
    if not source_image.is_file():
        raise FileNotFoundError(source_image)

    target_dir = output_root / sample_id
    target_dir.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(source_mesh, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError(f"invalid mesh: {source_mesh}")
    transformed = transform_mesh(mesh)
    transformed.export(target_dir / "whole.obj")
    shutil.copy2(source_image, target_dir / "whole.png")

    metadata = {
        "id": sample_id,
        "source_mesh": str(source_mesh),
        "source_image": str(source_image),
        "coordinate_frame": "URDF-Anything+ model frame: +Y up, +Z front",
        "partnet_to_model_xyz": "(-y, z, -x)",
        "partnet_to_model_matrix": PARTNET_TO_MODEL.tolist(),
        "source_bounds": np.asarray(mesh.bounds).tolist(),
        "transformed_bounds": np.asarray(transformed.bounds).tolist(),
        "vertices": int(len(transformed.vertices)),
        "faces": int(len(transformed.faces)),
    }
    (target_dir / "orientation.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--ids",
        nargs="*",
        help="Sample IDs; omit to prepare every sample directory under input-root",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip samples with an existing mesh, image, and orientation metadata",
    )
    args = parser.parse_args()

    ids = args.ids or sorted(path.name for path in args.input_root.iterdir() if path.is_dir())
    if not ids:
        raise ValueError(f"no samples found under {args.input_root}")
    results = []
    for sample_id in ids:
        target_dir = args.output_root / sample_id
        expected = (
            target_dir / "whole.obj",
            target_dir / "whole.png",
            target_dir / "orientation.json",
        )
        if args.resume and all(path.is_file() for path in expected):
            metadata = json.loads(expected[-1].read_text(encoding="utf-8"))
            results.append({**metadata, "status": "skipped_valid"})
        else:
            results.append(
                {
                    **prepare_sample(args.input_root, args.output_root, sample_id),
                    "status": "ok",
                }
            )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
