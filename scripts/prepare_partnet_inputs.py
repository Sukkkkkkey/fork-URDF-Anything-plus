#!/usr/bin/env python3
"""Prepare the Particulate PartNet-Mobility test split for inference.

The source PartNet archives remain read-only.  Meshes and ground truth are
copied from the already validated Particulate preprocessing, while canonical
whole-object images are reconstructed from the per-part highlight renders in
the original archives.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
from typing import Iterable
import zipfile

import numpy as np
from PIL import Image
import trimesh


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--particulate-assets-root", type=Path, required=True)
    parser.add_argument("--particulate-gt-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--mesh-scale", type=float, default=2.0)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--blender", type=Path, default=shutil.which("blender"))
    parser.add_argument("--xvfb-run", type=Path, default=shutil.which("xvfb-run"))
    parser.add_argument(
        "--blender-render-script",
        type=Path,
        default=ROOT / "scripts/blender_render_obj.py",
    )
    return parser.parse_args()


def scale_obj(source: Path, destination: Path, scale: float) -> None:
    lines = []
    with source.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.split()
            if fields and fields[0] == "v" and len(fields) >= 4:
                xyz = [float(value) * scale for value in fields[1:4]]
                suffix = "" if len(fields) == 4 else " " + " ".join(fields[4:])
                line = f"v {xyz[0]:.9g} {xyz[1]:.9g} {xyz[2]:.9g}{suffix}\n"
            lines.append(line)
    destination.write_text("".join(lines), encoding="utf-8")


def render_names(archive: zipfile.ZipFile, object_id: str) -> list[str]:
    prefix = f"{object_id}/parts_render_after_merging/"
    return sorted(
        name
        for name in archive.namelist()
        if name.startswith(prefix) and name.lower().endswith(".png")
    )


def neutral_render(archive_path: Path, object_id: str) -> tuple[Image.Image, int]:
    with zipfile.ZipFile(archive_path) as archive:
        names = render_names(archive, object_id)
        if not names:
            raise FileNotFoundError(
                f"no parts_render_after_merging PNG files in {archive_path}"
            )
        images = [
            np.asarray(Image.open(BytesIO(archive.read(name))).convert("RGB"))
            for name in names
        ]
    shape = images[0].shape
    if any(image.shape != shape for image in images):
        raise ValueError(f"render sizes differ in {archive_path}")
    stack = np.stack(images, axis=0)
    # Highlighted parts are red; the largest green+blue value selects the
    # neutral rendering of each pixel from one of the other part images.
    scores = stack[..., 1].astype(np.uint16) + stack[..., 2].astype(np.uint16)
    best = scores.argmax(axis=0)
    rows, cols = np.indices(best.shape)
    neutral = stack[best, rows, cols]
    return Image.fromarray(neutral, mode="RGB"), len(names)


def copy_textures(source: Path, destination: Path) -> None:
    if not source.is_dir():
        return
    shutil.copytree(source, destination, dirs_exist_ok=True)


def load_entries(split_file: Path, split: str) -> Iterable[tuple[str, str]]:
    split_data = json.loads(split_file.read_text(encoding="utf-8"))
    if split not in split_data:
        raise KeyError(f"split {split!r} not found in {split_file}")
    for entry in split_data[split]:
        category, object_id = entry.split("/", maxsplit=1)
        yield category, object_id


def prepare_one(args: argparse.Namespace, category: str, object_id: str) -> dict:
    source_dir = args.particulate_assets_root / object_id
    source_obj = source_dir / "original.obj"
    source_mtl = source_dir / "original.mtl"
    source_gt = args.particulate_gt_root / f"{object_id}.npz"
    archive_path = args.raw_root / f"{object_id}.zip"
    for required in (source_obj, source_mtl, source_gt, archive_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    target_dir = args.output_root / args.split / object_id
    expected = (
        target_dir / "whole.obj",
        target_dir / "whole.png",
        target_dir / "original.mtl",
        target_dir / "info.json",
        args.output_root / "gt" / f"{object_id}.npz",
    )
    if not args.overwrite and all(path.is_file() for path in expected):
        existing = json.loads((target_dir / "info.json").read_text(encoding="utf-8"))
        return {**existing, "status": "skipped_valid"}

    target_dir.mkdir(parents=True, exist_ok=True)
    scale_obj(source_obj, target_dir / "whole.obj", args.mesh_scale)
    shutil.copy2(source_mtl, target_dir / "original.mtl")
    copy_textures(source_dir / "textures", target_dir / "textures")
    try:
        image, render_count = neutral_render(archive_path, object_id)
        image.save(target_dir / "whole.png")
        image_method = "per-pixel max green+blue over part-highlight renders"
    except FileNotFoundError:
        if args.blender is None:
            raise RuntimeError(
                f"{object_id} has no source render and Blender was not found"
            )
        blender_command = [
            str(args.blender),
            "-b",
            "--python",
            str(args.blender_render_script),
            "--",
            "--input",
            str(target_dir / "whole.obj"),
            "--output",
            str(target_dir / "whole.png"),
        ]
        if args.xvfb_run is not None:
            blender_command = [str(args.xvfb_run), "-a", *blender_command]
        subprocess.run(blender_command, check=True)
        rendered = Image.open(target_dir / "whole.png").convert("RGBA")
        white = Image.new("RGBA", rendered.size, (255, 255, 255, 255))
        Image.alpha_composite(white, rendered).convert("RGB").save(
            target_dir / "whole.png"
        )
        render_count = 0
        image_method = "Blender canonical fallback (source archive has no renders)"

    gt_dir = args.output_root / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_gt, gt_dir / f"{object_id}.npz")

    mesh = trimesh.load(target_dir / "whole.obj", force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError(f"invalid mesh generated for {object_id}")
    max_extent = float(mesh.extents.max())
    if not np.isclose(max_extent, 2.0, atol=1e-5):
        raise ValueError(f"unexpected max extent for {object_id}: {max_extent}")
    info = {
        "id": object_id,
        "category": category,
        "source_archive": str(archive_path),
        "source_mesh": str(source_obj),
        "source_gt": str(source_gt),
        "mesh_scale_from_particulate": args.mesh_scale,
        "bounds": mesh.bounds.tolist(),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "source_render_count": render_count,
        "image_method": image_method,
    }
    (target_dir / "info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {**info, "status": "ok"}


def main() -> None:
    args = parse_args()
    entries = list(load_entries(args.split_file, args.split))
    if args.ids:
        selected = set(args.ids)
        entries = [entry for entry in entries if entry[1] in selected]
    if args.limit is not None:
        entries = entries[: args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)
    items = []
    for index, (category, object_id) in enumerate(entries, start=1):
        try:
            item = prepare_one(args, category, object_id)
            print(f"[{index}/{len(entries)}] {object_id}: {item['status']}")
        except Exception as error:
            item = {
                "id": object_id,
                "category": category,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
            print(f"[{index}/{len(entries)}] {object_id}: {item['error']}")
        items.append(item)
    manifest = {
        "raw_root": str(args.raw_root),
        "particulate_assets_root": str(args.particulate_assets_root),
        "particulate_gt_root": str(args.particulate_gt_root),
        "split_file": str(args.split_file),
        "split": args.split,
        "requested": len(entries),
        "succeeded": sum(item["status"] in {"ok", "skipped_valid"} for item in items),
        "failed": sum(item["status"] == "failed" for item in items),
        "items": items,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if manifest["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
