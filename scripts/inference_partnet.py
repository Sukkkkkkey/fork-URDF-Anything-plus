#!/usr/bin/env python3
"""Batch URDF-Anything+ inference with Particulate-compatible exports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
for module_root in (ROOT, ROOT / "TripoSG"):
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))

from urdf_anything.inference import URDFInference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dino-path", type=Path, required=True)
    parser.add_argument("--triposg-vae-path", type=Path, required=True)
    parser.add_argument(
        "--model-config-path",
        type=Path,
        default=None,
        help="Optional config override; by default use config embedded in checkpoint",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-links", type=int, default=16)
    parser.add_argument("--num-tokens", type=int, default=512)
    parser.add_argument("--eot-threshold", type=float, default=0.5)
    parser.add_argument("--presence-threshold", type=float, default=0.5)
    parser.add_argument("--max-reconstruction-attempts", type=int, default=3)
    parser.add_argument("--overlap-chamfer-threshold", type=float, default=3e-2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def to_numpy(value: torch.Tensor | None, width: int) -> np.ndarray:
    if value is None:
        return np.zeros(width, dtype=np.float32)
    return np.asarray(value.detach().cpu(), dtype=np.float32).reshape(width)


def unit_axis(axis: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (axis / norm).astype(np.float32)


def axis_point_to_plucker(axis: np.ndarray, point: np.ndarray) -> np.ndarray:
    direction = unit_axis(axis)
    return np.concatenate([direction, np.cross(direction, point)]).astype(np.float32)


def save_prediction(result: dict, output_dir: Path, eval_scale: float = 0.5) -> None:
    generated = [item for item in result["results"] if item["pred_mesh"] is not None]
    count = len(generated)
    origins = np.stack([to_numpy(item["param1"], 3) for item in generated])
    axes = np.stack([unit_axis(to_numpy(item["param2"], 3)) for item in generated])
    limits = np.stack([to_numpy(item["param3"], 2) for item in generated])
    ordered_limits = np.sort(limits, axis=1)
    logits = np.stack([to_numpy(item["motion_type"], 2) for item in generated])
    eot_mse = np.asarray([item["eot_mse"] for item in result["results"]], dtype=np.float32)
    np.savez(
        output_dir / "prediction_raw.npz",
        origins=origins,
        axes=axes,
        limits=limits,
        motion_type_logits=logits,
        eot_mse=eot_mse,
    )

    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    vertices = []
    faces = []
    face_part_ids = []
    vertex_offset = 0
    for part_id, item in enumerate(generated):
        mesh = item["pred_mesh"].copy()
        part_vertices = np.clip(
            np.asarray(mesh.vertices, dtype=np.float64) * eval_scale,
            -0.5,
            0.5,
        )
        part_faces = np.asarray(mesh.faces, dtype=np.int64)
        vertices.append(part_vertices)
        faces.append(part_faces + vertex_offset)
        face_part_ids.append(np.full(len(part_faces), part_id, dtype=np.int32))
        vertex_offset += len(part_vertices)
    combined = trimesh.Trimesh(
        vertices=np.concatenate(vertices),
        faces=np.concatenate(faces),
        process=False,
    )
    combined.export(eval_dir / "pred.obj")

    hierarchy = np.asarray([(0, index) for index in range(1, count)], dtype=np.int64)
    if hierarchy.size == 0:
        hierarchy = np.empty((0, 2), dtype=np.int64)
    motion_classes = logits.argmax(axis=1) if count else np.empty(0, dtype=np.int64)
    is_revolute = np.zeros(count, dtype=bool)
    is_prismatic = np.zeros(count, dtype=bool)
    if count > 1:
        is_revolute[1:] = motion_classes[1:] == 0
        is_prismatic[1:] = motion_classes[1:] == 1
    revolute_plucker = np.zeros((count, 6), dtype=np.float32)
    prismatic_axis = np.zeros((count, 3), dtype=np.float32)
    for index in range(1, count):
        if is_revolute[index]:
            revolute_plucker[index] = axis_point_to_plucker(
                axes[index], origins[index] * eval_scale
            )
        elif is_prismatic[index]:
            prismatic_axis[index] = axes[index]
    revolute_range = np.where(
        is_revolute[:, None], ordered_limits, 0.0
    ).astype(np.float32)
    prismatic_range = np.where(
        is_prismatic[:, None], ordered_limits * eval_scale, 0.0
    ).astype(np.float32)
    np.savez(
        eval_dir / "pred.npz",
        face_part_ids=np.concatenate(face_part_ids),
        motion_hierarchy=hierarchy,
        is_part_revolute=is_revolute,
        is_part_prismatic=is_prismatic,
        revolute_plucker=revolute_plucker,
        revolute_range=revolute_range,
        prismatic_axis=prismatic_axis,
        prismatic_range=prismatic_range,
    )


def validate_urdf(output_dir: Path) -> dict:
    urdf_path = output_dir / "generated.urdf"
    root = ET.parse(urdf_path).getroot()
    links = {element.attrib["name"] for element in root.findall("link")}
    joints = root.findall("joint")
    missing_meshes = []
    for mesh in root.findall(".//mesh"):
        mesh_path = output_dir / mesh.attrib["filename"]
        if not mesh_path.is_file():
            missing_meshes.append(str(mesh_path))
    bad_references = []
    for joint in joints:
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        if parent not in links or child not in links:
            bad_references.append(joint.attrib["name"])
    invalid_numbers = []
    for element in root.findall(".//origin") + root.findall(".//axis"):
        for attribute in ("xyz", "rpy"):
            if attribute not in element.attrib:
                continue
            values = np.fromstring(element.attrib[attribute], sep=" ")
            if values.size != 3 or not np.isfinite(values).all():
                invalid_numbers.append(f"{element.tag}.{attribute}")
    for element in root.findall(".//limit"):
        values = np.asarray(
            [float(element.attrib["lower"]), float(element.attrib["upper"])]
        )
        if not np.isfinite(values).all() or values[0] > values[1]:
            invalid_numbers.append("limit")
    if missing_meshes or bad_references or invalid_numbers:
        raise ValueError(
            "invalid URDF: "
            f"missing_meshes={missing_meshes}, bad_references={bad_references}, "
            f"invalid_numbers={invalid_numbers}"
        )
    return {"links_including_base": len(links), "joints_including_fixed": len(joints)}


def set_determinism(seed: int, device: str) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main() -> None:
    args = parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(args.model_path)
    for path in (
        args.triposg_vae_path / "config.json",
        args.triposg_vae_path / "diffusion_pytorch_model.safetensors",
        args.dino_path / "config.json",
        args.dino_path / "preprocessor_config.json",
        args.dino_path / "model.safetensors",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    os.environ["URDF_ANYTHING_TRIPOSG_VAE_PATH"] = str(args.triposg_vae_path)
    set_determinism(args.seed, args.device)
    sample_dirs = sorted(path for path in args.input_root.iterdir() if path.is_dir())
    if args.ids:
        selected = set(args.ids)
        sample_dirs = [path for path in sample_dirs if path.name in selected]
    if args.limit is not None:
        sample_dirs = sample_dirs[: args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)

    inference = URDFInference(
        model_path=str(args.model_path),
        model_config_path=(
            str(args.model_config_path) if args.model_config_path else None
        ),
        dino_path=str(args.dino_path),
        device=args.device,
        num_tokens=args.num_tokens,
        eot_threshold=args.eot_threshold,
        presence_threshold=args.presence_threshold,
        max_reconstruction_attempts=args.max_reconstruction_attempts,
        overlap_chamfer_threshold=args.overlap_chamfer_threshold,
        max_links=args.max_links,
        seed=args.seed,
        vae_deterministic=True,
    )
    items = []
    for index, sample_dir in enumerate(sample_dirs, start=1):
        output_dir = args.output_root / sample_dir.name
        if args.resume and (output_dir / "complete.json").is_file():
            print(f"[{index}/{len(sample_dirs)}] {sample_dir.name}: skipped")
            items.append({"id": sample_dir.name, "status": "skipped_valid"})
            continue
        started = time.time()
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = inference.infer_from_image(
                image_path=str(sample_dir / "whole.png"),
                whole_mesh_path=str(sample_dir / "whole.obj"),
                output_dir=str(output_dir),
            )
            if result["num_links"] < 1:
                raise ValueError("model generated no links")
            save_prediction(result, output_dir)
            validation = validate_urdf(output_dir)
            item = {
                "id": sample_dir.name,
                "status": "ok",
                "num_links": result["num_links"],
                "elapsed_seconds": round(time.time() - started, 3),
                **validation,
            }
            (output_dir / "complete.json").write_text(
                json.dumps(item, indent=2) + "\n", encoding="utf-8"
            )
        except Exception as error:
            item = {
                "id": sample_dir.name,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": round(time.time() - started, 3),
            }
        items.append(item)
        print(f"[{index}/{len(sample_dirs)}] {sample_dir.name}: {item['status']}")
        (args.output_root / "manifest.json").write_text(
            json.dumps({"items": items}, indent=2) + "\n", encoding="utf-8"
        )
    failed = sum(item["status"] == "failed" for item in items)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
