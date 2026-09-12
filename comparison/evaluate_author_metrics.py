#!/usr/bin/env python3
"""Evaluate PartNet-Mobility predictions with the URDF-Anything+ metric family.

The paper defines IoU, F-score, squared-L2 Chamfer distance, and revolute
joint axis/origin/limit errors, but does not publish its evaluation code or all
discretization details.  This script therefore makes every local protocol
choice explicit in the output metadata and reports coverage alongside errors.
The result is a local operationalization, not the authors' unreleased protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Iterable
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh


DEFAULT_DOMAIN = (-1.01, 1.01)
DEFAULT_CD_MISSING_PENALTY = 12.0
PROTOCOL_VERSION = "comparison-local-v4-dual-iou"
PART_GEOMETRY_METRICS = ("iou_surface", "iou_volume", "fscore", "cd")
WHOLE_GEOMETRY_METRICS = PART_GEOMETRY_METRICS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", choices=("urdf-anything-plus", "particulate"), required=True
    )
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--points-per-part", type=int, default=100_000)
    parser.add_argument("--points-whole", type=int, default=100_000)
    parser.add_argument("--matching-points", type=int, default=2_048)
    parser.add_argument("--kd-workers", type=int, default=8)
    parser.add_argument("--fscore-threshold", type=float, default=0.02)
    parser.add_argument("--voxel-resolution", type=int, default=128)
    parser.add_argument(
        "--dense-vertex-face-threshold",
        type=int,
        default=100_000,
        help="Use dense mesh vertices directly above this face count",
    )
    parser.add_argument("--domain-min", type=float, default=DEFAULT_DOMAIN[0])
    parser.add_argument("--domain-max", type=float, default=DEFAULT_DOMAIN[1])
    parser.add_argument(
        "--missing-cd-penalty", type=float, default=DEFAULT_CD_MISSING_PENALTY
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--urdf-coordinate",
        choices=("raw", "world"),
        default="raw",
        help=(
            "Coordinate frame for URDF-Anything+ predictions; world applies "
            "the generated URDF fixed base-joint transform"
        ),
    )
    return parser.parse_args()


def stable_seed(seed: int, *components: str) -> int:
    digest = hashlib.sha256(
        (str(seed) + "\0" + "\0".join(components)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def unit_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"invalid axis vector: {vector.tolist()}")
    return vector / norm


def plucker_axis_point(plucker: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unit direction and the closest point on a Pluecker line."""
    plucker = np.asarray(plucker, dtype=np.float64).reshape(6)
    direction_raw = plucker[:3]
    norm = float(np.linalg.norm(direction_raw))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"invalid Pluecker direction: {direction_raw.tolist()}")
    direction = direction_raw / norm
    moment = plucker[3:] / norm
    point = np.cross(moment, direction)
    return direction, point


def axis_error(axis_pred: np.ndarray, axis_gt: np.ndarray) -> float:
    pred = unit_vector(axis_pred)
    gt = unit_vector(axis_gt)
    return float(np.arccos(np.clip(abs(float(np.dot(pred, gt))), 0.0, 1.0)))


def line_distance(
    point_pred: np.ndarray,
    axis_pred: np.ndarray,
    point_gt: np.ndarray,
    axis_gt: np.ndarray,
) -> float:
    """Shortest Euclidean distance between two infinite 3D lines."""
    pred_axis = unit_vector(axis_pred)
    gt_axis = unit_vector(axis_gt)
    delta = np.asarray(point_gt, dtype=np.float64) - np.asarray(
        point_pred, dtype=np.float64
    )
    normal = np.cross(pred_axis, gt_axis)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-10:
        return float(np.linalg.norm(np.cross(delta, pred_axis)))
    return float(abs(np.dot(delta, normal)) / normal_norm)


def limit_error(
    axis_pred: np.ndarray,
    limits_pred: np.ndarray,
    axis_gt: np.ndarray,
    limits_gt: np.ndarray,
) -> float:
    """Articulate-Anything motion-vector limit error with axis sign handling."""
    pred_axis = unit_vector(axis_pred)
    gt_axis = unit_vector(axis_gt)
    pred_limits = np.sort(np.asarray(limits_pred, dtype=np.float64).reshape(2))
    gt_limits = np.sort(np.asarray(limits_gt, dtype=np.float64).reshape(2))
    pred_motion = pred_axis * float(pred_limits[1] - pred_limits[0])
    gt_motion = gt_axis * float(gt_limits[1] - gt_limits[0])
    return float(
        min(
            np.linalg.norm(pred_motion - gt_motion),
            np.linalg.norm(-pred_motion - gt_motion),
        )
    )


def point_metrics(
    points_pred: np.ndarray,
    points_gt: np.ndarray,
    threshold: float,
    workers: int = 1,
) -> dict[str, float]:
    if len(points_pred) == 0 or len(points_gt) == 0:
        raise ValueError("point metrics require two non-empty point sets")
    pred = np.asarray(points_pred, dtype=np.float64)
    gt = np.asarray(points_gt, dtype=np.float64)
    distance_pred = cKDTree(gt).query(pred, workers=workers)[0]
    distance_gt = cKDTree(pred).query(gt, workers=workers)[0]
    precision = float(np.mean(distance_pred < threshold))
    recall = float(np.mean(distance_gt < threshold))
    denominator = precision + recall
    fscore = 0.0 if denominator == 0.0 else 2.0 * precision * recall / denominator
    chamfer = float(np.mean(distance_pred**2) + np.mean(distance_gt**2))
    return {
        "fscore": fscore,
        "cd": chamfer,
        "precision": precision,
        "recall": recall,
    }


def face_majority_labels(vertex_labels: np.ndarray, faces: np.ndarray) -> np.ndarray:
    labels = np.asarray(vertex_labels)[np.asarray(faces)]
    if labels.shape[1] != 3:
        raise ValueError("only triangular faces are supported")
    a, b, c = labels[:, 0], labels[:, 1], labels[:, 2]
    return np.where(a == b, a, np.where(a == c, a, np.where(b == c, b, a)))


def load_obj_raw_preserve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read OBJ geometry without allowing material groups to reorder it."""
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                fields = line.split()
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
            elif line.startswith("f "):
                indices = [int(token.split("/")[0]) - 1 for token in line.split()[1:]]
                for index in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[index], indices[index + 1]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def load_motion(path: Path, linear_scale: float) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        result = {key: np.array(archive[key], copy=True) for key in archive.files}
    if "revolute_plucker" not in result:
        raise KeyError(f"{path} has no revolute_plucker")
    result["revolute_plucker"] = np.asarray(
        result["revolute_plucker"], dtype=np.float64
    )
    result["revolute_plucker"][:, 3:] *= linear_scale
    return result


def load_gt(
    sample_id: str, asset_root: Path, gt_root: Path
) -> tuple[trimesh.Trimesh, np.ndarray, dict[str, np.ndarray]]:
    asset_dir = asset_root / sample_id
    vertices, faces = load_obj_raw_preserve(asset_dir / "original.obj")
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    with np.load(asset_dir / "meta.npz", allow_pickle=False) as archive:
        vertex_labels = np.asarray(archive["vert_to_bone"], dtype=np.int64)
    labels = face_majority_labels(vertex_labels, mesh.faces).astype(np.int64)
    mesh.apply_scale(2.0)
    motion = load_motion(gt_root / f"{sample_id}.npz", linear_scale=2.0)
    return mesh, labels, motion


def link_sort_key(path: Path) -> int:
    match = re.fullmatch(r"link_(\d+)\.obj", path.name)
    if match is None:
        raise ValueError(f"unexpected link filename: {path.name}")
    return int(match.group(1))


def load_urdf_root_transform(sample_dir: Path) -> np.ndarray:
    """Load the fixed base-joint transform from a generated URDF."""
    urdf_path = sample_dir / "generated.urdf"
    root = ET.parse(urdf_path).getroot()
    candidates = []
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        if joint.get("type") != "fixed" or parent is None:
            continue
        if parent.get("link") == "base":
            candidates.append(joint)
    if len(candidates) != 1:
        raise ValueError(
            f"expected one fixed root joint under {urdf_path}, found {len(candidates)}"
        )
    origin = candidates[0].find("origin")
    xyz = np.fromstring(
        "0 0 0" if origin is None else origin.get("xyz", "0 0 0"), sep=" "
    )
    rpy = np.fromstring(
        "0 0 0" if origin is None else origin.get("rpy", "0 0 0"), sep=" "
    )
    if xyz.shape != (3,) or rpy.shape != (3,):
        raise ValueError(f"invalid root origin in {urdf_path}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = xyz
    return transform


def transform_motion_to_world(
    motion: dict[str, np.ndarray], transform: np.ndarray
) -> dict[str, np.ndarray]:
    """Transform predicted axes and Pluecker lines by a rigid transform."""
    result = {key: np.array(value, copy=True) for key, value in motion.items()}
    rotation = np.asarray(transform[:3, :3], dtype=np.float64)
    translation = np.asarray(transform[:3, 3], dtype=np.float64)
    if "revolute_plucker" in result:
        pluckers = np.asarray(result["revolute_plucker"], dtype=np.float64)
        transformed = np.array(pluckers, copy=True)
        for index, plucker in enumerate(pluckers):
            if np.linalg.norm(plucker[:3]) < 1e-12:
                continue
            axis, point = plucker_axis_point(plucker)
            world_axis = rotation @ axis
            world_point = rotation @ point + translation
            transformed[index, :3] = world_axis
            transformed[index, 3:] = np.cross(world_axis, world_point)
        result["revolute_plucker"] = transformed
    if "prismatic_axis" in result:
        axes = np.asarray(result["prismatic_axis"], dtype=np.float64)
        result["prismatic_axis"] = axes @ rotation.T
    return result


def load_urdf_prediction(
    sample_id: str, prediction_root: Path, coordinate: str = "raw"
) -> tuple[trimesh.Trimesh, np.ndarray, dict[str, np.ndarray]]:
    sample_dir = prediction_root / sample_id
    paths = sorted(sample_dir.glob("link_*.obj"), key=link_sort_key)
    if not paths:
        raise FileNotFoundError(f"no link meshes under {sample_dir}")
    meshes: list[trimesh.Trimesh] = []
    labels: list[np.ndarray] = []
    for path in paths:
        part_id = link_sort_key(path)
        mesh = trimesh.load_mesh(path, process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"prediction did not load as one mesh: {path}")
        meshes.append(mesh)
        labels.append(np.full(len(mesh.faces), part_id, dtype=np.int64))
    combined = trimesh.util.concatenate(meshes)
    motion = load_motion(sample_dir / "eval" / "pred.npz", linear_scale=2.0)
    if coordinate == "world":
        transform = load_urdf_root_transform(sample_dir)
        combined.apply_transform(transform)
        motion = transform_motion_to_world(motion, transform)
    elif coordinate != "raw":
        raise ValueError(f"unsupported URDF prediction coordinate: {coordinate}")
    return combined, np.concatenate(labels), motion


def load_particulate_prediction(
    sample_id: str, prediction_root: Path
) -> tuple[trimesh.Trimesh, np.ndarray, dict[str, np.ndarray]]:
    eval_dir = prediction_root / sample_id / "eval"
    vertices, faces = load_obj_raw_preserve(eval_dir / "pred.obj")
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    motion = load_motion(eval_dir / "pred.npz", linear_scale=2.0)
    labels = np.asarray(motion["face_part_ids"], dtype=np.int64)
    mesh.apply_scale(2.0)
    return mesh, labels, motion


def validate_mesh_labels(mesh: trimesh.Trimesh, labels: np.ndarray, name: str) -> None:
    if len(mesh.faces) != len(labels):
        raise ValueError(f"{name}: {len(mesh.faces)} faces but {len(labels)} labels")
    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        raise ValueError(f"{name}: empty mesh")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError(f"{name}: non-finite vertices")
    if np.any(labels < 0):
        raise ValueError(f"{name}: negative part label")


def sample_faces(
    mesh: trimesh.Trimesh,
    face_indices: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    face_indices = np.asarray(face_indices, dtype=np.int64)
    if len(face_indices) == 0:
        return np.empty((0, 3), dtype=np.float64)
    areas = np.asarray(mesh.area_faces[face_indices], dtype=np.float64)
    area_sum = float(np.sum(areas))
    probabilities = None if area_sum <= 0.0 else areas / area_sum
    selected = rng.choice(face_indices, size=count, replace=True, p=probabilities)
    triangles = np.asarray(mesh.vertices[mesh.faces[selected]], dtype=np.float64)
    random_values = rng.random((count, 2))
    sqrt_u = np.sqrt(random_values[:, 0])
    barycentric = np.column_stack(
        (1.0 - sqrt_u, sqrt_u * (1.0 - random_values[:, 1]), sqrt_u * random_values[:, 1])
    )
    return np.einsum("ni,nij->nj", barycentric, triangles)


def sample_geometry(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    points_per_part: int,
    points_whole: int,
    seed: int,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    part_points = {
        int(part_id): sample_faces(
            mesh, np.flatnonzero(labels == part_id), points_per_part, rng
        )
        for part_id in np.unique(labels)
    }
    whole_points = sample_faces(mesh, np.arange(len(mesh.faces)), points_whole, rng)
    return part_points, whole_points


def fill_volume_voxels(
    surface_voxels: np.ndarray, resolution: int
) -> np.ndarray:
    """Fill a surface grid using Trimesh's orthographic solid criterion.

    A cell is interior when an occupied surface cell exists in both directions
    along all three coordinate axes.  This is stable for the small topological
    holes present in the PartNet CAD meshes, for which flood filling an
    allegedly watertight shell is not reliable.
    """
    indices = np.asarray(surface_voxels, dtype=np.int64)
    if resolution < 1:
        raise ValueError("resolution must be positive")
    voxel_count = resolution**3
    if np.any(indices < 0) or np.any(indices >= voxel_count):
        raise ValueError("surface voxel index outside evaluation grid")
    if len(indices) == 0:
        return indices.copy()

    surface = np.zeros((resolution,) * 3, dtype=bool)
    surface.flat[indices] = True
    volume = np.ones_like(surface)
    for axis in range(3):
        from_low = np.logical_or.accumulate(surface, axis=axis)
        reversed_surface = np.flip(surface, axis=axis)
        from_high = np.flip(
            np.logical_or.accumulate(reversed_surface, axis=axis), axis=axis
        )
        volume &= from_low & from_high
    return np.flatnonzero(volume).astype(np.int64, copy=False)


def union_voxel_parts(voxel_parts: dict[int, np.ndarray]) -> np.ndarray:
    nonempty = [indices for indices in voxel_parts.values() if len(indices)]
    return (
        np.unique(np.concatenate(nonempty)).astype(np.int64)
        if nonempty
        else np.empty(0, dtype=np.int64)
    )


def voxel_occupancy_variants(
    surface_parts: dict[int, np.ndarray], resolution: int
) -> tuple[
    dict[int, np.ndarray],
    np.ndarray,
    dict[int, np.ndarray],
    np.ndarray,
]:
    volume_parts = {
        part_id: fill_volume_voxels(indices, resolution)
        for part_id, indices in surface_parts.items()
    }
    return (
        surface_parts,
        union_voxel_parts(surface_parts),
        volume_parts,
        union_voxel_parts(volume_parts),
    )


def voxelize_surface_and_volume(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    resolution: int,
    domain_min: float,
    domain_max: float,
    dense_vertex_face_threshold: int,
) -> tuple[
    dict[int, np.ndarray],
    np.ndarray,
    dict[int, np.ndarray],
    np.ndarray,
]:
    """Rasterize labeled surfaces once and return surface and volume occupancy."""
    pitch = (domain_max - domain_min) / resolution
    if len(mesh.faces) >= dense_vertex_face_threshold:
        indices = np.floor((np.asarray(mesh.vertices) - domain_min) / pitch).astype(
            np.int64
        )
        valid = np.all((indices >= 0) & (indices < resolution), axis=1)
        linear = np.full(len(indices), -1, dtype=np.int64)
        linear[valid] = np.ravel_multi_index(
            tuple(indices[valid].T), (resolution,) * 3
        )
        unique_parts: dict[int, np.ndarray] = {}
        for part_id in np.unique(labels):
            vertex_indices = np.unique(mesh.faces[labels == part_id].reshape(-1))
            part_linear = linear[vertex_indices]
            unique_parts[int(part_id)] = np.unique(part_linear[part_linear >= 0])
        return voxel_occupancy_variants(unique_parts, resolution)

    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
    except ImportError as error:
        raise RuntimeError(
            "volume voxelization requires embreex; run in the particulate conda environment"
        ) from error

    if domain_max <= domain_min:
        raise ValueError("domain-max must be greater than domain-min")
    coordinates = domain_min + (np.arange(resolution) + 0.5) * pitch
    grid_a, grid_b = np.meshgrid(coordinates, coordinates, indexing="ij")
    cross_section = np.column_stack((grid_a.ravel(), grid_b.ravel()))
    intersector = RayMeshIntersector(mesh)
    voxel_parts: dict[int, list[np.ndarray]] = {
        int(part_id): [] for part_id in np.unique(labels)
    }

    for axis in range(3):
        perpendicular = [dimension for dimension in range(3) if dimension != axis]
        origins = np.empty((len(cross_section), 3), dtype=np.float64)
        origins[:, axis] = domain_min - 2.0 * pitch
        origins[:, perpendicular] = cross_section
        directions = np.zeros_like(origins)
        directions[:, axis] = 1.0
        locations, _, face_indices = intersector.intersects_location(
            origins, directions, multiple_hits=True
        )
        if len(locations) == 0:
            continue
        indices = np.floor((locations - domain_min) / pitch).astype(np.int64)
        valid = np.all((indices >= 0) & (indices < resolution), axis=1)
        indices = indices[valid]
        hit_labels = labels[np.asarray(face_indices, dtype=np.int64)[valid]]
        linear = np.ravel_multi_index(tuple(indices.T), (resolution,) * 3)
        for part_id in np.unique(hit_labels):
            voxel_parts[int(part_id)].append(linear[hit_labels == part_id])

    unique_parts: dict[int, np.ndarray] = {}
    for part_id, chunks in voxel_parts.items():
        unique_parts[part_id] = (
            np.unique(np.concatenate(chunks)).astype(np.int64)
            if chunks
            else np.empty(0, dtype=np.int64)
        )
    return voxel_occupancy_variants(unique_parts, resolution)


def voxel_iou(voxels_pred: np.ndarray, voxels_gt: np.ndarray) -> float:
    pred = np.asarray(voxels_pred, dtype=np.int64)
    gt = np.asarray(voxels_gt, dtype=np.int64)
    union_count = len(pred) + len(gt)
    if union_count == 0:
        return 1.0
    intersection = len(np.intersect1d(pred, gt, assume_unique=True))
    union = union_count - intersection
    return 0.0 if union == 0 else float(intersection / union)


def match_parts(
    pred_points: dict[int, np.ndarray],
    gt_points: dict[int, np.ndarray],
    matching_points: int,
    workers: int,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    pred_ids = sorted(pred_points)
    gt_ids = sorted(gt_points)
    cost = np.empty((len(pred_ids), len(gt_ids)), dtype=np.float64)
    pred_small = {key: value[:matching_points] for key, value in pred_points.items()}
    gt_small = {key: value[:matching_points] for key, value in gt_points.items()}
    pred_trees = {key: cKDTree(value) for key, value in pred_small.items()}
    gt_trees = {key: cKDTree(value) for key, value in gt_small.items()}
    for row, pred_id in enumerate(pred_ids):
        for column, gt_id in enumerate(gt_ids):
            pred_to_gt = gt_trees[gt_id].query(
                pred_small[pred_id], workers=workers
            )[0]
            gt_to_pred = pred_trees[pred_id].query(
                gt_small[gt_id], workers=workers
            )[0]
            cost[row, column] = float(
                np.mean(pred_to_gt**2) + np.mean(gt_to_pred**2)
            )
    rows, columns = linear_sum_assignment(cost)
    return [(pred_ids[row], gt_ids[column]) for row, column in zip(rows, columns)], cost


def directional_part_average(
    ids: Iterable[int],
    matched_values: dict[int, float],
    missing_value: float,
) -> float:
    values = [matched_values.get(int(part_id), missing_value) for part_id in ids]
    return float(np.mean(values)) if values else float(missing_value)


def evaluate_part_geometry(
    pred_points: dict[int, np.ndarray],
    gt_points: dict[int, np.ndarray],
    pred_surface_voxels: dict[int, np.ndarray],
    gt_surface_voxels: dict[int, np.ndarray],
    pred_volume_voxels: dict[int, np.ndarray],
    gt_volume_voxels: dict[int, np.ndarray],
    pairs: list[tuple[int, int]],
    threshold: float,
    missing_cd_penalty: float,
    workers: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pair_results: list[dict[str, Any]] = []
    by_pred: dict[str, dict[int, float]] = {
        metric: {} for metric in PART_GEOMETRY_METRICS
    }
    by_gt: dict[str, dict[int, float]] = {
        metric: {} for metric in PART_GEOMETRY_METRICS
    }
    for pred_id, gt_id in pairs:
        points = point_metrics(
            pred_points[pred_id], gt_points[gt_id], threshold, workers=workers
        )
        values = {
            "iou_surface": voxel_iou(
                pred_surface_voxels[pred_id], gt_surface_voxels[gt_id]
            ),
            "iou_volume": voxel_iou(
                pred_volume_voxels[pred_id], gt_volume_voxels[gt_id]
            ),
            "fscore": points["fscore"],
            "cd": points["cd"],
        }
        pair_results.append(
            {
                "pred_part_id": pred_id,
                "gt_part_id": gt_id,
                **values,
                "precision": points["precision"],
                "recall": points["recall"],
            }
        )
        for metric, value in values.items():
            by_pred[metric][pred_id] = value
            by_gt[metric][gt_id] = value

    missing = {
        "iou_surface": 0.0,
        "iou_volume": 0.0,
        "fscore": 0.0,
        "cd": missing_cd_penalty,
    }
    penalized: dict[str, float] = {}
    matched_only: dict[str, float | None] = {}
    for metric in PART_GEOMETRY_METRICS:
        forward = directional_part_average(
            pred_points.keys(), by_pred[metric], missing[metric]
        )
        backward = directional_part_average(
            gt_points.keys(), by_gt[metric], missing[metric]
        )
        penalized[metric] = (forward + backward) / 2.0
        matched_values = [item[metric] for item in pair_results]
        matched_only[metric] = (
            float(np.mean(matched_values)) if matched_values else None
        )
    return {"penalized": penalized, "matched_only": matched_only}, pair_results


def evaluate_joints(
    pred_motion: dict[str, np.ndarray],
    gt_motion: dict[str, np.ndarray],
    pairs: list[tuple[int, int]],
) -> dict[str, Any]:
    pred_to_gt = {pred_id: gt_id for pred_id, gt_id in pairs}
    gt_to_pred = {gt_id: pred_id for pred_id, gt_id in pairs}
    gt_revolute = np.flatnonzero(np.asarray(gt_motion["is_part_revolute"], dtype=bool))
    pred_revolute = np.asarray(pred_motion["is_part_revolute"], dtype=bool)
    records: list[dict[str, Any]] = []
    missing: list[int] = []
    wrong_type: list[dict[str, int]] = []
    for gt_id_raw in gt_revolute:
        gt_id = int(gt_id_raw)
        pred_id = gt_to_pred.get(gt_id)
        if pred_id is None:
            missing.append(gt_id)
            continue
        if pred_id >= len(pred_revolute) or not bool(pred_revolute[pred_id]):
            wrong_type.append({"gt_part_id": gt_id, "pred_part_id": pred_id})
            continue
        pred_axis, pred_point = plucker_axis_point(
            pred_motion["revolute_plucker"][pred_id]
        )
        gt_axis, gt_point = plucker_axis_point(gt_motion["revolute_plucker"][gt_id])
        records.append(
            {
                "gt_part_id": gt_id,
                "pred_part_id": pred_id,
                "axis_error_rad": axis_error(pred_axis, gt_axis),
                "origin_error": line_distance(
                    pred_point, pred_axis, gt_point, gt_axis
                ),
                "limit_error_rad": limit_error(
                    pred_axis,
                    pred_motion["revolute_range"][pred_id],
                    gt_axis,
                    gt_motion["revolute_range"][gt_id],
                ),
            }
        )
    total = len(gt_revolute)
    matched = sum(gt_id in gt_to_pred for gt_id in map(int, gt_revolute))
    correct = len(records)
    return {
        "gt_revolute_count": total,
        "geometry_matched_count": matched,
        "correct_type_count": correct,
        "coverage": 1.0 if total == 0 else correct / total,
        "type_accuracy_on_geometry_matched": (
            None if matched == 0 else correct / matched
        ),
        "missing_gt_part_ids": missing,
        "wrong_type": wrong_type,
        "errors": records,
        "unused_pred_to_gt": pred_to_gt,
    }


def evaluate_object(
    sample_id: str,
    method: str,
    prediction_root: Path,
    gt_root: Path,
    asset_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.time()
    gt_mesh, gt_labels, gt_motion = load_gt(sample_id, asset_root, gt_root)
    if method == "urdf-anything-plus":
        pred_mesh, pred_labels, pred_motion = load_urdf_prediction(
            sample_id, prediction_root, args.urdf_coordinate
        )
    else:
        pred_mesh, pred_labels, pred_motion = load_particulate_prediction(
            sample_id, prediction_root
        )
    validate_mesh_labels(gt_mesh, gt_labels, "GT")
    validate_mesh_labels(pred_mesh, pred_labels, method)

    gt_points, gt_whole_points = sample_geometry(
        gt_mesh,
        gt_labels,
        args.points_per_part,
        args.points_whole,
        stable_seed(args.seed, sample_id, "gt"),
    )
    pred_points, pred_whole_points = sample_geometry(
        pred_mesh,
        pred_labels,
        args.points_per_part,
        args.points_whole,
        stable_seed(args.seed, sample_id, method),
    )
    pairs, cost = match_parts(
        pred_points, gt_points, args.matching_points, args.kd_workers
    )
    (
        gt_surface_voxels,
        gt_whole_surface_voxels,
        gt_volume_voxels,
        gt_whole_volume_voxels,
    ) = voxelize_surface_and_volume(
        gt_mesh,
        gt_labels,
        args.voxel_resolution,
        args.domain_min,
        args.domain_max,
        args.dense_vertex_face_threshold,
    )
    (
        pred_surface_voxels,
        pred_whole_surface_voxels,
        pred_volume_voxels,
        pred_whole_volume_voxels,
    ) = voxelize_surface_and_volume(
        pred_mesh,
        pred_labels,
        args.voxel_resolution,
        args.domain_min,
        args.domain_max,
        args.dense_vertex_face_threshold,
    )
    parts, pair_results = evaluate_part_geometry(
        pred_points,
        gt_points,
        pred_surface_voxels,
        gt_surface_voxels,
        pred_volume_voxels,
        gt_volume_voxels,
        pairs,
        args.fscore_threshold,
        args.missing_cd_penalty,
        args.kd_workers,
    )
    whole_points = point_metrics(
        pred_whole_points,
        gt_whole_points,
        args.fscore_threshold,
        workers=args.kd_workers,
    )
    whole = {
        "iou_surface": voxel_iou(
            pred_whole_surface_voxels, gt_whole_surface_voxels
        ),
        "iou_volume": voxel_iou(
            pred_whole_volume_voxels, gt_whole_volume_voxels
        ),
        "fscore": whole_points["fscore"],
        "cd": whole_points["cd"],
        "precision": whole_points["precision"],
        "recall": whole_points["recall"],
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "sample_id": sample_id,
        "method": method,
        "counts": {
            "pred_parts": len(pred_points),
            "gt_parts": len(gt_points),
            "matched_parts": len(pairs),
            "pred_faces": len(pred_mesh.faces),
            "gt_faces": len(gt_mesh.faces),
            "pred_surface_voxels": len(pred_whole_surface_voxels),
            "gt_surface_voxels": len(gt_whole_surface_voxels),
            "pred_volume_voxels": len(pred_whole_volume_voxels),
            "gt_volume_voxels": len(gt_whole_volume_voxels),
        },
        "matching": {
            "pairs": pair_results,
            "cost_matrix_squared_cd": cost,
            "pred_part_ids": sorted(pred_points),
            "gt_part_ids": sorted(gt_points),
        },
        "geometry": {"parts": parts, "whole": whole},
        "joints": evaluate_joints(pred_motion, gt_motion, pairs),
        "elapsed_seconds": time.time() - started,
    }


def scalar_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return {"mean": None, "std": None, "count": 0}
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "count": len(array),
    }


def summarize(
    object_results: list[dict[str, Any]], method: str, metadata: dict[str, Any]
) -> dict[str, Any]:
    geometry: dict[str, Any] = {"parts": {}, "whole": {}}
    for variant in ("penalized", "matched_only"):
        geometry["parts"][variant] = {
            metric: scalar_summary(
                [
                    result["geometry"]["parts"][variant][metric]
                    for result in object_results
                    if result["geometry"]["parts"][variant][metric] is not None
                ]
            )
            for metric in PART_GEOMETRY_METRICS
        }
    geometry["whole"] = {
        metric: scalar_summary(
            [result["geometry"]["whole"][metric] for result in object_results]
        )
        for metric in WHOLE_GEOMETRY_METRICS
    }
    joint_records = [
        record
        for result in object_results
        for record in result["joints"]["errors"]
    ]
    gt_revolute_count = sum(
        result["joints"]["gt_revolute_count"] for result in object_results
    )
    geometry_matched_count = sum(
        result["joints"]["geometry_matched_count"] for result in object_results
    )
    correct_type_count = len(joint_records)
    joints = {
        "axis_error_rad": scalar_summary(
            [record["axis_error_rad"] for record in joint_records]
        ),
        "origin_error": scalar_summary(
            [record["origin_error"] for record in joint_records]
        ),
        "limit_error_rad": scalar_summary(
            [record["limit_error_rad"] for record in joint_records]
        ),
        "gt_revolute_count": gt_revolute_count,
        "geometry_matched_count": geometry_matched_count,
        "correct_type_count": correct_type_count,
        "coverage": (
            None if gt_revolute_count == 0 else correct_type_count / gt_revolute_count
        ),
        "type_accuracy_on_geometry_matched": (
            None
            if geometry_matched_count == 0
            else correct_type_count / geometry_matched_count
        ),
    }
    return {
        "method": method,
        "object_count": len(object_results),
        "metadata": metadata,
        "geometry": geometry,
        "joints": joints,
        "counts": {
            "pred_parts": sum(result["counts"]["pred_parts"] for result in object_results),
            "gt_parts": sum(result["counts"]["gt_parts"] for result in object_results),
            "matched_parts": sum(
                result["counts"]["matched_parts"] for result in object_results
            ),
        },
        "elapsed_seconds": sum(result["elapsed_seconds"] for result in object_results),
        "sample_ids": [result["sample_id"] for result in object_results],
    }


def discover_ids(args: argparse.Namespace) -> list[str]:
    available = sorted(path.stem for path in args.gt_root.glob("*.npz"))
    if args.ids:
        requested = set(args.ids)
        missing = sorted(requested - set(available))
        if missing:
            raise FileNotFoundError(f"requested GT IDs not found: {missing}")
        available = [sample_id for sample_id in available if sample_id in requested]
    if args.limit is not None:
        available = available[: args.limit]
    if not available:
        raise ValueError("no samples selected")
    return available


def cache_is_compatible(
    result: dict[str, Any], sample_id: str, method: str
) -> bool:
    return (
        result.get("protocol_version") == PROTOCOL_VERSION
        and result.get("sample_id") == sample_id
        and result.get("method") == method
    )


def main() -> None:
    args = parse_args()
    ids = discover_ids(args)
    object_dir = args.output_dir / "objects"
    object_results: list[dict[str, Any]] = []
    print(f"Evaluating {args.method} on {len(ids)} objects", flush=True)
    for index, sample_id in enumerate(ids, start=1):
        output_path = object_dir / f"{sample_id}.json"
        use_cache = False
        if args.resume and output_path.is_file():
            result = json.loads(output_path.read_text(encoding="utf-8"))
            use_cache = cache_is_compatible(result, sample_id, args.method)
        if use_cache:
            status = "cached"
        else:
            result = evaluate_object(
                sample_id,
                args.method,
                args.prediction_root,
                args.gt_root,
                args.asset_root,
                args,
            )
            write_json(output_path, result)
            status = f"{result['elapsed_seconds']:.1f}s"
        object_results.append(result)
        print(f"[{index:02d}/{len(ids):02d}] {sample_id}: {status}", flush=True)

    metadata = {
        "protocol_name": "URDF-Anything+ metric family / local operationalization",
        "protocol_version": PROTOCOL_VERSION,
        "official_evaluation_code_available": False,
        "paper_defined_components": {
            "iou": "intersection over union of shape volume / occupied voxels",
            "fscore_threshold": 0.02,
            "chamfer": "sum of two directional mean squared-L2 distances",
            "joint_metrics": "Articulate-Anything axis, origin, and limit errors",
        },
        "local_choice_warning": (
            "voxelization, point counts, part matching, unmatched-part handling, "
            "aggregation, and conditional joint scope are not specified by the paper"
        ),
        "coordinate_domain": [args.domain_min, args.domain_max],
        "prediction_coordinate": (
            f"generated URDF {args.urdf_coordinate} coordinates"
            if args.method == "urdf-anything-plus"
            else "Particulate eval export coordinates"
        ),
        "normalization": "object coordinates scaled to approximately [-1, 1]^3",
        "voxel_iou": {
            "resolution": args.voxel_resolution,
            "ray_samples_per_voxel_cross_section": 1,
            "surface_rasterization": "dense mesh vertices at or above face threshold; tri-axial center rays otherwise",
            "dense_vertex_face_threshold": args.dense_vertex_face_threshold,
            "variants": {
                "iou_surface": {
                    "definition": "intersection/union of occupied surface voxels",
                    "solid_fill": False,
                },
                "iou_volume": {
                    "definition": "intersection/union of occupied solid volume voxels",
                    "solid_fill": True,
                    "volume_fill": "orthographic: bounded by surface voxels in both directions on all three axes",
                    "paper_primary": True,
                },
            },
        },
        "fscore_threshold": args.fscore_threshold,
        "chamfer": "bidirectional mean squared L2; two directional means are summed",
        "points_per_part": args.points_per_part,
        "points_whole": args.points_whole,
        "matching": "one-to-one Hungarian assignment on squared-L2 Chamfer",
        "matching_points_per_part": args.matching_points,
        "kd_tree_workers": args.kd_workers,
        "part_aggregation": "mean of pred-to-GT and GT-to-pred part means",
        "matched_only_variant_role": (
            "diagnostic over one-to-one matched parts; recommended before the "
            "unmatched-part stress test, but not an official paper protocol"
        ),
        "penalized_variant_role": (
            "local missing-part stress test; not an official or primary paper metric"
        ),
        "unmatched_part_values": {
            "iou_surface": 0.0,
            "iou_volume": 0.0,
            "fscore": 0.0,
            "cd": args.missing_cd_penalty,
        },
        "joint_scope": "GT revolute joints with geometry match and predicted revolute type",
        "axis_sign_invariant": True,
        "origin_error": "shortest distance between infinite predicted and GT axes",
        "limit_error": "Articulate-Anything motion-vector error with axis sign flip",
        "seed": args.seed,
    }
    summary = summarize(object_results, args.method, metadata)
    write_json(args.output_dir / "summary.json", summary)
    print(f"Summary: {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
