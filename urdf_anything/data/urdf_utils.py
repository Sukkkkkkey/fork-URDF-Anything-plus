"""Load and parse info.json; extract URDF parameters for links."""
import os
import json
import torch
import torch.nn.functional as F


def load_info_json(obj_dir):
    """Load and parse info.json in the given object directory."""
    try:
        info_path = os.path.join(obj_dir, "info.json")
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"load info.json failed {obj_dir}: {e}")
    return None


def get_urdf_params_from_info(info_data, link_idx):
    """
    Extract URDF-related parameters for a link from info.json data.
    Returns (origin_xyz, axis_xyz, lower_upper_limits, motion_type) or (None, None, None, None).
    """
    try:
        if info_data is None or "links" not in info_data:
            return None, None, None, None

        links = info_data["links"]
        if link_idx >= len(links):
            return None, None, None, None

        link = links[link_idx]
        if link_idx == 0:
            return None, None, None, None

        if (
            "origin_xyz" in link
            and "axis_xyz" in link
            and "lower" in link
            and "upper" in link
            and "motion_type" in link
        ):
            origin_xyz = [float(x) for x in link["origin_xyz"].split()]
            axis_xyz = [float(x) for x in link["axis_xyz"].split()]
            lower_upper_limits = [float(link["lower"]), float(link["upper"])]
            motion_type = None
            motion_type_str = link["motion_type"].lower()
            if motion_type_str == "revolute":
                motion_type = 0
            elif motion_type_str == "prismatic":
                motion_type = 1
            else:
                motion_type = None

            return (
                torch.tensor(origin_xyz, dtype=torch.float32),
                torch.tensor(axis_xyz, dtype=torch.float32),
                torch.tensor(lower_upper_limits, dtype=torch.float32),
                torch.tensor(motion_type, dtype=torch.long) if motion_type is not None else None,
            )

        return None, None, None, None
    except Exception as e:
        print(f"extract URDF parameters from info.json failed: {e}")
        return None, None, None, None


def build_motion_cue(origin_xyz, axis_xyz, lower_upper_limits, motion_type):
    """Build the 10D motion-history cue used by the autoregressive ablation."""
    origin_xyz = torch.as_tensor(origin_xyz, dtype=torch.float32)
    axis_xyz = F.normalize(
        torch.as_tensor(axis_xyz, dtype=torch.float32), dim=0, eps=1e-6
    )
    limits = torch.tanh(
        torch.as_tensor(lower_upper_limits, dtype=torch.float32)
    )
    motion_type = int(torch.as_tensor(motion_type).item())
    if motion_type not in (0, 1):
        raise ValueError(f"unsupported motion type index: {motion_type}")
    motion_one_hot = F.one_hot(
        torch.tensor(motion_type, device=origin_xyz.device), num_classes=2
    ).to(torch.float32)
    return torch.cat([origin_xyz, axis_xyz, limits, motion_one_hot], dim=0)


def get_motion_history_from_info(info_data, link_idx):
    """Return GT cues for movable links generated before ``link_idx``."""
    empty = torch.empty((0, 10), dtype=torch.float32)
    if info_data is None or "links" not in info_data or link_idx <= 1:
        return empty

    history = []
    history_end = min(int(link_idx), len(info_data["links"]))
    for previous_idx in range(1, history_end):
        origin, axis, limits, motion_type = get_urdf_params_from_info(
            info_data, previous_idx
        )
        if any(value is None for value in (origin, axis, limits, motion_type)):
            continue
        history.append(build_motion_cue(origin, axis, limits, motion_type))
    return torch.stack(history) if history else empty
