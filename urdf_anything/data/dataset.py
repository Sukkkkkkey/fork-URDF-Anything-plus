"""Cached dataset and collate for DiT training."""
import os
import torch
import trimesh
from torch.utils.data import Dataset

from .urdf_utils import (
    get_motion_history_from_info,
    get_urdf_params_from_info,
    load_info_json,
)

from .cache import load_single_data_item


class CachedDataset(Dataset):
    """Dataset that loads data from cache - on-demand loading, memory efficient, supports new cache structure and multiple datasets."""

    def __init__(self, data_index_list, data_root_map, split="train", train_eot=False):
        """
        Args:
            data_index_list: Data index list loaded from cache (can come from multiple datasets)
            data_root_map: Dataset root directory mapping {cache_name: data_root}
            split: 'train' or 'test'
            train_eot: Whether to include EOT data (link_name == 'eot')
        """
        if train_eot:
            self.data_index_list = data_index_list
        else:
            self.data_index_list = [
                data_info for data_info in data_index_list
                if data_info.get("link_name") != "eot"
            ]

        self.data_root_map = data_root_map
        self.split = split
        self.train_eot = train_eot

        unique_obj_ids = set([data_info["obj_id"] for data_info in self.data_index_list])

        dataset_stats = {}
        eot_count = 0
        non_eot_count = 0
        for data_info in self.data_index_list:
            file_path = data_info["file_path"]
            for cache_name in self.data_root_map.keys():
                if cache_name in file_path:
                    dataset_stats[cache_name] = dataset_stats.get(cache_name, 0) + 1
                    break
            if data_info.get("link_name") == "eot":
                eot_count += 1
            else:
                non_eot_count += 1

        print(f"{split} dataset: {len(self.data_index_list)} samples from {len(unique_obj_ids)} objects")
        if train_eot:
            print(f"  - EOT samples: {eot_count}")
            print(f"  - Non-EOT samples: {non_eot_count}")
        for dataset_name, count in dataset_stats.items():
            print(f"  - {dataset_name}: {count} samples")

    def __len__(self):
        return len(self.data_index_list)

    def __getitem__(self, idx):
        data_info = self.data_index_list[idx]

        obj_id = data_info["obj_id"]
        link_idx = data_info["link_idx"]
        whole_image_name = data_info.get("whole_image_name")

        if whole_image_name is None:
            raise ValueError(f"whole_image_name missing in data_info: {data_info}")

        file_path = data_info["file_path"]
        data_root = None
        for cache_name, root in self.data_root_map.items():
            if cache_name in file_path:
                data_root = root
                break

        if data_root is None:
            raise ValueError(f"Cannot determine dataset root directory: {file_path}")

        obj_dir = os.path.join(data_root, obj_id)

        try:
            data = load_single_data_item(
                file_path,
                link_idx=link_idx,
                whole_image_name=whole_image_name,
            )
        except Exception as e:
            print(f"Failed to load data {file_path} (link_idx={link_idx}, whole_image_name={whole_image_name}): {e}")
            return None

        info_data = load_info_json(obj_dir)
        data["motion_history"] = get_motion_history_from_info(info_data, link_idx)
        origin_xyz, axis_xyz, lower_upper_limits, motion_type = get_urdf_params_from_info(info_data, link_idx)

        if origin_xyz is not None and axis_xyz is not None and lower_upper_limits is not None and motion_type is not None:
            data["urdf_origin"] = origin_xyz
            data["urdf_axis"] = axis_xyz
            data["has_urdf"] = True
            data["lower_upper_limits"] = lower_upper_limits
            data["motion_type"] = motion_type
        else:
            data["urdf_origin"] = torch.zeros(3, dtype=torch.float32)
            data["urdf_axis"] = torch.zeros(3, dtype=torch.float32)
            data["has_urdf"] = False
            data["lower_upper_limits"] = torch.tensor([0.0, 0.0], dtype=torch.float32)
            data["motion_type"] = torch.tensor(-1, dtype=torch.long)
        return data


def collate_fn(batch):
    """Custom batch processing function."""
    encode_pres = torch.cat([item["encode_pre"] for item in batch], dim=0)
    encode_wholes = torch.cat([item["encode_whole"] for item in batch], dim=0)
    target_labels = torch.cat([item["target_label"] for item in batch], dim=0)
    dino_features = torch.stack([item["dino_features"] for item in batch])

    urdf_origins = torch.stack([item["urdf_origin"] for item in batch])
    urdf_axes = torch.stack([item["urdf_axis"] for item in batch])
    has_urdf = torch.tensor([item["has_urdf"] for item in batch])
    lower_upper_limits = torch.stack([item["lower_upper_limits"] for item in batch])
    motion_types = torch.stack([item["motion_type"] for item in batch])
    link_indices = torch.tensor([item["link_idx"] for item in batch])
    motion_history_lengths = torch.tensor(
        [item["motion_history"].shape[0] for item in batch], dtype=torch.long
    )
    max_history_length = int(motion_history_lengths.max().item())
    motion_histories = torch.zeros(
        len(batch), max_history_length, 10, dtype=torch.float32
    )
    for index, item in enumerate(batch):
        history_length = item["motion_history"].shape[0]
        if history_length:
            motion_histories[index, :history_length] = item["motion_history"]

    is_eot = torch.tensor([item.get("link_name") == "eot" for item in batch])
    return {
        "encode_pres": encode_pres.detach(),
        "encode_wholes": encode_wholes.detach(),
        "target_labels": target_labels.detach(),
        "dino_features": dino_features.detach(),
        "urdf_origins": urdf_origins.detach(),
        "urdf_axes": urdf_axes.detach(),
        "has_urdf": has_urdf.detach(),
        "link_indices": link_indices.detach(),
        "is_eot": is_eot.detach(),
        "ids": [item["id"] for item in batch],
        "link_names": [item["link_name"] for item in batch],
        "lower_upper_limits": lower_upper_limits.detach(),
        "motion_types": motion_types.detach(),
        "motion_histories": motion_histories.detach(),
        "motion_history_lengths": motion_history_lengths.detach(),
    }


def get_spatial_part_orders(obj_dir, info_data):
    """Assign root=0, then movable parts by AABB minimum in Z-X-Y order."""
    links = info_data.get("links", []) if info_data is not None else []
    if not links:
        raise ValueError(f"no links found for set sample: {obj_dir}")
    movable = []
    for link_idx, link in enumerate(links[1:], start=1):
        mesh_path = os.path.join(obj_dir, link["obj"])
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
            raise ValueError(f"cannot load part mesh for ordering: {mesh_path}")
        minimum = mesh.bounds[0]
        movable.append((link_idx, (minimum[2], minimum[0], minimum[1])))
    movable.sort(key=lambda item: (*item[1], item[0]))
    orders = {0: 0}
    orders.update(
        {link_idx: order for order, (link_idx, _) in enumerate(movable, start=1)}
    )
    return orders


class CachedSetDataset(Dataset):
    """One object/image per sample, with parts randomly assigned to fixed slots."""

    def __init__(
        self,
        data_index_list,
        data_root_map,
        split="train",
        num_part_slots=5,
    ):
        self.data_root_map = data_root_map
        self.split = split
        self.num_part_slots = int(num_part_slots)
        if self.num_part_slots < 1:
            raise ValueError("num_part_slots must be at least 1")

        grouped = {}
        for data_info in data_index_list:
            if data_info.get("link_name") == "eot":
                continue
            key = (data_info["file_path"], data_info["whole_image_name"])
            grouped.setdefault(key, data_info)
        self.data_index_list = list(grouped.values())
        print(
            f"{split} set dataset: {len(self.data_index_list)} objects/images, "
            f"{self.num_part_slots} slots each"
        )

    def __len__(self):
        return len(self.data_index_list)

    def _find_data_root(self, file_path):
        for cache_name, root in self.data_root_map.items():
            if cache_name in file_path:
                return root
        raise ValueError(f"Cannot determine dataset root directory: {file_path}")

    def __getitem__(self, idx):
        data_info = self.data_index_list[idx]
        file_path = data_info["file_path"]
        whole_image_name = data_info["whole_image_name"]
        data_root = self._find_data_root(file_path)
        obj_dir = os.path.join(data_root, data_info["obj_id"])
        info_data = load_info_json(obj_dir)
        links = info_data.get("links", []) if info_data is not None else []
        num_parts = len(links)
        if num_parts > self.num_part_slots:
            raise ValueError(
                f"object {data_info['obj_id']} has {num_parts} parts, exceeding "
                f"num_part_slots={self.num_part_slots}"
            )
        if num_parts == 0:
            raise ValueError(f"object {data_info['obj_id']} has no parts")

        part_orders = get_spatial_part_orders(obj_dir, info_data)
        first_part = load_single_data_item(
            file_path,
            link_idx=0,
            whole_image_name=whole_image_name,
        )
        latent_shape = first_part["target_label"].shape[1:]
        target_labels = torch.zeros(
            self.num_part_slots, *latent_shape, dtype=torch.float32
        )
        urdf_origins = torch.zeros(self.num_part_slots, 3, dtype=torch.float32)
        urdf_axes = torch.zeros(self.num_part_slots, 3, dtype=torch.float32)
        lower_upper_limits = torch.zeros(
            self.num_part_slots, 2, dtype=torch.float32
        )
        motion_types = torch.full(
            (self.num_part_slots,), -1, dtype=torch.long
        )
        has_urdf = torch.zeros(self.num_part_slots, dtype=torch.bool)
        presence = torch.zeros(self.num_part_slots, dtype=torch.float32)
        part_order_targets = torch.full(
            (self.num_part_slots,), -1, dtype=torch.long
        )
        link_indices = torch.full(
            (self.num_part_slots,), -1, dtype=torch.long
        )
        part_names = [""] * self.num_part_slots

        if self.split == "train":
            assigned_slots = torch.randperm(self.num_part_slots)[:num_parts]
        else:
            assigned_slots = torch.arange(num_parts)

        for link_idx, slot_tensor in enumerate(assigned_slots):
            slot = int(slot_tensor.item())
            part = (
                first_part
                if link_idx == 0
                else load_single_data_item(
                    file_path,
                    link_idx=link_idx,
                    whole_image_name=whole_image_name,
                )
            )
            target_labels[slot] = part["target_label"].squeeze(0).float()
            presence[slot] = 1.0
            part_order_targets[slot] = part_orders[link_idx]
            link_indices[slot] = link_idx
            part_names[slot] = links[link_idx]["name"]

            origin, axis, limits, motion_type = get_urdf_params_from_info(
                info_data, link_idx
            )
            if all(
                value is not None
                for value in (origin, axis, limits, motion_type)
            ):
                urdf_origins[slot] = origin
                urdf_axes[slot] = axis
                lower_upper_limits[slot] = limits
                motion_types[slot] = motion_type
                has_urdf[slot] = True

        return {
            "encode_whole": first_part["encode_whole"].squeeze(0).float(),
            "target_labels": target_labels,
            "dino_features": first_part["dino_features"].float(),
            "urdf_origins": urdf_origins,
            "urdf_axes": urdf_axes,
            "lower_upper_limits": lower_upper_limits,
            "motion_types": motion_types,
            "has_urdf": has_urdf,
            "presence": presence,
            "part_order_targets": part_order_targets,
            "link_indices": link_indices,
            "obj_id": data_info["obj_id"],
            "part_names": part_names,
        }


def set_collate_fn(batch):
    tensor_keys = (
        "encode_whole",
        "target_labels",
        "dino_features",
        "urdf_origins",
        "urdf_axes",
        "lower_upper_limits",
        "motion_types",
        "has_urdf",
        "presence",
        "part_order_targets",
        "link_indices",
    )
    result = {
        key: torch.stack([item[key] for item in batch]).detach()
        for key in tensor_keys
    }
    result["obj_ids"] = [item["obj_id"] for item in batch]
    result["part_names"] = [item["part_names"] for item in batch]
    return result
