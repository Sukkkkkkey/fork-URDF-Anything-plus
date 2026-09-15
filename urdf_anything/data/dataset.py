"""Cached dataset and collate for DiT training."""
import os
import torch
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
