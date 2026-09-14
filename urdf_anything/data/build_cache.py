import os
import json
import random
import gc
import argparse
from datetime import datetime

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from urdf_anything.model import load_obj_surface
from TripoSG.triposg.models.autoencoders import TripoSGVAEModel

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

device = None
dtype = None
cache_dir = None
dino_processor = None
dino_model = None
vae = None


def compute_dino_features(whole_image):
    with torch.no_grad():
        whole_image = whole_image.float()
        proc_whole = dino_processor(
            images=whole_image.unsqueeze(0), return_tensors="pt"
        ).to(device)
        out_whole = dino_model(**proc_whole).last_hidden_state[:, 1:, :]
        feat = out_whole.float().cpu()
    return feat.squeeze(0)


def load_and_encode_obj(obj_path, num_pc=204800, token_length=2048):
    points = load_obj_surface(obj_path, num_pc=num_pc)
    points = points.to(device, dtype=dtype)
    with torch.no_grad():
        encoded = (
            vae.encode(points, num_tokens=token_length)
            .latent_dist.sample()
            .float()
            .cpu()
        )
    return encoded


def load_image_tensor(image_path):
    return (
        torch.tensor(np.array(Image.open(image_path).convert("RGB"))).float()
        / 255.0
    )


def get_available_whole_images(id_path):
    candidates = []
    for file_name in os.listdir(os.path.join(id_path, "images")):
        if file_name.lower().endswith(IMAGE_EXTENSIONS):
            candidates.append(file_name)
    candidates.sort()
    if not candidates:
        raise FileNotFoundError(f"{id_path} has no available image.")
    return candidates


def get_item_from_info(id_path, info_data, link_idx, encode_whole, token_length=2048):
    obj_id = info_data["id"]
    links = info_data["links"]
    if link_idx >= len(links):
        raise ValueError(
            f"link_idx {link_idx} out of range, object has {len(links)} links"
        )
    current_link = links[link_idx]
    current_obj_path = os.path.join(id_path, current_link["obj"])
    target_label = load_and_encode_obj(current_obj_path, token_length=token_length)
    if link_idx == 0:
        encode_pre = torch.zeros_like(encode_whole).float().cpu()
    else:
        all_prev_points = []
        num_pc_per_link = max(1, 204800 // link_idx)
        for prev_idx in range(link_idx):
            prev_link = links[prev_idx]
            prev_obj_path = os.path.join(id_path, prev_link["obj"])
            prev_points = load_obj_surface(prev_obj_path, num_pc=num_pc_per_link)
            if prev_points.dim() == 3 and prev_points.shape[0] == 1:
                prev_points = prev_points.squeeze(0)
            all_prev_points.append(prev_points)
        combined_points = torch.cat(all_prev_points, dim=0)
        if combined_points.shape[0] != 204800:
            indices = torch.randperm(combined_points.shape[0])[:204800]
            combined_points = combined_points[indices]
        combined_points = combined_points.to(device, dtype=dtype).unsqueeze(0)
        with torch.no_grad():
            encode_pre = (
                vae.encode(combined_points, num_tokens=token_length)
                .latent_dist.sample()
                .float()
                .cpu()
            )
    id_name = f"{obj_id}_{current_link['name']}"
    return {
        "encode_pre": encode_pre,
        "target_label": target_label,
        "id": id_name,
        "link_idx": link_idx,
        "link_name": current_link["name"],
    }


def build_eot_item(info_data, encode_whole):
    target_label = torch.zeros_like(encode_whole).float().cpu()
    id_name = f"{info_data['id']}_eot"
    encode_pre = (
        encode_whole.clone().cpu()
        if encode_whole.is_cuda
        else encode_whole.clone()
    ).float()
    return {
        "encode_pre": encode_pre,
        "target_label": target_label,
        "id": id_name,
        "link_idx": len(info_data["links"]),
        "link_name": "eot",
    }


def save_cache_data_for_dataset(
    data_root,
    cache_name,
    token_length,
    train_test_split_path=None,
    allow_single_object_overlap=False,
    random_seed=42,
):
    global device, dtype, cache_dir, dino_processor, dino_model, vae
    print(f"\nBegin to save cache for dataset: {data_root}")
    print(f"Cache name: {cache_name}")
    all_ids = [
        d
        for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d)) and not d.startswith(".")
    ]
    all_ids.sort()
    print(f"Found {len(all_ids)} objects")
    current_cache_dir = os.path.join(cache_dir, cache_name)
    os.makedirs(current_cache_dir, exist_ok=True)
    total_samples = 0
    failed_items = []
    all_data = []

    for obj_id in tqdm(all_ids, desc="Processing objects"):
        id_path = os.path.join(data_root, obj_id)
        info_json_path = os.path.join(id_path, "info.json")
        if not os.path.exists(info_json_path):
            failed_items.append({"id": obj_id, "reason": "missing_info_json"})
            continue
        try:
            with open(info_json_path, "r", encoding="utf-8") as f:
                info_data = json.load(f)
            image_candidates = get_available_whole_images(id_path)
            if "whole_obj" in info_data:
                whole_obj_path = os.path.join(id_path, info_data["whole_obj"])
            else:
                whole_obj_path = os.path.join(id_path, "whole.obj")
            if not os.path.exists(whole_obj_path):
                raise FileNotFoundError(f"whole.obj not found: {whole_obj_path}")
            encode_whole = load_and_encode_obj(
                whole_obj_path, token_length=token_length
            )
            whole_image_features = {}
            for whole_image_name in image_candidates:
                whole_image_path = os.path.join(id_path, "images", whole_image_name)
                whole_image = load_image_tensor(whole_image_path)
                whole_image_features[whole_image_name] = compute_dino_features(
                    whole_image
                )
                del whole_image
                torch.cuda.empty_cache()
            num_links = len(info_data["links"])
            items = []
            for link_idx in range(num_links):
                data = get_item_from_info(
                    id_path, info_data, link_idx, encode_whole, token_length
                )
                items.append(data)
                total_samples += len(image_candidates)
            eot_data = build_eot_item(info_data, encode_whole)
            total_samples += len(image_candidates)
            obj_cache_data = {
                "obj_id": obj_id,
                "encode_whole": encode_whole,
                "whole_image_features": whole_image_features,
                "items": items,
                "eot_item": eot_data,
                "num_links": num_links,
                "num_images": len(image_candidates),
            }
            obj_cache_path = os.path.join(current_cache_dir, f"{obj_id}.pt")
            torch.save(obj_cache_data, obj_cache_path)
            for item in items:
                for whole_image_name in image_candidates:
                    all_data.append(
                        {
                            "id": item["id"],
                            "obj_id": obj_id,
                            "link_idx": item["link_idx"],
                            "link_name": item["link_name"],
                            "whole_image_name": whole_image_name,
                            "file_path": obj_cache_path,
                        }
                    )
            for whole_image_name in image_candidates:
                all_data.append(
                    {
                        "id": eot_data["id"],
                        "obj_id": obj_id,
                        "link_idx": eot_data["link_idx"],
                        "link_name": eot_data["link_name"],
                        "whole_image_name": whole_image_name,
                        "file_path": obj_cache_path,
                    }
                )
            del whole_image_features, encode_whole, items, obj_cache_data
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"Error processing {obj_id}: {e}")
            import traceback
            traceback.print_exc()
            failed_items.append({"id": obj_id, "reason": str(e)})
            continue

    if not all_data:
        raise RuntimeError(
            f"No cache samples were built for {data_root}; failures={failed_items}"
        )

    print("\nsplit train/test (by object)...")
    if train_test_split_path and os.path.isfile(train_test_split_path):
        split_method = "split_file_by_object"
        with open(train_test_split_path, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        train_obj_ids = set(str(x) for x in split_data.get("train", []))
        test_obj_ids = set(str(x) for x in split_data.get("test", []))
        print(f"  using train/test split: {train_test_split_path} (train {len(train_obj_ids)}, test {len(test_obj_ids)} ids)")
        for d in all_data:
            oid = str(d["obj_id"])
            if oid not in train_obj_ids and oid not in test_obj_ids:
                train_obj_ids.add(oid)
        train_data = [d for d in all_data if str(d["obj_id"]) in train_obj_ids]
        test_data = [d for d in all_data if str(d["obj_id"]) in test_obj_ids]
    else:
        random.seed(random_seed)
        unique_obj_ids = sorted(set(str(d["obj_id"]) for d in all_data))
        shuffled_obj_ids = unique_obj_ids.copy()
        random.shuffle(shuffled_obj_ids)
        total_obj_count = len(shuffled_obj_ids)
        if total_obj_count == 1 and allow_single_object_overlap:
            split_method = "single_object_smoke_overlap"
            train_obj_ids = set(shuffled_obj_ids)
            test_obj_ids = set(shuffled_obj_ids)
        else:
            split_method = "random_by_object"
            train_obj_count = int(total_obj_count * 0.9)
            train_obj_ids = set(shuffled_obj_ids[:train_obj_count])
            test_obj_ids = set(shuffled_obj_ids[train_obj_count:])
        train_data = [d for d in all_data if str(d["obj_id"]) in train_obj_ids]
        test_data = [d for d in all_data if str(d["obj_id"]) in test_obj_ids]
    torch.save(train_data, os.path.join(current_cache_dir, "train_data.pt"))
    torch.save(test_data, os.path.join(current_cache_dir, "test_data.pt"))
    torch.save(all_data, os.path.join(current_cache_dir, "full_data.pt"))
    metadata = {
        "data_root": data_root,
        "cache_name": cache_name,
        "total_objects": len(all_ids),
        "total_samples": total_samples,
        "train_objects": len(train_obj_ids),
        "test_objects": len(test_obj_ids),
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "train_ratio": (
            len(train_obj_ids) / len(set(str(d["obj_id"]) for d in all_data))
            if all_data
            else 0.0
        ),
        "test_ratio": (
            len(test_obj_ids) / len(set(str(d["obj_id"]) for d in all_data))
            if all_data
            else 0.0
        ),
        "split_method": split_method,
        "failed_items": failed_items,
        "device": device,
        "dtype": str(dtype),
        "random_seed": random_seed,
        "timestamp": datetime.now().isoformat(),
    }
    with open(
        os.path.join(current_cache_dir, "metadata.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\ncache completed!")
    print(f"  - total samples: {total_samples}")
    print(f"  - train samples: {len(train_data)}")
    print(f"  - test samples: {len(test_data)}")
    print(f"  - cache path: {current_cache_dir}")
    return current_cache_dir, all_data, train_data, test_data, metadata


def main():
    """CLI entry: parse args, load VAE/DINO, run save_cache_data_for_dataset on configured datasets."""
    global device, dtype, cache_dir, dino_processor, dino_model, vae
    parser = argparse.ArgumentParser()
    parser.add_argument("--token_length", type=int, default=512)
    parser.add_argument("--dino_model_path", type=str, default="DINOv3")
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument(
        "--triposg_vae_path",
        type=str,
        default=os.environ.get(
            "URDF_ANYTHING_TRIPOSG_VAE_PATH",
            "TripoSG/pretrained_weights/TripoSG",
        ),
        help="TripoSG root containing vae/ or the VAE directory itself",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Build one explicitly selected dataset root instead of all default categories",
    )
    parser.add_argument(
        "--data_normalized_root",
        type=str,
        default="data_normalized",
        help="Root containing all default *_urdf category directories",
    )
    parser.add_argument(
        "--cache_name",
        type=str,
        default=None,
        help="Cache name used with --data_root (defaults to the data directory name)",
    )
    parser.add_argument(
        "--allow_single_object_overlap",
        action="store_true",
        help="Use the same sole object for train and validation in a smoke test",
    )
    parser.add_argument(
        "--train_test_split",
        type=str,
        default=None,
        help="Object split JSON (defaults to <data_normalized_root>/train_test_split.json for full data)",
    )
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    token_length = args.token_length
    vae_kwargs = {"local_files_only": True}
    if os.path.isfile(os.path.join(args.triposg_vae_path, "config.json")):
        pass
    elif os.path.isfile(
        os.path.join(args.triposg_vae_path, "vae", "config.json")
    ):
        vae_kwargs["subfolder"] = "vae"
    else:
        raise FileNotFoundError(
            "Cannot find TripoSG VAE config.json under "
            f"{args.triposg_vae_path}"
        )
    vae = TripoSGVAEModel.from_pretrained(
        args.triposg_vae_path,
        **vae_kwargs,
    ).to(device, dtype=dtype)
    vae.eval()
    dino_processor = AutoImageProcessor.from_pretrained(
        args.dino_model_path, local_files_only=True
    )
    dino_model = AutoModel.from_pretrained(
        args.dino_model_path, local_files_only=True
    ).to(device)
    dino_model.eval()
    cache_dir = args.cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    if args.data_root:
        train_test_split_path = args.train_test_split
        datasets = [
            {
                "data_root": args.data_root,
                "cache_name": args.cache_name
                or f"{os.path.basename(os.path.normpath(args.data_root)).lower()}_eot",
            }
        ]
    else:
        train_test_split_path = args.train_test_split or os.path.join(
            args.data_normalized_root, "train_test_split.json"
        )
        datasets = [
            {"data_root": os.path.join(args.data_normalized_root, "Laptop_urdf"), "cache_name": "laptop_eot"},
            {"data_root": os.path.join(args.data_normalized_root, "Refrigerator_urdf"), "cache_name": "refrigerator_eot"},
            {"data_root": os.path.join(args.data_normalized_root, "Dishwasher_urdf"), "cache_name": "dishwasher_eot"},
            {"data_root": os.path.join(args.data_normalized_root, "Microwave_urdf"), "cache_name": "microwave_eot"},
            {"data_root": os.path.join(args.data_normalized_root, "Faucet_urdf"), "cache_name": "faucet_eot"},
            {"data_root": os.path.join(args.data_normalized_root, "Display_urdf"), "cache_name": "display_eot"},
            {"data_root": os.path.join(args.data_normalized_root, "Door_urdf"), "cache_name": "door_eot"},
            {"data_root": os.path.join(args.data_normalized_root, "Knife_urdf"), "cache_name": "knife_eot"},
            {"data_root": os.path.join(args.data_normalized_root, "Scissors_urdf"), "cache_name": "scissors_eot"},
            {"data_root": os.path.join(args.data_normalized_root, "StorageFurniture_urdf"), "cache_name": "storagefurniture_eot"},
        ]
    successful_datasets = 0
    for dataset_config in datasets:
        try:
            cache_path, all_data, train_data, test_data, metadata = save_cache_data_for_dataset(
                dataset_config["data_root"],
                dataset_config["cache_name"] + f"_token{token_length}",
                token_length,
                train_test_split_path=train_test_split_path,
                allow_single_object_overlap=args.allow_single_object_overlap,
                random_seed=args.seed,
            )
            print(f"\n{'='*60}")
            print(f"dataset: {dataset_config['data_root']}")
            print(f"cache path: {cache_path}")
            print(f"total samples: {len(all_data)}")
            print(f"train samples: {len(train_data)}")
            print(f"test samples: {len(test_data)}")
            if metadata.get("failed_items"):
                print(f"failed items: {len(metadata['failed_items'])}")
            print(f"{'='*60}\n")
            successful_datasets += 1
        except Exception as e:
            print(f"\nerror processing dataset {dataset_config['data_root']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    if successful_datasets == 0:
        raise RuntimeError("Cache building failed for every selected dataset")
