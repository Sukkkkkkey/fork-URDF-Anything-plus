"""
Thin training entry: argparse, load data, build trainer, run training.
Run from repo root (codes/):  python scripts/train.py ...
"""
import os
import sys
import random

# Ensure repo root (codes/) is on path when running as scripts/train.py
_script_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_script_dir)
if _root not in sys.path:
    sys.path.insert(0, _root)

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from urdf_anything.data import load_cache_data
from urdf_anything.data import (
    CachedDataset,
    CachedSetDataset,
    collate_fn,
    set_collate_fn,
)
from urdf_anything.training import setup_distributed, cleanup_distributed, DiTTrainer


def main():
    parser = argparse.ArgumentParser(description="Train DiT model")
    parser.add_argument(
        "--cache_path",
        type=str,
        nargs="+",
        required=True,
        help="Cache data path (can specify multiple, separated by spaces)",
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        default="urdf_anything/model/URDFModel_config.yaml",
        help="URDFModel config file path",
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=200, help="Maximum training epochs")
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help="Checkpoint save interval (number of epochs)",
    )
    parser.add_argument(
        "--init_mode",
        type=str,
        default="train_from_scratch",
        choices=["train_from_scratch", "resume_from_ckpt", "inference"],
        help="Model initialization mode",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint path (required for resume_from_ckpt and inference modes)",
    )
    parser.add_argument(
        "--train_urdf_params",
        type=str,
        choices=["True", "False"],
        default="False",
        help="Whether to train URDF parameters (origin, axis, limits, motion_type)",
    )
    parser.add_argument(
        "--train_eot",
        type=str,
        choices=["True", "False"],
        default="False",
        help="Whether to train EOT (End of Token) data",
    )
    parser.add_argument(
        "--no_3d_whole",
        type=str,
        choices=["True", "False"],
        default="False",
        help="Whether to remove 3D whole features from conditions",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=None,
        help="Dropout rate for encode_pre (0-1, None means no dropout)",
    )
    parser.add_argument(
        "--history_mode",
        choices=["geometry", "geometry_motion"],
        default="geometry",
        help="Autoregressive history condition ablation",
    )
    parser.add_argument(
        "--generation_mode",
        choices=["autoregressive", "set"],
        default="autoregressive",
        help="Part generation architecture ablation",
    )
    parser.add_argument(
        "--num_part_slots",
        type=int,
        default=5,
        help="Fixed number of slots used by generation_mode=set",
    )
    parser.add_argument(
        "--urdf_loss_timestep_ratio",
        type=float,
        default=0.3,
        help="URDF loss only when timestep < num_train_timesteps * ratio",
    )
    parser.add_argument(
        "--use_wandb",
        type=str,
        choices=["True", "False"],
        default="False",
        help="Whether to use wandb to log training",
    )
    parser.add_argument(
        "--save_checkpoint_dir",
        type=str,
        default=None,
        help="Checkpoint save directory (if None, use checkpoints/{experiment_name})",
    )
    parser.add_argument(
        "--save_optimizer",
        type=str,
        choices=["True", "False"],
        default="False",
        help="Whether to save optimizer state in checkpoint",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable best-effort deterministic CUDA behavior",
    )

    args = parser.parse_args()

    if args.num_part_slots < 1:
        parser.error("--num_part_slots must be at least 1")
    if args.generation_mode == "set" and args.history_mode != "geometry":
        parser.error("set mode is an independent geometry-only ablation")
    if args.generation_mode == "set" and args.train_eot == "True":
        parser.error("set mode uses presence supervision instead of EOT")

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}" if world_size > 1 else "cuda"
    process_seed = args.seed + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

    training_config = {
        "model_config_path": args.model_config_path,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_epochs": args.max_epochs,
        "save_interval": args.save_interval,
        "init_mode": args.init_mode,
        "checkpoint_path": args.checkpoint_path,
        "save_checkpoint_dir": args.save_checkpoint_dir,
        "train_urdf_params": args.train_urdf_params == "True",
        "train_eot": args.train_eot == "True",
        "no_3d_whole": args.no_3d_whole == "True",
        "encode_pre_dropout_rate": args.dropout_rate,
        "urdf_loss_timestep_ratio": args.urdf_loss_timestep_ratio,
        "save_optimizer": args.save_optimizer == "True",
        "use_wandb": args.use_wandb == "True",
        "seed": args.seed,
        "deterministic": args.deterministic,
        "history_mode": args.history_mode,
        "generation_mode": args.generation_mode,
        "num_part_slots": args.num_part_slots,
    }

    if local_rank == 0:
        print("Loading data index from cache...")

    cache_paths = args.cache_path
    all_train_data = []
    all_test_data = []
    data_root_map = {}

    for cache_path in cache_paths:
        if not os.path.exists(cache_path):
            if local_rank == 0:
                print(f"Warning: cache path does not exist, skipping {cache_path}")
            continue
        if local_rank == 0:
            print(f"\nLoading dataset: {cache_path}")
        train_data_index, train_metadata = load_cache_data(cache_path, split="train")
        test_data_index, test_metadata = load_cache_data(cache_path, split="test")
        cache_name = train_metadata["cache_name"]
        data_root = train_metadata["data_root"]
        data_root_map[cache_name] = data_root
        all_train_data.extend(train_data_index)
        all_test_data.extend(test_data_index)

    if len(all_train_data) == 0:
        raise ValueError("No training data found!")

    if local_rank == 0:
        print(f"\nTotal:")
        print(f"  Train set: {len(all_train_data)} samples")
        print(f"  Test set: {len(all_test_data)} samples")
        print(f"  Datasets: {list(data_root_map.keys())}")

    train_eot = training_config.get("train_eot", False)
    if training_config["generation_mode"] == "set":
        train_dataset = CachedSetDataset(
            all_train_data,
            data_root_map,
            split="train",
            num_part_slots=training_config["num_part_slots"],
        )
        val_dataset = CachedSetDataset(
            all_test_data,
            data_root_map,
            split="test",
            num_part_slots=training_config["num_part_slots"],
        )
        selected_collate_fn = set_collate_fn
    else:
        train_dataset = CachedDataset(
            all_train_data, data_root_map, split="train", train_eot=train_eot
        )
        val_dataset = CachedDataset(
            all_test_data, data_root_map, split="test", train_eot=train_eot
        )
        selected_collate_fn = collate_fn

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            seed=args.seed,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=args.seed,
        )
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=0,
        collate_fn=selected_collate_fn,
        pin_memory=False,
        generator=torch.Generator().manual_seed(process_seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=0,
        collate_fn=selected_collate_fn,
        pin_memory=False,
        generator=torch.Generator().manual_seed(process_seed),
    )

    trainer = DiTTrainer(
        training_config,
        device=device,
        local_rank=local_rank,
        world_size=world_size,
        rank=rank,
        cache_paths=cache_paths,
        data_root_map=data_root_map,
    )

    try:
        trainer.train(
            train_loader,
            val_loader,
            max_epochs=training_config["max_epochs"],
            save_interval=training_config.get("save_interval", 10),
        )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
