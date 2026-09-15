import os
import sys
import json
import platform
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from urdf_anything.model import URDFModel, load_model_config

import wandb


class DiTTrainer:
    """DiT trainer - supports cached data, improved naming conventions, and distributed training."""

    def __init__(self, config, device="cuda", local_rank=0, world_size=1, rank=0, cache_paths=None, data_root_map=None):
        self.config = config
        self.no_3d_whole = self.config.get("no_3d_whole", False)
        self.device = device
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1
        self.use_wandb = config.get("use_wandb", False) and wandb is not None
        self.save_optimizer = config.get("save_optimizer", False)

        self.cache_paths = cache_paths if cache_paths is not None else []
        self.data_root_map = data_root_map if data_root_map is not None else {}

        self.training_start_time = datetime.now().isoformat()
        self.experiment_name = self._generate_experiment_name()

        model_config_path = self.config.get(
            "model_config_path", "urdf_anything/model/URDFModel_config.yaml"
        )
        self.model_config = load_model_config(model_config_path)
        self.model_config["ditrunner"]["history_mode"] = self.config.get(
            "history_mode", "geometry"
        )
        self.model_config["ditrunner"]["generation_mode"] = self.config.get(
            "generation_mode", "autoregressive"
        )
        self.model_config["ditrunner"]["num_part_slots"] = self.config.get(
            "num_part_slots", 5
        )

        init_mode = self.config.get("init_mode", "train_from_scratch")
        checkpoint_path = self.config.get("checkpoint_path", None)

        self.model = URDFModel(
            config_dict=self.model_config,
            device=device,
            init_mode=init_mode,
            checkpoint_path=checkpoint_path,
        ).to(device)
        self.generation_mode = self.config.get(
            "generation_mode", "autoregressive"
        )
        if self.generation_mode == "set":
            self.model.DiTRunner.model.gradient_checkpointing = True

        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank)

        model_to_train = self.model.module if self.is_distributed else self.model
        for name, param in model_to_train.named_parameters():
            if "SoT" in name:
                param.requires_grad = True
            elif "feature3d_adapter" in name:
                param.requires_grad = True
            elif "dino_adapter" in name:
                param.requires_grad = True
            elif "DiTRunner" in name and "pos_emb" not in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 0.01),
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        noise_scheduler_config = self.model_config["ditrunner"]["noise_scheduler"]
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_config["num_train_timesteps"],
            beta_start=noise_scheduler_config["beta_start"],
            prediction_type=noise_scheduler_config["prediction_type"],
            clip_sample=noise_scheduler_config["clip_sample"],
        )
        self.num_train_timesteps = noise_scheduler_config["num_train_timesteps"]

        urdf_loss_timestep_ratio = config.get("urdf_loss_timestep_ratio", 0.3)
        self.urdf_loss_timestep_threshold = int(self.num_train_timesteps * urdf_loss_timestep_ratio)
        if self.rank == 0:
            print(f"URDF loss is only computed when timestep < {self.urdf_loss_timestep_threshold} (ratio={urdf_loss_timestep_ratio})")

        if self.rank == 0:
            if self.config.get("save_checkpoint_dir") is not None:
                self.save_checkpoint_dir = f"{self.config.get('save_checkpoint_dir')}/{self.experiment_name}"
                os.makedirs(self.save_checkpoint_dir, exist_ok=True)
            else:
                self.save_checkpoint_dir = f"checkpoints/{self.experiment_name}"
                os.makedirs(self.save_checkpoint_dir, exist_ok=True)

            if self.use_wandb and wandb is not None:
                wandb.init(
                    project="urdf-diffusion",
                    name=self.experiment_name,
                    config=self.config,
                    dir=self.save_checkpoint_dir,
                )
        else:
            self.save_checkpoint_dir = None
        self.global_step = 0

    def _generate_experiment_name(self):
        """Generate experiment name containing hyperparameters."""
        lr = self.config.get("learning_rate", 1e-4)
        batch_size = self.config.get("batch_size", 2)
        epochs = self.config.get("max_epochs", 100)
        eot = self.config.get("train_eot", False)
        urdf_params = self.config.get("train_urdf_params", False)
        no_3d_whole = self.config.get("no_3d_whole", False)
        history_mode = self.config.get("history_mode", "geometry")
        generation_mode = self.config.get("generation_mode", "autoregressive")
        num_part_slots = self.config.get("num_part_slots", 5)

        lr_str = f"lr{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
        batch_str = f"bs{batch_size}"
        epoch_str = f"ep{epochs}"
        eot_str = "_eot" if eot else ""
        urdf_params_str = "_urdf-params" if urdf_params else ""
        no_3d_whole_str = "_no3dwhole" if no_3d_whole else ""
        history_str = "_motion-history" if history_mode == "geometry_motion" else ""
        generation_str = (
            f"_set{num_part_slots}" if generation_mode == "set" else ""
        )
        experiment_name = f"{lr_str}_{batch_str}_{epoch_str}{eot_str}{urdf_params_str}{no_3d_whole_str}{history_str}{generation_str}"
        return experiment_name

    def train_step(self, batch):
        """Single training step."""
        if self.generation_mode == "set":
            return self._train_step_set(batch)
        self.model.train()
        self.optimizer.zero_grad()

        dino_features_raw = batch["dino_features"].to(self.device)
        model_to_use = self.model.module if self.is_distributed else self.model
        dino_features = model_to_use.dino_adapter(dino_features_raw)
        encode_wholes = batch["encode_wholes"].to(self.device)
        target_labels = batch["target_labels"].to(self.device)
        link_indices = batch["link_indices"]

        urdf_origins = batch["urdf_origins"].to(self.device)
        urdf_axes = batch["urdf_axes"].to(self.device)
        has_urdf = batch["has_urdf"].to(self.device)
        lower_upper_limits = batch["lower_upper_limits"].to(self.device)
        motion_types = batch["motion_types"].to(self.device)
        is_eot = (
            batch["is_eot"].to(self.device)
            if isinstance(batch["is_eot"], torch.Tensor)
            else torch.tensor(batch["is_eot"], dtype=torch.bool, device=self.device)
        )
        is_non_eot = ~is_eot

        encode_pres = []
        for i, link_idx in enumerate(link_indices):
            if link_idx == 0:
                batch_size = encode_wholes.shape[0]
                expanded_sot = model_to_use.SoT.expand(batch_size, -1, -1)
                encode_pres.append(expanded_sot[i : i + 1])
            else:
                encode_pres.append(batch["encode_pres"][i : i + 1].to(self.device))
        encode_pres = torch.cat(encode_pres, dim=0)

        link_indices_tensor = (
            torch.tensor(link_indices, dtype=torch.long, device=self.device)
            if not isinstance(link_indices, torch.Tensor)
            else link_indices.to(self.device)
        )
        non_first_link_mask = link_indices_tensor > 0
        if non_first_link_mask.any():
            encode_pres = encode_pres.clone()
            dropout_rate = self.config.get("encode_pre_dropout_rate", None)
            if dropout_rate is not None and dropout_rate > 0:
                mask = torch.rand_like(encode_pres) > dropout_rate
                encode_pres[non_first_link_mask] = encode_pres[non_first_link_mask] * mask[non_first_link_mask]

        cond = {"dino": dino_features, "encode_pre": encode_pres, "encode_whole": encode_wholes}
        if self.config.get("history_mode") == "geometry_motion":
            cond["motion_history"] = batch["motion_histories"].to(self.device)
            cond["motion_history_lengths"] = batch[
                "motion_history_lengths"
            ].to(self.device)

        noise = torch.randn_like(target_labels)
        timesteps = torch.randint(0, self.num_train_timesteps, (target_labels.shape[0],), device=self.device)
        noisy_latents = self.noise_scheduler.add_noise(target_labels, noise, timesteps)

        link_indices_tensor = (
            link_indices.to(self.device)
            if isinstance(link_indices, torch.Tensor)
            else torch.tensor(link_indices, dtype=torch.long, device=self.device)
        )
        non_first_link_mask = link_indices_tensor > 0
        urdf_mask = has_urdf.bool() & non_first_link_mask & is_non_eot

        if getattr(self, "no_3d_whole", False):
            encoder_hidden_states_2 = cond["encode_pre"]
        else:
            encoder_hidden_states_2 = torch.cat([cond["encode_pre"], cond["encode_whole"]], dim=1)

        model_output = model_to_use.DiTRunner.model(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=cond["dino"],
            encoder_hidden_states_2=encoder_hidden_states_2,
            motion_history=cond.get("motion_history"),
            motion_history_lengths=cond.get("motion_history_lengths"),
        )

        if isinstance(model_output, dict):
            latent_pred = model_output["latent"]
            param1_pred = model_output["param1"]
            param2_pred = model_output["param2"]
            param3_pred = model_output["param3"]
            motion_type_pred = model_output.get("motion_type", None)
        else:
            if len(model_output) == 5:
                latent_pred, param1_pred, param2_pred, param3_pred, motion_type_pred = model_output
            else:
                latent_pred, param1_pred, param2_pred, param3_pred = model_output
                motion_type_pred = None

        noise_scheduler_config = self.model_config["ditrunner"]["noise_scheduler"]
        if noise_scheduler_config["prediction_type"] == "epsilon":
            target = noise
        elif noise_scheduler_config["prediction_type"] == "v_prediction":
            target = self.noise_scheduler.get_velocity(target_labels, noise, timesteps)
        else:
            target = target_labels

        if is_eot.any():
            eot_latent_loss = F.mse_loss(latent_pred[is_eot], target[is_eot])
        else:
            eot_latent_loss = torch.tensor(0.0, device=self.device)
        if is_non_eot.any():
            non_eot_latent_loss = F.mse_loss(latent_pred[is_non_eot], target[is_non_eot])
        else:
            non_eot_latent_loss = torch.tensor(0.0, device=self.device)
        latent_loss = non_eot_latent_loss + eot_latent_loss * 0.1
        total_loss = latent_loss

        urdf_timestep_mask = None
        param1_loss = None
        param2_loss = None
        param3_loss = None
        motion_type_loss = None
        motion_type_accuracy = None
        urdf_count = 0

        if urdf_mask.any() and self.config.get("train_urdf_params", False):
            timestep_mask = timesteps < self.urdf_loss_timestep_threshold
            urdf_timestep_mask = urdf_mask & timestep_mask
            if urdf_timestep_mask.any():
                param1_target = urdf_origins[urdf_timestep_mask]
                param2_target = urdf_axes[urdf_timestep_mask]
                param3_target = lower_upper_limits[urdf_timestep_mask]
                param1_pred_masked = param1_pred[urdf_timestep_mask]
                urdf_origins_masked = urdf_origins[urdf_timestep_mask]
                urdf_axes_masked = urdf_axes[urdf_timestep_mask]
                axes_abs = torch.abs(urdf_axes_masked)
                max_vals, max_indices = torch.max(axes_abs, dim=1)
                axes_abs_copy = axes_abs.clone()
                axes_abs_copy[torch.arange(len(max_indices), device=self.device), max_indices] = -1
                second_max_vals = torch.max(axes_abs_copy, dim=1)[0]
                is_onehot = (max_vals > 0.9) & (second_max_vals < 0.1)
                dim_mask = torch.ones(len(max_indices), 3, dtype=torch.bool, device=self.device)
                onehot_indices = torch.where(is_onehot)[0]
                if len(onehot_indices) > 0:
                    dim_mask[onehot_indices, max_indices[onehot_indices]] = False
                param1_loss = F.l1_loss(param1_pred_masked[dim_mask], param1_target[dim_mask])
                param2_pred_masked = param2_pred[urdf_timestep_mask]
                param2_loss = F.l1_loss(param2_pred_masked, param2_target)
                param3_pred_masked = param3_pred[urdf_timestep_mask]
                param3_loss = F.l1_loss(param3_pred_masked, param3_target)
                motion_type_loss = torch.tensor(0.0, device=self.device)
                motion_type_accuracy = 0.0
                urdf_count = urdf_timestep_mask.sum().item()
                if motion_type_pred is not None and urdf_count > 0:
                    motion_type_pred_masked = motion_type_pred[urdf_timestep_mask]
                    motion_types_masked = motion_types[urdf_timestep_mask]
                    class_counts = torch.bincount(motion_types_masked, minlength=2).float()
                    class_weights = class_counts.sum() / (2 * class_counts + 1e-6)
                    motion_type_loss = F.cross_entropy(
                        motion_type_pred_masked, motion_types_masked, weight=class_weights
                    )
                    pred_classes = torch.argmax(motion_type_pred_masked, dim=1)
                    correct = (pred_classes == motion_types_masked).float()
                    motion_type_accuracy = correct.mean().item()
                total_loss = (
                    latent_loss
                    + 0.01 * param1_loss
                    + param2_loss * 0.01
                    + param3_loss * 0.01
                    + motion_type_loss * 0.01
                )
            else:
                total_loss = latent_loss

        total_loss.backward()
        self.optimizer.step()

        if self.rank == 0 and self.use_wandb and wandb is not None:
            log_dict = {
                "train/total_loss": total_loss.item(),
                "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                "train/step": self.global_step,
            }
            if is_eot.any():
                log_dict["train/eot_latent_loss"] = eot_latent_loss.item()
            if is_non_eot.any():
                log_dict["train/non_eot_latent_loss"] = non_eot_latent_loss.item()
            wandb.log(log_dict)
            if (
                urdf_timestep_mask is not None
                and urdf_timestep_mask.any()
                and param1_loss is not None
            ):
                log_dict_urdf = {
                    "train/param1_loss": param1_loss.item(),
                    "train/param2_loss": param2_loss.item(),
                    "train/param3_loss": param3_loss.item(),
                }
                if motion_type_pred is not None and urdf_count > 0 and motion_type_loss is not None:
                    log_dict_urdf["train/motion_type_loss"] = motion_type_loss.item()
                    # log_dict_urdf["train/motion_type_accuracy"] = motion_type_accuracy
                    # log_dict_urdf["train/urdf_count"] = urdf_count
                    # unique_labels, counts = torch.unique(
                    #     motion_types[urdf_timestep_mask], return_counts=True
                    # )
                    # for label, count in zip(unique_labels, counts):
                    #     label_name = "revolute" if label.item() == 0 else "prismatic"
                    #     log_dict_urdf[f"train/motion_type_{label_name}_count"] = count.item()
                wandb.log(log_dict_urdf)

        self.global_step += 1
        return total_loss.item()

    def _compute_set_losses(self, batch):
        """Forward and compute the simultaneous fixed-slot objective."""
        model_to_use = self.model.module if self.is_distributed else self.model
        dino_features = model_to_use.dino_adapter(
            batch["dino_features"].to(self.device)
        )
        encode_wholes = batch["encode_whole"].to(self.device)
        target_labels = batch["target_labels"].to(self.device)
        batch_size, num_slots, seq_length, latent_dim = target_labels.shape
        if num_slots != self.config.get("num_part_slots", 5):
            raise ValueError(
                f"batch has {num_slots} slots, expected "
                f"{self.config.get('num_part_slots', 5)}"
            )

        noise = torch.randn_like(target_labels)
        object_timesteps = torch.randint(
            0, self.num_train_timesteps, (batch_size,), device=self.device
        )
        timesteps = object_timesteps[:, None].expand(-1, num_slots).reshape(-1)
        target_flat = target_labels.reshape(-1, seq_length, latent_dim)
        noise_flat = noise.reshape_as(target_flat)
        noisy_latents = self.noise_scheduler.add_noise(
            target_flat, noise_flat, timesteps
        )

        model_output = model_to_use.DiTRunner.model(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=dino_features.repeat_interleave(
                num_slots, dim=0
            ),
            encoder_hidden_states_2=encode_wholes.repeat_interleave(
                num_slots, dim=0
            ),
        )
        latent_pred = model_output["latent"].reshape_as(target_labels)
        presence_pred = model_output["presence"].reshape(batch_size, num_slots)
        part_order_pred = model_output["part_order"].reshape(
            batch_size, num_slots, num_slots
        )
        param1_pred = model_output["param1"].reshape(batch_size, num_slots, 3)
        param2_pred = model_output["param2"].reshape(batch_size, num_slots, 3)
        param3_pred = model_output["param3"].reshape(batch_size, num_slots, 2)
        motion_type_pred = model_output["motion_type"].reshape(
            batch_size, num_slots, 2
        )

        noise_scheduler_config = self.model_config["ditrunner"]["noise_scheduler"]
        if noise_scheduler_config["prediction_type"] == "epsilon":
            geometry_target = noise
        elif noise_scheduler_config["prediction_type"] == "v_prediction":
            geometry_target = self.noise_scheduler.get_velocity(
                target_flat, noise_flat, timesteps
            ).reshape_as(target_labels)
        else:
            geometry_target = target_labels

        presence_target = batch["presence"].to(self.device)
        real_mask = presence_target.bool()
        padding_mask = ~real_mask
        zero = latent_pred.sum() * 0.0
        geometry_real = (
            F.mse_loss(latent_pred[real_mask], geometry_target[real_mask])
            if real_mask.any()
            else zero
        )
        geometry_padding = (
            F.mse_loss(
                latent_pred[padding_mask], geometry_target[padding_mask]
            )
            if padding_mask.any()
            else zero
        )
        geometry_loss = geometry_real + 0.1 * geometry_padding
        presence_loss = F.binary_cross_entropy_with_logits(
            presence_pred, presence_target
        )

        part_order_targets = batch["part_order_targets"].to(self.device)
        part_order_loss = (
            F.cross_entropy(
                part_order_pred[real_mask], part_order_targets[real_mask]
            )
            if real_mask.any()
            else zero
        )

        timestep_mask = object_timesteps[:, None] < self.urdf_loss_timestep_threshold
        motion_mask = (
            real_mask
            & batch["has_urdf"].to(self.device).bool()
            & (part_order_targets > 0)
            & timestep_mask
        )
        if motion_mask.any() and self.config.get("train_urdf_params", False):
            origin_loss = F.l1_loss(
                param1_pred[motion_mask],
                batch["urdf_origins"].to(self.device)[motion_mask],
            )
            axis_loss = F.l1_loss(
                param2_pred[motion_mask],
                batch["urdf_axes"].to(self.device)[motion_mask],
            )
            limits_loss = F.l1_loss(
                param3_pred[motion_mask],
                batch["lower_upper_limits"].to(self.device)[motion_mask],
            )
            motion_type_loss = F.cross_entropy(
                motion_type_pred[motion_mask],
                batch["motion_types"].to(self.device)[motion_mask],
            )
        else:
            origin_loss = zero
            axis_loss = zero
            limits_loss = zero
            motion_type_loss = zero

        auxiliary_loss = (
            presence_loss
            + part_order_loss
            + origin_loss
            + axis_loss
            + limits_loss
            + motion_type_loss
        )
        total_loss = geometry_loss + 0.01 * auxiliary_loss
        return {
            "total": total_loss,
            "geometry": geometry_loss,
            "geometry_real": geometry_real,
            "geometry_padding": geometry_padding,
            "presence": presence_loss,
            "part_order": part_order_loss,
            "origin": origin_loss,
            "axis": axis_loss,
            "limits": limits_loss,
            "motion_type": motion_type_loss,
        }

    def _train_step_set(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        losses = self._compute_set_losses(batch)
        losses["total"].backward()
        self.optimizer.step()
        if self.rank == 0 and self.use_wandb and wandb is not None:
            wandb.log(
                {
                    f"train/{name}_loss": value.item()
                    for name, value in losses.items()
                }
                | {
                    "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                    "train/step": self.global_step,
                }
            )
        self.global_step += 1
        return losses["total"].item()

    def _validate_set(self, val_loader):
        self.model.eval()
        totals = {}
        num_batches = 0
        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc="Validating", disable=self.local_rank != 0
            ):
                losses = self._compute_set_losses(batch)
                for name, value in losses.items():
                    totals[name] = totals.get(name, 0.0) + value.item()
                num_batches += 1
        if num_batches == 0:
            raise ValueError("validation loader is empty")
        self.last_set_val_metrics = {
            name: value / num_batches for name, value in totals.items()
        }
        metrics = self.last_set_val_metrics
        return (
            metrics["total"],
            metrics["origin"],
            metrics["axis"],
            metrics["limits"],
            metrics["geometry"],
            0.0,
            0.0,
            metrics["motion_type"],
        )

    def validate(self, val_loader):
        """Validation - use complete denoising process."""
        if self.generation_mode == "set":
            return self._validate_set(val_loader)
        self.model.eval()
        total_loss = 0
        num_batches = 0
        total_param1_loss = 0
        total_param2_loss = 0
        total_param3_loss = 0
        total_motion_type_loss = 0
        total_motion_type_accuracy = 0.0
        total_latent_loss = 0
        total_eot_latent_loss = 0
        total_non_eot_latent_loss = 0
        eot_batch_count = 0
        non_eot_batch_count = 0
        urdf_count = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", disable=self.local_rank != 0):
                dino_features_raw = batch["dino_features"].to(self.device)
                model_to_use = self.model.module if self.is_distributed else self.model
                dino_features = model_to_use.dino_adapter(dino_features_raw)
                encode_wholes = batch["encode_wholes"].to(self.device)
                target_labels = batch["target_labels"].to(self.device)
                link_indices = batch["link_indices"]
                urdf_origins = batch["urdf_origins"].to(self.device)
                urdf_axes = batch["urdf_axes"].to(self.device)
                has_urdf = batch["has_urdf"].to(self.device)
                lower_upper_limits = batch["lower_upper_limits"].to(self.device)
                motion_types = batch["motion_types"].to(self.device)
                is_eot = (
                    batch["is_eot"].to(self.device)
                    if isinstance(batch["is_eot"], torch.Tensor)
                    else torch.tensor(batch["is_eot"], dtype=torch.bool, device=self.device)
                )
                is_non_eot = ~is_eot

                encode_pres = []
                for i, link_idx in enumerate(link_indices):
                    if link_idx == 0:
                        batch_size = encode_wholes.shape[0]
                        expanded_sot = model_to_use.SoT.expand(batch_size, -1, -1)
                        encode_pres.append(expanded_sot[i : i + 1])
                    else:
                        encode_pres.append(batch["encode_pres"][i : i + 1].to(self.device))
                encode_pres = torch.cat(encode_pres, dim=0)
                cond = {"dino": dino_features, "encode_pre": encode_pres, "encode_whole": encode_wholes}
                if self.config.get("history_mode") == "geometry_motion":
                    cond["motion_history"] = batch["motion_histories"].to(
                        self.device
                    )
                    cond["motion_history_lengths"] = batch[
                        "motion_history_lengths"
                    ].to(self.device)

                noisy_latents = torch.randn_like(target_labels)
                timesteps = torch.randint(
                    0, self.num_train_timesteps, (target_labels.shape[0],), device=self.device
                )
                predicted_latents = self.noise_scheduler.add_noise(
                    target_labels, noisy_latents, timesteps
                )
                link_indices_tensor = (
                    link_indices.to(self.device)
                    if isinstance(link_indices, torch.Tensor)
                    else torch.tensor(link_indices, dtype=torch.long, device=self.device)
                )
                non_first_link_mask = link_indices_tensor > 0
                urdf_mask = has_urdf.bool() & non_first_link_mask & is_non_eot

                if getattr(self, "no_3d_whole", False):
                    encoder_hidden_states_2 = cond["encode_pre"]
                else:
                    encoder_hidden_states_2 = torch.cat(
                        [cond["encode_pre"], cond["encode_whole"]], dim=1
                    )
                model_output = model_to_use.DiTRunner.model(
                    hidden_states=predicted_latents,
                    timestep=timesteps,
                    encoder_hidden_states=cond["dino"],
                    encoder_hidden_states_2=encoder_hidden_states_2,
                    motion_history=cond.get("motion_history"),
                    motion_history_lengths=cond.get("motion_history_lengths"),
                )
                if isinstance(model_output, dict):
                    latent_pred = model_output["latent"]
                    param1_pred = model_output["param1"]
                    param2_pred = model_output["param2"]
                    param3_pred = model_output["param3"]
                    motion_type_pred = model_output.get("motion_type", None)
                else:
                    if len(model_output) == 5:
                        latent_pred, param1_pred, param2_pred, param3_pred, motion_type_pred = model_output
                    else:
                        latent_pred, param1_pred, param2_pred, param3_pred = model_output
                        motion_type_pred = None

                noise_scheduler_config = self.model_config["ditrunner"]["noise_scheduler"]
                if noise_scheduler_config["prediction_type"] == "epsilon":
                    target = noisy_latents
                elif noise_scheduler_config["prediction_type"] == "v_prediction":
                    target = self.noise_scheduler.get_velocity(
                        target_labels, noisy_latents, timesteps
                    )
                else:
                    target = target_labels

                if is_eot.any():
                    eot_latent_loss = F.mse_loss(latent_pred[is_eot], target[is_eot])
                    total_eot_latent_loss += eot_latent_loss.item()
                    eot_batch_count += 1
                if is_non_eot.any():
                    non_eot_latent_loss = F.mse_loss(
                        latent_pred[is_non_eot], target[is_non_eot]
                    )
                    total_non_eot_latent_loss += non_eot_latent_loss.item()
                    non_eot_batch_count += 1

                latent_loss = F.mse_loss(latent_pred, target)
                link_indices_tensor = (
                    link_indices.to(self.device)
                    if isinstance(link_indices, torch.Tensor)
                    else torch.tensor(link_indices, dtype=torch.long, device=self.device)
                )
                non_first_link_mask = link_indices_tensor > 0
                urdf_mask = has_urdf.bool() & non_first_link_mask & is_non_eot
                total_batch_loss = latent_loss

                if urdf_mask.any() and self.config.get("train_urdf_params", False):
                    timestep_mask = timesteps < self.urdf_loss_timestep_threshold
                    urdf_timestep_mask = urdf_mask & timestep_mask
                    if urdf_timestep_mask.any():
                        param1_target = urdf_origins[urdf_timestep_mask]
                        param2_target = urdf_axes[urdf_timestep_mask]
                        param3_target = lower_upper_limits[urdf_timestep_mask]
                        urdf_samples_in_batch = urdf_timestep_mask.sum().item()
                        urdf_count += urdf_samples_in_batch
                        param2_pred_masked = param2_pred[urdf_timestep_mask]
                        param2_loss = F.l1_loss(param2_pred_masked, param2_target)
                        param1_pred_masked = param1_pred[urdf_timestep_mask]
                        urdf_origins_masked = urdf_origins[urdf_timestep_mask]
                        urdf_axes_masked = urdf_axes[urdf_timestep_mask]
                        axes_abs = torch.abs(urdf_axes_masked)
                        max_vals, max_indices = torch.max(axes_abs, dim=1)
                        axes_abs_copy = axes_abs.clone()
                        axes_abs_copy[
                            torch.arange(len(max_indices), device=self.device), max_indices
                        ] = -1
                        second_max_vals = torch.max(axes_abs_copy, dim=1)[0]
                        is_onehot = (max_vals > 0.9) & (second_max_vals < 0.1)
                        dim_mask = torch.ones(
                            len(max_indices), 3, dtype=torch.bool, device=self.device
                        )
                        onehot_indices = torch.where(is_onehot)[0]
                        if len(onehot_indices) > 0:
                            dim_mask[onehot_indices, max_indices[onehot_indices]] = False
                        param1_loss = F.l1_loss(
                            param1_pred_masked[dim_mask], param1_target[dim_mask]
                        )
                        param3_pred_masked = param3_pred[urdf_timestep_mask]
                        param3_loss = F.l1_loss(param3_pred_masked, param3_target)
                        motion_type_loss = torch.tensor(0.0, device=self.device)
                        batch_accuracy = 0.0
                        if motion_type_pred is not None and urdf_samples_in_batch > 0:
                            motion_type_pred_masked = motion_type_pred[urdf_timestep_mask]
                            motion_types_masked = motion_types[urdf_timestep_mask]
                            class_counts = torch.bincount(
                                motion_types_masked, minlength=2
                            ).float()
                            class_weights = class_counts.sum() / (
                                2 * class_counts + 1e-6
                            )
                            motion_type_loss = F.cross_entropy(
                                motion_type_pred_masked,
                                motion_types_masked,
                                weight=class_weights,
                            )
                            pred_classes = torch.argmax(
                                motion_type_pred_masked, dim=1
                            )
                            correct = (
                                pred_classes == motion_types_masked
                            ).float()
                            batch_accuracy = correct.mean().item()
                        total_batch_loss = (
                            latent_loss
                            + 0.01 * param1_loss
                            + param2_loss * 0.01
                            + param3_loss * 0.01
                            + motion_type_loss * 0.01
                        )
                        total_param1_loss += param1_loss.item() * urdf_samples_in_batch
                        total_param2_loss += param2_loss.item() * urdf_samples_in_batch
                        total_param3_loss += param3_loss.item() * urdf_samples_in_batch
                        total_motion_type_loss += (
                            motion_type_loss.item() * urdf_samples_in_batch
                        )
                        total_motion_type_accuracy += (
                            batch_accuracy * urdf_samples_in_batch
                        )
                total_loss += total_batch_loss.item()
                total_latent_loss += latent_loss.item()
                num_batches += 1

        avg_eot_loss = (
            total_eot_latent_loss / eot_batch_count if eot_batch_count > 0 else 0.0
        )
        avg_non_eot_loss = (
            total_non_eot_latent_loss / non_eot_batch_count
            if non_eot_batch_count > 0
            else 0.0
        )
        avg_param1_loss = total_param1_loss / urdf_count if urdf_count > 0 else 0.0
        avg_param2_loss = total_param2_loss / urdf_count if urdf_count > 0 else 0.0
        avg_param3_loss = total_param3_loss / urdf_count if urdf_count > 0 else 0.0
        avg_motion_type_loss = (
            total_motion_type_loss / urdf_count if urdf_count > 0 else 0.0
        )
        return (
            total_loss / num_batches,
            avg_param1_loss,
            avg_param2_loss,
            avg_param3_loss,
            total_latent_loss / num_batches,
            avg_eot_loss,
            avg_non_eot_loss,
            avg_motion_type_loss,
        )

    def train(self, train_loader, val_loader, max_epochs=100, save_interval=10):
        """Training loop."""
        best_val_loss = float("inf")
        noise_scheduler_config = self.model_config["ditrunner"]["noise_scheduler"]
        print("prediction_type:", noise_scheduler_config["prediction_type"])
        print("train_urdf_params:", self.config.get("train_urdf_params", False))
        print("generation_mode:", self.generation_mode)
        if self.generation_mode == "set":
            print(
                "gradient_checkpointing:",
                self.model.DiTRunner.model.gradient_checkpointing
                if not self.is_distributed
                else self.model.module.DiTRunner.model.gradient_checkpointing,
            )
        if self.rank == 0:
            config_path = os.path.join(self.save_checkpoint_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(self.config, f, indent=2)

        for epoch in range(max_epochs):
            if self.is_distributed:
                train_loader.sampler.set_epoch(epoch)
                val_loader.sampler.set_epoch(epoch)

            train_losses = []
            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{max_epochs}",
                disable=self.local_rank != 0,
            )
            for batch in pbar:
                loss = self.train_step(batch)
                train_losses.append(loss)
                if self.local_rank == 0:
                    pbar.set_postfix({"loss": f"{loss:.6f}"})

            avg_train_loss = np.mean(train_losses)
            (
                val_loss,
                val_param1_loss,
                val_param2_loss,
                val_param3_loss,
                val_latent_loss,
                val_eot_latent_loss,
                val_non_eot_latent_loss,
                val_motion_type_loss,
            ) = self.validate(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            if self.rank == 0:
                print(
                    f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                    f"Val Param1 Loss: {val_param1_loss:.6f}, Val Param2 Loss: {val_param2_loss:.6f}, "
                    f"Val Param3 Loss: {val_param3_loss:.6f}, Val Latent Loss: {val_latent_loss:.6f}, "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )
                if self.config.get("train_eot", False):
                    print(
                        f"  Val EOT Loss: {val_eot_latent_loss:.6f}, Val Non-EOT Loss: {val_non_eot_latent_loss:.6f}"
                    )
                if val_motion_type_loss > 0:
                    print(f"  Val Motion Type Loss: {val_motion_type_loss:.6f}")
                if self.generation_mode == "set":
                    set_metrics = self.last_set_val_metrics
                    print(
                        "  Val Presence Loss: "
                        f"{set_metrics['presence']:.6f}, Val Part Order Loss: "
                        f"{set_metrics['part_order']:.6f}, Val Geometry Real Loss: "
                        f"{set_metrics['geometry_real']:.6f}, "
                        "Val Geometry Padding Loss: "
                        f"{set_metrics['geometry_padding']:.6f}"
                    )

            if self.rank == 0:
                if (epoch + 1) % save_interval == 0:
                    model_to_save = self.model.module if self.is_distributed else self.model
                    model_state = model_to_save.get_state_dict_for_saving()
                    checkpoint = {
                        "epoch": epoch,
                        "model_state_dict": model_state,
                        "val_loss": val_loss,
                    }
                    if self.save_optimizer:
                        checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
                    checkpoint.update(
                        {
                            "config": self.config,
                            "model_config": self.model_config,
                            "experiment_name": self.experiment_name,
                            "cache_paths": self.cache_paths,
                            "data_root_map": self.data_root_map,
                            "learning_rate": self.optimizer.param_groups[0]["lr"],
                            "weight_decay": self.optimizer.param_groups[0]["weight_decay"],
                            "batch_size": self.config.get("batch_size", None),
                            "max_epochs": self.config.get("max_epochs", None),
                            "save_interval": save_interval,
                            "train_urdf_params": self.config.get("train_urdf_params", False),
                            "train_eot": self.config.get("train_eot", False),
                            "encode_pre_dropout_rate": self.config.get(
                                "encode_pre_dropout_rate", None
                            ),
                            "init_mode": self.config.get("init_mode", "train_from_scratch"),
                            "history_mode": self.config.get(
                                "history_mode", "geometry"
                            ),
                            "generation_mode": self.generation_mode,
                            "num_part_slots": self.config.get(
                                "num_part_slots", 5
                            ),
                            "checkpoint_path": self.config.get("checkpoint_path", None),
                            "save_checkpoint_dir": self.save_checkpoint_dir,
                            "model_config_path": self.config.get("model_config_path", None),
                            "use_wandb": self.use_wandb,
                            "device": str(self.device),
                            "world_size": self.world_size,
                            "is_distributed": self.is_distributed,
                            "training_start_time": self.training_start_time,
                            "checkpoint_save_time": datetime.now().isoformat(),
                            "python_version": sys.version,
                            "torch_version": torch.__version__,
                            "platform": platform.platform(),
                            "save_optimizer": self.save_optimizer,
                        }
                    )
                    torch.save(
                        checkpoint,
                        os.path.join(self.save_checkpoint_dir, f"epoch_{epoch+1}.pth"),
                    )

            if self.rank == 0 and self.use_wandb and wandb is not None:
                log_dict = {
                    "epoch/train_loss": avg_train_loss,
                    "epoch/val_loss": val_loss,
                    "epoch/val_latent_loss": val_latent_loss,
                    "epoch/learning_rate": self.optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                }
                if self.config.get("train_eot", False):
                    log_dict["epoch/val_eot_latent_loss"] = val_eot_latent_loss
                    log_dict["epoch/val_non_eot_latent_loss"] = val_non_eot_latent_loss
                if val_param1_loss > 0:
                    log_dict["epoch/val_param1_loss"] = val_param1_loss
                if val_param2_loss > 0:
                    log_dict["epoch/val_param2_loss"] = val_param2_loss
                if val_param3_loss > 0:
                    log_dict["epoch/val_param3_loss"] = val_param3_loss
                if val_motion_type_loss > 0:
                    log_dict["epoch/val_motion_type_loss"] = val_motion_type_loss
                wandb.log(log_dict)

        if self.rank == 0:
            if self.use_wandb and wandb is not None:
                wandb.finish()
            print(f"Training completed! Experiment name: {self.experiment_name}")
            print(f"Best validation loss: {best_val_loss:.6f}")
