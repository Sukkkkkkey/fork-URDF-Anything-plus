"""DiT runner: MultiOutputDiT + scheduler, conditional sampling."""
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from safetensors.torch import load_file
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from .multi_output_dit import MultiOutputDiTModel
from .utils import load_model_config


class DiTRunner(nn.Module):
    def __init__(
        self,
        config,
        dit_pretrained_path=None,
        load_dit_pretrained=True,
        device="cuda",
    ):
        super().__init__()
        self.model = MultiOutputDiTModel(
            num_attention_heads=16,
            width=2048,
            in_channels=64,
            num_layers=21,
            cross_attention_dim=1024,
            additional_output_dims=(3, 3, 2),
            shared_hidden_dim=512,
            history_mode=config.get("history_mode", "geometry"),
            motion_history_dim=config.get("motion_history_dim", 10),
            motion_history_hidden_dim=config.get(
                "motion_history_hidden_dim", 256
            ),
            generation_mode=config.get("generation_mode", "autoregressive"),
            num_part_slots=config.get("num_part_slots", 5),
        ).to(device)

        if load_dit_pretrained:
            if dit_pretrained_path is None:
                dit_pretrained_path = config.get(
                    "dit_pretrained_path",
                    "TripoSG/transformer/diffusion_pytorch_model.safetensors",
                )
            if os.path.exists(dit_pretrained_path):
                state_dict = load_file(dit_pretrained_path)
                model_dict = self.model.state_dict()
                compatible_dict = {
                    k: v
                    for k, v in state_dict.items()
                    if k in model_dict and v.shape == model_dict[k].shape
                }
                model_dict.update(compatible_dict)
                missing, unexpected = self.model.load_state_dict(model_dict, strict=False)
                print(f"==> Loaded MultiOutputDiT checkpoint from {dit_pretrained_path}")
                if missing:
                    print("Missing keys:", missing)
                if unexpected:
                    print("Unexpected keys:", unexpected)
            else:
                print(
                    f"Warning: DiT pretrained weight path not found: {dit_pretrained_path}, using random initialization"
                )

        noise_scheduler_config = config["noise_scheduler"]
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=noise_scheduler_config["num_train_timesteps"],
            beta_start=noise_scheduler_config["beta_start"],
            prediction_type=noise_scheduler_config["prediction_type"],
            clip_sample=noise_scheduler_config["clip_sample"],
        )
        self.num_train_timesteps = noise_scheduler_config["num_train_timesteps"]
        self.num_inference_timesteps = noise_scheduler_config["num_inference_timesteps"]
        self.prediction_type = noise_scheduler_config["prediction_type"]
        self.generation_mode = config.get("generation_mode", "autoregressive")
        self.num_part_slots = int(config.get("num_part_slots", 5))

    def conditional_sample(self, cond, generator: torch.Generator | None = None):
        noisy_latents = torch.randn(
            cond["encode_whole"].shape,
            device=cond["encode_whole"].device,
            generator=generator,
        )
        batch_size = cond["encode_whole"].shape[0]
        device = cond["encode_whole"].device
        noisy_param1 = torch.randn(
            batch_size, 3, device=device, generator=generator
        )
        noisy_param2 = torch.randn(
            batch_size, 3, device=device, generator=generator
        )
        noisy_param3 = torch.randn(
            batch_size, 2, device=device, generator=generator
        )
        motion_type_final = None
        motion_type_pred = None
        self.noise_scheduler.set_timesteps(
            self.num_inference_timesteps, device=device
        )

        for t in tqdm(self.noise_scheduler.timesteps, desc="Sampling"):
            t = t.to(device=device)
            if t.dim() == 0:
                t = t.unsqueeze(0)
            elif t.dim() > 1:
                t = t.flatten()
            use_3d_whole = cond.get("use_3d_whole", True)
            if use_3d_whole:
                encoder_hidden_states_2 = torch.cat(
                    [cond["encode_pre"], cond["encode_whole"]], dim=1
                )
            else:
                encoder_hidden_states_2 = cond["encode_pre"]
            model_output = self.model(
                hidden_states=noisy_latents,
                timestep=t,
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
                    (
                        latent_pred,
                        param1_pred,
                        param2_pred,
                        param3_pred,
                        motion_type_pred,
                    ) = model_output
                else:
                    latent_pred, param1_pred, param2_pred, param3_pred = model_output
                    motion_type_pred = None
            t_int = t.item() if t.numel() == 1 else t[0].item()
            noisy_latents = self.noise_scheduler.step(
                latent_pred, t_int, noisy_latents
            ).prev_sample
            noisy_param1 = param1_pred
            noisy_param2 = param2_pred
            noisy_param3 = param3_pred
            if motion_type_pred is not None:
                motion_type_final = motion_type_pred
        return {
            "latent": noisy_latents,
            "param1": noisy_param1,
            "param2": noisy_param2,
            "param3": noisy_param3,
            "motion_type": motion_type_final,
        }

    def conditional_sample_set(self, cond, generator: torch.Generator | None = None):
        """Denoise all fixed part slots together without inter-slot attention."""
        if self.generation_mode != "set":
            raise ValueError("conditional_sample_set requires generation_mode=set")
        batch_size, seq_length, latent_dim = cond["encode_whole"].shape
        device = cond["encode_whole"].device
        noisy_latents = torch.randn(
            batch_size,
            self.num_part_slots,
            seq_length,
            latent_dim,
            device=device,
            generator=generator,
        )
        noisy_latents = noisy_latents.flatten(0, 1)
        dino = cond["dino"].repeat_interleave(self.num_part_slots, dim=0)
        encode_whole = cond["encode_whole"].repeat_interleave(
            self.num_part_slots, dim=0
        )
        self.noise_scheduler.set_timesteps(
            self.num_inference_timesteps, device=device
        )
        model_output = None
        for timestep in tqdm(self.noise_scheduler.timesteps, desc="Sampling set"):
            timestep = timestep.to(device=device)
            if timestep.dim() == 0:
                timestep = timestep.unsqueeze(0)
            model_timesteps = timestep.expand(batch_size * self.num_part_slots)
            model_output = self.model(
                hidden_states=noisy_latents,
                timestep=model_timesteps,
                encoder_hidden_states=dino,
                encoder_hidden_states_2=encode_whole,
            )
            timestep_int = int(timestep.flatten()[0].item())
            noisy_latents = self.noise_scheduler.step(
                model_output["latent"], timestep_int, noisy_latents
            ).prev_sample

        if model_output is None:
            raise RuntimeError("set sampler has no inference timesteps")

        def reshape_slots(value):
            return value.reshape(batch_size, self.num_part_slots, *value.shape[1:])

        return {
            "latent": noisy_latents.reshape(
                batch_size, self.num_part_slots, seq_length, latent_dim
            ),
            "param1": reshape_slots(model_output["param1"]),
            "param2": reshape_slots(model_output["param2"]),
            "param3": reshape_slots(model_output["param3"]),
            "motion_type": reshape_slots(model_output["motion_type"]),
            "presence": reshape_slots(model_output["presence"]),
            "part_order": reshape_slots(model_output["part_order"]),
        }
