"""URDFModel: DiT runner + 3D VAE + adapters, checkpoint loading."""
import os
import torch
import torch.nn as nn
from TripoSG.triposg.models.autoencoders import TripoSGVAEModel

from .utils import load_model_config, get_default_config_path
from .dit_runner import DiTRunner
from urdf_anything.data.urdf_utils import build_motion_cue


class URDFModel(nn.Module):
    def __init__(
        self,
        config_path=None,
        config_dict=None,
        device="cuda",
        init_mode="train_from_scratch",
        checkpoint_path=None,
    ):
        super().__init__()
        self.device = device
        self.init_mode = init_mode

        if config_path is not None:
            self.config = load_model_config(config_path)
        elif config_dict is not None:
            self.config = config_dict
        else:
            default_config_path = get_default_config_path()
            if os.path.exists(default_config_path):
                self.config = load_model_config(default_config_path)
            else:
                raise ValueError(
                    "Must provide config_path or config_dict, or ensure default config file exists"
                )

        load_3d_model = init_mode == "inference"
        load_dit_pretrained = init_mode == "train_from_scratch"
        dit_config = self.config["ditrunner"].copy()
        dit_pretrained_path = os.environ.get(
            "URDF_ANYTHING_TRIPOSG_DIT_PATH",
            self.config.get(
                "dit_pretrained_path",
                "TripoSG/transformer/diffusion_pytorch_model.safetensors",
            ),
        )
        self.DiTRunner = DiTRunner(
            dit_config,
            dit_pretrained_path=dit_pretrained_path,
            load_dit_pretrained=load_dit_pretrained,
            device=device,
        ).to(device)

        self.SoT = nn.Parameter(
            torch.randn(
                1,
                self.config["ditrunner"]["seq_length"],
                self.config["sample"]["latent_dim"],
            ),
            requires_grad=True,
        )
        self.EoT = torch.zeros_like(self.SoT)
        self.current_latent = None
        self.current_param1 = None
        self.current_param2 = None

        if load_3d_model:
            self.Build_3D_EncoderDecoder()
            self.Encoder_3D = self.model_3d.encode
            self.Decoder_3D = self.model_3d.decode
        else:
            self.model_3d = None
            self.Encoder_3D = None
            self.Decoder_3D = None

        dino_hidden_dim = self.config.get("dino_hidden_dim", 4096)
        self.dino_adapter = nn.Sequential(
            nn.Linear(dino_hidden_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
        )
        self.feature3d_adapter = nn.Sequential(
            nn.Linear(64, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
        )

        if init_mode in ["resume_from_ckpt", "inference"]:
            if checkpoint_path is None:
                raise ValueError(f"{init_mode} mode needs to provide checkpoint_path")
            self.load_checkpoint(checkpoint_path, load_3d_model=False)

    def Build_3D_EncoderDecoder(self):
        model_3d_config = self.config.get("model_3d", {})
        pretrained_path = os.environ.get(
            "URDF_ANYTHING_TRIPOSG_VAE_PATH",
            model_3d_config.get(
                "pretrained_path", "TripoSG/pretrained_weights/TripoSG"
            ),
        )
        subfolder = model_3d_config.get("subfolder", "vae")
        self.model_3d = TripoSGVAEModel.from_pretrained(
            pretrained_path, subfolder=subfolder
        ).to(self.device)

    def load_checkpoint(self, checkpoint_path, load_3d_model=False):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        model_state_dict = checkpoint.get("model_state_dict", checkpoint)
        if not load_3d_model:
            filtered_state_dict = {
                k: v
                for k, v in model_state_dict.items()
                if not k.startswith("model_3d.")
            }
            model_state_dict = filtered_state_dict
        missing, unexpected = self.load_state_dict(model_state_dict, strict=False)
        print(f"Loaded checkpoint: {checkpoint_path}")
        if missing:
            missing_filtered = [k for k in missing if not k.startswith("model_3d.")]
            if missing_filtered:
                print(
                    f"Missing keys: {missing_filtered[:10]}..."
                    if len(missing_filtered) > 10
                    else f"Missing keys: {missing_filtered}"
                )
        if unexpected:
            print(
                f"Unexpected keys: {unexpected[:10]}..."
                if len(unexpected) > 10
                else f"Unexpected keys: {unexpected}"
            )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path,
        config_path=None,
        device="cuda",
        init_mode="inference",
    ):
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if config_path is None and "model_config" in checkpoint:
            config_dict = checkpoint["model_config"]
            return cls(
                config_dict=config_dict,
                device=device,
                init_mode=init_mode,
                checkpoint_path=checkpoint_path,
            )
        if config_path is None:
            config_path = get_default_config_path()
        return cls(
            config_path=config_path,
            device=device,
            init_mode=init_mode,
            checkpoint_path=checkpoint_path,
        )

    def get_state_dict_for_saving(self):
        state_dict = self.state_dict()
        filtered_state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("model_3d.")
        }
        return filtered_state_dict

    def forward(self, input_dict):
        results = []
        motion_history = []
        for idx, dino_feat in enumerate(input_dict["dino_list"]):
            if idx == 0:
                encode_pre = self.SoT
            else:
                encode_pre = self.current_latent
            encode_whole = input_dict["encode_whole"]
            cond = {
                "dino": dino_feat,
                "encode_pre": encode_pre,
                "encode_whole": encode_whole,
            }
            if self.config["ditrunner"].get("history_mode") == "geometry_motion":
                if motion_history:
                    history_tensor = torch.stack(motion_history).unsqueeze(0)
                else:
                    history_tensor = torch.empty(1, 0, 10, device=self.device)
                cond["motion_history"] = history_tensor.to(self.device)
                cond["motion_history_lengths"] = torch.tensor(
                    [len(motion_history)], dtype=torch.long, device=self.device
                )
            output = self.DiTRunner.conditional_sample(cond)
            self.current_latent = output["latent"]
            self.current_param1 = output["param1"]
            self.current_param2 = output["param2"]
            self.current_motion_type = output.get("motion_type", None)
            if idx > 0 and self.current_motion_type is not None:
                motion_history.append(
                    build_motion_cue(
                        self.current_param1[0],
                        self.current_param2[0],
                        output["param3"][0],
                        torch.argmax(self.current_motion_type[0]),
                    ).to(self.device)
                )
            mesh, points, self.current_latent = self.Sample_SDF([output["latent"]])
            results.append(
                {
                    "latent": output["latent"],
                    "param1": output["param1"],
                    "param2": output["param2"],
                    "mesh": mesh,
                    "points": points,
                }
            )
        return results
