"""Multi-output DiT model: latent + param1/2/3 + motion_type heads."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from .dit_triposg import TripoSGDiTModel


class MultiOutputDiTModel(TripoSGDiTModel):
    def __init__(
        self,
        num_attention_heads: int = 16,
        width: int = 2048,
        in_channels: int = 64,
        num_layers: int = 21,
        cross_attention_dim: int = 1024,
        use_cross_attention_2: bool = True,
        cross_attention_2_dim: int = 64,
        additional_output_dims: tuple = (3, 3, 2),
        shared_hidden_dim: int = 512,
        history_mode: str = "geometry",
        motion_history_dim: int = 10,
        motion_history_hidden_dim: int = 256,
    ):
        super().__init__(
            num_attention_heads=num_attention_heads,
            width=width,
            in_channels=in_channels,
            num_layers=num_layers,
            cross_attention_dim=cross_attention_dim,
            use_cross_attention_2=use_cross_attention_2,
            cross_attention_2_dim=cross_attention_2_dim,
        )
        self.additional_output_dims = additional_output_dims
        self.shared_hidden_dim = shared_hidden_dim
        if history_mode not in ("geometry", "geometry_motion"):
            raise ValueError(f"unsupported history_mode: {history_mode}")
        self.history_mode = history_mode
        self.motion_history_hidden_dim = motion_history_hidden_dim
        if self.history_mode == "geometry_motion":
            self.motion_history_encoder = nn.GRU(
                input_size=motion_history_dim,
                hidden_size=motion_history_hidden_dim,
                num_layers=1,
                batch_first=True,
            )
            for block in self.blocks:
                block.enable_motion_adaln(motion_history_hidden_dim)
        else:
            self.motion_history_encoder = None
        self.proj_out = nn.Linear(self.inner_dim, self.shared_hidden_dim, bias=True)
        self._create_output_heads()

    def encode_motion_history(self, motion_history, motion_history_lengths):
        batch_size = motion_history.shape[0]
        condition = motion_history.new_zeros(
            batch_size, self.motion_history_hidden_dim
        )
        nonempty = motion_history_lengths > 0
        if not nonempty.any():
            return condition
        indices = nonempty.nonzero(as_tuple=False).flatten()
        sequences = motion_history.index_select(0, indices)
        lengths = motion_history_lengths.index_select(0, indices).to("cpu")
        packed = pack_padded_sequence(
            sequences,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.motion_history_encoder(packed)
        return condition.index_copy(0, indices, hidden[-1])

    def _create_output_heads(self):
        self.latent_head = nn.Sequential(
            nn.Linear(self.shared_hidden_dim, self.out_channels),
        )
        self.param1_head = nn.Sequential(
            nn.Linear(self.shared_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.additional_output_dims[0]),
        )
        self.param1_attention = nn.Sequential(
            nn.Linear(self.shared_hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.param2_head = nn.Sequential(
            nn.Linear(self.shared_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.additional_output_dims[1]),
        )
        self.param2_attention = nn.Sequential(
            nn.Linear(self.shared_hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.param3_head = nn.Sequential(
            nn.Linear(self.shared_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.additional_output_dims[2]),
        )
        self.param3_attention = nn.Sequential(
            nn.Linear(self.shared_hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.motion_type_head = nn.Sequential(
            nn.Linear(self.shared_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.motion_type_attention = nn.Sequential(
            nn.Linear(self.shared_hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states=None,
        encoder_hidden_states_2=None,
        image_rotary_emb=None,
        attention_kwargs=None,
        motion_history=None,
        motion_history_lengths=None,
        return_dict=True,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            attention_kwargs.pop("scale", 1.0)
        _, N, _ = hidden_states.shape
        temb = self.time_embed(timestep).to(hidden_states.dtype)
        temb = self.time_proj(temb)
        temb = temb.unsqueeze(dim=1)
        motion_condition = None
        if self.history_mode == "geometry_motion":
            if motion_history is None or motion_history_lengths is None:
                raise ValueError(
                    "geometry_motion mode requires motion history and lengths"
                )
            motion_condition = self.encode_motion_history(
                motion_history, motion_history_lengths
            )
        hidden_states = self.proj_in(hidden_states)
        hidden_states = torch.cat([temb, hidden_states], dim=1)
        skips = []
        for layer, block in enumerate(self.blocks):
            skip = None if layer <= self.config.num_layers // 2 else skips.pop()
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                ckpt_kwargs = (
                    {"use_reentrant": False} if torch.__version__ >= "1.11.0" else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    encoder_hidden_states_2,
                    temb,
                    image_rotary_emb,
                    skip,
                    attention_kwargs,
                    motion_condition,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_2=encoder_hidden_states_2,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    skip=skip,
                    attention_kwargs=attention_kwargs,
                    motion_condition=motion_condition,
                )
            if layer < self.config.num_layers // 2:
                skips.append(hidden_states)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states[:, -N:]
        shared_features = self.proj_out(hidden_states)
        latent_output = self.latent_head(shared_features)
        param1_output = self.param1_head(shared_features)
        param2_output = self.param2_head(shared_features)
        param3_output = self.param3_head(shared_features)
        param1_attention_weights = F.softmax(self.param1_attention(shared_features), dim=1)
        param1_global = (param1_output * param1_attention_weights).sum(dim=1)
        param2_attention_weights = F.softmax(self.param2_attention(shared_features), dim=1)
        param2_global = (param2_output * param2_attention_weights).sum(dim=1)
        param3_attention_weights = F.softmax(self.param3_attention(shared_features), dim=1)
        param3_global = (param3_output * param3_attention_weights).sum(dim=1)
        motion_type_attention_weights = F.softmax(
            self.motion_type_attention(shared_features), dim=1
        )
        motion_type_global = (
            self.motion_type_head(shared_features) * motion_type_attention_weights
        ).sum(dim=1)
        if not return_dict:
            return (
                latent_output,
                param1_global,
                param2_global,
                param3_global,
                motion_type_global,
            )
        return {
            "latent": latent_output,
            "param1": param1_global,
            "param2": param2_global,
            "param3": param3_global,
            "motion_type": motion_type_global,
        }
