import os
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from transformers import AutoImageProcessor, AutoModel

from urdf_anything.model import URDFModel
from urdf_anything.data.urdf_utils import build_motion_cue
from TripoSG.triposg.inference_utils import hierarchical_extract_geometry

from .utils import resize_image_tensor, ensure_watertight
from .urdf_io import construct_urdf

class URDFInference:
    def __init__(
        self,
        model_path: str,
        model_config_path: Optional[str] = None,
        dino_path: str = "dinov3",
        device: str = "cuda",
        num_tokens: int = 512,
        eot_threshold: float = 0.5,
        max_reconstruction_attempts: int = 5,
        overlap_chamfer_threshold: float = 1e-4,
        max_links: int = 16,
        seed: int = 42,
        vae_deterministic: bool = False,
    ):
        self.device = device
        self.num_tokens = num_tokens
        self.eot_threshold = eot_threshold
        self.max_reconstruction_attempts = max_reconstruction_attempts
        self.overlap_chamfer_threshold = overlap_chamfer_threshold
        if max_links < 1:
            raise ValueError("max_links must be at least 1")
        self.max_links = int(max_links)
        self.seed = int(seed)
        self.vae_deterministic = bool(vae_deterministic)

        print(f"loading model: {model_path}")
        self.model = URDFModel.from_checkpoint(
            checkpoint_path=model_path,
            config_path=model_config_path,
            device=device,
            init_mode="inference",
        )
        self.model.eval()
        self.model = self.model.to(device)
        self.history_mode = self.model.config["ditrunner"].get(
            "history_mode", "geometry"
        )

        print(f"loading DINO model: {dino_path}")
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_path)
        self.dino_model = AutoModel.from_pretrained(dino_path).to(device)
        for param in self.dino_model.parameters():
            param.requires_grad = False

        self.eot_token = torch.zeros_like(self.model.SoT).to(device)
        print("model loaded")

    def process_image(self, image_path: str) -> torch.Tensor:
        """Process image and return DINO features."""
        whole_image = (
            torch.tensor(np.array(Image.open(image_path).convert("RGB"))).float()
            / 255.0
        )
        whole_image = resize_image_tensor(whole_image, target_size=(512, 512))
        with torch.no_grad():
            proc_whole = self.dino_processor(
                images=whole_image.unsqueeze(0), return_tensors="pt"
            )
            proc_whole = {k: v.to(self.device) for k, v in proc_whole.items()}
            out_whole = self.dino_model(**proc_whole).last_hidden_state[:, 1:, :]
            out_whole = out_whole.to(self.device)
            dino_features = self.model.dino_adapter(out_whole)
        return dino_features

    def mesh_to_encode_whole(
        self, mesh: trimesh.Trimesh, num_pc: int = 204800
    ) -> torch.Tensor:
        if isinstance(mesh, trimesh.Scene):
            geometries = list(mesh.geometry.values())
            if not geometries:
                raise ValueError("input scene contains no mesh geometry")
            mesh = trimesh.util.concatenate(geometries)
        if not isinstance(mesh, trimesh.Trimesh):
            if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
                vertices = (
                    mesh.vertices.cpu().numpy()
                    if isinstance(mesh.vertices, torch.Tensor)
                    else mesh.vertices
                )
                faces = (
                    mesh.faces.cpu().numpy()
                    if isinstance(mesh.faces, torch.Tensor)
                    else mesh.faces
                )
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            else:
                raise ValueError(
                    f"cannot convert to trimesh.Trimesh: {type(mesh)}"
                )
        # Trimesh surface sampling uses NumPy RNG under the hood.
        # Seed it to keep point sampling reproducible.
        np.random.seed(self.seed + 101)
        surface_points, face_indices = mesh.sample(num_pc, return_index=True)
        face_normals = mesh.face_normals[face_indices]
        points = torch.cat(
            [
                torch.FloatTensor(surface_points),
                torch.FloatTensor(face_normals),
            ],
            dim=-1,
        ).unsqueeze(0).to(self.device)
        with torch.no_grad():
            encode_output = self.model.model_3d.encode(
                points,
                num_tokens=self.num_tokens,
                return_dict=False,
                seed=self.seed + 201,
            )
            if isinstance(encode_output, tuple):
                posterior = encode_output[0]
            else:
                posterior = encode_output.latent_dist

            if self.vae_deterministic:
                gen = torch.Generator(device=self.device)
                gen.manual_seed(self.seed + 301)
                encode_whole = posterior.sample(generator=gen)
            else:
                encode_whole = posterior.sample()
        return encode_whole

    def encode_prev_meshes(
        self, prev_meshes: List[trimesh.Trimesh]
    ) -> torch.Tensor:
        """Encode previous link meshes to encode_pre."""
        if len(prev_meshes) == 0:
            return self.model.SoT
        if len(prev_meshes) == 1:
            combined_mesh = prev_meshes[0]
        else:
            combined_mesh = trimesh.util.concatenate(prev_meshes)
        np.random.seed(self.seed + 1000 + len(prev_meshes))
        surface_points, face_indices = combined_mesh.sample(
            204800, return_index=True
        )
        point_normals = combined_mesh.face_normals[face_indices]
        prev_points = (
            torch.cat(
                [
                    torch.FloatTensor(surface_points),
                    torch.FloatTensor(point_normals),
                ],
                dim=-1,
            )
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            encode_pre_output = self.model.model_3d.encode(
                prev_points,
                num_tokens=self.num_tokens,
                return_dict=False,
                seed=self.seed + 1000 + len(prev_meshes),
            )
            if isinstance(encode_pre_output, tuple):
                posterior = encode_pre_output[0]
            else:
                posterior = encode_pre_output.latent_dist

            if self.vae_deterministic:
                gen = torch.Generator(device=self.device)
                gen.manual_seed(self.seed + 2000 + len(prev_meshes))
                encode_pre = posterior.sample(generator=gen)
            else:
                encode_pre = posterior.sample()
        return encode_pre

    def chamfer_distance_between_meshes(
        self,
        mesh_a: trimesh.Trimesh,
        mesh_b: trimesh.Trimesh,
        n_points: int = 10000,
    ) -> float:
        points_a = np.asarray(mesh_a.sample(n_points))
        points_b = np.asarray(mesh_b.sample(n_points))
        tree_a = cKDTree(points_a)
        tree_b = cKDTree(points_b)
        dist_a_to_b, _ = tree_b.query(points_a)
        dist_b_to_a, _ = tree_a.query(points_b)
        return float(
            np.mean(dist_a_to_b**2) + np.mean(dist_b_to_a**2)
        )

    def generate_link(
        self,
        dino_features: torch.Tensor,
        encode_pre: torch.Tensor,
        encode_whole: torch.Tensor,
        link_idx: int,
        generator: Optional[torch.Generator] = None,
        motion_history: Optional[List[torch.Tensor]] = None,
    ) -> Dict:
        """Generate one link: DiT sample -> mesh (or EoT)."""
        cond = {
            "dino": dino_features,
            "encode_pre": encode_pre,
            "encode_whole": encode_whole,
        }
        if self.history_mode == "geometry_motion":
            motion_history = motion_history or []
            if motion_history:
                history_tensor = torch.stack(motion_history).unsqueeze(0)
            else:
                history_tensor = torch.empty(1, 0, 10)
            cond["motion_history"] = history_tensor.to(self.device)
            cond["motion_history_lengths"] = torch.tensor(
                [len(motion_history)], dtype=torch.long, device=self.device
            )
        with torch.no_grad():
            output = self.model.DiTRunner.conditional_sample(
                cond, generator=generator
            )
            predicted_latent = output["latent"]
            param1_pred = output["param1"]
            param2_pred = output["param2"]
            param3_pred = output.get("param3", None)
            motion_type_pred = output.get("motion_type", None)
        mse = F.mse_loss(predicted_latent, self.eot_token).item()
        is_eot = mse < self.eot_threshold
        pred_mesh = None
        if not is_eot:
            predicted_latent_batch = (
                predicted_latent.unsqueeze(0)
                if predicted_latent.dim() == 2
                else predicted_latent
            )
            geometric_func = lambda x, latent=predicted_latent_batch: self.model.model_3d.decode(
                latent, sampled_points=x
            ).sample
            output_geom = hierarchical_extract_geometry(
                geometric_func,
                device=self.device,
                bounds=(-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
                dense_octree_depth=7,
                hierarchical_octree_depth=8,
            )
            meshes = [
                trimesh.Trimesh(mesh_v_f[0].astype(np.float32), mesh_v_f[1])
                for mesh_v_f in output_geom
                if mesh_v_f is not None and mesh_v_f[0] is not None
            ]
            if meshes:
                pred_mesh = ensure_watertight(meshes[0])
        return {
            "pred_mesh": pred_mesh,
            "predicted_latent": predicted_latent,
            "param1": param1_pred.cpu()[0] if param1_pred is not None else None,
            "param2": param2_pred.cpu()[0] if param2_pred is not None else None,
            "param3": param3_pred.cpu()[0] if param3_pred is not None else None,
            "motion_type": (
                motion_type_pred.cpu()[0]
                if motion_type_pred is not None
                else None
            ),
            "is_eot": is_eot,
            "eot_mse": mse,
        }

    def reconstruct_link_with_retry(
        self,
        dino_features: torch.Tensor,
        encode_pre: torch.Tensor,
        encode_whole: torch.Tensor,
        link_idx: int,
        motion_history: Optional[List[torch.Tensor]] = None,
    ) -> Optional[Dict]:
        """Generate link with retries."""
        for attempt in range(self.max_reconstruction_attempts):
            try:
                # Make sampling reproducible even with retries.
                # Different links/attempts use different seeds deterministically.
                attempt_seed = self.seed + link_idx * 1000 + attempt
                gen = torch.Generator(device=self.device)
                gen.manual_seed(attempt_seed)
                result = self.generate_link(
                    dino_features,
                    encode_pre,
                    encode_whole,
                    link_idx,
                    generator=gen,
                    motion_history=motion_history,
                )
                if result["pred_mesh"] is not None or result["is_eot"]:
                    if attempt > 0:
                        print(
                            f"  Link {link_idx} successfully generated after {attempt + 1} attempts"
                        )
                    return result
                print(
                    f"  Link {link_idx} failed after {attempt + 1} attempts, retrying..."
                )
            except Exception as e:
                print(
                    f"  Link {link_idx} failed after {attempt + 1} attempts: {e}"
                )
                if attempt == self.max_reconstruction_attempts - 1:
                    print(
                        f"  Link {link_idx} reached maximum attempts, skipping"
                    )
                    return None
        return None

    def infer_from_image(
        self,
        image_path: str,
        whole_mesh: Optional[trimesh.Trimesh] = None,
        whole_mesh_path: Optional[str] = None,
        output_dir: str = "output",
        save_meshes: bool = True,
        save_urdf: bool = True,
    ) -> Dict:
        os.makedirs(output_dir, exist_ok=True)
        print(f"processing image: {image_path}")
        dino_features = self.process_image(image_path)

        if whole_mesh is not None:
            print("using provided whole mesh")
            mesh = whole_mesh
        elif whole_mesh_path is not None:
            print(f"loading whole mesh from file: {whole_mesh_path}")
            mesh = trimesh.load(whole_mesh_path, force="mesh")
        else:
            raise ValueError("must provide whole_mesh or whole_mesh_path")

        print("encoding whole mesh...")
        encode_whole = self.mesh_to_encode_whole(mesh)
        print("\ngenerating links...")
        link_idx = 0
        encode_pre = self.model.SoT
        prev_meshes = []
        all_results = []
        motion_history = []

        while link_idx < self.max_links:
            print(f"\ngenerating Link {link_idx}...")
            result = self.reconstruct_link_with_retry(
                dino_features,
                encode_pre,
                encode_whole,
                link_idx,
                motion_history=motion_history,
            )
            if result is None:
                print(f"Link {link_idx} failed, stopping generation")
                break
            if result["is_eot"]:
                print(
                    f"detected EoT (MSE: {result['eot_mse']:.6f}), stopping generation"
                )
                break
            if result["pred_mesh"] is not None:
                if prev_meshes:
                    cd_prev = self.chamfer_distance_between_meshes(
                        prev_meshes[-1], result["pred_mesh"]
                    )
                    print(
                        f"  Chamfer Distance with previous link geometry: {cd_prev:.6f}"
                    )
                    if cd_prev < self.overlap_chamfer_threshold:
                        print(
                            f"  current link geometry overlaps (CD={cd_prev:.6f}), stopping"
                        )
                        break
                prev_meshes.append(result["pred_mesh"])
                if save_meshes:
                    mesh_path = os.path.join(
                        output_dir, f"link_{link_idx}.obj"
                    )
                    result["pred_mesh"].export(mesh_path)
                    print(f"  saving mesh: {mesh_path}")
                if link_idx > 0:
                    if result["param1"] is not None:
                        print(f"  Origin: {result['param1'].numpy()}")
                    if result["param2"] is not None:
                        print(f"  Axis: {result['param2'].numpy()}")
                    if result["param3"] is not None:
                        print(f"  Limits: {result['param3'].numpy()}")
                    if result["motion_type"] is not None:
                        motion_type_class = torch.argmax(
                            result["motion_type"]
                        ).item()
                        motion_name = (
                            "revolute"
                            if motion_type_class == 0
                            else "prismatic"
                        )
                        print(f"  Motion Type: {motion_name}")
                    if all(
                        result[key] is not None
                        for key in ("param1", "param2", "param3", "motion_type")
                    ):
                        motion_history.append(
                            build_motion_cue(
                                result["param1"],
                                result["param2"],
                                result["param3"],
                                torch.argmax(result["motion_type"]),
                            )
                        )
            all_results.append({"link_idx": link_idx, **result})
            link_idx += 1
            if link_idx < self.max_links:
                encode_pre = self.encode_prev_meshes(prev_meshes)

        if link_idx >= self.max_links:
            print(f"reached max_links={self.max_links}, stopping generation")

        if save_urdf and len(prev_meshes) > 0:
            urdf_path = os.path.join(output_dir, "generated.urdf")
            print(f"\nGenerating URDF: {urdf_path}")
            for i, result in enumerate(all_results):
                if result["pred_mesh"] is not None:
                    link_name = f"link_{i}"
                    origin_xyz = (
                        result["param1"].numpy()
                        if result["param1"] is not None
                        else np.array([0, 0, 0])
                    )
                    axis_xyz = (
                        result["param2"].numpy()
                        if result["param2"] is not None
                        else np.array([0, 0, 1])
                    )
                    mesh_path = f"link_{i}.obj"
                    lower_upper_limits = None
                    if result["param3"] is not None:
                        lower_upper_limits = result["param3"].numpy().tolist()
                    motion_type = result["motion_type"]
                    construct_urdf(
                        link_idx=i,
                        link_name=link_name,
                        origin_xyz=origin_xyz,
                        axis_xyz=axis_xyz,
                        obj_path=mesh_path,
                        urdf_path=urdf_path,
                        lower_upper_limits=lower_upper_limits,
                        motion_type=motion_type,
                    )
        return {
            "num_links": len(prev_meshes),
            "results": all_results,
            "output_dir": output_dir,
        }
