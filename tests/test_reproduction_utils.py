import tempfile
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import torch
import trimesh


ROOT = Path(__file__).resolve().parents[1]
for module_root in (ROOT, ROOT / "TripoSG"):
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))

from scripts.inference_partnet import save_prediction, validate_urdf
from urdf_anything.inference.urdf_io import construct_urdf


class ReproductionUtilitiesTest(unittest.TestCase):
    def test_partnet_obj_loads_as_mesh_for_inference(self):
        source = Path(
            "/data2/LiuShuqi/data/processed/PartNetMobility_URDF-Anything-plus/"
            "test/12252/whole.obj"
        )
        if not source.is_file():
            self.skipTest("prepared PartNet smoke input is unavailable")
        mesh = trimesh.load(source, force="mesh")
        self.assertIsInstance(mesh, trimesh.Trimesh)
        self.assertFalse(mesh.is_empty)

    def test_urdf_preserves_normalized_continuous_axis(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trimesh.creation.box().export(root / "link_0.obj")
            trimesh.creation.box().export(root / "link_1.obj")
            urdf_path = root / "generated.urdf"
            construct_urdf(0, "link_0", None, None, "link_0.obj", urdf_path)
            axis = np.array([-0.2, -0.9, 0.3], dtype=np.float64)
            construct_urdf(
                1,
                "link_1",
                np.array([0.1, 0.2, 0.3]),
                axis,
                "link_1.obj",
                urdf_path,
                [0.5, -0.5],
                "revolute",
            )
            joint = ET.parse(urdf_path).find(".//joint[@name='joint_1']")
            saved = np.fromstring(joint.find("axis").attrib["xyz"], sep=" ")
            np.testing.assert_allclose(saved, axis / np.linalg.norm(axis), atol=1e-6)
            limits = joint.find("limit").attrib
            self.assertEqual((limits["lower"], limits["upper"]), ("-0.500000", "0.500000"))
            self.assertEqual(validate_urdf(root)["links_including_base"], 3)

    def test_particulate_export_shapes_and_scale(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            meshes = [trimesh.creation.box(), trimesh.creation.icosphere(subdivisions=1)]
            items = []
            for index, mesh in enumerate(meshes):
                mesh.export(root / f"link_{index}.obj")
                items.append(
                    {
                        "pred_mesh": mesh,
                        "param1": torch.tensor([0.1, 0.2, 0.3]),
                        "param2": torch.tensor([-0.2, -0.9, 0.3]),
                        "param3": torch.tensor([1.0, -1.0]),
                        "motion_type": torch.tensor([1.0, 0.0]),
                        "eot_mse": 1.0,
                    }
                )
            save_prediction({"results": items}, root)
            exported = np.load(root / "eval/pred.npz")
            self.assertEqual(exported["motion_hierarchy"].shape, (1, 2))
            self.assertEqual(exported["revolute_plucker"].shape, (2, 6))
            np.testing.assert_allclose(exported["revolute_range"][1], [-1.0, 1.0])
            self.assertEqual(len(exported["face_part_ids"]), sum(len(m.faces) for m in meshes))
            pred_mesh = trimesh.load(root / "eval/pred.obj", force="mesh", process=False)
            self.assertAlmostEqual(float(pred_mesh.extents.max()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
