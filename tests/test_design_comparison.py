from pathlib import Path
import sys
import tempfile
import unittest

import torch
import trimesh


ROOT = Path(__file__).resolve().parents[1]
for module_root in (ROOT, ROOT / "TripoSG"):
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))

from urdf_anything.data import CachedSetDataset, load_cache_data
from urdf_anything.data.dataset import get_spatial_part_orders
from urdf_anything.data.urdf_utils import get_motion_history_from_info
from urdf_anything.inference.runner import select_and_order_part_slots
from urdf_anything.model.dit_triposg import DiTBlock


class MotionHistoryDesignTest(unittest.TestCase):
    def setUp(self):
        self.info = {
            "links": [
                {"name": "body"},
                {
                    "name": "door_1",
                    "origin_xyz": "0.1 0.2 0.3",
                    "axis_xyz": "2 0 0",
                    "lower": 0.0,
                    "upper": 1.57,
                    "motion_type": "revolute",
                },
                {
                    "name": "door_2",
                    "origin_xyz": "0.4 0.5 0.6",
                    "axis_xyz": "0 -3 0",
                    "lower": -0.25,
                    "upper": 0.5,
                    "motion_type": "prismatic",
                },
            ]
        }

    def test_history_contains_only_previous_movable_parts(self):
        self.assertEqual(get_motion_history_from_info(self.info, 0).shape, (0, 10))
        self.assertEqual(get_motion_history_from_info(self.info, 1).shape, (0, 10))

        third_part_history = get_motion_history_from_info(self.info, 2)
        self.assertEqual(third_part_history.shape, (1, 10))
        torch.testing.assert_close(
            third_part_history[0, 3:6], torch.tensor([1.0, 0.0, 0.0])
        )
        torch.testing.assert_close(
            third_part_history[0, 6:8], torch.tanh(torch.tensor([0.0, 1.57]))
        )
        torch.testing.assert_close(
            third_part_history[0, 8:10], torch.tensor([1.0, 0.0])
        )

        eot_history = get_motion_history_from_info(self.info, 3)
        self.assertEqual(eot_history.shape, (2, 10))
        torch.testing.assert_close(
            eot_history[1, 3:6], torch.tensor([0.0, -1.0, 0.0])
        )
        torch.testing.assert_close(
            eot_history[1, 8:10], torch.tensor([0.0, 1.0])
        )

    def test_motion_adaln_starts_as_identity_modulation(self):
        block = DiTBlock(
            dim=8,
            num_attention_heads=2,
            cross_attention_dim=4,
            cross_attention_norm_type=None,
            cross_attention_2_dim=4,
            cross_attention_2_norm_type=None,
        )
        block.enable_motion_adaln(condition_dim=3)
        condition = torch.randn(2, 3)
        scale, shift = block.motion_adaln(condition).chunk(2, dim=-1)
        torch.testing.assert_close(scale, torch.zeros_like(scale))
        torch.testing.assert_close(shift, torch.zeros_like(shift))


class SetDenoisingDesignTest(unittest.TestCase):
    def test_part_orders_follow_z_x_y_aabb_minimum(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            links = [{"name": "root", "obj": "root.obj"}]
            trimesh.creation.box().export(root / "root.obj")
            translations = ((0.0, 0.0, 2.0), (3.0, 0.0, 1.0), (2.0, 4.0, 1.0))
            for index, translation in enumerate(translations, start=1):
                mesh = trimesh.creation.box()
                mesh.apply_translation(translation)
                obj_name = f"part_{index}.obj"
                mesh.export(root / obj_name)
                links.append({"name": f"part_{index}", "obj": obj_name})
            orders = get_spatial_part_orders(root, {"links": links})
            self.assertEqual(orders, {0: 0, 3: 1, 2: 2, 1: 3})

    def test_presence_fallback_and_unique_order_assignment(self):
        presence_logits = torch.tensor([-4.0, -3.0, -2.0])
        order_logits = torch.zeros(3, 3)
        self.assertEqual(
            select_and_order_part_slots(presence_logits, order_logits, 0.5),
            [2],
        )

        presence_logits = torch.tensor([4.0, 4.0, -4.0])
        order_logits = torch.tensor(
            [[5.0, 1.0, 0.0], [4.0, 3.0, 0.0]]
        )
        ordered = select_and_order_part_slots(
            presence_logits, order_logits, 0.5
        )
        self.assertEqual(ordered, [0, 1])
        self.assertEqual(len(set(ordered)), 2)

    def test_microwave_smoke_cache_forms_one_five_slot_sample(self):
        cache_path = Path(
            "/data2/LiuShuqi/output/URDF-Anything-plus/design-comparison/"
            "cache/microwave_7201_token512"
        )
        if not cache_path.is_dir():
            self.skipTest("design-comparison smoke cache is unavailable")
        data_index, metadata = load_cache_data(cache_path, split="test")
        dataset = CachedSetDataset(
            data_index,
            {metadata["cache_name"]: metadata["data_root"]},
            split="test",
            num_part_slots=5,
        )
        sample = dataset[0]
        self.assertEqual(sample["target_labels"].shape, (5, 512, 64))
        self.assertEqual(int(sample["presence"].sum().item()), 3)
        self.assertEqual(
            sorted(sample["part_order_targets"][sample["presence"].bool()].tolist()),
            [0, 1, 2],
        )
        self.assertTrue(
            torch.count_nonzero(
                sample["target_labels"][~sample["presence"].bool()]
            ).item()
            == 0
        )


if __name__ == "__main__":
    unittest.main()
