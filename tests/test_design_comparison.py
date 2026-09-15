from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
for module_root in (ROOT, ROOT / "TripoSG"):
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))

from urdf_anything.data.urdf_utils import get_motion_history_from_info
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


if __name__ == "__main__":
    unittest.main()
