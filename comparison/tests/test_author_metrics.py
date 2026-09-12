import math
import unittest

import numpy as np

from comparison.evaluate_author_metrics import (
    PROTOCOL_VERSION,
    axis_error,
    cache_is_compatible,
    evaluate_part_geometry,
    face_majority_labels,
    fill_volume_voxels,
    limit_error,
    line_distance,
    plucker_axis_point,
    point_metrics,
    transform_motion_to_world,
    voxel_iou,
)


class AuthorMetricTests(unittest.TestCase):
    def test_cache_requires_current_protocol_sample_and_method(self):
        result = {
            "protocol_version": PROTOCOL_VERSION,
            "sample_id": "12252",
            "method": "urdf-anything-plus",
        }
        self.assertTrue(
            cache_is_compatible(result, "12252", "urdf-anything-plus")
        )
        self.assertFalse(cache_is_compatible(result, "12252", "particulate"))
        result["protocol_version"] = "comparison-local-v3-volume-iou"
        self.assertFalse(
            cache_is_compatible(result, "12252", "urdf-anything-plus")
        )

    def test_plucker_point_is_closest_axis_point(self):
        direction, point = plucker_axis_point([0, 0, 2, 0, 2, 0])
        np.testing.assert_allclose(direction, [0, 0, 1])
        np.testing.assert_allclose(point, [1, 0, 0])

    def test_axis_and_limit_are_sign_invariant(self):
        self.assertAlmostEqual(axis_error([1, 0, 0], [-1, 0, 0]), 0.0)
        self.assertAlmostEqual(
            limit_error([1, 0, 0], [-1, 2], [-1, 0, 0], [-2, 1]), 0.0
        )

    def test_axis_error_is_unsigned_right_angle(self):
        self.assertAlmostEqual(axis_error([1, 0, 0], [0, 1, 0]), math.pi / 2)

    def test_line_distance_parallel_and_skew(self):
        self.assertAlmostEqual(
            line_distance([0, 0, 0], [0, 0, 1], [2, 0, 4], [0, 0, 1]),
            2.0,
        )
        self.assertAlmostEqual(
            line_distance([0, 0, 0], [1, 0, 0], [0, 0, 3], [0, 1, 0]),
            3.0,
        )

    def test_identical_point_metrics(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        result = point_metrics(points, points, threshold=0.02)
        self.assertEqual(result["fscore"], 1.0)
        self.assertEqual(result["cd"], 0.0)

    def test_voxel_iou(self):
        self.assertAlmostEqual(voxel_iou(np.array([1, 2]), np.array([2, 3])), 1 / 3)

    def test_part_geometry_emits_both_iou_variants(self):
        points = {0: np.array([[0.0, 0.0, 0.0]])}
        result, pairs = evaluate_part_geometry(
            points,
            points,
            {0: np.array([1, 2])},
            {0: np.array([2, 3])},
            {0: np.array([1, 2, 3])},
            {0: np.array([2, 3, 4])},
            [(0, 0)],
            threshold=0.02,
            missing_cd_penalty=12.0,
            workers=1,
        )
        self.assertAlmostEqual(result["matched_only"]["iou_surface"], 1 / 3)
        self.assertAlmostEqual(result["matched_only"]["iou_volume"], 1 / 2)
        self.assertAlmostEqual(pairs[0]["iou_surface"], 1 / 3)
        self.assertAlmostEqual(pairs[0]["iou_volume"], 1 / 2)

    def test_volume_fill_fills_closed_voxel_shell(self):
        resolution = 5
        grid = np.zeros((resolution,) * 3, dtype=bool)
        grid[1:4, 1:4, 1:4] = True
        grid[2, 2, 2] = False
        surface = np.flatnonzero(grid)
        filled = fill_volume_voxels(surface, resolution)
        self.assertEqual(len(filled), 27)
        self.assertIn(np.ravel_multi_index((2, 2, 2), grid.shape), filled)

    def test_volume_fill_does_not_fill_open_sheet(self):
        resolution = 5
        grid = np.zeros((resolution,) * 3, dtype=bool)
        grid[2, 1:4, 1:4] = True
        filled = fill_volume_voxels(np.flatnonzero(grid), resolution)
        np.testing.assert_array_equal(filled, np.flatnonzero(grid))

    def test_face_majority(self):
        vertices = np.array([0, 0, 1, 2])
        faces = np.array([[0, 1, 2], [1, 2, 3]])
        np.testing.assert_array_equal(face_majority_labels(vertices, faces), [0, 0])

    def test_world_transform_preserves_and_moves_axis_line(self):
        direction = np.array([1.0, 0.0, 0.0])
        point = np.array([0.0, 2.0, 3.0])
        motion = {
            "revolute_plucker": np.array(
                [np.concatenate([direction, np.cross(direction, point)])]
            ),
            "prismatic_axis": np.array([[1.0, 0.0, 0.0]]),
        }
        transform = np.array(
            [
                [0.0, -1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, -2.0],
                [0.0, 0.0, 1.0, 0.5],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        result = transform_motion_to_world(motion, transform)
        result_axis, result_point = plucker_axis_point(
            result["revolute_plucker"][0]
        )
        expected_axis = transform[:3, :3] @ direction
        expected_point = transform[:3, :3] @ point + transform[:3, 3]
        np.testing.assert_allclose(result_axis, expected_axis)
        np.testing.assert_allclose(result["prismatic_axis"][0], expected_axis)
        self.assertAlmostEqual(
            line_distance(result_point, result_axis, expected_point, expected_axis),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
