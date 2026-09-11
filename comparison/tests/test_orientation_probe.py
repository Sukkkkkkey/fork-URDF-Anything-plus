import unittest

import numpy as np
import trimesh

from comparison.prepare_orientation_probe import PARTNET_TO_MODEL, transform_mesh


class OrientationProbeTests(unittest.TestCase):
    def test_maps_partnet_up_and_front_to_model_convention(self):
        np.testing.assert_allclose(PARTNET_TO_MODEL @ [0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(PARTNET_TO_MODEL @ [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(np.linalg.det(PARTNET_TO_MODEL), 1.0)

    def test_mesh_transform_preserves_extent_values(self):
        mesh = trimesh.creation.box(extents=[2.0, 3.0, 4.0])
        transformed = transform_mesh(mesh)
        np.testing.assert_allclose(transformed.extents, [3.0, 4.0, 2.0])


if __name__ == "__main__":
    unittest.main()
