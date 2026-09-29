import unittest

import numpy as np

from dxa_qc.geometry import build_qc_annotation


class GeometryTests(unittest.TestCase):
    def test_lumbar_annotation_contains_axis_guides_and_angle(self):
        image = np.zeros((220, 160), dtype=np.float32)
        for y in range(20, 200):
            x = 65 + y // 12
            image[y, x - 5:x + 6] = 1.0
        result = build_qc_annotation(image, "lumbar_spine")
        self.assertEqual(result["kind"], "lumbar_geometry")
        self.assertEqual(len(result["guides"]), 4)
        self.assertGreater(abs(result["angle_degrees"]), 1.0)

    def test_hip_annotation_contains_roi(self):
        image = np.zeros((180, 180), dtype=np.float32)
        image[40:150, 55:130] = 1.0
        result = build_qc_annotation(image, "proximal_femur")
        self.assertEqual(result["kind"], "hip_geometry")
        self.assertIsNotNone(result["roi"])


if __name__ == "__main__":
    unittest.main()
