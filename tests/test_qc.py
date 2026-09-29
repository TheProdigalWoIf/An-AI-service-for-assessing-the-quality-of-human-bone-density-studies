import unittest

import numpy as np

from dxa_qc.qc import analyze_pixels, classify_region


class QCTests(unittest.TestCase):
    def test_spine_shape(self):
        image = np.zeros((300, 280), dtype=np.float32)
        image[20:285, 118:162] = 1
        self.assertEqual(classify_region(image), "lumbar_spine")

    def test_femur_shape(self):
        image = np.zeros((300, 280), dtype=np.float32)
        for y in range(20, 285):
            x = min(250, 30 + y // 2)
            image[y, x - 10:x + 10] = 1
        self.assertEqual(classify_region(image), "proximal_femur")

    def test_non_dxa(self):
        image = np.tile(np.linspace(.2, .8, 280), (300, 1)).astype(np.float32)
        result = analyze_pixels(image, "p", "s", "i")
        self.assertEqual(result.violation_type, "not_dxa_image")
        self.assertEqual(result.quality_class, 1)
        self.assertEqual(result.decision_status, "needs_review")

    def test_spine_result_is_final_without_side(self):
        image = np.zeros((300, 280), dtype=np.float32)
        image[20:285, 118:162] = 1
        result = analyze_pixels(image, "p", "s", "i")
        self.assertEqual(result.anatomical_region, "lumbar_spine")
        self.assertEqual(result.side, "none")
        self.assertEqual(result.decision_status, "final")

    def test_femur_side_is_not_invented(self):
        """Правила не определяют сторону бедра — только needs_review."""
        image = np.zeros((300, 280), dtype=np.float32)
        for y in range(20, 285):
            x = min(250, 30 + y // 2)
            image[y, x - 10:x + 10] = 1
        result = analyze_pixels(image, "p", "s", "i")
        self.assertEqual(result.anatomical_region, "proximal_femur")
        self.assertEqual(result.side, "none")
        self.assertEqual(result.decision_status, "needs_review")


if __name__ == "__main__":
    unittest.main()
