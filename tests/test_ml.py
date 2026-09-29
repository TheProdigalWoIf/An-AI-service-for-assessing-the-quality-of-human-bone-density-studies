import unittest

import numpy as np
import torch

from dxa_qc.ml import (
    REGIONS,
    SIDES,
    VIOLATIONS,
    DXAMultiTaskModel,
    allowed_violations,
    violation_vector,
)


class ViolationsTests(unittest.TestCase):
    def test_five_expert_criteria(self):
        self.assertEqual(len(VIOLATIONS), 5)

    def test_allowed_violations_by_region(self):
        spine = allowed_violations("lumbar_spine")
        hip = allowed_violations("proximal_femur")
        self.assertEqual(spine, {
            "spine_positioning_incorrect", "spine_axis_misalignment", "spine_artifact_or_object",
        })
        self.assertEqual(hip, {"hip_positioning_incorrect", "hip_roi_incorrect"})
        self.assertFalse(spine & hip)
        self.assertEqual(allowed_violations("unknown"), set())

    def test_violation_vector(self):
        vector = violation_vector("spine_axis_misalignment;hip_roi_incorrect")
        self.assertEqual(vector.tolist(), [0.0, 1.0, 0.0, 0.0, 1.0])
        self.assertEqual(violation_vector("none").tolist(), [0.0] * 5)
        self.assertEqual(violation_vector("").tolist(), [0.0] * 5)
        with self.assertRaises(ValueError):
            violation_vector("low_contrast")  # код из старой таксономии отвергается


class ModelTests(unittest.TestCase):
    def test_forward_shapes(self):
        model = DXAMultiTaskModel(pretrained=False).eval()
        with torch.inference_mode():
            region, side, quality, violation = model(torch.zeros(2, 3, 224, 224))
        self.assertEqual(region.shape, (2, len(REGIONS)))
        self.assertEqual(side.shape, (2, len(SIDES)))
        self.assertEqual(quality.shape, (2,))
        self.assertEqual(violation.shape, (2, len(VIOLATIONS)))


if __name__ == "__main__":
    unittest.main()
