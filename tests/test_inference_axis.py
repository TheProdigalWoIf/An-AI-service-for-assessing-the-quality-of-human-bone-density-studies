import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dxa_qc.geometry import build_qc_annotation, spine_axis_metrics
from dxa_qc.inference import SPINE_AXIS_LIMIT_DEG


def bar_image(angle_degrees: float) -> np.ndarray:
    """Синтетический «снимок»: яркая прямая полоса под заданным углом на чёрном фоне."""
    image = np.zeros((400, 300), dtype=np.float32)
    angle_rad = np.deg2rad(angle_degrees)
    for y in range(30, 370):
        x = int(150 + (y - 200) * np.tan(angle_rad))
        image[y, max(0, x - 8):min(300, x + 8)] = 0.9
    return image


def curved_image() -> np.ndarray:
    """Изогнутая «колонка» (сколиоз): верх уходит влево, низ — вправо, хорда вертикальна."""
    image = np.zeros((400, 300), dtype=np.float32)
    for y in range(30, 370):
        x = int(150 + 60 * np.sin(np.pi * (y - 30) / 340))
        image[y, max(0, x - 8):min(300, x + 8)] = 0.9
    return image


class SpineAxisMetricsTests(unittest.TestCase):
    def test_vertical_bar_is_straight(self):
        self.assertLess(abs(spine_axis_metrics(bar_image(0.0))["tilt_deg"]), 1.0)

    def test_tilted_bar_tilt_is_measured(self):
        for tilt in (8.0, -8.0):
            measured = spine_axis_metrics(bar_image(tilt))["tilt_deg"]
            self.assertAlmostEqual(abs(measured), abs(tilt), delta=1.5,
                                   msg=f"tilt={tilt}, measured={measured}")

    def test_curved_column_has_small_tilt_but_large_segment_angle(self):
        """Сколиоз: наклон общей прямой мал, а изгиб (верх vs низ) — большой."""
        metrics = spine_axis_metrics(curved_image())
        self.assertLess(abs(metrics["tilt_deg"]), 2.0, "изогнутая колонка не должна выглядеть наклонённой")
        self.assertGreater(metrics["segment_angle_deg"], 8.0, "изгиб должен быть виден в угле между участками")

    def test_angle_same_as_overlay(self):
        image = bar_image(7.0)
        self.assertAlmostEqual(
            spine_axis_metrics(image)["tilt_deg"],
            build_qc_annotation(image, "lumbar_spine")["angle_degrees"],
            delta=0.01,
            msg="угол решения должен совпадать с углом на превью",
        )

    def test_limit_matches_methodology(self):
        self.assertEqual(SPINE_AXIS_LIMIT_DEG, 5.0)


if __name__ == "__main__":
    unittest.main()
