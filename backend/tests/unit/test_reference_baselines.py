import json
import unittest
from pathlib import Path

from app.learning.reference_baselines import (
    alpha_beta,
    constant_velocity_2pt,
    evaluate_baselines,
    track_centres,
)

RECORDED = Path(__file__).resolve().parents[3] / "datasets" / "recorded"
VIDEO_C = RECORDED / "heldout_video_c_calibration_tracks.json"
VIDEO_D = RECORDED / "heldout_video_d_calibration_tracks.json"


def tracks(points):
    return {"model": "sam-3.1", "boxes": [
        {"frame_index": i, "object_id": 0,
         "bbox_xyxy_px": [x - 5, y - 5, x + 5, y + 5]}
        for i, (x, y) in enumerate(points)]}


class SyntheticBaselineTests(unittest.TestCase):
    def test_constant_velocity_is_exact_for_both(self):
        pts = [(10 + 3 * t, 50 - 2 * t) for t in range(20)]
        result = evaluate_baselines(tracks(pts))
        self.assertAlmostEqual(
            result["constant_velocity_2pt"]["mean_error_px"], 0.0)
        self.assertAlmostEqual(result["alpha_beta"]["mean_error_px"], 0.0)

    def test_targets_start_at_frame_two(self):
        pts = [(t, t) for t in range(10)]
        self.assertEqual([f for f, _ in constant_velocity_2pt(pts)],
                         list(range(2, 10)))
        self.assertEqual([f for f, _ in alpha_beta(pts)], list(range(2, 10)))

    def test_forecast_ignores_target_observation(self):
        pts = [(t, 0.0) for t in range(10)]
        changed = pts[:9] + [(999.0, 999.0)]
        self.assertEqual(alpha_beta(pts)[-1], alpha_beta(changed)[-1])
        self.assertEqual(constant_velocity_2pt(pts)[-1],
                         constant_velocity_2pt(changed)[-1])

    def test_rejects_frame_gap(self):
        data = tracks([(0, 0), (1, 1), (2, 2)])
        data["boxes"][2]["frame_index"] = 3
        with self.assertRaises(ValueError):
            track_centres(data)

    def test_rejects_multiple_objects(self):
        data = tracks([(0, 0), (1, 1), (2, 2)])
        data["boxes"][1]["object_id"] = 1
        with self.assertRaises(ValueError):
            track_centres(data)


@unittest.skipUnless(VIDEO_C.exists() and VIDEO_D.exists(),
                     "recorded tracks not available")
class ExploratoryValueTests(unittest.TestCase):
    """Values first seen in exploratory (post-hoc) review of Video C/D."""

    def test_video_c(self):
        r = evaluate_baselines(json.loads(VIDEO_C.read_text(encoding="utf-8")))
        self.assertEqual(r["constant_velocity_2pt"]["evaluated_predictions"], 48)
        self.assertEqual(r["constant_velocity_2pt"]["mean_error_px"], 9.015013)
        self.assertEqual(r["alpha_beta"]["mean_error_px"], 11.513307)

    def test_video_d(self):
        r = evaluate_baselines(json.loads(VIDEO_D.read_text(encoding="utf-8")))
        self.assertEqual(r["constant_velocity_2pt"]["mean_error_px"], 0.010417)
        self.assertEqual(r["alpha_beta"]["mean_error_px"], 0.010417)


if __name__ == "__main__":
    unittest.main()