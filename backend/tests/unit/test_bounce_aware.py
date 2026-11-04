import json
import unittest
from pathlib import Path

import numpy as np

from app.learning.bounce_aware import (
    bounce_aware,
    boxes_from_tracks,
    compare,
    observed_bounces,
    score,
)
from app.learning.probabilistic import kalman_cv

RECORDED = Path(__file__).resolve().parents[3] / "datasets" / "recorded"
VIDEO_C = RECORDED / "heldout_video_c_calibration_tracks.json"


def box(cx, cy, half=10.0):
    return [cx - half, cy - half, cx + half, cy + half]


def bouncing_ball(floor_bottom=400.0, half=10.0, drops=3, e=0.6):
    """Synthetic ball falling under gravity, bouncing off a floor."""
    y, vy, g, out = 100.0, 0.0, 3.0, []
    floor_centre = floor_bottom - half
    for t in range(140):
        out.append(box(200 + t, y, half))
        vy += g
        y += vy
        if y > floor_centre:
            y = floor_centre - (y - floor_centre) * e
            vy = -vy * e
    return out


class BounceDetectionTests(unittest.TestCase):
    def test_detects_bounces_of_synthetic_ball(self):
        boxes = bouncing_ball()
        self.assertGreaterEqual(len(observed_bounces(boxes)), 3)

    def test_no_bounce_on_straight_line(self):
        boxes = [box(10 + 3 * t, 50 + 2 * t) for t in range(30)]
        self.assertEqual(observed_bounces(boxes), [])


class ForecasterTests(unittest.TestCase):
    def test_plain_variant_matches_kalman_cv(self):
        boxes = bouncing_ball()
        centres = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
        plain = bounce_aware(boxes, reset=False, floor_memory=False)
        reference = kalman_cv(centres, 50.0, 4.0)
        for a, (t, mean, cov) in zip(plain, reference):
            self.assertEqual(a["frame"], t)
            self.assertTrue(np.allclose(a["point"], mean))
            self.assertTrue(np.allclose(a["components"][0][2], cov))

    def test_forecast_ignores_target_frame(self):
        boxes = bouncing_ball()
        changed = boxes[:-1] + [box(999, 999)]
        a = bounce_aware(boxes)[-1]
        b = bounce_aware(changed)[-1]
        self.assertTrue(np.allclose(a["point"], b["point"]))
        self.assertEqual(a["p_bounce"], b["p_bounce"])

    def test_floor_unknown_before_first_bounce(self):
        boxes = bouncing_ball()
        first = observed_bounces(boxes)[0]
        for f in bounce_aware(boxes):
            if f["frame"] <= first + 1:
                self.assertIsNone(f["floor_bottom_px"])

    def test_floor_memory_helps_later_bounces(self):
        boxes = bouncing_ball()
        results = compare(boxes)

        def later(name):
            b = [x for x in results[name]["bounces"] if x["order"] == "later"]
            return sum(x["mean_error_px"] for x in b) / len(b)

        self.assertLess(later("kalman+reset+floor"), later("kalman+reset"))
        self.assertLess(results["kalman+reset+floor"]["mean_nll"],
                        results["kalman"]["mean_nll"])

    def test_mixture_weights_sum_to_one(self):
        for f in bounce_aware(bouncing_ball()):
            self.assertAlmostEqual(sum(w for w, _, _ in f["components"]), 1.0)

    def test_tracks_with_gaps_rejected(self):
        tracks = {"boxes": [{"object_id": "0", "frame_index": i,
                             "bbox_xyxy_px": box(i, i)} for i in (0, 1, 3)]}
        with self.assertRaises(ValueError):
            boxes_from_tracks(tracks)


@unittest.skipUnless(VIDEO_C.exists(), "recorded tracks not available")
class VideoCDevelopmentTests(unittest.TestCase):
    """Development values on Video C (already examined; not validation)."""

    def test_video_c_values(self):
        boxes = boxes_from_tracks(json.loads(VIDEO_C.read_text(encoding="utf-8")))
        self.assertEqual(observed_bounces(boxes), [36, 42])
        results = compare(boxes)
        self.assertAlmostEqual(results["kalman"]["mean_error_px"], 9.40191, places=4)
        self.assertLess(results["kalman+reset+floor"]["mean_error_px"],
                        results["kalman+reset"]["mean_error_px"])


if __name__ == "__main__":
    unittest.main()