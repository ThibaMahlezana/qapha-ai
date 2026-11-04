import copy
import json
import unittest
from pathlib import Path

from app.pipeline.timeline import MODELS, build_timeline, zone_ellipse

RECORDED = Path(__file__).resolve().parents[3] / "datasets" / "recorded"
VIDEO_C = RECORDED / "heldout_video_c_calibration_tracks.json"


def straight_line(n=12):
    return {"model": "sam-3.1", "concept": "ball", "boxes": [
        {"object_id": "0", "frame_index": i,
         "bbox_xyxy_px": [10 + 4 * i, 20, 20 + 4 * i, 30]}
        for i in range(n)]}


class TimelineTests(unittest.TestCase):
    def test_first_two_frames_have_no_predictions(self):
        t = build_timeline(straight_line())
        self.assertIsNone(t["frames"][0]["predictions"])
        self.assertIsNone(t["frames"][1]["predictions"])
        self.assertEqual(set(t["frames"][2]["predictions"]), set(MODELS))

    def test_constant_velocity_exact_on_straight_line(self):
        t = build_timeline(straight_line())
        self.assertEqual(t["summary"]["constant_velocity_2pt"]["mean_error_px"], 0)

    def test_forecasts_do_not_see_their_target(self):
        a = build_timeline(straight_line())
        changed = straight_line()
        changed["boxes"][-1]["bbox_xyxy_px"] = [500, 500, 510, 510]
        b = build_timeline(changed)
        for m in MODELS:
            self.assertEqual(a["frames"][-1]["predictions"][m]["xy"],
                             b["frames"][-1]["predictions"][m]["xy"])

    def test_adaptive_equals_selected_expert(self):
        t = build_timeline(straight_line())
        for f in t["frames"][2:]:
            chosen = f["adaptive"]["selection"]
            self.assertEqual(f["predictions"]["adaptive"]["xy"],
                             f["predictions"][chosen]["xy"])

    def test_rejects_two_objects(self):
        data = straight_line()
        data["boxes"][3]["object_id"] = "1"
        with self.assertRaises(ValueError):
            build_timeline(data)

    def test_unsorted_boxes_accepted(self):
        data = straight_line()
        shuffled = copy.deepcopy(data)
        shuffled["boxes"].reverse()
        self.assertEqual(build_timeline(data)["summary"],
                         build_timeline(shuffled)["summary"])

    def test_zone_present_and_ellipse_sane(self):
        t = build_timeline(straight_line())
        z = t["frames"][5]["zone"]
        self.assertTrue(z["observed_inside"])
        self.assertGreaterEqual(z["rx_px"], z["ry_px"])
        self.assertGreater(z["ry_px"], 0)
        self.assertEqual(t["zone_summary"]["coverage"]["90"], 1.0)

    def test_zone_ellipse_matches_covariance(self):
        e = zone_ellipse([[16.0, 0.0], [0.0, 4.0]], "90")
        self.assertAlmostEqual(e["rx_px"], (4.605170 * 16) ** 0.5, places=3)
        self.assertAlmostEqual(e["ry_px"], (4.605170 * 4) ** 0.5, places=3)
        self.assertAlmostEqual(abs(e["angle_deg"]) % 180, 0.0, places=3)

    @unittest.skipUnless(VIDEO_C.exists(), "recorded tracks not available")
    def test_matches_registered_video_c_values(self):
        t = build_timeline(json.loads(VIDEO_C.read_text(encoding="utf-8")))
        s = t["summary"]
        self.assertEqual(s["adaptive"]["mean_error_px"], 11.482721)
        self.assertEqual(s["policy"]["mean_error_px"], 9.261511)
        self.assertEqual(s["persistence"]["mean_error_px"], 13.387468)
        self.assertEqual(s["constant_velocity_2pt"]["mean_error_px"], 9.015013)
        self.assertEqual(s["alpha_beta"]["mean_error_px"], 11.513307)
        self.assertEqual(t["warning_frames"], [36])
        self.assertEqual(t["zone_summary"]["coverage"]["90"], 0.8958)
        self.assertEqual(t["final_learning_state_sha256"],
                         "77206fa7fde18dc30ef8be6c7bfb1bf8671a1f436472d6bc2a1d8113bff81040")


if __name__ == "__main__":
    unittest.main()