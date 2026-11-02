
import copy
import unittest

from app.experiments.video_forecast import freeze, verify


def make_report(end):
    return {
        "outcome": "completed",
        "boxes": [
            {
                "frame_index": frame,
                "object_id": "1",
                "bbox_xyxy_px": [
                    frame,
                    2 * frame,
                    frame + 10,
                    2 * frame + 10,
                ],
            }
            for frame in range(end + 1)
        ],
    }


class VideoForecastTests(unittest.TestCase):

    def test_freezes_and_verifies_linear_motion(self):
        frozen = freeze(make_report(49))

        self.assertEqual(
            frozen["predicted_center_px"],
            {"x": 65.0, "y": 125.0},
        )

        evaluation = verify(frozen, make_report(60))

        self.assertEqual(
            evaluation["euclidean_error_px"], 0
        )

    def test_rejects_future_data_at_freeze(self):
        with self.assertRaisesRegex(ValueError, "future frames"):
            freeze(make_report(60))

    def test_detects_tampering_and_prefix_changes(self):
        frozen = freeze(make_report(49))

        changed = copy.deepcopy(frozen)
        changed["predicted_center_px"]["y"] = 999

        with self.assertRaisesRegex(ValueError, "modified"):
            verify(changed, make_report(60))

        altered = make_report(60)
        altered["boxes"][2]["bbox_xyxy_px"] = [
            100, 100, 110, 110
        ]

        with self.assertRaisesRegex(ValueError, "differ"):
            verify(frozen, altered)

    def test_rejects_missing_target_and_gapped_window(self):
        frozen = freeze(make_report(49))

        with self.assertRaisesRegex(ValueError, "unavailable"):
            verify(frozen, make_report(59))

        gapped = make_report(49)
        gapped["boxes"] = [
            box for box in gapped["boxes"]
            if box["frame_index"] != 47
        ]

        with self.assertRaisesRegex(ValueError, "consecutive"):
            freeze(gapped)


if __name__ == "__main__":
    unittest.main()
