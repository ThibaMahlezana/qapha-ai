
import unittest

from app.experiments.heldout_forecast import freeze_forecast


def fixtures():
    tracking = {
        "outcome": "completed",
        "boxes": [
            {
                "frame_index": f,
                "object_id": "0",
                "bbox_xyxy_px": [
                    2 * f - 2,
                    f * f - 2,
                    2 * f + 2,
                    f * f + 2,
                ],
            }
            for f in range(10)
        ],
    }
    manifest = {
        "schema": "qapha.calibration_clip.v1",
        "future_frames_included": False,
        "frame_range_inclusive": [0, 9],
        "frames_verified": 10,
        "calibration_sha256": "test",
        "fps": 60.0,
    }
    return tracking, manifest


class HeldoutForecastTests(unittest.TestCase):

    def test_known_quadratic_motion(self):
        result = freeze_forecast(
            *fixtures(), cutoff=9, training_samples=8,
            horizons=(1,),
        )
        predicted = result["predictions"][0]
        self.assertAlmostEqual(
            predicted["quadratic_y_center_px"][1], 100
        )
        self.assertAlmostEqual(
            predicted["quadratic_y_center_px"][0], 20
        )

    def test_rejects_future_tracking(self):
        tracking, manifest = fixtures()
        tracking["boxes"].append({
            "frame_index": 10,
            "object_id": "0",
            "bbox_xyxy_px": [1, 1, 5, 5],
        })
        with self.assertRaises(ValueError):
            freeze_forecast(
                tracking, manifest, cutoff=9,
                training_samples=8,
            )

    def test_requires_consecutive_history(self):
        tracking, manifest = fixtures()
        tracking["boxes"] = [
            box for box in tracking["boxes"]
            if box["frame_index"] != 8
        ]
        with self.assertRaises(ValueError):
            freeze_forecast(
                tracking, manifest, cutoff=9,
                training_samples=8,
            )

    def test_rejects_leaky_manifest(self):
        tracking, manifest = fixtures()
        manifest["future_frames_included"] = True
        with self.assertRaises(ValueError):
            freeze_forecast(
                tracking, manifest, cutoff=9,
                training_samples=8,
            )


if __name__ == "__main__":
    unittest.main()
