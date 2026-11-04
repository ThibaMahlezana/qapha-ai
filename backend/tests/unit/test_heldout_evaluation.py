
import copy
import unittest

from app.experiments.heldout_forecast import freeze_forecast
from app.experiments.heldout_evaluation import evaluate


def fixtures():
    calibration = {
        "outcome": "completed",
        "concept": "red ball",
        "boxes": [
            {
                "frame_index": frame,
                "object_id": "0",
                "bbox_xyxy_px": [
                    frame * 2 - 2,
                    frame * frame - 2,
                    frame * 2 + 2,
                    frame * frame + 2,
                ],
            }
            for frame in range(10)
        ],
    }

    manifest = {
        "schema": "qapha.calibration_clip.v1",
        "source_sha256": "source-test",
        "calibration_sha256": "calibration-test",
        "frame_range_inclusive": [0, 9],
        "frames_verified": 10,
        "future_frames_included": False,
        "fps": 60.0,
    }

    protocol = {
        "schema": "qapha.heldout_protocol.v1",
        "video": {
            "sha256": "source-test",
            "fps": 60.0,
            "resolution_px": [720, 1280],
        },
        "calibration_frames_inclusive": [0, 9],
        "target_frames": [10, 11],
    }

    frozen = freeze_forecast(
        calibration,
        manifest,
        cutoff=9,
        training_samples=8,
        horizons=(1, 2),
    )

    verification = {
        "outcome": "completed",
        "concept": "red ball",
        "boxes": [
            {
                "frame_index": 0,
                "object_id": "0",
                "bbox_xyxy_px": [18, 98, 22, 102],
            },
            {
                "frame_index": 1,
                "object_id": "0",
                "bbox_xyxy_px": [20, 119, 24, 123],
            },
        ],
    }

    return frozen, calibration, manifest, verification, protocol


class HeldoutEvaluationTests(unittest.TestCase):

    def test_evaluates_valid_forecast(self):
        report = evaluate(
            *fixtures(),
            verification_start=10,
        )

        self.assertTrue(report["calibration_reproduced_exactly"])
        self.assertEqual(len(report["evaluations"]), 2)
        self.assertFalse(report["evaluations"][0]["boundary_touch"])
        self.assertAlmostEqual(
            report["evaluations"][0]["errors_px"]["quadratic_y"],
            0.0,
        )

    def test_detects_tampered_frozen_prediction(self):
        values = list(fixtures())
        values[0] = copy.deepcopy(values[0])
        values[0]["predictions"][0]["linear_center_px"][0] += 5

        with self.assertRaises(ValueError):
            evaluate(*values, verification_start=10)

    def test_detects_wrong_source_video(self):
        values = list(fixtures())
        values[4] = copy.deepcopy(values[4])
        values[4]["video"]["sha256"] = "wrong"

        with self.assertRaises(ValueError):
            evaluate(*values, verification_start=10)

    def test_rejects_missing_target_observation(self):
        values = list(fixtures())
        values[3] = copy.deepcopy(values[3])
        values[3]["boxes"].pop()

        with self.assertRaises(ValueError):
            evaluate(*values, verification_start=10)

    def test_flags_image_boundary_contact(self):
        values = list(fixtures())
        values[3] = copy.deepcopy(values[3])
        values[3]["boxes"][0]["bbox_xyxy_px"] = [0, 98, 4, 102]

        report = evaluate(*values, verification_start=10)

        self.assertTrue(report["evaluations"][0]["boundary_touch"])


if __name__ == "__main__":
    unittest.main()
