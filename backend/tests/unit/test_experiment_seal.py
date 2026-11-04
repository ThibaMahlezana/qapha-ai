
import copy
import unittest

from app.experiments.experiment_seal import (
    canonical_hash,
    seal_experiment,
)
from app.experiments.heldout_forecast import freeze_forecast
from app.experiments.model_selection import select_model


def fixtures():
    source_hash = "a" * 64
    calibration_hash = "b" * 64

    tracks = {
        "outcome": "completed",
        "concept": "red ball",
        "boxes": [
            {
                "frame_index": frame,
                "object_id": "0",
                "bbox_xyxy_px": [
                    100 + frame - 5,
                    100 + frame * frame / 4 - 5,
                    100 + frame + 5,
                    100 + frame * frame / 4 + 5,
                ],
            }
            for frame in range(40)
        ],
    }

    manifest = {
        "schema": "qapha.calibration_clip.v1",
        "source_sha256": source_hash,
        "calibration_sha256": calibration_hash,
        "frame_range_inclusive": [0, 39],
        "frames_verified": 40,
        "future_frames_included": False,
        "fps": 60.0,
        "resolution_px": [720, 1280],
    }

    protocol = {
        "schema": "qapha.heldout_protocol.v1",
        "video": {
            "sha256": source_hash,
            "fps": 60.0,
            "resolution_px": [720, 1280],
        },
        "calibration_frames_inclusive": [0, 39],
        "target_frames": [44, 49, 60],
    }

    selection = select_model(
        tracks,
        manifest,
        cutoff=39,
        object_id="0",
        training_samples=12,
        horizon=3,
        step=3,
    )

    forecast = freeze_forecast(
        tracks,
        manifest,
        cutoff=39,
        object_id="0",
        training_samples=12,
        horizons=(5, 10, 21),
    )

    return selection, forecast, protocol, manifest, tracks


class ExperimentSealTests(unittest.TestCase):

    def test_seals_reproducible_experiment(self):
        result = seal_experiment(*fixtures())

        self.assertTrue(result["selection_reproduced_exactly"])
        self.assertTrue(result["forecast_reproduced_exactly"])
        self.assertFalse(result["verification_frames_used"])

        body = {
            k: v for k, v in result.items()
            if k != "seal_sha256"
        }
        self.assertEqual(
            result["seal_sha256"],
            canonical_hash(body),
        )

    def test_rejects_changed_selection(self):
        values = list(fixtures())
        values[0] = copy.deepcopy(values[0])
        values[0]["selected_model"] = (
            "linear"
            if values[0]["selected_model"] == "quadratic_y"
            else "quadratic_y"
        )

        with self.assertRaises(ValueError):
            seal_experiment(*values)

    def test_rejects_changed_forecast(self):
        values = list(fixtures())
        values[1] = copy.deepcopy(values[1])
        values[1]["predictions"][0]["linear_center_px"][0] += 10

        with self.assertRaises(ValueError):
            seal_experiment(*values)

    def test_rejects_changed_source(self):
        values = list(fixtures())
        values[2] = copy.deepcopy(values[2])
        values[2]["video"]["sha256"] = "c" * 64

        with self.assertRaises(ValueError):
            seal_experiment(*values)

    def test_rejects_future_tracking(self):
        values = list(fixtures())
        values[4] = copy.deepcopy(values[4])
        values[4]["boxes"].append({
            "frame_index": 40,
            "object_id": "0",
            "bbox_xyxy_px": [1, 1, 10, 10],
        })

        with self.assertRaises(ValueError):
            seal_experiment(*values)


if __name__ == "__main__":
    unittest.main()
