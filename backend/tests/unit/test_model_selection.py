
import unittest

from app.experiments.model_selection import select_model


def fixtures(quadratic=False):
    boxes = []

    for frame in range(40):
        x = 100 + 2 * frame
        y = 100 + 3 * frame

        if quadratic:
            y += 0.25 * frame * frame

        boxes.append({
            "frame_index": frame,
            "object_id": "0",
            "bbox_xyxy_px": [
                x - 4, y - 4, x + 4, y + 4
            ],
        })

    tracks = {
        "outcome": "completed",
        "boxes": boxes,
    }

    manifest = {
        "schema": "qapha.calibration_clip.v1",
        "frame_range_inclusive": [0, 39],
        "frames_verified": 40,
        "future_frames_included": False,
        "calibration_sha256": "test",
    }

    return tracks, manifest


class ModelSelectionTests(unittest.TestCase):

    def test_selects_linear_for_constant_velocity(self):
        result = select_model(
            *fixtures(), cutoff=39
        )
        self.assertEqual(result["selected_model"], "linear")

    def test_selects_quadratic_for_acceleration(self):
        result = select_model(
            *fixtures(quadratic=True), cutoff=39
        )
        self.assertEqual(
            result["selected_model"], "quadratic_y"
        )

    def test_rejects_future_observations(self):
        tracks, manifest = fixtures()
        tracks["boxes"].append({
            "frame_index": 40,
            "object_id": "0",
            "bbox_xyxy_px": [0, 0, 10, 10],
        })
        with self.assertRaises(ValueError):
            select_model(tracks, manifest, cutoff=39)

    def test_rejects_future_in_manifest(self):
        tracks, manifest = fixtures()
        manifest["future_frames_included"] = True

        with self.assertRaises(ValueError):
            select_model(tracks, manifest, cutoff=39)

    def test_requires_enough_validation_folds(self):
        with self.assertRaises(ValueError):
            select_model(
                *fixtures(),
                cutoff=39,
                min_folds=100,
            )


if __name__ == "__main__":
    unittest.main()
