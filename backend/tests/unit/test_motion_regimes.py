
import unittest

from app.physics.motion_regimes import analyze_regimes


def fixtures(points):
    boxes = []

    for frame, (x, y) in enumerate(points):
        boxes.append({
            "frame_index": frame,
            "object_id": "0",
            "bbox_xyxy_px": [
                x - 2, y - 2, x + 2, y + 2,
            ],
        })

    return (
        {"outcome": "completed", "boxes": boxes},
        {
            "schema": "qapha.calibration_clip.v1",
            "frame_range_inclusive": [0, len(points) - 1],
            "frames_verified": len(points),
            "future_frames_included": False,
            "calibration_sha256": "test",
        },
    )


class MotionRegimesTests(unittest.TestCase):

    def test_vertical_motion(self):
        points = [(100, 100 + frame * 10)
                  for frame in range(12)]

        result = analyze_regimes(*fixtures(points))

        self.assertEqual(
            result["latest_regime"],
            "vertical_dominant",
        )

    def test_horizontal_motion(self):
        points = [(100 + frame * 5, 100)
                  for frame in range(12)]

        result = analyze_regimes(*fixtures(points))

        self.assertEqual(
            result["latest_regime"],
            "approximately_horizontal",
        )

    def test_stationary(self):
        points = [(100, 100) for _ in range(12)]

        result = analyze_regimes(*fixtures(points))

        self.assertEqual(
            result["latest_regime"],
            "approximately_stationary",
        )

    def test_detects_transition(self):
        points = (
            [(100, 100 + f * 10) for f in range(10)]
            + [(100 + (f + 1) * 10, 190)
               for f in range(10)]
        )

        result = analyze_regimes(*fixtures(points))

        self.assertEqual(
            result["latest_regime"],
            "approximately_horizontal",
        )
        self.assertTrue(result["transitions"])

    def test_rejects_future_observations(self):
        points = [(100, 100 + f * 10) for f in range(12)]
        tracks, manifest = fixtures(points)

        tracks["boxes"].append({
            "frame_index": 12,
            "object_id": "0",
            "bbox_xyxy_px": [0, 0, 10, 10],
        })

        with self.assertRaises(ValueError):
            analyze_regimes(tracks, manifest)


if __name__ == "__main__":
    unittest.main()
