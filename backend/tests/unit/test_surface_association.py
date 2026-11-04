
import unittest

from app.physics.surface_association import (
    analyze_surface_association,
)


def fixtures():
    # Synthetic descent, reversal, then horizontal support.
    centers_y = [
        50, 80, 110, 140, 170, 200, 220,
        210, 205, 210, 220, 220, 220, 220, 220, 220,
    ]

    boxes = []

    for frame, y in enumerate(centers_y):
        x = 100 + frame * 2
        boxes.append({
            "frame_index": frame,
            "object_id": "0",
            "bbox_xyxy_px": [
                x - 10, y - 10,
                x + 10, y + 10,
            ],
        })

    tracks = {"outcome": "completed", "boxes": boxes}

    manifest = {
        "schema": "qapha.calibration_clip.v1",
        "frame_range_inclusive": [0, len(centers_y) - 1],
        "frames_verified": len(centers_y),
        "future_frames_included": False,
        "calibration_sha256": "test",
    }

    return tracks, manifest


class SurfaceAssociationTests(unittest.TestCase):

    def test_detects_candidate_support(self):
        result = analyze_surface_association(*fixtures())

        self.assertEqual(
            result["latest_regime"],
            "candidate_surface_constrained_motion",
        )
        self.assertEqual(
            result["candidate_support"]["bottom_y_px"],
            230.0,
        )

    def test_detects_candidate_reversal(self):
        result = analyze_surface_association(*fixtures())

        self.assertTrue(result["candidate_events"])
        self.assertEqual(
            result["candidate_events"][0]["frame_index"], 6
        )
        self.assertFalse(
            result["candidate_events"][0]["physical_collision_verified"]
        )

    def test_no_claim_of_verified_surface(self):
        result = analyze_surface_association(*fixtures())

        self.assertFalse(
            result["candidate_support"][
                "independent_surface_geometry_verified"
            ]
        )

    def test_rejects_future_frames(self):
        tracks, manifest = fixtures()

        tracks["boxes"].append({
            "frame_index": 16,
            "object_id": "0",
            "bbox_xyxy_px": [0, 0, 10, 10],
        })

        with self.assertRaises(ValueError):
            analyze_surface_association(tracks, manifest)

    def test_unstable_height_does_not_establish_support(self):
        tracks, manifest = fixtures()

        for item in tracks["boxes"][-6:]:
            frame = item["frame_index"]
            item["bbox_xyxy_px"][1] += (frame - 10) * 10
            item["bbox_xyxy_px"][3] += (frame - 10) * 10

        result = analyze_surface_association(tracks, manifest)

        self.assertEqual(
            result["latest_regime"], "support_not_established"
        )
        self.assertIsNone(
            result["candidate_support"]["bottom_y_px"]
        )


if __name__ == "__main__":
    unittest.main()
