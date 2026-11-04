
import copy
import unittest

from app.physics.geometry_contact_evidence import associate_evidence


def fixtures():
    manifest = {
        "schema": "qapha.calibration_clip.v1",
        "calibration_sha256": "test-calibration",
        "frame_range_inclusive": [0, 49],
        "frames_verified": 50,
        "future_frames_included": False,
    }

    surface = {
        "schema": "qapha.surface_association.v1",
        "calibration_sha256": "test-calibration",
        "cutoff_frame": 49,
        "object_id": "0",
        "latest_regime": "candidate_surface_constrained_motion",
        "candidate_support": {
            "bottom_y_px": 679.0,
        },
        "candidate_events": [
            {
                "frame_index": 36,
                "classification": "candidate_contact_reversal",
            }
        ],
        "tracking_quality_flags": [],
    }

    measurements = [
        {
            "frame_index": frame,
            "edge_y_px": 655,
            "gradient_strength": 48.0,
        }
        for frame in (0, 0, 49, 49)
    ]

    return measurements, surface, manifest


class GeometryContactEvidenceTests(unittest.TestCase):

    def test_identifies_stable_visual_edge(self):
        report = associate_evidence(*fixtures())

        self.assertTrue(
            report["visual_geometry"]["stable_edge"]
        )
        self.assertEqual(
            report["visual_geometry"][
                "candidate_visual_edge_y_px"
            ],
            655.0,
        )
        self.assertEqual(
            report["object_geometry_relation"][
                "vertical_offset_px"
            ],
            24.0,
        )

    def test_never_verifies_collision(self):
        report = associate_evidence(*fixtures())

        self.assertFalse(
            report["physical_collision_verified"]
        )
        self.assertFalse(
            report["candidate_events"][0][
                "physical_collision_verified"
            ]
        )

    def test_rejects_mismatched_calibration(self):
        values = list(fixtures())
        values[1] = copy.deepcopy(values[1])
        values[1]["calibration_sha256"] = "wrong"

        with self.assertRaises(ValueError):
            associate_evidence(*values)

    def test_rejects_future_calibration(self):
        values = list(fixtures())
        values[2] = copy.deepcopy(values[2])
        values[2]["future_frames_included"] = True

        with self.assertRaises(ValueError):
            associate_evidence(*values)

    def test_unstable_visual_edge_not_verified(self):
        values = list(fixtures())
        values[0] = copy.deepcopy(values[0])
        values[0][-1]["edge_y_px"] = 680

        report = associate_evidence(*values)

        self.assertFalse(
            report["visual_geometry"]["stable_edge"]
        )
        self.assertIsNone(
            report["visual_geometry"][
                "candidate_visual_edge_y_px"
            ]
        )


if __name__ == "__main__":
    unittest.main()
