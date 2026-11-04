
import copy
import unittest

from app.reasoning.contact_hypotheses import build_contact_hypotheses


def fixtures():
    manifest = {
        "schema": "qapha.calibration_clip.v1",
        "frame_range_inclusive": [0, 49],
        "frames_verified": 50,
        "future_frames_included": False,
        "calibration_sha256": "calibration-test",
    }

    surface = {
        "schema": "qapha.surface_association.v1",
        "stage": "calibration_only",
        "object_id": "0",
        "cutoff_frame": 49,
        "calibration_sha256": "calibration-test",
        "latest_regime": "candidate_surface_constrained_motion",
        "candidate_support": {
            "bottom_y_px": 679.0,
            "stable_height": True,
            "horizontal_travel_px": 8.0,
        },
        "candidate_events": [
            {
                "frame_index": 36,
                "classification": "candidate_contact_reversal",
                "physical_collision_verified": False,
            },
        ],
    }

    geometry = {
        "schema": "qapha.geometry_contact_evidence.v1",
        "stage": "calibration_only",
        "object_id": "0",
        "cutoff_frame": 49,
        "calibration_sha256": "calibration-test",
        "physical_collision_verified": False,
        "visual_geometry": {
            "stable_edge": True,
            "candidate_visual_edge_y_px": 655.0,
            "physical_contact_plane_verified": False,
        },
        "object_geometry_relation": {
            "stable_bottom_y_px": 679.0,
            "vertical_offset_px": 24.0,
        },
        "candidate_events": [
            {
                "frame_index": 36,
                "motion_evidence": "candidate_contact_reversal",
                "independent_visual_edge_available": True,
                "physical_collision_verified": False,
            },
        ],
        "tracking_quality_flags": [
            {
                "from_frame": 34,
                "to_frame": 35,
                "displacement_px": 173.0,
                "requires_review": True,
            }
        ],
    }

    return geometry, surface, manifest


class ContactHypothesesTests(unittest.TestCase):

    def test_builds_event_and_support_hypotheses(self):
        result = build_contact_hypotheses(*fixtures())
        self.assertEqual(result["hypothesis_count"], 2)
        self.assertEqual(
            result["hypotheses"][1]["type"],
            "possible_surface_constrained_motion",
        )

    def test_never_verifies_physical_collision(self):
        result = build_contact_hypotheses(*fixtures())
        self.assertFalse(result["physical_collision_verified"])
        self.assertTrue(all(
            h["physical_collision_verified"] is False
            for h in result["hypotheses"]
        ))

    def test_flags_large_jump_near_reversal(self):
        result = build_contact_hypotheses(*fixtures())
        self.assertTrue(result["hypotheses"][0]["quality_warning"])

    def test_rejects_mismatched_calibration(self):
        values = list(fixtures())
        values[1] = copy.deepcopy(values[1])
        values[1]["calibration_sha256"] = "wrong"

        with self.assertRaises(ValueError):
            build_contact_hypotheses(*values)

    def test_rejects_inconsistent_event_frames(self):
        values = list(fixtures())
        values[0] = copy.deepcopy(values[0])
        values[0]["candidate_events"][0]["frame_index"] = 42

        with self.assertRaises(ValueError):
            build_contact_hypotheses(*values)


if __name__ == "__main__":
    unittest.main()
