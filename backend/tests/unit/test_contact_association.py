
import unittest

from app.experiments.contact_association import (
    associate_contact_events,
)


def fixture():
    return {
        "schema": "qapha.contact_audit.v1",
        "stage": "post_prediction_evaluation_only",
        "object_id": "1",
        "forecast_cutoff_frame": 49,
        "reviewed_edge_id": "edge_0",
        "audits": [{
            "predicted_impact_frame": 60.050853,
            "audit_frame_window": [54, 66],
            "closest_bottom_to_edge": {
                "frame_index": 60,
                "bottom_edge_distance_px": 3.0,
            },
            "vertical_down_to_up_reversals": [{
                "frame_index": 61,
                "delta_before_px": 34.5,
                "delta_after_px": -30.0,
            }],
            "physical_collision_verified": False,
        }],
    }


class ContactAssociationTests(unittest.TestCase):

    def test_associates_close_approach_then_reversal(self):
        result = associate_contact_events(fixture())

        self.assertEqual(
            result["associations"][0]["association"],
            "candidate_temporal_contact",
        )
        self.assertEqual(
            result["associations"][0]["matched_reversal_frames"],
            [61],
        )
        self.assertFalse(result["physical_collision_verified"])

    def test_rejects_far_approach(self):
        audit = fixture()
        audit["audits"][0]["closest_bottom_to_edge"][
            "bottom_edge_distance_px"
        ] = 38.0

        result = associate_contact_events(audit)

        self.assertEqual(
            result["associations"][0]["association"],
            "not_associated",
        )

    def test_rejects_reversal_before_approach_or_too_late(self):
        for frame in (59, 63):
            audit = fixture()
            audit["audits"][0][
                "vertical_down_to_up_reversals"
            ][0]["frame_index"] = frame

            result = associate_contact_events(audit)

            self.assertEqual(
                result["associations"][0][
                    "matched_reversal_frames"
                ],
                [],
            )

    def test_missing_closest_does_not_claim_contact(self):
        audit = fixture()
        audit["audits"][0]["closest_bottom_to_edge"] = None

        result = associate_contact_events(audit)

        self.assertEqual(
            result["associations"][0]["association"],
            "not_associated",
        )

    def test_rejects_invalid_provenance_and_thresholds(self):
        audit = fixture()
        audit["stage"] = "forecast_input"

        with self.assertRaises(ValueError):
            associate_contact_events(audit)

        with self.assertRaises(ValueError):
            associate_contact_events(
                fixture(),
                max_delay_frames=-1,
            )

        with self.assertRaises(ValueError):
            associate_contact_events(
                fixture(),
                max_distance_px=float("nan"),
            )


if __name__ == "__main__":
    unittest.main()
