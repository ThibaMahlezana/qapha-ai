
import copy
import unittest

from app.experiments.contact_audit import audit_contact


def fixtures():
    forecast = {
        "schema": "qapha.conditional_contact.v1",
        "object_id": "1",
        "cutoff_frame": 4,
        "contact_assumption": {
            "edge_id": "edge_0",
            "physical_surface_verified": False,
        },
        "hypothetical_impacts": [
            {
                "time_after_cutoff_frames": 5.0,
                "image_x_px": 50.0,
            }
        ],
    }

    geometry = {
        "schema": "qapha.scene_geometry.v1",
        "candidates": [
            {
                "endpoints_px": [
                    [0, 100], [100, 100]
                ]
            }
        ],
    }

    ys = {
        7: 65,
        8: 80,
        9: 92,
        10: 78,
        11: 65,
    }

    tracking = {
        "outcome": "completed",
        "boxes": [
            {
                "object_id": "1",
                "frame_index": frame,
                "bbox_xyxy_px": [
                    40, y - 10,
                    60, y + 10,
                ],
            }
            for frame, y in ys.items()
        ],
    }

    return forecast, tracking, geometry


class ContactAuditTests(unittest.TestCase):

    def test_detects_near_edge_reversal_without_proving_contact(self):
        result = audit_contact(*fixtures())

        report = result["audits"][0]

        self.assertTrue(report["any_reversal_near_edge"])
        self.assertFalse(
            report["physical_collision_verified"]
        )
        self.assertEqual(
            report["vertical_down_to_up_reversals"][0][
                "frame_index"
            ],
            9,
        )

    def test_far_edge_is_not_near(self):
        forecast, tracking, geometry = fixtures()

        geometry["candidates"][0]["endpoints_px"] = [
            [200, 100], [300, 100]
        ]

        result = audit_contact(
            forecast, tracking, geometry
        )

        self.assertFalse(
            result["audits"][0]["any_reversal_near_edge"]
        )

    def test_missing_frames_do_not_invent_reversal(self):
        forecast, tracking, geometry = fixtures()

        tracking["boxes"] = [
            box for box in tracking["boxes"]
            if box["frame_index"] != 9
        ]

        result = audit_contact(
            forecast, tracking, geometry
        )

        self.assertEqual(
            result["audits"][0][
                "vertical_down_to_up_reversals"
            ],
            [],
        )

    def test_invalid_input_and_duplicates_rejected(self):
        forecast, tracking, geometry = fixtures()

        duplicate = copy.deepcopy(tracking)
        duplicate["boxes"].append(
            copy.deepcopy(duplicate["boxes"][0])
        )

        with self.assertRaises(ValueError):
            audit_contact(forecast, duplicate, geometry)

        with self.assertRaises(ValueError):
            audit_contact(
                forecast, tracking, geometry,
                window_frames=0,
            )

        invalid = copy.deepcopy(forecast)
        invalid["contact_assumption"]["edge_id"] = "edge_99"

        with self.assertRaises(ValueError):
            audit_contact(invalid, tracking, geometry)


if __name__ == "__main__":
    unittest.main()
