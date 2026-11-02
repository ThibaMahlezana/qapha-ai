
import json
import unittest
from types import SimpleNamespace

from app.reasoning.temporal_reasoner import (
    prepare_evidence,
    validate_response,
    reason,
)


def scene():
    return {
        "schema": "qapha.temporal_scene.v1",
        "frame_window": [35, 49],
        "reviewed_edges": [
            {
                "edge_id": "edge_1",
                "physical_surface_verified": False,
            }
        ],
        "objects": [
            {
                "object_id": "0",
                "status": "measured_image_motion",
                "motion_class": "moving",
                "relationships": [
                    {
                        "edge_id": "edge_1",
                        "distance_first_px": 80,
                        "distance_last_px": 35,
                        "distance_change_px": -45,
                        "distance_trend": "approaching",
                        "near_at_end": False,
                        "physical_contact_verified": False,
                    }
                ],
            },
            {
                "object_id": "2",
                "status": "measured_image_motion",
                "motion_class": "approximately_stationary",
                "relationships": [
                    {
                        "edge_id": "edge_1",
                        "distance_first_px": 28.5,
                        "distance_last_px": 28.1,
                        "distance_change_px": -0.4,
                        "distance_trend": "approximately_constant_distance",
                        "near_at_end": True,
                        "physical_contact_verified": False,
                    }
                ],
            },
        ],
    }


def answers():
    return {
        "assessments": [
            {
                "object_id": "0",
                "edge_id": "edge_1",
                "assessment": "approaching_candidate_edge",
                "reason": "Image distance decreased.",
                "next_observation": "Observe intervening frames.",
            },
            {
                "object_id": "2",
                "edge_id": "edge_1",
                "assessment": "stationary_near_edge",
                "reason": "Small displacement near line.",
                "next_observation": "Measure depth if available.",
            },
        ]
    }


class FakeClient:
    def __init__(self, payload):
        self.calls = []
        self.payload = payload
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.create
            )
        )

    def create(self, **kwargs):
        self.calls.append(kwargs)

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(self.payload)
                    )
                )
            ]
        )


class TemporalReasonerTests(unittest.TestCase):

    def test_prepares_evidence(self):
        result = prepare_evidence(scene())

        self.assertEqual(
            result["frame_window"], [35, 49]
        )
        self.assertEqual(len(result["objects"]), 2)

    def test_accepts_grounded_assessments(self):
        result = validate_response(
            answers(), prepare_evidence(scene())
        )

        self.assertEqual(len(result["assessments"]), 2)
        self.assertFalse(
            result["physical_contact_verified"]
        )

    def test_rejects_invented_or_unsupported_edges(self):
        bad = answers()
        bad["assessments"][0]["edge_id"] = "edge_99"

        with self.assertRaises(ValueError):
            validate_response(
                bad, prepare_evidence(scene())
            )

    def test_rejects_unsupported_classification(self):
        bad = answers()
        bad["assessments"][0]["assessment"] = (
            "stationary_near_edge"
        )

        with self.assertRaises(ValueError):
            validate_response(
                bad, prepare_evidence(scene())
            )

    def test_rejects_missing_objects(self):
        bad = answers()
        bad["assessments"].pop()

        with self.assertRaises(ValueError):
            validate_response(
                bad, prepare_evidence(scene())
            )

    def test_mocked_muse_call(self):
        client = FakeClient(answers())

        result = reason(
            scene(), client=client
        )

        self.assertEqual(
            result["status"],
            "unverified_hypotheses",
        )
        self.assertEqual(
            client.calls[0]["response_format"]["type"],
            "json_schema",
        )

    def test_empty_scene_skips_api(self):
        data = scene()
        data["objects"] = []

        result = reason(data)

        self.assertEqual(result["assessments"], [])


if __name__ == "__main__":
    unittest.main()
