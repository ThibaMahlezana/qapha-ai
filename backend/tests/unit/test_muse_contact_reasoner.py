
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.reasoning.muse_contact_reasoner import (
    digest,
    prepare_packet,
    validate_response,
    reason,
)


def fixtures():
    records = [
        {
            "evidence_id": "candidate_reversal_frame_36",
            "frame_index": 36,
            "hypothesis_type": "possible_surface_interaction",
            "observed_evidence": {
                "vertical_direction_reversal": True,
            },
            "candidate_explanation": "Possible interaction",
            "quality_warning": True,
            "quality_flags": [{"requires_review": True}],
            "limitations": ["Contact not verified"],
            "physical_collision_verified": False,
        },
        {
            "evidence_id": "candidate_support_frame_49",
            "frame_index": 49,
            "hypothesis_type": "possible_surface_constrained_motion",
            "observed_evidence": {
                "stable_object_bottom_y_px": 679,
            },
            "candidate_explanation": "Possible support",
            "quality_warning": False,
            "quality_flags": [],
            "limitations": ["No contact plane"],
            "physical_collision_verified": False,
        },
    ]

    body = {
        "schema": "qapha.muse_evidence_packet.v1",
        "stage": "calibration_only",
        "object_id": "0",
        "cutoff_frame": 49,
        "calibration_sha256": "calibration-test",
        "source_artifacts": {},
        "scene_observations": {
            "independent_contact_plane_verified": False,
            "visual_edge_y_px": 655,
        },
        "evidence_records": records,
        "allowed_evidence_ids": sorted(
            r["evidence_id"] for r in records
        ),
        "reasoning_task": {
            "permitted_conclusions": [
                "possible_surface_interaction",
                "possible_surface_constrained_motion",
                "insufficient_evidence",
            ],
        },
    }

    return {**body, "packet_sha256": digest(body)}


def proposal():
    return {
        "assessments": [
            {
                "evidence_id": "candidate_reversal_frame_36",
                "frame_index": 36,
                "conclusion": "possible_surface_interaction",
                "reason": "A vertical reversal was measured.",
                "alternative_explanation": "Tracking error.",
                "next_observation": "Check frame-by-frame masks.",
            },
            {
                "evidence_id": "candidate_support_frame_49",
                "frame_index": 49,
                "conclusion": "possible_surface_constrained_motion",
                "reason": "The tracked height remained stable.",
                "alternative_explanation": "Perspective effects.",
                "next_observation": "Measure a physical surface.",
            },
        ]
    }


class MuseContactReasonerTests(unittest.TestCase):

    def test_accepts_grounded_assessments(self):
        result = validate_response(
            proposal(), prepare_packet(fixtures())
        )
        self.assertEqual(result["assessment_count"], 2)
        self.assertFalse(result["physical_collision_verified"])
        self.assertTrue(
            result["assessments"][0]["input_quality_warning"]
        )

    def test_rejects_invented_evidence_id(self):
        data = proposal()
        data["assessments"][0]["evidence_id"] = "invented"

        with self.assertRaises(ValueError):
            validate_response(data, prepare_packet(fixtures()))

    def test_rejects_wrong_frame(self):
        data = proposal()
        data["assessments"][0]["frame_index"] = 70

        with self.assertRaises(ValueError):
            validate_response(data, prepare_packet(fixtures()))

    def test_rejects_missing_assessment(self):
        data = proposal()
        data["assessments"].pop()

        with self.assertRaises(ValueError):
            validate_response(data, prepare_packet(fixtures()))

    def test_rejects_incompatible_conclusion(self):
        data = proposal()
        data["assessments"][0]["conclusion"] = (
            "possible_surface_constrained_motion"
        )

        with self.assertRaises(ValueError):
            validate_response(data, prepare_packet(fixtures()))

    def test_rejects_tampered_packet(self):
        packet = copy.deepcopy(fixtures())
        packet["cutoff_frame"] = 70

        with self.assertRaises(ValueError):
            prepare_packet(packet)

    def test_mocked_muse_request(self):
        client = MagicMock()
        client.chat.completions.create.return_value = (
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(proposal())
                        )
                    )
                ]
            )
        )

        result = reason(fixtures(), client=client)

        self.assertEqual(result["assessment_count"], 2)
        client.chat.completions.create.assert_called_once()

        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "muse-spark-1.3")
        self.assertEqual(
            kwargs["response_format"]["type"],
            "json_schema",
        )


if __name__ == "__main__":
    unittest.main()
