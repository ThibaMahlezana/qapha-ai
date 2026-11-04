
import copy
import unittest

from app.reasoning.muse_contact_reasoner import (
    digest,
    prepare_packet,
    validate_response,
)
from app.reasoning.reasoning_audit import audit_reasoning


def fixtures():
    record = {
        "evidence_id": "candidate_reversal_frame_36",
        "frame_index": 36,
        "hypothesis_type": "possible_surface_interaction",
        "observed_evidence": {
            "vertical_direction_reversal": True,
        },
        "candidate_explanation": "Possible interaction",
        "quality_warning": True,
        "quality_flags": [{"requires_review": True}],
        "limitations": ["No verified collision"],
        "physical_collision_verified": False,
    }

    body = {
        "schema": "qapha.muse_evidence_packet.v1",
        "stage": "calibration_only",
        "object_id": "0",
        "cutoff_frame": 49,
        "calibration_sha256": "test",
        "source_artifacts": {},
        "scene_observations": {
            "independent_contact_plane_verified": False,
        },
        "evidence_records": [record],
        "allowed_evidence_ids": [record["evidence_id"]],
        "reasoning_task": {
            "permitted_conclusions": [
                "possible_surface_interaction",
                "possible_surface_constrained_motion",
                "insufficient_evidence",
            ],
        },
    }

    packet = {**body, "packet_sha256": digest(body)}

    proposal = {
        "assessments": [{
            "evidence_id": "candidate_reversal_frame_36",
            "frame_index": 36,
            "conclusion": "insufficient_evidence",
            "reason": "Measurement quality is uncertain.",
            "alternative_explanation": "Tracking error.",
            "next_observation": "Inspect consecutive masks.",
        }]
    }

    reasoning = validate_response(
        proposal, prepare_packet(packet)
    )

    return packet, reasoning


class ReasoningAuditTests(unittest.TestCase):

    def test_valid_audit(self):
        result = audit_reasoning(*fixtures())

        self.assertTrue(result["coverage_complete"])
        self.assertTrue(result["structural_validation_passed"])
        self.assertEqual(result["assessment_count"], 1)

    def test_quality_warning_requires_review(self):
        result = audit_reasoning(*fixtures())

        self.assertTrue(result["manual_review_required"])
        self.assertIn(
            "input_tracking_quality_warning",
            result["review_items"][0]["reasons"],
        )

    def test_rejects_wrong_packet_hash(self):
        packet, reasoning = fixtures()
        reasoning = copy.deepcopy(reasoning)
        reasoning["source_packet_sha256"] = "wrong"

        with self.assertRaises(ValueError):
            audit_reasoning(packet, reasoning)

    def test_rejects_tampered_assessment(self):
        packet, reasoning = fixtures()
        reasoning = copy.deepcopy(reasoning)
        reasoning["assessments"][0][
            "physical_collision_verified"
        ] = True

        with self.assertRaises(ValueError):
            audit_reasoning(packet, reasoning)

    def test_rejects_missing_assessment(self):
        packet, reasoning = fixtures()
        reasoning = copy.deepcopy(reasoning)
        reasoning["assessments"] = []
        reasoning["assessment_count"] = 0

        with self.assertRaises(ValueError):
            audit_reasoning(packet, reasoning)


if __name__ == "__main__":
    unittest.main()
