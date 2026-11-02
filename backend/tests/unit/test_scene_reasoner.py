
"""Offline tests for constrained scene reasoning."""

import json
import unittest
from types import SimpleNamespace

from app.reasoning.scene_reasoner import (
    prepare_evidence,
    validate_proposal,
    reason,
)


def graph():
    return {
        "schema": "qapha.scene_graph.v1",
        "frame_index": 49,
        "tracked_objects": [
            {"object_id": "2"}
        ],
        "reviewed_edges": [
            {"edge_id": "edge_1"}
        ],
        "relationships": [
            {
                "subject": "2",
                "object": "edge_1",
                "predicate": "image_proximity_to_edge",
                "distance_px": 28.126,
                "threshold_px": 30.0,
                "physical_contact_verified": False,
                "evidence_frame_index": 49,
            }
        ],
    }


def proposal():
    return {
        "hypotheses": [
            {
                "object_id": "2",
                "edge_id": "edge_1",
                "hypothesis": "proximity_only",
                "confidence": 0.8,
                "evidence_frame_index": 49,
                "reason": "Only image proximity is measured.",
                "next_observation": (
                    "Compare positions over consecutive frames."
                ),
            }
        ],
    }


class FakeClient:
    def __init__(self, result):
        self.result = result
        self.calls = []

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
                        content=json.dumps(self.result)
                    )
                )
            ]
        )


class SceneReasonerTests(unittest.TestCase):

    def test_evidence_references_real_nodes(self):
        evidence = prepare_evidence(graph())

        self.assertEqual(
            evidence["relationships"][0]["object_id"],
            "2",
        )

        self.assertFalse(
            evidence["relationships"][0]["contact_verified"]
        )

    def test_rejects_invented_object(self):
        bad = proposal()
        bad["hypotheses"][0]["object_id"] = "99"

        with self.assertRaises(ValueError):
            validate_proposal(
                bad,
                prepare_evidence(graph()),
            )

    def test_rejects_unsupported_hypothesis(self):
        bad = proposal()
        bad["hypotheses"][0]["hypothesis"] = "confirmed_collision"

        with self.assertRaises(ValueError):
            validate_proposal(
                bad,
                prepare_evidence(graph()),
            )

    def test_rejects_missing_or_bad_confidence(self):
        bad = proposal()
        bad["hypotheses"][0]["confidence"] = 1.5

        with self.assertRaises(ValueError):
            validate_proposal(
                bad,
                prepare_evidence(graph()),
            )

    def test_mocked_muse_request(self):
        client = FakeClient(proposal())

        result = reason(graph(), client=client)

        self.assertEqual(
            result["status"],
            "unverified_hypotheses",
        )

        self.assertEqual(
            len(result["hypotheses"]), 1
        )

        call = client.calls[0]

        self.assertEqual(
            call["model"], "muse-spark-1.3"
        )

        self.assertEqual(
            call["response_format"]["type"],
            "json_schema",
        )

    def test_empty_relationships_no_api_call(self):
        empty = graph()
        empty["relationships"] = []

        result = reason(empty)

        self.assertEqual(
            result["hypotheses"], []
        )


if __name__ == "__main__":
    unittest.main()
