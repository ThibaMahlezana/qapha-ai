
import copy
import unittest

from app.evaluation.event_annotations import (
    create_annotation_template,
    digest,
    validate_annotations,
)


def fixture():
    body = {
        "schema": "qapha.muse_evidence_packet.v1",
        "stage": "calibration_only",
        "object_id": "0",
        "cutoff_frame": 49,
        "evidence_records": [
            {"frame_index": 36},
            {"frame_index": 42},
            {"frame_index": 49},
        ],
    }
    return {**body, "packet_sha256": digest(body)}


class EventAnnotationTests(unittest.TestCase):

    def test_creates_template(self):
        data = create_annotation_template(fixture())

        self.assertEqual(
            [item["frame_index"] for item in data["annotation_frames"]],
            [36, 42, 49],
        )
        self.assertEqual(data["stage"], "review_pending")

    def test_reviewer_sees_no_muse_conclusions(self):
        data = create_annotation_template(fixture())
        content = str(data["annotation_frames"])

        self.assertNotIn("possible_surface_interaction", content)
        self.assertNotIn("model_conclusion", content)

    def test_detects_tampered_packet(self):
        packet = fixture()
        packet["cutoff_frame"] = 70

        with self.assertRaises(ValueError):
            create_annotation_template(packet)

    def test_rejects_incomplete_review(self):
        data = create_annotation_template(fixture())
        data["annotation_frames"][0]["reviewed"] = True

        with self.assertRaises(ValueError):
            validate_annotations(data)

    def test_accepts_completed_review(self):
        data = create_annotation_template(fixture())

        for item in data["annotation_frames"]:
            item["reviewed"] = True
            item["motion_label"] = "uncertain"
            item["contact_visibility"] = "occluded_or_ambiguous"
            item["tracking_quality_issue_visible"] = False
            item["reviewer_notes"] = "Unable to establish contact."

        result = validate_annotations(data)

        self.assertTrue(result["complete"])
        self.assertEqual(result["completed"], 3)


if __name__ == "__main__":
    unittest.main()
