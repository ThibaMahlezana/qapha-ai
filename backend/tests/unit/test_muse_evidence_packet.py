
import copy
import unittest

from app.reasoning.muse_evidence_packet import (
    build_packet,
    digest,
)


def fixtures():
    manifest = {
        "schema": "qapha.calibration_clip.v1",
        "frame_range_inclusive": [0, 49],
        "frames_verified": 50,
        "future_frames_included": False,
        "calibration_sha256": "test-calibration",
    }

    surface = {
        "schema": "qapha.surface_association.v1",
        "stage": "calibration_only",
        "object_id": "0",
        "cutoff_frame": 49,
        "calibration_sha256": "test-calibration",
    }

    geometry = {
        "schema": "qapha.geometry_contact_evidence.v1",
        "stage": "calibration_only",
        "object_id": "0",
        "cutoff_frame": 49,
        "calibration_sha256": "test-calibration",
        "physical_collision_verified": False,
        "visual_geometry": {
            "stable_edge": True,
            "candidate_visual_edge_y_px": 655.0,
            "edge_spread_px": 0,
            "physical_contact_plane_verified": False,
        },
        "object_geometry_relation": {
            "stable_bottom_y_px": 679.0,
            "vertical_offset_px": 24.0,
        },
    }

    hypotheses = {
        "schema": "qapha.contact_hypotheses.v1",
        "stage": "calibration_only",
        "object_id": "0",
        "cutoff_frame": 49,
        "calibration_sha256": "test-calibration",
        "hypothesis_count": 1,
        "physical_collision_verified": False,
        "hypotheses": [
            {
                "id": "candidate_reversal_frame_36",
                "type": "possible_surface_interaction",
                "frame_index": 36,
                "status": "hypothesis_not_verified",
                "physical_collision_verified": False,
                "observed_evidence": {
                    "vertical_direction_reversal": True,
                },
                "inference": "Possible environmental interaction.",
                "quality_warning": True,
                "quality_flags": [
                    {"requires_review": True}
                ],
                "limitations": [
                    "Contact is not physically verified."
                ],
            }
        ],
    }

    hypotheses["geometry_evidence_sha256"] = digest(geometry)
    hypotheses["surface_evidence_sha256"] = digest(surface)

    return hypotheses, geometry, surface, manifest


class MuseEvidencePacketTests(unittest.TestCase):

    def test_creates_evidence_packet(self):
        packet = build_packet(*fixtures())

        self.assertEqual(len(packet["evidence_records"]), 1)
        self.assertEqual(
            packet["allowed_evidence_ids"],
            ["candidate_reversal_frame_36"],
        )

        body = {
            key: value for key, value in packet.items()
            if key != "packet_sha256"
        }
        self.assertEqual(packet["packet_sha256"], digest(body))

    def test_rejects_tampered_geometry(self):
        values = list(fixtures())
        values[1] = copy.deepcopy(values[1])
        values[1]["visual_geometry"][
            "candidate_visual_edge_y_px"
        ] = 600.0

        with self.assertRaises(ValueError):
            build_packet(*values)

    def test_rejects_future_frames(self):
        values = list(fixtures())
        values[3] = copy.deepcopy(values[3])
        values[3]["future_frames_included"] = True

        with self.assertRaises(ValueError):
            build_packet(*values)

    def test_rejects_duplicate_evidence_id(self):
        values = list(fixtures())
        values[0] = copy.deepcopy(values[0])

        duplicated = copy.deepcopy(
            values[0]["hypotheses"][0]
        )
        values[0]["hypotheses"].append(duplicated)
        values[0]["hypothesis_count"] = 2

        with self.assertRaises(ValueError):
            build_packet(*values)

    def test_rejects_verified_collision_claim(self):
        values = list(fixtures())
        values[0] = copy.deepcopy(values[0])
        values[0]["physical_collision_verified"] = True

        with self.assertRaises(ValueError):
            build_packet(*values)


if __name__ == "__main__":
    unittest.main()
