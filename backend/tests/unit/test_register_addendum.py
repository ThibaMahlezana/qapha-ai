import tempfile
import unittest
from pathlib import Path

from app.experiments.register_addendum import (
    INHERITED,
    LOCKED,
    build_addendum,
    digest,
    file_sha256,
)

STAMP = "2026-11-04T07:00:00+00:00"


class AddendumTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.code = root / "backend"
        self.recorded = root / "recorded"
        self.recorded.mkdir()
        for rel in (*INHERITED.values(), *LOCKED.values()):
            path = self.code / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(rel.encode("utf-8"))
        body = {
            "experiment_id": "video_e_uncertainty_response",
            "video": "heldout_video_e.mp4",
            "inherited_implementation_sha256": {
                name: file_sha256(self.code / rel)
                for name, rel in INHERITED.items()},
        }
        self.original = {**body, "registration_sha256": digest(body)}

    def tearDown(self):
        self.tmp.cleanup()

    def build(self):
        return build_addendum(self.original, self.code, self.recorded, STAMP)

    def test_sealed_and_linked_to_original(self):
        result = self.build()
        body = {k: v for k, v in result.items()
                if k != "registration_sha256"}
        self.assertEqual(digest(body), result["registration_sha256"])
        self.assertEqual(result["amends_registration_sha256"],
                         self.original["registration_sha256"])
        self.assertFalse(result["source_video_present_at_registration"])

    def test_deterministic(self):
        self.assertEqual(self.build(), self.build())

    def test_locks_policy_hash(self):
        result = self.build()
        self.assertEqual(
            result["locked_implementation_sha256"]["warning_policy"],
            file_sha256(self.code / LOCKED["warning_policy"]))

    def test_records_video_hash_when_present(self):
        (self.recorded / "heldout_video_e.mp4").write_bytes(b"video")
        result = self.build()
        self.assertTrue(result["source_video_present_at_registration"])
        self.assertEqual(result["source_video_sha256"],
                         file_sha256(self.recorded / "heldout_video_e.mp4"))

    def test_tampered_original_rejected(self):
        self.original["video"] = "other.mp4"
        with self.assertRaises(ValueError):
            self.build()

    def test_changed_inherited_module_rejected(self):
        (self.code / INHERITED["adaptive_loop"]).write_bytes(b"edited")
        with self.assertRaises(ValueError):
            self.build()

    def test_refuses_after_tracking_exists(self):
        (self.recorded / "heldout_video_e_calibration_tracks.json").write_text("{}")
        with self.assertRaises(ValueError):
            self.build()


if __name__ == "__main__":
    unittest.main()