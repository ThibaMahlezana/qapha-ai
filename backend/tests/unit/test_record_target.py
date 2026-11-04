import json
import tempfile
import unittest
from pathlib import Path

from app.experiments.record_target import (
    build_target_record,
    digest,
    file_sha256,
)


class TargetRecordTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        body = {"experiment_id": "e", "video": "heldout_video_e.mp4"}
        self.reg = self.dir / "reg.json"
        self.reg.write_text(json.dumps(
            {**body, "registration_sha256": digest(body)}), encoding="utf-8")
        clip = self.dir / "heldout_video_e_calibration.mp4"
        clip.write_bytes(b"clip")
        self.manifest = self.dir / "manifest.json"
        self.manifest.write_text(json.dumps({
            "calibration_video": str(clip),
            "calibration_sha256": file_sha256(clip),
            "source_sha256": "abc"}), encoding="utf-8")
        self.preview = self.dir / "heldout_video_e_preview.jpg"
        self.preview.write_bytes(b"jpg")

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, concept="soccer ball"):
        return build_target_record(self.reg, self.manifest, self.preview,
                                   "the ball", concept, [], "t")

    def test_sealed_record(self):
        record = self.build()
        body = {k: v for k, v in record.items() if k != "record_sha256"}
        self.assertEqual(digest(body), record["record_sha256"])
        self.assertEqual(record["sam_concept"], "soccer ball")
        self.assertEqual(record["preview_image_sha256"],
                         file_sha256(self.preview))

    def test_refuses_after_tracking(self):
        (self.dir / "heldout_video_e_calibration_tracks.json").write_text("{}")
        with self.assertRaises(ValueError):
            self.build()

    def test_refuses_modified_clip(self):
        (self.dir / "heldout_video_e_calibration.mp4").write_bytes(b"x")
        with self.assertRaises(ValueError):
            self.build()

    def test_requires_concept(self):
        with self.assertRaises(ValueError):
            self.build(concept="  ")


if __name__ == "__main__":
    unittest.main()