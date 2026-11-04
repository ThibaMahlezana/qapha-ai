import json
import tempfile
import unittest
from pathlib import Path

from app.experiments.record_intake_deviation import (
    archive_attempt,
    digest,
    file_sha256,
    record_existing_archive,
)


def sealed(path, body):
    Path(path).write_text(json.dumps(
        {**body, "registration_sha256": digest(body)}), encoding="utf-8")


class DeviationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        sealed(self.dir / "reg.json", {"id": "e"})
        sealed(self.dir / "add.json", {"id": "e-add"})
        (self.dir / "heldout_video_e.mp4").write_bytes(b"video")
        (self.dir / "heldout_video_e_protocol.json").write_text("{}")
        (self.dir / "heldout_video_d.mp4").write_bytes(b"other")
        (self.dir / "heldout_video_e2.mp4").write_bytes(b"not-matched")

    def tearDown(self):
        self.tmp.cleanup()

    def run_archive(self, reg="reg.json"):
        return archive_attempt(self.dir, "heldout_video_e", "attempt1",
                               self.dir / reg, self.dir / "add.json",
                               "target absent", ["preview.jpg"], "t")

    def test_moves_only_candidate_files(self):
        original = file_sha256(self.dir / "heldout_video_e.mp4")
        record = self.run_archive()
        self.assertEqual([e["file"] for e in record["archived_files"]],
                         ["heldout_video_e.mp4",
                          "heldout_video_e_protocol.json"])
        self.assertFalse((self.dir / "heldout_video_e.mp4").exists())
        self.assertEqual(file_sha256(self.dir / "attempt1" /
                                     "heldout_video_e.mp4"), original)
        self.assertTrue((self.dir / "heldout_video_d.mp4").exists())
        self.assertTrue((self.dir / "heldout_video_e2.mp4").exists())

    def test_record_is_sealed(self):
        record = self.run_archive()
        body = {k: v for k, v in record.items() if k != "record_sha256"}
        self.assertEqual(digest(body), record["record_sha256"])
        self.assertFalse(record["tracking_performed"])

    def test_refuses_when_tracks_exist(self):
        (self.dir / "heldout_video_e_calibration_tracks.json").write_text("{}")
        with self.assertRaises(ValueError):
            self.run_archive()
        self.assertTrue((self.dir / "heldout_video_e.mp4").exists())

    def test_refuses_tampered_registration(self):
        data = json.loads((self.dir / "reg.json").read_text())
        data["id"] = "changed"
        (self.dir / "reg.json").write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            self.run_archive()
        self.assertTrue((self.dir / "heldout_video_e.mp4").exists())

    def test_refuses_existing_archive(self):
        (self.dir / "attempt1").mkdir()
        with self.assertRaises(FileExistsError):
            self.run_archive()

    def test_record_written_before_files_move(self):
        seen = {}

        def write(record):
            seen["archive_existed"] = (self.dir / "attempt1").exists()
            seen["video_in_place"] = (self.dir / "heldout_video_e.mp4").exists()

        archive_attempt(self.dir, "heldout_video_e", "attempt1",
                        self.dir / "reg.json", self.dir / "add.json",
                        "r", [], "t", write_record=write)
        self.assertEqual(seen, {"archive_existed": False,
                                "video_in_place": True})


class ExistingArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        sealed(self.dir / "reg.json", {"id": "e"})
        sealed(self.dir / "add.json", {"id": "e-add"})
        self.archive = self.dir / "attempt1"
        self.archive.mkdir()
        video = self.archive / "heldout_video_e.mp4"
        video.write_bytes(b"original")
        clip = self.archive / "heldout_video_e_calibration.mp4"
        clip.write_bytes(b"clip")
        (self.archive / "heldout_video_e_protocol.json").write_text(
            json.dumps({"video": {"sha256": file_sha256(video)}}),
            encoding="utf-8")
        (self.archive / "heldout_video_e_calibration_manifest.json").write_text(
            json.dumps({"source_sha256": file_sha256(video),
                        "calibration_video": "C:/old/place/" + clip.name,
                        "calibration_sha256": file_sha256(clip)}),
            encoding="utf-8")
        (self.dir / "heldout_video_e.mp4").write_bytes(b"replacement")

    def tearDown(self):
        self.tmp.cleanup()

    def run_record(self):
        return record_existing_archive(
            self.dir, "heldout_video_e", "attempt1", self.dir / "reg.json",
            self.dir / "add.json", "r", [], "t")

    def test_verifies_and_records_replacement(self):
        record = self.run_record()
        self.assertTrue(all(
            record["integrity_checks_against_intake_hashes"].values()))
        self.assertEqual(record["archive_method"],
                         "moved_manually_before_record_sealed")
        self.assertEqual(record["replacement_candidate_sha256_at_record_time"],
                         file_sha256(self.dir / "heldout_video_e.mp4"))
        body = {k: v for k, v in record.items() if k != "record_sha256"}
        self.assertEqual(digest(body), record["record_sha256"])

    def test_rejects_modified_archived_video(self):
        (self.archive / "heldout_video_e.mp4").write_bytes(b"edited")
        with self.assertRaises(ValueError):
            self.run_record()

    def test_rejects_tracking_artifacts(self):
        (self.dir / "heldout_video_e_calibration_tracks.json").write_text("{}")
        with self.assertRaises(ValueError):
            self.run_record()


if __name__ == "__main__":
    unittest.main()