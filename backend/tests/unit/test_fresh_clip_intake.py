import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.experiments.fresh_clip_intake import digest, file_sha256, run_intake


def write_clip(path, frames, size=(64, 48)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             30.0, size)
    for i in range(frames):
        writer.write(np.full((size[1], size[0], 3), i % 255, dtype=np.uint8))
    writer.release()


def registration(clips, frames=20):
    body = {"clips": clips,
            "analysis": {"frames_per_clip": f"0..{frames - 1} (first {frames} frames)"}}
    return {**body, "registration_sha256": digest(body)}


class IntakeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.clips = root / "clips"
        self.analysis = root / "analysis"
        self.clips.mkdir()
        write_clip(self.clips / "drop_050cm.mp4", 40)
        write_clip(self.clips / "roll_wall.mp4", 10, size=(48, 64))
        self.entries = [
            {"clip_id": "F01", "file": "drop_050cm.mp4", "role": "drop_050cm",
             "group": "bounce", "sha256": file_sha256(self.clips / "drop_050cm.mp4")},
            {"clip_id": "F06", "file": "roll_wall.mp4", "role": "roll_wall",
             "group": "rolling_safety_check",
             "sha256": file_sha256(self.clips / "roll_wall.mp4")},
        ]

    def tearDown(self):
        self.tmp.cleanup()

    def test_cuts_window_and_reports_short_clips(self):
        m = run_intake(registration(self.entries), self.clips, self.analysis)
        by = {c["role"]: c for c in m["clips"]}
        self.assertEqual(by["drop_050cm"]["status"], "ok")
        self.assertEqual(by["drop_050cm"]["frames_written"], 20)
        self.assertEqual(by["roll_wall"]["status"], "short")
        self.assertTrue((self.analysis / "drop_050cm.mp4").exists())
        self.assertTrue((self.analysis / "drop_050cm_preview.jpg").exists())
        capture = cv2.VideoCapture(str(self.analysis / "drop_050cm.mp4"))
        count = 0
        while capture.read()[0]:
            count += 1
        capture.release()
        self.assertEqual(count, 20)

    def test_changed_clip_rejected(self):
        write_clip(self.clips / "drop_050cm.mp4", 41)
        with self.assertRaises(ValueError):
            run_intake(registration(self.entries), self.clips, self.analysis)

    def test_missing_clip_reported_not_skipped(self):
        (self.clips / "roll_wall.mp4").unlink()
        m = run_intake(registration(self.entries), self.clips, self.analysis)
        self.assertEqual({c["role"]: c["status"] for c in m["clips"]}["roll_wall"],
                         "missing")

    def test_never_overwrites(self):
        run_intake(registration(self.entries), self.clips, self.analysis)
        with self.assertRaises(FileExistsError):
            run_intake(registration(self.entries), self.clips, self.analysis)

    def test_tampered_registration_rejected(self):
        reg = registration(self.entries)
        reg["clips"][0]["role"] = "other"
        with self.assertRaises(ValueError):
            run_intake(reg, self.clips, self.analysis)


if __name__ == "__main__":
    unittest.main()