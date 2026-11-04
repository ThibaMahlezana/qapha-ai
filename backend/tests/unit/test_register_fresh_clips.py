import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.experiments.register_fresh_clips import build_registry, digest


def write_clip(path, frames):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             25.0, (64, 48))
    for i in range(frames):
        writer.write(np.full((48, 64, 3), i, dtype=np.uint8))
    writer.release()


class FreshClipRegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        write_clip(self.dir / "b_long.mp4", 80)
        write_clip(self.dir / "a_short.mp4", 30)
        (self.dir / "notes.txt").write_text("ignored")
        (self.dir / "broken.mp4").write_bytes(b"not a video")

    def tearDown(self):
        self.tmp.cleanup()

    def test_registers_videos_in_name_order(self):
        reg = build_registry(self.dir, "t")
        self.assertEqual([c["file"] for c in reg["clips"]],
                         ["a_short.mp4", "b_long.mp4", "broken.mp4"])
        self.assertEqual([c["clip_id"] for c in reg["clips"]],
                         ["F01", "F02", "F03"])

    def test_length_and_readability_flags(self):
        clips = {c["file"]: c for c in build_registry(self.dir, "t")["clips"]}
        self.assertFalse(clips["a_short.mp4"]["meets_minimum_length"])
        self.assertTrue(clips["b_long.mp4"]["meets_minimum_length"])
        self.assertFalse(clips["broken.mp4"]["readable"])

    def test_sealed_and_records_no_viewing(self):
        reg = build_registry(self.dir, "t")
        body = {k: v for k, v in reg.items() if k != "registry_sha256"}
        self.assertEqual(digest(body), reg["registry_sha256"])
        self.assertEqual(reg["frames_viewed_at_registration"], 0)

    def test_fingerprint_changes_if_clip_changes(self):
        before = build_registry(self.dir, "t")["clips"][1]["sha256"]
        write_clip(self.dir / "b_long.mp4", 81)
        after = build_registry(self.dir, "t")["clips"][1]["sha256"]
        self.assertNotEqual(before, after)

    def test_custom_id_prefix_and_role(self):
        reg = build_registry(self.dir, "t", id_prefix="H", note="originals")
        self.assertEqual([c["clip_id"] for c in reg["clips"]], ["H01", "H02", "H03"])
        self.assertEqual(reg["clips"][1]["role"], "b_long")
        self.assertEqual(reg["note"], "originals")

    def test_bad_prefix_rejected(self):
        with self.assertRaises(ValueError):
            build_registry(self.dir, "t", id_prefix="h1")

    def test_empty_folder_rejected(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(ValueError):
                build_registry(empty, "t")


if __name__ == "__main__":
    unittest.main()