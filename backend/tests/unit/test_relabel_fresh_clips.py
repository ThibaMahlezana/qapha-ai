import tempfile
import unittest
from pathlib import Path

from app.experiments.register_fresh_clips import build_registry
from app.experiments.relabel_fresh_clips import relabel


class RelabelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "WhatsApp A.mp4").write_bytes(b"clip-a")
        (self.dir / "WhatsApp B.mp4").write_bytes(b"clip-b")
        self.previous = build_registry(self.dir, "t0")

    def tearDown(self):
        self.tmp.cleanup()

    def rename(self, old, new):
        (self.dir / old).rename(self.dir / new)

    def test_renamed_clips_keep_ids_and_link_back(self):
        self.rename("WhatsApp A.mp4", "drop_050cm.mp4")
        self.rename("WhatsApp B.mp4", "still_control.mp4")
        reg = relabel(self.previous, self.dir, "t1", "note")
        by_role = {c["role"]: c for c in reg["clips"]}
        self.assertEqual(by_role["drop_050cm"]["clip_id"], "F01")
        self.assertEqual(by_role["drop_050cm"]["previous_file"], "WhatsApp A.mp4")
        self.assertEqual(by_role["still_control"]["clip_id"], "F02")
        self.assertEqual(reg["supersedes_registry_sha256"],
                         self.previous["registry_sha256"])
        self.assertFalse(reg["viewing"]["viewed_by_method_developer"])

    def test_edited_clip_rejected(self):
        self.rename("WhatsApp A.mp4", "drop_050cm.mp4")
        self.rename("WhatsApp B.mp4", "still_control.mp4")
        (self.dir / "drop_050cm.mp4").write_bytes(b"clip-a-edited")
        with self.assertRaises(ValueError):
            relabel(self.previous, self.dir, "t1", "note")

    def test_missing_clip_rejected(self):
        self.rename("WhatsApp A.mp4", "drop_050cm.mp4")
        (self.dir / "WhatsApp B.mp4").unlink()
        with self.assertRaises(ValueError):
            relabel(self.previous, self.dir, "t1", "note")

    def test_added_clip_rejected(self):
        self.rename("WhatsApp A.mp4", "drop_050cm.mp4")
        self.rename("WhatsApp B.mp4", "still_control.mp4")
        (self.dir / "extra_clip.mp4").write_bytes(b"new")
        with self.assertRaises(ValueError):
            relabel(self.previous, self.dir, "t1", "note")

    def test_bad_name_rejected(self):
        self.rename("WhatsApp A.mp4", "Drop 50cm.mp4")
        self.rename("WhatsApp B.mp4", "still_control.mp4")
        with self.assertRaises(ValueError):
            relabel(self.previous, self.dir, "t1", "note")

    def test_tampered_previous_registry_rejected(self):
        self.previous["clips"][0]["file"] = "other.mp4"
        with self.assertRaises(ValueError):
            relabel(self.previous, self.dir, "t1", "note")


if __name__ == "__main__":
    unittest.main()