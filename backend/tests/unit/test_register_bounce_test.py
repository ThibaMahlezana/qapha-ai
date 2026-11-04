import tempfile
import unittest
from pathlib import Path

from app.experiments import register_bounce_test as rbt


def registry(clips=None, analysed=False):
    clips = clips or [
        {"clip_id": "F01", "file": "drop_050cm.mp4", "role": "drop_050cm",
         "sha256": "a"},
        {"clip_id": "F06", "file": "roll_wall.mp4", "role": "roll_wall",
         "sha256": "b"},
    ]
    body = {"clips": clips,
            "viewing": {"tracking_or_analysis_performed": analysed}}
    return {**body, "registry_sha256": rbt.digest(body)}


class BounceRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.analysis = Path(self.tmp.name) / "analysis"

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, reg=None):
        return rbt.build_registration(reg or registry(), rbt.Path(rbt.__file__)
                                      .resolve().parents[2], self.analysis, "t")

    def test_sealed_and_bound_to_registry(self):
        reg = self.build()
        body = {k: v for k, v in reg.items() if k != "registration_sha256"}
        self.assertEqual(rbt.digest(body), reg["registration_sha256"])
        self.assertEqual(reg["clip_registry_sha256"],
                         registry()["registry_sha256"])

    def test_groups_rolling_clips_for_safety_check(self):
        groups = {c["role"]: c["group"] for c in self.build()["clips"]}
        self.assertEqual(groups["drop_050cm"], "bounce")
        self.assertEqual(groups["roll_wall"], "rolling_safety_check")

    def test_locks_code_hashes(self):
        code = self.build()["locked_code_sha256"]
        self.assertEqual(set(code), {"bounce_aware", "probabilistic"})
        self.assertTrue(all(len(h) == 64 for h in code.values()))

    def test_refuses_if_analysis_folder_has_files(self):
        self.analysis.mkdir()
        (self.analysis / "f01_tracks.json").write_text("{}")
        with self.assertRaises(ValueError):
            self.build()

    def test_refuses_already_analysed_registry(self):
        with self.assertRaises(ValueError):
            self.build(registry(analysed=True))

    def test_refuses_tampered_registry(self):
        reg = registry()
        reg["clips"][0]["role"] = "drop_999cm"
        with self.assertRaises(ValueError):
            self.build(reg)


if __name__ == "__main__":
    unittest.main()