import json
import shutil
import tempfile
import unittest
from pathlib import Path

from app.experiments import bounce_test_evaluation as bte


def box(cx, cy, half=10.0):
    return [cx - half, cy - half, cx + half, cy + half]


def bouncing(frames=90, floor_bottom=400.0, half=10.0, e=0.6):
    y, vy, out = 100.0, 0.0, []
    floor_centre = floor_bottom - half
    for t in range(frames):
        out.append(box(200 + t, y, half))
        vy += 3.0
        y += vy
        if y > floor_centre:
            y = floor_centre - (y - floor_centre) * e
            vy = -vy * e
    return out


def rolling(frames=90):
    return [box(50 + 4 * t, 300) for t in range(frames)]


def tracks(boxes, concept="ball", ids=None):
    return {"model": "sam-3.1", "concept": concept, "boxes": [
        {"object_id": (ids[i] if ids else "0"), "frame_index": i,
         "bbox_xyxy_px": b} for i, b in enumerate(boxes)]}


def seal(body, key):
    return {**body, key: bte.digest(body)}


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        clips = [
            {"clip_id": "F01", "role": "drop_a", "group": "bounce"},
            {"clip_id": "F02", "role": "drop_b", "group": "bounce"},
            {"clip_id": "F06", "role": "roll_wall", "group": "rolling_safety_check"},
            {"clip_id": "F07", "role": "missing_one", "group": "bounce"},
        ]
        code = {n: bte.file_sha256(bte.CODE_ROOT / rel)
                for n, rel in bte.CODE.items()}
        self.reg = seal({"clips": clips, "locked_code_sha256": code,
                         "analysis": {"frames_per_clip": "0..89 (first 90 frames)",
                                      "sam_concept": "ball"}},
                        "registration_sha256")
        self.manifest = seal({"registration_sha256":
                              self.reg["registration_sha256"]}, "manifest_sha256")
        self.write("drop_a", tracks(bouncing()))
        self.write("drop_b", tracks(bouncing(floor_bottom=380.0)))
        self.write("roll_wall", tracks(rolling()))

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, role, data):
        (self.dir / f"{role}_tracks.json").write_text(json.dumps(data),
                                                      encoding="utf-8")

    def run_eval(self, code_root=bte.CODE_ROOT):
        return bte.evaluate(self.reg, self.manifest, self.dir, code_root)

    def test_reports_every_clip_including_untracked(self):
        report = self.run_eval()
        by = {c["role"]: c for c in report["clips"]}
        self.assertEqual(len(by), 4)
        self.assertFalse(by["missing_one"]["eligible"])
        self.assertEqual(by["missing_one"]["reason"], "not tracked")
        self.assertEqual(report["eligible_clips"], "3 of 4")

    def test_hypotheses_are_computed(self):
        h = self.run_eval()["hypotheses"]
        self.assertTrue(h["H1_primary"]["testable"])
        self.assertIn("supported", h["H1_primary"])
        self.assertTrue(h["H2"]["testable"])
        self.assertTrue(h["H3_safety"]["testable"])
        self.assertTrue(h["H3_safety"]["per_clip"]["roll_wall"]["within_5_percent"])

    def test_wrong_concept_is_ineligible(self):
        self.write("drop_a", tracks(bouncing(), concept="tennis ball"))
        by = {c["role"]: c for c in self.run_eval()["clips"]}
        self.assertFalse(by["drop_a"]["eligible"])

    def test_two_objects_ineligible(self):
        ids = ["0"] * 89 + ["1"]
        self.write("drop_a", tracks(bouncing(), ids=ids))
        by = {c["role"]: c for c in self.run_eval()["clips"]}
        self.assertEqual(by["drop_a"]["reason"], "2 tracked objects")

    def test_missing_frames_ineligible(self):
        data = tracks(bouncing())
        del data["boxes"][40]
        self.write("drop_a", data)
        by = {c["role"]: c for c in self.run_eval()["clips"]}
        self.assertEqual(by["drop_a"]["reason"], "1 frames without the ball")

    def test_changed_code_refused(self):
        with tempfile.TemporaryDirectory() as root:
            for rel in bte.CODE.values():
                (Path(root) / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(bte.CODE_ROOT / rel, Path(root) / rel)
            (Path(root) / bte.CODE["bounce_aware"]).write_text("# edited",
                                                               encoding="utf-8")
            with self.assertRaises(ValueError):
                self.run_eval(code_root=root)

    def test_manifest_from_other_registration_refused(self):
        self.manifest = seal({"registration_sha256": "other"}, "manifest_sha256")
        with self.assertRaises(ValueError):
            self.run_eval()


if __name__ == "__main__":
    unittest.main()