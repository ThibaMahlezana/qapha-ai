import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from app.experiments import registered_policy_evaluation as rpe

RECORDED = Path(__file__).resolve().parents[3] / "datasets" / "recorded"
VIDEO_C = RECORDED / "heldout_video_c_calibration_tracks.json"
VIDEO_D = RECORDED / "heldout_video_d_calibration_tracks.json"
INHERITED = ("adaptive_loop", "adaptive_session",
             "prediction_protocol", "reliability_monitor")


def seal(body, key):
    return {**body, key: rpe.digest(body)}


def make_documents(code_root=rpe.CODE_ROOT):
    code = {n: rpe.file_sha256(Path(code_root) / p)
            for n, p in rpe.CODE_PATHS.items()}
    registration = seal({
        "experiment_id": "test",
        "video": "v.mp4",
        "frame_range_inclusive": [0, 49],
        "prediction_targets_inclusive": [2, 49],
        "tracking_requirements": {"unique_object_ids": 1,
                                  "required_frames": 50},
        "informativeness": {"minimum_total_path_length_px": 50.0,
                            "minimum_maximum_step_px": 5.0},
        "baseline": {"name": "original_adaptive_selector",
                     "window": 6, "decay": 0.85},
        "experimental_policy": {
            "name": "warning_triggered_persistence",
            "recent_window": 5, "absolute_threshold_px": 15.0,
            "relative_factor": 3.0, "minimum_baseline_px": 2.0,
            "override_model": "persistence",
            "override_duration_frames": 3},
        "inherited_implementation_sha256": {n: code[n] for n in INHERITED},
    }, "registration_sha256")
    addendum = seal({
        "amends_registration_sha256": registration["registration_sha256"],
        "locked_implementation_sha256": {
            "warning_policy": code["warning_policy"],
            "reference_baselines": code["reference_baselines"]},
    }, "registration_sha256")
    manifest = {"calibration_sha256": "clip"}
    target = seal({"registration_sha256": registration["registration_sha256"],
                   "calibration_video_sha256": "clip"}, "record_sha256")
    return registration, addendum, target, manifest


@unittest.skipUnless(VIDEO_C.exists() and VIDEO_D.exists(),
                     "recorded tracks not available")
class RegisteredEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.docs = make_documents()
        self.c = json.loads(VIDEO_C.read_text(encoding="utf-8"))
        self.d = json.loads(VIDEO_D.read_text(encoding="utf-8"))

    def test_reproduces_video_c_through_frozen_protocol(self):
        report = rpe.evaluate(*self.docs, self.c)
        self.assertEqual(report["outcome"], "evaluated")
        self.assertEqual(report["evaluated_predictions"], 48)
        models = report["all_models"]
        self.assertEqual(models["original_adaptive_selector"]["mean_error_px"],
                         11.482721)
        self.assertEqual(models["warning_triggered_persistence"]["mean_error_px"],
                         9.261511)
        self.assertEqual(models["fixed_persistence"]["mean_error_px"], 13.387468)
        self.assertEqual(models["fixed_linear_window6"]["mean_error_px"],
                         13.54104)
        self.assertEqual(models["constant_velocity_2pt"]["mean_error_px"],
                         9.015013)
        self.assertEqual(report["primary"]["warning_frames"], [36])
        self.assertEqual(report["final_learning_state_sha256"],
                         "77206fa7fde18dc30ef8be6c7bfb1bf8671a1f436472d6bc2a1d8113bff81040")

    def test_box_order_does_not_matter(self):
        shuffled = copy.deepcopy(self.c)
        shuffled["boxes"].reverse()
        a = rpe.evaluate(*self.docs, self.c)
        b = rpe.evaluate(*self.docs, shuffled)
        self.assertEqual(a["all_models"], b["all_models"])

    def test_stationary_video_is_non_informative_but_reported(self):
        report = rpe.evaluate(*self.docs, self.d)
        self.assertFalse(report["informativeness"]["informative"])
        self.assertEqual(report["outcome_label"], "report_non_informative")
        self.assertEqual(report["outcome"], "evaluated")

    def test_missing_frame_stops_as_incomplete(self):
        broken = copy.deepcopy(self.c)
        del broken["boxes"][10]
        report = rpe.evaluate(*self.docs, broken)
        self.assertEqual(report["outcome"], "report_incomplete")
        self.assertEqual(report["tracking"]["missing_frames"], [10])
        self.assertNotIn("all_models", report)

    def test_second_object_stops_as_incomplete(self):
        broken = copy.deepcopy(self.c)
        broken["boxes"][5]["object_id"] = "1"
        report = rpe.evaluate(*self.docs, broken)
        self.assertEqual(report["outcome"], "report_incomplete")


class IntegrityTests(unittest.TestCase):
    def test_changed_locked_code_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in rpe.CODE_PATHS.values():
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(rpe.CODE_ROOT / rel, root / rel)
            docs = make_documents(root)
            rpe.verify_integrity(*docs, code_root=root)
            with (root / rpe.CODE_PATHS["warning_policy"]).open("a") as f:
                f.write("\n# edited\n")
            with self.assertRaises(ValueError):
                rpe.verify_integrity(*docs, code_root=root)

    @unittest.skipUnless(VIDEO_C.exists(), "recorded tracks not available")
    def test_concept_must_match_target_record(self):
        registration, addendum, target, manifest = make_documents()
        target = seal({k: v for k, v in target.items() if k != "record_sha256"}
                      | {"sam_concept": "tennis ball"}, "record_sha256")
        tracks = json.loads(VIDEO_C.read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            rpe.evaluate(registration, addendum, target, manifest, tracks)

    def test_target_for_other_clip_rejected(self):
        registration, addendum, target, manifest = make_documents()
        with self.assertRaises(ValueError):
            rpe.verify_integrity(registration, addendum, target,
                                 {"calibration_sha256": "other"})


if __name__ == "__main__":
    unittest.main()