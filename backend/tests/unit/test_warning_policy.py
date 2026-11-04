import json
import unittest
from pathlib import Path

from app.learning.warning_policy import (
    _digest,
    apply_policy,
    evaluate,
    load_registration,
)

RECORDED = Path(__file__).resolve().parents[3] / "datasets" / "recorded"
VIDEO_C = RECORDED / "heldout_video_c_calibration_tracks.json"
VIDEO_D = RECORDED / "heldout_video_d_calibration_tracks.json"
REG_D = RECORDED / "video_d_policy_preregistration.json"
REG_E = RECORDED / "video_e_policy_preregistration.json"

DEFAULTS = {
    "window": 6, "decay": 0.85, "recent_window": 5,
    "absolute_threshold_px": 15.0, "relative_factor": 3.0,
    "minimum_baseline_px": 2.0, "override_duration_frames": 3,
}


def event(frame, adaptive, persistence=None, linear=None,
          selection="linear"):
    return {
        "frame_index": frame,
        "history_cutoff_frame": frame - 1,
        "selection": selection,
        "errors_px": {
            "adaptive": adaptive,
            "persistence": adaptive if persistence is None else persistence,
            "linear": adaptive if linear is None else linear,
        },
    }


class SyntheticPolicyTests(unittest.TestCase):
    def test_no_warning_keeps_baseline(self):
        events = [event(f, 1.0) for f in range(2, 20)]
        records = apply_policy(events)
        self.assertFalse(any(r["warning"] for r in records))
        self.assertTrue(all(r["candidate_model"] == "linear"
                            for r in records))

    def test_warning_overrides_three_frames_after_spike(self):
        errs = [1.0] * 8 + [100.0] + [1.0] * 6
        events = [event(2 + i, e, persistence=0.5)
                  for i, e in enumerate(errs)]
        records = apply_policy(events)
        spike_frame = 2 + 8
        warned = [r["frame"] for r in records if r["warning"]]
        self.assertEqual(warned, [spike_frame + 1])
        overridden = [r["frame"] for r in records if r["override_active"]]
        self.assertEqual(overridden,
                         [spike_frame + 1, spike_frame + 2, spike_frame + 3])

    def test_warning_never_uses_current_error(self):
        errs = [1.0] * 8 + [100.0]
        records = apply_policy([event(2 + i, e) for i, e in enumerate(errs)])
        self.assertFalse(records[-1]["warning"])

    def test_rejects_gap_in_frames(self):
        with self.assertRaises(ValueError):
            apply_policy([event(2, 1.0), event(4, 1.0)])

    def test_registration_tamper_detected(self):
        reg = {"baseline": {"name": "original_adaptive_selector"},
               "experimental_policy": {}}
        reg["registration_sha256"] = _digest(reg)
        reg["baseline"]["window"] = 7
        with self.assertRaises(ValueError):
            load_registration(reg)


@unittest.skipUnless(VIDEO_C.exists() and VIDEO_D.exists(),
                     "recorded tracks not available")
class HistoricalReproductionTests(unittest.TestCase):
    """The committed code must reproduce previously reported results."""

    def test_video_c_diagnostic(self):
        result = evaluate(json.loads(VIDEO_C.read_text(encoding="utf-8")), DEFAULTS)
        self.assertEqual(result["evaluated_predictions"], 48)
        self.assertEqual(result["warning_frames"], [36])
        self.assertEqual(result["decisions_changed"], [36, 37, 38])
        self.assertEqual(result["baseline"]["mean_error_px"], 11.482721)
        self.assertEqual(result["candidate"]["mean_error_px"], 9.261511)

    def test_video_d_evaluation(self):
        params = load_registration(json.loads(REG_D.read_text(encoding="utf-8")))
        result = evaluate(json.loads(VIDEO_D.read_text(encoding="utf-8")), params)
        self.assertEqual(result["warning_frames"], [])
        self.assertFalse(result["policy_activated"])
        self.assertEqual(result["baseline"]["mean_error_px"], 0.010417)
        self.assertEqual(result["candidate"]["mean_error_px"], 0.010417)

    @unittest.skipUnless(REG_E.exists(), "Video E registration missing")
    def test_video_e_registration_parameters_match(self):
        params = load_registration(json.loads(REG_E.read_text(encoding="utf-8")))
        self.assertEqual(
            params["registration_sha256"],
            "88367a5bca1c50f8c7a99b08b2ee553a4567ce1a5d9d9d6de9daa73f82f6bf71")
        self.assertEqual({k: params[k] for k in DEFAULTS}, DEFAULTS)


if __name__ == "__main__":
    unittest.main()