"""Run a registered warning-policy experiment in its registered order.

Order (each stage is recorded even if a later stage is not reached):
  1. Integrity: registration, addendum, locked code, target record.
  2. Tracking requirements (one object, every registered frame).
  3. Informativeness, judged on motion only, before any forecast.
  4. Frozen one-step predictions via prediction_protocol
     (freeze_next_prediction -> observe_and_update), then the
     registered policy and the addendum's secondary baselines.
Results are written once and never overwritten; unfavourable outcomes
are reported exactly like favourable ones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from app.learning.prediction_protocol import (
    create_protocol,
    freeze_next_prediction,
    observe_and_update,
)
from app.learning.reference_baselines import evaluate_baselines
from app.learning.warning_policy import apply_policy, load_registration


SCHEMA = "qapha.registered_policy_evaluation.v1"
CODE_ROOT = Path(__file__).resolve().parents[2]
CODE_PATHS = {
    "adaptive_loop": "app/learning/adaptive_loop.py",
    "adaptive_session": "app/learning/adaptive_session.py",
    "prediction_protocol": "app/learning/prediction_protocol.py",
    "reliability_monitor": "app/learning/reliability_monitor.py",
    "warning_policy": "app/learning/warning_policy.py",
    "reference_baselines": "app/learning/reference_baselines.py",
}


def digest(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _sealed(data, key):
    body = {k: v for k, v in data.items() if k != key}
    return digest(body) == data.get(key)


def verify_integrity(registration, addendum, target, manifest,
                     code_root=CODE_ROOT):
    if not _sealed(registration, "registration_sha256"):
        raise ValueError("Registration integrity check failed")
    if not _sealed(addendum, "registration_sha256"):
        raise ValueError("Addendum integrity check failed")
    if addendum["amends_registration_sha256"] != registration["registration_sha256"]:
        raise ValueError("Addendum does not amend this registration")
    if not _sealed(target, "record_sha256"):
        raise ValueError("Target record integrity check failed")
    if target["registration_sha256"] != registration["registration_sha256"]:
        raise ValueError("Target record belongs to another registration")
    if target["calibration_video_sha256"] != manifest["calibration_sha256"]:
        raise ValueError("Target record refers to another calibration clip")

    expected = {**registration["inherited_implementation_sha256"],
                **addendum["locked_implementation_sha256"]}
    actual = {name: file_sha256(Path(code_root) / rel)
              for name, rel in CODE_PATHS.items()}
    mismatched = sorted(n for n in CODE_PATHS if actual[n] != expected[n])
    if mismatched:
        raise ValueError(f"Code differs from registered hashes: {mismatched}")
    return {"code_sha256": actual,
            "registration_sha256": registration["registration_sha256"],
            "addendum_sha256": addendum["registration_sha256"],
            "target_record_sha256": target["record_sha256"]}


def check_tracking(tracks, registration):
    req = registration["tracking_requirements"]
    first, last = registration["frame_range_inclusive"]
    boxes = tracks.get("boxes", [])
    ids = sorted({str(b.get("object_id")) for b in boxes})
    frames = sorted(b.get("frame_index") for b in boxes)
    wanted = list(range(first, last + 1))
    result = {
        "model": tracks.get("model"),
        "concept": tracks.get("concept"),
        "unique_object_ids": len(ids),
        "frames_present": len(set(frames)),
        "duplicate_frames": len(frames) - len(set(frames)),
        "missing_frames": sorted(set(wanted) - set(frames)),
    }
    result["complete"] = (
        tracks.get("model") == "sam-3.1"
        and len(ids) == req["unique_object_ids"]
        and result["duplicate_frames"] == 0
        and not result["missing_frames"]
        and len(set(frames)) == req["required_frames"]
    )
    return result


def ordered_tracks(tracks):
    """Copy of tracks with boxes sorted by frame (values unchanged)."""
    return {**tracks, "boxes": sorted(tracks["boxes"],
                                      key=lambda b: b["frame_index"])}


def centres(tracks):
    out = []
    for b in tracks["boxes"]:
        x0, y0, x1, y1 = b["bbox_xyxy_px"]
        out.append(((x0 + x1) / 2, (y0 + y1) / 2))
    return out


def check_informativeness(points, registration):
    crit = registration["informativeness"]
    steps = [math.hypot(b[0] - a[0], b[1] - a[1])
             for a, b in zip(points, points[1:])]
    path = sum(steps)
    largest = max(steps) if steps else 0.0
    return {
        "total_path_length_px": round(path, 6),
        "maximum_step_px": round(largest, 6),
        "minimum_total_path_length_px": crit["minimum_total_path_length_px"],
        "minimum_maximum_step_px": crit["minimum_maximum_step_px"],
        "informative": (path >= crit["minimum_total_path_length_px"]
                        and largest >= crit["minimum_maximum_step_px"]),
    }


def frozen_sequence(points, object_id, window, decay):
    """Freeze each prediction before its observation is revealed."""
    protocol = create_protocol(object_id, window, decay)
    events, fingerprints = [], []
    for frame, xy in enumerate(points):
        pending = None
        if frame >= 2:
            protocol, pending = freeze_next_prediction(protocol)
        protocol, evaluation = observe_and_update(protocol, frame, list(xy))
        if pending is not None:
            fingerprints.append(pending["prediction_sha256"])
            events.append({
                "frame_index": frame,
                "history_cutoff_frame": frame - 1,
                "selection": evaluation["selection"],
                "errors_px": evaluation["errors_px"],
                "prediction_sha256": pending["prediction_sha256"],
            })
    return events, protocol["session"]["state_sha256"]


def _stats(errors):
    return {"mean_error_px": round(sum(errors) / len(errors), 6),
            "maximum_error_px": round(max(errors), 6),
            "errors_over_25px": sum(e > 25 for e in errors)}


def evaluate(registration, addendum, target, manifest, tracks,
             code_root=CODE_ROOT):
    report = {"schema": SCHEMA,
              "experiment_id": registration["experiment_id"],
              "integrity": verify_integrity(registration, addendum, target,
                                            manifest, code_root)}

    if "sam_concept" in target and tracks.get("concept") != target["sam_concept"]:
        raise ValueError("Tracks were not produced with the recorded concept")

    tracking = check_tracking(tracks, registration)
    report["tracking"] = tracking
    if not tracking["complete"]:
        report["outcome"] = "report_incomplete"
        return report

    tracks = ordered_tracks(tracks)
    points = centres(tracks)
    report["informativeness"] = check_informativeness(points, registration)
    if not report["informativeness"]["informative"]:
        report["outcome_label"] = "report_non_informative"

    params = load_registration(registration)
    events, final_state = frozen_sequence(
        points, str(tracks["boxes"][0]["object_id"]),
        params["window"], params["decay"])
    first, last = registration["prediction_targets_inclusive"]
    if [e["frame_index"] for e in events] != list(range(first, last + 1)):
        raise RuntimeError("Prediction targets differ from registration")

    records = apply_policy(
        events,
        recent_window=params["recent_window"],
        absolute_threshold_px=params["absolute_threshold_px"],
        relative_factor=params["relative_factor"],
        minimum_baseline_px=params["minimum_baseline_px"],
        override_duration_frames=params["override_duration_frames"],
    )
    baseline = _stats([r["baseline_error_px"] for r in records])
    candidate = _stats([r["candidate_error_px"] for r in records])
    secondary = evaluate_baselines(tracks)
    models = {
        "original_adaptive_selector": baseline,
        "warning_triggered_persistence": candidate,
        "fixed_persistence": _stats([e["errors_px"]["persistence"]
                                     for e in events]),
        "fixed_linear_window6": _stats([e["errors_px"]["linear"]
                                        for e in events]),
        "constant_velocity_2pt": {k: secondary["constant_velocity_2pt"][k]
                                  for k in ("mean_error_px",
                                            "maximum_error_px",
                                            "errors_over_25px")},
        "alpha_beta": {k: secondary["alpha_beta"][k]
                       for k in ("mean_error_px", "maximum_error_px",
                                 "errors_over_25px")},
    }
    for i, r in enumerate(records):
        r["prediction_sha256"] = events[i]["prediction_sha256"]

    report.update({
        "outcome": "evaluated",
        "evaluated_predictions": len(records),
        "final_learning_state_sha256": final_state,
        "primary": {
            "metric": "mean_euclidean_error_px",
            "baseline": baseline,
            "candidate": candidate,
            "mean_error_difference_px": round(
                candidate["mean_error_px"] - baseline["mean_error_px"], 6),
            "warning_frames": [r["frame"] for r in records if r["warning"]],
            "policy_overrides": [r["frame"] for r in records
                                 if r["candidate_model"] != r["baseline_model"]],
        },
        "all_models": models,
        "exploratory_expectations": {
            "E1_cv2pt_beats_adaptive":
                models["constant_velocity_2pt"]["mean_error_px"]
                < baseline["mean_error_px"],
            "E2_policy_beats_adaptive":
                candidate["mean_error_px"] < baseline["mean_error_px"],
        },
        "records": records,
        "limitations": [
            "One video; no generalization is established.",
            "Pixel errors against SAM box centres, not physical ground truth.",
            "Policy selects among frozen baseline forecasts; it does not "
            "change the learner's updates.",
            "Perspective changes pixel scale across the clip.",
        ],
    })
    return report


def main():
    parser = argparse.ArgumentParser()
    for name in ("registration", "addendum", "target", "manifest",
                 "tracks", "output"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
    report = evaluate(load(args.registration), load(args.addendum),
                      load(args.target), load(args.manifest),
                      load(args.tracks))
    report["tracks_sha256"] = file_sha256(args.tracks)
    report["report_sha256"] = digest(report)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    summary = {k: report.get(k) for k in
               ("outcome", "outcome_label", "tracking", "informativeness",
                "evaluated_predictions", "primary", "all_models",
                "exploratory_expectations", "report_sha256")}
    print(json.dumps(summary, indent=2))
    print("Saved:", output)


if __name__ == "__main__":
    main()