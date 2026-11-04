"""Registered warning-triggered persistence policy.

Implements the experimental policy of video_d/video_e preregistrations:
when the reliability monitor warns before a target frame, select the
persistence forecast for that frame and the following frames until a
cooldown of `override_duration_frames` expires.  Overlapping warnings
extend the cooldown to at least the registered duration.

Limitation (registered): the policy changes which already-frozen candidate
forecast is selected.  It does not change the baseline learner's updates.
"""

from __future__ import annotations

import hashlib
import json

from app.learning.adaptive_loop import replay
from app.learning.reliability_monitor import assess_reliability


SCHEMA = "qapha.warning_policy_evaluation.v1"
MODELS = ("persistence", "linear")


def _digest(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_registration(registration):
    """Verify a preregistration and return its locked parameters."""
    if not isinstance(registration, dict):
        raise ValueError("Invalid registration")
    body = {k: v for k, v in registration.items()
            if k != "registration_sha256"}
    if _digest(body) != registration.get("registration_sha256"):
        raise ValueError("Registration integrity check failed")

    baseline = registration["baseline"]
    policy = registration["experimental_policy"]
    if baseline.get("name") != "original_adaptive_selector":
        raise ValueError("Unsupported baseline")
    if policy.get("name") != "warning_triggered_persistence":
        raise ValueError("Unsupported policy")
    if policy.get("override_model") != "persistence":
        raise ValueError("Unsupported override model")

    return {
        "registration_sha256": registration["registration_sha256"],
        "window": baseline["window"],
        "decay": baseline["decay"],
        "recent_window": policy["recent_window"],
        "absolute_threshold_px": policy["absolute_threshold_px"],
        "relative_factor": policy["relative_factor"],
        "minimum_baseline_px": policy["minimum_baseline_px"],
        "override_duration_frames": policy["override_duration_frames"],
    }


def apply_policy(
    events,
    *,
    recent_window=5,
    absolute_threshold_px=15.0,
    relative_factor=3.0,
    minimum_baseline_px=2.0,
    override_duration_frames=3,
):
    """Replay the policy causally over adaptive-loop events."""
    if not isinstance(events, list) or not events:
        raise ValueError("Adaptive events required")
    if (type(override_duration_frames) is not int
            or override_duration_frames < 1):
        raise ValueError("Invalid override duration")

    history = []
    cooldown = 0
    records = []
    previous_frame = None

    for event in events:
        frame = event.get("frame_index")
        if (type(frame) is not int
                or event.get("history_cutoff_frame") != frame - 1
                or (previous_frame is not None
                    and frame != previous_frame + 1)):
            raise ValueError("Invalid event sequence")
        errors = event.get("errors_px", {})
        if event.get("selection") not in MODELS or any(
                name not in errors for name in (*MODELS, "adaptive")):
            raise ValueError("Incomplete event")

        # Warning for this frame uses only errors from earlier frames.
        assessment = assess_reliability(
            history,
            recent_window=recent_window,
            absolute_threshold_px=absolute_threshold_px,
            relative_factor=relative_factor,
            minimum_baseline_px=minimum_baseline_px,
        )
        if assessment["warning"]:
            cooldown = max(cooldown, override_duration_frames)

        overridden = cooldown > 0
        candidate_model = "persistence" if overridden else event["selection"]
        if cooldown > 0:
            cooldown -= 1

        records.append({
            "frame": frame,
            "warning": assessment["warning"],
            "override_active": overridden,
            "baseline_model": event["selection"],
            "candidate_model": candidate_model,
            "baseline_error_px": errors["adaptive"],
            "candidate_error_px": errors[candidate_model],
        })

        # Only now does the current baseline error enter the history.
        history.append(float(errors["adaptive"]))
        previous_frame = frame

    return records


def _summary(errors):
    return {
        "mean_error_px": round(sum(errors) / len(errors), 6),
        "maximum_error_px": round(max(errors), 6),
        "errors_over_25px": sum(e > 25 for e in errors),
    }


def evaluate(track_data, params):
    """Run baseline learner and registered policy on one SAM track."""
    report = replay(track_data, window=params["window"],
                    decay=params["decay"])
    records = apply_policy(
        report["events"],
        recent_window=params["recent_window"],
        absolute_threshold_px=params["absolute_threshold_px"],
        relative_factor=params["relative_factor"],
        minimum_baseline_px=params["minimum_baseline_px"],
        override_duration_frames=params["override_duration_frames"],
    )
    baseline = _summary([r["baseline_error_px"] for r in records])
    candidate = _summary([r["candidate_error_px"] for r in records])
    changed = [r["frame"] for r in records
               if r["candidate_model"] != r["baseline_model"]]
    return {
        "schema": SCHEMA,
        "registration_sha256": params.get("registration_sha256"),
        "parameters": {k: v for k, v in params.items()
                       if k != "registration_sha256"},
        "evaluated_predictions": len(records),
        "warning_frames": [r["frame"] for r in records if r["warning"]],
        "decisions_changed": changed,
        "policy_activated": bool(changed),
        "baseline": baseline,
        "candidate": candidate,
        "mean_error_difference_px": round(
            candidate["mean_error_px"] - baseline["mean_error_px"], 6),
        "records": records,
        "limitations": [
            "Policy selects among frozen baseline forecasts; "
            "it does not alter the learner's updates.",
            "Pixel errors against SAM tracks are not physical ground truth.",
            "One video cannot establish generalization.",
        ],
    }