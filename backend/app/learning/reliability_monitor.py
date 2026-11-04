
"""Causal prediction-reliability monitoring.

All warnings for target frame N use errors from frames strictly before N.
This monitor is diagnostic; its thresholds are not calibrated probabilities.
"""

from __future__ import annotations

import math
from statistics import median


def _valid_error(value):
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and value >= 0
    )


def assess_reliability(
    previous_errors,
    *,
    recent_window=5,
    absolute_threshold_px=15.0,
    relative_factor=3.0,
    minimum_baseline_px=2.0,
):
    """Assess the next forecast using past errors only."""

    if not isinstance(previous_errors, list):
        raise ValueError("Previous errors must be a list")

    if any(not _valid_error(e) for e in previous_errors):
        raise ValueError("Invalid prediction error")

    if type(recent_window) is not int or recent_window < 2:
        raise ValueError("Invalid recent window")

    for value in (
        absolute_threshold_px,
        relative_factor,
        minimum_baseline_px,
    ):
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("Invalid threshold")

    if relative_factor <= 1:
        raise ValueError("Relative factor must exceed one")

    if len(previous_errors) < recent_window + 1:
        return {
            "status": "insufficient_history",
            "warning": False,
            "past_errors_used": len(previous_errors),
            "latest_prior_error_px": (
                previous_errors[-1] if previous_errors else None
            ),
            "historical_baseline_px": None,
            "trigger_threshold_px": None,
            "reason": "Not enough previous errors for comparison",
        }

    # Exclude the latest prior error when establishing a baseline.
    # Otherwise, a large surprise would inflate its own threshold.
    reference = previous_errors[-(recent_window + 1):-1]
    baseline = float(median(reference))

    threshold = max(
        float(absolute_threshold_px),
        float(relative_factor) * max(
            baseline, float(minimum_baseline_px)
        ),
    )

    latest = float(previous_errors[-1])
    warning = latest > threshold

    return {
        "status": (
            "elevated_prediction_uncertainty"
            if warning else "normal_recent_error"
        ),
        "warning": warning,
        "past_errors_used": len(previous_errors),
        "latest_prior_error_px": round(latest, 6),
        "historical_baseline_px": round(baseline, 6),
        "trigger_threshold_px": round(threshold, 6),
        "reason": (
            "Latest prior prediction error exceeded "
            "the historical comparison threshold"
            if warning else
            "No abrupt error increase detected by this rule"
        ),
    }


def monitor_timeline(events):
    """Produce a warning for each frame using only earlier frame errors."""

    if not isinstance(events, list) or not events:
        raise ValueError("Timeline events required")

    history = []
    assessments = []
    previous_frame = None

    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Invalid timeline event")

        frame = event.get("frame_index")
        cutoff = event.get("history_cutoff_frame")
        error = event.get("adaptive_error_px")

        if (
            type(frame) is not int
            or frame < 0
            or cutoff != frame - 1
            or (
                previous_frame is not None
                and frame != previous_frame + 1
            )
        ):
            raise ValueError("Invalid timeline sequence")

        if not _valid_error(error):
            raise ValueError("Invalid timeline error")

        # Assessment happens BEFORE current error is revealed.
        assessment = assess_reliability(history)

        assessments.append({
            "frame_index": frame,
            "history_cutoff_frame": cutoff,
            "warning_generated_before_current_observation": True,
            **assessment,
        })

        # Only now may the current error enter the history.
        history.append(float(error))
        previous_frame = frame

    return {
        "schema": "qapha.reliability_monitor.v1",
        "stage": "historical_calibration_diagnostic",
        "evaluated_frames": len(assessments),
        "warnings": sum(
            item["warning"] for item in assessments
        ),
        "assessments": assessments,
        "limitations": [
            "Flags changes in prediction-error history, not physical events.",
            "A warning requires at least one earlier prediction error.",
            "Fixed thresholds are diagnostic, not statistically calibrated.",
            "This is not prospective validation.",
        ],
    }
