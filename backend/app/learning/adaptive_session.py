
"""Resumable, integrity-checked adaptive learning sessions."""

from __future__ import annotations

import hashlib
import json
import math

from app.learning.adaptive_loop import AdaptiveTracker


SCHEMA = "qapha.adaptive_session.v1"

BODY_FIELDS = {
    "schema",
    "object_id",
    "window",
    "decay",
    "observations",
}


def fingerprint(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def seal(body):
    return {
        **body,
        "state_sha256": fingerprint(body),
    }


def create_session(object_id="0", window=6, decay=0.85):
    """Create an empty online learning session."""

    if not isinstance(object_id, str) or not object_id:
        raise ValueError("Invalid object ID")

    # Reuse validation from the existing tracker.
    AdaptiveTracker(window=window, decay=decay)

    return seal({
        "schema": SCHEMA,
        "object_id": object_id,
        "window": window,
        "decay": decay,
        "observations": [],
    })


def restore_tracker(state):
    """Verify the checkpoint and rebuild its derived model state."""

    if not isinstance(state, dict):
        raise ValueError("Invalid session")

    if set(state) != BODY_FIELDS | {"state_sha256"}:
        raise ValueError("Unexpected session fields")

    body = {
        key: state[key]
        for key in BODY_FIELDS
    }

    if body["schema"] != SCHEMA:
        raise ValueError("Unsupported session schema")

    if (
        not isinstance(body["object_id"], str)
        or not body["object_id"]
    ):
        raise ValueError("Invalid object ID")

    if fingerprint(body) != state["state_sha256"]:
        raise ValueError("Session integrity check failed")

    tracker = AdaptiveTracker(
        window=body["window"],
        decay=body["decay"],
    )

    observations = body["observations"]

    if not isinstance(observations, list):
        raise ValueError("Invalid observations")

    for expected_frame, observation in enumerate(observations):
        if (
            not isinstance(observation, dict)
            or set(observation) != {"frame_index", "xy"}
            or type(observation["frame_index"]) is not int
            or observation["frame_index"] != expected_frame
        ):
            raise ValueError("Invalid observation sequence")

        tracker.step(
            observation["frame_index"],
            observation["xy"],
        )

    return tracker


def advance_session(state, frame_index, xy):
    """
    Predict from prior frames, then score and learn from xy.

    Returns (new_state, prediction_event).
    The original state is not modified.
    """

    tracker = restore_tracker(state)

    expected_frame = len(state["observations"])

    if (
        type(frame_index) is not int
        or frame_index != expected_frame
    ):
        raise ValueError(
            f"Expected frame {expected_frame}, received {frame_index}"
        )

    if (
        not isinstance(xy, (list, tuple))
        or len(xy) != 2
        or any(
            type(v) not in (int, float)
            or not math.isfinite(v)
            for v in xy
        )
    ):
        raise ValueError("Invalid observation coordinates")

    point = [float(xy[0]), float(xy[1])]

    # The existing tracker guarantees forecast-before-update.
    event = tracker.step(frame_index, point)

    body = {
        key: state[key]
        for key in BODY_FIELDS
    }

    body["observations"] = [
        *state["observations"],
        {
            "frame_index": frame_index,
            "xy": point,
        },
    ]

    return seal(body), event


def session_summary(state):
    """Read model performance without modifying the checkpoint."""

    tracker = restore_tracker(state)

    result = {
        "schema": "qapha.adaptive_session_summary.v1",
        "object_id": state["object_id"],
        "observations_seen": len(state["observations"]),
        "last_observed_frame": (
            len(state["observations"]) - 1
        ),
        "next_expected_frame": len(state["observations"]),
        "state_sha256": state["state_sha256"],
    }

    if tracker.count:
        report = tracker.report()

        result["evaluated_predictions"] = tracker.count
        result["mean_error_px"] = report["mean_error_px"]
        result["selection_counts"] = report["selection_counts"]
        result["current_model_scores"] = dict(tracker.loss)
    else:
        result["evaluated_predictions"] = 0
        result["mean_error_px"] = None
        result["selection_counts"] = {
            "persistence": 0,
            "linear": 0,
        }
        result["current_model_scores"] = dict(tracker.loss)

    return result
