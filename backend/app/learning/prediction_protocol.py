
"""Freeze predictions before observations update the adaptive learner."""

from __future__ import annotations

import hashlib
import json
import math

from app.learning.adaptive_session import (
    advance_session,
    create_session,
    restore_tracker,
)


SCHEMA = "qapha.prediction_protocol.v1"


def digest(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _seal(body):
    return {**body, "protocol_sha256": digest(body)}


def _verify(protocol):
    if not isinstance(protocol, dict):
        raise ValueError("Invalid protocol")

    expected = {"schema", "session", "pending", "protocol_sha256"}

    if set(protocol) != expected or protocol["schema"] != SCHEMA:
        raise ValueError("Invalid protocol schema")

    body = {
        key: protocol[key]
        for key in ("schema", "session", "pending")
    }

    if digest(body) != protocol["protocol_sha256"]:
        raise ValueError("Protocol integrity check failed")

    tracker = restore_tracker(protocol["session"])

    pending = protocol["pending"]

    if pending is not None:
        if not isinstance(pending, dict):
            raise ValueError("Invalid pending prediction")

        content = {
            key: value
            for key, value in pending.items()
            if key != "prediction_sha256"
        }

        if digest(content) != pending.get("prediction_sha256"):
            raise ValueError("Prediction integrity check failed")

        next_frame = len(protocol["session"]["observations"])

        if pending.get("target_frame") != next_frame:
            raise ValueError("Pending prediction frame mismatch")

        if pending.get("history_cutoff_frame") != next_frame - 1:
            raise ValueError("Invalid prediction history cutoff")

        if pending.get("source_state_sha256") != (
            protocol["session"]["state_sha256"]
        ):
            raise ValueError("Prediction source state mismatch")

        if len(protocol["session"]["observations"]) < 2:
            raise ValueError("Insufficient history for prediction")

    return tracker


def create_protocol(object_id="0", window=6, decay=0.85):
    session = create_session(object_id, window, decay)

    return _seal({
        "schema": SCHEMA,
        "session": session,
        "pending": None,
    })


def freeze_next_prediction(protocol):
    """
    Predict the next frame without observing its actual position.

    A second forecast is forbidden while one is pending.
    """
    tracker = _verify(protocol)

    if protocol["pending"] is not None:
        raise ValueError("An unresolved prediction already exists")

    observations = protocol["session"]["observations"]

    if len(observations) < 2:
        raise ValueError("At least two observations are required")

    target_frame = len(observations)
    forecasts = tracker._forecast(target_frame)

    if forecasts is None:
        raise ValueError("Predictor is not ready")

    selection = min(
        ("persistence", "linear"),
        key=lambda name: (
            float("inf")
            if tracker.loss[name] is None
            else tracker.loss[name],
            0 if name == "persistence" else 1,
        ),
    )

    prediction_body = {
        "schema": "qapha.frozen_online_prediction.v1",
        "object_id": protocol["session"]["object_id"],
        "history_cutoff_frame": target_frame - 1,
        "target_frame": target_frame,
        "source_state_sha256": protocol["session"]["state_sha256"],
        "selection": selection,
        "predictions_xy": {
            name: [round(float(v), 4) for v in xy]
            for name, xy in forecasts.items()
        },
        "scores_before_observation": dict(tracker.loss),
    }

    pending = {
        **prediction_body,
        "prediction_sha256": digest(prediction_body),
    }

    next_protocol = _seal({
        "schema": SCHEMA,
        "session": protocol["session"],
        "pending": pending,
    })

    return next_protocol, pending


def observe_and_update(protocol, frame_index, xy):
    """
    Resolve a frozen prediction using the actual observation.

    Frames 0 and 1 are bootstrap observations without forecasts.
    Later frames require a pending prediction.
    """
    _verify(protocol)

    observations = protocol["session"]["observations"]
    expected_frame = len(observations)

    if type(frame_index) is not int or frame_index != expected_frame:
        raise ValueError("Unexpected observation frame")

    if (
        not isinstance(xy, (list, tuple))
        or len(xy) != 2
        or any(
            type(value) not in (int, float)
            or not math.isfinite(value)
            for value in xy
        )
    ):
        raise ValueError("Invalid observation")

    pending = protocol["pending"]

    if expected_frame < 2:
        if pending is not None:
            raise ValueError("Bootstrap cannot have pending predictions")
    elif pending is None:
        raise ValueError("A frozen prediction is required")

    updated_session, event = advance_session(
        protocol["session"], frame_index, xy
    )

    if pending is None:
        if event is not None:
            raise ValueError("Unexpected bootstrap forecast")

        evaluation = None
    else:
        if event is None:
            raise ValueError("Missing prediction evaluation")

        if (
            event["selection"] != pending["selection"]
            or event["predictions_xy"] != pending["predictions_xy"]
            or event["scores_before_observation"]
            != pending["scores_before_observation"]
            or event["history_cutoff_frame"]
            != pending["history_cutoff_frame"]
        ):
            raise ValueError("Frozen prediction mismatch")

        evaluation = {
            "schema": "qapha.online_prediction_evaluation.v1",
            "prediction_sha256": pending["prediction_sha256"],
            "target_frame": frame_index,
            "selection": pending["selection"],
            "observed_xy": event["observed_xy"],
            "errors_px": event["errors_px"],
            "scores_after_observation": event[
                "scores_after_observation"
            ],
        }

    updated_protocol = _seal({
        "schema": SCHEMA,
        "session": updated_session,
        "pending": None,
    })

    return updated_protocol, evaluation
