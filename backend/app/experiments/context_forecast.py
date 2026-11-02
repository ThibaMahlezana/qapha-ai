
"""Causal event-aware pixel forecast; no physical-contact claims."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def forecast(
    tracking,
    events,
    *,
    object_id="1",
    cutoff=49,
    target=70,
    lookback=12,
    min_samples=6,
):
    if (
        tracking.get("outcome") != "completed"
        or not isinstance(tracking.get("boxes"), list)
    ):
        raise ValueError("Invalid SAM tracks")

    if (
        events.get("schema") != "qapha.motion_events.v1"
        or not isinstance(events.get("events"), list)
    ):
        raise ValueError("Invalid event report")

    if (
        any(
            type(v) is not int
            for v in (cutoff, target, lookback, min_samples)
        )
        or cutoff < 1
        or target <= cutoff
        or lookback < 2
        or min_samples < 3
        or lookback < min_samples
    ):
        raise ValueError("Invalid forecast window")

    if not isinstance(object_id, str) or not object_id:
        raise ValueError("Invalid object ID")

    window = events.get("frame_window")
    if (
        not isinstance(window, list)
        or len(window) != 2
        or any(type(x) is not int for x in window)
        or window[0] < 0
        or window[1] > cutoff
    ):
        raise ValueError("Events exceed observation cutoff")

    positions = {}

    for box in tracking["boxes"]:
        frame = box.get("frame_index")

        if type(frame) is not int or frame < 0:
            raise ValueError("Invalid tracking frame")

        # Never use future detections when fitting the forecast.
        if frame > cutoff or str(box.get("object_id")) != object_id:
            continue

        b = box.get("bbox_xyxy_px")

        if (
            not isinstance(b, list)
            or len(b) != 4
            or any(
                type(v) not in (int, float)
                or not np.isfinite(v)
                for v in b
            )
        ):
            raise ValueError("Invalid bounding box")

        if b[2] <= b[0] or b[3] <= b[1] or frame in positions:
            raise ValueError("Invalid/duplicate bounding box")

        positions[frame] = (
            (b[0] + b[2]) / 2.0,
            (b[1] + b[3]) / 2.0,
        )

    reversals = []

    for event in events["events"]:
        f = event.get("frame_index")

        if (
            type(f) is not int
            or not window[0] <= f <= window[1]
        ):
            raise ValueError("Invalid event frame")

        if f > cutoff:
            raise ValueError("Future event not allowed")

        if (
            str(event.get("object_id")) == object_id
            and event.get("event") in (
                "horizontal_direction_reversal",
                "vertical_direction_reversal",
            )
        ):
            reversals.append(f)

    latest = max(reversals, default=None)

    # Find the consecutive tracking history ending at cutoff.
    # Never fit a line across missing detections.
    start = cutoff

    while start - 1 in positions:
        start -= 1

    start = max(start, cutoff - lookback + 1)
    original_start = start

    policy = "recent_linear_baseline"

    if latest is not None and latest >= start - 1:
        candidate = max(start, latest + 1)

        if cutoff - candidate + 1 >= min_samples:
            start = candidate
            policy = "post_reversal_linear"
        else:
            policy = (
                "recent_linear_insufficient_post_event_samples"
            )

    frames = list(range(start, cutoff + 1))

    if (
        len(frames) < min_samples
        or any(f not in positions for f in frames)
    ):
        raise ValueError(
            "Insufficient contiguous tracking history"
        )

    xs = np.asarray(frames, dtype=float)
    xy = np.asarray(
        [positions[f] for f in frames],
        dtype=float,
    )

    # Centre the time axis at the last observed frame.
    centred = xs - float(cutoff)

    matrix = np.column_stack((
        centred,
        np.ones_like(centred),
    ))

    coefficients, *_ = np.linalg.lstsq(
        matrix,
        xy,
        rcond=None,
    )

    predicted = (
        np.array([target - cutoff, 1.0]) @ coefficients
    ).tolist()

    history = [
        {
            "frame_index": f,
            "center_px": list(positions[f]),
        }
        for f in frames
    ]

    canonical = json.dumps(
        history,
        sort_keys=True,
        separators=(",", ":"),
    )

    return {
        "schema": "qapha.context_forecast.v1",
        "object_id": object_id,
        "cutoff_frame": cutoff,
        "target_frame": target,
        "policy": policy,
        "latest_observed_reversal_frame": latest,
        "train_frames": [start, cutoff],
        "baseline_train_start_without_event_policy": (
            original_start
        ),
        "sample_count": len(frames),
        "predicted_center_px": [
            round(v, 6) for v in predicted
        ],
        "history_sha256": hashlib.sha256(
            canonical.encode()
        ).hexdigest(),
        "history": history,
        "warning": (
            "Image-pixel linear extrapolation, "
            "not a collision model or proof of improvement."
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tracks", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--object-id", default="1")
    parser.add_argument("--cutoff", type=int, default=49)
    parser.add_argument("--target", type=int, default=70)
    parser.add_argument("--lookback", type=int, default=12)

    args = parser.parse_args()

    files = [
        Path(args.tracks).resolve(),
        Path(args.events).resolve(),
        Path(args.output).resolve(),
    ]

    if len(set(files)) != 3:
        parser.error("Inputs and output must be distinct")

    result = forecast(
        json.loads(
            files[0].read_text(encoding="utf-8")
        ),
        json.loads(
            files[1].read_text(encoding="utf-8")
        ),
        object_id=args.object_id,
        cutoff=args.cutoff,
        target=args.target,
        lookback=args.lookback,
    )

    files[2].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Prevent accidental overwriting of frozen results.
    with files[2].open("x", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2)

    print(json.dumps(
        {
            key: value
            for key, value in result.items()
            if key != "history"
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
