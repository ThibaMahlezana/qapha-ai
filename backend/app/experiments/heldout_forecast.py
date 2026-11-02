
"""Freeze linear and quadratic-y forecasts using calibration-only SAM data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def freeze_forecast(tracks, calibration, *,
                    cutoff=49, object_id="0",
                    training_samples=12,
                    horizons=(5, 10, 21)):
    if tracks.get("outcome") != "completed":
        raise ValueError("Incomplete tracking")
    if not isinstance(tracks.get("boxes"), list):
        raise ValueError("Invalid tracking boxes")
    if calibration.get("schema") != "qapha.calibration_clip.v1":
        raise ValueError("Invalid calibration manifest")
    if calibration.get("future_frames_included") is not False:
        raise ValueError("Calibration must exclude future frames")
    if calibration.get("frame_range_inclusive") != [0, cutoff]:
        raise ValueError("Calibration cutoff mismatch")
    if calibration.get("frames_verified") != cutoff + 1:
        raise ValueError("Calibration frame count mismatch")
    if type(cutoff) is not int or cutoff < 0:
        raise ValueError("Invalid cutoff")
    if type(training_samples) is not int or training_samples < 5:
        raise ValueError("At least five training samples required")
    if not isinstance(object_id, str) or not object_id:
        raise ValueError("Invalid object ID")
    if (not horizons or len(set(horizons)) != len(horizons)
            or any(type(h) is not int or h <= 0 for h in horizons)):
        raise ValueError("Invalid prediction horizons")

    positions = {}
    for box in tracks["boxes"]:
        frame = box.get("frame_index")
        if type(frame) is not int or frame < 0 or frame > cutoff:
            raise ValueError("Tracking contains invalid/future frame")
        if str(box.get("object_id")) != object_id:
            continue

        b = box.get("bbox_xyxy_px")
        if (not isinstance(b, list) or len(b) != 4
                or any(type(v) not in (int, float)
                       or not math.isfinite(v) for v in b)
                or b[2] <= b[0] or b[3] <= b[1]
                or frame in positions):
            raise ValueError("Invalid/duplicate object observation")

        positions[frame] = [
            (b[0] + b[2]) / 2,
            (b[1] + b[3]) / 2,
        ]

    start = cutoff - training_samples + 1
    frames = list(range(start, cutoff + 1))
    if start < 0 or any(f not in positions for f in frames):
        raise ValueError("Insufficient consecutive calibration history")

    t = np.asarray(frames, dtype=float) - cutoff
    xy = np.asarray([positions[f] for f in frames], dtype=float)

    linear_design = np.column_stack((np.ones_like(t), t))
    linear = np.linalg.lstsq(linear_design, xy, rcond=None)[0]

    quadratic_design = np.column_stack(
        (np.ones_like(t), t, 0.5 * t * t)
    )
    quadratic_y = np.linalg.lstsq(
        quadratic_design, xy[:, 1], rcond=None
    )[0]

    predictions = []
    for h in sorted(horizons):
        x = float(linear[0, 0] + linear[1, 0] * h)
        linear_y = float(linear[0, 1] + linear[1, 1] * h)
        quad_y = float(
            quadratic_y[0]
            + quadratic_y[1] * h
            + 0.5 * quadratic_y[2] * h * h
        )
        predictions.append({
            "frame_index": cutoff + h,
            "linear_center_px": [round(x, 6), round(linear_y, 6)],
            "quadratic_y_center_px": [round(x, 6), round(quad_y, 6)],
        })

    history = [
        {"frame_index": f, "center_px": positions[f]}
        for f in frames
    ]
    canonical = json.dumps(
        history, sort_keys=True, separators=(",", ":")
    )

    return {
        "schema": "qapha.heldout_forecast.v1",
        "stage": "frozen_before_future_verification",
        "object_id": object_id,
        "cutoff_frame": cutoff,
        "training_frames": [start, cutoff],
        "training_sample_count": training_samples,
        "training_sha256": hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
        "calibration_video_sha256": calibration["calibration_sha256"],
        "fps": calibration["fps"],
        "image_y_acceleration_px_per_frame2": round(
            float(quadratic_y[2]), 6
        ),
        "predictions": predictions,
        "limitations": (
            "Pixel-space extrapolation only. "
            "Neither model simulates ring collisions. "
            "Repeated tracking boxes can bias parameter fitting. "
            "No future observations used."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--object-id", default="0")
    parser.add_argument("--cutoff", type=int, default=49)
    parser.add_argument("--training-samples", type=int, default=12)
    parser.add_argument("--horizons", default="5,10,21")
    args = parser.parse_args()

    paths = [
        Path(args.tracks).resolve(),
        Path(args.calibration_manifest).resolve(),
        Path(args.output).resolve(),
    ]
    if len(set(paths)) != 3:
        parser.error("All paths must differ")

    try:
        horizons = tuple(
            int(x.strip()) for x in args.horizons.split(",")
        )
    except ValueError:
        parser.error("Invalid horizons")

    result = freeze_forecast(
        json.loads(paths[0].read_text(encoding="utf-8")),
        json.loads(paths[1].read_text(encoding="utf-8")),
        cutoff=args.cutoff,
        object_id=args.object_id,
        training_samples=args.training_samples,
        horizons=horizons,
    )

    paths[2].parent.mkdir(parents=True, exist_ok=True)
    with paths[2].open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
