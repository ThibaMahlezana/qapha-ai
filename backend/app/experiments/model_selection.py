
"""Calibration-only walk-forward motion-model selection."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def select_model(
    tracks,
    manifest,
    *,
    cutoff=49,
    object_id="0",
    training_samples=12,
    horizon=3,
    step=3,
    min_folds=4,
    required_improvement=0.10,
):
    if manifest.get("schema") != "qapha.calibration_clip.v1":
        raise ValueError("Invalid calibration manifest")

    if manifest.get("future_frames_included") is not False:
        raise ValueError("Future frames are not permitted")

    if manifest.get("frame_range_inclusive") != [0, cutoff]:
        raise ValueError("Calibration cutoff mismatch")

    if manifest.get("frames_verified") != cutoff + 1:
        raise ValueError("Calibration frame count mismatch")

    if tracks.get("outcome") != "completed":
        raise ValueError("Tracking is incomplete")

    if any(
        type(value) is not int or value <= 0
        for value in (training_samples, horizon, step, min_folds)
    ):
        raise ValueError("Invalid validation parameters")

    if training_samples < 5:
        raise ValueError("At least five training frames required")

    if not (0 <= required_improvement < 1):
        raise ValueError("Invalid improvement threshold")

    positions = {}

    for entry in tracks.get("boxes", []):
        frame = entry.get("frame_index")

        if type(frame) is not int or not 0 <= frame <= cutoff:
            raise ValueError("Tracking contains an invalid/future frame")

        if str(entry.get("object_id")) != object_id:
            continue

        bbox = entry.get("bbox_xyxy_px")

        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in bbox
            )
            or bbox[2] <= bbox[0]
            or bbox[3] <= bbox[1]
            or frame in positions
        ):
            raise ValueError("Invalid/duplicate object observation")

        positions[frame] = np.array(
            [(bbox[0] + bbox[2]) / 2,
             (bbox[1] + bbox[3]) / 2],
            dtype=float,
        )

    if not positions or cutoff not in positions:
        raise ValueError("Object not tracked at calibration cutoff")

    first = min(positions)
    first_origin = first + training_samples - 1
    last_origin = cutoff - horizon

    folds = []

    for origin in range(first_origin, last_origin + 1, step):
        window = list(
            range(origin - training_samples + 1, origin + 1)
        )
        target = origin + horizon

        # A fold is valid only when every required observation exists.
        if any(f not in positions for f in window + [target]):
            continue

        t = np.array(window, dtype=float) - origin
        xy = np.array([positions[f] for f in window])

        linear_design = np.column_stack(
            (np.ones_like(t), t)
        )
        linear = np.linalg.lstsq(
            linear_design, xy, rcond=None
        )[0]

        quadratic_design = np.column_stack(
            (np.ones_like(t), t, 0.5 * t * t)
        )
        quadratic_y = np.linalg.lstsq(
            quadratic_design, xy[:, 1], rcond=None
        )[0]

        h = float(horizon)

        linear_prediction = np.array([
            linear[0, 0] + linear[1, 0] * h,
            linear[0, 1] + linear[1, 1] * h,
        ])

        quadratic_prediction = np.array([
            linear_prediction[0],
            quadratic_y[0]
            + quadratic_y[1] * h
            + 0.5 * quadratic_y[2] * h * h,
        ])

        observed = positions[target]

        folds.append({
            "training_frames": [window[0], origin],
            "validation_frame": target,
            "errors_px": {
                "linear": round(
                    float(np.linalg.norm(
                        linear_prediction - observed
                    )), 6
                ),
                "quadratic_y": round(
                    float(np.linalg.norm(
                        quadratic_prediction - observed
                    )), 6
                ),
            },
        })

    if len(folds) < min_folds:
        raise ValueError(
            f"Insufficient validation folds: "
            f"{len(folds)} < {min_folds}"
        )

    mean_errors = {
        model: sum(
            fold["errors_px"][model] for fold in folds
        ) / len(folds)
        for model in ("linear", "quadratic_y")
    }

    linear_error = mean_errors["linear"]
    quadratic_error = mean_errors["quadratic_y"]

    # Prefer the simpler model unless the improvement is meaningful.
    selected = (
        "quadratic_y"
        if quadratic_error < linear_error * (1 - required_improvement)
        else "linear"
    )

    history = [
        {
            "frame_index": frame,
            "center_px": positions[frame].tolist(),
        }
        for frame in sorted(positions)
    ]

    fingerprint = hashlib.sha256(
        json.dumps(
            history, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()

    return {
        "schema": "qapha.model_selection.v1",
        "stage": "calibration_only_model_selection",
        "object_id": object_id,
        "cutoff_frame": cutoff,
        "calibration_sha256": manifest["calibration_sha256"],
        "tracking_observations_sha256": fingerprint,
        "policy": {
            "method": "walk_forward",
            "training_samples": training_samples,
            "validation_horizon_frames": horizon,
            "origin_step_frames": step,
            "minimum_folds": min_folds,
            "required_relative_improvement": required_improvement,
            "tie_breaker": "linear",
        },
        "fold_count": len(folds),
        "folds": folds,
        "mean_validation_error_px": {
            key: round(value, 6)
            for key, value in mean_errors.items()
        },
        "selected_model": selected,
        "note": (
            "Selection uses calibration observations only. "
            "Validation outcomes are historical calibration frames, "
            "not held-out verification frames. "
            "Policy validation on Video B is retrospective because "
            "Video B verification results are already known."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cutoff", type=int, default=49)
    parser.add_argument("--object-id", default="0")
    parser.add_argument("--training-samples", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--step", type=int, default=3)
    args = parser.parse_args()

    paths = [
        Path(args.tracks).resolve(),
        Path(args.calibration_manifest).resolve(),
        Path(args.output).resolve(),
    ]

    if len(set(paths)) != 3:
        parser.error("Input and output paths must differ")

    result = select_model(
        json.loads(paths[0].read_text(encoding="utf-8")),
        json.loads(paths[1].read_text(encoding="utf-8")),
        cutoff=args.cutoff,
        object_id=args.object_id,
        training_samples=args.training_samples,
        horizon=args.horizon,
        step=args.step,
    )

    paths[2].parent.mkdir(parents=True, exist_ok=True)

    with paths[2].open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
