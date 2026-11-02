
"""Freeze and evaluate two image-plane motion models.

The acceleration model does not account for collisions or camera calibration.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def digest(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def extract(report, object_id):
    if report.get("outcome") != "completed" or not isinstance(
        report.get("boxes"), list
    ):
        raise ValueError("Expected completed SAM report with boxes")

    seen = set()
    points = []

    for box in report["boxes"]:
        frame = box.get("frame_index")

        if type(frame) is not int or frame < 0:
            raise ValueError("Invalid frame index")

        if str(box.get("object_id")) != object_id:
            continue

        coords = box.get("bbox_xyxy_px")

        if (
            not isinstance(coords, list)
            or len(coords) != 4
            or any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in coords
            )
        ):
            raise ValueError("Invalid bounding box")

        x1, y1, x2, y2 = coords

        if x2 <= x1 or y2 <= y1 or frame in seen:
            raise ValueError("Invalid or duplicate object box")

        seen.add(frame)
        points.append({
            "frame": frame,
            "x": (x1 + x2) / 2,
            "y": (y1 + y2) / 2,
        })

    return sorted(points, key=lambda p: p["frame"])


def freeze_comparison(
    report,
    *,
    object_id="1",
    fps=25.0,
    last_frame=49,
    target_frame=70,
    window=7,
):
    if (
        type(fps) not in (int, float)
        or not math.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError("Invalid fps")

    if (
        type(last_frame) is not int
        or last_frame < 0
        or type(target_frame) is not int
        or target_frame <= last_frame
        or type(window) is not int
        or window < 3
    ):
        raise ValueError("Invalid frame/window configuration")

    if (
        report.get("outcome") != "completed"
        or not isinstance(report.get("boxes"), list)
    ):
        raise ValueError("Invalid report")

    # No future observations may be used during freezing.
    if any(
        type(box.get("frame_index")) is not int
        or box["frame_index"] < 0
        or box["frame_index"] > last_frame
        for box in report["boxes"]
    ):
        raise ValueError("Freeze input includes invalid or future frames")

    points = extract(report, str(object_id))
    recent = points[-window:]

    expected_frames = list(
        range(last_frame - window + 1, last_frame + 1)
    )

    if [p["frame"] for p in recent] != expected_frames:
        raise ValueError(
            "Need consecutive recent observations ending at last_frame"
        )

    t = np.array([
        (p["frame"] - last_frame) / fps for p in recent
    ], dtype=float)

    future_t = (target_frame - last_frame) / fps

    xs = np.array([p["x"] for p in recent], dtype=float)
    ys = np.array([p["y"] for p in recent], dtype=float)

    # Horizontal motion: same linear model for both predictions.
    cx = np.polyfit(t, xs, 1)

    # Model A: constant vertical velocity.
    cy_linear = np.polyfit(t, ys, 1)

    # Model B: constant vertical acceleration.
    cy_quadratic = np.polyfit(t, ys, 2)

    if not all(
        np.isfinite(coefficients).all()
        for coefficients in (cx, cy_linear, cy_quadratic)
    ):
        raise ValueError("Nonfinite fitted coefficients")

    common_x = round(float(np.polyval(cx, future_t)), 4)

    predictions = {
        "constant_velocity": {
            "x": common_x,
            "y": round(
                float(np.polyval(cy_linear, future_t)), 4
            ),
        },
        "constant_vertical_acceleration": {
            "x": common_x,
            "y": round(
                float(np.polyval(cy_quadratic, future_t)), 4
            ),
        },
    }

    result = {
        "schema": "qapha.model_comparison.v1",
        "object_id": str(object_id),
        "fps": float(fps),
        "last_frame": last_frame,
        "target_frame": target_frame,
        "window": window,
        "observed_prefix_sha256": digest(points),
        "predictions": predictions,
        "fitted_vertical_acceleration_px_s2": round(
            float(2 * cy_quadratic[0]), 4
        ),
        "status": "frozen_unverified",
        "limitations": (
            "Uncalibrated image-plane fit; assumes no future "
            "collision or regime change"
        ),
    }

    result["sha256"] = digest(result)
    return result


def verify_comparison(frozen, report):
    if (
        frozen.get("schema") != "qapha.model_comparison.v1"
        or frozen.get("status") != "frozen_unverified"
    ):
        raise ValueError("Invalid frozen prediction")

    content = {
        key: value
        for key, value in frozen.items()
        if key != "sha256"
    }

    if digest(content) != frozen.get("sha256"):
        raise ValueError("Frozen prediction was modified")

    points = extract(report, frozen["object_id"])

    prefix = [
        p for p in points
        if p["frame"] <= frozen["last_frame"]
    ]

    if digest(prefix) != frozen["observed_prefix_sha256"]:
        raise ValueError(
            "Calibration observations differ; merge original "
            "prefix with future observations"
        )

    target = [
        p for p in points
        if p["frame"] == frozen["target_frame"]
    ]

    if len(target) != 1:
        raise ValueError("Target observation unavailable")

    actual = {
        "x": target[0]["x"],
        "y": target[0]["y"],
    }

    errors = {
        name: round(
            math.hypot(
                prediction["x"] - actual["x"],
                prediction["y"] - actual["y"],
            ),
            4,
        )
        for name, prediction in frozen["predictions"].items()
    }

    return {
        "status": "verified",
        "target_frame": frozen["target_frame"],
        "object_id": frozen["object_id"],
        "actual_center_px": actual,
        "predicted_centers_px": frozen["predictions"],
        "errors_px": errors,
        "better_model_on_this_example": min(
            errors, key=errors.get
        ),
        "prediction_sha256": frozen["sha256"],
        "note": (
            "One held-out image-plane comparison; "
            "not evidence of general superiority"
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare frozen video-motion models"
    )

    subs = parser.add_subparsers(
        dest="command", required=True
    )

    make = subs.add_parser("freeze")
    make.add_argument("--tracks", required=True)
    make.add_argument("--output", required=True)
    make.add_argument("--object-id", default="1")
    make.add_argument("--fps", type=float, default=25.0)
    make.add_argument("--last-frame", type=int, default=49)
    make.add_argument("--target-frame", type=int, default=70)
    make.add_argument("--window", type=int, default=7)

    check = subs.add_parser("verify")
    check.add_argument("--frozen", required=True)
    check.add_argument("--tracks", required=True)
    check.add_argument("--output", required=True)

    args = parser.parse_args()

    output = Path(args.output).resolve()
    source = Path(args.tracks).resolve()

    if args.command == "freeze":
        if output == source:
            parser.error("Cannot overwrite input")

        report = json.loads(
            source.read_text(encoding="utf-8")
        )

        result = freeze_comparison(
            report,
            object_id=args.object_id,
            fps=args.fps,
            last_frame=args.last_frame,
            target_frame=args.target_frame,
            window=args.window,
        )

    else:
        frozen_path = Path(args.frozen).resolve()

        if output in (source, frozen_path):
            parser.error("Cannot overwrite input")

        frozen = json.loads(
            frozen_path.read_text(encoding="utf-8")
        )

        report = json.loads(
            source.read_text(encoding="utf-8")
        )

        result = verify_comparison(frozen, report)

    output.parent.mkdir(parents=True, exist_ok=True)

    # Never overwrite an earlier experiment artifact.
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
