
"""Freeze a video prediction before verifying it against future SAM observations."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(value):
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        allow_nan=False
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def observations(report, object_id):
    if (
        report.get("outcome") != "completed"
        or not isinstance(report.get("boxes"), list)
    ):
        raise ValueError("Expected completed SAM tracking data")

    points = []
    seen = set()

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
            raise ValueError("Invalid or duplicate detection")

        seen.add(frame)
        points.append({
            "frame": frame,
            "x": (x1 + x2) / 2,
            "y": (y1 + y2) / 2,
        })

    return sorted(points, key=lambda p: p["frame"])


def freeze(report, object_id="1", fps=25.0,
           last_frame=49, target_frame=60, window=7):

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid FPS")

    if (
        type(last_frame) is not int
        or last_frame < 0
        or type(target_frame) is not int
        or target_frame <= last_frame
    ):
        raise ValueError("Target must follow observed frames")

    if type(window) is not int or window < 2:
        raise ValueError("Window must contain at least two samples")

    if not isinstance(report.get("boxes"), list):
        raise ValueError("Missing SAM boxes")

    # Absolutely no future frames may enter the prediction.
    for box in report["boxes"]:
        frame = box.get("frame_index")
        if type(frame) is not int or frame < 0:
            raise ValueError("Invalid frame index")
        if frame > last_frame:
            raise ValueError("Freeze input contains future frames")

    points = observations(report, str(object_id))

    if len(points) < window or points[-1]["frame"] != last_frame:
        raise ValueError("Insufficient recent observations")

    selected = points[-window:]

    if [p["frame"] for p in selected] != list(
        range(last_frame - window + 1, last_frame + 1)
    ):
        raise ValueError("Recent observations must be consecutive")

    # Least-squares constant-velocity model.
    times = [(p["frame"] - last_frame) / fps for p in selected]
    mean_t = sum(times) / window
    denominator = sum((t - mean_t) ** 2 for t in times)

    future_t = (target_frame - last_frame) / fps
    prediction = {}

    for axis in ("x", "y"):
        values = [p[axis] for p in selected]
        mean_v = sum(values) / window

        slope = sum(
            (t - mean_t) * (v - mean_v)
            for t, v in zip(times, values)
        ) / denominator

        intercept = mean_v - slope * mean_t

        prediction[axis] = round(
            intercept + slope * future_t, 4
        )

    result = {
        "schema": "qapha.video_forecast.v1",
        "model": "least_squares_constant_velocity",
        "object_id": str(object_id),
        "fps": fps,
        "last_frame": last_frame,
        "target_frame": target_frame,
        "window": window,
        "observed_prefix_sha256": digest(points),
        "predicted_center_px": prediction,
        "units": "image_pixels",
        "status": "frozen_unverified",
    }

    result["sha256"] = digest(result)
    return result


def verify(frozen, report):
    if frozen.get("schema") != "qapha.video_forecast.v1":
        raise ValueError("Invalid prediction schema")

    signed_fields = {
        key: value
        for key, value in frozen.items()
        if key != "sha256"
    }

    if (
        digest(signed_fields) != frozen.get("sha256")
        or frozen.get("status") != "frozen_unverified"
    ):
        raise ValueError("Frozen prediction has been modified")

    points = observations(report, frozen["object_id"])

    prefix = [
        p for p in points
        if p["frame"] <= frozen["last_frame"]
    ]

    if digest(prefix) != frozen["observed_prefix_sha256"]:
        raise ValueError("Calibration observations differ")

    targets = [
        p for p in points
        if p["frame"] == frozen["target_frame"]
    ]

    if len(targets) != 1:
        raise ValueError("Target observation unavailable")

    actual = {
        "x": targets[0]["x"],
        "y": targets[0]["y"],
    }

    predicted = frozen["predicted_center_px"]

    error = math.hypot(
        actual["x"] - predicted["x"],
        actual["y"] - predicted["y"],
    )

    return {
        "status": "verified",
        "object_id": frozen["object_id"],
        "target_frame": frozen["target_frame"],
        "prediction_sha256": frozen["sha256"],
        "predicted_center_px": predicted,
        "actual_center_px": actual,
        "euclidean_error_px": round(error, 4),
        "note": "Image-plane error, not physical distance",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Freeze or verify video predictions"
    )

    commands = parser.add_subparsers(
        dest="command", required=True
    )

    make = commands.add_parser("freeze")
    make.add_argument("--tracks", required=True)
    make.add_argument("--output", required=True)
    make.add_argument("--object-id", default="1")
    make.add_argument("--fps", type=float, default=25.0)
    make.add_argument("--last-frame", type=int, default=49)
    make.add_argument("--target-frame", type=int, default=60)
    make.add_argument("--window", type=int, default=7)

    check = commands.add_parser("verify")
    check.add_argument("--frozen", required=True)
    check.add_argument("--tracks", required=True)
    check.add_argument("--output", required=True)

    args = parser.parse_args()
    output = Path(args.output).resolve()

    if args.command == "freeze":
        source = Path(args.tracks).resolve()

        if output == source:
            parser.error("Cannot overwrite tracking input")

        report = json.loads(source.read_text(encoding="utf-8"))

        result = freeze(
            report,
            object_id=args.object_id,
            fps=args.fps,
            last_frame=args.last_frame,
            target_frame=args.target_frame,
            window=args.window,
        )

    else:
        frozen_path = Path(args.frozen).resolve()
        source = Path(args.tracks).resolve()

        if output in (source, frozen_path):
            parser.error("Cannot overwrite input")

        frozen = json.loads(
            frozen_path.read_text(encoding="utf-8")
        )
        report = json.loads(
            source.read_text(encoding="utf-8")
        )

        result = verify(frozen, report)

    output.parent.mkdir(parents=True, exist_ok=True)

    # Exclusive creation prevents accidental overwrites.
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
