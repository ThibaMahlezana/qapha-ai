
"""Calibration-only image-plane motion regime diagnostics."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def analyze_regimes(
    tracks,
    manifest,
    *,
    object_id="0",
    window_size=6,
    horizontal_min_px=5.0,
    vertical_min_px=5.0,
    stable_span_px=4.0,
    vertical_ratio=1.5,
):
    if manifest.get("schema") != "qapha.calibration_clip.v1":
        raise ValueError("Invalid calibration manifest")

    if manifest.get("future_frames_included") is not False:
        raise ValueError("Future frames are forbidden")

    frame_range = manifest.get("frame_range_inclusive")
    if (
        not isinstance(frame_range, list)
        or len(frame_range) != 2
        or frame_range[0] != 0
        or type(frame_range[1]) is not int
    ):
        raise ValueError("Invalid calibration frame range")

    cutoff = frame_range[1]

    if cutoff < 0 or manifest.get("frames_verified") != cutoff + 1:
        raise ValueError("Invalid calibration frame count")

    if tracks.get("outcome") != "completed":
        raise ValueError("Incomplete SAM tracking")

    if type(window_size) is not int or window_size < 3:
        raise ValueError("Window must contain at least three frames")

    thresholds = (
        horizontal_min_px,
        vertical_min_px,
        stable_span_px,
        vertical_ratio,
    )

    if any(
        type(v) not in (int, float)
        or not math.isfinite(v)
        or v <= 0
        for v in thresholds
    ):
        raise ValueError("Invalid motion thresholds")

    positions = {}

    for entry in tracks.get("boxes", []):
        frame = entry.get("frame_index")

        if type(frame) is not int or not 0 <= frame <= cutoff:
            raise ValueError("Invalid or future frame in tracking")

        if str(entry.get("object_id")) != object_id:
            continue

        bbox = entry.get("bbox_xyxy_px")

        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(
                type(v) not in (int, float)
                or not math.isfinite(v)
                for v in bbox
            )
            or bbox[2] <= bbox[0]
            or bbox[3] <= bbox[1]
            or frame in positions
        ):
            raise ValueError("Invalid or duplicate object box")

        positions[frame] = (
            (bbox[0] + bbox[2]) / 2,
            (bbox[1] + bbox[3]) / 2,
        )

    if not positions:
        raise ValueError("Selected object has no observations")

    windows = []

    # Evaluate each possible consecutive window.
    for end in range(window_size - 1, cutoff + 1):
        start = end - window_size + 1
        frames = range(start, end + 1)

        if any(frame not in positions for frame in frames):
            continue

        points = [positions[frame] for frame in frames]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        dx = xs[-1] - xs[0]
        dy = ys[-1] - ys[0]
        x_span = max(xs) - min(xs)
        y_span = max(ys) - min(ys)

        if x_span <= stable_span_px and y_span <= stable_span_px:
            label = "approximately_stationary"

        elif (
            y_span <= stable_span_px
            and abs(dx) >= horizontal_min_px
        ):
            label = "approximately_horizontal"

        elif (
            abs(dy) >= vertical_min_px
            and abs(dy) >= vertical_ratio * abs(dx)
        ):
            label = "vertical_dominant"

        else:
            label = "mixed_or_uncertain"

        windows.append({
            "start_frame": start,
            "end_frame": end,
            "label": label,
            "delta_x_px": round(dx, 6),
            "delta_y_px": round(dy, 6),
            "x_span_px": round(x_span, 6),
            "y_span_px": round(y_span, 6),
        })

    transitions = []

    for previous, current in zip(windows, windows[1:]):
        # Avoid describing gaps in tracking as observed transitions.
        if current["end_frame"] != previous["end_frame"] + 1:
            continue

        if current["label"] != previous["label"]:
            transitions.append({
                "frame_index": current["end_frame"],
                "from": previous["label"],
                "to": current["label"],
                "physical_contact_verified": False,
            })

    history = [
        {"frame_index": frame, "center_px": list(positions[frame])}
        for frame in sorted(positions)
    ]

    fingerprint = hashlib.sha256(
        json.dumps(
            history,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    return {
        "schema": "qapha.motion_regimes.v1",
        "stage": "calibration_only",
        "object_id": object_id,
        "cutoff_frame": cutoff,
        "calibration_sha256": manifest["calibration_sha256"],
        "observations_sha256": fingerprint,
        "parameters": {
            "window_size": window_size,
            "horizontal_min_px": horizontal_min_px,
            "vertical_min_px": vertical_min_px,
            "stable_span_px": stable_span_px,
            "vertical_ratio": vertical_ratio,
        },
        "window_count": len(windows),
        "label_counts": dict(
            Counter(window["label"] for window in windows)
        ),
        "latest_regime": (
            windows[-1]["label"]
            if windows and windows[-1]["end_frame"] == cutoff
            else "insufficient_recent_observations"
        ),
        "windows": windows,
        "transitions": transitions,
        "limitations": (
            "Descriptive image-plane classification, not physical "
            "contact inference. Thresholds are pixel-scale heuristics. "
            "Overlapping windows are correlated. The analyser does "
            "not modify frozen forecasts or model selection."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--object-id", default="0")
    parser.add_argument("--window-size", type=int, default=6)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = [
        Path(args.tracks).resolve(),
        Path(args.calibration_manifest).resolve(),
        Path(args.output).resolve(),
    ]

    if len(set(paths)) != 3:
        parser.error("All paths must differ")

    result = analyze_regimes(
        json.loads(paths[0].read_text(encoding="utf-8")),
        json.loads(paths[1].read_text(encoding="utf-8")),
        object_id=args.object_id,
        window_size=args.window_size,
    )

    paths[2].parent.mkdir(parents=True, exist_ok=True)

    with paths[2].open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)

    print(json.dumps({
        "schema": result["schema"],
        "object_id": result["object_id"],
        "window_count": result["window_count"],
        "label_counts": result["label_counts"],
        "latest_regime": result["latest_regime"],
        "transitions": result["transitions"],
        "output": str(paths[2]),
    }, indent=2))


if __name__ == "__main__":
    main()
