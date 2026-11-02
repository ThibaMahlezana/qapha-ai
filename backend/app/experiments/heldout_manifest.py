
"""Register a new evaluation clip before observing its future motion."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2


def register_video(video_path, *, cutoff=49, horizons=(5, 10, 21)):
    if type(cutoff) is not int or cutoff < 8:
        raise ValueError("cutoff must be an integer >= 8")

    if (
        not horizons
        or any(type(h) is not int or h <= 0 for h in horizons)
        or len(set(horizons)) != len(horizons)
    ):
        raise ValueError("horizons must be unique positive frame offsets")

    source = Path(video_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    # Only metadata is inspected here, not future object positions.
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("Cannot open video")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()

    if (
        not math.isfinite(fps)
        or fps <= 0
        or total_frames <= cutoff + max(horizons)
        or width <= 0
        or height <= 0
    ):
        raise ValueError(
            "Invalid video metadata or insufficient future frames"
        )

    return {
        "schema": "qapha.heldout_protocol.v1",
        "stage": "registered_before_future_tracking",
        "video": {
            "path": str(source),
            "sha256": digest.hexdigest(),
            "fps": fps,
            "frame_count_reported": total_frames,
            "resolution_px": [width, height],
        },
        "calibration_frames_inclusive": [0, cutoff],
        "target_frames": [
            cutoff + h for h in sorted(horizons)
        ],
        "evaluation_policy": {
            "perception_calibration_only": True,
            "geometry_from_calibration_only": True,
            "freeze_forecasts_before_verification": True,
            "verification_tracking_separate": True,
            "report_all_scenarios_and_horizons": True,
        },
        "transferred_experimental_assumptions": {
            "restitution": 0.6,
            "contact_offset_px": 0.0,
            "candidate_edge_selection": (
                "review geometry using calibration frames only"
            ),
            "contact_is_hypothetical": True,
        },
        "limitations": (
            "Metadata and full-file checksum do not read future "
            "object coordinates. This registry cannot prevent a "
            "person from viewing future frames. A new video's "
            "perspective and scale may make prior pixel thresholds "
            "inappropriate; do not tune them on verification frames. "
            "Reported frame count must be checked during decoding."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Register a new held-out video"
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cutoff", type=int, default=49)
    parser.add_argument("--horizons", default="5,10,21")
    args = parser.parse_args()

    source = Path(args.video).resolve()
    destination = Path(args.output).resolve()

    if source == destination:
        parser.error("Video and output must differ")

    try:
        offsets = tuple(
            int(s.strip())
            for s in args.horizons.split(",")
        )
    except ValueError:
        parser.error(
            "--horizons must contain comma-separated integers"
        )

    report = register_video(
        source,
        cutoff=args.cutoff,
        horizons=offsets,
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Preserve the original registration if rerun.
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
