
"""Extract candidate static 2D surface edges from video.

These are visual edges, NOT verified physical support surfaces.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def analyze_scene(
    video_path: str | Path,
    *,
    frames: int = 12,
    min_length_px: int = 90,
    max_candidates: int = 25,
) -> tuple[dict, np.ndarray]:

    if (
        not 1 <= frames <= 100
        or min_length_px < 10
        or not 1 <= max_candidates <= 100
    ):
        raise ValueError("Invalid sampling or candidate limit")

    path = Path(video_path).resolve()

    if not path.is_file():
        raise FileNotFoundError(path)

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise ValueError("Unable to open video")

    samples = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Invalid video metadata")

    # Only earliest frames are used to build scene context.
    try:
        for _ in range(frames):
            ok, frame = cap.read()

            if not ok:
                break

            samples.append(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            )
    finally:
        cap.release()

    if not samples:
        raise ValueError("Video contains no decodable frames")

    # Suppress moving foreground by taking a temporal median.
    background = np.median(
        np.stack(samples, axis=0),
        axis=0,
    ).astype(np.uint8)

    edges = cv2.Canny(
        cv2.GaussianBlur(background, (5, 5), 0),
        55,
        145,
    )

    raw = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=55,
        minLineLength=min_length_px,
        maxLineGap=14,
    )

    proposals = []

    if raw is not None:
        lines = np.asarray(raw).reshape(-1, 4)

        for line in lines:
            x1, y1, x2, y2 = map(int, line)

            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)

            # Exclude near-vertical edges for this initial pass.
            if abs(dx) < 0.35 * length:
                continue

            if x1 > x2:
                x1, y1, x2, y2 = x2, y2, x1, y1

            angle = math.degrees(
                math.atan2(y2 - y1, x2 - x1)
            )

            proposals.append({
                "endpoints_px": [
                    [x1, y1],
                    [x2, y2],
                ],
                "length_px": round(length, 2),
                "angle_deg_image": round(angle, 2),
                "kind": (
                    "approximately_horizontal"
                    if abs(angle) <= 8
                    else "inclined"
                ),
                "physical_surface_verified": False,
            })

    # Retain deterministic ordering.
    proposals.sort(
        key=lambda p: (
            -p["length_px"],
            p["endpoints_px"],
        )
    )

    proposals = proposals[:max_candidates]

    overlay = cv2.cvtColor(
        background,
        cv2.COLOR_GRAY2BGR,
    )

    for index, proposal in enumerate(proposals):
        (x1, y1), (x2, y2) = proposal["endpoints_px"]

        cv2.line(
            overlay,
            (x1, y1),
            (x2, y2),
            (30, 210, 255),
            2,
        )

        cv2.putText(
            overlay,
            str(index),
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (30, 210, 255),
            1,
        )

    report = {
        "schema": "qapha.scene_geometry.v1",
        "source": str(path),
        "coordinate_system": "image_pixels_y_down",
        "resolution_px": [width, height],
        "sampled_frames": len(samples),
        "sampling": "first_consecutive_frames",
        "candidate_count": len(proposals),
        "candidates": proposals,
        "limitations": (
            "Image edges only; some may be decoration, "
            "occlusion boundaries or duplicate edges. "
            "No contact, depth or slope-in-metres is established."
        ),
    }

    return report, overlay


def main():
    parser = argparse.ArgumentParser(
        description="Propose static scene edges for human review"
    )

    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overlay", required=True)
    parser.add_argument("--frames", type=int, default=12)

    args = parser.parse_args()

    source = Path(args.video).resolve()
    output = Path(args.output).resolve()
    overlay_path = Path(args.overlay).resolve()

    if len({source, output, overlay_path}) != 3:
        parser.error("Input and outputs must be distinct")

    report, image = analyze_scene(
        source,
        frames=args.frames,
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    overlay_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Do not overwrite previous experiment artifacts.
    with output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    if not cv2.imwrite(str(overlay_path), image):
        raise OSError("Failed to write diagnostic overlay")

    print(json.dumps({
        "sampled_frames": report["sampled_frames"],
        "candidate_count": report["candidate_count"],
        "output": str(output),
        "overlay": str(overlay_path),
    }, indent=2))


if __name__ == "__main__":
    main()
