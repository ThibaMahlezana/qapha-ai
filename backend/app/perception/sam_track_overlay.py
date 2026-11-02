
"""Visualize SAM video-tracking boxes, IDs and motion trails."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2


COLORS = [
    (60, 210, 255),
    (80, 230, 80),
    (255, 130, 80),
    (220, 100, 220),
    (80, 220, 220),
]


def validate_tracks(report: dict) -> dict[int, list[dict]]:
    """Validate and group SAM detections by source frame index."""
    if not isinstance(report, dict):
        raise ValueError("Tracking report must be an object")

    if report.get("outcome") != "completed":
        raise ValueError("SAM tracking must have completed")

    boxes = report.get("boxes")
    if not isinstance(boxes, list):
        raise ValueError("Tracking report requires a boxes list")

    grouped = defaultdict(list)
    seen = set()

    for box in boxes:
        frame = box.get("frame_index")
        object_id = str(box.get("object_id", ""))
        coords = box.get("bbox_xyxy_px")

        if type(frame) is not int or frame < 0:
            raise ValueError("Invalid frame index")

        if not object_id or object_id == "None":
            raise ValueError("Missing object ID")

        if (
            not isinstance(coords, list)
            or len(coords) != 4
            or not all(
                isinstance(v, (int, float))
                and not isinstance(v, bool)
                and math.isfinite(v)
                for v in coords
            )
        ):
            raise ValueError("Invalid bounding box")

        x1, y1, x2, y2 = coords

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Bounding box has invalid dimensions")

        key = (frame, object_id)

        if key in seen:
            raise ValueError("Duplicate object ID within a frame")

        seen.add(key)
        grouped[frame].append(box)

    return dict(grouped)


def render_overlay(
    video_path: str | Path,
    tracks_path: str | Path,
    output_path: str | Path,
    trail_length: int = 15,
) -> dict:
    """Create a diagnostic MP4 containing tracked objects."""

    if not 0 <= trail_length <= 100:
        raise ValueError("trail_length must be between 0 and 100")

    video_path = Path(video_path).resolve()
    tracks_path = Path(tracks_path).resolve()
    output_path = Path(output_path).resolve()

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not tracks_path.is_file():
        raise FileNotFoundError(f"Tracking JSON not found: {tracks_path}")

    if output_path == video_path:
        raise ValueError("Output cannot overwrite the source video")

    report = json.loads(tracks_path.read_text(encoding="utf-8"))
    grouped = validate_tracks(report)

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError("Cannot open source video")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError("Invalid video metadata")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise ValueError("Cannot create output video")

    object_ids = sorted(
        {str(box["object_id"]) for boxes in grouped.values() for box in boxes}
    )

    color_map = {
        object_id: COLORS[index % len(COLORS)]
        for index, object_id in enumerate(object_ids)
    }

    trails = defaultdict(list)
    rendered = 0
    frames_with_detections = 0

    try:
        while True:
            success, image = capture.read()

            if not success:
                break

            frame_index = rendered
            detections = grouped.get(frame_index, [])

            if detections:
                frames_with_detections += 1

            for box in detections:
                object_id = str(box["object_id"])
                x1, y1, x2, y2 = (
                    int(round(value))
                    for value in box["bbox_xyxy_px"]
                )

                center_x = int(round((x1 + x2) / 2))
                center_y = int(round((y1 + y2) / 2))

                color = color_map[object_id]

                cv2.rectangle(
                    image, (x1, y1), (x2, y2), color, 2
                )

                cv2.circle(
                    image, (center_x, center_y), 4, color, -1
                )

                cv2.putText(
                    image,
                    f"SAM ID {object_id}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

                trails[object_id].append((center_x, center_y))

                if trail_length:
                    trails[object_id] = trails[object_id][-trail_length:]
                else:
                    trails[object_id].clear()

                points = trails[object_id]

                for previous, current in zip(points, points[1:]):
                    cv2.line(
                        image, previous, current, color, 2
                    )

            timestamp_ms = frame_index * 1000.0 / fps

            cv2.putText(
                image,
                f"Frame {frame_index} | {timestamp_ms:.1f} ms "
                f"| Objects: {len(detections)}",
                (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(image)
            rendered += 1

    finally:
        capture.release()
        writer.release()

    if grouped and max(grouped) >= rendered:
        raise ValueError(
            "Tracking report references frames beyond the video"
        )

    return {
        "frames_rendered": rendered,
        "unique_object_ids": len(object_ids),
        "frames_with_detections": frames_with_detections,
        "fps": fps,
        "output": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Render SAM video tracking overlay"
    )

    parser.add_argument("--video", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trail-length", type=int, default=15)

    args = parser.parse_args()

    result = render_overlay(
        args.video,
        args.tracks,
        args.output,
        trail_length=args.trail_length,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
