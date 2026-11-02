
"""Analyse image-plane motion from SAM tracking boxes."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def analyze_tracks(report, fps, min_reversal_px=5.0):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")

    if not math.isfinite(min_reversal_px) or min_reversal_px < 0:
        raise ValueError("Invalid reversal threshold")

    if (
        report.get("outcome") != "completed"
        or not isinstance(report.get("boxes"), list)
    ):
        raise ValueError("Expected a completed SAM tracking report")

    grouped = defaultdict(list)
    seen = set()

    for box in report["boxes"]:
        frame = box.get("frame_index")
        object_id = box.get("object_id")
        coords = box.get("bbox_xyxy_px")

        if type(frame) is not int or frame < 0 or object_id is None:
            raise ValueError("Invalid frame or object ID")

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

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Nonpositive bounding box")

        key = (str(object_id), frame)

        if key in seen:
            raise ValueError("Duplicate object in frame")

        seen.add(key)

        grouped[str(object_id)].append({
            "frame_index": frame,
            "time_s": round(frame / fps, 6),
            "center_x_px": (x1 + x2) / 2,
            "center_y_px": (y1 + y2) / 2,
        })

    tracks = []

    for object_id in sorted(grouped):
        points = sorted(
            grouped[object_id],
            key=lambda p: p["frame_index"],
        )

        # Estimate image-plane velocity using finite differences.
        for index, point in enumerate(points):
            if len(points) < 2:
                point["vx_px_s"] = None
                point["vy_px_s"] = None
                continue

            if index == 0:
                left, right = 0, 1
            elif index == len(points) - 1:
                left, right = index - 1, index
            else:
                left, right = index - 1, index + 1

            a, b = points[left], points[right]

            # Do not calculate velocity across missing frames.
            if (
                b["frame_index"] - a["frame_index"]
                != right - left
            ):
                point["vx_px_s"] = None
                point["vy_px_s"] = None
                continue

            dt = (
                b["frame_index"] - a["frame_index"]
            ) / fps

            point["vx_px_s"] = round(
                (b["center_x_px"] - a["center_x_px"]) / dt,
                4,
            )

            point["vy_px_s"] = round(
                (b["center_y_px"] - a["center_y_px"]) / dt,
                4,
            )

        events = []

        # Image Y increases downward.
        # A candidate bounce is a downward-to-upward reversal.
        for index in range(1, len(points) - 1):
            a, b, c = points[index - 1:index + 2]

            if (
                b["frame_index"] - a["frame_index"] != 1
                or c["frame_index"] - b["frame_index"] != 1
            ):
                continue

            before = b["center_y_px"] - a["center_y_px"]
            after = c["center_y_px"] - b["center_y_px"]

            if (
                before >= min_reversal_px
                and after <= -min_reversal_px
            ):
                events.append({
                    "type": "candidate_bounce",
                    "frame_index": b["frame_index"],
                    "time_s": b["time_s"],
                    "center_y_px": b["center_y_px"],
                    "dy_before_px": before,
                    "dy_after_px": after,
                    "verified_collision": False,
                })

        tracks.append({
            "object_id": object_id,
            "samples": points,
            "candidate_events": events,
        })

    return {
        "coordinate_system": "image_pixels_y_down",
        "fps": fps,
        "velocity_unit": "pixels_per_second",
        "tracks": tracks,
        "note": (
            "Candidate bounces are vertical direction changes, "
            "not verified physical collisions."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract motion and candidate bounces"
    )

    parser.add_argument("--tracks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", required=True, type=float)
    parser.add_argument(
        "--min-reversal-px",
        type=float,
        default=5.0,
    )

    args = parser.parse_args()

    source = Path(args.tracks).resolve()
    output = Path(args.output).resolve()

    if source == output:
        parser.error("Output must differ from source")

    report = json.loads(
        source.read_text(encoding="utf-8")
    )

    result = analyze_tracks(
        report,
        args.fps,
        args.min_reversal_px,
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    events = [
        {"object_id": track["object_id"], **event}
        for track in result["tracks"]
        for event in track["candidate_events"]
    ]

    print(json.dumps({
        "objects": len(result["tracks"]),
        "samples": sum(
            len(track["samples"])
            for track in result["tracks"]
        ),
        "candidate_events": events,
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
