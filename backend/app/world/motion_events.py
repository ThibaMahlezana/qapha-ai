
"""Detect observed image-plane direction reversals in SAM object tracks."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def distance_to_segment(point, endpoints):
    (ax, ay), (bx, by) = endpoints
    dx, dy = bx - ax, by - ay
    norm = dx * dx + dy * dy

    if norm <= 0:
        raise ValueError("Zero-length edge")

    t = max(
        0.0,
        min(1.0, (
            (point[0] - ax) * dx
            + (point[1] - ay) * dy
        ) / norm),
    )

    return math.hypot(
        point[0] - (ax + t * dx),
        point[1] - (ay + t * dy),
    )


def detect_motion_events(
    geometry,
    tracking,
    *,
    start_frame=30,
    end_frame=49,
    edge_ids=(0, 1, 2, 3, 4),
    fps=25.0,
    min_step_px=5.0,
    near_px=30.0,
):
    if geometry.get("schema") != "qapha.scene_geometry.v1":
        raise ValueError("Invalid geometry schema")

    if (
        tracking.get("outcome") != "completed"
        or not isinstance(tracking.get("boxes"), list)
    ):
        raise ValueError("Invalid SAM tracking report")

    if (
        type(start_frame) is not int
        or type(end_frame) is not int
        or start_frame < 0
        or end_frame - start_frame < 2
    ):
        raise ValueError("At least three frames required")

    if any(
        type(v) not in (int, float)
        or not math.isfinite(v)
        or v <= 0
        for v in (fps, min_step_px)
    ):
        raise ValueError("Invalid fps or movement threshold")

    if (
        type(near_px) not in (int, float)
        or not math.isfinite(near_px)
        or near_px < 0
    ):
        raise ValueError("Invalid proximity threshold")

    candidates = geometry.get("candidates")

    if (
        not isinstance(candidates, list)
        or not edge_ids
        or len(edge_ids) != len(set(edge_ids))
    ):
        raise ValueError("Invalid edge selection")

    edges = []

    for idx in edge_ids:
        if type(idx) is not int or not 0 <= idx < len(candidates):
            raise ValueError("Unknown edge")

        points = candidates[idx].get("endpoints_px")

        if (
            not isinstance(points, list)
            or len(points) != 2
            or any(
                not isinstance(p, list)
                or len(p) != 2
                or any(
                    type(v) not in (int, float)
                    or not math.isfinite(v)
                    for v in p
                )
                for p in points
            )
        ):
            raise ValueError("Invalid edge coordinates")

        distance_to_segment(points[0], points)
        edges.append((f"edge_{idx}", points))

    tracks = {}

    for item in tracking["boxes"]:
        frame = item.get("frame_index")

        if type(frame) is not int or frame < 0:
            raise ValueError("Invalid SAM frame index")

        # Never use observations beyond the selected window.
        if not start_frame <= frame <= end_frame:
            continue

        oid = item.get("object_id")
        bounds = item.get("bbox_xyxy_px")

        if (
            oid is None
            or isinstance(oid, bool)
            or not isinstance(bounds, list)
            or len(bounds) != 4
            or any(
                type(v) not in (int, float)
                or not math.isfinite(v)
                for v in bounds
            )
        ):
            raise ValueError("Invalid SAM detection")

        x1, y1, x2, y2 = bounds

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid SAM bounding box")

        object_frames = tracks.setdefault(str(oid), {})

        if frame in object_frames:
            raise ValueError(
                "Duplicate detection for object/frame"
            )

        object_frames[frame] = (
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            y2,
        )

    events = []

    for oid in sorted(tracks):
        positions = tracks[oid]

        for frame in range(start_frame + 1, end_frame):
            # Never infer reversals across missing detections.
            if not all(
                f in positions
                for f in (frame - 1, frame, frame + 1)
            ):
                continue

            previous, current, following = (
                positions[frame - 1],
                positions[frame],
                positions[frame + 1],
            )

            for axis, axis_index in (
                ("horizontal", 0),
                ("vertical", 1),
            ):
                before = (
                    current[axis_index]
                    - previous[axis_index]
                )
                after = (
                    following[axis_index]
                    - current[axis_index]
                )

                if (
                    abs(before) < min_step_px
                    or abs(after) < min_step_px
                    or before * after >= 0
                ):
                    continue

                # Bottom-centre is an image-plane proxy,
                # not a verified contact point.
                foot = (current[0], current[2])

                distances = sorted(
                    (
                        (
                            edge_id,
                            distance_to_segment(foot, points),
                        )
                        for edge_id, points in edges
                    ),
                    key=lambda pair: (pair[1], pair[0]),
                )

                closest_id, closest_distance = distances[0]

                events.append({
                    "object_id": oid,
                    "frame_index": frame,
                    "time_s": round(frame / fps, 6),
                    "event": f"{axis}_direction_reversal",
                    "delta_before_px": round(before, 3),
                    "delta_after_px": round(after, 3),
                    "nearest_edge_id": closest_id,
                    "nearest_edge_distance_px": round(
                        closest_distance, 3
                    ),
                    "near_reviewed_edge": (
                        closest_distance <= near_px
                    ),
                    "physical_collision_verified": False,
                })

    events.sort(
        key=lambda event: (
            event["frame_index"],
            event["object_id"],
            event["event"],
        )
    )

    return {
        "schema": "qapha.motion_events.v1",
        "frame_window": [start_frame, end_frame],
        "coordinate_system": "image_pixels_y_down",
        "fps": fps,
        "reviewed_edge_ids": [
            edge_id for edge_id, _ in edges
        ],
        "events": events,
        "limitations": (
            "Three-frame pixel-centre reversals only. "
            "Box jitter, occlusion and perspective can "
            "produce false positives. Nearest visual edge "
            "is not a verified contact surface. "
            "No impact or bounce is established. "
            "Frames outside the window are not used."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Detect bounded SAM motion reversals"
    )

    parser.add_argument("--geometry", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=int, default=30)
    parser.add_argument("--end", type=int, default=49)
    parser.add_argument("--edge-ids", default="0,1,2,3,4")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument(
        "--min-step-px", type=float, default=5.0
    )
    parser.add_argument(
        "--near-px", type=float, default=30.0
    )

    args = parser.parse_args()

    try:
        edge_ids = tuple(
            int(x.strip())
            for x in args.edge_ids.split(",")
        )
    except ValueError:
        parser.error(
            "--edge-ids must be comma-separated integers"
        )

    source = Path(args.geometry).resolve()
    track_path = Path(args.tracks).resolve()
    output = Path(args.output).resolve()

    if len({source, track_path, output}) != 3:
        parser.error("Input and output paths must differ")

    result = detect_motion_events(
        json.loads(source.read_text(encoding="utf-8")),
        json.loads(track_path.read_text(encoding="utf-8")),
        start_frame=args.start,
        end_frame=args.end,
        edge_ids=edge_ids,
        fps=args.fps,
        min_step_px=args.min_step_px,
        near_px=args.near_px,
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(json.dumps({
        "frame_window": result["frame_window"],
        "event_count": len(result["events"]),
        "events": result["events"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
