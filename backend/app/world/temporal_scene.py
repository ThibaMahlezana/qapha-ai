
"""Temporal scene relationships from SAM tracks and reviewed edges.

All measurements are in image pixels. Proximity does not prove contact.
"""

import argparse
import json
import math
from pathlib import Path


def segment_distance(point, endpoints):
    (ax, ay), (bx, by) = endpoints
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy

    if length_sq <= 0:
        raise ValueError("Degenerate edge")

    t = max(
        0.0,
        min(
            1.0,
            ((point[0] - ax) * dx + (point[1] - ay) * dy)
            / length_sq,
        ),
    )
    return math.hypot(
        point[0] - (ax + t * dx),
        point[1] - (ay + t * dy),
    )


def build_temporal_scene(
    geometry,
    tracking,
    *,
    start_frame=35,
    end_frame=49,
    edge_ids=(0, 1, 2, 3, 4),
    fps=25.0,
    near_px=30.0,
    trend_px=5.0,
    movement_px=5.0,
):
    if geometry.get("schema") != "qapha.scene_geometry.v1":
        raise ValueError("Invalid geometry schema")

    if (
        tracking.get("outcome") != "completed"
        or not isinstance(tracking.get("boxes"), list)
    ):
        raise ValueError("Invalid tracking report")

    if (
        type(start_frame) is not int
        or type(end_frame) is not int
        or start_frame < 0
        or end_frame <= start_frame
    ):
        raise ValueError("Invalid frame window")

    if (
        type(fps) not in (int, float)
        or not math.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError("Invalid FPS")

    if any(
        type(v) not in (int, float)
        or not math.isfinite(v)
        or v < 0
        for v in (near_px, trend_px, movement_px)
    ):
        raise ValueError("Invalid thresholds")

    candidates = geometry.get("candidates")

    if (
        not isinstance(candidates, list)
        or not edge_ids
        or len(set(edge_ids)) != len(edge_ids)
    ):
        raise ValueError("Invalid edge selection")

    edges = []

    for edge_id in edge_ids:
        if (
            type(edge_id) is not int
            or not 0 <= edge_id < len(candidates)
        ):
            raise ValueError("Unknown edge")

        points = candidates[edge_id].get("endpoints_px")

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

        segment_distance(points[0], points)

        edges.append({
            "edge_id": f"edge_{edge_id}",
            "endpoints_px": points,
            "physical_surface_verified": False,
        })

    by_object = {}
    seen = set()

    for box in tracking["boxes"]:
        frame = box.get("frame_index")

        if type(frame) is not int or frame < 0:
            raise ValueError("Invalid tracking frame")

        # Exclude all observations after the selected window.
        if not start_frame <= frame <= end_frame:
            continue

        object_id = box.get("object_id")
        coords = box.get("bbox_xyxy_px")

        if (
            object_id is None
            or not isinstance(coords, list)
            or len(coords) != 4
            or any(
                type(v) not in (int, float)
                or not math.isfinite(v)
                for v in coords
            )
        ):
            raise ValueError("Invalid tracked object")

        x1, y1, x2, y2 = coords

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid bounding box")

        key = (str(object_id), frame)

        if key in seen:
            raise ValueError("Duplicate object in frame")

        seen.add(key)

        by_object.setdefault(str(object_id), {})[frame] = (
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            y2,
        )

    objects = []
    expected = list(range(start_frame, end_frame + 1))

    for object_id in sorted(by_object):
        track = by_object[object_id]

        if not all(frame in track for frame in expected):
            objects.append({
                "object_id": object_id,
                "status": "insufficient_temporal_coverage",
                "observed_frames": len(track),
                "expected_frames": len(expected),
                "relationships": [],
            })
            continue

        first = track[start_frame]
        last = track[end_frame]

        dx = last[0] - first[0]
        dy = last[1] - first[1]
        displacement = math.hypot(dx, dy)

        motion = (
            "approximately_stationary"
            if displacement <= movement_px
            else "moving"
        )

        relationships = []

        for edge in edges:
            endpoints = edge["endpoints_px"]

            # The box's bottom-centre is only an image-plane proxy.
            first_distance = segment_distance(
                (first[0], first[2]), endpoints
            )
            last_distance = segment_distance(
                (last[0], last[2]), endpoints
            )

            change = last_distance - first_distance

            if change < -trend_px:
                trend = "approaching"
            elif change > trend_px:
                trend = "receding"
            else:
                trend = "approximately_constant_distance"

            relationships.append({
                "edge_id": edge["edge_id"],
                "distance_first_px": round(first_distance, 3),
                "distance_last_px": round(last_distance, 3),
                "distance_change_px": round(change, 3),
                "distance_trend": trend,
                "near_at_end": last_distance <= near_px,
                "physical_contact_verified": False,
            })

        objects.append({
            "object_id": object_id,
            "status": "measured_image_motion",
            "observed_frames": len(track),
            "expected_frames": len(expected),
            "displacement_px": [
                round(dx, 3),
                round(dy, 3),
            ],
            "motion_class": motion,
            "relationships": relationships,
        })

    return {
        "schema": "qapha.temporal_scene.v1",
        "frame_window": [start_frame, end_frame],
        "fps": fps,
        "time_window_s": [
            round(start_frame / fps, 6),
            round(end_frame / fps, 6),
        ],
        "coordinate_system": "image_pixels_y_down",
        "reviewed_edges": edges,
        "objects": objects,
        "thresholds_px": {
            "near": near_px,
            "trend": trend_px,
            "movement": movement_px,
        },
        "limitations": (
            "2D image-plane distances only. "
            "No depth, contact, support, or metric geometry verified. "
            "Endpoint trends can miss intervening direction changes."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build temporal scene relationships"
    )

    parser.add_argument("--geometry", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=int, default=35)
    parser.add_argument("--end", type=int, default=49)
    parser.add_argument("--edge-ids", default="0,1,2,3,4")
    parser.add_argument("--fps", type=float, default=25.0)

    args = parser.parse_args()

    geometry_path = Path(args.geometry).resolve()
    tracks_path = Path(args.tracks).resolve()
    output = Path(args.output).resolve()

    if len({geometry_path, tracks_path, output}) != 3:
        parser.error("Inputs and output must be different")

    try:
        edge_ids = tuple(
            int(value.strip())
            for value in args.edge_ids.split(",")
        )
    except ValueError:
        parser.error("--edge-ids requires comma-separated integers")

    geometry = json.loads(
        geometry_path.read_text(encoding="utf-8")
    )
    tracking = json.loads(
        tracks_path.read_text(encoding="utf-8")
    )

    result = build_temporal_scene(
        geometry,
        tracking,
        start_frame=args.start,
        end_frame=args.end,
        edge_ids=edge_ids,
        fps=args.fps,
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    summary = {
        "objects": len(result["objects"]),
        "reviewed_edges": len(result["reviewed_edges"]),
        "relationships": sum(
            len(obj["relationships"])
            for obj in result["objects"]
        ),
        "summary": [
            {
                "object_id": obj["object_id"],
                "motion_class": obj.get(
                    "motion_class", obj["status"]
                ),
                "near_edges": [
                    rel["edge_id"]
                    for rel in obj["relationships"]
                    if rel["near_at_end"]
                ],
            }
            for obj in result["objects"]
        ],
        "output": str(output),
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
