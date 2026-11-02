
"""Build evidence-linked 2D scene graph from reviewed edges and SAM boxes.

Image-plane proximity is not proof of physical contact or support.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def segment_distance(px, py, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    if length2 <= 0:
        raise ValueError("Degenerate segment")
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def build_scene_graph(
    geometry, tracking, *, frame_index=49,
    edge_ids=(0, 1, 2, 3, 4), near_px=30.0
):
    if type(frame_index) is not int or frame_index < 0:
        raise ValueError("Invalid frame index")
    if not math.isfinite(near_px) or near_px < 0:
        raise ValueError("Invalid proximity threshold")
    if geometry.get("schema") != "qapha.scene_geometry.v1":
        raise ValueError("Unknown geometry schema")
    candidates = geometry.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Missing scene candidates")
    if tracking.get("outcome") != "completed" or not isinstance(
        tracking.get("boxes"), list
    ):
        raise ValueError("Expected completed SAM tracking")
    if len(edge_ids) != len(set(edge_ids)) or any(
        type(i) is not int or i < 0 or i >= len(candidates)
        for i in edge_ids
    ):
        raise ValueError("Invalid reviewed edge IDs")

    edges = []
    for index in edge_ids:
        candidate = candidates[index]
        points = candidate["endpoints_px"]
        if (
            len(points) != 2
            or any(
                len(p) != 2
                or any(
                    not isinstance(v, (int, float))
                    or isinstance(v, bool)
                    or not math.isfinite(v)
                    for v in p
                )
                for p in points
            )
        ):
            raise ValueError("Invalid segment coordinates")
        segment_distance(points[0][0], points[0][1], *points)
        edges.append({
            "edge_id": f"edge_{index}",
            "source_candidate_index": index,
            "endpoints_px": points,
            "review_status": "manually_selected_visual_edge",
            "physical_surface_verified": False,
        })

    objects, relations, seen = [], [], set()

    for box in tracking["boxes"]:
        if box.get("frame_index") != frame_index:
            continue

        oid = str(box.get("object_id", ""))
        coords = box.get("bbox_xyxy_px")

        if not oid or oid == "None" or oid in seen:
            raise ValueError("Missing or duplicate tracked object ID")
        if (
            not isinstance(coords, list)
            or len(coords) != 4
            or any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in coords
            )
        ):
            raise ValueError("Invalid object bounds")

        x1, y1, x2, y2 = coords
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid object dimensions")
        seen.add(oid)

        center = [(x1 + x2) / 2, (y1 + y2) / 2]
        bottom_center = [center[0], y2]

        objects.append({
            "object_id": oid,
            "bbox_xyxy_px": coords,
            "center_px": center,
            "bottom_center_px": bottom_center,
            "evidence": {
                "source": "sam_tracking",
                "frame_index": frame_index,
            },
        })

        for edge in edges:
            distance = segment_distance(
                *bottom_center, *edge["endpoints_px"]
            )
            if distance <= near_px:
                relations.append({
                    "subject": oid,
                    "predicate": "image_proximity_to_edge",
                    "object": edge["edge_id"],
                    "distance_px": round(distance, 3),
                    "threshold_px": near_px,
                    "physical_contact_verified": False,
                    "evidence_frame_index": frame_index,
                })

    objects.sort(key=lambda item: item["object_id"])
    relations.sort(key=lambda item: (
        item["subject"], item["object"]
    ))

    return {
        "schema": "qapha.scene_graph.v1",
        "coordinate_system": "image_pixels_y_down",
        "frame_index": frame_index,
        "reviewed_edges": edges,
        "tracked_objects": objects,
        "relationships": relations,
        "limitations": (
            "2D image proximity only; does not establish support, "
            "collision, depth, or metric slope."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Merge reviewed scene edges with SAM observations"
    )
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frame", type=int, default=49)
    parser.add_argument("--edge-ids", default="0,1,2,3,4")
    parser.add_argument("--near-px", type=float, default=30.0)
    args = parser.parse_args()

    try:
        edge_ids = tuple(
            int(x.strip()) for x in args.edge_ids.split(",")
        )
    except ValueError:
        parser.error("--edge-ids must be comma-separated integers")

    inputs = [
        Path(args.geometry).resolve(),
        Path(args.tracks).resolve(),
    ]
    output = Path(args.output).resolve()

    if output in inputs:
        parser.error("Output must not overwrite input")

    geometry = json.loads(
        inputs[0].read_text(encoding="utf-8")
    )
    tracks = json.loads(
        inputs[1].read_text(encoding="utf-8")
    )

    result = build_scene_graph(
        geometry, tracks,
        frame_index=args.frame,
        edge_ids=edge_ids,
        near_px=args.near_px,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps({
        "frame_index": args.frame,
        "reviewed_edges": len(result["reviewed_edges"]),
        "tracked_objects": len(result["tracked_objects"]),
        "proximity_relationships": len(result["relationships"]),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
