
"""Post-hoc audit of hypothetical contacts against SAM tracking.

Evaluation only: future observations must never feed back into
the original frozen forecast.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def distance_to_segment(point, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy

    if length2 <= 0:
        raise ValueError("Degenerate visual edge")

    t = max(
        0.0,
        min(1.0, (
            (point[0] - ax) * dx
            + (point[1] - ay) * dy
        ) / length2),
    )

    return math.hypot(
        point[0] - (ax + t * dx),
        point[1] - (ay + t * dy),
    )


def audit_contact(
    forecast,
    tracking,
    geometry,
    *,
    window_frames=6,
    near_px=30.0,
    min_step_px=5.0,
):
    if forecast.get("schema") != "qapha.conditional_contact.v1":
        raise ValueError("Invalid frozen forecast")

    if (
        tracking.get("outcome") != "completed"
        or not isinstance(tracking.get("boxes"), list)
    ):
        raise ValueError("Invalid verification tracks")

    if (
        geometry.get("schema") != "qapha.scene_geometry.v1"
        or not isinstance(geometry.get("candidates"), list)
    ):
        raise ValueError("Invalid geometry")

    if (
        type(window_frames) is not int
        or window_frames < 1
    ):
        raise ValueError("Invalid audit window")

    if any(
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < 0
        for value in (near_px, min_step_px)
    ):
        raise ValueError("Invalid thresholds")

    assumption = forecast.get("contact_assumption")
    impacts = forecast.get("hypothetical_impacts")
    cutoff = forecast.get("cutoff_frame")
    object_id = forecast.get("object_id")

    if not isinstance(assumption, dict):
        raise ValueError("Missing contact assumption")

    if not isinstance(impacts, list):
        raise ValueError("Missing hypothetical impacts")

    if type(cutoff) is not int or cutoff < 0:
        raise ValueError("Invalid forecast cutoff")

    if not isinstance(object_id, str) or not object_id:
        raise ValueError("Invalid forecast object")

    if assumption.get("physical_surface_verified") is not False:
        raise ValueError("Unexpected verified contact surface")

    edge_id = assumption.get("edge_id")

    if (
        not isinstance(edge_id, str)
        or not edge_id.startswith("edge_")
        or not edge_id[5:].isdigit()
    ):
        raise ValueError("Invalid selected edge")

    index = int(edge_id[5:])
    candidates = geometry["candidates"]

    if not 0 <= index < len(candidates):
        raise ValueError("Selected edge not found")

    points = candidates[index].get("endpoints_px")

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

    # Also rejects zero-length segments.
    distance_to_segment(points[0], *points)

    observations = {}

    for box in tracking["boxes"]:
        if str(box.get("object_id")) != object_id:
            continue

        frame = box.get("frame_index")
        bounds = box.get("bbox_xyxy_px")

        if type(frame) is not int or frame < 0:
            raise ValueError("Invalid frame")

        if (
            not isinstance(bounds, list)
            or len(bounds) != 4
            or any(
                type(v) not in (int, float)
                or not math.isfinite(v)
                for v in bounds
            )
        ):
            raise ValueError("Invalid bounding box")

        x1, y1, x2, y2 = bounds

        if x2 <= x1 or y2 <= y1 or frame in observations:
            raise ValueError("Invalid or duplicate detection")

        observations[frame] = {
            "center": ((x1 + x2) / 2, (y1 + y2) / 2),
            "bottom": ((x1 + x2) / 2, y2),
            "half_height": (y2 - y1) / 2,
        }

    reports = []

    for impact in impacts:
        elapsed = impact.get("time_after_cutoff_frames")
        predicted_x = impact.get("image_x_px")

        if any(
            type(v) not in (int, float)
            or not math.isfinite(v)
            for v in (elapsed, predicted_x)
        ) or elapsed <= 0:
            raise ValueError("Invalid hypothetical impact")

        predicted_frame = cutoff + elapsed
        nearest_frame = math.floor(predicted_frame + 0.5)

        start = max(
            cutoff + 1,
            nearest_frame - window_frames,
        )
        end = nearest_frame + window_frames

        sampled = []

        for frame in range(start, end + 1):
            observation = observations.get(frame)

            if observation is None:
                continue

            sampled.append({
                "frame_index": frame,
                "center_px": list(observation["center"]),
                "bottom_center_px": list(observation["bottom"]),
                "half_height_px": round(
                    observation["half_height"], 3
                ),
                "center_edge_distance_px": round(
                    distance_to_segment(
                        observation["center"], *points
                    ), 3
                ),
                "bottom_edge_distance_px": round(
                    distance_to_segment(
                        observation["bottom"], *points
                    ), 3
                ),
            })

        closest = min(
            sampled,
            key=lambda item: (
                item["bottom_edge_distance_px"],
                abs(item["frame_index"] - predicted_frame),
            ),
            default=None,
        )

        reversals = []

        for frame in range(start + 1, end):
            before = observations.get(frame - 1)
            current = observations.get(frame)
            after = observations.get(frame + 1)

            if any(
                value is None
                for value in (before, current, after)
            ):
                continue

            delta_before = (
                current["center"][1] - before["center"][1]
            )
            delta_after = (
                after["center"][1] - current["center"][1]
            )

            if (
                delta_before >= min_step_px
                and delta_after <= -min_step_px
            ):
                distance = distance_to_segment(
                    current["bottom"], *points
                )

                reversals.append({
                    "frame_index": frame,
                    "delta_before_px": round(
                        delta_before, 3
                    ),
                    "delta_after_px": round(
                        delta_after, 3
                    ),
                    "bottom_edge_distance_px": round(
                        distance, 3
                    ),
                    "near_visual_edge": distance <= near_px,
                })

        observed_at_nearest = observations.get(nearest_frame)

        reports.append({
            "predicted_impact_frame": round(
                predicted_frame, 6
            ),
            "predicted_impact_x_px": predicted_x,
            "audit_frame_window": [start, end],
            "observed_frame_count": len(sampled),
            "observed_at_nearest_frame": (
                {
                    "frame_index": nearest_frame,
                    "center_px": list(
                        observed_at_nearest["center"]
                    ),
                    "bottom_center_px": list(
                        observed_at_nearest["bottom"]
                    ),
                }
                if observed_at_nearest is not None
                else None
            ),
            "closest_bottom_to_edge": closest,
            "vertical_down_to_up_reversals": reversals,
            "any_reversal_near_edge": any(
                event["near_visual_edge"]
                for event in reversals
            ),
            "physical_collision_verified": False,
        })

    return {
        "schema": "qapha.contact_audit.v1",
        "stage": "post_prediction_evaluation_only",
        "object_id": object_id,
        "forecast_cutoff_frame": cutoff,
        "reviewed_edge_id": edge_id,
        "audits": reports,
        "limitations": (
            "SAM box geometry and 2D proximity do not establish "
            "3D contact. A motion reversal near a visual edge "
            "is suggestive, not proof of collision. "
            "Future verification observations were used only "
            "for this audit, never to update the frozen forecast."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Audit hypothetical impacts against observed tracks"
    )

    for name in ("forecast", "tracks", "geometry", "output"):
        parser.add_argument("--" + name, required=True)

    parser.add_argument(
        "--window-frames", type=int, default=6
    )
    parser.add_argument(
        "--near-px", type=float, default=30.0
    )
    args = parser.parse_args()

    paths = [
        Path(getattr(args, name)).resolve()
        for name in (
            "forecast", "tracks", "geometry", "output"
        )
    ]

    if len(set(paths)) != 4:
        parser.error("Input and output paths must differ")

    data = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in paths[:3]
    ]

    result = audit_contact(
        *data,
        window_frames=args.window_frames,
        near_px=args.near_px,
    )

    paths[3].parent.mkdir(parents=True, exist_ok=True)

    with paths[3].open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
