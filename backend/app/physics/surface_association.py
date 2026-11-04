
"""Infer possible surface interaction from calibration-only object tracks."""

import argparse
import json
import math
import statistics
from pathlib import Path


def analyze_surface_association(
    tracks,
    manifest,
    *,
    object_id="0",
    stable_frames=6,
    stable_tolerance_px=3.0,
    contact_tolerance_px=5.0,
    min_horizontal_travel_px=5.0,
    min_descent_px=5.0,
    min_reversal_px=2.0,
    large_jump_px=100.0,
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
        or frame_range[1] < 0
    ):
        raise ValueError("Invalid calibration frame range")

    cutoff = frame_range[1]
    if manifest.get("frames_verified") != cutoff + 1:
        raise ValueError("Calibration frame count mismatch")

    if tracks.get("outcome") != "completed":
        raise ValueError("Incomplete tracking")

    if type(stable_frames) is not int or stable_frames < 3:
        raise ValueError("At least three stable frames required")

    parameters = (
        stable_tolerance_px,
        contact_tolerance_px,
        min_horizontal_travel_px,
        min_descent_px,
        min_reversal_px,
        large_jump_px,
    )
    if any(
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value <= 0
        for value in parameters
    ):
        raise ValueError("Invalid threshold")

    observations = {}

    for entry in tracks.get("boxes", []):
        frame = entry.get("frame_index")

        if type(frame) is not int or not 0 <= frame <= cutoff:
            raise ValueError("Invalid or future frame detected")

        if str(entry.get("object_id")) != object_id:
            continue

        box = entry.get("bbox_xyxy_px")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(
                type(v) not in (int, float)
                or not math.isfinite(v)
                for v in box
            )
            or box[2] <= box[0]
            or box[3] <= box[1]
            or frame in observations
        ):
            raise ValueError("Invalid or duplicate observation")

        x1, y1, x2, y2 = box
        observations[frame] = {
            "center_x": (x1 + x2) / 2,
            "center_y": (y1 + y2) / 2,
            "bottom_y": y2,
        }

    recent_frames = list(
        range(cutoff - stable_frames + 1, cutoff + 1)
    )

    if any(frame not in observations for frame in recent_frames):
        raise ValueError("Missing consecutive recent observations")

    recent = [observations[frame] for frame in recent_frames]

    bottom_values = [item["bottom_y"] for item in recent]
    center_y_values = [item["center_y"] for item in recent]

    bottom_span = max(bottom_values) - min(bottom_values)
    center_y_span = max(center_y_values) - min(center_y_values)
    horizontal_travel = abs(
        recent[-1]["center_x"] - recent[0]["center_x"]
    )

    stable = (
        bottom_span <= stable_tolerance_px
        and center_y_span <= stable_tolerance_px
    )

    horizontal = horizontal_travel >= min_horizontal_travel_px

    if stable and horizontal:
        latest_regime = "candidate_surface_constrained_motion"
    elif stable:
        latest_regime = "candidate_stationary_support"
    else:
        latest_regime = "support_not_established"

    candidate_level = (
        float(statistics.median(bottom_values))
        if stable else None
    )

    candidate_events = []

    if candidate_level is not None:
        for frame in range(1, cutoff):
            if not all(
                f in observations for f in (frame - 1, frame, frame + 1)
            ):
                continue

            before = observations[frame - 1]
            current = observations[frame]
            after = observations[frame + 1]

            downward_change = (
                current["center_y"] - before["center_y"]
            )
            upward_change = (
                after["center_y"] - current["center_y"]
            )
            distance = abs(
                current["bottom_y"] - candidate_level
            )

            if (
                downward_change >= min_descent_px
                and upward_change <= -min_reversal_px
                and distance <= contact_tolerance_px
            ):
                candidate_events.append({
                    "frame_index": frame,
                    "bottom_distance_to_candidate_level_px": round(
                        distance, 6
                    ),
                    "preceding_center_dy_px": round(
                        downward_change, 6
                    ),
                    "following_center_dy_px": round(
                        upward_change, 6
                    ),
                    "classification": "candidate_contact_reversal",
                    "physical_collision_verified": False,
                })

    large_jumps = []

    for frame in range(1, cutoff + 1):
        if frame not in observations or frame - 1 not in observations:
            continue

        previous = observations[frame - 1]
        current = observations[frame]

        displacement = math.hypot(
            current["center_x"] - previous["center_x"],
            current["center_y"] - previous["center_y"],
        )

        if displacement > large_jump_px:
            large_jumps.append({
                "from_frame": frame - 1,
                "to_frame": frame,
                "displacement_px": round(displacement, 6),
                "requires_review": True,
            })

    return {
        "schema": "qapha.surface_association.v1",
        "stage": "calibration_only",
        "object_id": object_id,
        "cutoff_frame": cutoff,
        "calibration_sha256": manifest["calibration_sha256"],
        "latest_regime": latest_regime,
        "candidate_support": {
            "bottom_y_px": candidate_level,
            "source": "median_of_recent_tracked_object_bottoms",
            "frames_inclusive": [
                recent_frames[0], recent_frames[-1]
            ],
            "bottom_span_px": round(bottom_span, 6),
            "center_y_span_px": round(center_y_span, 6),
            "horizontal_travel_px": round(horizontal_travel, 6),
            "stable_height": stable,
            "independent_surface_geometry_verified": False,
        },
        "candidate_events": candidate_events,
        "tracking_quality_flags": large_jumps,
        "parameters": {
            "stable_frames": stable_frames,
            "stable_tolerance_px": stable_tolerance_px,
            "contact_tolerance_px": contact_tolerance_px,
            "min_horizontal_travel_px": min_horizontal_travel_px,
            "min_descent_px": min_descent_px,
            "min_reversal_px": min_reversal_px,
            "large_jump_px": large_jump_px,
        },
        "limitations": (
            "Candidate support level is inferred from the object itself, "
            "not independently measured from scene geometry. "
            "Reversals and proximity do not verify physical contact. "
            "Thresholds are pixel-scale heuristics."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--object-id", default="0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = [
        Path(args.tracks).resolve(),
        Path(args.calibration_manifest).resolve(),
        Path(args.output).resolve(),
    ]

    if len(set(paths)) != 3:
        parser.error("Input and output paths must differ")

    result = analyze_surface_association(
        json.loads(paths[0].read_text(encoding="utf-8")),
        json.loads(paths[1].read_text(encoding="utf-8")),
        object_id=args.object_id,
    )

    paths[2].parent.mkdir(parents=True, exist_ok=True)

    with paths[2].open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
