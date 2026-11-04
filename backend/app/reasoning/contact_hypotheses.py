
"""Create auditable physical hypotheses from calibration evidence."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def fingerprint(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_contact_hypotheses(geometry, surface, manifest):
    if geometry.get("schema") != "qapha.geometry_contact_evidence.v1":
        raise ValueError("Invalid geometry evidence")

    if surface.get("schema") != "qapha.surface_association.v1":
        raise ValueError("Invalid surface evidence")

    if manifest.get("schema") != "qapha.calibration_clip.v1":
        raise ValueError("Invalid calibration manifest")

    if manifest.get("future_frames_included") is not False:
        raise ValueError("Future frames are not permitted")

    frame_range = manifest.get("frame_range_inclusive")
    if (
        not isinstance(frame_range, list)
        or len(frame_range) != 2
        or frame_range[0] != 0
        or type(frame_range[1]) is not int
        or frame_range[1] < 0
    ):
        raise ValueError("Invalid calibration range")

    cutoff = frame_range[1]

    if manifest.get("frames_verified") != cutoff + 1:
        raise ValueError("Calibration frame count mismatch")

    calibration_hash = manifest.get("calibration_sha256")

    for report in (geometry, surface):
        if report.get("stage") != "calibration_only":
            raise ValueError("Evidence must be calibration-only")

        if report.get("calibration_sha256") != calibration_hash:
            raise ValueError("Calibration fingerprint mismatch")

        if report.get("cutoff_frame") != cutoff:
            raise ValueError("Evidence cutoff mismatch")

    object_id = surface.get("object_id")

    if (
        not isinstance(object_id, str)
        or not object_id
        or geometry.get("object_id") != object_id
    ):
        raise ValueError("Object identity mismatch")

    visual = geometry.get("visual_geometry")
    relation = geometry.get("object_geometry_relation")
    support = surface.get("candidate_support")

    if not all(isinstance(x, dict) for x in (visual, relation, support)):
        raise ValueError("Missing evidence sections")

    edge_available = visual.get("stable_edge") is True

    if visual.get("physical_contact_plane_verified") is not False:
        raise ValueError("Unsupported contact-plane verification claim")

    edge_y = visual.get("candidate_visual_edge_y_px")

    if edge_available:
        if (
            type(edge_y) not in (int, float)
            or not math.isfinite(edge_y)
        ):
            raise ValueError("Invalid visual edge")
    elif edge_y is not None:
        raise ValueError("Unstable edge must not have a verified position")

    stable_bottom = support.get("bottom_y_px")
    if stable_bottom is not None and (
        type(stable_bottom) not in (int, float)
        or not math.isfinite(stable_bottom)
    ):
        raise ValueError("Invalid support position")

    if relation.get("stable_bottom_y_px") != stable_bottom:
        raise ValueError("Object-bottom evidence mismatch")

    if edge_available and stable_bottom is not None:
        expected_offset = round(stable_bottom - edge_y, 6)
        if relation.get("vertical_offset_px") != expected_offset:
            raise ValueError("Geometry offset mismatch")

    if geometry.get("physical_collision_verified") is not False:
        raise ValueError("Unsupported physical verification claim")

    events = surface.get("candidate_events", [])
    geometry_events = geometry.get("candidate_events", [])

    if not isinstance(events, list) or not isinstance(geometry_events, list):
        raise ValueError("Invalid event collections")

    event_frames = [e.get("frame_index") for e in events]
    geometry_frames = [e.get("frame_index") for e in geometry_events]

    if (
        event_frames != geometry_frames
        or event_frames != sorted(set(event_frames))
        or any(type(f) is not int or not 0 <= f <= cutoff
               for f in event_frames)
    ):
        raise ValueError("Inconsistent event frames")

    jumps = geometry.get("tracking_quality_flags", [])
    if not isinstance(jumps, list):
        raise ValueError("Invalid quality flags")

    hypotheses = []

    for event, measured_event in zip(events, geometry_events):
        if (
            event.get("classification") != "candidate_contact_reversal"
            or event.get("physical_collision_verified") is not False
            or measured_event.get("motion_evidence")
            != "candidate_contact_reversal"
            or measured_event.get("physical_collision_verified") is not False
            or measured_event.get("independent_visual_edge_available")
            is not edge_available
        ):
            raise ValueError("Inconsistent reversal evidence")

        frame = event["frame_index"]

        nearby_quality_flags = [
            flag
            for flag in jumps
            if (
                type(flag.get("from_frame")) is int
                and type(flag.get("to_frame")) is int
                and flag["from_frame"] <= frame + 2
                and flag["to_frame"] >= frame - 2
            )
        ]

        hypotheses.append({
            "id": f"candidate_reversal_frame_{frame}",
            "type": "possible_surface_interaction",
            "frame_index": frame,
            "status": "hypothesis_not_verified",
            "observed_evidence": {
                "vertical_direction_reversal": True,
                "reversal_frame": frame,
                "stable_visual_edge_detected": edge_available,
                "candidate_visual_edge_y_px": edge_y,
            },
            "inference": (
                "The motion reversal may reflect an interaction "
                "with the environment."
            ),
            "quality_warning": bool(nearby_quality_flags),
            "quality_flags": nearby_quality_flags,
            "limitations": [
                "Visual floor edge is not a calibrated contact plane.",
                "No event-time 3D contact position was established.",
                "Motion reversal does not establish collision causality.",
            ],
            "physical_collision_verified": False,
        })

    if surface.get("latest_regime") == "candidate_surface_constrained_motion":
        if stable_bottom is None or support.get("stable_height") is not True:
            raise ValueError("Surface-constrained hypothesis lacks stable height")

        hypotheses.append({
            "id": f"candidate_support_frame_{cutoff}",
            "type": "possible_surface_constrained_motion",
            "frame_index": cutoff,
            "status": "hypothesis_not_verified",
            "observed_evidence": {
                "stable_object_bottom_y_px": stable_bottom,
                "horizontal_travel_px": support["horizontal_travel_px"],
                "stable_visual_edge_detected": edge_available,
                "candidate_visual_edge_y_px": edge_y,
                "object_bottom_to_edge_offset_px": (
                    round(stable_bottom - edge_y, 6)
                    if edge_available else None
                ),
            },
            "inference": (
                "The object may be moving along a supporting surface."
            ),
            "quality_warning": False,
            "quality_flags": [],
            "limitations": [
                "Constant image height does not establish physical support.",
                "The floor edge and contact plane may differ.",
            ],
            "physical_collision_verified": False,
        })

    return {
        "schema": "qapha.contact_hypotheses.v1",
        "stage": "calibration_only",
        "object_id": object_id,
        "cutoff_frame": cutoff,
        "calibration_sha256": calibration_hash,
        "geometry_evidence_sha256": fingerprint(geometry),
        "surface_evidence_sha256": fingerprint(surface),
        "hypothesis_count": len(hypotheses),
        "hypotheses": hypotheses,
        "physical_collision_verified": False,
        "note": (
            "Hypotheses are deterministic interpretations of "
            "observed motion and visual geometry. They are not "
            "verified physical contact events."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--surface", required=True)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = [
        Path(args.geometry).resolve(),
        Path(args.surface).resolve(),
        Path(args.calibration_manifest).resolve(),
        Path(args.output).resolve(),
    ]

    if len(set(paths)) != 4:
        parser.error("All paths must differ")

    inputs = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in paths[:3]
    ]

    result = build_contact_hypotheses(*inputs)

    paths[3].parent.mkdir(parents=True, exist_ok=True)

    with paths[3].open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
