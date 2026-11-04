
"""Combine independent visual geometry and tracked-object motion evidence."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def measure_floor_edge(
    video_path,
    manifest,
    *,
    regions=((40, 180), (950, 1150)),
    search_range=(620, 695),
):
    if manifest.get("schema") != "qapha.calibration_clip.v1":
        raise ValueError("Invalid calibration manifest")

    if manifest.get("future_frames_included") is not False:
        raise ValueError("Future frames prohibited")

    cutoff = manifest["frame_range_inclusive"][1]
    if manifest.get("frame_range_inclusive") != [0, cutoff]:
        raise ValueError("Invalid calibration range")

    if manifest.get("frames_verified") != cutoff + 1:
        raise ValueError("Calibration frame count mismatch")

    if sha256_file(video_path) != manifest["calibration_sha256"]:
        raise ValueError("Calibration video SHA-256 mismatch")

    width, height = manifest["resolution_px"]
    top, bottom = search_range

    if not (0 <= top < bottom < height):
        raise ValueError("Invalid search range")

    if not regions or any(
        not (0 <= left < right <= width)
        for left, right in regions
    ):
        raise ValueError("Invalid background regions")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Cannot open calibration video")

    measurements = []

    try:
        for frame_index in (0, cutoff):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()

            if not ok:
                raise RuntimeError(
                    f"Cannot decode calibration frame {frame_index}"
                )

            if frame.shape[:2] != (height, width):
                raise ValueError("Unexpected frame resolution")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            for left, right in regions:
                strip = gray[top:bottom + 1, left:right]
                profile = np.median(
                    strip.astype(np.float32), axis=1
                )
                gradient = np.abs(np.diff(profile))

                best = int(np.argmax(gradient))

                measurements.append({
                    "frame_index": frame_index,
                    "region_x": [left, right],
                    "edge_y_px": top + best + 1,
                    "gradient_strength": round(
                        float(gradient[best]), 3
                    ),
                })
    finally:
        cap.release()

    return measurements


def associate_evidence(
    measurements,
    surface,
    manifest,
    *,
    max_edge_spread_px=3.0,
    min_gradient_strength=10.0,
):
    if surface.get("schema") != "qapha.surface_association.v1":
        raise ValueError("Invalid surface-association report")

    if manifest.get("schema") != "qapha.calibration_clip.v1":
        raise ValueError("Invalid calibration manifest")

    if manifest.get("future_frames_included") is not False:
        raise ValueError("Future frames prohibited")

    if (
        surface.get("calibration_sha256")
        != manifest.get("calibration_sha256")
    ):
        raise ValueError("Calibration fingerprint mismatch")

    cutoff = manifest["frame_range_inclusive"][1]

    if (
        surface.get("cutoff_frame") != cutoff
        or manifest.get("frames_verified") != cutoff + 1
    ):
        raise ValueError("Calibration cutoff mismatch")

    if (
        not measurements
        or len(measurements) < 4
        or any(
            type(m.get("frame_index")) is not int
            or m["frame_index"] not in (0, cutoff)
            or type(m.get("edge_y_px")) is not int
            or not math.isfinite(m.get("gradient_strength", float("nan")))
            for m in measurements
        )
        or {m["frame_index"] for m in measurements} != {0, cutoff}
    ):
        raise ValueError("Invalid background measurements")

    edge_values = [m["edge_y_px"] for m in measurements]
    edge_spread = max(edge_values) - min(edge_values)

    strengths_sufficient = all(
        m["gradient_strength"] >= min_gradient_strength
        for m in measurements
    )

    stable_visual_edge = (
        edge_spread <= max_edge_spread_px
        and strengths_sufficient
    )

    edge_y = (
        float(np.median(edge_values))
        if stable_visual_edge
        else None
    )

    support = surface.get("candidate_support")
    if not isinstance(support, dict):
        raise ValueError("Missing candidate support")

    stable_bottom = support.get("bottom_y_px")

    if stable_bottom is not None and (
        type(stable_bottom) not in (int, float)
        or not math.isfinite(stable_bottom)
    ):
        raise ValueError("Invalid object bottom position")

    offset = (
        round(stable_bottom - edge_y, 6)
        if edge_y is not None and stable_bottom is not None
        else None
    )

    if offset is None:
        relation = "not_determined"
    elif offset > 0:
        relation = "object_bottom_below_visual_edge"
    elif offset < 0:
        relation = "object_bottom_above_visual_edge"
    else:
        relation = "object_bottom_at_visual_edge"

    events = [
        {
            "frame_index": event["frame_index"],
            "motion_evidence": event["classification"],
            "independent_visual_edge_available": stable_visual_edge,
            "physical_collision_verified": False,
        }
        for event in surface.get("candidate_events", [])
    ]

    return {
        "schema": "qapha.geometry_contact_evidence.v1",
        "stage": "calibration_only",
        "object_id": surface["object_id"],
        "cutoff_frame": cutoff,
        "calibration_sha256": manifest["calibration_sha256"],
        "visual_geometry": {
            "measurements": measurements,
            "stable_edge": stable_visual_edge,
            "edge_spread_px": edge_spread,
            "candidate_visual_edge_y_px": edge_y,
            "measurement_method": (
                "Strongest grayscale row transition "
                "in selected background regions"
            ),
            "physical_contact_plane_verified": False,
        },
        "object_geometry_relation": {
            "stable_bottom_y_px": stable_bottom,
            "vertical_offset_px": offset,
            "relation": relation,
            "candidate_surface_constrained_motion": (
                surface.get("latest_regime")
                == "candidate_surface_constrained_motion"
            ),
        },
        "candidate_events": events,
        "tracking_quality_flags": surface.get(
            "tracking_quality_flags", []
        ),
        "physical_collision_verified": False,
        "limitations": [
            "The visual floor edge is not necessarily a contact plane.",
            "The edge measurement uses manually chosen background regions.",
            "Object-bottom offset is not physical penetration depth.",
            "Motion reversal plus scene geometry does not prove causality.",
            "Only calibration frames are used.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--surface-association", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    video = Path(args.video).resolve()
    manifest_path = Path(args.calibration_manifest).resolve()
    surface_path = Path(args.surface_association).resolve()
    output = Path(args.output).resolve()

    if len({video, manifest_path, surface_path, output}) != 4:
        parser.error("All file paths must differ")

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    surface = json.loads(
        surface_path.read_text(encoding="utf-8")
    )

    measurements = measure_floor_edge(video, manifest)
    result = associate_evidence(
        measurements, surface, manifest
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
