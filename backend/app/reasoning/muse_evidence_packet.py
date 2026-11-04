
"""Build an evidence-bounded, calibration-only Muse reasoning packet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ALLOWED_TYPES = {
    "possible_surface_interaction",
    "possible_surface_constrained_motion",
}


def digest(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_packet(hypotheses, geometry, surface, manifest):
    if hypotheses.get("schema") != "qapha.contact_hypotheses.v1":
        raise ValueError("Invalid hypotheses schema")

    if geometry.get("schema") != "qapha.geometry_contact_evidence.v1":
        raise ValueError("Invalid geometry schema")

    if surface.get("schema") != "qapha.surface_association.v1":
        raise ValueError("Invalid surface schema")

    if manifest.get("schema") != "qapha.calibration_clip.v1":
        raise ValueError("Invalid calibration schema")

    if manifest.get("future_frames_included") is not False:
        raise ValueError("Future frames must not be included")

    window = manifest.get("frame_range_inclusive")
    if (
        not isinstance(window, list)
        or len(window) != 2
        or window[0] != 0
        or type(window[1]) is not int
        or window[1] < 0
        or manifest.get("frames_verified") != window[1] + 1
    ):
        raise ValueError("Invalid calibration frame window")

    cutoff = window[1]
    calibration_hash = manifest.get("calibration_sha256")
    object_id = hypotheses.get("object_id")

    if not isinstance(calibration_hash, str) or not calibration_hash:
        raise ValueError("Missing calibration fingerprint")

    if not isinstance(object_id, str) or not object_id:
        raise ValueError("Invalid object ID")

    for report in (hypotheses, geometry, surface):
        if report.get("stage") != "calibration_only":
            raise ValueError("Report is not calibration-only")

        if report.get("cutoff_frame") != cutoff:
            raise ValueError("Cutoff mismatch")

        if report.get("object_id") != object_id:
            raise ValueError("Object mismatch")

        if report.get("calibration_sha256") != calibration_hash:
            raise ValueError("Calibration fingerprint mismatch")

    if hypotheses.get("geometry_evidence_sha256") != digest(geometry):
        raise ValueError("Geometry evidence fingerprint mismatch")

    if hypotheses.get("surface_evidence_sha256") != digest(surface):
        raise ValueError("Surface evidence fingerprint mismatch")

    if any(
        report.get("physical_collision_verified") is not False
        for report in (hypotheses, geometry)
    ):
        raise ValueError("Unsupported collision-verification claim")

    records = hypotheses.get("hypotheses")

    if (
        not isinstance(records, list)
        or hypotheses.get("hypothesis_count") != len(records)
    ):
        raise ValueError("Invalid hypothesis collection")

    known_ids = set()
    evidence_records = []

    for item in records:
        if not isinstance(item, dict):
            raise ValueError("Invalid hypothesis item")

        identifier = item.get("id")
        frame = item.get("frame_index")
        kind = item.get("type")

        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in known_ids
            or kind not in ALLOWED_TYPES
            or type(frame) is not int
            or not 0 <= frame <= cutoff
            or item.get("status") != "hypothesis_not_verified"
            or item.get("physical_collision_verified") is not False
            or not isinstance(item.get("observed_evidence"), dict)
            or type(item.get("quality_warning")) is not bool
            or not isinstance(item.get("quality_flags"), list)
            or not isinstance(item.get("limitations"), list)
        ):
            raise ValueError("Invalid hypothesis evidence")

        known_ids.add(identifier)

        evidence_records.append({
            "evidence_id": identifier,
            "frame_index": frame,
            "hypothesis_type": kind,
            "observed_evidence": item["observed_evidence"],
            "candidate_explanation": item.get("inference", ""),
            "quality_warning": item["quality_warning"],
            "quality_flags": item["quality_flags"],
            "limitations": item["limitations"],
            "physical_collision_verified": False,
        })

    visual = geometry.get("visual_geometry")
    relation = geometry.get("object_geometry_relation")

    if not isinstance(visual, dict) or not isinstance(relation, dict):
        raise ValueError("Missing geometry evidence")

    if visual.get("physical_contact_plane_verified") is not False:
        raise ValueError("Contact plane cannot be assumed verified")

    packet_body = {
        "schema": "qapha.muse_evidence_packet.v1",
        "stage": "calibration_only",
        "object_id": object_id,
        "cutoff_frame": cutoff,
        "calibration_sha256": calibration_hash,
        "source_artifacts": {
            "hypotheses_sha256": digest(hypotheses),
            "geometry_sha256": digest(geometry),
            "surface_sha256": digest(surface),
        },
        "scene_observations": {
            "stable_visual_edge": visual.get("stable_edge"),
            "visual_edge_y_px": visual.get(
                "candidate_visual_edge_y_px"
            ),
            "visual_edge_spread_px": visual.get(
                "edge_spread_px"
            ),
            "object_bottom_y_px": relation.get(
                "stable_bottom_y_px"
            ),
            "object_to_visual_edge_offset_px": relation.get(
                "vertical_offset_px"
            ),
            "independent_contact_plane_verified": False,
        },
        "evidence_records": evidence_records,
        "allowed_evidence_ids": sorted(known_ids),
        "reasoning_task": {
            "objective": (
                "Compare plausible explanations for the observed "
                "motion transitions and stable-height movement."
            ),
            "requirements": [
                "Reference only evidence IDs supplied in this packet.",
                "Distinguish direct observations from hypotheses.",
                "Consider alternative explanations and tracking error.",
                "Never claim a physically verified collision.",
                "Do not infer world-scale acceleration or restitution.",
                "Do not claim access to frames beyond the cutoff.",
                "State what additional evidence would test each explanation.",
            ],
            "permitted_conclusions": [
                "possible_surface_interaction",
                "possible_surface_constrained_motion",
                "insufficient_evidence",
            ],
        },
    }

    return {
        **packet_body,
        "packet_sha256": digest(packet_body),
    }


def main():
    parser = argparse.ArgumentParser()

    for name in (
        "hypotheses",
        "geometry",
        "surface",
        "calibration-manifest",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True)

    args = parser.parse_args()

    names = (
        "hypotheses",
        "geometry",
        "surface",
        "calibration_manifest",
    )

    inputs = [
        Path(getattr(args, name)).resolve()
        for name in names
    ]
    output = Path(args.output).resolve()

    if len(set(inputs + [output])) != 5:
        parser.error("Input and output paths must differ")

    data = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in inputs
    ]

    packet = build_packet(*data)

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("x", encoding="utf-8") as stream:
        json.dump(packet, stream, indent=2)

    print(json.dumps({
        "schema": packet["schema"],
        "stage": packet["stage"],
        "object_id": packet["object_id"],
        "cutoff_frame": packet["cutoff_frame"],
        "evidence_count": len(packet["evidence_records"]),
        "allowed_evidence_ids": packet["allowed_evidence_ids"],
        "packet_sha256": packet["packet_sha256"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
