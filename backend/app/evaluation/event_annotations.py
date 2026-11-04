
"""Generate review annotations without exposing Muse conclusions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MOTION_LABELS = {
    "not_reviewed",
    "direction_reversal_visible",
    "approximately_horizontal_visible",
    "no_clear_motion_change",
    "uncertain",
}

CONTACT_LABELS = {
    "not_reviewed",
    "visible_contact_candidate",
    "no_visible_contact",
    "occluded_or_ambiguous",
}

def digest(data):
    content = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def create_annotation_template(packet):
    if not isinstance(packet, dict):
        raise ValueError("Invalid packet")

    if packet.get("schema") != "qapha.muse_evidence_packet.v1":
        raise ValueError("Unsupported packet")

    if packet.get("stage") != "calibration_only":
        raise ValueError("Expected calibration-only packet")

    body = {
        key: value
        for key, value in packet.items()
        if key != "packet_sha256"
    }

    if digest(body) != packet.get("packet_sha256"):
        raise ValueError("Packet fingerprint mismatch")

    cutoff = packet.get("cutoff_frame")
    if type(cutoff) is not int or cutoff < 0:
        raise ValueError("Invalid cutoff")

    records = packet.get("evidence_records")
    if not isinstance(records, list) or not records:
        raise ValueError("No evidence records")

    frames = []

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid evidence record")

        frame = record.get("frame_index")
        if type(frame) is not int or not 0 <= frame <= cutoff:
            raise ValueError("Invalid evidence frame")

        frames.append(frame)

    if len(set(frames)) != len(frames):
        raise ValueError("Multiple records at same frame not supported")

    annotations = []

    for frame in sorted(frames):
        annotations.append({
            "frame_index": frame,
            "reviewed": False,
            "motion_label": "not_reviewed",
            "contact_visibility": "not_reviewed",
            "tracking_quality_issue_visible": None,
            "reviewer_notes": "",
        })

    return {
        "schema": "qapha.event_annotations.v1",
        "stage": "review_pending",
        "source_packet_sha256": packet["packet_sha256"],
        "object_id": packet["object_id"],
        "calibration_cutoff_frame": cutoff,
        "annotation_frames": annotations,
        "allowed_motion_labels": sorted(MOTION_LABELS),
        "allowed_contact_labels": sorted(CONTACT_LABELS),
        "review_protocol": {
            "inspect_frames_before_and_after": True,
            "use_calibration_video_only": True,
            "hide_model_reasoning_from_reviewer": True,
            "do_not_infer_contact_from_direction_change_alone": True,
            "unknown_is_an_acceptable_answer": True,
        },
        "limitations": (
            "This template is not ground truth until independently "
            "reviewed. Visual contact is not proof of physical causality. "
            "Video C is a diagnostic, non-blinded experiment."
        ),
    }


def validate_annotations(data):
    if data.get("schema") != "qapha.event_annotations.v1":
        raise ValueError("Invalid annotation schema")

    frames = data.get("annotation_frames")

    if not isinstance(frames, list) or not frames:
        raise ValueError("Missing annotations")

    seen = set()
    completed = 0

    for entry in frames:
        frame = entry.get("frame_index")

        if type(frame) is not int or frame < 0 or frame in seen:
            raise ValueError("Invalid or duplicate frame")

        seen.add(frame)

        if type(entry.get("reviewed")) is not bool:
            raise ValueError("Invalid reviewed flag")

        if entry.get("motion_label") not in MOTION_LABELS:
            raise ValueError("Invalid motion label")

        if entry.get("contact_visibility") not in CONTACT_LABELS:
            raise ValueError("Invalid contact label")

        if entry.get("tracking_quality_issue_visible") not in (
            None, True, False
        ):
            raise ValueError("Invalid tracking quality assessment")

        if not isinstance(entry.get("reviewer_notes"), str):
            raise ValueError("Invalid reviewer notes")

        if entry["reviewed"]:
            if (
                entry["motion_label"] == "not_reviewed"
                or entry["contact_visibility"] == "not_reviewed"
                or entry["tracking_quality_issue_visible"] is None
            ):
                raise ValueError("Reviewed annotation is incomplete")
            completed += 1

    return {
        "total": len(frames),
        "completed": completed,
        "complete": completed == len(frames),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.packet).resolve()
    output = Path(args.output).resolve()

    if source == output:
        parser.error("Output must differ from source")

    packet = json.loads(source.read_text(encoding="utf-8"))
    result = create_annotation_template(packet)

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)

    print(json.dumps({
        "schema": result["schema"],
        "stage": result["stage"],
        "frames_for_review": [
            entry["frame_index"]
            for entry in result["annotation_frames"]
        ],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
