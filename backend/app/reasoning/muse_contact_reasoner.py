
"""Evidence-constrained Muse Spark reasoning about possible interactions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from app.reasoning.scene_reasoner import MODEL


CONCLUSIONS = {
    "possible_surface_interaction",
    "possible_surface_constrained_motion",
    "insufficient_evidence",
}

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["assessments"],
    "properties": {
        "assessments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "evidence_id",
                    "frame_index",
                    "conclusion",
                    "reason",
                    "alternative_explanation",
                    "next_observation",
                ],
                "properties": {
                    "evidence_id": {"type": "string"},
                    "frame_index": {"type": "integer"},
                    "conclusion": {
                        "type": "string",
                        "enum": sorted(CONCLUSIONS),
                    },
                    "reason": {"type": "string"},
                    "alternative_explanation": {"type": "string"},
                    "next_observation": {"type": "string"},
                },
            },
        },
    },
}


def digest(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prepare_packet(packet):
    """Validate and extract the bounded evidence supplied to Muse."""
    if not isinstance(packet, dict):
        raise ValueError("Invalid evidence packet")

    if packet.get("schema") != "qapha.muse_evidence_packet.v1":
        raise ValueError("Unsupported evidence packet")

    if packet.get("stage") != "calibration_only":
        raise ValueError("Only calibration evidence is permitted")

    cutoff = packet.get("cutoff_frame")
    if type(cutoff) is not int or cutoff < 0:
        raise ValueError("Invalid calibration cutoff")

    object_id = packet.get("object_id")
    if not isinstance(object_id, str) or not object_id:
        raise ValueError("Invalid object ID")

    body = {
        key: value
        for key, value in packet.items()
        if key != "packet_sha256"
    }

    if digest(body) != packet.get("packet_sha256"):
        raise ValueError("Evidence packet checksum mismatch")

    records = packet.get("evidence_records")
    allowed = packet.get("allowed_evidence_ids")

    if not isinstance(records, list) or not isinstance(allowed, list):
        raise ValueError("Missing evidence records")

    if not records:
        raise ValueError("No evidence records to assess")

    if len(records) > 100:
        raise ValueError("Evidence packet exceeds record limit")

    mapping = {}

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid evidence record")

        evidence_id = record.get("evidence_id")
        frame = record.get("frame_index")
        kind = record.get("hypothesis_type")

        if (
            not isinstance(evidence_id, str)
            or not evidence_id
            or evidence_id in mapping
            or type(frame) is not int
            or not 0 <= frame <= cutoff
            or kind not in (
                "possible_surface_interaction",
                "possible_surface_constrained_motion",
            )
            or record.get("physical_collision_verified") is not False
            or type(record.get("quality_warning")) is not bool
            or not isinstance(record.get("observed_evidence"), dict)
            or not isinstance(record.get("quality_flags"), list)
            or not isinstance(record.get("limitations"), list)
        ):
            raise ValueError("Invalid evidence record")

        mapping[evidence_id] = record

    if allowed != sorted(mapping):
        raise ValueError("Allowed evidence IDs do not match records")

    task = packet.get("reasoning_task")
    if not isinstance(task, dict):
        raise ValueError("Missing reasoning task")

    if set(task.get("permitted_conclusions", [])) != CONCLUSIONS:
        raise ValueError("Unexpected permitted conclusions")

    scene = packet.get("scene_observations")
    if not isinstance(scene, dict):
        raise ValueError("Missing scene observations")

    if scene.get("independent_contact_plane_verified") is not False:
        raise ValueError("Unsupported physical contact-plane claim")

    return {
        "packet_sha256": packet["packet_sha256"],
        "object_id": object_id,
        "cutoff_frame": cutoff,
        "scene_observations": scene,
        "evidence_records": records,
        "allowed_evidence_ids": allowed,
        "reasoning_task": task,
    }


def validate_response(proposal, evidence):
    """Reject invented references, missing records and unsupported claims."""
    if (
        not isinstance(proposal, dict)
        or set(proposal) != {"assessments"}
        or not isinstance(proposal["assessments"], list)
    ):
        raise ValueError("Invalid Muse response structure")

    records = {
        record["evidence_id"]: record
        for record in evidence["evidence_records"]
    }

    expected_fields = {
        "evidence_id",
        "frame_index",
        "conclusion",
        "reason",
        "alternative_explanation",
        "next_observation",
    }

    seen = set()
    validated = []

    for item in proposal["assessments"]:
        if not isinstance(item, dict) or set(item) != expected_fields:
            raise ValueError("Unexpected assessment fields")

        evidence_id = item["evidence_id"]

        if evidence_id not in records or evidence_id in seen:
            raise ValueError("Invented or duplicate evidence reference")

        seen.add(evidence_id)
        record = records[evidence_id]

        if (
            type(item["frame_index"]) is not int
            or item["frame_index"] != record["frame_index"]
        ):
            raise ValueError("Evidence frame mismatch")

        conclusion = item["conclusion"]

        if conclusion not in CONCLUSIONS:
            raise ValueError("Unsupported conclusion")

        if (
            conclusion != "insufficient_evidence"
            and conclusion != record["hypothesis_type"]
        ):
            raise ValueError(
                "Conclusion is inconsistent with evidence type"
            )

        for field in (
            "reason",
            "alternative_explanation",
            "next_observation",
        ):
            value = item[field]

            if not isinstance(value, str):
                raise ValueError(
                    f"Invalid {field} for evidence_id={evidence_id}: "
                    f"expected string, received {type(value).__name__}"
                )

            length = len(value.strip())

            if not 1 <= length <= 2000:
                raise ValueError(
                    f"Invalid {field} for evidence_id={evidence_id}: "
                    f"length={length}, allowed=1..2000"
                )

        validated.append({
            **item,
            "physical_collision_verified": False,
            "status": "unverified_reasoning_assessment",
            "input_quality_warning": record["quality_warning"],
        })

    if seen != set(records):
        raise ValueError("Muse did not address every evidence record")

    return {
        "schema": "qapha.muse_contact_reasoning.v1",
        "stage": "calibration_only",
        "model": MODEL,
        "object_id": evidence["object_id"],
        "cutoff_frame": evidence["cutoff_frame"],
        "source_packet_sha256": evidence["packet_sha256"],
        "assessment_count": len(validated),
        "assessments": validated,
        "physical_collision_verified": False,
        "limitations": [
            "Muse explanations are hypotheses, not measurements.",
            "Textual explanations are not independently fact-verified.",
            "Evidence IDs and frame references were validated.",
            "Physical contact was not verified.",
            "No future observations were provided to Muse.",
        ],
    }


def reason(packet, client=None):
    """Call the existing Muse API pattern with validated evidence."""
    evidence = prepare_packet(packet)

    if client is None:
        from openai import OpenAI

        key = os.environ.get("MODEL_API_KEY")
        if not key:
            raise RuntimeError("MODEL_API_KEY is not set")

        client = OpenAI(
            base_url="https://api.meta.ai/v1",
            api_key=key,
            timeout=60.0,
            max_retries=1,
        )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "developer",
                "content": (
                    "You are a cautious physical-world reasoning "
                    "assistant. Use ONLY the supplied evidence packet. "
                    "Return exactly one assessment per evidence record. "
                    "Copy each evidence_id and frame_index exactly. "
                    "Choose only a permitted conclusion, consistent "
                    "with that evidence record's hypothesis_type. "
                    "An insufficient_evidence conclusion is always "
                    "allowed. Never claim verified contact, collision, "
                    "support forces, calibrated gravity, or causality. "
                    "Distinguish observations from inferred mechanisms. "
                    "Discuss plausible alternatives such as tracking "
                    "error, occlusion, projection, or another interaction. "
                    "For quality warnings, explicitly acknowledge "
                    "measurement uncertainty. "
                    "The next_observation must identify additional "
                    "evidence that would help test the hypothesis. "
                    "Do not assume access to future frames. "
                    "Keep reason, alternative_explanation, and "
                    "next_observation concise, preferably under 300 "
                    "characters each. Return only the requested structured JSON."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    evidence, allow_nan=False
                ),
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "qapha_contact_assessments",
                "strict": True,
                "schema": SCHEMA,
            },
        },
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("Muse returned empty content")

    try:
        proposal = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("Muse returned invalid JSON") from exc

    return validate_response(proposal, evidence)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.packet).resolve()
    output = Path(args.output).resolve()

    if source == output:
        parser.error("Output cannot overwrite evidence packet")

    packet = json.loads(source.read_text(encoding="utf-8"))
    result = reason(packet)

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
