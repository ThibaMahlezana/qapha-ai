
"""Step 7C: Evidence-constrained Muse Spark scene reasoning."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

MODEL = "muse-spark-1.3"

ALLOWED = {
    "proximity_only",
    "possible_contact",
    "insufficient_evidence",
}

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["hypotheses"],
    "properties": {
        "hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "object_id",
                    "edge_id",
                    "hypothesis",
                    "confidence",
                    "evidence_frame_index",
                    "reason",
                    "next_observation",
                ],
                "properties": {
                    "object_id": {"type": "string"},
                    "edge_id": {"type": "string"},
                    "hypothesis": {
                        "type": "string",
                        "enum": sorted(ALLOWED),
                    },
                    "confidence": {
                        "type": "number",
                    },
                    "evidence_frame_index": {
                        "type": "integer",
                    },
                    "reason": {"type": "string"},
                    "next_observation": {"type": "string"},
                },
            },
        },
    },
}


def prepare_evidence(graph: dict) -> dict:
    """Expose only existing scene-graph evidence."""

    if graph.get("schema") != "qapha.scene_graph.v1":
        raise ValueError("Invalid scene graph schema")

    frame = graph.get("frame_index")
    objects = graph.get("tracked_objects")
    edges = graph.get("reviewed_edges")
    relations = graph.get("relationships")

    if type(frame) is not int or frame < 0:
        raise ValueError("Invalid evidence frame")

    if not all(isinstance(x, list) for x in (
        objects, edges, relations
    )):
        raise ValueError("Missing scene graph collections")

    object_ids = [str(o["object_id"]) for o in objects]
    edge_ids = [str(e["edge_id"]) for e in edges]

    if len(set(object_ids)) != len(object_ids):
        raise ValueError("Duplicate object IDs")

    if len(set(edge_ids)) != len(edge_ids):
        raise ValueError("Duplicate edge IDs")

    evidence = []

    for relation in relations:
        oid = str(relation["subject"])
        eid = str(relation["object"])
        distance = relation["distance_px"]
        threshold = relation["threshold_px"]

        if oid not in object_ids or eid not in edge_ids:
            raise ValueError("Relationship references unknown node")

        if relation.get("predicate") != "image_proximity_to_edge":
            raise ValueError("Unsupported relationship")

        if relation.get("evidence_frame_index") != frame:
            raise ValueError("Relationship frame mismatch")

        if relation.get("physical_contact_verified") is not False:
            raise ValueError("Only unverified contacts supported")

        if not all(
            type(x) in (int, float)
            and math.isfinite(x)
            and x >= 0
            for x in (distance, threshold)
        ):
            raise ValueError("Invalid distance")

        if distance > threshold:
            raise ValueError("Relationship exceeds threshold")

        evidence.append({
            "object_id": oid,
            "edge_id": eid,
            "distance_px": distance,
            "threshold_px": threshold,
            "frame_index": frame,
            "contact_verified": False,
        })

    return {
        "frame_index": frame,
        "coordinate_system": "image_pixels_y_down",
        "relationships": evidence,
        "limitations": (
            "Single-frame 2D proximity only. "
            "No verified contact, support, depth, "
            "surface material, or physical slope."
        ),
    }


def validate_proposal(proposal: dict, evidence: dict) -> dict:
    """Reject ungrounded references and unsupported fields."""

    if not isinstance(proposal, dict) or set(proposal) != {
        "hypotheses"
    }:
        raise ValueError("Invalid proposal structure")

    hypotheses = proposal["hypotheses"]

    if not isinstance(hypotheses, list):
        raise ValueError("Hypotheses must be a list")

    allowed_pairs = {
        (r["object_id"], r["edge_id"])
        for r in evidence["relationships"]
    }

    seen = set()

    for item in hypotheses:
        if not isinstance(item, dict) or set(item) != {
            "object_id", "edge_id", "hypothesis",
            "confidence", "evidence_frame_index",
            "reason", "next_observation",
        }:
            raise ValueError("Invalid hypothesis fields")

        pair = (item["object_id"], item["edge_id"])

        if pair not in allowed_pairs or pair in seen:
            raise ValueError("Invented or duplicate evidence reference")

        seen.add(pair)

        if item["hypothesis"] not in ALLOWED:
            raise ValueError("Unsupported hypothesis")

        if item["evidence_frame_index"] != evidence["frame_index"]:
            raise ValueError("Incorrect evidence frame")

        confidence = item["confidence"]

        if (
            type(confidence) not in (int, float)
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("Invalid confidence")

        for key in ("reason", "next_observation"):
            if (
                not isinstance(item[key], str)
                or not 1 <= len(item[key].strip()) <= 500
            ):
                raise ValueError(f"Invalid {key}")

    # Do not accept responses that silently omit relationships.
    if seen != allowed_pairs:
        raise ValueError("Proposal did not address all relationships")

    return {
        "schema": "qapha.scene_reasoning.v1",
        "model": MODEL,
        "frame_index": evidence["frame_index"],
        "hypotheses": hypotheses,
        "physical_contact_verified": False,
        "status": "unverified_hypotheses",
        "note": (
            "Muse hypotheses are not measurements. "
            "Confidence values are model estimates, "
            "not calibrated probabilities."
        ),
    }


def reason(graph: dict, client=None) -> dict:
    """Call Muse with bounded evidence and validate its reply."""

    evidence = prepare_evidence(graph)

    if not evidence["relationships"]:
        return validate_proposal(
            {"hypotheses": []}, evidence
        )

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
                    "You are a cautious physical-world hypothesis "
                    "generator. Use ONLY the supplied observations. "
                    "Return one hypothesis per listed relationship. "
                    "Never invent object IDs, edges or measurements. "
                    "Image proximity alone does NOT prove contact, "
                    "support, collision, slope or gravity. "
                    "Prefer proximity_only or insufficient_evidence "
                    "unless another conclusion is clearly justified. "
                    "Treat your confidence as uncalibrated. "
                    "The next_observation must describe a useful "
                    "additional measurement, not an action."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(evidence),
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "scene_hypotheses",
                "strict": True,
                "schema": SCHEMA,
            },
        },
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("Muse returned no content")

    return validate_proposal(json.loads(content), evidence)


def main():
    parser = argparse.ArgumentParser(
        description="Generate evidence-constrained scene hypotheses"
    )
    parser.add_argument("--graph", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.graph).resolve()
    output = Path(args.output).resolve()

    if output == source:
        parser.error("Output cannot overwrite input")

    graph = json.loads(source.read_text(encoding="utf-8"))
    result = reason(graph)

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
