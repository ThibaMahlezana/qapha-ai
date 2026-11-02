
"""Evidence-constrained Muse reasoning for temporal scene observations."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

MODEL = "muse-spark-1.3"

CATEGORIES = (
    "motion_only",
    "stationary_near_edge",
    "approaching_candidate_edge",
    "insufficient_evidence",
)

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
                    "object_id", "edge_id", "assessment",
                    "reason", "next_observation",
                ],
                "properties": {
                    "object_id": {"type": "string"},
                    "edge_id": {"type": "string"},
                    "assessment": {
                        "type": "string",
                        "enum": list(CATEGORIES),
                    },
                    "reason": {"type": "string"},
                    "next_observation": {"type": "string"},
                },
            },
        },
    },
}


def prepare_evidence(scene: dict) -> dict:
    if (
        not isinstance(scene, dict)
        or scene.get("schema") != "qapha.temporal_scene.v1"
    ):
        raise ValueError("Unsupported temporal scene schema")

    window = scene.get("frame_window")

    if (
        not isinstance(window, list)
        or len(window) != 2
        or any(type(x) is not int or x < 0 for x in window)
        or window[1] <= window[0]
    ):
        raise ValueError("Invalid frame window")

    edges = scene.get("reviewed_edges")
    objects = scene.get("objects")

    if not isinstance(edges, list) or not isinstance(objects, list):
        raise ValueError("Missing edge/object collections")

    edge_ids = [edge.get("edge_id") for edge in edges]

    if (
        any(not isinstance(e, str) or not e for e in edge_ids)
        or len(edge_ids) != len(set(edge_ids))
    ):
        raise ValueError("Invalid edge IDs")

    for edge in edges:
        if edge.get("physical_surface_verified") is not False:
            raise ValueError("Unexpected verified surface")

    seen = set()
    summaries = []

    for obj in objects:
        oid = obj.get("object_id")
        status = obj.get("status")

        if not isinstance(oid, str) or not oid or oid in seen:
            raise ValueError("Invalid object ID")

        seen.add(oid)

        if status not in (
            "measured_image_motion",
            "insufficient_temporal_coverage",
        ):
            raise ValueError("Unknown motion status")

        relations = obj.get("relationships")

        if not isinstance(relations, list):
            raise ValueError("Missing object relationships")

        by_edge = {}

        for rel in relations:
            eid = rel.get("edge_id")

            if eid not in edge_ids or eid in by_edge:
                raise ValueError("Invalid or duplicate edge relation")

            if rel.get("physical_contact_verified") is not False:
                raise ValueError("Contact is unverified")

            distances = [
                rel.get(k)
                for k in (
                    "distance_first_px",
                    "distance_last_px",
                    "distance_change_px",
                )
            ]

            if any(
                type(v) not in (int, float)
                or not math.isfinite(v)
                for v in distances
            ):
                raise ValueError("Invalid relation distances")

            if any(v < 0 for v in distances[:2]):
                raise ValueError("Negative distance")

            trend = rel.get("distance_trend")

            if trend not in (
                "approaching",
                "receding",
                "approximately_constant_distance",
            ):
                raise ValueError("Unknown trend")

            if type(rel.get("near_at_end")) is not bool:
                raise ValueError("Invalid proximity flag")

            by_edge[eid] = {
                "edge_id": eid,
                "distance_first_px": distances[0],
                "distance_last_px": distances[1],
                "distance_change_px": distances[2],
                "distance_trend": trend,
                "near_at_end": rel["near_at_end"],
            }

        if status == "measured_image_motion":
            if obj.get("motion_class") not in (
                "moving",
                "approximately_stationary",
            ):
                raise ValueError("Unknown motion class")

            if set(by_edge) != set(edge_ids):
                raise ValueError("Incomplete edge observations")

            motion_class = obj["motion_class"]
        else:
            if by_edge:
                raise ValueError(
                    "Gapped tracks must not have temporal relations"
                )

            motion_class = "unknown"

        summaries.append({
            "object_id": oid,
            "status": status,
            "motion_class": motion_class,
            "relationships": [
                by_edge[eid]
                for eid in edge_ids
                if eid in by_edge
            ],
        })

    return {
        "frame_window": window,
        "objects": summaries,
        "limitations": (
            "Image-plane endpoint trends only; no verified "
            "contact or depth. A decreasing distance does not "
            "prove continuous approach. No claim of rolling, "
            "support, collision or physical slope is justified."
        ),
    }


def validate_response(response: dict, evidence: dict) -> dict:
    if not isinstance(response, dict) or set(response) != {
        "assessments"
    }:
        raise ValueError("Invalid response shape")

    items = response["assessments"]

    if not isinstance(items, list):
        raise ValueError("Assessments must be list")

    objects = {
        obj["object_id"]: obj
        for obj in evidence["objects"]
    }

    seen = set()

    for item in items:
        if not isinstance(item, dict) or set(item) != {
            "object_id",
            "edge_id",
            "assessment",
            "reason",
            "next_observation",
        }:
            raise ValueError("Unexpected assessment fields")

        oid = item["object_id"]
        eid = item["edge_id"]
        category = item["assessment"]

        if (
            not isinstance(oid, str)
            or oid not in objects
            or oid in seen
        ):
            raise ValueError("Unknown or duplicate object")

        seen.add(oid)

        if not isinstance(eid, str) or category not in CATEGORIES:
            raise ValueError("Invalid edge or category")

        for key in ("reason", "next_observation"):
            value = item[key]

            if (
                not isinstance(value, str)
                or not 1 <= len(value.strip()) <= 400
            ):
                raise ValueError("Missing or oversized explanation")

        obj = objects[oid]
        relations = {
            rel["edge_id"]: rel
            for rel in obj["relationships"]
        }

        if category in ("motion_only", "insufficient_evidence"):
            if eid != "none":
                raise ValueError(
                    "This category cannot cite an edge"
                )

        elif eid not in relations:
            raise ValueError("Invented edge reference")

        if (
            category == "motion_only"
            and obj["motion_class"] != "moving"
        ):
            raise ValueError("Motion not established")

        if category == "stationary_near_edge" and not (
            obj["motion_class"] == "approximately_stationary"
            and relations[eid]["near_at_end"]
        ):
            raise ValueError(
                "Stationary-near-edge claim not supported"
            )

        if category == "approaching_candidate_edge" and not (
            obj["motion_class"] == "moving"
            and relations[eid]["distance_trend"] == "approaching"
        ):
            raise ValueError(
                "Approaching-edge claim not supported"
            )

    if seen != set(objects):
        raise ValueError("Missing assessments")

    return {
        "schema": "qapha.temporal_reasoning.v1",
        "model": MODEL,
        "frame_window": evidence["frame_window"],
        "assessments": items,
        "status": "unverified_hypotheses",
        "physical_contact_verified": False,
        "note": (
            "Model explanations are hypotheses, "
            "not physical measurements."
        ),
    }


def reason(scene: dict, client=None) -> dict:
    evidence = prepare_evidence(scene)

    if not evidence["objects"]:
        return validate_response(
            {"assessments": []}, evidence
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
                    "Assess every observed object exactly once, "
                    "using only supplied evidence. "
                    "For motion_only and insufficient_evidence "
                    "set edge_id to 'none'. "
                    "For stationary_near_edge cite an edge with "
                    "near_at_end true and stationary motion. "
                    "For approaching_candidate_edge cite an edge "
                    "with distance_trend approaching and "
                    "motion_class moving. "
                    "Endpoint approach is not proof of continuous "
                    "approach. Use insufficient_evidence for "
                    "missing coverage. Never assert confirmed "
                    "contact, support, bounce, slope or 3D "
                    "separation. Describe useful future observations. "
                    "Return only the JSON schema response."
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
                "name": "temporal_assessments",
                "strict": True,
                "schema": SCHEMA,
            },
        },
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("Empty Muse response")

    return validate_response(
        json.loads(content), evidence
    )


def main():
    parser = argparse.ArgumentParser(
        description="Reason about measured temporal scene"
    )

    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    source = Path(args.scene).resolve()
    output = Path(args.output).resolve()

    if source == output:
        parser.error("Source and destination must differ")

    scene = json.loads(
        source.read_text(encoding="utf-8")
    )

    result = reason(scene)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
