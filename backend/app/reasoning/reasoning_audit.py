
"""Audit Muse assessments against their original evidence packet."""

import argparse
import json
from collections import Counter
from pathlib import Path

from app.reasoning.muse_contact_reasoner import (
    digest,
    prepare_packet,
    validate_response,
)


PROPOSAL_FIELDS = (
    "evidence_id",
    "frame_index",
    "conclusion",
    "reason",
    "alternative_explanation",
    "next_observation",
)


def audit_reasoning(packet, reasoning):
    evidence = prepare_packet(packet)

    if reasoning.get("schema") != "qapha.muse_contact_reasoning.v1":
        raise ValueError("Unexpected reasoning schema")

    if reasoning.get("stage") != "calibration_only":
        raise ValueError("Reasoning must use calibration evidence")

    if reasoning.get("source_packet_sha256") != evidence["packet_sha256"]:
        raise ValueError("Reasoning packet fingerprint mismatch")

    if reasoning.get("object_id") != evidence["object_id"]:
        raise ValueError("Reasoning object mismatch")

    if reasoning.get("cutoff_frame") != evidence["cutoff_frame"]:
        raise ValueError("Reasoning cutoff mismatch")

    if reasoning.get("physical_collision_verified") is not False:
        raise ValueError("Unsupported verified collision claim")

    assessments = reasoning.get("assessments")

    if not isinstance(assessments, list):
        raise ValueError("Missing reasoning assessments")

    if reasoning.get("assessment_count") != len(assessments):
        raise ValueError("Assessment count mismatch")

    # Revalidate the model's original structured fields.
    proposal = {"assessments": []}

    for assessment in assessments:
        if not isinstance(assessment, dict):
            raise ValueError("Invalid assessment")

        if set(assessment) != set(PROPOSAL_FIELDS) | {
            "physical_collision_verified",
            "status",
            "input_quality_warning",
        }:
            raise ValueError("Unexpected assessment fields")

        proposal["assessments"].append({
            key: assessment[key]
            for key in PROPOSAL_FIELDS
        })

    reproduced = validate_response(proposal, evidence)

    if reasoning != reproduced:
        raise ValueError(
            "Saved reasoning differs from validated Muse response"
        )

    records = {
        record["evidence_id"]: record
        for record in evidence["evidence_records"]
    }

    review_items = []
    conclusion_counts = Counter()
    results = []

    for assessment in assessments:
        identifier = assessment["evidence_id"]
        record = records[identifier]
        conclusion = assessment["conclusion"]

        conclusion_counts[conclusion] += 1

        reasons = []

        if record["quality_warning"]:
            reasons.append("input_tracking_quality_warning")

        if conclusion != "insufficient_evidence":
            reasons.append("physical_hypothesis_requires_review")

        results.append({
            "evidence_id": identifier,
            "frame_index": assessment["frame_index"],
            "input_hypothesis_type": record["hypothesis_type"],
            "model_conclusion": conclusion,
            "evidence_reference_valid": True,
            "frame_reference_valid": True,
            "input_quality_warning_preserved": (
                assessment["input_quality_warning"]
                == record["quality_warning"]
            ),
            "physical_collision_verified": False,
            "review_reasons": reasons,
        })

        if reasons:
            review_items.append({
                "evidence_id": identifier,
                "reasons": reasons,
            })

    return {
        "schema": "qapha.reasoning_audit.v1",
        "stage": "calibration_only",
        "source_packet_sha256": evidence["packet_sha256"],
        "reasoning_sha256": digest(reasoning),
        "object_id": evidence["object_id"],
        "cutoff_frame": evidence["cutoff_frame"],
        "evidence_count": len(records),
        "assessment_count": len(assessments),
        "coverage_complete": True,
        "reference_validation_passed": True,
        "structural_validation_passed": True,
        "conclusion_counts": dict(conclusion_counts),
        "results": results,
        "review_items": review_items,
        "manual_review_required": bool(review_items),
        "physical_collision_verified": False,
        "limitations": [
            "Structural validity does not establish scientific truth.",
            "The audit does not independently verify explanatory prose.",
            "Review flags do not establish that an assessment is wrong.",
            "Physical hypotheses require additional observation.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", required=True)
    parser.add_argument("--reasoning", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = [
        Path(args.packet).resolve(),
        Path(args.reasoning).resolve(),
        Path(args.output).resolve(),
    ]

    if len(set(paths)) != 3:
        parser.error("Input and output paths must differ")

    packet = json.loads(paths[0].read_text(encoding="utf-8"))
    reasoning = json.loads(paths[1].read_text(encoding="utf-8"))

    report = audit_reasoning(packet, reasoning)

    paths[2].parent.mkdir(parents=True, exist_ok=True)

    with paths[2].open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
