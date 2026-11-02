
"""Associate post-hoc closest approaches with subsequent vertical reversals.

This is evaluation, not a collision detector or a forecasting input.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def associate_contact_events(
    audit: dict,
    *,
    max_distance_px: float = 10.0,
    max_delay_frames: int = 2,
) -> dict:
    if (
        not isinstance(audit, dict)
        or audit.get("schema") != "qapha.contact_audit.v1"
    ):
        raise ValueError("Invalid contact audit schema")

    if audit.get("stage") != "post_prediction_evaluation_only":
        raise ValueError("Expected post-prediction audit")

    if (
        type(max_distance_px) not in (int, float)
        or not math.isfinite(max_distance_px)
        or max_distance_px < 0
    ):
        raise ValueError("Invalid distance threshold")

    if type(max_delay_frames) is not int or max_delay_frames < 0:
        raise ValueError("Invalid delay threshold")

    oid = audit.get("object_id")
    eid = audit.get("reviewed_edge_id")
    cutoff = audit.get("forecast_cutoff_frame")
    reports = audit.get("audits")

    if (
        not isinstance(oid, str)
        or not oid
        or not isinstance(eid, str)
        or not eid
        or type(cutoff) is not int
        or cutoff < 0
        or not isinstance(reports, list)
    ):
        raise ValueError("Invalid audit metadata")

    results = []

    for index, report in enumerate(reports):
        if (
            not isinstance(report, dict)
            or report.get("physical_collision_verified") is not False
        ):
            raise ValueError("Invalid audit report")

        predicted = report.get("predicted_impact_frame")
        window = report.get("audit_frame_window")
        closest = report.get("closest_bottom_to_edge")
        reversals = report.get("vertical_down_to_up_reversals")

        if (
            type(predicted) not in (int, float)
            or not math.isfinite(predicted)
            or predicted <= cutoff
        ):
            raise ValueError("Invalid predicted frame")

        if (
            not isinstance(window, list)
            or len(window) != 2
            or any(type(v) is not int for v in window)
            or window[0] <= cutoff
            or window[1] < window[0]
        ):
            raise ValueError("Invalid audit window")

        if not isinstance(reversals, list):
            raise ValueError("Invalid reversal observations")

        if closest is None:
            closest_frame = None
            closest_distance = None
        else:
            if not isinstance(closest, dict):
                raise ValueError("Invalid closest observation")

            closest_frame = closest.get("frame_index")
            closest_distance = closest.get(
                "bottom_edge_distance_px"
            )

            if (
                type(closest_frame) is not int
                or not window[0] <= closest_frame <= window[1]
                or type(closest_distance) not in (int, float)
                or not math.isfinite(closest_distance)
                or closest_distance < 0
            ):
                raise ValueError("Invalid closest observation")

        matching = []

        for item in reversals:
            if not isinstance(item, dict):
                raise ValueError("Invalid reversal")

            frame = item.get("frame_index")
            before = item.get("delta_before_px")
            after = item.get("delta_after_px")

            if (
                type(frame) is not int
                or not window[0] <= frame <= window[1]
                or any(
                    type(v) not in (int, float)
                    or not math.isfinite(v)
                    for v in (before, after)
                )
                or before <= 0
                or after >= 0
            ):
                raise ValueError(
                    "Invalid downward-to-upward reversal"
                )

            if (
                closest_frame is not None
                and closest_distance <= max_distance_px
                and 0 <= frame - closest_frame <= max_delay_frames
            ):
                matching.append(frame)

        matching.sort()

        results.append({
            "audit_index": index,
            "predicted_impact_frame": predicted,
            "closest_approach_frame": closest_frame,
            "closest_bottom_edge_distance_px": closest_distance,
            "matched_reversal_frames": matching,
            "association": (
                "candidate_temporal_contact"
                if matching
                else "not_associated"
            ),
            "physical_collision_verified": False,
        })

    return {
        "schema": "qapha.contact_association.v1",
        "stage": "post_prediction_evaluation_only",
        "object_id": oid,
        "edge_id": eid,
        "forecast_cutoff_frame": cutoff,
        "criteria": {
            "max_closest_approach_distance_px":
                max_distance_px,
            "max_approach_to_reversal_delay_frames":
                max_delay_frames,
            "require_reversal_after_or_at_approach": True,
        },
        "associations": results,
        "physical_collision_verified": False,
        "limitations": (
            "Retrospective image-plane association, not proof "
            "of contact. Uses the upstream audit's single "
            "closest approach per impact and observed reversal "
            "triplets. Does not establish 3D geometry, forces, "
            "or a calibrated probability. Do not use verification "
            "observations to change an already frozen forecast."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Associate post-hoc closest approaches and reversals"
        )
    )

    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--max-distance-px", type=float, default=10.0
    )
    parser.add_argument(
        "--max-delay-frames", type=int, default=2
    )

    args = parser.parse_args()

    source = Path(args.audit).resolve()
    destination = Path(args.output).resolve()

    if source == destination:
        parser.error("Input and output must differ")

    result = associate_contact_events(
        json.loads(source.read_text(encoding="utf-8")),
        max_distance_px=args.max_distance_px,
        max_delay_frames=args.max_delay_frames,
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with destination.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
