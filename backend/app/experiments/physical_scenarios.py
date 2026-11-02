
"""Pixel-space free-flight candidate; never asserts physical contact."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def predict_scenarios(tracks: dict, events: dict, geometry: dict, *,
                      object_id: str = "1", cutoff: int = 49,
                      horizons: tuple[int, ...] = (5, 10, 21),
                      max_history: int = 8,
                      edge_ids: tuple[int, ...] = (0, 1, 2, 3, 4),
                      near_px: float = 30.0) -> dict:
    """Fit linear and vertical-acceleration candidates after a reversal."""
    if tracks.get("outcome") != "completed" or not isinstance(tracks.get("boxes"), list):
        raise ValueError("Invalid SAM tracking data")
    if events.get("schema") != "qapha.motion_events.v1" or not isinstance(events.get("events"), list):
        raise ValueError("Invalid motion-event data")
    if geometry.get("schema") != "qapha.scene_geometry.v1" or not isinstance(geometry.get("candidates"), list):
        raise ValueError("Invalid geometry data")
    if type(cutoff) is not int or cutoff < 4 or not isinstance(object_id, str) or not object_id:
        raise ValueError("Invalid cutoff or object ID")
    if (type(max_history) is not int or max_history < 5 or not horizons or
        any(type(h) is not int or h <= 0 for h in horizons) or len(set(horizons)) != len(horizons)):
        raise ValueError("Invalid forecast horizons or history length")
    if type(near_px) not in (float, int) or not math.isfinite(near_px) or near_px < 0:
        raise ValueError("Invalid proximity threshold")

    selected_edges = []
    if len(edge_ids) != len(set(edge_ids)):
        raise ValueError("Duplicate edge ID")
    for idx in edge_ids:
        if type(idx) is not int or not 0 <= idx < len(geometry["candidates"]):
            raise ValueError("Invalid edge ID")
        a, b = geometry["candidates"][idx]["endpoints_px"]
        pts = np.asarray([a, b], dtype=float)
        if pts.shape != (2, 2) or not np.isfinite(pts).all() or np.allclose(pts[0], pts[1]):
            raise ValueError("Invalid segment")
        selected_edges.append((f"edge_{idx}", pts))

    window = events.get("frame_window")
    if (not isinstance(window, list) or len(window) != 2 or
        any(type(f) is not int for f in window) or window[0] < 0 or
        window[1] > cutoff or window[0] > window[1]):
        raise ValueError("Invalid event frame window")

    reversals = []
    for e in events["events"]:
        f = e.get("frame_index")
        if type(f) is not int or not window[0] <= f <= window[1]:
            raise ValueError("Invalid event frame")
        if (str(e.get("object_id")) == object_id and
            e.get("event") in ("vertical_direction_reversal", "horizontal_direction_reversal")):
            reversals.append(f)

    if not reversals:
        raise ValueError("No observed reversal: this scenario requires an event")
    last_event = max(reversals)

    history = {}
    for box in tracks["boxes"]:
        f = box.get("frame_index")
        if type(f) is not int or f < 0:
            raise ValueError("Invalid frame")
        if f > cutoff or str(box.get("object_id")) != object_id:
            continue
        b = box.get("bbox_xyxy_px")
        if (not isinstance(b, list) or len(b) != 4 or
            any(type(x) not in (float, int) or not math.isfinite(x) for x in b) or
            b[2] <= b[0] or b[3] <= b[1] or f in history):
            raise ValueError("Invalid or duplicate bounds")
        history[f] = np.array([
            (b[0] + b[2]) / 2,
            (b[1] + b[3]) / 2
        ], dtype=float)

    start = max(last_event + 1, cutoff - max_history + 1)
    frames = list(range(start, cutoff + 1))

    if len(frames) < 5 or any(f not in history for f in frames):
        raise ValueError("At least five consecutive post-event samples needed")

    t = np.asarray(frames, dtype=float) - cutoff
    values = np.asarray([history[f] for f in frames])

    design_linear = np.column_stack((np.ones_like(t), t))
    fit_x = np.linalg.lstsq(
        design_linear, values[:, 0], rcond=None
    )[0]
    fit_y_linear = np.linalg.lstsq(
        design_linear, values[:, 1], rcond=None
    )[0]

    design_ballistic = np.column_stack((
        np.ones_like(t),
        t,
        0.5 * t * t
    ))
    fit_y_ballistic = np.linalg.lstsq(
        design_ballistic, values[:, 1], rcond=None
    )[0]

    residual_y = values[:, 1] - design_ballistic @ fit_y_ballistic
    rms = float(np.sqrt(np.mean(residual_y**2)))

    def nearby_edges(x: float, y: float) -> list[dict]:
        matches = []
        for eid, segment in selected_edges:
            a, b = segment
            d = b - a
            u = float(np.clip(
                np.dot(np.array([x, y]) - a, d) / np.dot(d, d),
                0, 1
            ))
            distance = float(np.linalg.norm(
                np.array([x, y]) - (a + u * d)
            ))
            if distance <= near_px:
                matches.append({
                    "edge_id": eid,
                    "distance_px": round(distance, 3)
                })
        return sorted(
            matches, key=lambda e: (e["distance_px"], e["edge_id"])
        )

    forecasts = []

    for h in sorted(horizons):
        x = float(fit_x[0] + fit_x[1] * h)
        linear_y = float(
            fit_y_linear[0] + fit_y_linear[1] * h
        )
        ballistic_y = float(
            fit_y_ballistic[0]
            + fit_y_ballistic[1] * h
            + 0.5 * fit_y_ballistic[2] * h * h
        )

        forecasts.append({
            "frame_index": cutoff + h,
            "linear_center_px": [
                round(x, 6), round(linear_y, 6)
            ],
            "free_flight_candidate_center_px": [
                round(x, 6), round(ballistic_y, 6)
            ],
            "candidate_near_visual_edges_center_only": (
                nearby_edges(x, ballistic_y)
            ),
        })

    return {
        "schema": "qapha.physical_scenarios.v1",
        "object_id": object_id,
        "cutoff_frame": cutoff,
        "last_observed_reversal_frame": last_event,
        "training_frames": [start, cutoff],
        "training_sample_count": len(frames),
        "image_y_acceleration_px_per_frame2": round(
            float(fit_y_ballistic[2]), 6
        ),
        "vertical_fit_rms_px": round(rms, 6),
        "forecasts": forecasts,
        "physical_contact_verified": False,
        "limitations": (
            "Vertical acceleration is apparent image-space curvature, "
            "not calibrated gravity. Screening uses object centre "
            "rather than contact geometry. Visual edges are unverified "
            "surfaces; no rebound is simulated. No fitted quantity "
            "is a calibrated uncertainty interval."
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    for name in ("tracks", "events", "geometry", "output"):
        parser.add_argument("--" + name, required=True)

    parser.add_argument("--object-id", default="1")
    parser.add_argument("--cutoff", type=int, default=49)
    parser.add_argument("--horizons", default="5,10,21")
    args = parser.parse_args()

    paths = [
        Path(getattr(args, name)).resolve()
        for name in ("tracks", "events", "geometry", "output")
    ]

    if len(set(paths)) != 4:
        parser.error("All input and output paths must differ")

    try:
        horizons = tuple(
            int(x.strip()) for x in args.horizons.split(",")
        )
    except ValueError:
        parser.error("--horizons requires comma-separated integers")

    result = predict_scenarios(
        *(
            json.loads(p.read_text(encoding="utf-8"))
            for p in paths[:3]
        ),
        object_id=args.object_id,
        cutoff=args.cutoff,
        horizons=horizons,
    )

    paths[3].parent.mkdir(parents=True, exist_ok=True)
    with paths[3].open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
