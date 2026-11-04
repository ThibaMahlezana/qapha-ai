"""Build a frame-by-frame replay timeline for the laboratory UI.

For each frame the timeline holds the observed SAM box and, from frame 2
on, every model's forecast made BEFORE that frame was observed:
  persistence, linear (window 6), adaptive (the selector's choice),
  policy (warning-triggered persistence), constant_velocity_2pt, alpha_beta.
The adaptive learner runs through prediction_protocol, so each adaptive
forecast carries the fingerprint it was frozen with.

This is a read-only view: it reuses the existing learners unchanged and
does not alter any registered experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from app.learning.prediction_protocol import (
    create_protocol,
    freeze_next_prediction,
    observe_and_update,
)
from app.learning.bounce_aware import bounce_aware
from app.learning.probabilistic import CHI2_2DOF, gaussian_score, kalman_cv
from app.learning.reference_baselines import alpha_beta, constant_velocity_2pt
from app.learning.warning_policy import apply_policy


SCHEMA = "qapha.lab_timeline.v1"
MODELS = ("persistence", "linear", "adaptive", "policy",
          "constant_velocity_2pt", "alpha_beta", "kalman_cv", "bounce_aware")
WINDOW, DECAY = 6, 0.85
KALMAN_Q, KALMAN_R = 50.0, 4.0  # set on development videos C and E


def zone_ellipse(cov, level="90"):
    """Semi-axes (px) and rotation (deg) of a Gaussian prediction ellipse."""
    values, vectors = np.linalg.eigh(np.asarray(cov, float))
    scale = CHI2_2DOF[level]
    major = int(np.argmax(values))
    vx, vy = vectors[:, major]
    return {
        "rx_px": round(math.sqrt(scale * values[major]), 4),
        "ry_px": round(math.sqrt(scale * values[1 - major]), 4),
        "angle_deg": round(math.degrees(math.atan2(vy, vx)), 4),
    }


def slice_tracks(tracks, first_frame, last_frame):
    """Keep frames first..last of the most-tracked object, renumbered from 0."""
    per_id = {}
    for b in tracks.get("boxes", []):
        per_id.setdefault(str(b["object_id"]), []).append(b)
    oid = max(per_id, key=lambda k: (len(per_id[k]), k))
    kept = [dict(b, frame_index=b["frame_index"] - first_frame)
            for b in per_id[oid] if first_frame <= b["frame_index"] <= last_frame]
    return {**tracks, "boxes": kept}


def _sorted_boxes(tracks):
    if tracks.get("model") != "sam-3.1":
        raise ValueError("unexpected tracker model")
    boxes = sorted(tracks.get("boxes", []), key=lambda b: b["frame_index"])
    if len({str(b["object_id"]) for b in boxes}) != 1:
        raise ValueError("timeline supports exactly one tracked object")
    if [b["frame_index"] for b in boxes] != list(range(len(boxes))):
        raise ValueError("frames must be consecutive from 0")
    if len(boxes) < 3:
        raise ValueError("at least three frames required")
    return boxes


def _centre(box):
    x0, y0, x1, y1 = box["bbox_xyxy_px"]
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def _round(xy):
    return [round(float(xy[0]), 4), round(float(xy[1]), 4)]


def _stats(errors):
    return {"mean_error_px": round(sum(errors) / len(errors), 6),
            "maximum_error_px": round(max(errors), 6),
            "errors_over_25px": sum(e > 25 for e in errors)}


def build_timeline(tracks, *, video=None):
    boxes = _sorted_boxes(tracks)
    points = [_centre(b) for b in boxes]
    object_id = str(boxes[0]["object_id"])

    # Adaptive learner through the frozen-prediction protocol.
    protocol = create_protocol(object_id, WINDOW, DECAY)
    frozen = {}
    events = []
    for frame, xy in enumerate(points):
        pending = None
        if frame >= 2:
            protocol, pending = freeze_next_prediction(protocol)
        protocol, evaluation = observe_and_update(protocol, frame, list(xy))
        if pending is not None:
            frozen[frame] = pending
            events.append({"frame_index": frame,
                           "history_cutoff_frame": frame - 1,
                           "selection": pending["selection"],
                           "errors_px": evaluation["errors_px"]})

    policy = {r["frame"]: r for r in apply_policy(events)}
    cv2 = dict(constant_velocity_2pt(points))
    ab = dict(alpha_beta(points))
    kalman = {t: (mean, cov) for t, mean, cov in
              kalman_cv(points, KALMAN_Q, KALMAN_R)}
    bounce = {f["frame"]: f for f in
              bounce_aware([b["bbox_xyxy_px"] for b in boxes])}
    zone_rows = []

    frames, errors = [], {m: [] for m in MODELS}
    for frame, box in enumerate(boxes):
        observed = points[frame]
        entry = {"frame": frame,
                 "observed": {"bbox_xyxy_px": box["bbox_xyxy_px"],
                              "centre_xy": _round(observed)},
                 "predictions": None}
        if frame in frozen:
            p = frozen[frame]
            experts = p["predictions_xy"]
            rec = policy[frame]
            # Errors use the exact values each model produced (the
            # protocol's experts are already stored at 4 decimals);
            # rounding below is for display only.
            xy = {
                "persistence": experts["persistence"],
                "linear": experts["linear"],
                "adaptive": experts[p["selection"]],
                "policy": experts[rec["candidate_model"]],
                "constant_velocity_2pt": cv2[frame],
                "alpha_beta": ab[frame],
                "kalman_cv": tuple(kalman[frame][0]),
                "bounce_aware": tuple(bounce[frame]["point"]),
            }
            entry["predictions"] = {}
            for m in MODELS:
                e = math.hypot(xy[m][0] - observed[0], xy[m][1] - observed[1])
                errors[m].append(e)
                entry["predictions"][m] = {"xy": _round(xy[m]),
                                           "error_px": round(e, 6)}
            entry["adaptive"] = {
                "selection": p["selection"],
                "scores_before_observation": p["scores_before_observation"],
                "prediction_sha256": p["prediction_sha256"],
            }
            mean, cov = kalman[frame]
            nll, d2 = gaussian_score(mean, cov, observed)
            ellipse = zone_ellipse(cov, "90")
            entry["zone"] = {
                "model": "kalman_cv",
                "level": "90",
                "centre_xy": _round(mean),
                **ellipse,
                "observed_inside": d2 <= CHI2_2DOF["90"],
                "nll": round(nll, 6),
            }
            zone_rows.append((d2, nll))
            b = bounce[frame]
            entry["bounce"] = {
                "floor_bottom_px": b["floor_bottom_px"],
                "p_bounce": (None if b["p_bounce"] is None
                             else round(b["p_bounce"], 4)),
                "hypotheses": [
                    {"name": ("keeps falling" if i == 0 else "bounces"),
                     "weight": round(w, 4), "centre_xy": _round(mu),
                     **zone_ellipse(cov, "90")}
                    for i, (w, mu, cov) in enumerate(b["components"])
                ],
            }
            entry["policy"] = {"warning": rec["warning"],
                               "override_active": rec["override_active"],
                               "selection": rec["candidate_model"]}
        frames.append(entry)

    return {
        "schema": SCHEMA,
        "source": {"concept": tracks.get("concept"),
                   "object_id": object_id,
                   "tracker": tracks.get("model")},
        "video": video,
        "frame_count": len(frames),
        "first_predicted_frame": 2,
        "models": list(MODELS),
        "parameters": {"window": WINDOW, "decay": DECAY,
                       "policy": "warning_triggered_persistence (defaults)",
                       "kalman_cv": {"q": KALMAN_Q, "r": KALMAN_R},
                       "bounce_aware": "kalman+reset+floor, locked settings"},
        "summary": {m: _stats(errors[m]) for m in MODELS},
        "zone_summary": {
            "model": "kalman_cv",
            "coverage": {k: round(sum(d2 <= v for d2, _ in zone_rows)
                                  / len(zone_rows), 4)
                         for k, v in CHI2_2DOF.items()},
            "mean_nll": round(sum(n for _, n in zone_rows) / len(zone_rows), 6),
        },
        "warning_frames": [f for f, r in sorted(policy.items()) if r["warning"]],
        "final_learning_state_sha256": protocol["session"]["state_sha256"],
        "frames": frames,
        "limitations": [
            "Image-pixel coordinates of SAM box centres, not physical positions.",
            "One-frame-ahead forecasts only.",
            "Kalman zone settings (q=50, r=4) were chosen on development "
            "videos C and E; they are not independently validated.",
            "Read-only replay of recorded tracks; not a live stream.",
            "Bounce-aware model: the floor is learned from observed bounces; "
            "its bounce detector can be fooled by small wobbles (see tests "
            "v1 and v2).",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--video", help="video file shown in the lab")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--first-frame", type=int, default=None,
                        help="use only frames first..last of the clip")
    parser.add_argument("--last-frame", type=int, default=None)
    parser.add_argument("--label", default=None, help="short clip description")
    args = parser.parse_args()

    source, output = Path(args.tracks).resolve(), Path(args.output).resolve()
    if source == output:
        parser.error("output cannot overwrite tracks")
    raw = source.read_bytes()
    tracks = json.loads(raw)
    offset = 0
    if args.first_frame is not None:
        offset = args.first_frame
        last = args.last_frame if args.last_frame is not None else 10**9
        tracks = slice_tracks(tracks, offset, last)
    video = None
    if args.video:
        # Record the video's path relative to datasets/ so clips with the
        # same file name in different folders can never be confused.
        video_path = Path(args.video).resolve()
        datasets = Path(__file__).resolve().parents[3] / "datasets"
        try:
            ref = video_path.relative_to(datasets.resolve()).as_posix()
        except ValueError:
            ref = video_path.name
        video = {"file": ref, "fps": args.fps,
                 "resolution_px": [args.width, args.height],
                 "frame_offset": offset}
    timeline = build_timeline(tracks, video=video)
    if args.label:
        timeline["source"]["label"] = args.label
    timeline["source"]["tracks_file"] = source.name
    timeline["source"]["tracks_sha256"] = hashlib.sha256(raw).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    print(json.dumps({"frames": timeline["frame_count"],
                      "summary": timeline["summary"],
                      "warning_frames": timeline["warning_frames"]}, indent=2))
    print("Saved:", output)


if __name__ == "__main__":
    main()