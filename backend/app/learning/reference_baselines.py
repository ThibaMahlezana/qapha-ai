"""Fixed, non-adaptive reference forecasters for registered comparisons.

Each forecaster predicts frame t using only frames < t (one step ahead,
image-pixel box centres).  Neither model learns which predictor to trust;
they exist to test whether adaptive selection beats simple, well-known
alternatives.

constant_velocity_2pt: x_t = 2 * x_(t-1) - x_(t-2)
alpha_beta:            fixed-gain constant-velocity tracking filter
                       (alpha=0.85, beta=0.5, dt=1 frame), initialised
                       from frames 0 and 1.  Gains were set once during
                       exploratory review of Video C and are not tuned.
"""

from __future__ import annotations

import math

ALPHA = 0.85
BETA = 0.5


def track_centres(track_data):
    """Validated SAM box centres for one object, frames 0..N-1."""
    if track_data.get("model") != "sam-3.1":
        raise ValueError("unexpected tracker model")
    boxes = track_data.get("boxes")
    if not isinstance(boxes, list) or len(boxes) < 3:
        raise ValueError("at least three boxes required")
    if len({str(b.get("object_id")) for b in boxes}) != 1:
        raise ValueError("expected one tracked object")
    centres = []
    for expected, box in enumerate(boxes):
        if box.get("frame_index") != expected:
            raise ValueError("frames must be consecutive from 0")
        b = box.get("bbox_xyxy_px")
        if (not isinstance(b, list) or len(b) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v)
                       for v in b)
                or b[0] >= b[2] or b[1] >= b[3]):
            raise ValueError("invalid bounding box")
        centres.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
    return centres


def constant_velocity_2pt(centres):
    """Return [(frame, predicted_xy)] for frames 2..N-1."""
    out = []
    for t in range(2, len(centres)):
        (x1, y1), (x2, y2) = centres[t - 2], centres[t - 1]
        out.append((t, (2 * x2 - x1, 2 * y2 - y1)))
    return out


def alpha_beta(centres, alpha=ALPHA, beta=BETA):
    """Return [(frame, predicted_xy)] for frames 2..N-1."""
    if not (0 < alpha <= 1 and 0 < beta <= 2):
        raise ValueError("invalid gains")
    x = list(centres[1])
    v = [centres[1][0] - centres[0][0], centres[1][1] - centres[0][1]]
    out = []
    for t in range(2, len(centres)):
        pred = (x[0] + v[0], x[1] + v[1])
        out.append((t, pred))
        # Observation of frame t is used only after its forecast exists.
        r = (centres[t][0] - pred[0], centres[t][1] - pred[1])
        x = [pred[0] + alpha * r[0], pred[1] + alpha * r[1]]
        v = [v[0] + beta * r[0], v[1] + beta * r[1]]
    return out


FORECASTERS = {
    "constant_velocity_2pt": constant_velocity_2pt,
    "alpha_beta": alpha_beta,
}


def evaluate_baselines(track_data):
    centres = track_centres(track_data)
    result = {"schema": "qapha.reference_baselines.v1",
              "parameters": {"alpha": ALPHA, "beta": BETA}}
    for name, fn in FORECASTERS.items():
        errors = []
        records = []
        for frame, (px, py) in fn(centres):
            ox, oy = centres[frame]
            e = math.hypot(px - ox, py - oy)
            errors.append(e)
            records.append({"frame": frame,
                            "predicted_xy": [round(px, 4), round(py, 4)],
                            "error_px": round(e, 6)})
        result[name] = {
            "evaluated_predictions": len(errors),
            "mean_error_px": round(sum(errors) / len(errors), 6),
            "maximum_error_px": round(max(errors), 6),
            "errors_over_25px": sum(e > 25 for e in errors),
            "records": records,
        }
    return result