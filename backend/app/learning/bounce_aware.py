"""Bounce-aware forecasting: learn the floor by watching, then anticipate.

Builds on the constant-velocity Kalman filter (app.learning.probabilistic)
with two optional ingredients, so each one's contribution can be measured:

  reset         After a very surprising observation, stop trusting the old
                velocity and restart it from the last observed step.
                Generic adaptivity: no knowledge of the scene.
  floor_memory  When the ball's bottom edge is seen to go down then up, a
                floor is recorded at the deepest such point seen so far
                (sampled frames under-estimate contact depth).  Bounciness
                is a fixed assumption (restitution 0.6): estimating it from
                the frames either side of contact proved unreliable.
                Near the floor the forecast becomes a MIXTURE of two competing hypotheses:
                "keeps falling" and "bounces off the floor", weighted by
                the probability of contact.

Everything is causal: a forecast for frame t uses frames < t only, and
the floor is learned only from bounces that have already been observed.
Development module: settings are not yet registered for any test.
"""

from __future__ import annotations

import math

import numpy as np

from app.learning.probabilistic import CHI2_2DOF, gaussian_score

VARIANTS = {
    "kalman": {"reset": False, "floor_memory": False},
    "kalman+reset": {"reset": True, "floor_memory": False},
    "kalman+reset+floor": {"reset": True, "floor_memory": True},
}


def _phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def boxes_from_tracks(tracks):
    boxes = sorted(tracks["boxes"], key=lambda b: b["frame_index"])
    if [b["frame_index"] for b in boxes] != list(range(len(boxes))):
        raise ValueError("frames must be consecutive from 0")
    if len({str(b["object_id"]) for b in boxes}) != 1:
        raise ValueError("expected one tracked object")
    return [b["bbox_xyxy_px"] for b in boxes]


def observed_bounces(boxes, min_rise_px=1.0):
    """Frames where the bottom edge went down then came back up."""
    bottom = [b[3] for b in boxes]
    return [t - 1 for t in range(2, len(bottom))
            if bottom[t - 2] < bottom[t - 1]
            and bottom[t] < bottom[t - 1] - min_rise_px]


def bounce_aware(boxes, q=50.0, r=4.0, *, reset=True, floor_memory=True,
                 restitution=0.6, reset_d2=30.0, min_rise_px=1.0):
    """Yield one forecast per frame t >= 2, made from frames < t."""
    if q <= 0 or r <= 0:
        raise ValueError("q and r must be positive")
    F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    H = np.eye(2, 4)
    g = np.array([[0.5, 0, 1, 0], [0, 0.5, 0, 1]], float).T
    Q = q * g @ g.T
    R = r * np.eye(2)
    centres = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
    bottom = [b[3] for b in boxes]
    half = [(b[3] - b[1]) / 2 for b in boxes]

    def fresh(pos, vel):
        return (np.array([pos[0], pos[1], vel[0], vel[1]], float),
                np.diag([r, r, 2 * r + q, 2 * r + q]))

    c0, c1 = centres[0], centres[1]
    x, P = fresh(c1, (c1[0] - c0[0], c1[1] - c0[1]))
    floor, e = None, restitution  # restitution is an assumption, not learned
    out = []
    for t in range(2, len(centres)):
        x = F @ x
        P = F @ P @ F.T + Q
        S = H @ P @ H.T + R
        m = H @ x
        components = [(1.0, m.copy(), S.copy())]
        p_bounce = None
        if floor_memory and floor is not None:
            predicted_bottom = m[1] + half[t - 1]
            sigma_y = math.sqrt(S[1, 1])
            p_bounce = min(max(_phi((predicted_bottom - floor) / sigma_y),
                               1e-3), 1 - 1e-3)
            bounced = np.array([
                m[0],
                floor - e * max(predicted_bottom - floor, 0.0) - half[t - 1],
            ])
            components = [(1 - p_bounce, m.copy(), S.copy()),
                          (p_bounce, bounced, S.copy())]
        point = max(components, key=lambda c: c[0])[1]
        out.append({"frame": t, "components": components, "point": point,
                    "p_bounce": p_bounce, "floor_bottom_px": floor})

        # Only now is frame t observed.
        z = np.asarray(centres[t], float)
        lik = [w * math.exp(-gaussian_score(mu, cov, z)[0])
               for w, mu, cov in components]
        d2 = gaussian_score(m, S, z)[1]
        if len(components) == 2 and lik[1] > lik[0]:
            x, P = fresh(z, (x[2], -e * abs(x[3])))
        elif reset and d2 > reset_d2:
            prev = centres[t - 1]
            x, P = fresh(z, (z[0] - prev[0], z[1] - prev[1]))
        else:
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ (z - H @ x)
            P = (np.eye(4) - K @ H) @ P

        # A bounce seen at frame t-1 teaches where the floor is.  Frames
        # rarely land exactly on the moment of contact, so each observed
        # low point UNDER-estimates the floor's depth: keep the deepest.
        if (floor_memory and bottom[t - 2] < bottom[t - 1]
                and bottom[t] < bottom[t - 1] - min_rise_px):
            floor = bottom[t - 1] if floor is None else max(floor, bottom[t - 1])
    return out


def mixture_nll(components, observed):
    total = sum(w * math.exp(-gaussian_score(mu, cov, observed)[0])
                for w, mu, cov in components)
    return -math.log(max(total, 1e-300))


def score(forecasts, boxes):
    """Point error, mixture NLL, approximate 90 % coverage, bounce detail."""
    centres = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
    rows = []
    for f in forecasts:
        z = np.asarray(centres[f["frame"]], float)
        inside = any(gaussian_score(mu, cov, z)[1] <= CHI2_2DOF["90"]
                     for w, mu, cov in f["components"] if w >= 0.2)
        rows.append({"frame": f["frame"],
                     "error_px": float(np.linalg.norm(z - f["point"])),
                     "nll": mixture_nll(f["components"], z),
                     "inside90": inside,
                     "p_bounce": f["p_bounce"]})
    n = len(rows)
    by_frame = {r["frame"]: r for r in rows}
    bounces = []
    for i, b in enumerate(observed_bounces(boxes)):
        window = [by_frame[t] for t in (b, b + 1, b + 2) if t in by_frame]
        bounces.append({
            "bounce_frame": b,
            "order": "first" if i == 0 else "later",
            "mean_error_px": round(sum(r["error_px"] for r in window)
                                   / len(window), 3),
            "p_bounce_at_contact": (None if b not in by_frame
                                    or by_frame[b]["p_bounce"] is None
                                    else round(by_frame[b]["p_bounce"], 3)),
        })
    return {
        "evaluated_predictions": n,
        "mean_error_px": round(sum(r["error_px"] for r in rows) / n, 6),
        "mean_nll": round(sum(r["nll"] for r in rows) / n, 6),
        "inside90_approx": round(sum(r["inside90"] for r in rows) / n, 4),
        "bounces": bounces,
        "rows": rows,
    }


def compare(boxes):
    return {name: score(bounce_aware(boxes, **kw), boxes)
            for name, kw in VARIANTS.items()}


def main():
    """Development comparison: python -m app.learning.bounce_aware --tracks X"""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", required=True, action="append")
    args = parser.parse_args()
    for path in args.tracks:
        try:
            boxes = boxes_from_tracks(
                json.loads(Path(path).read_text(encoding="utf-8")))
        except ValueError as problem:
            print(Path(path).name, "| skipped:", problem)
            continue
        print(Path(path).name, "| observed bounces at frames",
              observed_bounces(boxes))
        for name, s in compare(boxes).items():
            later = [b for b in s["bounces"] if b["order"] == "later"]
            later_err = (round(sum(b["mean_error_px"] for b in later)
                               / len(later), 1) if later else None)
            first_err = next((b["mean_error_px"] for b in s["bounces"]
                              if b["order"] == "first"), None)
            print(f"  {name:20s} error={s['mean_error_px']:6.2f}  "
                  f"nll={s['mean_nll']:6.2f}  inside90~{s['inside90_approx']:.0%}  "
                  f"first_bounce_err={first_err}  later_bounces_err={later_err}")
            if name.endswith("floor"):
                print("    p(bounce) at each contact:",
                      [(b["bounce_frame"], b["p_bounce_at_contact"])
                       for b in s["bounces"]])


if __name__ == "__main__":
    main()