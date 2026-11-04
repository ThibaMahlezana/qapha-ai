"""Probabilistic one-step forecasters and proper scoring.

Each forecaster predicts frame t from frames < t as a 2-D Gaussian
(mean, covariance) in image pixels, so its uncertainty can be scored:

  * nll        negative log-likelihood of the observed centre (lower = better);
               punishes confident misses, rewards honest uncertainty.
  * coverage   how often the observation falls inside the 50/90/95 %
               prediction ellipse; a calibrated model matches those rates.

Models
  kalman_cv        constant-velocity Kalman filter (white-noise acceleration).
  cv2pt_empirical  the two-point constant-velocity guess with an isotropic
                   variance learned causally from its own past errors.

Development tool: parameters are not yet registered for any test.
"""

from __future__ import annotations

import math

import numpy as np

# Chi-square (2 degrees of freedom) quantiles for 50 / 90 / 95 % ellipses.
CHI2_2DOF = {"50": 1.386294, "90": 4.605170, "95": 5.991465}


def gaussian_score(mean, cov, observed):
    """Return (nll, squared Mahalanobis distance) for a 2-D Gaussian."""
    diff = np.asarray(observed, float) - np.asarray(mean, float)
    cov = np.asarray(cov, float)
    det = float(np.linalg.det(cov))
    if not det > 0:
        raise ValueError("covariance must be positive definite")
    d2 = float(diff @ np.linalg.solve(cov, diff))
    nll = 0.5 * (d2 + math.log(det) + 2 * math.log(2 * math.pi))
    return nll, d2


def kalman_cv(points, q=50.0, r=4.0):
    """Constant-velocity Kalman filter; yields (frame, mean, cov) from frame 2.

    q: acceleration noise (px^2 / frame^4): how much speed may change.
    r: measurement noise (px^2): how noisy SAM box centres are.
    """
    if q <= 0 or r <= 0:
        raise ValueError("q and r must be positive")
    F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
    g = np.array([[0.5, 0, 1, 0], [0, 0.5, 0, 1]], float).T  # 4x2
    Q = q * g @ g.T
    R = r * np.eye(2)
    p0, p1 = np.asarray(points[0], float), np.asarray(points[1], float)
    x = np.array([p1[0], p1[1], p1[0] - p0[0], p1[1] - p0[1]])
    P = np.diag([r, r, 2 * r + q, 2 * r + q])
    out = []
    for t in range(2, len(points)):
        x = F @ x
        P = F @ P @ F.T + Q
        S = H @ P @ H.T + R
        out.append((t, (H @ x).copy(), S.copy()))
        # Only now is frame t used.
        z = np.asarray(points[t], float)
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ (z - H @ x)
        P = (np.eye(4) - K @ H) @ P
    return out


def cv2pt_empirical(points, prior_var=25.0, prior_weight=2.0):
    """Two-point CV mean with variance from its own past squared errors."""
    out = []
    sum_sq, n = prior_var * prior_weight, prior_weight
    for t in range(2, len(points)):
        a, b = np.asarray(points[t - 2], float), np.asarray(points[t - 1], float)
        mean = 2 * b - a
        var = sum_sq / n / 2  # per-axis variance from mean squared 2-D error
        out.append((t, mean, var * np.eye(2)))
        err = np.asarray(points[t], float) - mean
        sum_sq += float(err @ err)
        n += 1
    return out


def score_forecasts(forecasts, points):
    """Summarise point error, NLL and ellipse coverage for one model."""
    rows = []
    for t, mean, cov in forecasts:
        nll, d2 = gaussian_score(mean, cov, points[t])
        err = float(np.linalg.norm(np.asarray(points[t], float) - mean))
        rows.append({"frame": t, "error_px": err, "nll": nll, "d2": d2,
                     "sigma_px": math.sqrt(float(np.trace(cov)) / 2)})
    n = len(rows)
    return {
        "evaluated_predictions": n,
        "mean_error_px": round(sum(r["error_px"] for r in rows) / n, 6),
        "mean_nll": round(sum(r["nll"] for r in rows) / n, 6),
        "coverage": {k: round(sum(r["d2"] <= v for r in rows) / n, 4)
                     for k, v in CHI2_2DOF.items()},
        "rows": rows,
    }


def main():
    """Development comparison: python -m app.learning.probabilistic --tracks X"""
    import argparse
    import json
    from pathlib import Path

    from app.learning.reference_baselines import track_centres

    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", required=True, action="append")
    args = parser.parse_args()
    for path in args.tracks:
        points = track_centres(json.loads(Path(path).read_text(encoding="utf-8")))
        print(Path(path).name)
        candidates = [("cv2pt_empirical", cv2pt_empirical(points))]
        candidates += [(f"kalman_cv q={q} r={r}", kalman_cv(points, q, r))
                       for q in (5.0, 50.0, 500.0) for r in (1.0, 4.0)]
        for label, forecasts in candidates:
            s = score_forecasts(forecasts, points)
            misses = [round(r["sigma_px"], 1) for r in s["rows"]
                      if r["error_px"] > 50]
            print(f"  {label:22s} error={s['mean_error_px']:7.2f}  "
                  f"nll={s['mean_nll']:6.2f}  inside90={s['coverage']['90']:.0%}  "
                  f"zone_at_big_misses_px={misses}")


if __name__ == "__main__":
    main()