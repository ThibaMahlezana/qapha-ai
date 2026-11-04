import math
import unittest

import numpy as np

from app.learning.probabilistic import (
    cv2pt_empirical,
    gaussian_score,
    kalman_cv,
    score_forecasts,
)


def line(n=20, vx=3.0, vy=-2.0):
    return [(10 + vx * t, 50 + vy * t) for t in range(n)]


class ScoringTests(unittest.TestCase):
    def test_gaussian_score_at_mean(self):
        nll, d2 = gaussian_score([0, 0], np.eye(2), [0, 0])
        self.assertAlmostEqual(d2, 0.0)
        self.assertAlmostEqual(nll, math.log(2 * math.pi))

    def test_mahalanobis_scales_with_variance(self):
        _, d2 = gaussian_score([0, 0], 4 * np.eye(2), [2, 0])
        self.assertAlmostEqual(d2, 1.0)

    def test_rejects_bad_covariance(self):
        with self.assertRaises(ValueError):
            gaussian_score([0, 0], np.zeros((2, 2)), [0, 0])

    def test_coverage_fields(self):
        pts = line()
        s = score_forecasts(kalman_cv(pts), pts)
        self.assertEqual(set(s["coverage"]), {"50", "90", "95"})
        self.assertEqual(s["evaluated_predictions"], len(pts) - 2)


class ForecasterTests(unittest.TestCase):
    def test_kalman_exact_on_straight_line(self):
        pts = line()
        s = score_forecasts(kalman_cv(pts), pts)
        self.assertLess(s["mean_error_px"], 1e-6)
        self.assertEqual(s["coverage"]["50"], 1.0)

    def test_kalman_forecast_ignores_target(self):
        pts = line()
        changed = pts[:-1] + [(999.0, 999.0)]
        a, b = kalman_cv(pts)[-1], kalman_cv(changed)[-1]
        self.assertTrue(np.allclose(a[1], b[1]) and np.allclose(a[2], b[2]))

    def test_more_process_noise_widens_zone(self):
        pts = line()
        narrow = kalman_cv(pts, q=1.0)[-1][2]
        wide = kalman_cv(pts, q=100.0)[-1][2]
        self.assertGreater(np.trace(wide), np.trace(narrow))

    def test_cv2pt_variance_grows_after_a_miss(self):
        pts = line(10) + [(500.0, 500.0)] + line(10)[-1:] * 2
        fc = cv2pt_empirical(pts)
        before = np.trace(fc[7][2])
        after = np.trace(fc[-1][2])
        self.assertGreater(after, before)

    def test_invalid_noise_rejected(self):
        with self.assertRaises(ValueError):
            kalman_cv(line(), q=0)


if __name__ == "__main__":
    unittest.main()