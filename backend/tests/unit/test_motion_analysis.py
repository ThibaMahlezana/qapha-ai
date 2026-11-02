
"""Tests for SAM image-plane motion analysis."""

import unittest

from app.physics.motion_analysis import analyze_tracks


def make_report(ys, frames=None):
    if frames is None:
        frames = list(range(len(ys)))

    return {
        "outcome": "completed",
        "boxes": [
            {
                "object_id": "1",
                "frame_index": frame,
                "bbox_xyxy_px": [
                    0, y - 5, 10, y + 5
                ],
            }
            for frame, y in zip(frames, ys)
        ],
    }


class MotionAnalysisTests(unittest.TestCase):

    def test_velocity_and_bounce(self):
        result = analyze_tracks(
            make_report([0, 10, 30, 10, 0]),
            fps=25,
        )

        track = result["tracks"][0]

        self.assertEqual(
            track["samples"][0]["vy_px_s"],
            250.0,
        )

        self.assertEqual(
            track["samples"][2]["vy_px_s"],
            0.0,
        )

        self.assertEqual(
            [e["frame_index"]
             for e in track["candidate_events"]],
            [2],
        )

        self.assertFalse(
            track["candidate_events"][0]["verified_collision"]
        )

    def test_gaps_not_misclassified(self):
        result = analyze_tracks(
            make_report([0, 20, 0], [0, 1, 4]),
            fps=25,
        )

        track = result["tracks"][0]

        self.assertEqual(
            track["candidate_events"],
            [],
        )

        self.assertIsNone(
            track["samples"][1]["vy_px_s"]
        )

    def test_rejects_invalid_input(self):
        for fps in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                analyze_tracks(
                    make_report([0, 5]),
                    fps,
                )

        duplicate = make_report(
            [1, 2],
            [0, 0],
        )

        with self.assertRaises(ValueError):
            analyze_tracks(duplicate, 25)

    def test_stationary_object(self):
        result = analyze_tracks(
            make_report([50, 50, 50]),
            fps=25,
        )

        track = result["tracks"][0]

        self.assertEqual(
            track["candidate_events"],
            [],
        )

        self.assertEqual(
            track["samples"][1]["vy_px_s"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
