
import copy
import unittest

from app.learning.adaptive_session import (
    advance_session,
    create_session,
    restore_tracker,
    session_summary,
)


POINTS = [
    [0.0, 0.0],
    [1.0, 0.0],
    [2.0, 0.0],
    [3.0, 0.0],
    [4.0, 0.0],
    [5.0, 0.0],
]


def feed(state, points, start=0):
    events = []

    for frame, point in enumerate(points, start=start):
        state, event = advance_session(state, frame, point)
        events.append(event)

    return state, events


class AdaptiveSessionTests(unittest.TestCase):

    def test_new_session_is_empty(self):
        state = create_session()
        summary = session_summary(state)

        self.assertEqual(summary["observations_seen"], 0)
        self.assertEqual(summary["next_expected_frame"], 0)

    def test_predict_before_learning(self):
        state, events = feed(create_session(), POINTS[:3])

        self.assertIsNone(events[0])
        self.assertIsNone(events[1])

        event = events[2]

        self.assertEqual(event["history_cutoff_frame"], 1)
        self.assertEqual(event["frame_index"], 2)
        self.assertEqual(
            event["predictions_xy"]["linear"],
            [2.0, 0.0],
        )

    def test_resume_matches_uninterrupted_session(self):
        continuous, _ = feed(create_session(), POINTS)

        partial, _ = feed(create_session(), POINTS[:3])

        # Simulate saving to JSON and restoring after restart.
        import json

        saved = json.dumps(partial)
        restored = json.loads(saved)

        resumed, _ = feed(
            restored,
            POINTS[3:],
            start=3,
        )

        self.assertEqual(continuous, resumed)
        self.assertEqual(
            session_summary(continuous)["mean_error_px"],
            session_summary(resumed)["mean_error_px"],
        )

    def test_rejects_modified_state(self):
        state, _ = feed(create_session(), POINTS[:3])

        modified = copy.deepcopy(state)
        modified["observations"][1]["xy"] = [999, 999]

        with self.assertRaises(ValueError):
            restore_tracker(modified)

    def test_rejects_skipped_and_future_frames(self):
        state, _ = feed(create_session(), POINTS[:3])

        with self.assertRaises(ValueError):
            advance_session(state, 5, [5.0, 0.0])

        with self.assertRaises(ValueError):
            advance_session(state, 2, [2.0, 0.0])


if __name__ == "__main__":
    unittest.main()
