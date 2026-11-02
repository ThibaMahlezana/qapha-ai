
import copy
import unittest

from app.experiments.context_forecast import forecast


def tracks():
    return {
        "outcome": "completed",
        "boxes": [
            {
                "object_id": "1",
                "frame_index": f,
                "bbox_xyxy_px": [
                    2 * f - 2,
                    3 * f - 2,
                    2 * f + 2,
                    3 * f + 2,
                ],
            }
            for f in range(20)
        ],
    }


def events(frame=10):
    return {
        "schema": "qapha.motion_events.v1",
        "frame_window": [0, 15],
        "events": [
            {
                "object_id": "1",
                "event": "vertical_direction_reversal",
                "frame_index": frame,
            }
        ],
    }


class ContextForecastTests(unittest.TestCase):

    def test_event_starts_post_reversal_training(self):
        result = forecast(
            tracks(),
            events(),
            cutoff=15,
            target=19,
            lookback=12,
        )

        self.assertEqual(
            result["policy"],
            "recent_linear_insufficient_post_event_samples",
        )

        result = forecast(
            tracks(),
            events(9),
            cutoff=15,
            target=19,
            lookback=12,
        )

        self.assertEqual(
            result["train_frames"], [10, 15]
        )

        self.assertEqual(
            result["policy"], "post_reversal_linear"
        )

        self.assertEqual(
            result["predicted_center_px"],
            [38.0, 57.0],
        )

    def test_no_event_uses_recent_linear(self):
        result = forecast(
            tracks(),
            events(0),
            cutoff=15,
            target=19,
            lookback=8,
        )

        self.assertEqual(
            result["policy"], "recent_linear_baseline"
        )

        self.assertEqual(
            result["train_frames"], [8, 15]
        )

    def test_future_tracks_do_not_change_forecast(self):
        base = tracks()
        extended = copy.deepcopy(base)

        extended["boxes"].append({
            "object_id": "1",
            "frame_index": 100,
            "bbox_xyxy_px": [0, 0, 900, 900],
        })

        self.assertEqual(
            forecast(
                base, events(9),
                cutoff=15, target=19,
            ),
            forecast(
                extended, events(9),
                cutoff=15, target=19,
            ),
        )

    def test_future_event_is_rejected(self):
        bad = events(16)
        bad["frame_window"] = [0, 16]

        with self.assertRaises(ValueError):
            forecast(
                tracks(),
                bad,
                cutoff=15,
                target=19,
            )

    def test_tracking_gap_is_not_bridged(self):
        bad = tracks()
        bad["boxes"] = [
            box for box in bad["boxes"]
            if box["frame_index"] != 13
        ]

        with self.assertRaises(ValueError):
            forecast(
                bad,
                events(9),
                cutoff=15,
                target=19,
            )


if __name__ == "__main__":
    unittest.main()
