
import unittest

from app.learning.reliability_monitor import (
    assess_reliability,
    monitor_timeline,
)


class ReliabilityMonitorTests(unittest.TestCase):

    def test_insufficient_history(self):
        result = assess_reliability([1, 2, 1])

        self.assertEqual(
            result["status"], "insufficient_history"
        )
        self.assertFalse(result["warning"])

    def test_stable_errors_do_not_trigger_warning(self):
        result = assess_reliability(
            [2, 2, 2, 2, 2, 2]
        )

        self.assertFalse(result["warning"])
        self.assertEqual(
            result["status"], "normal_recent_error"
        )

    def test_large_prior_error_triggers_warning(self):
        result = assess_reliability(
            [1, 2, 1, 3, 2, 40]
        )

        self.assertTrue(result["warning"])
        self.assertEqual(
            result["latest_prior_error_px"], 40
        )

    def test_current_error_cannot_trigger_its_own_warning(self):
        errors = [1, 1, 1, 1, 1, 1, 100]

        events = [
            {
                "frame_index": i + 2,
                "history_cutoff_frame": i + 1,
                "adaptive_error_px": error,
            }
            for i, error in enumerate(errors)
        ]

        report = monitor_timeline(events)

        # The final large error is not known when its forecast is made.
        self.assertFalse(
            report["assessments"][-1]["warning"]
        )

        events.append({
            "frame_index": 9,
            "history_cutoff_frame": 8,
            "adaptive_error_px": 1,
        })

        report = monitor_timeline(events)

        # The next forecast may use the previously observed large error.
        self.assertTrue(
            report["assessments"][-1]["warning"]
        )

    def test_rejects_invalid_errors_and_frame_gaps(self):
        with self.assertRaises(ValueError):
            assess_reliability([1, -5, 2])

        with self.assertRaises(ValueError):
            monitor_timeline([
                {
                    "frame_index": 2,
                    "history_cutoff_frame": 1,
                    "adaptive_error_px": 1,
                },
                {
                    "frame_index": 4,
                    "history_cutoff_frame": 3,
                    "adaptive_error_px": 2,
                },
            ])


if __name__ == "__main__":
    unittest.main()
