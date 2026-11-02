"""Offline integration tests: no Meta API calls or credentials required."""
import unittest

from app.experiments.muse_session import public_history, run_session
from app.physics.core import HiddenWorld, Session


class RecordingSelector:
    def __init__(self, speeds=(6.0, 2.0)):
        self.speeds = speeds
        self.histories = []

    def propose(self, allowed, history):
        self.histories.append(history)
        return self.speeds[len(self.histories) - 1], "Controlled test"


class MuseSessionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.session = Session(HiddenWorld(mu=0.23, k=0.34, seed=17))

    def test_two_experiments_and_frozen_forecast(self):
        selector = RecordingSelector()
        result = run_session(selector, self.session)
        self.assertEqual(len(result["experiments"]), 2)
        self.assertEqual(result["frozen_forecast"]["experiment_count"], 2)
        self.assertIn("absolute_error_m", result["evaluation"])
        self.assertEqual(selector.histories[0], [])
        self.assertEqual(len(selector.histories[1]), 1)
        self.assertIn("observations", selector.histories[1][0])
        self.assertNotIn("_mu", str(selector.histories))
        self.assertNotIn("_k", str(selector.histories))

    def test_rejects_disallowed_speed_before_observing(self):
        selector = RecordingSelector((7.0,))
        with self.assertRaisesRegex(ValueError, "not authorised"):
            run_session(selector, self.session)
        self.assertEqual(self.session.trials, [])

    def test_preexisting_budget_is_honoured(self):
        self.session.experiment(4.0)
        selector = RecordingSelector((2.0,))
        result = run_session(selector, self.session)
        self.assertEqual(len(result["experiments"]), 1)
        self.assertEqual(len(self.session.trials), 2)

    def test_public_history_is_only_measurements(self):
        self.session.experiment(2.0)
        history = public_history(self.session)
        self.assertEqual(set(history[0]), {"speed_mps", "observations"})
        self.assertEqual(set(history[0]["observations"][0]), {"time_s", "position_m", "sigma_m"})


if __name__ == "__main__":
    unittest.main()
