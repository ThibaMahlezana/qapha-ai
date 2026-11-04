
import copy
import json
import unittest

from app.learning.prediction_protocol import (
    create_protocol,
    freeze_next_prediction,
    observe_and_update,
)


def bootstrap():
    state = create_protocol()

    state, event = observe_and_update(state, 0, [0, 0])
    assert event is None

    state, event = observe_and_update(state, 1, [1, 0])
    assert event is None

    return state


class PredictionProtocolTests(unittest.TestCase):

    def test_freeze_before_observation(self):
        state = bootstrap()

        frozen_state, prediction = freeze_next_prediction(state)

        self.assertEqual(prediction["target_frame"], 2)
        self.assertEqual(prediction["history_cutoff_frame"], 1)
        self.assertEqual(
            prediction["predictions_xy"]["linear"],
            [2.0, 0.0],
        )

        self.assertEqual(
            len(frozen_state["session"]["observations"]),
            2,
        )

    def test_observe_scores_frozen_prediction(self):
        state = bootstrap()
        state, prediction = freeze_next_prediction(state)

        state, evaluation = observe_and_update(
            state, 2, [2, 0]
        )

        self.assertEqual(
            evaluation["prediction_sha256"],
            prediction["prediction_sha256"],
        )
        self.assertEqual(evaluation["errors_px"]["linear"], 0)
        self.assertIsNone(state["pending"])
        self.assertEqual(
            len(state["session"]["observations"]),
            3,
        )

    def test_cannot_observe_without_prediction(self):
        state = bootstrap()

        with self.assertRaises(ValueError):
            observe_and_update(state, 2, [2, 0])

    def test_cannot_replace_pending_prediction(self):
        state = bootstrap()
        state, _ = freeze_next_prediction(state)

        with self.assertRaises(ValueError):
            freeze_next_prediction(state)

    def test_resume_and_reject_tampering(self):
        state = bootstrap()
        state, prediction = freeze_next_prediction(state)

        restored = json.loads(json.dumps(state))

        restored, evaluation = observe_and_update(
            restored, 2, [2, 0]
        )

        self.assertEqual(
            evaluation["prediction_sha256"],
            prediction["prediction_sha256"],
        )

        modified = copy.deepcopy(state)
        modified["pending"]["target_frame"] = 20

        with self.assertRaises(ValueError):
            observe_and_update(modified, 2, [2, 0])


if __name__ == "__main__":
    unittest.main()
