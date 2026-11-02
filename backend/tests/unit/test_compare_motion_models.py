
import copy
import unittest

from app.experiments.compare_motion_models import (
    freeze_comparison,
    verify_comparison,
)


def report(last, *, curve=True):
    return {
        "outcome": "completed",
        "boxes": [
            {
                "object_id": "1",
                "frame_index": frame,
                "bbox_xyxy_px": [
                    frame,
                    frame * frame if curve else 2 * frame,
                    frame + 10,
                    (frame * frame if curve else 2 * frame) + 10,
                ],
            }
            for frame in range(last + 1)
        ],
    }


class ComparisonTests(unittest.TestCase):

    def test_quadratic_model_predicts_known_curve(self):
        frozen = freeze_comparison(
            report(49), target_frame=70
        )

        result = verify_comparison(
            frozen, report(70)
        )

        errors = result["errors_px"]

        self.assertLess(
            errors["constant_vertical_acceleration"],
            0.001,
        )

        self.assertGreater(
            errors["constant_velocity"],
            1,
        )

    def test_linear_motion_has_both_models_agree(self):
        frozen = freeze_comparison(
            report(49, curve=False),
            target_frame=70,
        )

        result = verify_comparison(
            frozen,
            report(70, curve=False),
        )

        for error in result["errors_px"].values():
            self.assertLess(error, 0.001)

    def test_rejects_future_leakage(self):
        with self.assertRaisesRegex(
            ValueError, "future"
        ):
            freeze_comparison(
                report(70), target_frame=70
            )

    def test_rejects_tampering_and_changed_history(self):
        frozen = freeze_comparison(report(49))

        changed = copy.deepcopy(frozen)

        changed["predictions"][
            "constant_velocity"
        ]["y"] = 0

        with self.assertRaisesRegex(
            ValueError, "modified"
        ):
            verify_comparison(changed, report(70))

        changed_history = report(70)

        changed_history["boxes"][0][
            "bbox_xyxy_px"
        ][1] += 1

        changed_history["boxes"][0][
            "bbox_xyxy_px"
        ][3] += 1

        with self.assertRaisesRegex(
            ValueError, "differ"
        ):
            verify_comparison(
                frozen, changed_history
            )

    def test_rejects_missing_future_and_gapped_recent_window(self):
        frozen = freeze_comparison(report(49))

        with self.assertRaisesRegex(
            ValueError, "unavailable"
        ):
            verify_comparison(
                frozen, report(69)
            )

        gapped = report(49)

        gapped["boxes"] = [
            box for box in gapped["boxes"]
            if box["frame_index"] != 46
        ]

        with self.assertRaisesRegex(
            ValueError, "consecutive"
        ):
            freeze_comparison(gapped)


if __name__ == "__main__":
    unittest.main()
