
import unittest

from app.world.motion_events import detect_motion_events


def geometry():
    return {
        "schema": "qapha.scene_geometry.v1",
        "candidates": [
            {
                "endpoints_px": [
                    [0, 100],
                    [100, 100],
                ]
            }
        ],
    }


def tracking(points, frames=None):
    if frames is None:
        frames = list(range(len(points)))

    return {
        "outcome": "completed",
        "boxes": [
            {
                "object_id": "1",
                "frame_index": frame,
                "bbox_xyxy_px": [
                    x - 10,
                    y - 10,
                    x + 10,
                    y + 10,
                ],
            }
            for frame, (x, y) in zip(frames, points)
        ],
    }


class MotionEventTests(unittest.TestCase):

    def call(self, tracks, **kwargs):
        return detect_motion_events(
            geometry(),
            tracks,
            start_frame=0,
            end_frame=4,
            edge_ids=(0,),
            **kwargs,
        )

    def test_vertical_reversal_near_edge_not_collision(self):
        result = self.call(tracking([
            (50, 45),
            (50, 65),
            (50, 90),
            (50, 70),
            (50, 50),
        ]))

        event = result["events"][0]

        self.assertEqual(
            event["event"],
            "vertical_direction_reversal",
        )
        self.assertEqual(event["frame_index"], 2)
        self.assertTrue(event["near_reviewed_edge"])
        self.assertFalse(
            event["physical_collision_verified"]
        )

    def test_horizontal_reversal(self):
        result = self.call(tracking([
            (10, 20),
            (30, 20),
            (50, 20),
            (30, 20),
            (10, 20),
        ]))

        self.assertEqual(
            [e["event"] for e in result["events"]],
            ["horizontal_direction_reversal"],
        )

    def test_gap_does_not_invent_reversal(self):
        result = self.call(tracking(
            [(50, 30), (50, 60), (50, 30)],
            [0, 2, 4],
        ))

        self.assertEqual(result["events"], [])

    def test_future_is_ignored(self):
        base = tracking([
            (50, 20),
            (50, 30),
            (50, 40),
            (50, 50),
            (50, 60),
        ])

        future = {
            "outcome": "completed",
            "boxes": (
                base["boxes"]
                + tracking(
                    [(50, 500)], [5]
                )["boxes"]
            ),
        }

        self.assertEqual(
            self.call(base),
            self.call(future),
        )

    def test_rejects_duplicate_and_bad_threshold(self):
        base = tracking([
            (50, 20),
            (50, 30),
            (50, 40),
            (50, 50),
            (50, 60),
        ])

        duplicate = {
            "outcome": "completed",
            "boxes": (
                base["boxes"]
                + [base["boxes"][1]]
            ),
        }

        with self.assertRaises(ValueError):
            self.call(duplicate)

        with self.assertRaises(ValueError):
            self.call(base, min_step_px=0)


if __name__ == "__main__":
    unittest.main()
