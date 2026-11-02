
import unittest

from app.world.temporal_scene import build_temporal_scene


def geometry():
    return {
        "schema": "qapha.scene_geometry.v1",
        "candidates": [
            {
                "endpoints_px": [[0, 100], [100, 100]]
            }
        ],
    }


def tracks(ys, frames=None):
    if frames is None:
        frames = list(range(len(ys)))

    return {
        "outcome": "completed",
        "boxes": [
            {
                "object_id": "1",
                "frame_index": frame,
                "bbox_xyxy_px": [40, y - 10, 60, y],
            }
            for frame, y in zip(frames, ys)
        ],
    }


class TemporalSceneTests(unittest.TestCase):

    def test_approaching_and_near_not_contact(self):
        result = build_temporal_scene(
            geometry(),
            tracks([20, 45, 80]),
            start_frame=0,
            end_frame=2,
            edge_ids=(0,),
        )

        obj = result["objects"][0]
        relation = obj["relationships"][0]

        self.assertEqual(obj["motion_class"], "moving")
        self.assertEqual(
            relation["distance_trend"], "approaching"
        )
        self.assertTrue(relation["near_at_end"])
        self.assertFalse(
            relation["physical_contact_verified"]
        )

    def test_stationary(self):
        result = build_temporal_scene(
            geometry(),
            tracks([70, 70, 70]),
            start_frame=0,
            end_frame=2,
            edge_ids=(0,),
        )

        obj = result["objects"][0]

        self.assertEqual(
            obj["motion_class"],
            "approximately_stationary",
        )
        self.assertEqual(
            obj["relationships"][0]["distance_trend"],
            "approximately_constant_distance",
        )

    def test_gap_does_not_infer_trend(self):
        result = build_temporal_scene(
            geometry(),
            tracks([20, 80], [0, 2]),
            start_frame=0,
            end_frame=2,
            edge_ids=(0,),
        )

        self.assertEqual(
            result["objects"][0]["status"],
            "insufficient_temporal_coverage",
        )
        self.assertEqual(
            result["objects"][0]["relationships"],
            [],
        )

    def test_future_frames_ignored(self):
        base = tracks([20, 40, 60])
        extended = tracks([20, 40, 60, 1000])

        first = build_temporal_scene(
            geometry(), base,
            start_frame=0,
            end_frame=2,
            edge_ids=(0,),
        )

        second = build_temporal_scene(
            geometry(), extended,
            start_frame=0,
            end_frame=2,
            edge_ids=(0,),
        )

        self.assertEqual(first, second)

    def test_invalid_and_duplicates(self):
        with self.assertRaises(ValueError):
            build_temporal_scene(
                geometry(),
                tracks([1, 2]),
                start_frame=0,
                end_frame=1,
                edge_ids=(2,),
            )

        duplicate = tracks([10, 20])
        duplicate["boxes"].append(
            duplicate["boxes"][0]
        )

        with self.assertRaises(ValueError):
            build_temporal_scene(
                geometry(),
                duplicate,
                start_frame=0,
                end_frame=1,
                edge_ids=(0,),
            )


if __name__ == "__main__":
    unittest.main()
