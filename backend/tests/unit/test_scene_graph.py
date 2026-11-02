
import unittest

from app.world.scene_graph import (
    build_scene_graph,
    segment_distance,
)


class SceneGraphTests(unittest.TestCase):
    def setUp(self):
        self.geometry = {
            "schema": "qapha.scene_geometry.v1",
            "candidates": [
                {"endpoints_px": [[0, 100], [100, 100]]},
                {"endpoints_px": [[0, 200], [100, 200]]},
            ],
        }
        self.tracks = {
            "outcome": "completed",
            "boxes": [
                {
                    "object_id": "1",
                    "frame_index": 49,
                    "bbox_xyxy_px": [40, 75, 60, 95],
                },
                {
                    "object_id": "1",
                    "frame_index": 50,
                    "bbox_xyxy_px": [40, 175, 60, 195],
                },
            ],
        }

    def test_distance_to_segment_clamped(self):
        self.assertAlmostEqual(
            segment_distance(50, 95, [0, 100], [100, 100]),
            5,
        )
        self.assertAlmostEqual(
            segment_distance(-10, 100, [0, 100], [100, 100]),
            10,
        )

    def test_graph_proximity_not_contact(self):
        result = build_scene_graph(
            self.geometry, self.tracks,
            edge_ids=(0,), near_px=10,
        )
        self.assertEqual(len(result["tracked_objects"]), 1)
        self.assertEqual(len(result["relationships"]), 1)
        self.assertFalse(
            result["relationships"][0]["physical_contact_verified"]
        )
        self.assertFalse(
            result["reviewed_edges"][0]["physical_surface_verified"]
        )

    def test_excludes_unreviewed_edges_and_future(self):
        result = build_scene_graph(
            self.geometry, self.tracks,
            edge_ids=(0,), near_px=10,
        )
        self.assertEqual(result["frame_index"], 49)
        self.assertEqual(
            [
                e["source_candidate_index"]
                for e in result["reviewed_edges"]
            ],
            [0],
        )
        self.assertEqual(
            result["tracked_objects"][0]["bbox_xyxy_px"],
            [40, 75, 60, 95],
        )

    def test_rejects_duplicates_and_invalid_edges(self):
        with self.assertRaises(ValueError):
            build_scene_graph(
                self.geometry, self.tracks, edge_ids=(0, 0)
            )

        with self.assertRaises(ValueError):
            build_scene_graph(
                self.geometry, self.tracks, edge_ids=(10,)
            )

        invalid = {
            "outcome": "completed",
            "boxes": [self.tracks["boxes"][0]] * 2,
        }
        with self.assertRaises(ValueError):
            build_scene_graph(
                self.geometry, invalid, edge_ids=(0,)
            )


if __name__ == "__main__":
    unittest.main()
