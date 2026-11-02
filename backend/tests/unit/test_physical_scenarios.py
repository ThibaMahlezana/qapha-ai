
import unittest

from app.experiments.physical_scenarios import predict_scenarios


def fixtures():
    tracks = {"outcome": "completed", "boxes": []}

    for f in range(20):
        t = f - 9
        x = 50 + 2 * t
        y = 40 + 3 * t + 0.5 * t * t

        tracks["boxes"].append({
            "object_id": "1",
            "frame_index": f,
            "bbox_xyxy_px": [
                x - 2, y - 2,
                x + 2, y + 2
            ],
        })

    events = {
        "schema": "qapha.motion_events.v1",
        "frame_window": [0, 9],
        "events": [{
            "object_id": "1",
            "frame_index": 4,
            "event": "vertical_direction_reversal",
        }],
    }

    geometry = {
        "schema": "qapha.scene_geometry.v1",
        "candidates": [{
            "endpoints_px": [[0, 100], [120, 100]]
        }],
    }

    return tracks, events, geometry


class PhysicalScenarioTests(unittest.TestCase):

    def test_known_acceleration_and_horizons(self):
        result = predict_scenarios(
            *fixtures(),
            cutoff=9,
            horizons=(2, 4),
            edge_ids=(0,),
        )

        self.assertAlmostEqual(
            result["image_y_acceleration_px_per_frame2"], 1.0
        )
        self.assertEqual(result["training_frames"], [5, 9])
        self.assertEqual(result["forecasts"][0]["frame_index"], 11)
        self.assertAlmostEqual(
            result["forecasts"][0][
                "free_flight_candidate_center_px"
            ][1],
            48.0,
        )
        self.assertFalse(result["physical_contact_verified"])

    def test_future_tracking_is_ignored(self):
        t, e, g = fixtures()

        result = predict_scenarios(
            t, e, g, cutoff=9, edge_ids=(0,)
        )

        t["boxes"].append({
            "object_id": "1",
            "frame_index": 20,
            "bbox_xyxy_px": [0, 0, 1000, 1000],
        })

        self.assertEqual(
            result,
            predict_scenarios(
                t, e, g, cutoff=9, edge_ids=(0,)
            ),
        )

    def test_requires_reversal(self):
        t, e, g = fixtures()
        e["events"] = []

        with self.assertRaises(ValueError):
            predict_scenarios(
                t, e, g, cutoff=9, edge_ids=(0,)
            )

    def test_rejects_gapped_post_event_history(self):
        t, e, g = fixtures()

        t["boxes"] = [
            b for b in t["boxes"]
            if b["frame_index"] != 7
        ]

        with self.assertRaises(ValueError):
            predict_scenarios(
                t, e, g, cutoff=9, edge_ids=(0,)
            )

    def test_rejects_future_event_and_bad_edge(self):
        t, e, g = fixtures()
        e["frame_window"] = [0, 11]

        with self.assertRaises(ValueError):
            predict_scenarios(
                t, e, g, cutoff=9, edge_ids=(0,)
            )

        t, e, g = fixtures()

        with self.assertRaises(ValueError):
            predict_scenarios(
                t, e, g, cutoff=9, edge_ids=(2,)
            )


if __name__ == "__main__":
    unittest.main()
