
import copy
import unittest

from app.experiments.conditional_contact import simulate


def fixtures():
    tracks = {
        "outcome": "completed",
        "boxes": [],
    }

    for frame in range(12):
        t = frame - 7
        x = 10 + t
        y = 20 + 2 * t + 0.5 * t * t

        tracks["boxes"].append({
            "object_id": "1",
            "frame_index": frame,
            "bbox_xyxy_px": [
                x - 2,
                y - 2,
                x + 2,
                y + 2,
            ],
        })

    geometry = {
        "schema": "qapha.scene_geometry.v1",
        "candidates": [
            {
                "endpoints_px": [
                    [0, 35],
                    [100, 35],
                ]
            }
        ],
    }

    return tracks, geometry


class ConditionalContactTests(unittest.TestCase):

    def test_contact_and_divergence(self):
        result = simulate(
            *fixtures(),
            cutoff=7,
            train_start=3,
            horizons=(1, 3, 6),
            edge_id=0,
        )

        self.assertGreaterEqual(
            len(result["hypothetical_impacts"]),
            1,
        )

        final = result["forecasts"][-1]

        self.assertNotEqual(
            final["no_contact_center_px"],
            final["conditional_contact_center_px"],
        )

        self.assertFalse(
            result["contact_assumption"][
                "physical_surface_verified"
            ]
        )

    def test_no_contact_when_edge_out_of_path(self):
        tracks, geometry = fixtures()

        geometry["candidates"][0]["endpoints_px"] = [
            [200, 35],
            [300, 35],
        ]

        result = simulate(
            tracks,
            geometry,
            cutoff=7,
            train_start=3,
            horizons=(1, 3),
            edge_id=0,
        )

        self.assertEqual(
            result["hypothetical_impacts"],
            [],
        )

        for forecast in result["forecasts"]:
            self.assertAlmostEqual(
                forecast["no_contact_center_px"][1],
                forecast[
                    "conditional_contact_center_px"
                ][1],
                places=4,
            )

    def test_future_data_ignored(self):
        tracks, geometry = fixtures()

        baseline = simulate(
            tracks,
            geometry,
            cutoff=7,
            train_start=3,
            horizons=(1, 3),
            edge_id=0,
        )

        tracks["boxes"].append({
            "object_id": "1",
            "frame_index": 100,
            "bbox_xyxy_px": [
                0, 0, 999, 999,
            ],
        })

        self.assertEqual(
            baseline,
            simulate(
                tracks,
                geometry,
                cutoff=7,
                train_start=3,
                horizons=(1, 3),
                edge_id=0,
            ),
        )

    def test_gaps_and_duplicate_rejected(self):
        tracks, geometry = fixtures()

        tracks["boxes"] = [
            box for box in tracks["boxes"]
            if box["frame_index"] != 5
        ]

        with self.assertRaises(ValueError):
            simulate(
                tracks,
                geometry,
                cutoff=7,
                train_start=3,
                edge_id=0,
            )

        tracks, geometry = fixtures()

        tracks["boxes"].append(
            copy.deepcopy(tracks["boxes"][5])
        )

        with self.assertRaises(ValueError):
            simulate(
                tracks,
                geometry,
                cutoff=7,
                train_start=3,
                edge_id=0,
            )

    def test_bad_assumptions(self):
        tracks, geometry = fixtures()

        with self.assertRaises(ValueError):
            simulate(
                tracks,
                geometry,
                cutoff=7,
                train_start=3,
                restitution=1.2,
            )

        with self.assertRaises(ValueError):
            simulate(
                tracks,
                geometry,
                cutoff=7,
                train_start=3,
                edge_id=99,
            )

        geometry["candidates"][0]["endpoints_px"] = [
            [0, 0],
            [0, 100],
        ]

        with self.assertRaises(ValueError):
            simulate(
                tracks,
                geometry,
                cutoff=7,
                train_start=3,
                edge_id=0,
            )


if __name__ == "__main__":
    unittest.main()
