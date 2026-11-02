
"""Offline tests of candidate scene geometry extraction."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.context.scene_geometry import analyze_scene


class SceneGeometryTests(unittest.TestCase):

    def make_video(self, folder):
        path = folder / "scene.mp4"

        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            25,
            (320, 240),
        )

        self.assertTrue(writer.isOpened())

        for i in range(15):
            image = np.zeros(
                (240, 320, 3),
                dtype=np.uint8,
            )

            cv2.line(
                image,
                (20, 80),
                (300, 80),
                (255, 255, 255),
                3,
            )

            cv2.line(
                image,
                (20, 140),
                (270, 200),
                (255, 255, 255),
                3,
            )

            cv2.circle(
                image,
                (30 + i * 14, 45),
                8,
                (40, 40, 220),
                -1,
            )

            writer.write(image)

        writer.release()
        return path

    def test_finds_static_horizontal_and_inclined_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, overlay = analyze_scene(
                self.make_video(Path(tmp)),
                frames=12,
                min_length_px=70,
            )

            self.assertEqual(
                report["sampled_frames"], 12
            )

            self.assertEqual(
                overlay.shape, (240, 320, 3)
            )

            self.assertTrue(any(
                item["kind"] == "approximately_horizontal"
                for item in report["candidates"]
            ))

            self.assertTrue(any(
                item["kind"] == "inclined"
                for item in report["candidates"]
            ))

            self.assertTrue(all(
                not item["physical_surface_verified"]
                for item in report["candidates"]
            ))

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            analyze_scene("missing.mp4", frames=0)

        with self.assertRaises(FileNotFoundError):
            analyze_scene("missing.mp4")

    def test_deterministic_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = self.make_video(Path(tmp))

            first, _ = analyze_scene(
                video,
                frames=8,
                min_length_px=70,
            )

            second, _ = analyze_scene(
                video,
                frames=8,
                min_length_px=70,
            )

            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
