
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.experiments.heldout_manifest import register_video


class HeldoutManifestTests(unittest.TestCase):

    def make_video(self, folder, frames=80):
        path = folder / "clip.mp4"

        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            25.0,
            (64, 48),
        )

        self.assertTrue(writer.isOpened())

        for i in range(frames):
            image = np.full(
                (48, 64, 3),
                i % 255,
                dtype=np.uint8,
            )
            writer.write(image)

        writer.release()
        return path

    def test_registration_and_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = self.make_video(Path(tmp))
            result = register_video(video)

            self.assertEqual(
                result["calibration_frames_inclusive"],
                [0, 49],
            )
            self.assertEqual(
                result["target_frames"],
                [54, 59, 70],
            )
            self.assertEqual(
                len(result["video"]["sha256"]),
                64,
            )
            self.assertTrue(
                result["evaluation_policy"][
                    "verification_tracking_separate"
                ]
            )

    def test_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = self.make_video(Path(tmp))

            self.assertEqual(
                register_video(video),
                register_video(video),
            )

    def test_rejects_insufficient_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = self.make_video(
                Path(tmp),
                frames=60,
            )

            with self.assertRaises(ValueError):
                register_video(video)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            register_video(
                "missing.mp4",
                cutoff=3,
            )

        with self.assertRaises(ValueError):
            register_video(
                "missing.mp4",
                horizons=(5, 5),
            )

        with self.assertRaises(FileNotFoundError):
            register_video("missing.mp4")


if __name__ == "__main__":
    unittest.main()
