
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.experiments.calibration_clip import (
    extract_calibration,
    sha256_file,
)


class CalibrationClipTests(unittest.TestCase):

    def create_fixture(self, directory):
        source = directory / "source.mp4"

        writer = cv2.VideoWriter(
            str(source),
            cv2.VideoWriter_fourcc(*"mp4v"),
            25.0,
            (64, 48),
        )

        self.assertTrue(writer.isOpened())

        for frame in range(20):
            image = np.full(
                (48, 64, 3),
                frame * 10,
                dtype=np.uint8,
            )

            writer.write(image)

        writer.release()

        return source, {
            "schema": "qapha.heldout_protocol.v1",
            "stage": "registered_before_future_tracking",
            "video": {
                "path": str(source),
                "sha256": sha256_file(source),
                "fps": 25.0,
                "resolution_px": [64, 48],
            },
            "calibration_frames_inclusive": [0, 9],
            "evaluation_policy": {
                "perception_calibration_only": True,
            },
        }

    def test_extracts_only_calibration_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source, protocol = self.create_fixture(directory)

            result = extract_calibration(
                protocol,
                output_video=directory / "calibration.mp4",
            )

            self.assertEqual(result["frames_written"], 10)
            self.assertEqual(result["frames_verified"], 10)
            self.assertFalse(result["future_frames_included"])
            self.assertTrue(source.is_file())

    def test_rejects_modified_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _, protocol = self.create_fixture(directory)

            protocol["video"]["sha256"] = "0" * 64

            with self.assertRaises(ValueError):
                extract_calibration(
                    protocol,
                    output_video=directory / "calibration.mp4",
                )

    def test_never_overwrites_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source, protocol = self.create_fixture(directory)

            with self.assertRaises(ValueError):
                extract_calibration(
                    protocol,
                    output_video=source,
                )

    def test_rejects_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _, protocol = self.create_fixture(directory)

            destination = directory / "calibration.mp4"
            destination.write_bytes(b"existing")

            with self.assertRaises(FileExistsError):
                extract_calibration(
                    protocol,
                    output_video=destination,
                )


if __name__ == "__main__":
    unittest.main()
