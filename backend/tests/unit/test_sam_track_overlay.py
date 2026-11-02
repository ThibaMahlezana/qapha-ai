
"""Unit tests for SAM video tracking diagnostics."""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.perception.sam_track_overlay import (
    render_overlay,
    validate_tracks,
)


class SamTrackOverlayTests(unittest.TestCase):

    def test_valid_tracking_records(self):
        report = {
            "outcome": "completed",
            "boxes": [
                {
                    "frame_index": 0,
                    "object_id": "0",
                    "bbox_xyxy_px": [10, 20, 30, 40],
                },
                {
                    "frame_index": 1,
                    "object_id": "0",
                    "bbox_xyxy_px": [15, 25, 35, 45],
                },
            ],
        }

        grouped = validate_tracks(report)

        self.assertEqual(len(grouped), 2)
        self.assertEqual(grouped[0][0]["object_id"], "0")
        self.assertEqual(grouped[1][0]["object_id"], "0")

    def test_rejects_duplicate_object(self):
        box = {
            "frame_index": 0,
            "object_id": "0",
            "bbox_xyxy_px": [10, 20, 30, 40],
        }

        report = {
            "outcome": "completed",
            "boxes": [box, dict(box)],
        }

        with self.assertRaises(ValueError):
            validate_tracks(report)

    def test_renders_annotated_video(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)

            video = folder / "source.mp4"
            tracks = folder / "tracks.json"
            output = folder / "annotated.mp4"

            writer = cv2.VideoWriter(
                str(video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                25,
                (160, 120),
            )

            self.assertTrue(writer.isOpened())

            for frame_index in range(3):
                image = np.zeros(
                    (120, 160, 3),
                    dtype=np.uint8,
                )

                cv2.circle(
                    image,
                    (30 + frame_index * 10, 60),
                    10,
                    (255, 255, 255),
                    -1,
                )

                writer.write(image)

            writer.release()

            report = {
                "outcome": "completed",
                "boxes": [
                    {
                        "frame_index": index,
                        "object_id": "0",
                        "bbox_xyxy_px": [
                            20 + index * 10,
                            50,
                            40 + index * 10,
                            70,
                        ],
                    }
                    for index in range(3)
                ],
            }

            tracks.write_text(
                json.dumps(report),
                encoding="utf-8",
            )

            result = render_overlay(
                video,
                tracks,
                output,
            )

            self.assertEqual(result["frames_rendered"], 3)
            self.assertEqual(result["unique_object_ids"], 1)
            self.assertEqual(
                result["frames_with_detections"], 3
            )

            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
