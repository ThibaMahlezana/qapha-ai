import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.perception.color_tracker import locate_colored_object, track_extracted_frames


class ColorTrackerTests(unittest.TestCase):
    def test_red_object_centre(self):
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.circle(image, (65, 45), 14, (0, 0, 255), -1)
        result = locate_colored_object(image, "red")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["center_px"][0], 65, delta=1)
        self.assertAlmostEqual(result["center_px"][1], 45, delta=1)

    def test_no_object_returns_none(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertIsNone(locate_colored_object(image, "red"))

    def test_manifest_tracking_and_missing_detection(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            frames = []
            for idx in range(2):
                image = np.zeros((100, 120, 3), dtype=np.uint8)
                if idx == 0:
                    cv2.rectangle(image, (20, 20), (45, 45), (0, 0, 255), -1)
                name = f"frame_{idx:08d}.jpg"
                self.assertTrue(cv2.imwrite(str(root / name), image))
                frames.append({"frame_index": idx, "timestamp_ms": idx * 500, "file": name})
            (root / "manifest.json").write_text(json.dumps({"frames": frames}), encoding="utf-8")
            result = track_extracted_frames(root / "manifest.json", output_path=root / "tracks.json")
            self.assertEqual(result["frames_checked"], 2)
            self.assertEqual(result["frames_detected"], 1)
            self.assertTrue((root / "tracks.json").exists())
            self.assertFalse(result["detections"][1]["detected"])

    def test_unsupported_colour_rejected(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            locate_colored_object(image, "purple")


if __name__ == "__main__":
    unittest.main()
