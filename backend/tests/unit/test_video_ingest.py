"""Step 6A tests; generate a tiny local AVI so no footage is needed."""
import json
import tempfile
import unittest
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

from app.perception.video_ingest import extract_frames


@unittest.skipUnless(cv2 is not None and np is not None, "Install opencv-python and numpy")
class VideoIngestTests(unittest.TestCase):
    def test_extract_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "example.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
            if not writer.isOpened():
                self.skipTest("OpenCV MJPG encoder unavailable")
            for i in range(20):
                img = np.zeros((48, 64, 3), dtype=np.uint8)
                img[:, :, :] = i * 10
                writer.write(img)
            writer.release()
            result = extract_frames(video, root / "out", interval_s=0.5, max_frames=10)
            self.assertEqual([f["frame_index"] for f in result["frames"]], [0, 5, 10, 15])
            self.assertEqual([f["timestamp_ms"] for f in result["frames"]], [0, 500, 1000, 1500])
            self.assertEqual(len(list((root / "out").glob("*.jpg"))), 4)
            self.assertEqual(len(json.loads((root / "out" / "manifest.json").read_text())["frames"]), 4)

    def test_missing_video_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                extract_frames(Path(temp) / "missing.mp4", Path(temp) / "out")

    def test_invalid_interval_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            video = Path(temp) / "empty.mp4"
            video.touch()
            with self.assertRaises(ValueError):
                extract_frames(video, Path(temp) / "out", interval_s=0)


if __name__ == "__main__":
    unittest.main()
