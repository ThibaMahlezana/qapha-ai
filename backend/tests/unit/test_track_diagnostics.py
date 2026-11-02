
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.perception.track_diagnostics import annotate_video, summarize_tracks


class DiagnosticsTests(unittest.TestCase):
    def test_track_summary(self):
        report = {"frames": [
            {"frame_index": 0, "objects": [{"track_id": 1, "color": "red"}]},
            {"frame_index": 1, "objects": [{"track_id": 1, "color": "red"}, {"track_id": 2, "color": "blue"}]},
        ]}
        self.assertEqual(summarize_tracks(report)[0]["detections"], 2)
        self.assertEqual(len(summarize_tracks(report)), 2)

    def test_annotates_three_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            video = folder / "source.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (160, 120))
            self.assertTrue(writer.isOpened())
            for i in range(3):
                image = np.zeros((120, 160, 3), dtype=np.uint8)
                cv2.circle(image, (30 + i * 10, 60), 8, (0, 0, 255), -1)
                writer.write(image)
            writer.release()
            report = {"frames": [{"frame_index": i, "objects": [{"track_id": 1, "color": "red", "center_px": [30 + i * 10, 60], "bbox_xywh_px": [22 + i * 10, 52, 16, 16]}]} for i in range(3)]}
            tracks = folder / "tracks.json"
            tracks.write_text(json.dumps(report))
            output = folder / "annotated.mp4"
            result = annotate_video(video, tracks, output)
            self.assertEqual(result["frames_rendered"], 3)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_invalid_source_and_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            tracks = folder / "tracks.json"
            tracks.write_text(json.dumps({"frames": [{"frame_index": 1, "objects": []}, {"frame_index": 1, "objects": []}]}))
            with self.assertRaises(FileNotFoundError):
                annotate_video(folder / "missing.mp4", tracks, folder / "out.mp4")
            self.assertEqual(summarize_tracks({"frames": []}), [])


if __name__ == "__main__":
    unittest.main()
