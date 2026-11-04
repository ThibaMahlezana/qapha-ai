import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.experiments.calibration_preview import _sha256, render_preview


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.clip = Path(self.tmp.name) / "clip.mp4"
        writer = cv2.VideoWriter(str(self.clip),
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 25.0, (320, 240))
        for i in range(50):
            frame = np.full((240, 320, 3), i * 5, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        self.manifest = {"calibration_video": str(self.clip),
                         "calibration_sha256": _sha256(self.clip),
                         "frame_range_inclusive": [0, 49]}

    def tearDown(self):
        self.tmp.cleanup()

    def test_renders_three_tiles(self):
        sheet = render_preview(self.manifest, (0, 25, 49), tile_width=200)
        self.assertEqual(sheet.shape[1], 600)
        self.assertEqual(sheet.shape[0], 150)

    def test_rejects_frame_outside_calibration(self):
        with self.assertRaises(ValueError):
            render_preview(self.manifest, (0, 50))

    def test_rejects_modified_clip(self):
        self.manifest["calibration_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            render_preview(self.manifest)


if __name__ == "__main__":
    unittest.main()