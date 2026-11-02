import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.perception.multi_object_tracker import NearestNeighbourTracker, detect_blobs, track_manifest


class MultiObjectTrackerTests(unittest.TestCase):
    def test_detect_multiple_colours(self):
        image = np.zeros((150, 200, 3), dtype=np.uint8)
        cv2.circle(image, (35, 60), 12, (0, 0, 255), -1)
        cv2.circle(image, (150, 60), 12, (255, 0, 0), -1)
        detected = detect_blobs(image)
        self.assertEqual({d['color'] for d in detected}, {'red', 'blue'})

    def test_same_object_keeps_identity(self):
        tracker = NearestNeighbourTracker()
        first = tracker.update([{'color': 'red', 'center_px': [10, 20]}], 0)
        second = tracker.update([{'color': 'red', 'center_px': [20, 20]}], 40)
        self.assertEqual(first[0]['track_id'], second[0]['track_id'])

    def test_missed_frame_and_expiration(self):
        tracker = NearestNeighbourTracker(max_missed=1)
        initial = tracker.update([{'color': 'green', 'center_px': [30, 30]}], 0)[0]['track_id']
        self.assertEqual(tracker.update([], 40), [])
        self.assertEqual(tracker.update([{'color': 'green', 'center_px': [35, 30]}], 80)[0]['track_id'], initial)
        tracker.update([], 120)
        tracker.update([], 160)
        self.assertNotEqual(tracker.update([{'color': 'green', 'center_px': [35, 30]}], 200)[0]['track_id'], initial)

    def test_manifest_integration(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            items = []
            for i in range(3):
                image = np.zeros((100, 150, 3), dtype=np.uint8)
                cv2.circle(image, (20 + i * 8, 40), 10, (0, 0, 255), -1)
                name = f'frame_{i}.png'
                cv2.imwrite(str(base / name), image)
                items.append({'file': name, 'frame_index': i, 'timestamp_ms': 40 * i})
            (base / 'manifest.json').write_text(json.dumps({'frames': items}))
            result = track_manifest(base / 'manifest.json', base / 'tracks.json')
            self.assertEqual(result['frames_checked'], 3)
            self.assertEqual(result['unique_track_ids'], 1)
            self.assertEqual(result['total_detections'], 3)
            self.assertTrue((base / 'tracks.json').exists())


if __name__ == '__main__':
    unittest.main()
