import json
import tempfile
import unittest
from pathlib import Path

try:
    import cv2
    import numpy as np
    from fastapi.testclient import TestClient
    from app.api.lab import create_app
    AVAILABLE = True
except ImportError:  # fastapi/httpx not installed
    AVAILABLE = False

from app.pipeline.timeline import build_timeline


def tracks(n=10):
    return {"model": "sam-3.1", "concept": "ball", "boxes": [
        {"object_id": "0", "frame_index": i,
         "bbox_xyxy_px": [10 + 3 * i, 20, 20 + 3 * i, 30]}
        for i in range(n)]}


@unittest.skipUnless(AVAILABLE, "fastapi/httpx not installed")
class LabApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        data = Path(self.tmp.name)
        (data / "timelines").mkdir()
        (data / "recorded").mkdir()
        writer = cv2.VideoWriter(str(data / "recorded" / "clip.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 25.0, (64, 48))
        for i in range(10):
            writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
        writer.release()
        timeline = build_timeline(tracks(), video={
            "file": "clip.mp4", "fps": 25.0, "resolution_px": [64, 48]})
        (data / "timelines" / "demo.json").write_text(
            json.dumps(timeline), encoding="utf-8")
        bad = dict(timeline, video={"file": "../secret.mp4"})
        (data / "timelines" / "escape.json").write_text(
            json.dumps(bad), encoding="utf-8")
        self.client = TestClient(create_app(data))

    def tearDown(self):
        self.tmp.cleanup()

    def test_health(self):
        self.assertEqual(self.client.get("/api/health").json()["status"], "ok")

    def test_lists_timelines(self):
        names = [t["name"] for t in self.client.get("/api/timelines").json()]
        self.assertEqual(names, ["demo", "escape"])

    def test_returns_timeline(self):
        body = self.client.get("/api/timelines/demo").json()
        self.assertEqual(body["frame_count"], 10)

    def test_returns_jpeg_frame(self):
        r = self.client.get("/api/timelines/demo/frames/3")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "image/jpeg")
        self.assertTrue(r.content.startswith(b"\xff\xd8"))

    def test_frame_out_of_range(self):
        self.assertEqual(
            self.client.get("/api/timelines/demo/frames/10").status_code, 404)

    def test_rejects_bad_names_and_path_escape(self):
        self.assertEqual(
            self.client.get("/api/timelines/..%2Fx").status_code, 404)
        self.assertEqual(
            self.client.get("/api/timelines/a.b").status_code, 400)
        self.assertEqual(
            self.client.get("/api/timelines/escape/frames/0").status_code, 404)

    def test_frame_offset_and_analysis_folder(self):
        data = Path(self.tmp.name)
        folder = data / "fresh_clip_analysis_v2"
        folder.mkdir()
        import shutil
        shutil.copy(data / "recorded" / "clip.mp4", folder / "fresh.mp4")
        timeline = build_timeline(tracks(5), video={
            "file": "fresh.mp4", "fps": 25.0, "resolution_px": [64, 48],
            "frame_offset": 4})
        (data / "timelines" / "fresh.json").write_text(
            json.dumps(timeline), encoding="utf-8")
        client = TestClient(create_app(data))
        self.assertEqual(client.get("/api/timelines/fresh/frames/4").status_code, 200)
        self.assertEqual(client.get("/api/timelines/fresh/frames/5").status_code, 404)
        direct = client.get("/api/timelines/demo/frames/8").content
        shifted = client.get("/api/timelines/fresh/frames/4").content
        self.assertEqual(direct, shifted)

    def test_concurrent_frame_requests_all_succeed(self):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as pool:
            codes = list(pool.map(
                lambda i: self.client.get(f"/api/timelines/demo/frames/{i % 10}").status_code,
                range(40)))
        self.assertEqual(set(codes), {200})

    def test_same_name_in_two_folders_uses_recorded_path(self):
        data = Path(self.tmp.name)
        for folder, frames in (("fresh_clip_analysis", 3), ("fresh_clip_analysis_v2", 10)):
            (data / folder).mkdir()
            writer = cv2.VideoWriter(str(data / folder / "same.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 48))
            for i in range(frames):
                writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
            writer.release()
        timeline = build_timeline(tracks(5), video={
            "file": "fresh_clip_analysis_v2/same.mp4", "fps": 25.0,
            "resolution_px": [64, 48], "frame_offset": 4})
        (data / "timelines" / "same.json").write_text(json.dumps(timeline),
                                                      encoding="utf-8")
        client = TestClient(create_app(data))
        self.assertEqual(client.get("/api/timelines/same/frames/4").status_code, 200)

    def test_path_outside_allowed_folders_rejected(self):
        data = Path(self.tmp.name)
        timeline = build_timeline(tracks(5), video={
            "file": "../outside.mp4", "fps": 25.0, "resolution_px": [64, 48]})
        (data / "timelines" / "bad.json").write_text(json.dumps(timeline),
                                                     encoding="utf-8")
        client = TestClient(create_app(data))
        self.assertEqual(client.get("/api/timelines/bad/frames/0").status_code, 404)

    def test_only_get_allowed(self):
        self.assertEqual(
            self.client.post("/api/timelines/demo").status_code, 405)


if __name__ == "__main__":
    unittest.main()
