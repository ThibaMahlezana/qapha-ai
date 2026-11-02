import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.perception.sam_video_smoke import make_input, normalise_events, summarize, validate_video


class SamVideoSmokeTests(unittest.TestCase):
    def test_request_video_and_prompt(self):
        request = make_input("file-123", "ball")
        self.assertEqual(request[0]["content"][1], {"type": "input_video", "file_id": "file-123"})
        with self.assertRaises(ValueError):
            make_input("invalid", "ball")
        with self.assertRaises(ValueError):
            make_input("file-123", " ")

    def test_validates_local_video(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "clip.mp4"
            with self.assertRaises(ValueError):
                validate_video(path)
            path.write_bytes(b"video-placeholder")
            self.assertEqual(validate_video(path), path.resolve())
            with self.assertRaises(ValueError):
                validate_video(Path(root) / "not.mp3")

    def test_summarizes_tracks_and_rejects_failure(self):
        frame = lambda i: SimpleNamespace(frame_index=i)
        records = [SimpleNamespace(kind="box", object_id="4", frame=frame(i),
                                   left=10, top=20, right=30, bottom=40) for i in (0, 1, 3)]
        records.append(SimpleNamespace(kind="mask"))
        result = SimpleNamespace(outcome=SimpleNamespace(status="completed"),
                                 diagnostics=[], records=records)
        summary = summarize(result, "ball")
        self.assertEqual(summary["unique_object_ids"], 1)
        self.assertEqual(summary["tracks"][0]["visible_frames"], 3)
        self.assertEqual(summary["mask_count"], 1)
        result.outcome.status = "failed"
        with self.assertRaises(RuntimeError):
            summarize(result, "ball")

    def test_synthesizes_completion_event(self):
        async def source():
            yield {"type": "response.content_part.done", "item_id": "x", "output_index": 0,
                   "content_index": 0, "part": {"type": "output_text", "text": "done"}}
        async def collect():
            return [event async for event in normalise_events(source())]
        import asyncio
        events = asyncio.run(collect())
        self.assertEqual([e["type"] for e in events],
                         ["response.content_part.done", "response.output_text.done"])


if __name__ == "__main__":
    unittest.main()
