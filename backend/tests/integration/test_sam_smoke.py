import base64
import json
import tempfile
import unittest
from pathlib import Path

from app.perception.sam_smoke import extract_sse_text, inspect_output, make_request


class SamSmokeTests(unittest.TestCase):
    def test_request_uses_meta_sam_and_inline_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "frame.jpg"
            image.write_bytes(b"fake-jpeg-for-payload-test")
            payload = make_request(image, "ball")
            self.assertEqual(payload["model"], "sam-3.1")
            self.assertTrue(payload["stream"])
            self.assertEqual(payload["input"][0]["content"][0]["text"], "ball")
            url = payload["input"][0]["content"][1]["image_url"]
            self.assertEqual(base64.b64decode(url.split(",", 1)[1]), image.read_bytes())

    def test_sse_chunk_assembly_and_box_inspection(self):
        lines = [
            'data: ' + json.dumps({"type": "response.output_text.delta", "delta": "<0f>0<|box;x1=1;y1=2;"}),
            'data: ' + json.dumps({"type": "response.output_text.delta", "delta": "x2=12;y2=14;w=100;h=80;c=0.9|><|mask;x=0;y=0;data=...|>"}),
            'data: [DONE]',
        ]
        result = inspect_output(extract_sse_text(lines))
        self.assertEqual(result["box_count"], 1)
        self.assertEqual(result["boxes"][0]["bbox_xyxy_px"], [1, 2, 12, 14])
        self.assertEqual(result["mask_tokens_present"], 1)

    def test_invalid_inputs_and_empty_response(self):
        with self.assertRaises(FileNotFoundError):
            make_request(Path("not_present_image.jpg"))
        self.assertEqual(inspect_output("")["box_count"], 0)
        with self.assertRaises(RuntimeError):
            extract_sse_text(['data: {"type":"response.failed"}'])


if __name__ == "__main__":
    unittest.main()
