"""One-image SAM 3.1 smoke test. No tracking or physics assumptions."""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://api.meta.ai/v1/responses"
BOX_RE = re.compile(r"<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)(?:;c=([\d.]+))?\|>")


def make_request(image: Path, prompt: str = "ball") -> dict:
    """Construct documented Responses API request without sending it."""
    if not image.is_file():
        raise FileNotFoundError(f"Image not found: {image}")
    if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Image must be JPEG or PNG")
    if not prompt.strip() or len(prompt) > 100:
        raise ValueError("Provide a short, nonempty object noun phrase")
    mime = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
    data_url = f"data:{mime};base64,{base64.b64encode(image.read_bytes()).decode('ascii')}"
    if len(data_url) > 20_000_000:
        raise ValueError("Image too large for inline API request")
    return {
        "model": "sam-3.1",
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": prompt.strip()},
            {"type": "input_image", "image_url": data_url},
        ]}],
        "stream": True,
        "metadata": {"include_confidence": "true"},
    }


def extract_sse_text(lines) -> str:
    """Collect text deltas from a Meta Responses SSE stream."""
    parts = []
    for raw in lines:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        event = json.loads(data)
        kind = event.get("type")
        if kind in {"response.failed", "error"}:
            raise RuntimeError(f"SAM API failure: {json.dumps(event)[:600]}")
        if kind == "response.output_text.delta":
            parts.append(event.get("delta", ""))
    return "".join(parts)


def inspect_output(raw: str) -> dict:
    """Extract boxes for inspection; raw masks remain in the saved response."""
    boxes = []
    for m in BOX_RE.finditer(raw):
        x1, y1, x2, y2, w, h = map(int, m.groups()[:6])
        boxes.append({"bbox_xyxy_px": [x1, y1, x2, y2], "image_size_px": [w, h],
                      "confidence": float(m.group(7)) if m.group(7) else None})
    return {"box_count": len(boxes), "boxes": boxes, "mask_tokens_present": raw.count("<|mask;")}


def run(image: Path, output: Path, prompt: str = "ball") -> dict:
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        raise ValueError("MODEL_API_KEY environment variable is not set")
    payload = make_request(image, prompt)
    request = urllib.request.Request(API_URL, data=json.dumps(payload).encode("utf-8"), headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = extract_sse_text(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(1200).decode("utf-8", errors="replace")
        raise RuntimeError(f"SAM HTTP {exc.code}: {detail}") from exc
    result = inspect_output(raw)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"model": "sam-3.1", "prompt": prompt, "image": str(image.resolve()),
                                  "raw_segmentation": raw, "inspection": result}, indent=2), encoding="utf-8")
    return {**result, "output": str(output), "note": "Boxes are unverified until visually inspected; masks preserved raw."}


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe SAM 3.1 whole-object segmentation on one frame")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="ball")
    args = parser.parse_args()
    try:
        print(json.dumps(run(Path(args.image), Path(args.output), args.prompt), indent=2))
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
