"""Step 6G: hosted SAM 3.1 video tracking smoke test.

Uses Meta's documented Files + Responses API and the official SAM stream parser.
Stores only derived geometry/track summary; the original uploaded video remains local.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from contextlib import aclosing
from pathlib import Path

API_BASE = "https://api.meta.ai/v1"
MODEL = "sam-3.1"


def validate_video(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != ".mp4":
        raise ValueError("Provide an existing .mp4 video file")
    if path.stat().st_size == 0:
        raise ValueError("Video is empty")
    return path


def make_input(file_id: str, concept: str) -> list[dict]:
    if not file_id.startswith("file-"):
        raise ValueError("Expected a Meta file ID beginning with 'file-'")
    if not concept.strip():
        raise ValueError("Concept must not be empty")
    return [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": concept.strip()},
        {"type": "input_video", "file_id": file_id},
    ]}]


async def normalise_events(stream):
    """Official parser needs a text-done event when only content-part-done arrives."""
    completed = set()
    async for event in stream:
        data = event.model_dump() if hasattr(event, "model_dump") else dict(event)
        lane = (data.get("item_id"), data.get("output_index"), data.get("content_index"))
        if data.get("type") == "response.output_text.done":
            if lane not in completed:
                completed.add(lane)
                yield data
            continue
        yield data
        part = data.get("part")
        if (data.get("type") == "response.content_part.done" and lane not in completed
                and isinstance(part, dict) and part.get("type") == "output_text"
                and isinstance(part.get("text"), str)):
            completed.add(lane)
            yield {"type": "response.output_text.done", "item_id": data.get("item_id"),
                   "output_index": data.get("output_index"), "content_index": data.get("content_index"),
                   "text": part["text"]}


def summarize(result, concept: str) -> dict:
    if result.outcome.status != "completed":
        raise RuntimeError(f"SAM video request incomplete: {result.outcome.status}")
    errors = [d.message for d in result.diagnostics if d.severity == "error"]
    if errors:
        raise RuntimeError("SAM parser errors: " + "; ".join(errors[:3]))
    boxes = []
    frames_by_id: dict[str, set[int]] = {}
    for record in result.records:
        if record.kind != "box":
            continue
        frame = None if record.frame is None else record.frame.frame_index
        obj_id = str(record.object_id)
        boxes.append({"object_id": obj_id, "frame_index": frame,
                      "bbox_xyxy_px": [record.left, record.top, record.right, record.bottom]})
        if frame is not None:
            frames_by_id.setdefault(obj_id, set()).add(frame)
    return {
        "model": MODEL, "concept": concept, "outcome": "completed",
        "box_count": len(boxes),
        "mask_count": sum(r.kind == "mask" for r in result.records),
        "unique_object_ids": len(frames_by_id),
        "tracks": [{"object_id": k, "first_frame": min(v), "last_frame": max(v),
                    "visible_frames": len(v)} for k, v in sorted(frames_by_id.items())],
        "boxes": boxes,
    }


async def run(video: Path, output: Path, concept: str = "ball") -> dict:
    # Import dependencies lazily so unit tests do not need API credentials.
    from openai import AsyncOpenAI
    from meta_sam_parser import parse_responses_stream, video_segmentation_format

    video = validate_video(video)
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        raise RuntimeError("MODEL_API_KEY environment variable is missing")
    if output.resolve() == video:
        raise ValueError("Output must not overwrite the video")

    async with AsyncOpenAI(base_url=API_BASE, api_key=key, timeout=180.0) as api:
        with video.open("rb") as handle:
            uploaded = await api.files.create(file=(video.name, handle, "video/mp4"), purpose="user_data")
        file_id = uploaded.id
        request_input = make_input(file_id, concept)
        print("Video uploaded. Running SAM 3.1 video tracking...", flush=True)
        stream = await api.responses.create(model=MODEL, stream=True, input=request_input)
        async with stream, aclosing(normalise_events(stream)) as events:
            parsed = parse_responses_stream(events, video_segmentation_format())
            async with parsed:
                async for _snapshot in parsed:
                    pass
                result = await parsed.final_result()
    summary = summarize(result, concept)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"outcome": summary["outcome"], "box_count": summary["box_count"],
            "mask_count": summary["mask_count"], "unique_object_ids": summary["unique_object_ids"],
            "tracks": summary["tracks"], "output": str(output.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM 3.1 hosted video tracking smoke test")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concept", default="ball")
    args = parser.parse_args()
    try:
        print(json.dumps(asyncio.run(run(args.video, args.output, args.concept)), indent=2))
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
