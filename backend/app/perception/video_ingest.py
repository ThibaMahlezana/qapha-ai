"""Step 6A: inspect a local video and export timestamped sampled frames.

Run from backend/: python -m app.perception.video_ingest --video PATH
No Meta API calls and no object tracking in this step.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def extract_frames(video_path: str | Path, output_dir: str | Path, interval_s: float = 1.0, max_frames: int = 20) -> dict:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required: python -m pip install opencv-python") from exc

    source = Path(video_path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"Video file does not exist: {source}")
    if not math.isfinite(interval_s) or interval_s <= 0:
        raise ValueError("interval_s must be a positive finite number")
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1")

    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not decode video: {source}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no valid frame rate; cannot compute reliable timestamps")
        reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise ValueError("Video dimensions are invalid")

        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        frames = []
        frame_index = 0
        next_sample_s = 0.0
        while len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_s = frame_index / fps  # Constant-frame-rate assumption
            if timestamp_s + 1e-9 >= next_sample_s:
                filename = f"frame_{frame_index:08d}.jpg"
                if not cv2.imwrite(str(destination / filename), frame):
                    raise OSError(f"Could not write frame: {destination / filename}")
                frames.append({"frame_index": frame_index, "timestamp_ms": round(timestamp_s * 1000, 3), "file": filename})
                next_sample_s += interval_s
                while next_sample_s <= timestamp_s + 1e-9:
                    next_sample_s += interval_s
            frame_index += 1
        if not frames:
            raise ValueError("Video contains no decodable frames")
    finally:
        capture.release()

    manifest = {
        "source": str(source),
        "fps": fps,
        "width": width,
        "height": height,
        "reported_frame_count": reported_frames,
        "timestamp_method": "frame_index/fps (constant frame rate assumed)",
        "sample_interval_s": interval_s,
        "frames": frames,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Qapha Step 6A: timestamped frame extraction")
    parser.add_argument("--video", required=True, help="Path to a local video (e.g. MP4)")
    parser.add_argument("--output", default="../datasets/recorded/extracted", help="Output folder for frames + manifest")
    parser.add_argument("--interval", type=float, default=1.0, help="Sample every N seconds (default 1)")
    parser.add_argument("--max-frames", type=int, default=20, help="Maximum frames to save (default 20)")
    args = parser.parse_args()
    try:
        manifest = extract_frames(args.video, args.output, args.interval, args.max_frames)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(json.dumps({"source": manifest["source"], "fps": manifest["fps"], "resolution": f'{manifest["width"]}x{manifest["height"]}', "frames_saved": len(manifest["frames"]), "first_timestamp_ms": manifest["frames"][0]["timestamp_ms"], "last_timestamp_ms": manifest["frames"][-1]["timestamp_ms"], "output": str(Path(args.output).resolve())}, indent=2))


if __name__ == "__main__":
    main()
