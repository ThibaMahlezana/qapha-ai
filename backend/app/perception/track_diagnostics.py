
"""Render provisional tracking detections over the ORIGINAL video (Step 6D).

This is a diagnostic renderer, not a tracker or identity-correction algorithm.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def summarize_tracks(report: dict) -> list[dict]:
    if not isinstance(report.get("frames"), list):
        raise ValueError("Tracking report must contain a frames list")
    tracks = defaultdict(list)
    for frame in report["frames"]:
        for obj in frame["objects"]:
            tracks[int(obj["track_id"])].append((int(frame["frame_index"]), obj["color"]))
    return [
        {"track_id": tid, "color": observations[0][1],
         "detections": len(observations), "first_frame": observations[0][0],
         "last_frame": observations[-1][0]}
        for tid, observations in sorted(tracks.items())
    ]


def annotate_video(video_path: str | Path, tracks_path: str | Path, output_path: str | Path, trail_length: int = 12) -> dict:
    import cv2
    if trail_length < 0 or trail_length > 100:
        raise ValueError("trail_length must be between 0 and 100")
    video = Path(video_path).expanduser().resolve()
    track_file = Path(tracks_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not video.is_file() or not track_file.is_file():
        raise FileNotFoundError("Video or tracks JSON does not exist")
    if output == video:
        raise ValueError("Output must not overwrite the source video")
    report = json.loads(track_file.read_text(encoding="utf-8"))
    summary = summarize_tracks(report)
    frames = report["frames"]
    indices = [int(frame["frame_index"]) for frame in frames]
    if indices != sorted(set(indices)) or any(i < 0 for i in indices):
        raise ValueError("Frame indices must be unique, nonnegative and increasing")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError("Could not open source video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Invalid source video metadata")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError("Could not create output video")
    history: dict[int, list[tuple[int, int]]] = defaultdict(list)
    frame_lookup = {i: f for i, f in zip(indices, frames)}
    rendered = 0
    try:
        for index in range(indices[-1] + 1 if indices else 0):
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"Video ended before report frame {index}")
            row = frame_lookup.get(index)
            if row is not None:
                for obj in row["objects"]:
                    tid = int(obj["track_id"])
                    x, y, w, h = map(int, obj["bbox_xywh_px"])
                    cx, cy = map(int, obj["center_px"])
                    color = (255, 180, 30) if obj["color"] == "blue" else (60, 80, 255) if obj["color"] == "red" else (70, 220, 70)
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(image, f"ID {tid} {obj['color']}", (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1, cv2.LINE_AA)
                    history[tid].append((cx, cy))
                    history[tid] = history[tid][-max(trail_length, 1):]
                    if trail_length > 1 and len(history[tid]) > 1:
                        for p, q in zip(history[tid], history[tid][1:]):
                            cv2.line(image, p, q, color, 2)
                cv2.putText(image, f"Frame {index} | detections {len(row['objects'])} | provisional IDs", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(image)
            rendered += 1
    finally:
        cap.release()
        writer.release()
    return {"frames_rendered": rendered, "tracks": len(summary), "short_tracks_3_frames_or_less": sum(t["detections"] <= 3 for t in summary), "track_summary": summary, "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description="Overlay tracked colour blobs onto a video for inspection")
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = annotate_video(args.video, args.tracks, args.output)
    except (ValueError, FileNotFoundError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
