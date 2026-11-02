"""Step 6C: conservative multi-colour tracking on timestamped extracted frames.

HSV baseline only. Track IDs are provisional; occlusion/crossings can cause switches.
Coordinates are pixels, not metres. Does not infer physical motion or identity.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from app.perception.color_tracker import HSV_RANGES


def detect_blobs(image, colors=("red", "blue", "green"), min_area=80.0):
    import cv2
    import numpy as np
    if not math.isfinite(min_area) or min_area <= 0:
        raise ValueError("min_area must be positive")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    found = []
    for color in colors:
        if color not in HSV_RANGES:
            raise ValueError(f"Unsupported color: {color}")
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low, high in HSV_RANGES[color]:
            mask |= cv2.inRange(hsv, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            moments = cv2.moments(cnt)
            if moments["m00"] <= 0:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            found.append({"color": color, "center_px": [round(moments["m10"] / moments["m00"], 2), round(moments["m01"] / moments["m00"], 2)], "bbox_xywh_px": [int(x), int(y), int(w), int(h)], "area_px2": round(float(area), 2)})
    return sorted(found, key=lambda d: (d["color"], d["center_px"][0], d["center_px"][1]))


class NearestNeighbourTracker:
    def __init__(self, max_distance_px=65.0, max_missed=2):
        if not math.isfinite(max_distance_px) or max_distance_px <= 0 or max_missed < 0:
            raise ValueError("Invalid tracker settings")
        self.max_distance_px = max_distance_px
        self.max_missed = max_missed
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, timestamp_ms):
        if not math.isfinite(timestamp_ms):
            raise ValueError("Invalid timestamp")
        # Greedy, mutually exclusive nearest-neighbour assignment by colour.
        candidates = []
        for tid, track in self.tracks.items():
            for j, detection in enumerate(detections):
                if track["color"] != detection["color"]:
                    continue
                dx = track["center_px"][0] - detection["center_px"][0]
                dy = track["center_px"][1] - detection["center_px"][1]
                distance = math.hypot(dx, dy)
                if distance <= self.max_distance_px:
                    candidates.append((distance, tid, j))
        used_tracks, used_detections = set(), set()
        matches = {}
        for _, tid, j in sorted(candidates):
            if tid not in used_tracks and j not in used_detections:
                used_tracks.add(tid)
                used_detections.add(j)
                matches[tid] = j
        output = []
        for tid in list(self.tracks):
            if tid in matches:
                detection = detections[matches[tid]]
                self.tracks[tid].update(center_px=detection["center_px"], missed=0)
                output.append({"track_id": tid, **detection})
            else:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed:
                    del self.tracks[tid]
        for j, detection in enumerate(detections):
            if j not in used_detections:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"color": detection["color"], "center_px": detection["center_px"], "missed": 0}
                output.append({"track_id": tid, **detection})
        return sorted(output, key=lambda x: x["track_id"])


def track_manifest(manifest_path, output_path=None, colors=("red", "blue", "green"), min_area=80.0, max_distance_px=65.0, max_missed=2):
    import cv2
    path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest.get("frames"), list):
        raise ValueError("Manifest lacks frames")
    tracker = NearestNeighbourTracker(max_distance_px, max_missed)
    frames = []
    for item in manifest["frames"]:
        rel = Path(item["file"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe path in manifest")
        image = cv2.imread(str(path.parent / rel))
        if image is None:
            raise ValueError(f"Unreadable frame: {rel}")
        detections = detect_blobs(image, colors, min_area)
        frames.append({"frame_index": item["frame_index"], "timestamp_ms": item["timestamp_ms"], "objects": tracker.update(detections, float(item["timestamp_ms"]))})
    report = {"source_manifest": str(path), "method": "HSV blobs + greedy nearest neighbour (provisional IDs)", "coordinate_system": "image_pixels", "frames_checked": len(frames), "unique_track_ids": len({o["track_id"] for f in frames for o in f["objects"]}), "total_detections": sum(len(f["objects"]) for f in frames), "frames": frames}
    if output_path:
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description="Qapha Step 6C: provisional multi-object colour tracks")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", default="../datasets/recorded/multi_tracks.json")
    p.add_argument("--max-distance", type=float, default=65)
    p.add_argument("--max-missed", type=int, default=2)
    args = p.parse_args()
    try:
        result = track_manifest(args.manifest, args.output, max_distance_px=args.max_distance, max_missed=args.max_missed)
    except (ValueError, OSError, KeyError) as exc:
        p.exit(1, f"ERROR: {exc}\n")
    print(json.dumps({k: result[k] for k in ("frames_checked", "unique_track_ids", "total_detections")}, indent=2))
    print(f"Saved: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
