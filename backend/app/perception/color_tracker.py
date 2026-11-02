"""Step 6B: controlled-colour object tracking in Step 6A extracted frames.

This is a deterministic perception baseline, NOT general-purpose object recognition.
Coordinates are image pixels; no physical units or identity across multiple objects.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


HSV_RANGES = {
    "red": [((0, 90, 70), (10, 255, 255)), ((170, 90, 70), (179, 255, 255))],
    "blue": [((90, 90, 70), (135, 255, 255))],
    "green": [((35, 80, 60), (85, 255, 255))],
}


def locate_colored_object(image, color: str = "red", min_area_px: float = 80.0):
    """Return largest colour blob's bounding box/centroid, or None when absent."""
    import cv2
    import numpy as np

    if color not in HSV_RANGES:
        raise ValueError(f"Unsupported colour: {color}. Choose {', '.join(HSV_RANGES)}")
    if min_area_px <= 0:
        raise ValueError("min_area_px must be positive")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in HSV_RANGES[color]:
        mask |= cv2.inRange(hsv, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < min_area_px:
        return None
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    return {"center_px": [round(moments["m10"] / moments["m00"], 2), round(moments["m01"] / moments["m00"], 2)], "bbox_xywh_px": [x, y, w, h], "area_px2": round(area, 2)}


def track_extracted_frames(manifest_path: str | Path, color: str = "red", min_area_px: float = 80.0, output_path: str | Path | None = None) -> dict:
    import cv2

    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(manifest.get("frames"), list):
        raise ValueError("Manifest lacks frames list")
    detections = []
    for item in manifest["frames"]:
        frame_name = Path(item["file"])
        if frame_name.is_absolute() or ".." in frame_name.parts:
            raise ValueError("Invalid frame path in manifest")
        image = cv2.imread(str(manifest_file.parent / frame_name))
        if image is None:
            raise ValueError(f"Unreadable frame: {frame_name}")
        located = locate_colored_object(image, color, min_area_px)
        detections.append({"frame_index": item["frame_index"], "timestamp_ms": item["timestamp_ms"], "detected": located is not None, "measurement": located})
    report = {"source_manifest": str(manifest_file), "method": "HSV colour segmentation (controlled conditions only)", "color": color, "coordinate_system": "image_pixels", "frames_checked": len(detections), "frames_detected": sum(d["detected"] for d in detections), "detections": detections}
    if output_path is not None:
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Qapha Step 6B: controlled-colour tracking")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--color", choices=list(HSV_RANGES), default="red")
    parser.add_argument("--min-area", type=float, default=80.0)
    parser.add_argument("--output", default="../datasets/recorded/tracks.json")
    args = parser.parse_args()
    try:
        result = track_extracted_frames(args.manifest, args.color, args.min_area, args.output)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(json.dumps({"frames_checked": result["frames_checked"], "frames_detected": result["frames_detected"], "colour": args.color, "output": str(Path(args.output).resolve()), "first_detection": next((d for d in result["detections"] if d["detected"]), None)}, indent=2))


if __name__ == "__main__":
    main()
