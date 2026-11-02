"""Visualize SAM 3.1 single-image segmentation from an existing smoke-test JSON.

Requires: opencv-python, numpy, meta-sam-parser. No API call is made.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


_RECORD = re.compile(r"(?:^|,)(\d+)<\|box;([^|]+)\|><\|mask;([^|]+)\|>")


def _fields(token: str) -> dict[str, str]:
    fields = {}
    for part in token.split(";"):
        key, sep, value = part.partition("=")
        if sep:
            fields[key] = value
    return fields


def parse_image_records(raw: str) -> list[dict]:
    """Parse the small single-image subset emitted by the current Qapha smoke test.

    Mask byte decoding is delegated to Meta's official meta_sam_parser.
    """
    if not isinstance(raw, str) or not raw.startswith("<0f>"):
        raise ValueError("Expected SAM single-image output starting with <0f>")
    content = raw[4:].strip()
    if not content:
        return []
    result = []
    position = 0
    for match in _RECORD.finditer(content):
        if match.start() != position:
            raise ValueError("Malformed or unsupported SAM record syntax")
        position = match.end()
        box = _fields(match.group(2))
        mask = _fields(match.group(3))
        try:
            x1, y1, x2, y2 = (int(box[k]) for k in ("x1", "y1", "x2", "y2"))
            frame_w, frame_h = int(box["w"]), int(box["h"])
            height, width, encoded = mask["data"].split(",", 2)
            mh, mw = int(height), int(width)
        except (KeyError, ValueError) as exc:
            raise ValueError("Invalid SAM record fields") from exc
        if not (0 <= x1 <= x2 < frame_w and 0 <= y1 <= y2 < frame_h):
            raise ValueError("Invalid box coordinates")
        if not (mh > 0 and mw > 0 and encoded and encoded[0] in "~!"):
            raise ValueError("Invalid mask dimensions or encoding")
        if (mw, mh) != (x2 - x1 + 1, y2 - y1 + 1):
            raise ValueError("Mask dimensions do not match bounding box")
        result.append({
            "id": int(match.group(1)), "box": (x1, y1, x2, y2),
            "frame_size": (frame_w, frame_h), "mask_size": (mw, mh),
            "encoding": "lossless" if encoded[0] == "~" else "one_bit",
            "payload": encoded, "confidence": float(box["c"]) if "c" in box else None,
        })
    if position != len(content):
        raise ValueError("Trailing invalid SAM output")
    return result


def render_overlay(image_path: str, result_path: str, output_path: str) -> dict:
    import cv2
    import numpy as np
    from meta_sam_parser import SegmentationMask, decode_mask_to_raster

    image_file, json_file, output_file = (Path(p).expanduser().resolve() for p in (image_path, result_path, output_path))
    if not image_file.is_file() or not json_file.is_file():
        raise FileNotFoundError("Image or SAM result JSON not found")
    if output_file == image_file:
        raise ValueError("Do not overwrite the input image")
    image = cv2.imread(str(image_file))
    if image is None:
        raise ValueError("Unable to decode input image")
    report = json.loads(json_file.read_text(encoding="utf-8"))
    records = parse_image_records(report["raw_segmentation"])
    palette = [(40, 190, 40), (220, 80, 40), (40, 80, 240), (160, 60, 200)]
    counts = []
    for record in records:
        if record["frame_size"] != (image.shape[1], image.shape[0]):
            raise ValueError("SAM image dimensions differ from input image")
        mw, mh = record["mask_size"]
        mask = SegmentationMask(encoding=record["encoding"], payload=record["payload"], width=mw, height=mh)
        binary = np.frombuffer(decode_mask_to_raster(mask), dtype=np.uint8).reshape((mh, mw))
        x1, y1, x2, y2 = record["box"]
        region = image[y1:y2+1, x1:x2+1]
        foreground = binary.astype(bool)
        color = np.array(palette[record["id"] % len(palette)], dtype=np.float32)
        region[foreground] = (region[foreground].astype(np.float32) * 0.55 + color * 0.45).astype(np.uint8)
        cv2.rectangle(image, (x1, y1), (x2, y2), tuple(map(int, color)), 2)
        cv2.putText(image, f"SAM {record['id']}", (x1, max(20, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, .55, tuple(map(int, color)), 2)
        counts.append(int(binary.sum()))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_file), image):
        raise OSError("Failed to save annotated image")
    return {"objects": len(records), "mask_areas_px2": counts, "output": str(output_file)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay existing Meta SAM masks on a source frame")
    parser.add_argument("--image", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(render_overlay(args.image, args.result, args.output), indent=2))
    except (ValueError, KeyError, FileNotFoundError, OSError, ImportError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
