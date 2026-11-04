"""Render a labelled preview of selected calibration frames.

Reads ONLY the calibration-only clip (frames 0..cutoff), verified against
its manifest, so no future frame can appear in the preview.  Used to
choose and record the tracking target before SAM is run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render_preview(manifest, frames=(0, 25, 49), tile_width=640):
    clip = Path(manifest["calibration_video"])
    if _sha256(clip) != manifest.get("calibration_sha256"):
        raise ValueError("Calibration clip does not match its manifest")
    last = manifest["frame_range_inclusive"][1]
    if any(type(f) is not int or not 0 <= f <= last for f in frames):
        raise ValueError("Preview frames must lie inside the calibration range")

    wanted = set(frames)
    found = {}
    capture = cv2.VideoCapture(str(clip))
    if not capture.isOpened():
        raise ValueError("Cannot open calibration clip")
    try:
        index = 0
        while len(found) < len(wanted):
            ok, image = capture.read()
            if not ok:
                break
            if index in wanted:
                found[index] = image
            index += 1
    finally:
        capture.release()
    if set(found) != wanted:
        raise ValueError(f"Could not decode frames {sorted(wanted - set(found))}")

    tiles = []
    for f in frames:
        image = found[f]
        h, w = image.shape[:2]
        tile = cv2.resize(image, (tile_width, round(h * tile_width / w)))
        cv2.rectangle(tile, (0, 0), (190, 44), (0, 0, 0), -1)
        cv2.putText(tile, f"frame {f}", (10, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    return np.hstack(tiles)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", default="0,25,49")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    frames = tuple(int(s) for s in args.frames.split(","))
    sheet = render_preview(manifest, frames)
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError("Could not write preview image")
    print(json.dumps({"preview": str(output), "frames": list(frames),
                      "calibration_sha256": manifest["calibration_sha256"]},
                     indent=2))


if __name__ == "__main__":
    main()