"""Seal a batch of fresh, unopened clips BEFORE any frame is viewed.

Records each video's SHA-256 fingerprint and container metadata (frame
rate, reported frame count, resolution, file size).  Only metadata is
read: no frame is decoded, so no motion is seen.  The manifest is
written once and never overwritten.  Later experiments must reference
clips by these fingerprints, so a clip cannot be silently replaced.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import cv2


SCHEMA = "qapha.fresh_clip_registry.v1"
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def digest(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clip_metadata(path):
    """Container metadata only; no frame is read."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return {"readable": False}
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    ok = math.isfinite(fps) and fps > 0 and frames > 0 and width > 0
    return {
        "readable": bool(ok),
        "fps": round(fps, 4) if math.isfinite(fps) else None,
        "frame_count_reported": frames,
        "resolution_px": [width, height],
        "duration_s": round(frames / fps, 3) if ok else None,
    }


def build_registry(folder, registered_at_utc, minimum_frames=71, id_prefix="F",
                   note=None):
    folder = Path(folder)
    if not (id_prefix.isalpha() and id_prefix.isupper()):
        raise ValueError("id_prefix must be capital letters, e.g. F or H")
    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
    if not files:
        raise ValueError(f"No video files found in {folder}")
    clips = []
    for index, path in enumerate(files, start=1):
        meta = clip_metadata(path)
        frames = meta.get("frame_count_reported") or 0
        clips.append({
            "clip_id": f"{id_prefix}{index:02d}",
            "role": path.stem,
            "file": path.name,
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
            **meta,
            "meets_minimum_length": bool(meta["readable"]
                                         and frames >= minimum_frames),
        })
    body = {
        "schema": SCHEMA,
        "registered_at_utc_self_reported": registered_at_utc,
        "folder": folder.name,
        "frames_viewed_at_registration": 0,
        "note": note,
        "minimum_frames": minimum_frames,
        "clips": clips,
        "rules": [
            "Clips are a held-out test set: no tracking or analysis until a "
            "method is registered against them.",
            "Eligibility previews (if any) must be recorded before tracking.",
            "Failures and ineligible clips are reported, never dropped silently.",
        ],
    }
    return {**body, "registry_sha256": digest(body)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--id-prefix", default="F",
                        help="letters for clip IDs, e.g. H for H01, H02 ...")
    parser.add_argument("--note", default=None,
                        help="optional sentence recorded in the registry")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    registry = build_registry(args.folder, now.isoformat(),
                              id_prefix=args.id_prefix, note=args.note)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(registry, stream, indent=2)
    for c in registry["clips"]:
        print(f"{c['clip_id']}  {c['file']:<32} readable={c['readable']!s:<5} "
              f"fps={c.get('fps')}  frames={c.get('frame_count_reported')}  "
              f"res={c.get('resolution_px')}  long_enough={c['meets_minimum_length']}")
    print("registry_sha256:", registry["registry_sha256"])
    print("Saved:", output)


if __name__ == "__main__":
    main()