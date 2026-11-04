"""Cut the registered analysis window from every fresh clip.

For each clip in a sealed bounce-test registration: verify its SHA-256,
write the first N frames to <analysis-dir>/<role>.mp4, write a labelled
preview sheet, and record everything in an intake manifest.  Every clip
is processed; none is skipped by choice.  Existing outputs are never
overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


SCHEMA = "qapha.fresh_clip_intake.v1"


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


def frames_from_registration(registration):
    """Number of frames to keep, or None for the whole clip."""
    text = registration["analysis"]["frames_per_clip"]
    if text == "all frames":
        return None
    return int(text.split("(first ")[1].split(" ")[0])


def cut_clip(source, destination, frames):
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        return {"status": "unreadable", "frames_written": 0}
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    writer, kept = None, []
    try:
        while frames is None or len(kept) < frames:
            ok, image = capture.read()
            if not ok:
                break
            if writer is None:
                h, w = image.shape[:2]
                writer = cv2.VideoWriter(str(destination),
                                         cv2.VideoWriter_fourcc(*"mp4v"),
                                         float(fps), (w, h))
            writer.write(image)
            kept.append(image)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    status = "ok" if frames is None or len(kept) == frames else "short"
    return {"status": status, "frames_written": len(kept), "fps": float(fps),
            "resolution_px": ([kept[0].shape[1], kept[0].shape[0]]
                              if kept else None),
            "images": kept}


def preview_sheet(images, picks, tile_width=320):
    tiles = []
    for i in picks:
        if i >= len(images):
            continue
        img = images[i]
        h, w = img.shape[:2]
        tile = cv2.resize(img, (tile_width, round(h * tile_width / w)))
        cv2.rectangle(tile, (0, 0), (120, 30), (0, 0, 0), -1)
        cv2.putText(tile, f"frame {i}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    height = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.zeros((height - t.shape[0], t.shape[1], 3),
                                    np.uint8)]) for t in tiles]
    return np.hstack(tiles)


def run_intake(registration, clip_folder, analysis_dir):
    body = {k: v for k, v in registration.items()
            if k != "registration_sha256"}
    if digest(body) != registration.get("registration_sha256"):
        raise ValueError("Registration integrity check failed")
    frames = frames_from_registration(registration)
    clip_folder, analysis_dir = Path(clip_folder), Path(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for clip in registration["clips"]:
        source = clip_folder / clip["file"]
        entry = {"clip_id": clip["clip_id"], "role": clip["role"],
                 "group": clip["group"]}
        if not source.is_file():
            entries.append({**entry, "status": "missing"})
            continue
        if file_sha256(source) != clip["sha256"]:
            raise ValueError(f"{clip['file']} differs from the registration")
        cut_path = analysis_dir / f"{clip['role']}.mp4"
        preview_path = analysis_dir / f"{clip['role']}_preview.jpg"
        for p in (cut_path, preview_path):
            if p.exists():
                raise FileExistsError(p)
        result = cut_clip(source, cut_path, frames)
        images = result.pop("images", [])
        if images:
            n = len(images)
            picks = sorted({0, n // 3, (2 * n) // 3, n - 1})
            cv2.imwrite(str(preview_path), preview_sheet(images, picks))
        entries.append({
            **entry, **result,
            "analysis_video": cut_path.name if images else None,
            "analysis_video_sha256": (file_sha256(cut_path)
                                      if images else None),
            "preview": preview_path.name if images else None,
        })
    manifest = {
        "schema": SCHEMA,
        "registration_sha256": registration["registration_sha256"],
        "frames_per_clip": frames,
        "note": ("All clips are tracked regardless of preview appearance; "
                 "eligibility is decided only by the registered rule."),
        "clips": entries,
    }
    return {**manifest, "manifest_sha256": digest(manifest)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", required=True)
    parser.add_argument("--clips", required=True)
    parser.add_argument("--analysis-dir", required=True)
    args = parser.parse_args()
    analysis = Path(args.analysis_dir)
    output = analysis / "intake_manifest.json"
    if output.exists():
        raise FileExistsError(output)
    registration = json.loads(Path(args.registration).read_text(encoding="utf-8"))
    manifest = run_intake(registration, args.clips, analysis)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    for c in manifest["clips"]:
        print(f"{c['clip_id']}  {c['role']:<18} {c['status']:<10} "
              f"frames={c.get('frames_written')}  res={c.get('resolution_px')}")
    print("manifest_sha256:", manifest["manifest_sha256"])
    print("Saved:", output)


if __name__ == "__main__":
    main()