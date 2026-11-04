"""Seal the tracking-target choice BEFORE SAM tracking is run.

Links the chosen target and SAM concept text to the exact calibration
clip and the preview image used to choose it.  Refuses if any tracking
or evaluation artifact for the clip already exists, so the target
cannot be chosen (or re-chosen) after seeing results.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


SCHEMA = "qapha.target_selection.v1"
TRACKING_MARKERS = ("tracks", "evaluation", "annotated", "replay")


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


def build_target_record(registration_path, manifest_path, preview_path,
                        description, sam_concept, notes, recorded_at_utc):
    registration = json.loads(Path(registration_path).read_text(encoding="utf-8"))
    body = {k: v for k, v in registration.items()
            if k != "registration_sha256"}
    if digest(body) != registration.get("registration_sha256"):
        raise ValueError("Registration integrity check failed")

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    clip = Path(manifest["calibration_video"])
    if file_sha256(clip) != manifest["calibration_sha256"]:
        raise ValueError("Calibration clip does not match its manifest")

    stem = Path(registration["video"]).stem
    existing = [p.name for p in clip.parent.glob(f"{stem}_*")
                if any(m in p.name for m in TRACKING_MARKERS)]
    if existing:
        raise ValueError(f"Tracking artifacts already exist: {existing}")

    if not description.strip() or not sam_concept.strip():
        raise ValueError("Description and SAM concept are required")

    record = {
        "schema": SCHEMA,
        "experiment_id": registration["experiment_id"],
        "registration_sha256": registration["registration_sha256"],
        "recorded_at_utc_self_reported": recorded_at_utc,
        "source_video_sha256": manifest["source_sha256"],
        "calibration_video_sha256": manifest["calibration_sha256"],
        "preview_image": Path(preview_path).name,
        "preview_image_sha256": file_sha256(preview_path),
        "target_description": description.strip(),
        "sam_concept": sam_concept.strip(),
        "notes": notes,
        "tracking_performed": False,
        "no_target_reselection_after_performance_review": True,
    }
    return {**record, "record_sha256": digest(record)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--preview", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--sam-concept", required=True)
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    record = build_target_record(
        args.registration, args.manifest, args.preview, args.description,
        args.sam_concept, args.note, now.isoformat())
    with output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps(record, indent=2))
    print("Saved:", output)


if __name__ == "__main__":
    main()