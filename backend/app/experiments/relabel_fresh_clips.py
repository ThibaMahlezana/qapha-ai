"""Re-register renamed fresh clips, proving they are the same files.

Renaming does not change a file's SHA-256, so every clip in the folder
must match exactly one clip in the previous sealed registry, and every
previously registered clip must still be present.  Nothing may be added,
removed or edited.  The new registry records the descriptive names (the
role of each clip), who has viewed the clips, and links to the previous
registry.  The previous registry file is never modified.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

from app.experiments.register_fresh_clips import (
    VIDEO_SUFFIXES,
    clip_metadata,
    digest,
    file_sha256,
)


SCHEMA = "qapha.fresh_clip_registry.v2"
NAME = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)*\.(mp4|mov|m4v|avi|mkv)$")


def verify_registry(registry):
    body = {k: v for k, v in registry.items() if k != "registry_sha256"}
    if digest(body) != registry.get("registry_sha256"):
        raise ValueError("Previous registry integrity check failed")


def relabel(previous, folder, registered_at_utc, viewing_note):
    verify_registry(previous)
    folder = Path(folder)
    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
    bad_names = [p.name for p in files if not NAME.match(p.name)]
    if bad_names:
        raise ValueError(
            "Use lowercase names like drop_050cm.mp4 (letters, digits, "
            f"underscores): {bad_names}")

    old_by_hash = {c["sha256"]: c for c in previous["clips"]}
    if len(old_by_hash) != len(previous["clips"]):
        raise ValueError("Previous registry contains duplicate clips")
    new_hashes = {}
    for p in files:
        h = file_sha256(p)
        if h in new_hashes:
            raise ValueError(f"Duplicate clip: {p.name} = {new_hashes[h]}")
        new_hashes[h] = p.name

    added = sorted(n for h, n in new_hashes.items() if h not in old_by_hash)
    missing = sorted(c["file"] for h, c in old_by_hash.items()
                     if h not in new_hashes)
    if added or missing:
        raise ValueError(f"Clips differ from the sealed registry. "
                         f"New or edited: {added}; missing: {missing}")

    clips = []
    for p in files:
        h = file_sha256(p)
        old = old_by_hash[h]
        clips.append({
            "clip_id": old["clip_id"],
            "file": p.name,
            "role": p.stem,
            "previous_file": old["file"],
            "sha256": h,
            "size_bytes": p.stat().st_size,
            **clip_metadata(p),
            "meets_minimum_length": old["meets_minimum_length"],
        })
    body = {
        "schema": SCHEMA,
        "registered_at_utc_self_reported": registered_at_utc,
        "folder": folder.name,
        "supersedes_registry_sha256": previous["registry_sha256"],
        "content_unchanged_since_previous_registry": True,
        "viewing": {
            "viewed_by_owner_for_labelling": True,
            "viewed_by_method_developer": False,
            "tracking_or_analysis_performed": False,
            "note": viewing_note,
        },
        "clips": clips,
        "rules": previous.get("rules", []),
    }
    return {**body, "registry_sha256": digest(body)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", required=True)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--note", default="Owner watched each clip once at "
                        "normal speed to identify and rename it.")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    previous = json.loads(Path(args.previous).read_text(encoding="utf-8"))
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    registry = relabel(previous, args.folder, now.isoformat(), args.note)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(registry, stream, indent=2)
    for c in registry["clips"]:
        print(f"{c['clip_id']}  {c['file']:<26} was {c['previous_file']}")
    print("registry_sha256:", registry["registry_sha256"])
    print("Saved:", output)


if __name__ == "__main__":
    main()