"""Archive an ineligible candidate video and seal a deviation record.

Used only BEFORE any tracking: if a candidate clip fails eligibility on
calibration frames (e.g. target absent at frame 0), its intake files are
moved, unmodified, into an archive folder and a sealed record explains
why.  Nothing is deleted.  Refuses if tracks or evaluations exist.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


SCHEMA = "qapha.intake_deviation.v1"
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


def verified_hash(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {k: v for k, v in data.items() if k != "registration_sha256"}
    if digest(body) != data.get("registration_sha256"):
        raise ValueError(f"Integrity check failed: {Path(path).name}")
    return data["registration_sha256"]


def candidate_files(recorded, prefix):
    files = sorted(p for p in recorded.glob(f"{prefix}*") if p.is_file())
    return [p for p in files
            if p.name == f"{prefix}.mp4" or p.name.startswith(f"{prefix}_")]


def _refuse_tracking(names):
    tracked = [n for n in names if any(m in n for m in TRACKING_MARKERS)]
    if tracked:
        raise ValueError(f"Tracking artifacts exist; not an intake "
                         f"deviation: {tracked}")


def _base_record(registration, addendum, reason, evidence,
                 registered_at_utc, archive_name, entries):
    return {
        "schema": SCHEMA,
        "registration_sha256": verified_hash(registration),
        "addendum_sha256": verified_hash(addendum),
        "recorded_at_utc_self_reported": registered_at_utc,
        "stage": "pre_tracking_intake",
        "decision": "candidate_ineligible_replaced_before_tracking",
        "reason": reason,
        "evidence": evidence,
        "tracking_performed": False,
        "evaluation_performed": False,
        "archived_to": archive_name,
        "archived_files": entries,
        "parameters_changed": False,
        "limitations": [
            "Eligibility was judged by a person from calibration frames.",
            "Self-reported timestamp; git commit provides external record.",
        ],
    }


def _seal(record):
    return {**record, "record_sha256": digest(record)}


def archive_attempt(recorded, prefix, archive_name, registration,
                    addendum, reason, evidence, registered_at_utc,
                    write_record=None):
    """Seal the record FIRST (via write_record), then move the files."""
    recorded = Path(recorded)
    files = candidate_files(recorded, prefix)
    if not any(p.name == f"{prefix}.mp4" for p in files):
        raise FileNotFoundError(f"{prefix}.mp4 not found")
    _refuse_tracking(p.name for p in files)
    archive = recorded / archive_name
    if archive.exists():
        raise FileExistsError(archive)

    entries = [{"file": p.name, "sha256": file_sha256(p)} for p in files]
    record = _base_record(registration, addendum, reason, evidence,
                          registered_at_utc, archive_name, entries)
    record["archive_method"] = "moved_by_script_after_record_sealed"
    record = _seal(record)
    if write_record is not None:
        write_record(record)

    archive.mkdir()
    for p in files:
        p.rename(archive / p.name)
    for entry in entries:
        if file_sha256(archive / entry["file"]) != entry["sha256"]:
            raise RuntimeError(f"Archive verification failed: {entry['file']}")
    return record


def record_existing_archive(recorded, prefix, archive_name, registration,
                            addendum, reason, evidence, registered_at_utc):
    """Seal a record for files that were already moved by hand.

    Archived files are checked against the hashes written at intake time
    (protocol and calibration manifest), so the record can show they are
    the unmodified originals.
    """
    recorded = Path(recorded)
    archive = recorded / archive_name
    if not archive.is_dir():
        raise FileNotFoundError(archive)
    files = sorted(p for p in archive.iterdir() if p.is_file())
    names = [p.name for p in files]
    _refuse_tracking(names)
    _refuse_tracking(p.name for p in candidate_files(recorded, prefix))

    video = archive / f"{prefix}.mp4"
    protocol_path = archive / f"{prefix}_protocol.json"
    manifest_path = archive / f"{prefix}_calibration_manifest.json"
    for required in (video, protocol_path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    video_hash = file_sha256(video)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calibration = archive / Path(manifest["calibration_video"]).name
    checks = {
        "video_matches_protocol":
            protocol["video"]["sha256"] == video_hash,
        "video_matches_calibration_manifest":
            manifest["source_sha256"] == video_hash,
        "calibration_clip_matches_manifest":
            calibration.is_file()
            and file_sha256(calibration) == manifest["calibration_sha256"],
    }
    if not all(checks.values()):
        raise ValueError(f"Archived files do not match intake hashes: {checks}")

    entries = [{"file": p.name, "sha256": file_sha256(p)} for p in files]
    record = _base_record(registration, addendum, reason, evidence,
                          registered_at_utc, archive_name, entries)
    record["archive_method"] = "moved_manually_before_record_sealed"
    record["integrity_checks_against_intake_hashes"] = checks
    replacement = recorded / f"{prefix}.mp4"
    record["replacement_candidate_sha256_at_record_time"] = (
        file_sha256(replacement) if replacement.is_file() else None)
    return _seal(record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recorded-dir", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--archive-name", required=True)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--addendum", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--evidence", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--already-archived", action="store_true",
                        help="files were moved by hand; record only")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)

    def write(record):
        with output.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)

    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    common = (args.recorded_dir, args.prefix, args.archive_name,
              args.registration, args.addendum, args.reason, args.evidence,
              now.isoformat())
    if args.already_archived:
        record = record_existing_archive(*common)
        write(record)
    else:
        record = archive_attempt(*common, write_record=write)
    print(json.dumps({k: record[k] for k in
                      ("record_sha256", "decision", "archived_to")}, indent=2))
    for entry in record["archived_files"]:
        print("archived:", entry["file"])
    print("Saved:", output)


if __name__ == "__main__":
    main()