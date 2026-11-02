
"""Create a calibration-only video from a registered held-out clip."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def extract_calibration(
    protocol: dict,
    *,
    output_video: str | Path,
) -> dict:
    if protocol.get("schema") != "qapha.heldout_protocol.v1":
        raise ValueError("Invalid held-out protocol")

    if protocol.get("stage") != (
        "registered_before_future_tracking"
    ):
        raise ValueError("Unexpected protocol stage")

    split = protocol.get("calibration_frames_inclusive")
    video = protocol.get("video")
    policy = protocol.get("evaluation_policy")

    if (
        not isinstance(split, list)
        or len(split) != 2
        or split[0] != 0
        or type(split[1]) is not int
        or split[1] < 0
        or not isinstance(video, dict)
        or not isinstance(policy, dict)
        or policy.get("perception_calibration_only") is not True
    ):
        raise ValueError("Invalid calibration specification")

    source = Path(video["path"]).resolve()
    destination = Path(output_video).resolve()

    if not source.is_file():
        raise FileNotFoundError(source)

    if destination == source:
        raise ValueError("Cannot overwrite source video")

    if destination.suffix.lower() != ".mp4":
        raise ValueError("Output must be an MP4 file")

    if destination.exists():
        raise FileExistsError(destination)

    source_hash = sha256_file(source)

    if source_hash != video.get("sha256"):
        raise ValueError(
            "Source video fingerprint does not match registration"
        )

    expected_frames = split[1] + 1

    fps = video.get("fps")
    resolution = video.get("resolution_px")

    if (
        type(fps) not in (int, float)
        or not math.isfinite(fps)
        or fps <= 0
        or not isinstance(resolution, list)
        or len(resolution) != 2
        or any(type(v) is not int or v <= 0 for v in resolution)
    ):
        raise ValueError("Invalid registered video metadata")

    width, height = resolution

    destination.parent.mkdir(
        parents=True, exist_ok=True
    )

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("Cannot open source video")

    writer = None
    written = 0

    try:
        actual_fps = capture.get(cv2.CAP_PROP_FPS)

        if not math.isclose(
            actual_fps, fps, rel_tol=0.001, abs_tol=0.01
        ):
            raise ValueError("Source FPS changed since registration")

        writer = cv2.VideoWriter(
            str(destination),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError("Cannot create calibration video")

        # Strictly read frames 0 through cutoff.
        # No verification frame is submitted to the writer.
        for _ in range(expected_frames):
            ok, frame = capture.read()

            if not ok:
                raise ValueError(
                    "Source ended before calibration cutoff"
                )

            if frame.shape[:2] != (height, width):
                raise ValueError("Unexpected source resolution")

            writer.write(frame)
            written += 1

    finally:
        capture.release()

        if writer is not None:
            writer.release()

    # Verify the resulting file is decodable and contains
    # exactly the intended number of frames.
    check = cv2.VideoCapture(str(destination))
    if not check.isOpened():
        raise RuntimeError("Cannot verify calibration clip")

    decoded = 0

    try:
        while True:
            ok, _ = check.read()

            if not ok:
                break

            decoded += 1
    finally:
        check.release()

    if decoded != expected_frames:
        raise RuntimeError(
            f"Frame-count mismatch: expected {expected_frames}, "
            f"decoded {decoded}"
        )

    return {
        "schema": "qapha.calibration_clip.v1",
        "source_sha256": source_hash,
        "source_video": str(source),
        "calibration_video": str(destination),
        "calibration_sha256": sha256_file(destination),
        "frame_range_inclusive": [0, split[1]],
        "frames_written": written,
        "frames_verified": decoded,
        "fps": fps,
        "resolution_px": resolution,
        "future_frames_included": False,
        "note": (
            "Calibration-only MP4. Re-encoding can change "
            "pixel values. Frame indexing is preserved, "
            "but this is not a lossless copy. "
            "Verification must use a separate pipeline."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create a calibration-only held-out video"
    )

    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--output-manifest", required=True)

    args = parser.parse_args()

    protocol_path = Path(args.protocol).resolve()
    output_video = Path(args.output_video).resolve()
    manifest_path = Path(args.output_manifest).resolve()

    if len({
        protocol_path, output_video, manifest_path
    }) != 3:
        parser.error("All paths must differ")

    if manifest_path.exists():
        raise FileExistsError(manifest_path)

    protocol = json.loads(
        protocol_path.read_text(encoding="utf-8")
    )

    report = extract_calibration(
        protocol,
        output_video=output_video,
    )

    manifest_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
