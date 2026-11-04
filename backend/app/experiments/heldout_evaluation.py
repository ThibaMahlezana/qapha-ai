
"""Audit frozen Qapha predictions against future SAM observations."""

import argparse
import json
import math
from pathlib import Path

from app.experiments.heldout_forecast import freeze_forecast


def evaluate(frozen, calibration_tracks, calibration_manifest,
             verification_tracks, protocol, *, verification_start=50):

    if (frozen.get("schema") != "qapha.heldout_forecast.v1"
            or frozen.get("stage") != "frozen_before_future_verification"):
        raise ValueError("Invalid frozen forecast")

    if protocol.get("schema") != "qapha.heldout_protocol.v1":
        raise ValueError("Invalid evaluation protocol")

    cutoff = frozen["cutoff_frame"]
    if type(cutoff) is not int or cutoff < 0:
        raise ValueError("Invalid cutoff")

    if verification_start != cutoff + 1:
        raise ValueError("Verification clip must start after cutoff")

    if protocol.get("calibration_frames_inclusive") != [0, cutoff]:
        raise ValueError("Protocol cutoff mismatch")

    targets = protocol.get("target_frames")
    predictions = frozen.get("predictions")

    if (not isinstance(targets, list) or not targets
            or any(type(t) is not int for t in targets)
            or targets != sorted(set(targets))):
        raise ValueError("Invalid registered targets")

    if (not isinstance(predictions, list)
            or [p.get("frame_index") for p in predictions] != targets):
        raise ValueError("Forecast targets differ from protocol")

    horizons = tuple(target - cutoff for target in targets)
    if any(h <= 0 for h in horizons):
        raise ValueError("Targets must be after cutoff")

    count = frozen.get("training_sample_count")
    if (type(count) is not int or count < 5
            or frozen.get("training_frames")
            != [cutoff - count + 1, cutoff]):
        raise ValueError("Invalid training interval")

    if (calibration_manifest.get("source_sha256")
            != protocol.get("video", {}).get("sha256")):
        raise ValueError("Original video fingerprint mismatch")

    if (calibration_manifest.get("calibration_sha256")
            != frozen.get("calibration_video_sha256")):
        raise ValueError("Calibration fingerprint mismatch")

    if frozen.get("fps") != protocol["video"]["fps"]:
        raise ValueError("Frame-rate mismatch")

    # Rebuild the predictions entirely from calibration evidence.
    reproduced = freeze_forecast(
        calibration_tracks,
        calibration_manifest,
        cutoff=cutoff,
        object_id=frozen["object_id"],
        training_samples=count,
        horizons=horizons,
    )

    if reproduced != frozen:
        raise ValueError(
            "Frozen forecast does not match calibration observations"
        )

    if (verification_tracks.get("outcome") != "completed"
            or not isinstance(verification_tracks.get("boxes"), list)):
        raise ValueError("Verification tracking incomplete")

    if verification_tracks.get("concept") != calibration_tracks.get("concept"):
        raise ValueError("Calibration and verification prompts differ")

    width, height = protocol["video"]["resolution_px"]

    if (type(width) is not int or type(height) is not int
            or width <= 0 or height <= 0):
        raise ValueError("Invalid image dimensions")

    observations = {}
    max_index = max(targets) - verification_start

    for entry in verification_tracks["boxes"]:
        index = entry.get("frame_index")

        if type(index) is not int or not 0 <= index <= max_index:
            raise ValueError("Unexpected verification frame")

        if str(entry.get("object_id")) != frozen["object_id"]:
            raise ValueError("Unexpected verification object ID")

        if index in observations:
            raise ValueError("Duplicate verification observation")

        bbox = entry.get("bbox_xyxy_px")

        if (not isinstance(bbox, list) or len(bbox) != 4
                or any(type(v) not in (int, float)
                       or not math.isfinite(v) for v in bbox)):
            raise ValueError("Invalid bounding box")

        x1, y1, x2, y2 = bbox

        if not (0 <= x1 < x2 <= width
                and 0 <= y1 < y2 <= height):
            raise ValueError("Bounding box outside image")

        observations[index] = bbox

    evaluations = []

    for prediction in predictions:
        original_frame = prediction["frame_index"]
        clip_frame = original_frame - verification_start

        if clip_frame not in observations:
            raise ValueError(
                f"Missing observation for frame {original_frame}"
            )

        x1, y1, x2, y2 = observations[clip_frame]
        actual = [(x1 + x2) / 2, (y1 + y2) / 2]

        errors = {}

        for model, field in (
            ("linear", "linear_center_px"),
            ("quadratic_y", "quadratic_y_center_px"),
        ):
            predicted = prediction[field]

            if (not isinstance(predicted, list)
                    or len(predicted) != 2
                    or any(type(v) not in (int, float)
                           or not math.isfinite(v)
                           for v in predicted)):
                raise ValueError("Invalid prediction position")

            errors[model] = round(
                math.dist(actual, predicted), 6
            )

        evaluations.append({
            "original_frame": original_frame,
            "verification_clip_frame": clip_frame,
            "observed_center_px": actual,
            "bbox_xyxy_px": [x1, y1, x2, y2],
            "boundary_touch": (
                x1 <= 0 or y1 <= 0
                or x2 >= width or y2 >= height
            ),
            "errors_px": errors,
        })

    mean_errors = {
        model: round(
            sum(row["errors_px"][model] for row in evaluations)
            / len(evaluations),
            6,
        )
        for model in ("linear", "quadratic_y")
    }

    return {
        "schema": "qapha.heldout_evaluation.v1",
        "frozen_training_sha256": frozen["training_sha256"],
        "calibration_reproduced_exactly": True,
        "source_sha256_matches_protocol": True,
        "image_resolution_px": [width, height],
        "evaluations": evaluations,
        "mean_error_px": mean_errors,
        "limitations": [
            "Object identity across clips requires visual confirmation.",
            "Verification JSON does not cryptographically prove clip origin.",
            "Errors use SAM bounding-box centres, not manual ground truth.",
            "Neither model was selected in advance as the winner.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()

    names = (
        "forecast",
        "calibration-tracks",
        "calibration-manifest",
        "verification-tracks",
        "protocol",
        "output",
    )

    for name in names:
        parser.add_argument(f"--{name}", required=True)

    args = parser.parse_args()

    input_paths = [
        Path(getattr(args, name.replace("-", "_"))).resolve()
        for name in names[:-1]
    ]
    output_path = Path(args.output).resolve()

    if output_path in input_paths:
        parser.error("Output path must differ from inputs")

    inputs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in input_paths
    ]

    report = evaluate(*inputs)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
