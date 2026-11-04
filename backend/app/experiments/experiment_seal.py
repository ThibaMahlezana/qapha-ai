
"""Reproduce and seal a calibration-only Qapha experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from app.experiments.heldout_forecast import freeze_forecast
from app.experiments.model_selection import select_model


def canonical_hash(value):
    data = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    return hashlib.sha256(data).hexdigest()


def valid_sha256(value):
    return (
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    )


def seal_experiment(
    selection,
    forecast,
    protocol,
    manifest,
    tracks,
):
    if selection.get("schema") != "qapha.model_selection.v1":
        raise ValueError("Invalid model-selection artifact")

    if forecast.get("schema") != "qapha.heldout_forecast.v1":
        raise ValueError("Invalid forecast artifact")

    if forecast.get("stage") != "frozen_before_future_verification":
        raise ValueError("Forecast must be frozen")

    if protocol.get("schema") != "qapha.heldout_protocol.v1":
        raise ValueError("Invalid protocol")

    if manifest.get("schema") != "qapha.calibration_clip.v1":
        raise ValueError("Invalid calibration manifest")

    if manifest.get("future_frames_included") is not False:
        raise ValueError("Future frames cannot be used for sealing")

    video = protocol.get("video")
    if not isinstance(video, dict):
        raise ValueError("Missing registered video")

    cutoff = forecast.get("cutoff_frame")
    object_id = forecast.get("object_id")

    if type(cutoff) is not int or cutoff < 0:
        raise ValueError("Invalid forecast cutoff")

    if not isinstance(object_id, str) or not object_id:
        raise ValueError("Invalid object ID")

    if selection.get("cutoff_frame") != cutoff:
        raise ValueError("Selection and forecast cutoffs differ")

    if selection.get("object_id") != object_id:
        raise ValueError("Selection and forecast objects differ")

    if protocol.get("calibration_frames_inclusive") != [0, cutoff]:
        raise ValueError("Registered calibration window differs")

    if manifest.get("frame_range_inclusive") != [0, cutoff]:
        raise ValueError("Calibration manifest window differs")

    if manifest.get("frames_verified") != cutoff + 1:
        raise ValueError("Calibration frame count differs")

    source_hash = video.get("sha256")
    calibration_hash = manifest.get("calibration_sha256")

    if not valid_sha256(source_hash):
        raise ValueError("Invalid original video fingerprint")

    if not valid_sha256(calibration_hash):
        raise ValueError("Invalid calibration fingerprint")

    if manifest.get("source_sha256") != source_hash:
        raise ValueError("Registered source video does not match")

    if forecast.get("calibration_video_sha256") != calibration_hash:
        raise ValueError("Forecast calibration fingerprint mismatch")

    if selection.get("calibration_sha256") != calibration_hash:
        raise ValueError("Selection calibration fingerprint mismatch")

    if forecast.get("fps") != video.get("fps"):
        raise ValueError("Forecast frame rate mismatch")

    if manifest.get("fps") != video.get("fps"):
        raise ValueError("Calibration frame rate mismatch")

    if manifest.get("resolution_px") != video.get("resolution_px"):
        raise ValueError("Calibration resolution mismatch")

    targets = protocol.get("target_frames")
    predictions = forecast.get("predictions")

    if (
        not isinstance(targets, list)
        or not targets
        or any(type(frame) is not int or frame <= cutoff
               for frame in targets)
        or targets != sorted(set(targets))
    ):
        raise ValueError("Invalid registered target frames")

    if (
        not isinstance(predictions, list)
        or [p.get("frame_index") for p in predictions] != targets
    ):
        raise ValueError("Forecast targets differ from protocol")

    policy = selection.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("Missing model-selection policy")

    if policy.get("method") != "walk_forward":
        raise ValueError("Unsupported selection method")

    if policy.get("tie_breaker") != "linear":
        raise ValueError("Unexpected tie-breaking policy")

    # Reproduce the selection decision using historical tracks ONLY.
    reproduced_selection = select_model(
        tracks,
        manifest,
        cutoff=cutoff,
        object_id=object_id,
        training_samples=policy["training_samples"],
        horizon=policy["validation_horizon_frames"],
        step=policy["origin_step_frames"],
        min_folds=policy["minimum_folds"],
        required_improvement=policy["required_relative_improvement"],
    )

    if reproduced_selection != selection:
        raise ValueError(
            "Model-selection record cannot be reproduced"
        )

    count = forecast.get("training_sample_count")

    if type(count) is not int or count < 5:
        raise ValueError("Invalid forecast training count")

    horizons = tuple(frame - cutoff for frame in targets)

    # Reproduce both candidate forecasts from the same calibration data.
    reproduced_forecast = freeze_forecast(
        tracks,
        manifest,
        cutoff=cutoff,
        object_id=object_id,
        training_samples=count,
        horizons=horizons,
    )

    if reproduced_forecast != forecast:
        raise ValueError(
            "Frozen predictions cannot be reproduced"
        )

    selected = selection.get("selected_model")

    model_fields = {
        "linear": "linear_center_px",
        "quadratic_y": "quadratic_y_center_px",
    }

    if selected not in model_fields:
        raise ValueError("Unsupported selected model")

    selected_field = model_fields[selected]

    selected_predictions = [
        {
            "frame_index": entry["frame_index"],
            "center_px": entry[selected_field],
        }
        for entry in predictions
    ]

    # Bind the complete records, rather than only selected fields.
    # This preserves the alternative model and the full policy.
    body = {
        "schema": "qapha.experiment_seal.v1",
        "stage": "calibration_only_sealed",
        "source_video_sha256": source_hash,
        "calibration_video_sha256": calibration_hash,
        "calibration_cutoff_frame": cutoff,
        "object_id": object_id,
        "selected_model": selected,
        "selected_predictions": selected_predictions,
        "selection_artifact_sha256": canonical_hash(selection),
        "forecast_artifact_sha256": canonical_hash(forecast),
        "protocol_artifact_sha256": canonical_hash(protocol),
        "calibration_manifest_artifact_sha256": canonical_hash(manifest),
        "tracking_artifact_sha256": canonical_hash(tracks),
        "selection_reproduced_exactly": True,
        "forecast_reproduced_exactly": True,
        "verification_frames_used": False,
        "limitations": (
            "This is a reproducible local record, not a digital "
            "signature or independently trusted timestamp. "
            "Video B is a retrospective selector diagnostic; "
            "use a new untouched video for prospective evaluation."
        ),
    }

    return {
        **body,
        "seal_sha256": canonical_hash(body),
    }


def main():
    parser = argparse.ArgumentParser()

    for name in (
        "selection",
        "forecast",
        "protocol",
        "calibration-manifest",
        "tracks",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True)

    args = parser.parse_args()

    names = (
        "selection",
        "forecast",
        "protocol",
        "calibration_manifest",
        "tracks",
    )

    input_paths = [
        Path(getattr(args, name)).resolve()
        for name in names
    ]

    output_path = Path(args.output).resolve()

    if len(set(input_paths + [output_path])) != 6:
        parser.error("Input and output paths must all differ")

    inputs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in input_paths
    ]

    result = seal_experiment(*inputs)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Do not overwrite an existing seal.
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
