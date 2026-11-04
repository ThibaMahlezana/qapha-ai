"""Write a sealed addendum to the Video E preregistration.

The addendum must be created BEFORE any Video E tracking exists.  It:
  * verifies the original registration's integrity (never edits it);
  * re-verifies the four inherited learner-module hashes;
  * locks the hashes of the policy and reference-baseline code;
  * adds descriptive secondary comparisons without changing the
    primary hypothesis, metric, thresholds or duration.

Limitation: a hash plus a self-reported timestamp does not prove when
the addendum was written.  Commit it to git immediately for an
external, timestamped record.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


SCHEMA = "qapha.registration_addendum.v1"

INHERITED = {
    "adaptive_loop": "app/learning/adaptive_loop.py",
    "adaptive_session": "app/learning/adaptive_session.py",
    "prediction_protocol": "app/learning/prediction_protocol.py",
    "reliability_monitor": "app/learning/reliability_monitor.py",
}
LOCKED = {
    "warning_policy": "app/learning/warning_policy.py",
    "reference_baselines": "app/learning/reference_baselines.py",
}
# Any of these existing means Video E has already been examined.
FORBIDDEN_PATTERNS = (
    "heldout_video_e_*tracks*.json",
    "heldout_video_e_*evaluation*.json",
    "heldout_video_e_*preview*",
    "heldout_video_e_*annotated*",
)


def digest(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_addendum(original, code_root, recorded_dir, registered_at_utc):
    body = {k: v for k, v in original.items() if k != "registration_sha256"}
    if digest(body) != original.get("registration_sha256"):
        raise ValueError("Original registration integrity check failed")
    if original.get("experiment_id") != "video_e_uncertainty_response":
        raise ValueError("Unexpected experiment")

    recorded_dir = Path(recorded_dir)
    seen = sorted(p.name for pattern in FORBIDDEN_PATTERNS
                  for p in recorded_dir.glob(pattern))
    if seen:
        raise ValueError(f"Video E artifacts already exist: {seen}")

    code_root = Path(code_root)
    expected = original["inherited_implementation_sha256"]
    for name, rel in INHERITED.items():
        if file_sha256(code_root / rel) != expected[name]:
            raise ValueError(f"Inherited module changed: {name}")

    video = recorded_dir / original["video"]
    addendum = {
        "schema": SCHEMA,
        "experiment_id": original["experiment_id"],
        "amends_registration_sha256": original["registration_sha256"],
        "registered_at_utc_self_reported": registered_at_utc,
        "status": "registered_before_video_e_tracking",
        "source_video_present_at_registration": video.exists(),
        "source_video_sha256": (file_sha256(video)
                                if video.exists() else None),
        "unchanged_from_original": [
            "primary_metric", "baseline", "experimental_policy",
            "informativeness", "tracking_requirements",
            "frame_range_inclusive", "prediction_targets_inclusive",
        ],
        "locked_implementation_sha256": {
            name: file_sha256(code_root / rel)
            for name, rel in LOCKED.items()
        },
        "secondary_comparisons": {
            "role": "descriptive; does not alter the primary decision",
            "models": {
                "fixed_persistence": "adaptive_loop persistence expert",
                "fixed_linear_window6": "adaptive_loop linear expert",
                "constant_velocity_2pt": "x_t = 2 x_(t-1) - x_(t-2)",
                "alpha_beta": {"alpha": 0.85, "beta": 0.5, "dt_frames": 1},
            },
            "metrics": [
                "mean_euclidean_error_px",
                "maximum_euclidean_error_px",
                "count_errors_over_25px",
            ],
            "prediction_targets_inclusive": [2, 49],
        },
        "exploratory_expectations": {
            "basis": "post-hoc review of Video C only",
            "E1": ("constant_velocity_2pt mean error is lower than "
                   "original_adaptive_selector mean error"),
            "E2": ("warning_triggered_persistence mean error is lower "
                   "than original_adaptive_selector mean error"),
            "note": "Expectations are recorded so that they can fail.",
        },
        "reporting": {
            "report_all_models_regardless_of_outcome": True,
            "non_informative_video_reported_as_such": True,
            "no_parameter_tuning": True,
        },
        "limitations": [
            "Self-reported timestamp; git commit provides external record.",
            "Secondary baselines were first examined on Video C.",
            "One video cannot establish generalization.",
        ],
    }
    return {**addendum, "registration_sha256": digest(addendum)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    original_path = Path(args.original).resolve()
    output = Path(args.output).resolve()
    if output == original_path:
        parser.error("output cannot overwrite the original registration")

    original = json.loads(original_path.read_text(encoding="utf-8"))
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    addendum = build_addendum(
        original,
        code_root=Path(__file__).resolve().parents[2],
        recorded_dir=original_path.parent,
        registered_at_utc=now.isoformat(),
    )
    with output.open("x", encoding="utf-8") as stream:
        json.dump(addendum, stream, indent=2)
    print(json.dumps({
        "registration_sha256": addendum["registration_sha256"],
        "amends": addendum["amends_registration_sha256"],
        "source_video_present": addendum["source_video_present_at_registration"],
        "locked_implementation_sha256":
            addendum["locked_implementation_sha256"],
    }, indent=2))
    print("Saved:", output)


if __name__ == "__main__":
    main()
    