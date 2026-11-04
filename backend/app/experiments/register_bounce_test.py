"""Register the floor-memory bounce test BEFORE any fresh clip is tracked.

Locks the code (by file hash), every setting, the frames to analyse, the
hypotheses and their decision rules, and binds them to a sealed fresh-clip
registry.  Refuses to run if any tracking/analysis artifact for the fresh
clips already exists.  Written once; never overwritten.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


SCHEMA = "qapha.bounce_test_registration.v1"
CODE = {
    "bounce_aware": "app/learning/bounce_aware.py",
    "probabilistic": "app/learning/probabilistic.py",
}
ROLLING_ROLES = ("roll_wall", "roll_ramp")


def digest(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_registration(clip_registry, code_root, analysis_dir,
                       registered_at_utc, frames=90, concept="ball",
                       segment_rule=None, exclude=None,
                       experiment_id="fresh_clips_floor_memory_v1",
                       extra_limitations=None):
    """frames=0 means the whole clip; exclude maps role -> reason."""
    exclude = exclude or {}
    body = {k: v for k, v in clip_registry.items() if k != "registry_sha256"}
    if digest(body) != clip_registry.get("registry_sha256"):
        raise ValueError("Clip registry integrity check failed")
    if clip_registry.get("viewing", {}).get("tracking_or_analysis_performed"):
        raise ValueError("Clips were already analysed")

    analysis_dir = Path(analysis_dir)
    if analysis_dir.exists() and any(analysis_dir.iterdir()):
        raise ValueError(f"Analysis folder is not empty: {analysis_dir}")

    clips = []
    for c in clip_registry["clips"]:
        clips.append({
            "clip_id": c["clip_id"],
            "file": c["file"],
            "role": c["role"],
            "sha256": c["sha256"],
            "group": ("excluded" if c["role"] in exclude
                      else "rolling_safety_check" if c["role"] in ROLLING_ROLES
                      else "bounce"),
            **({"excluded_reason": exclude[c["role"]]}
               if c["role"] in exclude else {}),
        })

    registration = {
        "schema": SCHEMA,
        "experiment_id": experiment_id,
        "registered_at_utc_self_reported": registered_at_utc,
        "clip_registry_sha256": clip_registry["registry_sha256"],
        "clips": clips,
        "locked_code_sha256": {name: file_sha256(Path(code_root) / rel)
                               for name, rel in CODE.items()},
        "settings": {
            "q": 50.0, "r": 4.0, "restitution": 0.6,
            "reset_d2": 30.0, "min_rise_px": 1.0,
        },
        "compared_variants": ["kalman", "kalman+reset", "kalman+reset+floor"],
        "also_reported": ["constant_velocity_2pt"],
        "analysis": {
            "frames_per_clip": ("all frames" if frames == 0 else
                                f"0..{frames - 1} (first {frames} frames)"),
            **({"segment_rule": segment_rule} if segment_rule else {}),
            "sam_concept": concept,
            "target_record_before_tracking": True,
            "eligibility": ("exactly one tracked object present in every "
                            "analysed frame; otherwise reported as "
                            "ineligible, never silently dropped"),
            "bounce_definition": ("bottom edge goes down then up by more "
                                  "than 1 px; frames b, b+1, b+2 scored"),
        },
        "hypotheses": {
            "H1_primary": {
                "claim": ("kalman+reset+floor has lower error than "
                          "kalman+reset at LATER bounces"),
                "metric": "mean error over frames b..b+2 of later bounces",
                "supported_if": ("lower when pooled over all eligible bounce "
                                 "clips AND lower in a majority of clips "
                                 "that have at least one later bounce"),
                "clips_without_later_bounce": "reported, excluded from H1",
            },
            "H2": {
                "claim": "floor memory lowers the overall surprise score",
                "metric": "mean NLL over all predictions",
                "supported_if": "lower pooled over all eligible clips",
            },
            "H3_safety": {
                "claim": "floor memory does not harm rolling clips",
                "metric": "mean error on rolling_safety_check clips",
                "supported_if": ("kalman+reset+floor within 5% of "
                                 "kalman+reset on each rolling clip"),
            },
        },
        "reporting": {
            "report_every_clip_and_variant": True,
            "report_unfavourable_results": True,
            "no_parameter_tuning_after_tracking": True,
        },
        "known_limitations": [
            "Clips were recompressed by WhatsApp (reduced resolution).",
            "The clip owner watched each clip once to label it; the method "
            "developer has not viewed any frame.",
            "Development data (videos C, E and a synthetic ball) shaped the "
            "method; this test is its first independent check.",
            "No two-ball or stationary clip in this set; Video D served as "
            "the stationary control during development.",
            "Pixel errors are in image space; scale differs between clips.",
            *(extra_limitations or []),
        ],
    }
    return {**registration, "registration_sha256": digest(registration)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-registry", required=True)
    parser.add_argument("--analysis-dir", required=True,
                        help="folder that will hold tracks/results; must be empty")
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--concept", default="ball")
    parser.add_argument("--segment-rule", default=None,
                        choices=[None, "longest_gap_free_run_min_30"])
    parser.add_argument("--exclude", action="append", default=[],
                        help="role=reason, e.g. roll_wall=analysed in v1")
    parser.add_argument("--experiment-id",
                        default="fresh_clips_floor_memory_v1")
    parser.add_argument("--limitation", action="append", default=[])
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    registry = json.loads(Path(args.clip_registry).read_text(encoding="utf-8"))
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    exclude = dict(item.split("=", 1) for item in args.exclude)
    reg = build_registration(registry, Path(__file__).resolve().parents[2],
                             args.analysis_dir, now.isoformat(),
                             args.frames, args.concept, args.segment_rule,
                             exclude, args.experiment_id, args.limitation)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(reg, stream, indent=2)
    for c in reg["clips"]:
        print(f"{c['clip_id']}  {c['role']:<18} group={c['group']}")
    print("locked code:", reg["locked_code_sha256"])
    print("registration_sha256:", reg["registration_sha256"])
    print("Saved:", output)


if __name__ == "__main__":
    main()