"""Evaluate the registered floor-memory bounce test on the fresh clips.

Applies exactly the rules sealed in the bounce-test registration:
integrity of registration, intake manifest and locked code; eligibility
(one tracked object in every analysed frame, registered SAM concept);
then H1 (later-bounce error), H2 (overall NLL) and H3 (rolling clips not
harmed).  Every clip is reported, eligible or not.  Written once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from app.learning.bounce_aware import compare
from app.learning.reference_baselines import constant_velocity_2pt

SCHEMA = "qapha.bounce_test_evaluation.v1"
CODE_ROOT = Path(__file__).resolve().parents[2]
CODE = {"bounce_aware": "app/learning/bounce_aware.py",
        "probabilistic": "app/learning/probabilistic.py"}
FLOOR, BASE = "kalman+reset+floor", "kalman+reset"


def digest(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sealed(data, key):
    return digest({k: v for k, v in data.items() if k != key}) == data.get(key)


def verify(registration, manifest, code_root=CODE_ROOT):
    if not _sealed(registration, "registration_sha256"):
        raise ValueError("Registration integrity check failed")
    if not _sealed(manifest, "manifest_sha256"):
        raise ValueError("Intake manifest integrity check failed")
    if manifest["registration_sha256"] != registration["registration_sha256"]:
        raise ValueError("Intake manifest belongs to another registration")
    changed = [n for n, rel in CODE.items()
               if file_sha256(Path(code_root) / rel)
               != registration["locked_code_sha256"][n]]
    if changed:
        raise ValueError(f"Locked code has changed: {changed}")


def frames_registered(registration):
    text = registration["analysis"]["frames_per_clip"]
    if text == "all frames":
        return None
    return int(text.split("(first ")[1].split(" ")[0])


MIN_RUN = 30


def longest_run(tracks, concept):
    """v2 rule: most-tracked object, longest gap-free run of >= 30 frames."""
    if tracks is None:
        return False, "not tracked", None, None
    if tracks.get("model") != "sam-3.1":
        return False, "unexpected tracker", None, None
    if tracks.get("concept") != concept:
        return False, f"tracked with concept {tracks.get('concept')!r}", None, None
    per_id = {}
    for b in tracks.get("boxes", []):
        per_id.setdefault(str(b.get("object_id")), {})[b["frame_index"]] = \
            b["bbox_xyxy_px"]
    if not per_id:
        return False, "0 tracked objects", None, None
    oid, frames = max(per_id.items(), key=lambda kv: (len(kv[1]), kv[0]))
    best_start, best_len, start, prev = None, 0, None, None
    for f in sorted(frames):
        if prev is None or f != prev + 1:
            start = f
        length = f - start + 1
        if length > best_len:
            best_start, best_len = start, length
        prev = f
    segment = {"object_id": oid, "objects_tracked": len(per_id),
               "first_frame": best_start,
               "last_frame": best_start + best_len - 1,
               "frames": best_len}
    if best_len < MIN_RUN:
        return False, f"longest tracked run {best_len} < {MIN_RUN} frames", \
            None, segment
    boxes = [frames[f] for f in range(best_start, best_start + best_len)]
    return True, "eligible", boxes, segment


def eligibility(tracks, frames, concept):
    if tracks is None:
        return False, "not tracked", None
    if tracks.get("model") != "sam-3.1":
        return False, "unexpected tracker", None
    if tracks.get("concept") != concept:
        return False, f"tracked with concept {tracks.get('concept')!r}", None
    boxes = tracks.get("boxes", [])
    ids = {str(b.get("object_id")) for b in boxes}
    if len(ids) != 1:
        return False, f"{len(ids)} tracked objects", None
    by_frame = {}
    for b in boxes:
        if b["frame_index"] in by_frame:
            return False, "duplicate frames", None
        by_frame[b["frame_index"]] = b["bbox_xyxy_px"]
    missing = [f for f in range(frames) if f not in by_frame]
    if missing:
        return False, f"{len(missing)} frames without the ball", None
    return True, "eligible", [by_frame[f] for f in range(frames)]


def evaluate_clip(boxes):
    results = compare(boxes)
    centres = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
    cv_errors = [math.hypot(p[0] - centres[t][0], p[1] - centres[t][1])
                 for t, p in constant_velocity_2pt(centres)]
    variants = {}
    for name, s in results.items():
        later = [b for b in s["bounces"] if b["order"] == "later"]
        variants[name] = {
            "evaluated_predictions": s["evaluated_predictions"],
            "mean_error_px": s["mean_error_px"],
            "mean_nll": s["mean_nll"],
            "inside90_approx": s["inside90_approx"],
            "bounces": s["bounces"],
            "later_bounce_errors": [b["mean_error_px"] for b in later],
        }
    variants["constant_velocity_2pt"] = {
        "evaluated_predictions": len(cv_errors),
        "mean_error_px": round(sum(cv_errors) / len(cv_errors), 6),
    }
    return variants


def evaluate(registration, manifest, analysis_dir, code_root=CODE_ROOT):
    verify(registration, manifest, code_root)
    frames = frames_registered(registration)
    concept = registration["analysis"]["sam_concept"]
    analysis_dir = Path(analysis_dir)
    clips = []
    for c in registration["clips"]:
        path = analysis_dir / f"{c['role']}_tracks.json"
        tracks = (json.loads(path.read_text(encoding="utf-8"))
                  if path.is_file() else None)
        entry = {"clip_id": c["clip_id"], "role": c["role"],
                 "group": c["group"],
                 "tracks_sha256": file_sha256(path) if path.is_file() else None}
        if c["group"] == "excluded":
            clips.append({**entry, "eligible": False,
                          "reason": "excluded: " + c.get("excluded_reason", "")})
            continue
        if registration["analysis"].get("segment_rule"):
            ok, reason, boxes, segment = longest_run(tracks, concept)
            entry["segment"] = segment
        else:
            ok, reason, boxes = eligibility(tracks, frames, concept)
        entry.update({"eligible": ok, "reason": reason})
        if ok:
            entry["variants"] = evaluate_clip(boxes)
        clips.append(entry)

    eligible = [c for c in clips if c["eligible"]]
    bounce = [c for c in eligible if c["group"] == "bounce"]
    with_later = [c for c in bounce
                  if c["variants"][FLOOR]["later_bounce_errors"]]

    def pooled(clist, variant):
        errs = [e for c in clist
                for e in c["variants"][variant]["later_bounce_errors"]]
        return round(sum(errs) / len(errs), 6) if errs else None

    if with_later:
        per_clip = {c["role"]: (
            sum(c["variants"][FLOOR]["later_bounce_errors"])
            < sum(c["variants"][BASE]["later_bounce_errors"]))
            for c in with_later}
        pf, pb = pooled(with_later, FLOOR), pooled(with_later, BASE)
        wins = sum(per_clip.values())
        h1 = {"testable": True,
              "pooled_later_bounce_error_px": {FLOOR: pf, BASE: pb},
              "clips_improved": f"{wins} of {len(per_clip)}",
              "per_clip_improved": per_clip,
              "supported": pf < pb and wins > len(per_clip) / 2}
    else:
        h1 = {"testable": False,
              "reason": "no eligible bounce clip has a later bounce"}

    if eligible:
        def mean_nll(variant):
            total = sum(c["variants"][variant]["mean_nll"]
                        * c["variants"][variant]["evaluated_predictions"]
                        for c in eligible)
            count = sum(c["variants"][variant]["evaluated_predictions"]
                        for c in eligible)
            return round(total / count, 6)
        nf, nb = mean_nll(FLOOR), mean_nll(BASE)
        h2 = {"testable": True, "pooled_mean_nll": {FLOOR: nf, BASE: nb},
              "supported": nf < nb}
    else:
        h2 = {"testable": False, "reason": "no eligible clips"}

    rolling = [c for c in eligible if c["group"] == "rolling_safety_check"]
    if rolling:
        checks = {c["role"]: {
            FLOOR: c["variants"][FLOOR]["mean_error_px"],
            BASE: c["variants"][BASE]["mean_error_px"],
            "within_5_percent": (c["variants"][FLOOR]["mean_error_px"]
                                 <= 1.05 * c["variants"][BASE]["mean_error_px"]),
        } for c in rolling}
        h3 = {"testable": True, "per_clip": checks,
              "supported": all(v["within_5_percent"] for v in checks.values())}
    else:
        h3 = {"testable": False, "reason": "no eligible rolling clips"}

    report = {
        "schema": SCHEMA,
        "registration_sha256": registration["registration_sha256"],
        "intake_manifest_sha256": manifest["manifest_sha256"],
        "eligible_clips": f"{len(eligible)} of {len(clips)}",
        "hypotheses": {"H1_primary": h1, "H2": h2, "H3_safety": h3},
        "clips": clips,
        "limitations": registration.get("known_limitations", []),
    }
    return {**report, "report_sha256": digest(report)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", required=True)
    parser.add_argument("--analysis-dir", required=True)
    args = parser.parse_args()
    analysis = Path(args.analysis_dir)
    output = analysis / "bounce_test_report.json"
    if output.exists():
        raise FileExistsError(output)
    load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
    report = evaluate(load(args.registration),
                      load(analysis / "intake_manifest.json"), analysis)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print("Eligible clips:", report["eligible_clips"])
    for c in report["clips"]:
        line = f"  {c['clip_id']} {c['role']:<16} {c['reason']}"
        if c.get("segment"):
            seg = c["segment"]
            line += f" [frames {seg['first_frame']}-{seg['last_frame']}]"
        if c["eligible"]:
            v = c["variants"]
            line += (f" | error  kalman={v['kalman']['mean_error_px']:.2f} "
                     f"reset={v[BASE]['mean_error_px']:.2f} "
                     f"floor={v[FLOOR]['mean_error_px']:.2f} "
                     f"cv2pt={v['constant_velocity_2pt']['mean_error_px']:.2f} "
                     f"| bounces={[b['bounce_frame'] for b in v[FLOOR]['bounces']]}")
        print(line)
    print(json.dumps(report["hypotheses"], indent=2))
    print("report_sha256:", report["report_sha256"])
    print("Saved:", output)


if __name__ == "__main__":
    main()