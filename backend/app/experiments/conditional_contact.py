
"""Freeze no-contact and hypothetical 2D rebound trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def simulate(
    tracks,
    geometry,
    *,
    object_id="1",
    cutoff=49,
    horizons=(5, 10, 21),
    train_start=42,
    edge_id=0,
    restitution=0.6,
    contact_offset_px=0.0,
    step_frames=0.05,
):
    """Hypothetical point-centre contact with a near-horizontal image edge.

    No future positions are read.
    The boundary is NOT a verified collision surface.
    """
    if (
        tracks.get("outcome") != "completed"
        or not isinstance(tracks.get("boxes"), list)
    ):
        raise ValueError("Invalid SAM track report")

    if (
        geometry.get("schema") != "qapha.scene_geometry.v1"
        or not isinstance(geometry.get("candidates"), list)
    ):
        raise ValueError("Invalid scene geometry")

    if not isinstance(object_id, str) or not object_id:
        raise ValueError("Invalid object ID")

    if (
        any(type(v) is not int for v in (
            cutoff, train_start, edge_id
        ))
        or train_start < 0
        or cutoff - train_start < 4
    ):
        raise ValueError("At least five training frames required")

    if (
        not horizons
        or any(
            type(h) is not int or h <= 0
            for h in horizons
        )
        or len(horizons) != len(set(horizons))
    ):
        raise ValueError("Invalid horizons")

    for name, value, lower, upper in (
        ("restitution", restitution, 0, 1),
        ("contact_offset_px", contact_offset_px, 0, 200),
        ("step_frames", step_frames, 0.001, 0.25),
    ):
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not lower <= value <= upper
        ):
            raise ValueError(f"Invalid {name}")

    candidates = geometry["candidates"]

    if not 0 <= edge_id < len(candidates):
        raise ValueError("Unknown geometry edge")

    endpoints = np.asarray(
        candidates[edge_id]["endpoints_px"],
        dtype=float,
    )

    if (
        endpoints.shape != (2, 2)
        or not np.isfinite(endpoints).all()
    ):
        raise ValueError("Invalid edge endpoints")

    (x1, y1), (x2, y2) = endpoints

    if abs(x2 - x1) < 1 or abs(y2 - y1) > 2:
        raise ValueError(
            "Contact candidate must be nearly horizontal"
        )

    contact_y = float(
        (y1 + y2) / 2 - contact_offset_px
    )

    xmin, xmax = sorted((float(x1), float(x2)))

    positions = {}

    for box in tracks["boxes"]:
        frame = box.get("frame_index")

        if type(frame) is not int or frame < 0:
            raise ValueError("Invalid frame index")

        # Future observations are excluded.
        if (
            frame > cutoff
            or str(box.get("object_id")) != object_id
        ):
            continue

        bounds = box.get("bbox_xyxy_px")

        if (
            not isinstance(bounds, list)
            or len(bounds) != 4
            or any(
                type(v) not in (int, float)
                or not math.isfinite(v)
                for v in bounds
            )
        ):
            raise ValueError("Invalid SAM box")

        if (
            bounds[2] <= bounds[0]
            or bounds[3] <= bounds[1]
            or frame in positions
        ):
            raise ValueError(
                "Degenerate or duplicate SAM box"
            )

        positions[frame] = (
            (bounds[0] + bounds[2]) / 2,
            (bounds[1] + bounds[3]) / 2,
        )

    frames = list(range(train_start, cutoff + 1))

    if any(frame not in positions for frame in frames):
        raise ValueError(
            "Missing consecutive training observations"
        )

    times = (
        np.asarray(frames, dtype=float) - cutoff
    )

    xy = np.asarray(
        [positions[frame] for frame in frames],
        dtype=float,
    )

    # Horizontal image motion: constant velocity.
    design_x = np.column_stack((
        np.ones_like(times),
        times,
    ))

    x0, vx = np.linalg.lstsq(
        design_x,
        xy[:, 0],
        rcond=None,
    )[0]

    # Vertical image motion: constant apparent acceleration.
    design_y = np.column_stack((
        np.ones_like(times),
        times,
        0.5 * times * times,
    ))

    y0, vy, acceleration = np.linalg.lstsq(
        design_y,
        xy[:, 1],
        rcond=None,
    )[0]

    elapsed = 0.0
    y = float(y0)
    velocity = float(vy)

    hypothetical_impacts = []
    predictions = {}
    settled = False

    # Integrate through requested horizons without reading future data.
    for target in sorted(horizons):
        while elapsed < target - 1e-10:
            dt = min(step_frames, target - elapsed)

            if settled:
                elapsed += dt
                continue

            next_y = (
                y
                + velocity * dt
                + 0.5 * acceleration * dt * dt
            )
            next_velocity = (
                velocity + acceleration * dt
            )

            next_x = x0 + vx * (elapsed + dt)

            crossed = (
                y < contact_y <= next_y
                and next_velocity > 0
                and xmin <= next_x <= xmax
            )

            if crossed:
                # Locate the crossing inside this step.
                left, right = 0.0, dt

                for _ in range(45):
                    middle = (left + right) / 2

                    middle_y = (
                        y
                        + velocity * middle
                        + 0.5 * acceleration * middle * middle
                    )

                    if middle_y < contact_y:
                        left = middle
                    else:
                        right = middle

                tau = (left + right) / 2
                hit_time = elapsed + tau
                hit_x = float(x0 + vx * hit_time)

                if not xmin <= hit_x <= xmax:
                    # Crossing occurred outside the finite segment.
                    y = next_y
                    velocity = next_velocity
                    elapsed += dt
                    continue

                impact_velocity = (
                    velocity + acceleration * tau
                )

                hypothetical_impacts.append({
                    "time_after_cutoff_frames": round(
                        hit_time, 6
                    ),
                    "image_x_px": round(hit_x, 6),
                    "impact_vertical_velocity_px_per_frame":
                        round(float(impact_velocity), 6),
                })

                remaining = dt - tau

                # Assumed rebound: this is a scenario,
                # NOT an observed collision.
                velocity = float(
                    -restitution * impact_velocity
                )
                y = contact_y

                if (
                    velocity >= 0
                    or (
                        acceleration > 0
                        and velocity * velocity
                        / (2 * acceleration) < 0.1
                    )
                ):
                    settled = True
                    velocity = 0.0
                else:
                    y = (
                        contact_y
                        + velocity * remaining
                        + 0.5 * acceleration
                        * remaining * remaining
                    )
                    velocity += acceleration * remaining

            else:
                y = next_y
                velocity = next_velocity

            elapsed += dt

        predictions[target] = [
            round(float(x0 + vx * target), 6),
            round(float(y), 6),
        ]

    training = [
        {
            "frame_index": frame,
            "center_px": list(positions[frame]),
        }
        for frame in frames
    ]

    canonical = json.dumps(
        training,
        sort_keys=True,
        separators=(",", ":"),
    )

    digest = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()

    return {
        "schema": "qapha.conditional_contact.v1",
        "object_id": object_id,
        "cutoff_frame": cutoff,
        "training_frames": [train_start, cutoff],
        "training_sha256": digest,
        "contact_assumption": {
            "edge_id": f"edge_{edge_id}",
            "contact_y_px": round(contact_y, 6),
            "contact_offset_px": contact_offset_px,
            "restitution": restitution,
            "physical_surface_verified": False,
        },
        "fitted_apparent_y_acceleration_px_per_frame2":
            round(float(acceleration), 6),
        "forecasts": [
            {
                "frame_index": cutoff + horizon,
                "no_contact_center_px": [
                    round(float(
                        x0 + vx * horizon
                    ), 6),
                    round(float(
                        y0
                        + vy * horizon
                        + 0.5 * acceleration
                        * horizon * horizon
                    ), 6),
                ],
                "conditional_contact_center_px":
                    predictions[horizon],
            }
            for horizon in sorted(horizons)
        ],
        "hypothetical_impacts": hypothetical_impacts,
        "limitations": (
            "2D image coordinates. The selected edge "
            "is not verified as a support plane. "
            "The contact offset and restitution are "
            "declared assumptions, not fitted physical facts. "
            "Hypothetical impacts are not observed collisions."
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    for name in ("tracks", "geometry", "output"):
        parser.add_argument(
            "--" + name,
            required=True,
        )

    parser.add_argument("--object-id", default="1")
    parser.add_argument("--cutoff", type=int, default=49)
    parser.add_argument(
        "--train-start", type=int, default=42
    )
    parser.add_argument(
        "--edge-id", type=int, default=0
    )
    parser.add_argument(
        "--horizons", default="5,10,21"
    )
    parser.add_argument(
        "--restitution", type=float, default=0.6
    )
    parser.add_argument(
        "--contact-offset-px",
        type=float,
        default=0.0,
    )

    args = parser.parse_args()

    paths = [
        Path(getattr(args, name)).resolve()
        for name in ("tracks", "geometry", "output")
    ]

    if len(set(paths)) != 3:
        parser.error("All paths must differ")

    try:
        horizons = tuple(
            int(value.strip())
            for value in args.horizons.split(",")
        )
    except ValueError:
        parser.error("Invalid horizons")

    result = simulate(
        json.loads(
            paths[0].read_text(encoding="utf-8")
        ),
        json.loads(
            paths[1].read_text(encoding="utf-8")
        ),
        object_id=args.object_id,
        cutoff=args.cutoff,
        train_start=args.train_start,
        edge_id=args.edge_id,
        horizons=horizons,
        restitution=args.restitution,
        contact_offset_px=args.contact_offset_px,
    )

    paths[2].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Do not overwrite a frozen experiment.
    with paths[2].open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
