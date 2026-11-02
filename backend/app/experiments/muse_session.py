"""Run a bounded Muse-driven experiment session against the offline hidden world.

From backend/: python -m app.experiments.muse_session
Only measured observations are supplied to the model; never ground truth.
"""
import json
from dataclasses import asdict
from typing import Protocol

from app.physics.core import HiddenWorld, Session
from app.providers.muse_agent import MuseExperimentSelector


class ExperimentSelector(Protocol):
    def propose(self, allowed: tuple[float, ...], history: list[dict]) -> tuple[float, str]: ...


def public_history(session: Session) -> list[dict]:
    """Expose only measured observations; hidden parameters remain in HiddenWorld."""
    return [
        {
            "speed_mps": trial.speed_mps,
            "observations": [asdict(o) for o in trial.observations],
        }
        for trial in session.trials
    ]


def run_session(selector: ExperimentSelector, session: Session, held_out_speed: float = 5.0) -> dict:
    """Select/execute allowed experiments and score a forecast after freezing it."""
    decisions = []
    while len(session.trials) < session.budget:
        allowed = tuple(float(s) for s in session.allowed)
        speed, reason = selector.propose(allowed, public_history(session))
        # Session.experiment independently enforces allowed speeds and the budget.
        trial = session.experiment(speed)
        decisions.append({
            "speed_mps": speed,
            "reason": reason,
            "observation_count": len(trial.observations),
        })

    forecast = session.forecast(held_out_speed)
    frozen_forecast = asdict(forecast)
    evaluation = session.world.score(forecast)
    return {"experiments": decisions, "frozen_forecast": frozen_forecast, "evaluation": evaluation}


def main() -> None:
    session = Session(HiddenWorld(mu=0.23, k=0.34, seed=17))
    result = run_session(MuseExperimentSelector(), session)
    print(json.dumps(result, indent=2))
    print("NOTE: This is one synthetic world, not a comparative AI benchmark.")


if __name__ == "__main__":
    main()
