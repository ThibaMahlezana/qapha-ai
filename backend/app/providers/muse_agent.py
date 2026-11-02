"""Bounded Meta Muse Spark experiment selector.

Run from qapha-ai/backend:
    python -m app.providers.muse_agent

Never send hidden world parameters or held-out outcomes to this adapter.
"""
import json
import math
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API_URL = "https://api.meta.ai/v1/chat/completions"
MODEL = "muse-spark-1.3"


def validate_selection(payload: object, allowed: tuple[float, ...]) -> tuple[float, str]:
    """Accept a single permitted numeric speed and a short explanation."""
    if not isinstance(payload, dict):
        raise ValueError("Response must be a JSON object")
    if set(payload) != {"speed_mps", "reason"}:
        raise ValueError("Response must contain only speed_mps and reason")
    speed = payload["speed_mps"]
    if type(speed) not in (int, float) or not math.isfinite(speed):
        raise ValueError("speed_mps must be a finite number, not a boolean or string")
    if speed not in allowed:
        raise ValueError("Experiment speed is not permitted")
    reason = payload["reason"]
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 500:
        raise ValueError("reason must be a non-empty string of at most 500 characters")
    return float(speed), reason.strip()


class MuseExperimentSelector:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("MODEL_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("MODEL_API_KEY is not set")

    def propose(self, allowed: tuple[float, ...], history: list[dict]) -> tuple[float, str]:
        if not allowed or any(type(x) not in (int, float) or not math.isfinite(x) for x in allowed):
            raise ValueError("Allowed speeds must be finite numbers")
        if len(set(allowed)) != len(allowed):
            raise ValueError("Allowed speeds must be unique")
        if not isinstance(history, list) or len(history) > 2:
            raise ValueError("History must contain at most two completed experiments")
        # Caller must provide only observations, never simulator ground truth.
        prompt = (
            "Choose ONE experiment to help distinguish unknown friction and linear drag. "
            "You do not know the true physical parameters. Only use the supplied history. "
            "Return ONLY valid JSON with exactly two keys: "
            '{"speed_mps": 4, "reason": "brief explanation"}. '
            f"Allowed speeds (m/s): {json.dumps(allowed)}. "
            f"Completed observations: {json.dumps(history, allow_nan=False)}"
        )
        request = Request(
            API_URL,
            data=json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}]}).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=60) as response:
                data = json.load(response)
        except HTTPError as exc:
            raise RuntimeError(f"Meta API HTTP {exc.code}") from None
        except (URLError, TimeoutError, OSError):
            raise RuntimeError("Meta API connection failed") from None
        try:
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError("Meta returned an invalid or non-JSON selection") from exc
        return validate_selection(parsed, allowed)

    def select(self, allowed: tuple[float, ...], history: list[dict]) -> float:
        """Compatibility method returning only the validated speed."""
        speed, _ = self.propose(allowed, history)
        return speed


def main() -> int:
    try:
        speed, reason = MuseExperimentSelector().propose((2.0, 4.0, 6.0), [])
    except (ValueError, RuntimeError) as exc:
        print(f"Selection test failed: {exc}")
        return 1
    print("SUCCESS: Muse Spark selected an authorised experiment.")
    print(f"Selected speed: {speed} m/s")
    print(f"Reason: {reason}")
    print("No simulation was executed. No hidden parameters were supplied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
