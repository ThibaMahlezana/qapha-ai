# Qapha AI — Foresight Milestone 1 (aligned with your folder structure)

This project is structured under `qapha-ai/backend/app/` exactly as agreed. The offline scientific core is implemented; video perception, context routing, UI, and live Meta validation are **not** yet implemented. Existing folders without implementation are kept as placeholders.

## Where the previous starter files went

| Old archive location | New location |
|---|---|
| `backend/qapha/core.py` | `backend/app/physics/core.py` |
| `backend/qapha/cli.py` | `backend/app/cli.py` |
| `backend/qapha/meta_agent.py` | `backend/app/providers/muse_agent.py` |
| `backend/tests/test_core.py` | `backend/tests/unit/test_core.py` |

## Run on Windows PowerShell

Open a terminal at the `qapha-ai` root:

```powershell
cd backend
python -m unittest discover -s tests/unit -v
python -m app.cli
python -m app.cli --benchmark --worlds 50
```

Python 3.10+ required. Offline simulation runs using the standard library only. Optional live Muse adapter uses the `openai` package and `MODEL_API_KEY`, and has **not** been live verified.

## Scope and status

- Implemented: hidden friction/drag simulator, numerical fitting, permitted experiment budget, fixed/random/diverse baselines, frozen forecast evaluation, offline unit tests.
- Not implemented: actual Meta experiment loop, SAM video tracking, world memory, scene recognition, reasoning router, FastAPI service, Angular frontend.
- The Meta adapter file exists as a placeholder implementation only and is not called by the offline benchmark.

## Import rule

When you add files, use absolute Python imports from `app`, e.g. `from app.physics.core import Session`. Execute Python commands from `backend/`.

## Merging with an existing folder

Unzip the archive into a temporary location and copy the **contents** of its `qapha-ai` folder into your existing `qapha-ai` folder, merging directories. Do not create a second nested `qapha-ai/qapha-ai/` folder. The existing frontend source is not replaced by this package (the archive contains only empty placeholder directories there).
