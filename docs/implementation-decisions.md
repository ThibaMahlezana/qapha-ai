# Architecture decision record

- Milestone 1 builds a pure-Python, CPU-only, deterministic core with no backend server or UI.
- Internal contracts: `Observation`, `Trial`, `Prediction`, `HiddenWorld`, `Session`.
- Evidence separation: the agent obtains `Trial` observations; only the evaluator accesses `HiddenWorld.score`.
- Experiments are bounded to the speeds 2, 4 and 6 m/s and at most two tests per world.
- A held-out 5 m/s test speed evaluates prediction transfer.
- Three no-AI experiment selectors provide reproducible reference results: fixed, random, and extreme-speed diversity heuristic. The heuristic is not a statistically optimal information-gain baseline.
- Optional Muse Spark interface is isolated and must be tested with credentials and network access; don't report its performance prematurely.
- SAM 3.1 and video ingestion are **not yet implemented**. Next step is a video observation adapter with pixel calibration, timestamp alignment and hand-annotated ground truth.
- Public datasets require access and licensing verification; recordings require consent and appropriate privacy controls.
