"""Small matched-world benchmark of live Muse experiment selection.

Run from backend/: python -m app.experiments.muse_benchmark --worlds 3
Requires MODEL_API_KEY. Each world uses 2 Muse API calls.
Ground-truth parameters are never passed to the selector.
"""
import argparse
import json
import random
import statistics
from app.experiments.muse_session import run_session
from app.physics.core import HiddenWorld, Session, fixed_selector, random_selector, diversity_selector


class BaselineSelector:
    def __init__(self, strategy: str, seed: int):
        self.strategy = strategy
        self.rng = random.Random(seed)

    def propose(self, allowed, history):
        speeds = [trial['speed_mps'] for trial in history]
        if self.strategy == 'fixed':
            speed = fixed_selector(allowed, speeds)
        elif self.strategy == 'random':
            speed = random_selector(allowed, speeds, self.rng)
        elif self.strategy == 'diverse':
            speed = diversity_selector(allowed, speeds)
        else:
            raise ValueError('Unsupported baseline')
        return float(speed), f'{self.strategy} baseline'


def compare(selectors: dict, worlds: list[tuple[float, float]], *, seed: int = 42) -> dict:
    """Compare selectors on the same worlds and observation RNG seeds.

    Each selector gets a NEW HiddenWorld for each run, preventing state leakage.
    """
    if not worlds:
        raise ValueError('At least one world is required')
    results = {name: {'errors_m': [], 'selections_mps': []} for name in selectors}
    for idx, (mu, k) in enumerate(worlds):
        for name, selector_factory in selectors.items():
            selector = selector_factory(idx)
            session = Session(HiddenWorld(mu, k, seed=seed + idx))
            outcome = run_session(selector, session, held_out_speed=5.)
            results[name]['errors_m'].append(outcome['evaluation']['absolute_error_m'])
            results[name]['selections_mps'].append([x['speed_mps'] for x in outcome['experiments']])
    return {
        name: {
            'worlds': len(worlds),
            'mean_absolute_error_m': statistics.mean(data['errors_m']),
            'errors_m': data['errors_m'],
            'selections_mps': data['selections_mps'],
        }
        for name, data in results.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worlds', type=int, default=3, help='Unseen worlds; 2 Muse API calls each')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.worlds <= 10:
        parser.error('--worlds must be between 1 and 10 to control API use')
    # Import only when running the live benchmark, so offline tests need no credentials.
    from app.providers.muse_agent import MuseExperimentSelector
    rng = random.Random(args.seed)
    worlds = [(rng.uniform(.08, .5), rng.uniform(.05, .8)) for _ in range(args.worlds)]
    selectors = {
        'fixed': lambda i: BaselineSelector('fixed', seed=args.seed + i),
        'random': lambda i: BaselineSelector('random', seed=args.seed + i),
        'diverse': lambda i: BaselineSelector('diverse', seed=args.seed + i),
        'muse': lambda i: MuseExperimentSelector(),
    }
    result = compare(selectors, worlds, seed=args.seed)
    print(json.dumps({'seed': args.seed, 'results': result}, indent=2))
    print('NOTE: Small exploratory synthetic benchmark. No general accuracy claim.')


if __name__ == '__main__':
    main()
