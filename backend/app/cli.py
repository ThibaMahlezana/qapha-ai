import argparse
import json
from dataclasses import asdict
from app.physics.core import HiddenWorld, Session, benchmark

def main():
    p=argparse.ArgumentParser(description='Qapha Foresight offline scientific prototype')
    p.add_argument('--benchmark',action='store_true')
    p.add_argument('--worlds',type=int,default=20)
    args=p.parse_args()
    if args.benchmark:
        print(json.dumps(benchmark(args.worlds),indent=2))
        return
    world=HiddenWorld(mu=.23,k=.34,seed=17)
    session=Session(world)
    for speed in (6.,2.):
        trial=session.experiment(speed)
        print(f'Calibration at {speed} m/s: {len(trial.observations)} noisy positions')
    forecast=session.forecast(5.)
    print('Frozen forecast:',json.dumps(asdict(forecast),indent=2))
    print('Revealed evaluation:',json.dumps(world.score(forecast),indent=2))

if __name__=='__main__': main()
