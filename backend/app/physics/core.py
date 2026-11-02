"""Deterministic hidden-world experiment; agent never receives the true parameters."""
from dataclasses import dataclass, asdict
import math
import random
from typing import Sequence

G = 9.81

@dataclass(frozen=True)
class Observation:
    time_s: float
    position_m: float
    sigma_m: float

@dataclass(frozen=True)
class Trial:
    speed_mps: float
    observations: tuple[Observation, ...]

@dataclass(frozen=True)
class Prediction:
    speed_mps: float
    stopping_distance_m: float
    estimated_mu: float
    estimated_k: float
    experiment_count: int


def trajectory(v0: float, mu: float, k: float, times: Sequence[float]):
    if not (v0 > 0 and mu > 0 and k >= 0):
        raise ValueError('v0 and mu must be positive; k must be nonnegative')
    a = mu * G
    if k < 1e-9:
        t_stop = v0 / a
        return tuple(v0 * min(t, t_stop) - .5 * a * min(t, t_stop)**2 for t in times)
    t_stop = math.log1p(k * v0/a) / k
    out=[]
    for t in times:
        t=min(t,t_stop)
        out.append(((v0+a/k)*(1-math.exp(-k*t))/k)-(a*t/k))
    return tuple(out)


def stopping_distance(v0: float, mu: float, k: float) -> float:
    a = mu * G
    if k < 1e-9:
        return v0*v0/(2*a)
    t = math.log1p(k*v0/a)/k
    return trajectory(v0,mu,k,(t,))[0]


class HiddenWorld:
    """Ground truth stays private; observations are the only public evidence."""
    def __init__(self, mu: float, k: float, seed: int = 0, sigma_m: float = .03):
        if not (.08 <= mu <= .5 and .05 <= k <= .8):
            raise ValueError('parameters outside benchmark bounds')
        self._mu, self._k = mu, k
        self._rng = random.Random(seed)
        self.sigma_m = sigma_m

    def observe(self, speed_mps: float, duration_s: float = 4., step_s: float = .25) -> Trial:
        if not (1 <= speed_mps <= 7):
            raise ValueError('speed outside experiment boundary')
        times = [round(i*step_s,8) for i in range(int(duration_s/step_s)+1)]
        ideal = trajectory(speed_mps,self._mu,self._k,times)
        return Trial(speed_mps,tuple(Observation(t, x+self._rng.gauss(0,self.sigma_m),self.sigma_m) for t,x in zip(times,ideal)))

    def score(self, prediction: Prediction) -> dict:
        true_d = stopping_distance(prediction.speed_mps,self._mu,self._k)
        return {'actual_stopping_distance_m':true_d, 'absolute_error_m':abs(prediction.stopping_distance_m-true_d)}


def fit_parameters(trials: Sequence[Trial]) -> tuple[float,float,float]:
    """Coarse-to-fine bounded least-squares, pure standard-library implementation.

    Returns mu,k,mean squared position error. The prior midpoint is used before any observations.
    """
    if not trials:
        return .29,.425,float('nan')
    def loss(mu,k):
        err=[]
        for trial in trials:
            obs=trial.observations
            predicted=trajectory(trial.speed_mps,mu,k,[o.time_s for o in obs])
            err.extend((o.position_m-p)**2 for o,p in zip(obs,predicted))
        return sum(err)/len(err)
    best=(float('inf'),.29,.425)
    for mu_steps,k_steps,mu_radius,k_radius in [(21,21,.21,.375),(15,15,.06,.12),(11,11,.012,.025)]:
        for i in range(mu_steps):
            mu=max(.08,min(.5,best[1]+(2*i/(mu_steps-1)-1)*mu_radius))
            for j in range(k_steps):
                k=max(.05,min(.8,best[2]+(2*j/(k_steps-1)-1)*k_radius))
                v=loss(mu,k)
                if v < best[0]: best=(v,mu,k)
    return best[1],best[2],best[0]


class Session:
    def __init__(self, world: HiddenWorld, allowed=(2.,4.,6.), budget=2):
        self.world=world
        self.allowed=tuple(allowed)
        self.budget=budget
        self.trials: list[Trial]=[]

    def experiment(self, speed: float) -> Trial:
        if speed not in self.allowed:
            raise ValueError('experiment not authorised')
        if len(self.trials) >= self.budget:
            raise ValueError('experiment budget exhausted')
        trial=self.world.observe(speed)
        self.trials.append(trial)
        return trial

    def forecast(self, speed: float) -> Prediction:
        if not (1 <= speed <=7): raise ValueError('held-out speed outside benchmark')
        mu,k,_=fit_parameters(self.trials)
        return Prediction(speed,stopping_distance(speed,mu,k),mu,k,len(self.trials))


def fixed_selector(allowed, history):
    return allowed[len(history)%len(allowed)]


def random_selector(allowed,history,rng):
    return rng.choice(allowed)


def diversity_selector(allowed,history):
    """Simple non-AI baseline; choose extreme speeds to diversify observations."""
    order=sorted(allowed)
    preference=[order[-1],order[0]]+[x for x in order if x not in (order[0],order[-1])]
    for x in preference:
        if x not in history: return x
    return preference[0]


def benchmark(world_count=20,seed=42):
    """Matched hidden worlds across three selectors. Honest baseline, not a Meta AI result."""
    rng=random.Random(seed)
    environments=[(rng.uniform(.08,.5),rng.uniform(.05,.8)) for _ in range(world_count)]
    results={}
    for name in ('fixed','random','diverse'):
        errors=[]
        for i,(mu,k) in enumerate(environments):
            session=Session(HiddenWorld(mu,k,seed=i+seed))
            choose_rng=random.Random(seed+i)
            for _ in range(session.budget):
                history=[t.speed_mps for t in session.trials]
                speed=(fixed_selector(session.allowed,history) if name=='fixed' else
                       random_selector(session.allowed,history,choose_rng) if name=='random' else
                       diversity_selector(session.allowed,history))
                session.experiment(speed)
            pred=session.forecast(5.)
            errors.append(session.world.score(pred)['absolute_error_m'])
        results[name]={'mean_absolute_error_m':sum(errors)/len(errors),'worlds':len(errors)}
    return results
