import math
import unittest
from app.physics.core import HiddenWorld, Session, trajectory, stopping_distance, fit_parameters, benchmark

class CoreTests(unittest.TestCase):
    def test_zero_drag_analytic(self):
        self.assertAlmostEqual(stopping_distance(4,.2,0),16/(2*.2*9.81),places=6)
    def test_motion_nonnegative_and_stopped(self):
        xs=trajectory(4,.2,.3,[i*.1 for i in range(200)])
        self.assertTrue(all(a<=b+1e-10 for a,b in zip(xs,xs[1:])))
        self.assertAlmostEqual(xs[-1],xs[-2],places=6)
    def test_hidden_parameter_estimation(self):
        w=HiddenWorld(.23,.34,seed=17,sigma_m=0)
        trials=[w.observe(2),w.observe(6)]
        mu,k,_=fit_parameters(trials)
        self.assertLess(abs(mu-.23),.03)
        self.assertLess(abs(k-.34),.08)
    def test_policy_and_budget(self):
        s=Session(HiddenWorld(.2,.3))
        with self.assertRaises(ValueError): s.experiment(5)
        s.experiment(2);s.experiment(6)
        with self.assertRaises(ValueError): s.experiment(4)
    def test_benchmark_reproducible(self):
        self.assertEqual(benchmark(3,42),benchmark(3,42))

if __name__=='__main__': unittest.main()
