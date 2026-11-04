import copy
import unittest
from app.learning.adaptive_loop import AdaptiveTracker, replay

class AdaptiveLoopTests(unittest.TestCase):
    def test_forecast_precedes_observation(self):
        tracker = AdaptiveTracker()
        tracker.step(0, (0, 0))
        tracker.step(1, (2, 0))
        event = tracker.step(2, (100, 0))
        self.assertEqual(event['history_cutoff_frame'], 1)
        self.assertEqual(event['predictions_xy']['linear'], [4.0, 0.0])
        self.assertEqual(event['selection'], 'persistence')
        self.assertEqual(event['scores_before_observation'], {'persistence': None, 'linear': None})
    def test_future_changes_do_not_rewrite_prior_predictions(self):
        a=AdaptiveTracker(); b=AdaptiveTracker()
        for i, point in enumerate([(0,0),(1,0),(2,0),(3,0)]):
            a.step(i,point);b.step(i,point)
        before=copy.deepcopy(a.events)
        a.step(4, (999,999));b.step(4,(4,0))
        self.assertEqual(a.events[:-1],before)
        self.assertEqual(a.events[-1]['predictions_xy'],b.events[-1]['predictions_xy'])
        self.assertEqual(a.events[-1]['selection'],b.events[-1]['selection'])
    def test_rejects_gaps_and_invalid_points(self):
        tracker=AdaptiveTracker(); tracker.step(0,(0,0))
        with self.assertRaises(ValueError):tracker.step(2,(2,0))
        with self.assertRaises(ValueError):tracker.step(1,(float('nan'),0))
        tracker.step(1,(1,0))
        with self.assertRaises(ValueError):tracker.step(1,(1,0))
    def test_adapts_after_velocity_change(self):
        tracker=AdaptiveTracker(window=3,decay=0.5)
        for frame,x in enumerate([0,10,20,30,40,40,40,40,40,40]):
            tracker.step(frame,(x,0))
        self.assertEqual(tracker.events[-1]['selection'],'persistence')
    def test_replay_requires_single_object_and_consecutive_frames(self):
        data={'model':'sam-3.1','boxes':[
            {'object_id':'0','frame_index':i,'bbox_xyxy_px':[i,0,i+10,10]}
            for i in range(4)]}
        self.assertEqual(replay(data)['evaluated_predictions'],2)
        broken=copy.deepcopy(data);broken['boxes'][2]['frame_index']=5
        with self.assertRaises(ValueError):replay(broken)
        broken=copy.deepcopy(data);broken['boxes'][2]['object_id']='1'
        with self.assertRaises(ValueError):replay(broken)

if __name__=='__main__': unittest.main()
