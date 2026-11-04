"""Leakage-safe online predictor comparison on SAM box centres."""
from __future__ import annotations
import argparse
import json
import math
from collections import deque
from pathlib import Path


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class AdaptiveTracker:
    """One-step prequential test: forecast, score, then learn."""

    def __init__(self, window=6, decay=0.85):
        if type(window) is not int or window < 2:
            raise ValueError('window must be >= 2')
        if type(decay) not in (float, int) or not 0 < decay < 1:
            raise ValueError('decay must be strictly between 0 and 1')
        self.window = window
        self.decay = decay
        self.history = deque(maxlen=window)
        self.loss = {'persistence': None, 'linear': None}
        self.totals = {'persistence': 0.0, 'linear': 0.0, 'adaptive': 0.0}
        self.count = 0
        self.events = []

    def _forecast(self, frame):
        if len(self.history) < 2:
            return None
        last_frame, last_xy = self.history[-1]
        if frame != last_frame + 1:
            raise ValueError('frames must be consecutive')
        old_frame, old_xy = self.history[0]
        dt = last_frame - old_frame
        velocity = ((last_xy[0] - old_xy[0]) / dt,
                    (last_xy[1] - old_xy[1]) / dt)
        return {
            'persistence': last_xy,
            'linear': (last_xy[0] + velocity[0], last_xy[1] + velocity[1]),
        }

    def step(self, frame, point):
        """Forecast frame using prior history; only then reveal point."""
        if type(frame) is not int or frame < 0:
            raise ValueError('invalid frame')
        if (not isinstance(point, (list, tuple)) or len(point) != 2
                or any(type(x) not in (int, float) or not math.isfinite(x)
                       for x in point)):
            raise ValueError('invalid point')
        point = (float(point[0]), float(point[1]))
        if self.history and frame != self.history[-1][0] + 1:
            raise ValueError('frames must be consecutive')
        if not self.history and frame != 0:
            raise ValueError('stream must start at frame 0')

        forecasts = self._forecast(frame)
        event = None
        if forecasts is not None:
            # Selection uses scores from PREVIOUS frames only.
            chosen = min(('persistence', 'linear'),
                         key=lambda name: (float('inf') if self.loss[name] is None
                                           else self.loss[name],
                                           0 if name == 'persistence' else 1))
            scores_before = dict(self.loss)
            errors = {name: distance(pos, point)
                      for name, pos in forecasts.items()}
            adaptive_error = errors[chosen]
            for name in ('persistence', 'linear'):
                previous = self.loss[name]
                self.loss[name] = (errors[name] if previous is None else
                                   self.decay * previous +
                                   (1 - self.decay) * errors[name])
                self.totals[name] += errors[name]
            self.totals['adaptive'] += adaptive_error
            self.count += 1
            event = {
                'frame_index': frame,
                'history_cutoff_frame': frame - 1,
                'selection': chosen,
                'scores_before_observation': scores_before,
                'predictions_xy': {name: [round(v, 4) for v in xy]
                                   for name, xy in forecasts.items()},
                'observed_xy': list(point),
                'errors_px': {**{name: round(value, 6)
                                 for name, value in errors.items()},
                              'adaptive': round(adaptive_error, 6)},
                'scores_after_observation': dict(self.loss),
            }
            self.events.append(event)
        self.history.append((frame, point))
        return event

    def report(self):
        if not self.count:
            raise ValueError('at least three observations required')
        return {
            'schema': 'qapha.adaptive_replay.v1',
            'method': 'prequential_one_step_expert_selection',
            'selection_rule': 'lowest previously observed EWMA pixel error',
            'window_frames': self.window,
            'decay': self.decay,
            'evaluated_predictions': self.count,
            'mean_error_px': {name: round(total / self.count, 6)
                              for name, total in self.totals.items()},
            'selection_counts': {name: sum(e['selection'] == name
                                           for e in self.events)
                                 for name in ('persistence', 'linear')},
            'events': self.events,
            'limitations': [
                'Predicts only one frame ahead in image-pixel coordinates.',
                'Online expert selection is not a learned physics model.',
                'Video C has already been examined; this is diagnostic replay.',
                'Tracker errors and abrupt changes affect observed loss.',
            ],
        }


def replay(track_data, window=6, decay=0.85):
    if track_data.get('model') != 'sam-3.1':
        raise ValueError('unexpected tracker model')
    boxes = track_data.get('boxes')
    if not isinstance(boxes, list) or not boxes:
        raise ValueError('missing boxes')
    ids = {str(box.get('object_id')) for box in boxes}
    if len(ids) != 1:
        raise ValueError('expected one tracked object')
    tracker = AdaptiveTracker(window=window, decay=decay)
    for box in boxes:
        bounds = box.get('bbox_xyxy_px')
        if (not isinstance(bounds, list) or len(bounds) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v)
                       for v in bounds)
                or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]):
            raise ValueError('invalid bounding box')
        xy = ((bounds[0] + bounds[2]) / 2,
              (bounds[1] + bounds[3]) / 2)
        tracker.step(box.get('frame_index'), xy)
    result = tracker.report()
    result['object_id'] = next(iter(ids))
    result['input_frames'] = len(boxes)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracks', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, target = Path(args.tracks).resolve(), Path(args.output).resolve()
    if source == target:
        parser.error('output cannot overwrite input')
    data = json.loads(source.read_text(encoding='utf-8'))
    report = replay(data)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({key: report[key] for key in
                      ('schema', 'input_frames', 'evaluated_predictions',
                       'mean_error_px', 'selection_counts')}, indent=2))
    print('Saved:', target)


if __name__ == '__main__':
    main()
