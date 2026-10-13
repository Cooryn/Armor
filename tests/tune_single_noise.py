"""Tune single-plate EKF on full native video observations (no external truth).

First build/run single_video_evaluation for each video to observations_1/2.csv.
Run: python -m tests.tune_single_noise --work-dir tests/outputs/single-noise
Scores use the first 70%; the final 30% is reported separately.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tests.reference.predictor.predictor import SinglePlateEKF


def video_segments(data):
    # Diagnostic clipping only: production single-plate tracking never resets
    # on these heuristics. Keep only contiguous, small-motion video fragments.
    yaw_step = np.diff(data.yaw.to_numpy())
    orientation = np.diff(data.orientation.to_numpy())
    return np.cumsum(np.r_[True, ~np.isfinite(yaw_step) | ~np.isfinite(orientation)
                          | (np.abs(yaw_step) > .035) | (np.abs(orientation) > .6)])


def replay(data, q, angle_sd, range_sd, segments=False, q_yaw=100.):
    model = SinglePlateEKF()
    model.process_noise = q
    model.yaw_process_noise = q_yaw
    model.R = np.diag([angle_sd**2, angle_sd**2, range_sd**2, .0576])
    states = []
    previous = None
    ids = video_segments(data) if segments else np.zeros(len(data))
    for index, row in enumerate(data.itertuples()):
        observation = np.array([row.yaw, row.pitch, row.distance, row.orientation])
        if segments and index and ids[index] != ids[index - 1]:
            model.is_initialized = False
        if model.is_initialized:
            model.predict((row.timestamp - previous) / 1000)
            if np.isfinite(observation).all():
                model.update(observation)
        elif np.isfinite(observation).all():
            model.initialize(observation)
        previous = row.timestamp
        states.append(model.state[:6].copy() if model.is_initialized else np.full(6, np.nan))
    return np.array(states)


def metrics(data, states, start, end):
    t = data.timestamp.to_numpy() / 1000
    yaw, pitch, distance = data[['yaw', 'pitch', 'distance']].to_numpy().T
    observed = np.column_stack([distance * np.cos(pitch) * np.sin(yaw),
                                distance * np.sin(pitch), distance * np.cos(pitch) * np.cos(yaw)])
    # Compare 50 ms forecasts with linearly interpolated future observations.
    # Both bracketing frames must have detections; never bridge a missing detection.
    j = np.searchsorted(t, t + .05)
    valid = (j < len(t)) & (np.arange(len(t)) >= start) & (j < end) & (t > 1)
    j = np.minimum(j, len(t) - 1)
    alpha = ((t + .05 - t[j - 1]) / (t[j] - t[j - 1]))[:, None]
    truth = observed[j - 1] * (1 - alpha) + observed[j] * alpha
    future = states[:, ::2] + .05 * states[:, 1::2]
    valid &= np.isfinite(truth).all(axis=1) & np.isfinite(future).all(axis=1)
    delta = future[valid] - truth[valid]
    segment = video_segments(data)
    continuous = valid & (segment == segment[j])
    future_yaw = np.arctan2(future[:, 0], future[:, 2])
    target_yaw = np.arctan2(truth[:, 0], truth[:, 2])
    angular = np.arctan2(np.sin(future_yaw - target_yaw), np.cos(future_yaw - target_yaw))
    current = states[:, ::2]
    present = valid & np.isfinite(observed).all(axis=1)
    lag_errors = []
    for lag in range(16):
        ids = np.arange(max(start, lag, 30), end)
        error = current[ids] - observed[ids - lag]
        lag_errors.append(np.nanmean(np.sum(error**2, axis=1)))
    acceleration = np.diff(current, n=2, axis=0)[max(start, 30):max(start, end - 2)]
    return dict(samples=int(valid.sum()),
                continuous_samples=int(continuous.sum()),
                continuous_future_rmse_m=float(np.sqrt(np.mean(np.sum((future[continuous] - truth[continuous])**2, axis=1)))),
                continuous_yaw_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(angular[continuous]**2)))),
                future_position_rmse_m=float(np.sqrt(np.mean(np.sum(delta**2, axis=1)))),
                future_yaw_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(angular[valid]**2)))),
                current_position_rmse_m=float(np.sqrt(np.mean(np.sum((current[present] - observed[present])**2, axis=1)))),
                lag_ms=int(np.argmin(lag_errors)) * 1000 / 30,
                second_difference_rms_m=float(np.sqrt(np.nanmean(np.sum(acceleration**2, axis=1)))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, default=Path('tests/outputs/single-noise'))
    parser.add_argument('--segments', action='store_true', help='Replay isolated continuous single-plate fragments')
    args = parser.parse_args()
    data = [pd.read_csv(args.work_dir / f'observations_{i}.csv') for i in (1, 2)]
    baseline = []
    for d in data:
        states = replay(d, .01, .04, .4)
        if args.segments:
            states = replay(d, .01, .04, .4, True)
        baseline.append([metrics(d, states, 0, int(len(d) * .7)), metrics(d, states, int(len(d) * .7), len(d))])
    rows = []
    candidates = [(.01, .04, .4)] + [(q, a, r) for q in (1., 10., 100., 1000.)
                                                    for a in (.002, .005, .01) for r in (.03, .05, .1)]
    for q, a, r in candidates:
        row = dict(q=q, angle_sd=a, range_sd=r)
        score = []
        for i, d in enumerate(data):
            states = replay(d, q, a, r, args.segments)
            for part, (start, end) in enumerate(((0, int(len(d) * .7)), (int(len(d) * .7), len(d)))):
                values = metrics(d, states, start, end)
                row.update({f'v{i+1}_{part}_{k}': v for k, v in values.items()})
                if part == 0:
                    score.append(.5 * values['continuous_future_rmse_m'] / baseline[i][part]['continuous_future_rmse_m']
                                 + .5 * values['current_position_rmse_m'] / baseline[i][part]['current_position_rmse_m'])
        row['score'] = float(np.mean(score))
        rows.append(row)
    ranking = pd.DataFrame(rows).sort_values('score')
    ranking.to_csv(args.work_dir / ('ranking_segments.csv' if args.segments else 'ranking.csv'), index=False)
    print(ranking[['q', 'angle_sd', 'range_sd', 'score']].head(12).to_string(index=False))
    (args.work_dir / 'baseline_metrics.json').write_text(json.dumps(baseline, indent=2))


if __name__ == '__main__':
    main()
