"""Ground-truth check for one physical plate, independent of rotating-video switches."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tests.tune_single_noise import replay


def trajectory(t, moving):
    if not moving:
        return np.column_stack([np.full_like(t, .2), np.full_like(t, .1), np.full_like(t, 3.)])
    return np.column_stack([.35 * np.sin(2 * np.pi * .65 * t) + .12 * np.sin(2 * np.pi * 1.1 * t),
                            .1 + .08 * np.sin(2 * np.pi * .4 * t),
                            3 + .2 * np.sin(2 * np.pi * .35 * t)])


def evaluate(q, angle_sd, range_sd):
    rows = []
    for fps in (30, 60):
        for moving in (False, True):
            t = np.arange(12 * fps) / fps
            truth = trajectory(t, moving)
            rng = np.random.default_rng(20261010 + fps + moving)
            observations = np.column_stack([np.arctan2(truth[:, 0], truth[:, 2]),
                                            np.arctan2(truth[:, 1], np.hypot(truth[:, 0], truth[:, 2])),
                                            np.linalg.norm(truth, axis=1)])
            # Conservative video-fragment second-difference scale, not a sensor calibration.
            observations += rng.normal(size=observations.shape) * [.003, .001, .035]
            data = pd.DataFrame(observations, columns=['yaw', 'pitch', 'distance'])
            data['timestamp'] = t * 1000
            states = replay(data, q, angle_sd, range_sd)
            valid = t >= 1
            current = states[:, ::2]
            future = current + .05 * states[:, 1::2]
            errors = current[valid] - truth[valid]
            future_errors = future[valid] - trajectory(t[valid] + .05, moving)
            lags = np.arange(0, .501, .001)
            lag = lags[np.argmin([np.mean((current[valid, 0] - trajectory(t[valid] - lag, moving)[:, 0])**2) for lag in lags])]
            rows.append(dict(fps=fps, moving=moving, current_rmse_m=float(np.sqrt(np.mean(np.sum(errors**2, axis=1)))),
                             future_rmse_m=float(np.sqrt(np.mean(np.sum(future_errors**2, axis=1)))),
                             x_lag_ms=float(lag * 1000) if moving else None))
    return rows


def main():
    work = Path('tests/outputs/single-noise')
    baseline = evaluate(.01, .04, .4)
    results = []
    for q in (1., 10., 100., 1000.):
        for a in (.002, .005, .01):
            for r in (.03, .05, .1):
                values = evaluate(q, a, r)
                score = np.mean([v['future_rmse_m'] / b['future_rmse_m'] for v, b in zip(values, baseline) if v['moving']])
                static = max(v['current_rmse_m'] for v in values if not v['moving'])
                results.append(dict(q=q, angle_sd=a, range_sd=r, score=score, static_rmse_m=static))
    ranking = pd.DataFrame(results).sort_values('score')
    ranking.to_csv(work / 'response_ranking.csv', index=False)
    (work / 'response.json').write_text(json.dumps(dict(baseline=baseline, selected=evaluate(100, .005, .1)), indent=2))
    print(ranking.head(12).to_string(index=False))
    print((work / 'response.json').read_text())


if __name__ == '__main__':
    main()
