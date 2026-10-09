"""Reproducible fixed-R sweep on recordings and synthetic sequences with known truth.

Build armor_noise_evaluation first, then run this module from the repository root.
Real-recording metrics measure future-observation consistency, not absolute accuracy.
"""
import argparse
import itertools
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['target_yaw', 'target_pitch', 'distance', 'armor_orientation_yaw']
BASELINE = np.array([.005, .005, .05, .05])


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def geometry(state, aid):
    yaw = state[6] + aid * np.pi / 2
    radius = state[8] + (state[9] if aid % 2 else 0)
    x = state[0] + radius * np.sin(yaw)
    y = state[2] + (state[10] if aid % 2 else 0)
    z = state[4] - radius * np.cos(yaw)
    return np.array([np.arctan2(x, z), np.arctan2(y, np.hypot(x, z)),
                     np.linalg.norm([x, y, z]), wrap(yaw)])


def recording(filename):
    data = pd.read_csv(ROOT / 'data' / filename)
    groups = dict(tuple(data.groupby('frame_id')))
    ids = np.array(sorted(groups))
    timestamps = np.array([groups[i].timestamp.iloc[0] / 1000 for i in ids])
    frames = []
    for fid in range(int(ids[0]), int(ids[-1]) + 1):
        z = groups[fid][FIELDS].to_numpy() if fid in groups else np.empty((0, 4))
        z = z[np.isfinite(z).all(axis=1)]
        frames.append((np.interp(fid, ids, timestamps), None, z))
    return filename.removesuffix('.csv'), False, frames


def synthetic(kind, noise_factor=1):
    rng = np.random.default_rng(20260930 + kind)
    frames = []
    for frame in range(420):
        t = frame / 30
        state = np.zeros(11)
        state[[0, 2, 4, 8, 9, 10]] = [.1, .2, 3, .26, .02, .025]
        if kind == 0:
            state[6], state[7] = 1.5 * t, 1.5
        elif kind == 1:
            u = min(t, 8.)
            state[0] += .3 * np.sin(.8 * u)
            state[4] += .25 * np.sin(.5 * u)
            state[6] = .8 * u + .8 * np.sin(1.3 * u)
            if t < 8:
                state[1], state[5], state[7] = .24 * np.cos(.8*u), .125 * np.cos(.5*u), .8 + 1.04 * np.cos(1.3*u)
        elif kind == 2:
            state[0] += .15 * np.sin(t)
            state[1], state[6], state[7] = .15 * np.cos(t), 6 * t, 6
        else:
            outbound, inbound = np.clip(t-2, 0, 3), np.clip(t-7, 0, 3)
            state[0] += .65*outbound - .55*inbound
            state[4] += .45*outbound - .45*inbound
            if 2 < t < 5:
                state[1], state[5] = .65, .45
            elif 7 < t < 10:
                state[1], state[5] = -.55, -.45
            state[6], state[7] = .6*min(t, 10), .6 if t < 10 else 0
        aid = min(range(4), key=lambda a: abs(wrap(state[6] + a * np.pi / 2)))
        z = []
        if not (100 <= frame < 111 or 250 <= frame < 254):
            for a in ([aid, (aid + 1) % 4] if frame % 3 == 0 else [aid]):
                measured = geometry(state, a) + rng.normal(size=4) * np.array([.003, .0005, .04, .10]) * noise_factor
                measured[[0, 1, 3]] = wrap(measured[[0, 1, 3]])
                if frame and frame % 83 == 0:
                    measured[2] += 1.2
                z.append(measured)
        frames.append((t, state, z))
    return f'synthetic_{kind}_{noise_factor}', True, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluator', type=Path, default=ROOT/'tests/build/Release/armor_noise_evaluation.exe')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'tests/outputs/fixed_noise_tuning')
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    sequences = [recording('camera_pose_raw_1.csv'), recording('pose_raw_2.csv')]
    sequences += [synthetic(k, factor) for factor in (1, 2) for k in range(4)]
    with (out/'sequences.txt').open('w') as f:
        f.write(f'{len(sequences)}\n')
        for name, is_synthetic, frames in sequences:
            f.write(f'{name} {int(is_synthetic)} {len(frames)}\n')
            for timestamp, truth, observations in frames:
                values = [timestamp, len(observations)]
                if is_synthetic:
                    values += truth.tolist()
                values += np.asarray(observations).flatten().tolist()
                f.write(' '.join(map(str, values)) + '\n')
    configs = {'baseline': BASELINE}
    grids = [([.005, .01, .02], [.001, .003, .01], [.04, .08, .16], [.08, .16, .32]),
             ([.02, .04, .07], [.01, .03, .07], [.08, .16, .24], [.16, .24, .32]),
             ([.03, .04, .05], [.02, .03, .04], [.20, .24, .28], [.20, .24, .28]),
             ([.02, .04, .07], [.02, .04, .07], [.32, .40, .56], [.20, .24, .28])]
    standards = sorted({std for grid in grids for std in itertools.product(*grid)})
    for index, std in enumerate(standards):
        configs[f'candidate_{index:02}'] = np.square(std)
    with (out/'candidates.txt').open('w') as f:
        for name, variances in configs.items():
            f.write(name + ' ' + ' '.join(map(str, variances)) + '\n')
    with (out/'measurements.csv').open('w') as f:
        subprocess.run([str(args.evaluator.resolve()), str((out/'sequences.txt').resolve()),
                        str((out/'candidates.txt').resolve())], stdout=f, check=True)
    metrics = pd.read_csv(out/'measurements.csv')
    baseline = metrics[metrics.candidate == 'baseline'].set_index(['sequence', 'split'])
    results = []
    for name, group in metrics.groupby('candidate', sort=False):
        row = {'candidate': name, **dict(zip(['yaw_variance', 'pitch_variance', 'distance_variance', 'plate_yaw_variance'], configs[name]))}
        for split in ('training', 'validation'):
            data = group[group.split == split].set_index(['sequence', 'split'])
            base = baseline.loc[data.index]
            real = ~data.index.get_level_values('sequence').str.startswith('synthetic')
            forecast = (data.loc[real, ['yaw_rmse', 'pitch_rmse', 'distance_rmse', 'plate_yaw_rmse']]
                        / base.loc[real, ['yaw_rmse', 'pitch_rmse', 'distance_rmse', 'plate_yaw_rmse']]).to_numpy().mean()
            truth = (data.loc[~real, ['center_truth_rmse', 'plate_truth_rmse']]
                     / base.loc[~real, ['center_truth_rmse', 'plate_truth_rmse']]).to_numpy().mean()
            jitter = (data.loc[real, ['center_step_rms', 'w_step_rms']]
                      / base.loc[real, ['center_step_rms', 'w_step_rms']]).to_numpy().mean()
            acceptance = data.loc[real, 'accepted'].sum() / base.loc[real, 'accepted'].sum()
            row[f'{split}_score'] = .45*forecast + .35*truth + .20*jitter + max(0, .98-acceptance)*5
            row[f'{split}_acceptance_ratio'] = acceptance
            row[f'{split}_forecast_ratio'] = forecast
            row[f'{split}_truth_ratio'] = truth
            row[f'{split}_jitter_ratio'] = jitter
            row[f'{split}_worst_forecast_ratio'] = (data.loc[real, ['yaw_rmse', 'pitch_rmse', 'distance_rmse', 'plate_yaw_rmse']]
                / base.loc[real, ['yaw_rmse', 'pitch_rmse', 'distance_rmse', 'plate_yaw_rmse']]).to_numpy().max()
            row[f'{split}_worst_truth_ratio'] = (data.loc[~real, ['center_truth_rmse', 'plate_truth_rmse']]
                / base.loc[~real, ['center_truth_rmse', 'plate_truth_rmse']]).to_numpy().max()
        results.append(row)
    ranking = pd.DataFrame(results).sort_values('training_score')
    ranking.to_csv(out/'ranking.csv', index=False)
    safe = ranking[(ranking.validation_score <= 1.02) & (ranking.validation_forecast_ratio <= 1.05)
                   & (ranking.validation_worst_forecast_ratio <= 1.20)
                   & (ranking.validation_worst_truth_ratio <= 1.10)
                   & (ranking.training_worst_truth_ratio <= 1.10)
                   & (ranking.validation_jitter_ratio <= 1.05)
                   & (ranking.training_acceptance_ratio >= .98) & (ranking.validation_acceptance_ratio >= .98)]
    chosen = safe.iloc[0].to_dict()
    chosen['standard_deviations'] = np.sqrt(configs[chosen['candidate']]).tolist()
    (out/'selected.json').write_text(json.dumps(chosen, indent=2), encoding='utf-8')
    print(safe.head(8).to_string(index=False))
    print('Selected:', json.dumps(chosen, indent=2))
    print('Metrics:', out/'measurements.csv')


if __name__ == '__main__':
    main()
