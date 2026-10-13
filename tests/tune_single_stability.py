"""Evaluate native single-plate Q/R candidates at the production 100 ms horizon.

Real camera recordings have no external position/yaw truth. Their step metrics
include target motion; synthetic stationary/moving sequences check noise vs lag.
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
BASELINE = (100., 100., .000025, .000025, .01, .0576)


def trajectory(t, kind):
    t = np.asarray(t)
    if kind == 'static':
        return np.column_stack([t * 0 + .2, t * 0 + .1, t * 0 + 3, t * 0 + .2])
    if kind == 'near':
        return np.column_stack([.08 * np.sin(2*np.pi*.65*t) + .03*np.sin(2*np.pi*1.1*t),
                                .04 + .02*np.sin(2*np.pi*.4*t), .35 + .03*np.sin(2*np.pi*.35*t),
                                .25*np.sin(2*np.pi*.5*t)])
    if kind == 'depth':
        return np.column_stack([t*0+.2, t*0+.1, 3+.3*np.sin(2*np.pi*.65*t),
                                .25*np.sin(2*np.pi*.5*t)])
    if kind == 'slow':
        return np.column_stack([.3*np.sin(2*np.pi*.25*t), .1+.05*np.sin(2*np.pi*.2*t),
                                3+.2*np.sin(2*np.pi*.2*t), .4*np.sin(2*np.pi*.25*t)])
    return np.column_stack([.35*np.sin(2*np.pi*.65*t)+.12*np.sin(2*np.pi*1.1*t),
                            .1+.08*np.sin(2*np.pi*.4*t), 3+.2*np.sin(2*np.pi*.35*t),
                            .4*np.sin(2*np.pi*.5*t)])


def synthetic(seed):
    sequences = []
    for fps, kind in itertools.product((30, 60), ('static', 'slow', 'fast', 'near', 'depth', 'dropout')):
        rng = np.random.default_rng(seed + fps + len(sequences) * 71)
        t = np.arange(12 * fps) / fps
        truth = trajectory(t, kind)
        x, y, z, orientation = truth.T
        obs = np.column_stack([np.arctan2(x, z), np.arctan2(y, np.hypot(x, z)),
                               np.linalg.norm(truth[:, :3], axis=1), orientation])
        sd = [.003, .001, .035, .08] if kind != 'near' else [.001, .0004, .004, .04]
        obs += rng.normal(size=obs.shape) * sd
        obs[:, 3] = (obs[:, 3] + np.pi) % (2*np.pi) - np.pi
        observations = [[row] for row in obs]
        if kind == 'dropout':
            for i in range(len(t)):
                if i % 10 == 4 or 5. <= t[i] < 5.2:
                    observations[i] = []
        sequences.append((f'synth_{kind}_{fps}', True, t*1000, truth, observations))
    return sequences


def recording(name, path):
    data = pd.read_csv(path)
    groups = {int(fid): group for fid, group in data.groupby('frame_id')}
    ids = np.array(sorted(groups))
    full = np.arange(ids[0], ids[-1] + 1)
    # Missing frame timestamps are interpolated for this diagnostic replay only.
    times = np.interp(full, ids, [groups[i].timestamp.iloc[0] for i in ids])
    observations = [groups[i][FIELDS].to_numpy().tolist() if i in groups else [] for i in full]
    return name, False, times, None, observations


def write_sequences(path, sequences):
    with path.open('w') as stream:
        stream.write(f'{len(sequences)}\n')
        for name, has_truth, times, truth, observations in sequences:
            stream.write(f'{name} {int(has_truth)} {len(times)}\n')
            for i, time in enumerate(times):
                values = [time, len(observations[i])]
                if has_truth:
                    values += truth[i].tolist()
                stream.write(' '.join(f'{v:.17g}' for v in values) + '\n')
                for observation in observations[i]:
                    stream.write(' '.join(f'{v:.17g}' for v in observation) + '\n')


def evaluate(exe, sequences, candidates, destination, details=None):
    command = [str(exe), str(sequences), str(candidates)]
    if details is not None:
        command.append(str(details))
    with destination.open('w') as stream:
        subprocess.run(command, stdout=stream, check=True)
    return pd.read_csv(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, default=ROOT / 'tests/outputs/single-stability-20261012')
    parser.add_argument('--exe', type=Path, default=ROOT / 'build/Release/single_noise_evaluation.exe')
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    records = [recording('camera_still_1', ROOT / 'tests/outputs/camera-fps-20261012/live-full-single/results/camera/single/offline/pose_raw_live.csv'),
               recording('camera_still_2', ROOT / 'tests/outputs/camera-fps-20261012/live-full-armor/results/camera/armor/offline/pose_raw_live.csv'),
               recording('camera_motion', ROOT / 'results/camera/single/offline/pose_raw_live.csv')]
    write_sequences(work / 'training.txt', synthetic(20261012) + records)
    write_sequences(work / 'validation.txt', synthetic(20261023))
    write_sequences(work / 'holdout.txt', synthetic(20261103))
    parameters = {'baseline': BASELINE}
    for q_pos, q_yaw, angle_sd, range_sd in itertools.product((30., 50., 60., 75., 100.), (30., 50., 75., 100.), (.005, .006, .008, .01), (.1, .15)):
        name = f'c{len(parameters):03}'
        parameters[name] = (q_pos, q_yaw, angle_sd**2, .000025, range_sd**2, .0576)
    with (work / 'candidates.txt').open('w') as stream:
        for name, values in parameters.items():
            stream.write(name + ' ' + ' '.join(f'{v:.17g}' for v in values) + '\n')
    metrics = evaluate(args.exe, work/'training.txt', work/'candidates.txt', work/'metrics.csv')
    base = metrics[metrics.candidate == 'baseline'].set_index('sequence')
    rankings = []
    for name, group in metrics.groupby('candidate', sort=False):
        group = group.set_index('sequence')
        still = ['synth_static_30', 'synth_static_60']
        motion = [key for key in group.index if key.startswith('synth') and key not in still]
        ratio = group.select_dtypes('number') / base.select_dtypes('number')
        # Motion protection prevents selection solely for a smoother picture.
        feasible = (group.loc[motion, 'current_rmse_m'].max() < .06 and
                    group.loc[motion, 'future_rmse_m'].max() < .125 and
                    ratio.loc[motion, 'future_rmse_m'].max() < 1.20 and
                    ratio.loc[motion, 'future_yaw_rmse_rad'].max() < 1.20)
        static_ratio = ratio.loc[still, 'future_rmse_m'].mean()
        yaw_ratio = ratio.loc[still, 'future_yaw_rmse_rad'].mean()
        motion_ratio = ratio.loc[motion, 'future_rmse_m'].mean()
        camera_step = ratio.loc[['camera_still_1', 'camera_still_2'], 'future_step_rms_m'].mean()
        score = .35*static_ratio + .25*yaw_ratio + .25*motion_ratio + .15*camera_step
        rankings.append(dict(candidate=name, feasible=feasible, score=score, static_ratio=static_ratio,
                             yaw_ratio=yaw_ratio, motion_ratio=motion_ratio, camera_step_ratio=camera_step,
                             q_pos=parameters[name][0], q_yaw=parameters[name][1],
                             angle_sd=np.sqrt(parameters[name][2]), range_sd=np.sqrt(parameters[name][4])))
    ranking = pd.DataFrame(rankings).sort_values(['feasible', 'score'], ascending=[False, True])
    ranking.to_csv(work/'ranking.csv', index=False)
    print(ranking.head(12).to_string(index=False), flush=True)
    if not ranking.feasible.any():
        raise RuntimeError('No candidate passed motion protection; keep production parameters')
    shortlist = ranking[ranking.feasible].head(12).candidate.tolist()
    with (work/'shortlist_candidates.txt').open('w') as stream:
        for name in ['baseline'] + shortlist:
            stream.write(name + ' ' + ' '.join(f'{v:.17g}' for v in parameters[name]) + '\n')
    validation = evaluate(args.exe, work/'validation.txt', work/'shortlist_candidates.txt', work/'validation.csv')
    base_validation = validation[validation.candidate == 'baseline'].set_index('sequence').select_dtypes('number')
    selected = None
    for name in shortlist:
        group = validation[validation.candidate == name].set_index('sequence').select_dtypes('number')
        motion = [key for key in group.index if 'static' not in key]
        ratio = group / base_validation
        if (ratio.loc[motion, 'future_rmse_m'].max() <= 1.20 and
                ratio.loc[motion, 'future_yaw_rmse_rad'].max() <= 1.20 and
                group.loc[motion, 'current_rmse_m'].max() < .06):
            selected = name
            break
    if selected is None:
        raise RuntimeError('No candidate passed validation; keep production parameters')
    with (work/'selected_candidates.txt').open('w') as stream:
        for name in ('baseline', selected):
            stream.write(name + ' ' + ' '.join(f'{v:.17g}' for v in parameters[name]) + '\n')
    evaluate(args.exe, work/'training.txt', work/'selected_candidates.txt', work/'comparison.csv', work/'states.csv')
    evaluate(args.exe, work/'holdout.txt', work/'selected_candidates.txt', work/'holdout.csv', work/'holdout_states.csv')
    (work/'selected.json').write_text(json.dumps(dict(candidate=selected, parameters=parameters[selected],
                                                   horizon_ms=100, training_seed=20261012,
                                                   validation_seed=20261023, holdout_seed=20261103), indent=2))
    print('Selected', selected, parameters[selected], flush=True)


if __name__ == '__main__':
    main()
