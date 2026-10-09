"""Compare the direct C++ model interfaces with NumPy on complete video frame clocks."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--python-root', type=Path, required=True)
parser.add_argument('--bin-dir', type=Path, required=True, help='Directory containing video_predictor_tests')
parser.add_argument('--work-dir', type=Path, required=True)
a = parser.parse_args()
sys.path.insert(0, str(a.python_root.resolve()))
from tests.reference.predictor.video_predictor import run_video_predictor

root = a.work_dir.resolve()
root.mkdir(parents=True, exist_ok=True)
exe = a.bin_dir.resolve() / ('video_predictor_tests.exe' if sys.platform == 'win32' else 'video_predictor_tests')
report = []


def compare(name, frames, model, horizon=50):
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    fixture = folder / 'frames.txt'
    with fixture.open('w') as stream:
        for fid, timestamp, observations in frames:
            stream.write(f'{fid} {timestamp:.17g} {len(observations)}\n')
            for p, z in observations:
                stream.write(' '.join(f'{value:.17g}' for value in np.r_[p, z]) + '\n')
    py, cpp = folder / 'python', folder / 'cpp'
    run_video_predictor(frames, model, py, horizon)
    subprocess.run([str(exe), '--replay', model, str(fixture), str(cpp), str(horizon)], check=True, capture_output=True)
    pf, cf = {p.name for p in py.iterdir()}, {p.name for p in cpp.iterdir()}
    assert pf == cf, (name, pf, cf)
    maximum, count = 0., 0
    for file in sorted(pf):
        if file.endswith('.txt'):
            assert (py / file).read_text() == (cpp / file).read_text(), (name, file)
            continue
        x, y = pd.read_csv(py / file), pd.read_csv(cpp / file)
        assert list(x.columns) == list(y.columns) and x.shape == y.shape, (name, file)
        count += len(x)
        for column in x:
            if pd.api.types.is_numeric_dtype(x[column]) and not pd.api.types.is_bool_dtype(x[column]):
                xx, yy = x[column].to_numpy(float), y[column].to_numpy(float)
                np.testing.assert_allclose(xx, yy, rtol=1e-7, atol=1e-8, equal_nan=True,
                                           err_msg=f'{name}/{file}/{column}')
                finite = np.isfinite(xx) & np.isfinite(yy)
                if finite.any():
                    maximum = max(maximum, float(np.max(np.abs(xx[finite] - yy[finite]))))
            else:
                pd.testing.assert_series_equal(x[column], y[column], check_dtype=False)
    report.append(dict(case=name, files=len(pf), csv_rows=count, max_absolute_difference=maximum))
    print(name, 'PASS', maximum, flush=True)
    return cpp


def observation(x, y, z, yaw=0):
    p = np.array([x, y, z], float)
    return p, np.array([np.arctan2(x, z), np.arctan2(y, np.hypot(x, z)), np.linalg.norm(p), yaw])


for suffix in ('1', '2'):
    source = a.python_root / 'data' / f'pose_raw_{suffix}.csv'
    if not source.exists():
        raise FileNotFoundError(f'Generate video {suffix} observations with Armor before running this comparison')
    data = pd.read_csv(source)
    if 'coordinate_frame' in data:
        assert data.coordinate_frame.eq('camera').all()
    cap = cv2.VideoCapture(str(a.python_root / 'assets/video' / f'video_{suffix}.avi'))
    fps, size = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    groups = {int(fid): [(r[['x', 'y', 'z']].to_numpy(float),
                         r[['target_yaw', 'target_pitch', 'distance', 'armor_orientation_yaw']].to_numpy(float))
                        for _, r in g.iterrows()] for fid, g in data.groupby('frame_id')}
    frames = [(i, i * 1000 / fps, groups.get(i, [])) for i in range(size)]
    for model in ('basic', 'polar', 'armor'):
        compare(f'real_{suffix}_{model}', frames, model)

synthetic = []
time = 0.
for i in range(100):
    time += (20, 33, 55)[i % 3]
    obs = []
    if 2 <= i < 92 and i not in range(20, 29) and i not in (44, 45):
        for aid in ([0, 1, 2, 3, 1] if i % 3 == 0 else [i % 4]):
            theta = 2.8 + i * .12 + aid * np.pi / 2
            o = observation(.1 + i * .002 + .26 * np.sin(theta), .2 + .015 * (aid % 2),
                            3 - .26 * np.cos(theta), (theta + np.pi) % (2 * np.pi) - np.pi)
            if i == 12: o[1][2] += 3
            if len(obs) == 4:
                o[0][:] *= 1.04
                o[1][2] *= 1.04
            obs.append(o)
    if i == 0: obs = [observation(0, 0, 0)]
    synthetic.append((i, time, obs))

for model in ('basic', 'polar', 'armor'):
    for horizon in (0, 50, 200):
        frames = [(i, t, [] if i == 0 else obs) for i, t, obs in synthetic] if model == 'armor' else synthetic
        output = compare(f'synthetic_{model}_{horizon}', frames, model, horizon)
        prefix = {'basic': '', 'polar': 'polar_', 'armor': 'armor_'}[model]
        state = pd.read_csv(output / f'{prefix}prediction_result_1.csv')
        assert state.frame_id.tolist() == list(range(3, 100))
        if model == 'armor':
            missed = state[state.status == 'prediction_only']
            assert not missed.empty and missed.armor_id.eq(-1).all()
            assert missed[['xa', 'za', 'pred_armor_yaw', 'obs_armor_yaw', 'err_distance']].isna().all().all()
            assert state.accepted_count.max() == 4
            futures = pd.read_csv(output / 'armor_future_prediction_1.csv')
            assert (futures.groupby('frame_id').armor_id.nunique() == 4).all()

selection = [(0, 0., [observation(0, 0, 4), observation(1, 0, 3), observation(-1, 0, 3)]),
             (1, 40., [observation(1, 0, 3), observation(-1, 0, 3)]), (2, 80., []),
             (3, 120., [observation(0, 0, 0)]), (4, 160., [])]
for model in ('basic', 'polar', 'armor'):
    frames = [(i, t, [] if i == 3 else obs) for i, t, obs in selection] if model == 'armor' else selection
    compare(f'selection_{model}', frames, model)
    for name, frames in [('empty', []), ('no_observations', [(i, i * 40., []) for i in range(6)]),
                         ('single_frame', [(0, 0., [observation(0, .2, 3)])])]:
        compare(f'{name}_{model}', frames, model)
wrap = [(i, i * 30., [observation(.01 - .002 * i, .2, -3)]) for i in range(20)]
for model in ('basic', 'polar', 'armor'):
    compare(f'target_wrap_{model}', wrap, model)

(root / 'verification.json').write_text(json.dumps(report, indent=2))
print(f'{len(report)} online comparison cases passed', flush=True)
