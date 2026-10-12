"""Software-only camera -> PnP -> fixed-frame EKF -> absolute command regression."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--work-dir', type=Path, required=True)
parser.add_argument('--filter-bin-dir', type=Path, default=Path(__file__).parent / 'build/Release')
a = parser.parse_args()
a.work_dir.mkdir(parents=True, exist_ok=True)
exe = a.filter_bin_dir.resolve() / ('pipeline_tests.exe' if sys.platform == 'win32' else 'pipeline_tests')
env = {k.upper(): v for k, v in os.environ.items()}
env['PATH'] = str(Path(sys.base_prefix) / 'Library/bin') + os.pathsep + env.get('PATH', '')
result = subprocess.run([str(exe), str(a.work_dir.resolve())], env=env, capture_output=True, text=True,
                        encoding='utf-8', errors='replace')
(a.work_dir / 'native.log').write_text(result.stdout + result.stderr)
if result.returncode:
    raise RuntimeError(result.stdout + result.stderr)
report = []
for model in range(2):
    model_name = ('single', 'armor')[model]
    directory = a.work_dir / str(model) / 'results/video' / model_name / 'fixture'
    raw = pd.read_csv(directory / 'pose_raw_fixture.csv')
    base = pd.read_csv(directory / 'pose_base_fixture.csv')
    poses = pd.read_csv(directory / 'camera_pose_fixture.csv')
    commands = pd.read_csv(directory / 'control_target_fixture.csv')
    assert raw.coordinate_frame.eq('camera').all() and base.coordinate_frame.eq('base').all()
    assert len(poses) == len(commands) == 12
    expected_pose_times = np.maximum(poses.timestamp.to_numpy() - 5, 0)
    for frame in (2, 6):
        expected_pose_times[frame] = expected_pose_times[frame - 1]
    np.testing.assert_allclose(poses.pose_timestamp_ms, expected_pose_times, atol=1e-7)
    # Frames without new feedback reuse the previous transform and retain its actual timestamp.
    transform_columns = ['qw', 'qx', 'qy', 'qz', 'x', 'y', 'z']
    for frame in (2, 6):
        np.testing.assert_allclose(poses.loc[frame, transform_columns].to_numpy(dtype=float),
                                   poses.loc[frame - 1, transform_columns].to_numpy(dtype=float), atol=0)
    assert commands.frame_id.tolist() == list(range(12))
    assert commands[commands.frame_id.between(5, 7)].valid.eq(1).all()
    assert commands.valid.eq(1).any(), 'controller never produced a valid absolute target'
    assert np.isfinite(commands[['yaw_rad', 'pitch_rad']].to_numpy()).all()
    np.testing.assert_allclose(base[['x', 'y', 'z']].to_numpy(), np.tile([.1, .2, 3], (len(base), 1)), atol=.003)
    assert base.base_reference_timestamp_ms.eq(0).all() and poses.base_reference_timestamp_ms.eq(0).all()
    origin_columns = ['base_origin_x_m', 'base_origin_y_m', 'base_origin_z_m']
    np.testing.assert_allclose(base[origin_columns].to_numpy(), 0, atol=0)
    np.testing.assert_allclose(poses[origin_columns].to_numpy(), 0, atol=0)
    np.testing.assert_allclose(base.base_reference_yaw_rad, .35, atol=1e-7)
    np.testing.assert_allclose(poses.base_reference_yaw_rad, .35, atol=1e-7)
    # The camera starts above horizontal with a real mounting offset from the yaw-axis origin.
    assert abs(poses.iloc[0].x - .13) < 1e-7
    assert abs(poses.iloc[0].y - (.01 - .015 * np.cos(.12) - .15 * np.sin(.12))) < 1e-7
    assert abs(poses.iloc[0].z - (-.015 * np.sin(.12) + .15 * np.cos(.12))) < 1e-7
    prefix = ('', 'armor_')[model]
    states = pd.read_csv(directory / f'{prefix}prediction_result_fixture.csv')
    assert states.coordinate_frame.eq('base').all() and len(states) == 11
    report.append(dict(model=model, frames=len(poses), observations=len(base), state_rows=len(states)))
(a.work_dir / 'verification.json').write_text(json.dumps(report, indent=2))
print('Two-model fixed-frame pipeline checks passed')
