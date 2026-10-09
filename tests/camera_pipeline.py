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
for model in range(3):
    directory = a.work_dir / str(model)
    raw = pd.read_csv(directory / 'data/pose_raw_fixture.csv')
    base = pd.read_csv(directory / 'data/pose_base_fixture.csv')
    poses = pd.read_csv(directory / 'data/camera_pose_fixture.csv')
    commands = pd.read_csv(directory / 'results/control_target_fixture.csv')
    assert raw.coordinate_frame.eq('camera').all() and base.coordinate_frame.eq('base').all()
    assert len(poses) == len(commands) == 12 and poses.synchronized.eq(1).all()
    assert commands.frame_id.tolist() == list(range(12))
    assert commands[commands.frame_id.between(5, 7)].valid.eq(0).all()
    assert commands.valid.eq(1).any(), 'controller never produced a valid absolute target'
    assert np.isfinite(commands[['yaw_rad', 'pitch_rad']].to_numpy()).all()
    np.testing.assert_allclose(base[['x', 'y', 'z']].to_numpy(), np.tile([.1, .2, 3], (len(base), 1)), atol=.003)
    assert base.base_reference_timestamp_ms.eq(0).all()
    prefix = ('', 'polar_', 'armor_')[model]
    states = pd.read_csv(directory / 'results' / f'{prefix}prediction_result_fixture.csv')
    assert states.coordinate_frame.eq('base').all() and len(states) == 11
    report.append(dict(model=model, frames=len(poses), observations=len(base), state_rows=len(states)))
(a.work_dir / 'verification.json').write_text(json.dumps(report, indent=2))
print('Three-model fixed-frame pipeline checks passed')
