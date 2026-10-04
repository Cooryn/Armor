"""Camera-only moving-platform simulation; no real calibration is assumed."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--bin-dir', type=Path, default=ROOT)
    parser.add_argument('--filter-bin-dir', type=Path, default=ROOT/'tests/build/Release')
    args = parser.parse_args()
    out = args.work_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)

    def run(exe, *options, success=True):
        suffix = '.exe' if sys.platform == 'win32' else ''
        result = subprocess.run([str((args.filter_bin_dir if exe == 'camera_filter_replay' else args.bin_dir).resolve() / (exe + suffix)), *map(str, options)],
                                capture_output=True, timeout=60)
        if (result.returncode == 0) != success:
            raise AssertionError((exe, result.returncode, result.stdout, result.stderr))
        return result

    def rotation(vector):
        return cv2.Rodrigues(np.asarray(vector, dtype=float))[0]

    axis = np.array([.01, .03, .02])
    offset = np.array([.06, -.02, .04])
    mount_angle = .08
    mount = rotation([0, 0, mount_angle])
    calibration = out / 'SIMULATION_calibration.csv'
    pd.DataFrame([dict(axis_x_m=axis[0], axis_y_m=axis[1], axis_z_m=axis[2],
        camera_x_m=offset[0], camera_y_m=offset[1], camera_z_m=offset[2],
        mount_qw=np.cos(mount_angle/2), mount_qx=0, mount_qy=0, mount_qz=np.sin(mount_angle/2))]).to_csv(calibration, index=False)
    center = np.array([.2, .1, 3.])
    heading = .25
    base_rotation = rotation([0, -heading, 0])
    base_position = center - .26 * base_rotation[:, 2]
    rows, telemetry, transforms = [], [], []
    for frame in range(120):
        yaw, pitch, roll = .6*np.sin(frame*.05), .15*np.sin(frame*.07), .03*np.sin(frame*.09)
        ry = rotation([0, yaw, 0])
        pr = rotation([pitch, 0, 0]) @ rotation([0, 0, roll])
        R = ry @ pr @ mount
        t = ry @ (axis + pr @ offset)
        transforms.append((R, t))
        camera_position = R.T @ (base_position - t)
        camera_rotation = R.T @ base_rotation
        rvec = cv2.Rodrigues(camera_rotation)[0].ravel()
        x, y, z = camera_position
        rows.append(dict(frame_id=frame, timestamp=frame*1000/30, x=x, y=y, z=z,
            target_yaw=np.arctan2(x, z), target_pitch=np.arctan2(y, np.hypot(x, z)),
            distance=np.linalg.norm(camera_position), armor_orientation_yaw=np.arctan2(camera_rotation[2, 0], camera_rotation[2, 2]),
            rvec_x=rvec[0], rvec_y=rvec[1], rvec_z=rvec[2], coordinate_frame='camera',
            detection_score=1, reprojection_error=0, pnp_candidate_count=1, pnp_used_temporal=0))
        telemetry.append(dict(timestamp=frame*1000/30, yaw_deg=np.degrees(yaw),
                              pitch_deg=np.degrees(pitch), roll_deg=np.degrees(roll)))
    camera_csv, telemetry_csv, base_csv = out/'pose_camera.csv', out/'SIMULATION_telemetry.csv', out/'pose_base.csv'
    pd.DataFrame(rows).to_csv(camera_csv, index=False)
    pd.DataFrame(telemetry).to_csv(telemetry_csv, index=False)
    frame_options = ['--telemetry', telemetry_csv, '--calibration', calibration]
    run('pose_base', '--input', camera_csv, '--output', base_csv, *frame_options)
    for origin_time in (-1, -2, float('nan'), float('inf')):
        for program, source in (('pose_base', camera_csv), ('camera_tracking', base_csv)):
            rejected_output = out/f'{program}_invalid_origin.csv'
            run(program, '--input', source, '--output', rejected_output,
                *frame_options, '--origin-time-ms', origin_time, success=False)
            assert not rejected_output.exists()
    shifted_telemetry, shifted_camera = out/'shifted_telemetry.csv', out/'shifted_camera.csv'
    for source, output in ((telemetry_csv, shifted_telemetry), (camera_csv, shifted_camera)):
        shifted = pd.read_csv(source)
        shifted.timestamp += 100
        shifted.to_csv(output, index=False)
    shifted_options = ['--telemetry', shifted_telemetry, '--calibration', calibration]
    shifted_output = out/'shifted_base.csv'
    run('pose_base', '--input', shifted_camera, '--output', shifted_output, *shifted_options)
    shifted_base = pd.read_csv(shifted_output)
    assert shifted_base.base_reference_timestamp_ms.eq(100).all()
    np.testing.assert_allclose(shifted_base[['x', 'y', 'z']], pd.read_csv(base_csv)[['x', 'y', 'z']], atol=1e-12)
    rejected_output = out/'zero_origin_outside_coverage.csv'
    run('pose_base', '--input', shifted_camera, '--output', rejected_output,
        *shifted_options, '--origin-time-ms', 0, success=False)
    assert not rejected_output.exists()
    origin = transforms[0][1]
    base = pd.read_csv(base_csv)
    np.testing.assert_allclose(base[['x', 'y', 'z']], np.tile(base_position-origin, (120, 1)), atol=1e-12)
    np.testing.assert_allclose(base.armor_orientation_yaw, heading, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(base[['qw', 'qx', 'qy', 'qz']], axis=1), 1, atol=1e-12)
    expected_q = np.array([np.cos(heading/2), 0, -np.sin(heading/2), 0])
    np.testing.assert_allclose(base[['qw', 'qx', 'qy', 'qz']], np.tile(expected_q, (120, 1)), atol=1e-12)
    assert base.coordinate_frame.eq('base').all()
    metadata = ['base_reference_timestamp_ms', 'base_origin_x_m', 'base_origin_y_m', 'base_origin_z_m']

    def filter_base(poses, folder):
        # Metadata belongs to the caller; the filter only receives typed observations.
        folder.mkdir(parents=True)
        fixture = folder/'observations.txt'
        poses[['frame_id', 'timestamp', 'target_yaw', 'target_pitch', 'distance',
               'armor_orientation_yaw']].to_csv(fixture, sep=' ', index=False, header=False)
        run('camera_filter_replay', fixture, folder)
        for name in ('states', 'futures'):
            path = folder/f'{name}.csv'
            result = pd.read_csv(path).merge(poses[['frame_id', 'coordinate_frame', *metadata]],
                                             on='frame_id', validate='many_to_one')
            result.to_csv(path, index=False)
        return folder/'states.csv', folder/'futures.csv'

    state_path, future_path = filter_base(base, out/'results')
    state = pd.read_csv(state_path)
    np.testing.assert_allclose(state[['xc', 'yc', 'zc']], np.tile(center-origin, (119, 1)), atol=1e-7)
    np.testing.assert_allclose(state[['vxc', 'vyc', 'vzc', 'w']], 0, atol=1e-7)
    assert state.coordinate_frame.eq('base').all() and state.accepted_count.eq(1).all()
    future = pd.read_csv(future_path)
    assert len(future) == 4*len(state) and future.coordinate_frame.eq('base').all()
    np.testing.assert_allclose(np.linalg.norm(future[['qw', 'qx', 'qy', 'qz']], axis=1), 1, atol=1e-12)
    commands_path = out/'camera_gimbal_sim.csv'
    run('camera_tracking', '--input', state_path, '--output', commands_path, *frame_options)
    commands = pd.read_csv(commands_path)
    for _, row in commands.iterrows():
        R, t = transforms[int(row.frame_id)]
        x, y, z = R.T @ (center - t)
        np.testing.assert_allclose([row.yaw_error_deg, row.pitch_error_deg],
            np.degrees([np.arctan2(x, z), -np.arctan2(y, np.hypot(x, z))]), atol=1e-6)
    assert commands.yaw_rate_dps.abs().le(60).all() and commands.pitch_rate_dps.abs().le(45).all()
    assert np.isfinite(commands[['yaw_rate_dps', 'pitch_rate_dps']]).all().all()
    dt = np.diff(commands.timestamp)/1000
    for col in ('yaw_rate_dps', 'pitch_rate_dps'):
        assert (np.abs(np.diff(commands[col])) <= 180*dt + 1e-9).all()

    reset_csv = out/'pose_reset.csv'
    reset_options = [*frame_options, '--origin-time-ms', 1000]
    run('pose_base', '--input', camera_csv, '--output', reset_csv, *reset_options)
    reset = pd.read_csv(reset_csv)
    np.testing.assert_allclose(reset[['x', 'y', 'z']], np.tile(base_position-transforms[30][1], (120, 1)), atol=1e-12)
    np.testing.assert_allclose(reset[['qw', 'qx', 'qy', 'qz']], base[['qw', 'qx', 'qy', 'qz']], atol=1e-12)
    reset_state, _ = filter_base(reset, out/'reset')
    run('camera_tracking', '--input', reset_state, '--output', out/'reset_commands.csv', *reset_options)
    reset_commands = pd.read_csv(out/'reset_commands.csv')
    cols = ['yaw_error_deg', 'pitch_error_deg', 'yaw_rate_dps', 'pitch_rate_dps']
    np.testing.assert_allclose(reset_commands[cols], commands[cols], atol=1e-7)
    run('camera_tracking', '--input', reset_state, '--output', out/'wrong_origin.csv', *frame_options, success=False)
    mixed = state.copy()
    mixed.loc[50, metadata] = pd.read_csv(reset_state).loc[50, metadata]
    mixed.to_csv(out/'mixed.csv', index=False)
    run('camera_tracking', '--input', out/'mixed.csv', '--output', out/'mixed_commands.csv',
        *frame_options, success=False)
    assert not (out/'mixed_commands.csv').exists()

    loss = state.copy()
    loss.loc[10:20, ['accepted_count', 'status']] = [0, 'prediction_only']
    loss.loc[30, 'zc'] = np.nan
    loss.to_csv(out/'loss.csv', index=False)
    run('camera_tracking', '--input', out/'loss.csv', '--output', out/'loss_commands.csv', *frame_options)
    stopped = pd.read_csv(out/'loss_commands.csv')
    assert not stopped.loc[10:20, 'control_valid'].any()
    assert stopped.loc[10:20, ['yaw_rate_dps', 'pitch_rate_dps']].eq(0).all().all()
    assert not stopped.loc[30, 'control_valid']
    run('camera_tracking', '--input', state_path, '--output', out/'missing_transform.csv', success=False)
    run('pose_base', '--input', camera_csv, '--output', out/'missing_calibration.csv', success=False)
    snapshot = camera_csv.read_bytes()
    run('pose_base', '--input', camera_csv, '--output', camera_csv, *frame_options, success=False)
    assert camera_csv.read_bytes() == snapshot
    invalid = pd.DataFrame(telemetry).iloc[1:]
    invalid.to_csv(out/'uncovered.csv', index=False)
    run('pose_base', '--input', camera_csv, '--output', out/'uncovered_result.csv',
        '--telemetry', out/'uncovered.csv', '--calibration', calibration, success=False)
    assert not (out/'uncovered_result.csv').exists()

    report = dict(status='passed', simulation_only=True, frames=120, state_rows=len(state),
        future_rows=len(future), command_rows=len(commands),
        max_position_error_m=float(np.abs(base[['x','y','z']].to_numpy()-(base_position-origin)).max()),
        max_stationary_center_speed_mps=float(state[['vxc','vyc','vzc']].abs().max().max()))
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print('Simulation output:', out)


if __name__ == '__main__':
    main()
