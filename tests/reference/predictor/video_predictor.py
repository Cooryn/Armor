"""Independent NumPy replay of every video frame, including missed detections."""
from pathlib import Path
import numpy as np
import pandas as pd
from .predictor import SinglePlateEKF, wrap_to_pi
from .predictor_polar import PolarEKF
from .predictor_armor import ArmorEKF

BASIC_COLUMNS = ('frame_id predicted_x observed_x error_x predicted_z observed_z error_z '
                 'predicted_yaw observed_yaw error_yaw predicted_distance observed_distance error_distance').split()
POLAR_COLUMNS = ('frame_id xc vxc yc vyc zc vzc body_yaw w r err_target_yaw err_target_pitch '
                 'err_distance err_armor_yaw obs_armor_yaw').split()
ARMOR_COLUMNS = ('frame_id timestamp prediction_horizon_ms prediction_timestamp future_xc future_yc future_zc '
                 'future_body_yaw xc yc zc vxc vyc vzc w xa za armor_id body_yaw pred_armor_yaw obs_armor_yaw '
                 'err_target_yaw err_target_pitch err_distance err_armor_yaw r dl dh observation_count '
                 'accepted_count rejected_count status').split()
DIAGNOSTIC_COLUMNS = ('frame_id observation_index accepted armor_id nis reason timestamp observed_distance '
                      'observed_armor_yaw best_candidate_id distance_residual').split()
FUTURE_COLUMNS = ('frame_id timestamp prediction_timestamp prediction_horizon_ms armor_id x y z '
                  'armor_orientation_yaw source_status').split()
GEOMETRY_COLUMNS = ['frame_id', 'timestamp', 'initialized', 'status']
for prefix in ('current', 'future'):
    GEOMETRY_COLUMNS += [f'{prefix}_center_{axis}' for axis in ('x', 'y', 'z')]
    GEOMETRY_COLUMNS += [f'{prefix}_plate_{i}_{axis}' for i in range(4) for axis in ('x', 'y', 'z', 'yaw')]


def run_video_predictor(frames, model, directory, horizon_ms=50):
    """frames = [(integer ID, timestamp ms, [(xyz array, spherical/yaw array)])]."""
    if not np.isfinite(horizon_ms) or horizon_ms < 0:
        raise ValueError('Invalid prediction horizon')
    ekf = {'basic': SinglePlateEKF, 'polar': PolarEKF, 'armor': ArmorEKF}[model]()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    results, diagnostics, futures, geometry, errors = [], [], [], [], []
    last_id, last_time = -1, -1

    def valid(o):
        p, z = o
        if model == 'armor':
            return ekf.valid_observation(z.reshape(4, 1))
        return (np.isfinite(p).all() and SinglePlateEKF.valid_observation(z[:3])
                and (model == 'basic' or np.isfinite(z).all()))

    def plate_poses(state):
        if model == 'basic':
            return None, [np.r_[state[::2], 0]]
        center, poses = state[[0, 2, 4]], []
        for aid in range(4):
            yaw = wrap_to_pi(state[6] + aid * np.pi / 2)
            radius = state[8] + (state[9] if model == 'armor' and aid % 2 else 0)
            y = state[2] + (state[10] if model == 'armor' and aid % 2 else 0)
            x = state[0] + radius * np.sin(yaw)
            poses.append(np.array([x, y, state[4] - radius * np.cos(yaw), yaw]))
        return center, poses

    for fid, timestamp, obs in frames:
        if (fid < 0 or fid != int(fid) or fid <= last_id or not np.isfinite(timestamp) or timestamp < 0
                or (last_id >= 0 and timestamp <= last_time)):
            raise ValueError('Invalid frame clock')
        dt = (timestamp - last_time) / 1000
        last_id, last_time = fid, timestamp
        selected = min((i for i, o in enumerate(obs) if valid(o)),
                       key=lambda i: obs[i][1][2], default=None)
        observation = obs[selected] if selected is not None else None
        all_obs = [dict(Z_obs=z.reshape(4, 1)) for _, z in obs]
        status = 'waiting'
        if not ekf.is_initialized:
            if observation is not None:
                p, z = observation
                if model == 'basic':
                    ekf.initialize(z[:3])
                elif model == 'polar':
                    ekf.X[:, 0] = [p[0] - .26 * np.sin(z[3]), 0, p[1], 0,
                                   p[2] + .26 * np.cos(z[3]), 0, z[3], 0, .26]
                    ekf.is_initialized = True
                else:
                    ekf.initialize(z.reshape(4, 1))
                status = 'initialized'
            if model == 'armor':
                for i, (_, z) in enumerate(obs):
                    accepted = i == selected
                    reason = ('invalid' if not valid(obs[i]) else
                              'initialization' if accepted else 'initialization_unused')
                    diagnostics.append([fid, i, accepted, 0 if accepted else -1, np.nan, reason,
                                        timestamp, z[2], z[3], -1, np.nan])
        else:
            ekf.predict(dt)
            e, updated = np.full(4, np.nan), False
            if model == 'basic':
                s = ekf.state.copy()
                predicted = ekf.h(s)
                p, z = observation if observation is not None else (np.full(3, np.nan), np.full(4, np.nan))
                if observation is not None:
                    e = np.array([s[0] - p[0], s[4] - p[2], wrap_to_pi(predicted[0] - z[0]), predicted[2] - z[2]])
                results.append([fid, s[0], p[0], e[0], s[4], p[2], e[1], predicted[0], z[0], e[2], predicted[2], z[2], e[3]])
                if observation is not None:
                    updated = ekf.update(z[:3])
            elif model == 'polar':
                z = observation[1] if observation is not None else np.full(4, np.nan)
                if observation is not None:
                    aid = round(wrap_to_pi(z[3] - ekf.X[6, 0]) / (np.pi / 2))
                    e = ekf.h(ekf.X, aid)[:, 0] - z
                    e[[0, 1, 3]] = wrap_to_pi(e[[0, 1, 3]])
                    ekf.update(z.reshape(4, 1))
                    updated = True
                results.append([fid, *ekf.X[:, 0], *e, z[3]])
            else:
                prior = ekf.X.copy()
                matches, ds = ekf.update_multi(all_obs)
                for d in ds:
                    z = obs[d['observation_index']][1]
                    diagnostics.append([fid, d['observation_index'], d['accepted'], d['armor_id'], d['nis'], d['reason'],
                                        timestamp, z[2], z[3], d['best_candidate_id'], d['distance_residual']])
                match = min(matches, key=lambda m: m['Z_obs'][2, 0], default=None)
                aid, observed_yaw, plate = -1, np.nan, np.full(4, np.nan)
                if match is not None:
                    aid = match['armor_id']
                    e, observed_yaw = -match['residual'][:, 0], match['Z_obs'][3, 0]
                    plate = plate_poses(prior[:, 0])[1][aid]
                updated = bool(matches)
                forecast = ekf.forecast(horizon_ms / 1000)
                f, s = forecast['state'][:, 0], ekf.X[:, 0]
                status = 'updated' if updated else 'prediction_only'
                for i, p in enumerate(forecast['plates']):
                    futures.append([fid, timestamp, timestamp + horizon_ms, horizon_ms, i, *p, status])
                results.append([fid, timestamp, horizon_ms, timestamp + horizon_ms, *f[[0, 2, 4, 6]],
                                *s[[0, 2, 4, 1, 3, 5, 7]], plate[0], plate[2], aid, s[6], plate[3], observed_yaw,
                                *e, *s[8:], len(obs), len(matches), len(obs) - len(matches), status])
            status = 'updated' if updated else 'prediction_only'
            errors.append(e)
        g = [fid, timestamp, ekf.is_initialized, status]
        if ekf.is_initialized:
            state = ekf.state.copy() if model == 'basic' else ekf.X[:, 0].copy()
            future = state.copy()
            for i in ([0, 2, 4] if model == 'basic' else [0, 2, 4, 6]):
                future[i] += future[i + 1] * horizon_ms / 1000
            if model != 'basic':
                future[6] = wrap_to_pi(future[6])
            for s in (state, future):
                center, poses = plate_poses(s)
                g.extend(center if center is not None else [np.nan] * 3)
                g.extend(np.asarray(poses).ravel())
                g.extend([np.nan] * (4 * (4 - len(poses))))
        else:
            g.extend([np.nan] * (len(GEOMETRY_COLUMNS) - 4))
        geometry.append(g)
    prefix = {'basic': '', 'polar': 'polar_', 'armor': 'armor_'}[model]
    columns = {'basic': BASIC_COLUMNS, 'polar': POLAR_COLUMNS, 'armor': ARMOR_COLUMNS}[model]
    pd.DataFrame(results, columns=columns).to_csv(directory / f'{prefix}prediction_result_1.csv', index=False)
    pd.DataFrame(geometry, columns=GEOMETRY_COLUMNS).to_csv(directory / 'geometry.csv', index=False)
    if model == 'armor':
        pd.DataFrame(diagnostics, columns=DIAGNOSTIC_COLUMNS).to_csv(directory / 'armor_observation_diagnostics_1.csv', index=False)
        pd.DataFrame(futures, columns=FUTURE_COLUMNS).to_csv(directory / 'armor_future_prediction_1.csv', index=False)
    names = ['x', 'z', 'yaw', 'distance'] if model == 'basic' else ['target_yaw', 'target_pitch', 'distance', 'armor_yaw']
    units = ['m', 'm', 'rad', 'm'] if model == 'basic' else ['rad', 'rad', 'm', 'rad']
    with (directory / f'{prefix}rmse_result_1.txt').open('w', newline='\n') as metrics:
        if model == 'armor':
            metrics.write('Metrics: prior residuals of accepted observations only; not ground-truth error.\n')
        for i, (name, unit) in enumerate(zip(names, units)):
            valid_errors = [e[i] for e in errors if np.isfinite(e[i])]
            value = np.sqrt(np.mean(np.square(valid_errors))) if valid_errors else np.nan
            metrics.write(f'RMSE_{name}: {value:.6f} {unit}\n')
