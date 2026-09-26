"""Offline CSV runner for ArmorEKF; exports numeric results only."""
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd

if __package__:
    from .predictor_armor import ArmorEKF
else:
    from predictor_armor import ArmorEKF

def run_predict_armor(csv_input_path, output_dir, suffix="1", adaptive_noise=True,
                      prediction_horizon_ms=50.0):
    if not np.isfinite(prediction_horizon_ms) or prediction_horizon_ms < 0:
        raise ValueError('Prediction horizon must be finite and nonnegative (ms)')
    if not os.path.exists(csv_input_path):
        print(f"错误: 找不到输入文件 {csv_input_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    data = pd.read_csv(csv_input_path)

    # CSV has no rows for missed detections. Interpolate only the video clock,
    # never observations, and expose every internal missing frame as prediction-only.
    if data.frame_id.nunique() < 2:
        print("No prediction exported: at least two detected frames are required.")
        return
    if (not np.isfinite(data[['frame_id', 'timestamp']].to_numpy()).all()
            or (data.frame_id < 0).any() or (data.frame_id % 1 != 0).any()):
        raise ValueError("Frame IDs and timestamps must be finite, valid video times")
    groups = dict(tuple(data.groupby('frame_id', sort=True)))
    frame_ids = np.array(sorted(groups), dtype=int)
    timestamps = np.array([groups[f].timestamp.iloc[0] for f in frame_ids])
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Timestamps must increase between frames; regenerate legacy CSV")
    if any(g.timestamp.nunique() != 1 for g in groups.values()):
        raise ValueError("All observations of a frame must share its timestamp")
    ekf = ArmorEKF(adaptive_noise=adaptive_noise)
    results, observation_log, future_results = [], [], []
    last_timestamp = None
    for frame_id in range(frame_ids[0], frame_ids[-1] + 1):
        timestamp = float(np.interp(frame_id, frame_ids, timestamps))
        group = groups.get(frame_id, data.iloc[:0])
        all_obs = [dict(Z_obs=row[['target_yaw', 'target_pitch', 'distance',
                                   'armor_orientation_yaw']].to_numpy(dtype=float).reshape(4, 1),
                        detection_score=row.get('detection_score', np.nan),
                        reprojection_error=row.get('reprojection_error', np.nan))
                   for _, row in group.iterrows()]
        valid = [o for o in all_obs if ekf.valid_observation(o['Z_obs'])]
        if not ekf.is_initialized:
            if not valid:
                for index in range(len(all_obs)):
                    observation_log.append(dict(frame_id=frame_id, observation_index=index,
                                                accepted=False, armor_id=-1, nis=np.nan, reason='invalid'))
                continue
            # One seed only; do not re-use it in a Kalman update.
            seed = min(valid, key=lambda o: o['Z_obs'][2, 0])
            ekf.initialize(seed['Z_obs'])
            last_timestamp = timestamp
            for index, obs in enumerate(all_obs):
                observation_log.append(dict(frame_id=frame_id, observation_index=index,
                                            accepted=obs is seed, armor_id=0 if obs is seed else -1,
                                            nis=np.nan, reason='initialization' if obs is seed else 'initialization_unused'))
            continue
        ekf.predict((timestamp - last_timestamp) / 1000.)
        last_timestamp = timestamp
        prior = ekf.X.copy()
        matches, diagnostics = ekf.update_multi(all_obs)
        for diagnostic in diagnostics:
            source = group.iloc[diagnostic['observation_index']]
            observation_log.append(dict(frame_id=frame_id, timestamp=timestamp,
                                        observed_distance=source['distance'],
                                        observed_armor_yaw=source['armor_orientation_yaw'],
                                        **diagnostic))
        match = min(matches, key=lambda m: m['Z_obs'][2, 0]) if matches else None
        if match is not None:
            best_id = match['armor_id']
            Z_pred = match['predicted']
            errors = -match['residual'][:, 0]
            obs_yaw = match['Z_obs'][3, 0]
        else:
            # Show the nearest predicted plate, with NO fabricated observation error.
            best_id = min(range(4), key=lambda aid: ekf.h(prior, aid)[2, 0])
            Z_pred = ekf.h(prior, best_id)
            errors = np.full(4, np.nan)
            obs_yaw = np.nan
        r_i = prior[8, 0] + (prior[9, 0] if best_id % 2 else 0)
        forecast = ekf.forecast(prediction_horizon_ms / 1000.)
        future = forecast['state']
        prediction_timestamp = timestamp + prediction_horizon_ms
        for aid, (x, y, z, yaw) in enumerate(forecast['plates']):
            future_results.append(dict(frame_id=frame_id, timestamp=timestamp,
                prediction_timestamp=prediction_timestamp, prediction_horizon_ms=prediction_horizon_ms,
                armor_id=aid, x=x, y=y, z=z, armor_orientation_yaw=yaw,
                source_status='updated' if matches else 'prediction_only'))
        results.append(dict(
            frame_id=frame_id, timestamp=timestamp,
            prediction_horizon_ms=prediction_horizon_ms, prediction_timestamp=prediction_timestamp,
            future_xc=future[0, 0], future_yc=future[2, 0], future_zc=future[4, 0],
            future_body_yaw=future[6, 0],
            xc=ekf.X[0, 0], yc=ekf.X[2, 0], zc=ekf.X[4, 0],
            vxc=ekf.X[1, 0], vyc=ekf.X[3, 0], vzc=ekf.X[5, 0], w=ekf.X[7, 0],
            xa=prior[0, 0] + r_i * np.sin(Z_pred[3, 0]),
            za=prior[4, 0] - r_i * np.cos(Z_pred[3, 0]), armor_id=best_id,
            body_yaw=ekf.X[6, 0], pred_armor_yaw=Z_pred[3, 0], obs_armor_yaw=obs_yaw,
            err_target_yaw=errors[0], err_target_pitch=errors[1],
            err_distance=errors[2], err_armor_yaw=errors[3],
            r=ekf.X[8, 0], dl=ekf.X[9, 0], dh=ekf.X[10, 0],
            observation_count=len(all_obs), accepted_count=len(matches),
            rejected_count=len(all_obs)-len(matches),
            status='updated' if matches else 'prediction_only'))

    if not results:
        print("No prediction exported: at least two detected frames are required.")
        return
    res_df = pd.DataFrame(results)
    pd.DataFrame(observation_log).to_csv(
        os.path.join(output_dir, f"armor_observation_diagnostics_{suffix}.csv"), index=False)

    # ==========================================
    # 1. 导出预测结果 CSV
    # ==========================================
    csv_out_path = os.path.join(output_dir, f'armor_prediction_result_{suffix}.csv')
    res_df.to_csv(csv_out_path, index=False)
    pd.DataFrame(future_results).to_csv(
        os.path.join(output_dir, f'armor_future_prediction_{suffix}.csv'), index=False)

    # 2. 导出误差文本 RMSE TXT
    rmse_tyaw = np.sqrt(np.mean(res_df['err_target_yaw']**2))
    rmse_tpitch = np.sqrt(np.mean(res_df['err_target_pitch']**2))
    rmse_dist = np.sqrt(np.mean(res_df['err_distance']**2))
    rmse_ayaw = np.sqrt(np.mean(res_df['err_armor_yaw']**2))
    txt_out_path = os.path.join(output_dir, f'armor_rmse_result_{suffix}.txt')
    with open(txt_out_path, 'w') as f:
        f.write("Metrics: prior residuals of accepted observations only; not ground-truth error.\n")
        f.write(f"RMSE_target_yaw: {rmse_tyaw:.6f} rad\n")
        f.write(f"RMSE_target_pitch: {rmse_tpitch:.6f} rad\n")
        f.write(f"RMSE_distance: {rmse_dist:.6f} m\n")
        f.write(f"RMSE_armor_yaw: {rmse_ayaw:.6f} rad\n")

    return res_df

def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suffix', default='1')
    parser.add_argument('--input', type=Path, help='Input pose CSV; overrides --suffix input')
    parser.add_argument('--output-dir', type=Path, default=root / 'results')
    parser.add_argument('--fixed-noise', action='store_true', help='Use legacy fixed observation covariance')
    parser.add_argument('--prediction-horizon-ms', type=float, default=50.0,
                        help='Forecast offset from each image timestamp, in ms (default: 50; 0 disables lead)')
    args = parser.parse_args()
    source = args.input if args.input is not None else root / 'data' / f'pose_raw_{args.suffix}.csv'
    if not np.isfinite(args.prediction_horizon_ms) or args.prediction_horizon_ms < 0:
        parser.error('--prediction-horizon-ms must be finite and nonnegative')
    run_predict_armor(source, args.output_dir, args.suffix, adaptive_noise=not args.fixed_noise,
                      prediction_horizon_ms=args.prediction_horizon_ms)

if __name__ == '__main__':
    main()
