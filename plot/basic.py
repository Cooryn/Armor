"""Plot saved basic predictor results."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ._cli import parse_paths


def plot_basic(prediction_csv, raw_csv, output_dir, suffix='2'):
    res_df = pd.read_csv(prediction_csv)
    if res_df.empty:
        raise ValueError('Prediction CSV contains no rows')
    os.makedirs(output_dir, exist_ok=True)
    raw = pd.read_csv(raw_csv)
    closest = raw.sort_values('distance').groupby('frame_id', sort=True).head(1).sort_values('frame_id')
    obs_trajectory = list(closest[['x', 'z']].itertuples(index=False, name=None))
    rmse_x, rmse_z, rmse_yaw, rmse_distance = [np.sqrt((res_df[c]**2).mean())
        for c in ['error_x', 'error_z', 'error_yaw', 'error_distance']]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, dpi=150)
    fig.suptitle('Predictor Errors', fontsize=16, fontweight='bold')

    axs[0, 0].plot(res_df['frame_id'], res_df['error_x'], color='#d62728', linewidth=1.5)
    axs[0, 0].set_ylabel('Error X (m)')
    axs[0, 0].set_title(f'RMSE_x: {rmse_x:.4f}')

    axs[0, 1].plot(res_df['frame_id'], res_df['error_z'], color='#1f77b4', linewidth=1.5)
    axs[0, 1].set_ylabel('Error Z (m)')
    axs[0, 1].set_title(f'RMSE_z: {rmse_z:.4f}')

    axs[1, 0].plot(res_df['frame_id'], res_df['error_yaw'], color='#2ca02c', linewidth=1.5)
    axs[1, 0].set_ylabel('Error Yaw (rad)')
    axs[1, 0].set_title(f'RMSE_yaw: {rmse_yaw:.4f}')
    axs[1, 0].set_xlabel('Frame ID')

    axs[1, 1].plot(res_df['frame_id'], res_df['error_distance'], color='#9467bd', linewidth=1.5)
    axs[1, 1].set_ylabel('Error Distance (m)')
    axs[1, 1].set_title(f'RMSE_distance: {rmse_distance:.4f}')
    axs[1, 1].set_xlabel('Frame ID')

    for ax in axs.flat:
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    img_out_path = os.path.join(output_dir, f'prediction_error_curve_{suffix}.png')
    plt.savefig(img_out_path)
    plt.close()
    print(f"已生成误差曲线: {img_out_path}")

    # ==========================================
    # 小车中心点二维位置：世界俯视图（x-z 平面）
    # 横轴 X（左右偏移），纵轴 Z（深度），越往上越远
    # ==========================================
    obs_traj = np.array(obs_trajectory)  # (N, 2): [x, z]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    fig.suptitle('Top-down Trajectory (X-Z plane)', fontsize=14, fontweight='bold')

    ax.plot(obs_traj[:, 0], obs_traj[:, 1], 'o-', color='#1f77b4',
            linewidth=1.5, markersize=3, label='Observed', alpha=0.8)
    ax.plot(res_df['predicted_x'], res_df['predicted_z'], 'x-', color='#d62728',
            linewidth=1.5, markersize=3, label='Predicted', alpha=0.8)

    ax.scatter(obs_traj[0, 0], obs_traj[0, 1], color='green', s=80,
               marker='o', zorder=5, label='Start')
    ax.scatter(obs_traj[-1, 0], obs_traj[-1, 1], color='black', s=80,
               marker='s', zorder=5, label='End')

    ax.set_xlabel('X (lateral, m)')
    ax.set_ylabel('Z (depth, m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    traj_out_path = os.path.join(output_dir, f'top_down_trajectory_{suffix}.png')
    plt.savefig(traj_out_path)
    plt.close()
    print(f"已生成俯视轨迹图: {traj_out_path}")

def main():
    args = parse_paths(__doc__)
    plot_basic(args.results_dir / f'prediction_result_{args.suffix}.csv',
                  args.data_dir / f'pose_raw_{args.suffix}.csv', args.output_dir, args.suffix)

if __name__ == '__main__':
    main()
