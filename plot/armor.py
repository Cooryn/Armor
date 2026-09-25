"""Export armor EKF plots from saved CSV files without rerunning the filter."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ._cli import parse_paths


def plot_armor(prediction_csv, raw_csv, output_dir, suffix='1'):
    res_df = pd.read_csv(prediction_csv)
    if res_df.empty:
        raise ValueError('Prediction CSV contains no rows')
    raw = pd.read_csv(raw_csv)
    raw = raw[raw.frame_id.isin(res_df.frame_id)]
    closest = raw.sort_values('distance').groupby('frame_id', sort=True).head(1).sort_values('frame_id')
    obs_trajectory = list(closest[['x', 'z']].itertuples(index=False, name=None))
    os.makedirs(output_dir, exist_ok=True)
    obs_traj = np.array(obs_trajectory if obs_trajectory else [(np.nan, np.nan)])  # (N, 2): [x, z]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    fig.suptitle('Top-down Trajectory (X-Z plane)', fontsize=14, fontweight='bold')

    ax.plot(obs_traj[:, 0], obs_traj[:, 1], 'o-', color='#1f77b4',
            linewidth=1.5, markersize=3, label='Observed', alpha=0.8)
    ax.plot(res_df['xa'], res_df['za'], 'x-', color='#d62728',
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

    # 4. 输出统一车体 yaw 曲线
    fig_byaw, ax_byaw = plt.subplots(figsize=(10, 4), dpi=150)
    ax_byaw.plot(res_df['frame_id'], res_df['body_yaw'], color='#2ca02c')
    ax_byaw.set_title('Body Yaw Curve')
    ax_byaw.set_xlabel('Frame ID'); ax_byaw.set_ylabel('Body Yaw (rad)')
    ax_byaw.grid(True, linestyle='--')
    fig_byaw.savefig(os.path.join(output_dir, f'body_yaw_curve_{suffix}.png'))
    plt.close(fig_byaw)

    # 5. 输出 Folded Armor Yaw (观测板向 vs 预测板向)
    fig_folded, ax_folded = plt.subplots(figsize=(10, 4), dpi=150)
    ax_folded.scatter(res_df['frame_id'], res_df['obs_armor_yaw'], color='red', s=5, alpha=0.5, label='Observed')
    ax_folded.plot(res_df['frame_id'], res_df['pred_armor_yaw'], color='blue', linewidth=1.5, label='Predicted (Folded)')
    ax_folded.set_title('Folded Armor Yaw Tracking')
    ax_folded.set_xlabel('Frame ID'); ax_folded.set_ylabel('Armor Yaw (rad)')
    ax_folded.legend(); ax_folded.grid(True, linestyle='--')
    fig_folded.savefig(os.path.join(output_dir, f'folded_armor_yaw_curve_{suffix}.png'))
    plt.close(fig_folded)

    # 6. 输出残差误差四宫格
    fig_err, axs_err = plt.subplots(2, 2, figsize=(12, 8), sharex=True, dpi=150)
    fig_err.suptitle('Armor Prediction Observation Errors', fontsize=16)
    axs_err[0, 0].plot(res_df['frame_id'], res_df['err_target_yaw'], color='red')
    axs_err[0, 0].set_title(f'Target Yaw Error')
    axs_err[0, 1].plot(res_df['frame_id'], res_df['err_target_pitch'], color='blue')
    axs_err[0, 1].set_title(f'Target Pitch Error')
    axs_err[1, 0].plot(res_df['frame_id'], res_df['err_distance'], color='green')
    axs_err[1, 0].set_title(f'Distance Error')
    axs_err[1, 1].plot(res_df['frame_id'], res_df['err_armor_yaw'], color='purple')
    axs_err[1, 1].set_title(f'Armor Yaw Error')
    for ax in axs_err.flat:
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_err.savefig(os.path.join(output_dir, f'armor_prediction_error_curve_{suffix}.png'))
    plt.close(fig_err)

    print("所有文件已保存至：", output_dir)


def main():
    args = parse_paths(__doc__, default_suffix='1')
    plot_armor(args.results_dir / f'armor_prediction_result_{args.suffix}.csv',
               args.data_dir / f'pose_raw_{args.suffix}.csv', args.output_dir, args.suffix)

if __name__ == '__main__':
    main()
