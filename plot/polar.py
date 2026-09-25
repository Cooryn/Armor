"""Plot saved polar predictor results."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ._cli import parse_paths


def plot_polar(prediction_csv, raw_csv, output_dir, suffix='2'):
    res_df = pd.read_csv(prediction_csv)
    if res_df.empty:
        raise ValueError('Prediction CSV contains no rows')
    os.makedirs(output_dir, exist_ok=True)
    rmse_tyaw, rmse_tpitch, rmse_dist, rmse_ayaw = [np.sqrt((res_df[c]**2).mean())
        for c in ['err_target_yaw', 'err_target_pitch', 'err_distance', 'err_armor_yaw']]
    fig1, axs1 = plt.subplots(3, 3, figsize=(16, 10), sharex=True, dpi=150)
    fig1.suptitle('Polar Predictor', fontsize=18, fontweight='bold')

    # 第一列：位置 (xc, yc, zc)
    axs1[0, 0].plot(res_df['frame_id'], res_df['xc'], color='#1f77b4', linewidth=2)
    axs1[0, 0].set_ylabel('xc (m)')
    axs1[1, 0].plot(res_df['frame_id'], res_df['yc'], color='#1f77b4', linewidth=2)
    axs1[1, 0].set_ylabel('yc (m)')
    axs1[2, 0].plot(res_df['frame_id'], res_df['zc'], color='#1f77b4', linewidth=2)
    axs1[2, 0].set_ylabel('zc (m)')
    axs1[2, 0].set_xlabel('Frame ID')

    # 第二列：速度 (vxc, vyc, vzc)
    axs1[0, 1].plot(res_df['frame_id'], res_df['vxc'], color='#ff7f0e', linewidth=2)
    axs1[0, 1].set_ylabel('vxc (m/s)')
    axs1[1, 1].plot(res_df['frame_id'], res_df['vyc'], color='#ff7f0e', linewidth=2)
    axs1[1, 1].set_ylabel('vyc (m/s)')
    axs1[2, 1].plot(res_df['frame_id'], res_df['vzc'], color='#ff7f0e', linewidth=2)
    axs1[2, 1].set_ylabel('vzc (m/s)')
    axs1[2, 1].set_xlabel('Frame ID')

    # 第三列：旋转姿态与结构参数 (body_yaw, w, r)
    axs1[0, 2].plot(res_df['frame_id'], res_df['body_yaw'], color='#2ca02c', linewidth=2)
    axs1[0, 2].set_ylabel('body_yaw (rad)')
    axs1[1, 2].plot(res_df['frame_id'], res_df['w'], color='#d62728', linewidth=2)
    axs1[1, 2].set_ylabel('w (rad/s)')
    axs1[2, 2].plot(res_df['frame_id'], res_df['r'], color='#9467bd', linewidth=2)
    axs1[2, 2].set_ylabel('Radius r (m)')
    axs1[2, 2].set_xlabel('Frame ID')

    # 统一设置网格线
    for ax in axs1.flat:
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    pred_curve_path = os.path.join(output_dir, f'polar_prediction_curve_{suffix}.png')
    plt.savefig(pred_curve_path)
    plt.close()
    print(f"生成全状态内部图: {pred_curve_path}")

    # 4. 导出 polar_error_curve.png
    fig2, axs2 = plt.subplots(2, 2, figsize=(12, 8), sharex=True, dpi=150)
    fig2.suptitle('Polar Observation Residuals', fontsize=16, fontweight='bold')

    axs2[0,0].plot(res_df['frame_id'], res_df['err_target_yaw'], alpha=0.8, color='red')
    axs2[0,0].set_title(f'Target Yaw Error (RMSE: {rmse_tyaw:.4f})')
    axs2[0,1].plot(res_df['frame_id'], res_df['err_target_pitch'], alpha=0.8, color='blue')
    axs2[0,1].set_title(f'Target Pitch Error (RMSE: {rmse_tpitch:.4f})')
    axs2[1,0].plot(res_df['frame_id'], res_df['err_distance'], alpha=0.8, color='green')
    axs2[1,0].set_title(f'Distance Error (RMSE: {rmse_dist:.4f})')
    axs2[1,1].plot(res_df['frame_id'], res_df['err_armor_yaw'], alpha=0.8, color='purple')
    axs2[1,1].set_title(f'Armor Yaw Error (RMSE: {rmse_ayaw:.4f})')

    for ax in axs2.flat:
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Frame ID')

    plt.tight_layout()
    err_curve_path = os.path.join(output_dir, f'polar_error_curve_{suffix}.png')
    plt.savefig(err_curve_path)
    plt.close()
    print(f"生成误差图: {err_curve_path}")

def main():
    args = parse_paths(__doc__)
    plot_polar(args.results_dir / f'polar_prediction_result_{args.suffix}.csv',
                  args.data_dir / f'pose_raw_{args.suffix}.csv', args.output_dir, args.suffix)

if __name__ == '__main__':
    main()
