import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ._cli import parse_paths


def plot_basic(prediction_csv, output_dir, suffix='2'):
    res_df = pd.read_csv(prediction_csv)
    if res_df.empty:
        raise ValueError('Prediction CSV contains no rows')
    os.makedirs(output_dir, exist_ok=True)
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

def main():
    args = parse_paths('Plot single-plate errors')
    plot_basic(args.results_dir / f'prediction_result_{args.suffix}.csv', args.output_dir, args.suffix)

if __name__ == '__main__':
    main()
