import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_raw_data(csv_path, output_dir, suffix="1"):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取 C++ 导出的 CSV 数据
    print(f"正在读取数据: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {csv_path}")
        return

    # 提取横坐标 (Frame ID)
    frames = df['frame_id']

    # ==========================================
    # 任务目标一：生成 pose_raw_curve.png
    # 包含数据：x, y, z, distance, target_pitch
    # ==========================================
    fig1, axs1 = plt.subplots(5, 1, figsize=(12, 14), sharex=True, dpi=150)
    fig1.suptitle('Raw Pose Observation Curves', fontsize=18, fontweight='bold')

    # X 轴坐标
    axs1[0].plot(frames, df['x'], color='#d62728', linewidth=1.5)
    axs1[0].set_ylabel('X (m)', fontweight='bold')
    
    # Y 轴坐标
    axs1[1].plot(frames, df['y'], color='#2ca02c', linewidth=1.5)
    axs1[1].set_ylabel('Y (m)', fontweight='bold')
    
    # Z 轴坐标 (深度)
    axs1[2].plot(frames, df['z'], color='#1f77b4', linewidth=1.5)
    axs1[2].set_ylabel('Z (m)', fontweight='bold')

    # Distance (直线距离)
    axs1[3].plot(frames, df['distance'], color='#9467bd', linewidth=1.5)
    axs1[3].set_ylabel('Distance (m)', fontweight='bold')

    # Target Pitch (目标俯仰角)
    axs1[4].plot(frames, df['target_pitch'], color='#ff7f0e', linewidth=1.5)
    axs1[4].set_ylabel('Target Pitch (rad)', fontweight='bold')
    axs1[4].set_xlabel('Frame ID', fontsize=12)

    # 统一设置网格
    for ax in axs1:
        ax.grid(True, alpha=0.4, linestyle='--')

    plt.tight_layout()
    pose_raw_path = os.path.join(output_dir, f'pose_raw_curve_{suffix}.png') # 动态拼接
    fig1.savefig(pose_raw_path)
    plt.close(fig1)
    print(f"已生成位姿观测曲线: {pose_raw_path}")

    # ==========================================
    # 任务目标二：生成 raw_yaw_curve.png
    # 包含数据：target_yaw, armor_orientation_yaw
    # ==========================================
    fig2, axs2 = plt.subplots(2, 1, figsize=(12, 8), sharex=True, dpi=150)
    fig2.suptitle('Raw Yaw Observation Curves', fontsize=18, fontweight='bold')

    # Target Yaw (目标相对于相机中心的偏航角)
    axs2[0].plot(frames, df['target_yaw'], color='#17becf', linewidth=1.5)
    axs2[0].set_ylabel('Target Yaw (rad)', fontweight='bold')
    
    # Armor Orientation Yaw (装甲板自身的绝对偏航角)
    axs2[1].plot(frames, df['armor_orientation_yaw'], color='#e377c2', linewidth=1.5)
    axs2[1].set_ylabel('Armor Yaw (rad)', fontweight='bold')
    axs2[1].set_xlabel('Frame ID', fontsize=12)

    # 统一设置网格
    for ax in axs2:
        ax.grid(True, alpha=0.4, linestyle='--')

    plt.tight_layout()
    raw_yaw_path = os.path.join(output_dir, f'raw_yaw_curve_{suffix}.png') # 动态拼接
    fig2.savefig(raw_yaw_path)
    plt.close(fig2)
    print(f"已生成偏航角观测曲线: {raw_yaw_path}")


if __name__ == "__main__":
    suffix = "2"

    input_csv_file = f"./data/pose_raw_{suffix}.csv"
    output_directory = "./results"

    plot_raw_data(input_csv_file, output_directory, suffix)