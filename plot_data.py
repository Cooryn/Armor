import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_pose_data(csv_file_path, output_image_path):
    # 检查文件是否存在
    if not os.path.exists(csv_file_path):
        print(f"找不到数据文件: {csv_file_path}")
        return

    # 读取 CSV 数据
    data = pd.read_csv(csv_file_path)

    # 创建一个 2x2 的子图画布 (分辨率调高一点，看着更专业)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    fig.suptitle('Armor Pose Visualization (Delivery 5)', fontsize=16)

    # 绘制 X 轴平移
    axs[0, 0].plot(data['Frame'], data['X'], color='r', label='X Position')
    axs[0, 0].set_title('X Position over Frames')
    axs[0, 0].set_ylabel('Meters (m)')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # 绘制 Y 轴平移
    axs[0, 1].plot(data['Frame'], data['Y'], color='g', label='Y Position')
    axs[0, 1].set_title('Y Position over Frames')
    axs[0, 1].set_ylabel('Meters (m)')
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # 绘制 Z 轴平移 (距离)
    axs[1, 0].plot(data['Frame'], data['Z'], color='b', label='Z Position')
    axs[1, 0].set_title('Z Position (Distance) over Frames')
    axs[1, 0].set_xlabel('Frames')
    axs[1, 0].set_ylabel('Meters (m)')
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # 绘制 Yaw 角
    axs[1, 1].plot(data['Frame'], data['Yaw'], color='purple', label='Yaw Angle')
    axs[1, 1].set_title('Yaw Angle over Frames')
    axs[1, 1].set_xlabel('Frames')
    axs[1, 1].set_ylabel('Degrees (deg)')
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    # 自动调整布局防重叠
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 保存图片
    plt.savefig(output_image_path)
    print(f"数据可视化图表已成功保存至: {output_image_path}")

if __name__ == '__main__':
    csv_path = './results/pose_data_2.csv'
    out_path = './results/exp_result_raw_2.png'
    plot_pose_data(csv_path, out_path)