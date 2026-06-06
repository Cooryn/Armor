import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_yaw_data(csv_file_path, output_image_path):
    if not os.path.exists(csv_file_path):
        print(f"找不到数据文件: {csv_file_path}")
        return

    data = pd.read_csv(csv_file_path)

    if 'Yaw' not in data.columns:
        print(f"CSV 格式不匹配！请确保 C++ 输出了 Yaw 列。")
        return

    plt.figure(figsize=(10, 6), dpi=150)

    plt.plot(data['Frame'], data['Yaw'], color='purple', label='Yaw', linewidth=2.0)
    
    plt.title('Vehicle Yaw Angle', fontsize=16, fontweight='bold')
    plt.xlabel('Frames', fontsize=12)
    plt.ylabel('Yaw Angle (Degrees)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=12)
    plt.tight_layout()

    plt.savefig(output_image_path)
    plt.close() 
    print(f"图像已成功保存至: {output_image_path}")

if __name__ == '__main__':
    os.makedirs('./results', exist_ok=True)

    plot_yaw_data('./results/pose_data_1.csv', './results/exp_result_raw_1.png')
    plot_yaw_data('./results/pose_data_2.csv', './results/exp_result_raw_2.png')