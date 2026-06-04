import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_yaw_data(csv_file_path, output_image_path):
    # 检查文件是否存在
    if not os.path.exists(csv_file_path):
        print(f"找不到数据文件: {csv_file_path}")
        return

    # 读取 CSV 数据
    data = pd.read_csv(csv_file_path)

    # 创建一个单子图画布 (分辨率150，尺寸10x6，比例更适合放进 PPT)
    plt.figure(figsize=(10, 6), dpi=150)

    # 绘制 Yaw 角数据，加粗一点线条 (linewidth=1.5) 以增强视觉冲击力
    plt.plot(data['Frame'], data['Yaw'], color='purple', label='Yaw Angle', linewidth=1.5)
    
    # 设置图表标题和坐标轴标签
    plt.title('Armor Yaw Angle over Frames (Delivery 5)', fontsize=16)
    plt.xlabel('Frames', fontsize=12)
    plt.ylabel('Yaw Angle (Degrees)', fontsize=12)
    
    # 增加虚线网格，方便直观读出 15° 的基准线
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 自动调整布局防重叠
    plt.tight_layout()

    # 保存图片
    plt.savefig(output_image_path)
    
    # 画完单图后必须关闭画布，释放内存，防止图表数据在下一个循环发生重叠
    plt.close() 
    print(f"Yaw 角数据可视化图表已成功保存至: {output_image_path}")

if __name__ == '__main__':

    csv_path_1 = './results/pose_data_1.csv'
    out_path_1 = './results/exp_result_improve_1.png' # 🚀 改为 improve_1
    plot_yaw_data(csv_path_1, out_path_1)

    csv_path_2 = './results/pose_data_2.csv'
    out_path_2 = './results/exp_result_improve_2.png' # 🚀 改为 improve_2
    plot_yaw_data(csv_path_2, out_path_2)