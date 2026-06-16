import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 角度归一化到 [-pi, pi]
def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class BasicPredictor:
    def __init__(self):
        # state = [x, vx, y, vy, z, vz]
        self.state = np.zeros((6, 1))
        
        # 状态转移矩阵 F (匀速模型)
        self.F = np.eye(6)
        
        # 观测矩阵 H
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0]
        ])
        
        self.P = np.eye(6) * 10.0
        self.Q = np.eye(6) * 0.01  # 过程噪声
        self.R = np.eye(3) * 0.1   # 观测噪声
        self.is_initialized = False

    def predict(self, dt):
        self.F[0, 1] = dt
        self.F[2, 3] = dt
        self.F[4, 5] = dt
        
        predicted_state = np.dot(self.F, self.state)
        predicted_P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return predicted_state, predicted_P

    def update(self, Z, predicted_state, predicted_P):
        S = np.dot(np.dot(self.H, predicted_P), self.H.T) + self.R
        K = np.dot(np.dot(predicted_P, self.H.T), np.linalg.inv(S))
        y = Z - np.dot(self.H, predicted_state)
        self.state = predicted_state + np.dot(K, y)
        I = np.eye(6)
        self.P = np.dot((I - np.dot(K, self.H)), predicted_P)

def run_predict(csv_input_path, output_dir):
    if not os.path.exists(csv_input_path):
        print(f"错误: 找不到输入文件 {csv_input_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    data = pd.read_csv(csv_input_path)
    
    predictor = BasicPredictor()
    results = []
    last_timestamp = None

    # 按帧遍历数据
    for frame_id, group in data.groupby('frame_id'):
        # 基础处理：如果画面有多块装甲板，只取距离最近的一块作为追踪目标
        closest_armor = group.sort_values(by='distance').iloc[0]
        
        observed_x = closest_armor['x']
        observed_y = closest_armor['y']
        observed_z = closest_armor['z']
        observed_yaw = closest_armor['target_yaw']
        observed_distance = closest_armor['distance']
        
        Z = np.array([[observed_x], [observed_y], [observed_z]])
        
        # 计算动态 dt
        current_timestamp = closest_armor['timestamp']
        if last_timestamp is None:
            dt = 1.0 / 30.0
        else:
            dt = (current_timestamp - last_timestamp) / 1000.0
            if dt <= 0: dt = 1.0 / 30.0
        last_timestamp = current_timestamp

        # 初始化第一帧
        if not predictor.is_initialized:
            predictor.state = np.array([[observed_x], [0], [observed_y], [0], [observed_z], [0]])
            predictor.is_initialized = True
            continue

        
        # 1. 根据第 k 帧状态预测第 k+1 帧状态
        predicted_state, predicted_P = predictor.predict(dt)
        
        predicted_x = predicted_state[0, 0]
        predicted_y = predicted_state[2, 0]
        predicted_z = predicted_state[4, 0]
        
        # 根据推导出的三维位置计算偏航角和距离
        predicted_yaw = np.arctan2(predicted_x, predicted_z)
        predicted_distance = np.sqrt(predicted_x**2 + predicted_y**2 + predicted_z**2)

        # 2 & 4. 使用第 k+1 帧 PnP 观测进行比较，计算预测误差
        error_x = predicted_x - observed_x
        error_z = predicted_z - observed_z
        error_yaw = wrap_to_pi(predicted_yaw - observed_yaw) 
        error_distance = predicted_distance - observed_distance

        # 3. 输出一帧预测结果
        results.append({
            'frame_id': frame_id,
            'predicted_x': predicted_x, 'observed_x': observed_x, 'error_x': error_x,
            'predicted_z': predicted_z, 'observed_z': observed_z, 'error_z': error_z,
            'predicted_yaw': predicted_yaw, 'observed_yaw': observed_yaw, 'error_yaw': error_yaw,
            'predicted_distance': predicted_distance, 'observed_distance': observed_distance, 'error_distance': error_distance
        })

        # 更新滤波器
        predictor.update(Z, predicted_state, predicted_P)

    # ==========================================
    # 结果结算与导出
    # ==========================================
    res_df = pd.DataFrame(results)
    
    # 导出 CSV
    csv_out_path = os.path.join(output_dir, f'prediction_result_{suffix}.csv')
    res_df.to_csv(csv_out_path, index=False)
    print(f"已输出预测结果: {csv_out_path}")

    # 5. 计算 RMSE
    rmse_x = np.sqrt(np.mean(res_df['error_x']**2))
    rmse_z = np.sqrt(np.mean(res_df['error_z']**2))
    rmse_yaw = np.sqrt(np.mean(res_df['error_yaw']**2))
    rmse_distance = np.sqrt(np.mean(res_df['error_distance']**2))

    txt_out_path = os.path.join(output_dir, f'rmse_result_{suffix}.txt')
    with open(txt_out_path, 'w') as f:
        f.write(f"RMSE_x: {rmse_x:.6f} m\n")
        f.write(f"RMSE_z: {rmse_z:.6f} m\n")
        f.write(f"RMSE_yaw: {rmse_yaw:.6f} rad\n")
        f.write(f"RMSE_distance: {rmse_distance:.6f} m\n")
    print(f"已计算RMSE: {txt_out_path}")

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

if __name__ == '__main__':
    suffix = "2"
    
    input_csv = os.path.join('./data', f'pose_raw_{suffix}.csv')
    output_directory = './results'
    run_predict(input_csv, output_directory)