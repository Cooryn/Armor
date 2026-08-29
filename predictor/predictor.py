import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

FPS = 30.0  # 视频帧率，与 main.cpp 的 VideoWriter 一致

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
        self.q = 0.01                # 加速度噪声方差 (m/s^2)^2；本数据目标近乎静止，取小值更平滑
        self.Q = np.zeros((6, 6))    # 占位，predict 时按标准 CV 模型重建
        self.R = np.eye(3) * 0.1     # 观测噪声；一步预测误差指标下，偏大 R 平滑后误差更小
        self.is_initialized = False

    def predict(self, dt):
        self.F[0, 1] = dt
        self.F[2, 3] = dt
        self.F[4, 5] = dt

        # 标准 CV（白噪声加速度）模型的过程噪声，随 dt 变化：
        # 每个轴 [x, vx] 分块 Q_axis = q * [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        block = np.array([[dt4 / 4.0, dt3 / 2.0],
                          [dt3 / 2.0, dt2]]) * self.q
        self.Q[:] = 0.0
        self.Q[0:2, 0:2] = block
        self.Q[2:4, 2:4] = block
        self.Q[4:6, 4:6] = block

        predicted_state = np.dot(self.F, self.state)
        predicted_P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return predicted_state, predicted_P

    def update(self, Z, predicted_state, predicted_P):
        S = np.dot(np.dot(self.H, predicted_P), self.H.T) + self.R
        # K = P H^T S^-1，等价于解 S K^T = H P，避免直接求逆
        K = np.linalg.solve(S, np.dot(self.H, predicted_P)).T
        y = Z - np.dot(self.H, predicted_state)
        self.state = predicted_state + np.dot(K, y)
        I = np.eye(6)
        self.P = np.dot((I - np.dot(K, self.H)), predicted_P)

def run_predict(csv_input_path, output_dir, suffix="1"):
    if not os.path.exists(csv_input_path):
        print(f"错误: 找不到输入文件 {csv_input_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    data = pd.read_csv(csv_input_path)

    predictor = BasicPredictor()
    results = []
    obs_trajectory = []  # 每帧小车中心点二维位置 (x, z)，用于俯视轨迹图
    last_frame_id = None

    # 按帧遍历数据
    for frame_id, group in data.groupby('frame_id'):
        # 基础处理：如果画面有多块装甲板，只取距离最近的一块作为追踪目标
        closest_armor = group.sort_values(by='distance').iloc[0]

        observed_x = closest_armor['x']
        observed_y = closest_armor['y']
        observed_z = closest_armor['z']
        observed_yaw = closest_armor['target_yaw']
        observed_distance = closest_armor['distance']

        obs_trajectory.append((observed_x, observed_z))

        Z = np.array([[observed_x], [observed_y], [observed_z]])

        # 计算动态 dt：用帧号差 / 帧率（main.cpp 的 timestamp 是处理耗时，不可靠）
        if last_frame_id is None:
            dt = 1.0 / FPS
        else:
            dt = (frame_id - last_frame_id) / FPS
            if dt <= 0:
                dt = 1.0 / FPS
        last_frame_id = frame_id

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

if __name__ == '__main__':
    suffix = "2"

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_csv = os.path.join(base_dir, 'data', f'pose_raw_{suffix}.csv')
    output_directory = os.path.join(base_dir, 'results')
    run_predict(input_csv, output_directory, suffix)