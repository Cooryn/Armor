import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class PolarEKF:
    def __init__(self):
        # [xc, vxc, yc, vyc, zc, vzc, body_yaw, w, r]^T
        self.X = np.zeros((9, 1))
        
        self.F = np.eye(9)
        self.P = np.eye(9) * 10.0
        self.P[8, 8] = 0.01
        
        # 过程噪声 Q
        self.Q = np.eye(9) * 0.01
        self.Q[1, 1] = 0.1  # vxc
        self.Q[3, 3] = 0.1  # vyc
        self.Q[5, 5] = 0.1  # vzc
        self.Q[7, 7] = 0.5  # w
        self.Q[8, 8] = 0.0001 # r
        
        # [target_yaw, target_pitch, distance, armor_orientation_yaw]
        self.R = np.diag([0.005, 0.005, 0.05, 0.05])
        
        self.is_initialized = False

    def predict(self, dt):
        self.F[0, 1] = dt  # xc += vxc * dt
        self.F[2, 3] = dt  # yc += vyc * dt
        self.F[4, 5] = dt  # zc += vzc * dt
        self.F[6, 7] = dt  # body_yaw += w * dt
        
        self.X = np.dot(self.F, self.X)
        self.X[6, 0] = wrap_to_pi(self.X[6, 0])
        
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def h(self, X_state, plate_idx):
        xc, yc, zc = X_state[0, 0], X_state[2, 0], X_state[4, 0]
        body_yaw, r = X_state[6, 0], X_state[8, 0]

        # plate_idx 可以是 -1, 0, 1, 2，代表与车头的 90 度倍数偏差
        current_plate_yaw = body_yaw + plate_idx * (np.pi / 2.0)
        
        # 用这块板子的角度去推算它的 3D 坐标
        xa = xc - r * np.sin(current_plate_yaw)
        za = zc - r * np.cos(current_plate_yaw)
        ya = yc 
        
        target_yaw = np.arctan2(xa, za)
        distance = np.sqrt(xa**2 + ya**2 + za**2)
        target_pitch = np.arctan2(ya, np.sqrt(xa**2 + za**2))
        
        return np.array([
            [wrap_to_pi(target_yaw)],
            [wrap_to_pi(target_pitch)],
            [distance],
            [wrap_to_pi(current_plate_yaw)]
        ])

    def get_jacobian(self, X_state, plate_idx):
        H = np.zeros((4, 9))
        eps = 1e-5
        Z_base = self.h(X_state, plate_idx)
        
        for i in range(9):
            X_eps = X_state.copy()
            X_eps[i, 0] += eps
            Z_eps = self.h(X_eps, plate_idx)
            
            diff = Z_eps - Z_base
            diff[0, 0] = wrap_to_pi(diff[0, 0])
            diff[1, 0] = wrap_to_pi(diff[1, 0])
            diff[3, 0] = wrap_to_pi(diff[3, 0])
            
            H[:, i] = (diff / eps).flatten()
        return H

    def update(self, Z_obs):
        obs_armor_yaw = Z_obs[3, 0]
        pred_body_yaw = self.X[6, 0]

        # 用相机看到的板子角度，减去我们预测的车头角度
        yaw_diff = wrap_to_pi(obs_armor_yaw - pred_body_yaw)
        
        # 将角度差除以 90 度并四舍五入。
        plate_idx = round(yaw_diff / (np.pi / 2.0))
        
        # 带着推断出来的板子编号，去计算预测观测量和雅可比矩阵
        Z_pred = self.h(self.X, plate_idx)
        H = self.get_jacobian(self.X, plate_idx)
        
        # 残差计算
        Y = Z_obs - Z_pred
        Y[0, 0] = wrap_to_pi(Y[0, 0])
        Y[1, 0] = wrap_to_pi(Y[1, 0])
        Y[3, 0] = wrap_to_pi(Y[3, 0])
        
        # 卡尔曼增益与更新
        S = np.dot(np.dot(H, self.P), H.T) + self.R
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
        
        # 因为残差是基于对应的装甲板算出来的，它会平滑地去修正车头 yaw
        self.X = self.X + np.dot(K, Y)
        # 注意：半径 r 理论上应当始终为正；如果此处变为负值说明滤波器已发散，应检查初始化和噪声参数
        self.X[6, 0] = wrap_to_pi(self.X[6, 0]) 
        
        I = np.eye(9)
        self.P = np.dot((I - np.dot(K, H)), self.P)

def run_predict_polar(csv_input_path, output_dir, suffix="1"):
    if not os.path.exists(csv_input_path):
        print(f"错误: 找不到输入文件 {csv_input_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    data = pd.read_csv(csv_input_path)
    
    ekf = PolarEKF()
    results = []
    last_timestamp = None

    for frame_id, group in data.groupby('frame_id'):
        # 取距离最近的装甲板进行观测
        obs = group.sort_values(by='distance').iloc[0]
        
        Z_obs = np.array([
            [obs['target_yaw']],
            [obs['target_pitch']],
            [obs['distance']],
            [obs['armor_orientation_yaw']]
        ])
        
        current_timestamp = obs['timestamp']
        if last_timestamp is None:
            dt = 1.0 / 30.0
        else:
            dt = (current_timestamp - last_timestamp) / 1000.0
            if dt <= 0: dt = 1.0 / 30.0
        last_timestamp = current_timestamp

        # 1. 第一帧冷启动初始化
        if not ekf.is_initialized:
            obs_yaw = obs['armor_orientation_yaw']
            r_init = 0.26 # 初始装甲板半径先验
            
            # 从相机的相对坐标反推车体旋转中心 (xc, zc)
            xc_init = obs['x'] + r_init * np.sin(obs_yaw)
            zc_init = obs['z'] + r_init * np.cos(obs_yaw)
            
            ekf.X = np.array([
                [xc_init], [0], [obs['y']], [0], [zc_init], [0], 
                [obs_yaw], [0], [r_init]
            ])
            ekf.is_initialized = True
            continue

        # 2. 预测步骤
        ekf.predict(dt)
        
        # 3. 更新步骤 (不需要再在外面写判断跳变的逻辑了)
        ekf.update(Z_obs)
        
        # 4. 记录误差用于绘图
        # 为了计算残差，我们需要再次推断当前的 plate_idx
        yaw_diff = wrap_to_pi(obs['armor_orientation_yaw'] - ekf.X[6, 0])
        plate_idx = round(yaw_diff / (np.pi / 2.0))
        
        Z_pred_final = ekf.h(ekf.X, plate_idx)
        
        error_target_yaw = wrap_to_pi(Z_pred_final[0, 0] - obs['target_yaw'])
        error_target_pitch = wrap_to_pi(Z_pred_final[1, 0] - obs['target_pitch'])
        error_distance = Z_pred_final[2, 0] - obs['distance']
        error_armor_yaw = wrap_to_pi(Z_pred_final[3, 0] - obs['armor_orientation_yaw'])

        results.append({
            'frame_id': frame_id,
            
            'xc': ekf.X[0, 0], 'vxc': ekf.X[1, 0],
            'yc': ekf.X[2, 0], 'vyc': ekf.X[3, 0],
            'zc': ekf.X[4, 0], 'vzc': ekf.X[5, 0],
            'body_yaw': ekf.X[6, 0], 'w': ekf.X[7, 0], 'r': ekf.X[8, 0],
            
            'err_target_yaw': error_target_yaw, 'err_target_pitch': error_target_pitch,
            'err_distance': error_distance, 'err_armor_yaw': error_armor_yaw,
            'obs_armor_yaw': obs['armor_orientation_yaw']
        })

    res_df = pd.DataFrame(results)
    
    # 1. 导出 CSV
    csv_out_path = os.path.join(output_dir, f'polar_prediction_result_{suffix}.csv')
    res_df.to_csv(csv_out_path, index=False)
    print(f"生成预测数据: {csv_out_path}")

    # 2. 导出 polar_rmse_result.txt
    rmse_tyaw = np.sqrt(np.mean(res_df['err_target_yaw']**2))
    rmse_tpitch = np.sqrt(np.mean(res_df['err_target_pitch']**2))
    rmse_dist = np.sqrt(np.mean(res_df['err_distance']**2))
    rmse_ayaw = np.sqrt(np.mean(res_df['err_armor_yaw']**2))

    txt_out_path = os.path.join(output_dir, f'polar_rmse_result_{suffix}.txt')
    with open(txt_out_path, 'w') as f:
        f.write(f"RMSE_target_yaw: {rmse_tyaw:.6f} rad\n")
        f.write(f"RMSE_target_pitch: {rmse_tpitch:.6f} rad\n")
        f.write(f"RMSE_distance: {rmse_dist:.6f} m\n")
        f.write(f"RMSE_armor_yaw: {rmse_ayaw:.6f} rad\n")
    print(f"生成误差统计: {txt_out_path}")

    # 3. 导出 polar_prediction_curve.png
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

if __name__ == '__main__':
    suffix = "2"

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_csv = os.path.join(base_dir, 'data', f'pose_raw_{suffix}.csv')
    output_directory = os.path.join(base_dir, 'results')
    run_predict_polar(input_csv, output_directory, suffix)