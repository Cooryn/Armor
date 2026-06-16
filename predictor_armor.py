import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class ArmorEKF:
    def __init__(self):
        # 🌟 11维状态量: [xc, vxc, yc, vyc, zc, vzc, body_yaw, w, r, dl, dh]^T
        self.X = np.zeros((11, 1))
        
        self.F = np.eye(11)
        self.P = np.eye(11) * 10.0
        # 物理结构参数极度自信，初始协方差给小点
        self.P[8, 8] = 0.01   # r
        self.P[9, 9] = 0.05   # dl
        self.P[10, 10] = 0.05 # dh
        
        # 过程噪声 Q
        self.Q = np.eye(11) * 0.01
        self.Q[1, 1] = 0.1   # vxc
        self.Q[3, 3] = 0.1   # vyc
        self.Q[5, 5] = 0.1   # vzc
        self.Q[7, 7] = 0.5   # w (角速度变化大)
        self.Q[8, 8] = 1e-5  # r (几乎不变)
        self.Q[9, 9] = 1e-4  # dl (微小自适应)
        self.Q[10, 10] = 1e-4 # dh (微小自适应)
        
        # 观测量: [target_yaw, target_pitch, distance, armor_orientation_yaw]
        self.R = np.diag([0.005, 0.005, 0.05, 0.05])
        
        self.is_initialized = False

    def predict(self, dt):
        """1. 建立车体中心运动模型"""
        self.F[0, 1] = dt  # xc += vxc * dt
        self.F[2, 3] = dt  # yc += vyc * dt
        self.F[4, 5] = dt  # zc += vzc * dt
        self.F[6, 7] = dt  # body_yaw += w * dt
        # r, dl, dh 的导数为 0，所以在转移矩阵中 F[i,i]=1 即可
        
        self.X = np.dot(self.F, self.X)
        self.X[6, 0] = wrap_to_pi(self.X[6, 0])
        
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def h(self, X_state, armor_id):
        """2. 建立四块装甲板与车体中心之间的几何关系"""
        xc, yc, zc = X_state[0, 0], X_state[2, 0], X_state[4, 0]
        body_yaw = X_state[6, 0]
        r, dl, dh = X_state[8, 0], X_state[9, 0], X_state[10, 0]

        # 计算理论朝向
        current_plate_yaw = body_yaw + armor_id * (np.pi / 2.0)
        
        # 🌟 核心：引入 dl 和 dh。判断是否为侧边装甲板 (id 1, 3)
        is_side = (armor_id % 2 != 0)
        
        # 半径补偿：侧板加 dl，高低补偿：侧板加 dh
        r_i = r + dl if is_side else r
        y_i = yc + dh if is_side else yc
        
        # 三维空间反推
        xa = xc - r_i * np.sin(current_plate_yaw)
        za = zc - r_i * np.cos(current_plate_yaw)
        ya = y_i 
        
        # 转回相机视角的极坐标
        target_yaw = np.arctan2(xa, za)
        distance = np.sqrt(xa**2 + ya**2 + za**2)
        target_pitch = np.arctan2(ya, np.sqrt(xa**2 + za**2))
        
        return np.array([
            [wrap_to_pi(target_yaw)],
            [wrap_to_pi(target_pitch)],
            [distance],
            [wrap_to_pi(current_plate_yaw)]
        ])

    def get_jacobian(self, X_state, armor_id):
        """有限差分计算 4x11 雅可比矩阵"""
        H = np.zeros((4, 11))
        eps = 1e-5
        Z_base = self.h(X_state, armor_id)
        
        for i in range(11):
            X_eps = X_state.copy()
            X_eps[i, 0] += eps
            Z_eps = self.h(X_eps, armor_id)
            
            diff = Z_eps - Z_base
            diff[0, 0] = wrap_to_pi(diff[0, 0])
            diff[1, 0] = wrap_to_pi(diff[1, 0])
            diff[3, 0] = wrap_to_pi(diff[3, 0])
            
            H[:, i] = (diff / eps).flatten()
        return H

    def update(self, Z_obs):
        """
        3 & 4. 预测器更新 (进阶方法：多重假设检验)
        """
        best_id = 0
        min_error = float('inf')
        best_Y = None
        
        # 提取观测噪声的方差，用于后续误差的量纲归一化
        R_diag = np.diag(self.R).reshape(4, 1)

        # 🌟 进阶逻辑：分别假设当前观测来自 0/1/2/3 四块装甲板
        for i in range(4):
            # 1. 假设是第 i 块装甲板，计算预测观测量
            Z_pred_i = self.h(self.X, i)
            
            # 2. 计算残差 Y
            Y_i = Z_obs - Z_pred_i
            Y_i[0, 0] = wrap_to_pi(Y_i[0, 0]) # target_yaw
            Y_i[1, 0] = wrap_to_pi(Y_i[1, 0]) # target_pitch
            Y_i[3, 0] = wrap_to_pi(Y_i[3, 0]) # armor_orientation_yaw
            
            # 3. 计算加权平方误差
            # 为什么除以 R_diag？因为角度(弧度)和距离(米)的单位不同。
            # 距离的误差可能是 0.1，角度的误差可能是 0.05。
            # 除以它们各自的传感器方差，能让它们在一个起跑线上公平比较（类似于马氏距离）。
            error_i = np.sum((Y_i ** 2) / R_diag)
            
            # 4. 记录误差最小的那一块
            if error_i < min_error:
                min_error = error_i
                best_id = i
                best_Y = Y_i

        # ==========================================
        # 带着挑选出来的【最优 armor_id】，去算雅可比矩阵并更新
        # ==========================================
        H = self.get_jacobian(self.X, best_id)
        
        S = np.dot(np.dot(H, self.P), H.T) + self.R
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
        
        # 状态修正
        self.X = self.X + np.dot(K, best_Y)
        self.X[6, 0] = wrap_to_pi(self.X[6, 0]) 
        
        I = np.eye(11)
        self.P = np.dot((I - np.dot(K, H)), self.P)
        
        return best_id

def run_predict_armor(csv_input_path, output_dir):
    if not os.path.exists(csv_input_path):
        print(f"错误: 找不到输入文件 {csv_input_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    data = pd.read_csv(csv_input_path)
    
    ekf = ArmorEKF()
    results = []
    last_timestamp = None

    for frame_id, group in data.groupby('frame_id'):
        obs = group.sort_values(by='distance').iloc[0]
        
        Z_obs = np.array([
            [obs['target_yaw']], [obs['target_pitch']],
            [obs['distance']], [obs['armor_orientation_yaw']]
        ])
        
        current_timestamp = obs['timestamp']
        if last_timestamp is None:
            dt = 1.0 / 30.0
        else:
            dt = (current_timestamp - last_timestamp) / 1000.0
            if dt <= 0: dt = 1.0 / 30.0
        last_timestamp = current_timestamp

        # 第一帧启动
        if not ekf.is_initialized:
            obs_yaw = obs['armor_orientation_yaw']
            r_init = 0.26
            
            xc_init = obs['x'] + r_init * np.sin(obs_yaw)
            zc_init = obs['z'] + r_init * np.cos(obs_yaw)
            
            # [xc, vxc, yc, vyc, zc, vzc, body_yaw, w, r, dl, dh]
            ekf.X = np.array([
                [xc_init], [0], [obs['y']], [0], [zc_init], [0], 
                [obs_yaw], [0], [r_init], [0], [0]
            ])
            ekf.is_initialized = True
            continue

        # 卡尔曼
        ekf.predict(dt)
        armor_id = ekf.update(Z_obs) 
        Z_pred_final = ekf.h(ekf.X, armor_id)
        
        # 记录误差用于图表输出
        error_target_yaw = wrap_to_pi(Z_pred_final[0, 0] - obs['target_yaw'])
        error_target_pitch = wrap_to_pi(Z_pred_final[1, 0] - obs['target_pitch'])
        error_distance = Z_pred_final[2, 0] - obs['distance']
        error_armor_yaw = wrap_to_pi(Z_pred_final[3, 0] - obs['armor_orientation_yaw'])
        
        # 为绘图记录当前装甲板的3D坐标
        is_side = (armor_id % 2 != 0)
        r_i = ekf.X[8,0] + ekf.X[9,0] if is_side else ekf.X[8,0]
        xa_pred = ekf.X[0,0] - r_i * np.sin(Z_pred_final[3, 0])
        za_pred = ekf.X[4,0] - r_i * np.cos(Z_pred_final[3, 0])

        results.append({
            'frame_id': frame_id, 'timestamp': current_timestamp,
            'xc': ekf.X[0, 0], 'zc': ekf.X[4, 0], 
            'xa': xa_pred, 'za': za_pred, 'armor_id': armor_id,
            'body_yaw': ekf.X[6, 0], 'pred_armor_yaw': Z_pred_final[3, 0],
            'obs_armor_yaw': obs['armor_orientation_yaw'],
            'err_target_yaw': error_target_yaw, 'err_target_pitch': error_target_pitch,
            'err_distance': error_distance, 'err_armor_yaw': error_armor_yaw
        })

    res_df = pd.DataFrame(results)
    
    # ==========================================
    # 严格对齐考核要求的 8 项产出物
    # ==========================================

    # 1. 导出预测结果 CSV
    csv_out_path = os.path.join(output_dir, f'armor_prediction_result_{suffix}.csv')
    res_df.to_csv(csv_out_path, index=False)

    # 2. 导出误差文本 RMSE TXT
    rmse_tyaw = np.sqrt(np.mean(res_df['err_target_yaw']**2))
    rmse_tpitch = np.sqrt(np.mean(res_df['err_target_pitch']**2))
    rmse_dist = np.sqrt(np.mean(res_df['err_distance']**2))
    rmse_ayaw = np.sqrt(np.mean(res_df['err_armor_yaw']**2))
    txt_out_path = os.path.join(output_dir, f'armor_rmse_result_{suffix}.txt')
    with open(txt_out_path, 'w') as f:
        f.write(f"RMSE_target_yaw: {rmse_tyaw:.6f} rad\n")
        f.write(f"RMSE_target_pitch: {rmse_tpitch:.6f} rad\n")
        f.write(f"RMSE_distance: {rmse_dist:.6f} m\n")
        f.write(f"RMSE_armor_yaw: {rmse_ayaw:.6f} rad\n")

    # 3. 输出车体中心轨迹图
    fig_center, ax_center = plt.subplots(figsize=(8, 8), dpi=150)
    ax_center.plot(res_df['xc'], res_df['zc'], color='black', alpha=0.5)
    scatter = ax_center.scatter(res_df['xc'], res_df['zc'], c=res_df['frame_id'], cmap='viridis', s=15)
    ax_center.set_title('Center Trajectory (xc, zc)')
    ax_center.set_xlabel('xc (m)'); ax_center.set_ylabel('zc (m)')
    ax_center.grid(True, linestyle='--')
    fig_center.colorbar(scatter, label='Frame ID')
    fig_center.savefig(os.path.join(output_dir, f'center_trajectory_curve_{suffix}.png'))
    plt.close(fig_center)

    # 4. 输出当前可见装甲板轨迹图 (按 ID 着色)
    fig_switch, ax_switch = plt.subplots(figsize=(8, 8), dpi=150)
    colors = ['red', 'blue', 'green', 'orange']
    for i in range(4):
        mask = res_df['armor_id'] == i
        ax_switch.scatter(res_df[mask]['xa'], res_df[mask]['za'], color=colors[i], label=f'Armor {i}', s=15, alpha=0.7)
    ax_switch.plot(res_df['xc'], res_df['zc'], color='black', linestyle='--', label='Center')
    ax_switch.set_title('Visible Armor Trajectory & Switching')
    ax_switch.set_xlabel('xa (m)'); ax_switch.set_ylabel('za (m)')
    ax_switch.legend(); ax_switch.grid(True, linestyle='--')
    fig_switch.savefig(os.path.join(output_dir, f'armor_switch_curve_{suffix}.png'))
    plt.close(fig_switch)

    # 5. 输出统一车体 yaw 曲线
    fig_byaw, ax_byaw = plt.subplots(figsize=(10, 4), dpi=150)
    ax_byaw.plot(res_df['frame_id'], res_df['body_yaw'], color='#2ca02c')
    ax_byaw.set_title('Body Yaw Curve')
    ax_byaw.set_xlabel('Frame ID'); ax_byaw.set_ylabel('Body Yaw (rad)')
    ax_byaw.grid(True, linestyle='--')
    fig_byaw.savefig(os.path.join(output_dir, f'body_yaw_curve_{suffix}.png'))
    plt.close(fig_byaw)

    # 6. 输出 Folded Armor Yaw (观测板向 vs 预测板向)
    fig_folded, ax_folded = plt.subplots(figsize=(10, 4), dpi=150)
    ax_folded.scatter(res_df['frame_id'], res_df['obs_armor_yaw'], color='red', s=5, alpha=0.5, label='Observed')
    ax_folded.plot(res_df['frame_id'], res_df['pred_armor_yaw'], color='blue', linewidth=1.5, label='Predicted (Folded)')
    ax_folded.set_title('Folded Armor Yaw Tracking')
    ax_folded.set_xlabel('Frame ID'); ax_folded.set_ylabel('Armor Yaw (rad)')
    ax_folded.legend(); ax_folded.grid(True, linestyle='--')
    fig_folded.savefig(os.path.join(output_dir, f'folded_armor_yaw_curve_{suffix}.png'))
    plt.close(fig_folded)

    # 7. 输出残差误差四宫格
    fig_err, axs_err = plt.subplots(2, 2, figsize=(12, 8), sharex=True, dpi=150)
    fig_err.suptitle('Armor Prediction Observation Errors', fontsize=16)
    axs_err[0,0].plot(res_df['frame_id'], res_df['err_target_yaw'], color='red')
    axs_err[0,0].set_title(f'Target Yaw Error')
    axs_err[0,1].plot(res_df['frame_id'], res_df['err_target_pitch'], color='blue')
    axs_err[0,1].set_title(f'Target Pitch Error')
    axs_err[1,0].plot(res_df['frame_id'], res_df['err_distance'], color='green')
    axs_err[1,0].set_title(f'Distance Error')
    axs_err[1,1].plot(res_df['frame_id'], res_df['err_armor_yaw'], color='purple')
    axs_err[1,1].set_title(f'Armor Yaw Error')
    for ax in axs_err.flat:
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_err.savefig(os.path.join(output_dir, f'armor_prediction_error_curve_{suffix}.png'))
    plt.close(fig_err)

    print("所有文件已保存至：", output_dir)

if __name__ == '__main__':
    # 🌟 统一的切换开关
    suffix = "1"
    
    input_csv = os.path.join('./data', f'pose_raw_{suffix}.csv') 
    output_directory = './results'
    run_predict_armor(input_csv, output_directory)