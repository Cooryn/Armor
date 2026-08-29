import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

FPS = 30.0  # 视频帧率，与 main.cpp 的 VideoWriter 一致

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class ArmorEKF:
    def __init__(self):
        # 🌟 11维状态量: [xc, vxc, yc, vyc, zc, vzc, body_yaw, w, r, dl, dh]^T
        self.X = np.zeros((11, 1))

        self.F = np.eye(11)
        self.P = np.eye(11) * 10.0
        # 物理结构参数初始协方差
        self.P[8, 8] = 0.01   # r
        self.P[9, 9] = 0.05   # dl
        self.P[10, 10] = 0.05 # dh

        # 过程噪声谱密度（单位：方差/秒），按 dt 动态构建 Q
        self.q_pos = 3.0     # 位置-速度对 (xc/vxc, yc/vyc, zc/vzc)
        self.q_yaw = 15.0    # 偏航角-角速度对 (body_yaw, w)
        self.q_r = 3e-4      # 底盘半径 r（几乎不变）
        self.q_dl = 3e-3     # 侧向偏移 dl
        self.q_dh = 3e-3     # 高度偏移 dh

        # 观测量: [target_yaw, target_pitch, distance, armor_orientation_yaw]
        self.R = np.diag([0.005, 0.005, 0.05, 0.05])

        self.is_initialized = False

    def predict(self, dt):
        """1. 建立车体中心运动模型（过程噪声按 dt 缩放）"""
        self.F[0, 1] = dt  # xc += vxc * dt
        self.F[2, 3] = dt  # yc += vyc * dt
        self.F[4, 5] = dt  # zc += vzc * dt
        self.F[6, 7] = dt  # body_yaw += w * dt

        self.X = np.dot(self.F, self.X)
        self.X[6, 0] = wrap_to_pi(self.X[6, 0])

        # 按 dt 构建离散化过程噪声 Q
        dt2 = dt * dt
        dt3 = dt2 * dt
        Q = np.zeros((11, 11))

        for i, j in [(0, 1), (2, 3), (4, 5)]:
            Q[i, i] = dt3 / 3.0 * self.q_pos
            Q[i, j] = dt2 / 2.0 * self.q_pos
            Q[j, i] = dt2 / 2.0 * self.q_pos
            Q[j, j] = dt * self.q_pos

        # body_yaw - w 对
        Q[6, 6] = dt3 / 3.0 * self.q_yaw
        Q[6, 7] = dt2 / 2.0 * self.q_yaw
        Q[7, 6] = dt2 / 2.0 * self.q_yaw
        Q[7, 7] = dt * self.q_yaw

        # 结构参数（无动力学，微小随机游走）
        Q[8, 8] = self.q_r * dt
        Q[9, 9] = self.q_dl * dt
        Q[10, 10] = self.q_dh * dt

        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + Q

    def h(self, X_state, armor_id):
        """2. 建立四块装甲板与车体中心之间的几何关系"""
        xc, yc, zc = X_state[0, 0], X_state[2, 0], X_state[4, 0]
        body_yaw = X_state[6, 0]
        r, dl, dh = X_state[8, 0], X_state[9, 0], X_state[10, 0]

        current_plate_yaw = body_yaw + armor_id * (np.pi / 2.0)

        is_side = (armor_id % 2 != 0)
        r_i = r + dl if is_side else r
        y_i = yc + dh if is_side else yc

        xa = xc - r_i * np.sin(current_plate_yaw)
        za = zc - r_i * np.cos(current_plate_yaw)
        ya = y_i

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

    def find_best_armor_id(self, Z_obs):
        """多重假设检验：对4块候选装甲板分别计算加权残差，返回最优ID（不修改状态）"""
        best_id = 0
        min_error = float('inf')
        R_diag = np.diag(self.R).reshape(4, 1)

        for i in range(4):
            Z_pred_i = self.h(self.X, i)
            Y_i = Z_obs - Z_pred_i
            Y_i[0, 0] = wrap_to_pi(Y_i[0, 0])
            Y_i[1, 0] = wrap_to_pi(Y_i[1, 0])
            Y_i[3, 0] = wrap_to_pi(Y_i[3, 0])
            error_i = np.sum((Y_i ** 2) / R_diag)

            if error_i < min_error:
                min_error = error_i
                best_id = i

        return best_id

    def update(self, Z_obs, armor_id=None):
        """
        卡尔曼更新步骤。
        若 armor_id 为 None，则自动做多重假设检验选择最优装甲板。
        使用 Joseph 形式保证 P 的对称正定性。
        """
        if armor_id is None:
            armor_id = self.find_best_armor_id(Z_obs)

        Z_pred = self.h(self.X, armor_id)
        Y = Z_obs - Z_pred
        Y[0, 0] = wrap_to_pi(Y[0, 0])
        Y[1, 0] = wrap_to_pi(Y[1, 0])
        Y[3, 0] = wrap_to_pi(Y[3, 0])

        H = self.get_jacobian(self.X, armor_id)
        S = np.dot(np.dot(H, self.P), H.T) + self.R
        K = np.linalg.solve(S, np.dot(H, self.P)).T

        self.X = self.X + np.dot(K, Y)
        self.X[6, 0] = wrap_to_pi(self.X[6, 0])
        self.X[8, 0] = np.clip(self.X[8, 0], 0.20, 0.30)  # r: 20~30cm

        # Joseph 形式
        I = np.eye(11)
        I_KH = I - np.dot(K, H)
        self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) + np.dot(np.dot(K, self.R), K.T)

        return armor_id

    def update_multi(self, observations):
        """
        使用同一帧内的多个观测顺序更新状态。
        observations: list of dict, 每个 dict 含:
          - 'Z_obs': (4,1) 观测向量
          - 'armor_id': int (可选，未指定则自动做多重假设检验选择最优装甲板)
        按距离升序处理。
        """
        sorted_obs = sorted(observations, key=lambda o: o['Z_obs'][2, 0])

        for obs in sorted_obs:
            Z_obs = obs['Z_obs']
            armor_id = obs.get('armor_id')

            if armor_id is None:
                armor_id = self.find_best_armor_id(Z_obs)

            Z_pred = self.h(self.X, armor_id)
            Y = Z_obs - Z_pred
            Y[0, 0] = wrap_to_pi(Y[0, 0])
            Y[1, 0] = wrap_to_pi(Y[1, 0])
            Y[3, 0] = wrap_to_pi(Y[3, 0])

            H = self.get_jacobian(self.X, armor_id)
            S = np.dot(np.dot(H, self.P), H.T) + self.R
            K = np.linalg.solve(S, np.dot(H, self.P)).T

            self.X = self.X + np.dot(K, Y)
            self.X[6, 0] = wrap_to_pi(self.X[6, 0])
            self.X[8, 0] = np.clip(self.X[8, 0], 0.20, 0.30)  # r: 20~30cm

            # Joseph 形式
            I = np.eye(11)
            I_KH = I - np.dot(K, H)
            self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) + np.dot(np.dot(K, self.R), K.T)


def run_predict_armor(csv_input_path, output_dir, suffix="1"):
    if not os.path.exists(csv_input_path):
        print(f"错误: 找不到输入文件 {csv_input_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    data = pd.read_csv(csv_input_path)

    ekf = ArmorEKF()
    results = []
    obs_trajectory = []      # 每帧最近装甲板的观测位置 (x, z)，用于俯视轨迹图
    last_frame_id = None
    first_frame_data = None  # 用于估算初始角速度
    skip_predict = False     # 初始化帧已做过 predict，跳过重复

    for frame_id, group in data.groupby('frame_id'):
        # 🌟 收集该帧所有装甲板检测
        all_obs = []
        for _, row in group.iterrows():
            Z = np.array([
                [row['target_yaw']], [row['target_pitch']],
                [row['distance']], [row['armor_orientation_yaw']]
            ])
            all_obs.append({'Z_obs': Z, 'armor_id': None})

        # 取距离最近的检测用于误差计算和时间戳
        closest = group.sort_values(by='distance').iloc[0]
        obs_trajectory.append((closest['x'], closest['z']))
        Z_closest = np.array([
            [closest['target_yaw']], [closest['target_pitch']],
            [closest['distance']], [closest['armor_orientation_yaw']]
        ])

        current_timestamp = closest['timestamp']

        # 计算动态 dt：用帧号差 / 帧率（main.cpp 的 timestamp 是处理耗时，不可靠）
        if last_frame_id is None:
            dt = 1.0 / FPS
        else:
            dt = (frame_id - last_frame_id) / FPS
            if dt <= 0:
                dt = 1.0 / FPS
        last_frame_id = frame_id

        # 第一帧：暂存数据，等第二帧算出角速度再初始化
        if first_frame_data is None:
            first_frame_data = {'closest': closest}
            continue

        # 第二帧：用两帧的 armor_orientation_yaw 估算初始角速度
        if not ekf.is_initialized:
            obs_yaw = first_frame_data['closest']['armor_orientation_yaw']
            obs_yaw2 = closest['armor_orientation_yaw']
            dt_init = dt  # 已在上面从帧号差算出
            if dt_init <= 0:
                dt_init = 1.0 / FPS

            # 用两帧 yaw 差估算角速度（处理角度环绕）
            w_init = wrap_to_pi(obs_yaw2 - obs_yaw) / dt_init

            r_init = 0.26
            first = first_frame_data['closest']
            xc_init = first['x'] + r_init * np.sin(obs_yaw)
            zc_init = first['z'] + r_init * np.cos(obs_yaw)

            # [xc, vxc, yc, vyc, zc, vzc, body_yaw, w, r, dl, dh]
            ekf.X = np.array([
                [xc_init], [0], [first['y']], [0], [zc_init], [0],
                [obs_yaw], [w_init], [r_init], [0], [0]
            ])
            ekf.is_initialized = True
            skip_predict = True  # 已在上方 predict，避免重复
            # predict: 从第一帧时刻推进到当前（第二帧）时刻
            ekf.predict(dt)
            # fall through to update with second frame observation

        # 卡尔曼
        if skip_predict:
            skip_predict = False
        else:
            ekf.predict(dt)

        # 🌟 用先验（预测）状态计算误差——反映真实的一步预测精度
        best_id = ekf.find_best_armor_id(Z_closest)
        Z_pred = ekf.h(ekf.X, best_id)

        error_target_yaw = wrap_to_pi(Z_pred[0, 0] - closest['target_yaw'])
        error_target_pitch = wrap_to_pi(Z_pred[1, 0] - closest['target_pitch'])
        error_distance = Z_pred[2, 0] - closest['distance']
        error_armor_yaw = wrap_to_pi(Z_pred[3, 0] - closest['armor_orientation_yaw'])

        # 为绘图记录当前装甲板的3D坐标（用先验状态）
        is_side = (best_id % 2 != 0)
        r_i = ekf.X[8, 0] + ekf.X[9, 0] if is_side else ekf.X[8, 0]
        xa_pred = ekf.X[0, 0] - r_i * np.sin(Z_pred[3, 0])
        za_pred = ekf.X[4, 0] - r_i * np.cos(Z_pred[3, 0])

        # 🌟 用该帧所有检测更新状态
        ekf.update_multi(all_obs)

        results.append({
            'frame_id': frame_id, 'timestamp': current_timestamp,
            'xc': ekf.X[0, 0], 'yc': ekf.X[2, 0], 'zc': ekf.X[4, 0],
            'xa': xa_pred, 'za': za_pred, 'armor_id': best_id,
            'body_yaw': ekf.X[6, 0], 'pred_armor_yaw': Z_pred[3, 0],
            'obs_armor_yaw': closest['armor_orientation_yaw'],
            'err_target_yaw': error_target_yaw, 'err_target_pitch': error_target_pitch,
            'err_distance': error_distance, 'err_armor_yaw': error_armor_yaw,
            'r': ekf.X[8, 0], 'dl': ekf.X[9, 0], 'dh': ekf.X[10, 0],
        })

    res_df = pd.DataFrame(results)

    # ==========================================
    # 1. 导出预测结果 CSV
    # ==========================================
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

    # ==========================================
    # 3. 俯视图：装甲板位置（x-z 平面），观测 vs 预测
    #    横轴 X（左右偏移），纵轴 Z（深度），越往上越远
    # ==========================================
    obs_traj = np.array(obs_trajectory)  # (N, 2): [x, z]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    fig.suptitle('Top-down Trajectory (X-Z plane)', fontsize=14, fontweight='bold')

    ax.plot(obs_traj[:, 0], obs_traj[:, 1], 'o-', color='#1f77b4',
            linewidth=1.5, markersize=3, label='Observed', alpha=0.8)
    ax.plot(res_df['xa'], res_df['za'], 'x-', color='#d62728',
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

    # 4. 输出统一车体 yaw 曲线
    fig_byaw, ax_byaw = plt.subplots(figsize=(10, 4), dpi=150)
    ax_byaw.plot(res_df['frame_id'], res_df['body_yaw'], color='#2ca02c')
    ax_byaw.set_title('Body Yaw Curve')
    ax_byaw.set_xlabel('Frame ID'); ax_byaw.set_ylabel('Body Yaw (rad)')
    ax_byaw.grid(True, linestyle='--')
    fig_byaw.savefig(os.path.join(output_dir, f'body_yaw_curve_{suffix}.png'))
    plt.close(fig_byaw)

    # 5. 输出 Folded Armor Yaw (观测板向 vs 预测板向)
    fig_folded, ax_folded = plt.subplots(figsize=(10, 4), dpi=150)
    ax_folded.scatter(res_df['frame_id'], res_df['obs_armor_yaw'], color='red', s=5, alpha=0.5, label='Observed')
    ax_folded.plot(res_df['frame_id'], res_df['pred_armor_yaw'], color='blue', linewidth=1.5, label='Predicted (Folded)')
    ax_folded.set_title('Folded Armor Yaw Tracking')
    ax_folded.set_xlabel('Frame ID'); ax_folded.set_ylabel('Armor Yaw (rad)')
    ax_folded.legend(); ax_folded.grid(True, linestyle='--')
    fig_folded.savefig(os.path.join(output_dir, f'folded_armor_yaw_curve_{suffix}.png'))
    plt.close(fig_folded)

    # 6. 输出残差误差四宫格
    fig_err, axs_err = plt.subplots(2, 2, figsize=(12, 8), sharex=True, dpi=150)
    fig_err.suptitle('Armor Prediction Observation Errors', fontsize=16)
    axs_err[0, 0].plot(res_df['frame_id'], res_df['err_target_yaw'], color='red')
    axs_err[0, 0].set_title(f'Target Yaw Error')
    axs_err[0, 1].plot(res_df['frame_id'], res_df['err_target_pitch'], color='blue')
    axs_err[0, 1].set_title(f'Target Pitch Error')
    axs_err[1, 0].plot(res_df['frame_id'], res_df['err_distance'], color='green')
    axs_err[1, 0].set_title(f'Distance Error')
    axs_err[1, 1].plot(res_df['frame_id'], res_df['err_armor_yaw'], color='purple')
    axs_err[1, 1].set_title(f'Armor Yaw Error')
    for ax in axs_err.flat:
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_err.savefig(os.path.join(output_dir, f'armor_prediction_error_curve_{suffix}.png'))
    plt.close(fig_err)

    print("所有文件已保存至：", output_dir)


if __name__ == '__main__':
    # 🌟 统一的切换开关
    suffix = "2"

    input_csv = os.path.join('./data', f'pose_raw_{suffix}.csv')
    output_directory = './results'
    run_predict_armor(input_csv, output_directory, suffix)
