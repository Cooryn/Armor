"""11-state armor EKF. No file I/O or plotting dependencies."""
import numpy as np

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class ArmorEKF:
    def __init__(self, max_yaw_error=np.deg2rad(45), pair_yaw_tolerance=np.deg2rad(25),
                 max_distance_error=0.5):
        # 🌟 11维状态量: [xc, vxc, yc, vyc, zc, vzc, body_yaw, w, r, dl, dh]^T
        self.X = np.zeros((11, 1))

        self.F = np.eye(11)
        self.P = np.eye(11) * 10.0
        # 物理结构参数初始协方差
        self.P[8, 8] = 0.01
        self.P[9, 9] = 0.05
        self.P[10, 10] = 0.05

        # 过程噪声谱密度（单位：方差/秒），按 dt 动态构建 Q
        self.q_pos = 3.0     # 位置-速度对 (xc/vxc, yc/vyc, zc/vzc)
        self.q_yaw = 15.0    # 偏航角-角速度对 (body_yaw, w)
        self.q_r = 3e-4      # 底盘半径 r（几乎不变）
        self.q_dl = 3e-3     # 侧向偏移 dl
        self.q_dh = 3e-3     # 高度偏移 dh

        # 观测量: [target_yaw, target_pitch, distance, armor_orientation_yaw]
        self.R = np.diag([0.0016, 0.0016, 0.16, 0.0576])

        self.max_yaw_error = max_yaw_error
        self.pair_yaw_tolerance = pair_yaw_tolerance
        self.max_distance_error = max_distance_error
        self.is_initialized = False

    def predict(self, dt):
        """1. 建立车体中心运动模型（过程噪声按 dt 缩放）"""
        self.F, Q = self._transition(dt)
        self.X = self.F @ self.X
        self.X[6, 0] = wrap_to_pi(self.X[6, 0])
        self.P = self.F @ self.P @ self.F.T + Q

    def _transition(self, dt):
        """Return transition and process noise without modifying the tracker."""
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and nonnegative")
        F = np.eye(11)
        F[0, 1] = F[2, 3] = F[4, 5] = F[6, 7] = dt

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

        return F, Q

    def forecast(self, horizon_s=0.05):
        """Predict from the current state, leaving X, P and F unchanged.

        Returns state (11x1), covariance (11x11), and four plate poses (4x4,
        columns x/y/z/yaw). The caller owns the timestamp; horizon is seconds.
        Poses use the existing yaw-only model in the same coordinate frame.
        """
        if not self.is_initialized:
            raise ValueError('Cannot forecast before initialization')
        F, Q = self._transition(horizon_s)
        state = F @ self.X
        state[6, 0] = wrap_to_pi(state[6, 0])
        covariance = F @ self.P @ F.T + Q
        poses = []
        for aid in range(4):
            yaw = wrap_to_pi(state[6, 0] + aid*np.pi/2)
            radius = state[8, 0] + (state[9, 0] if aid % 2 else 0)
            poses.append([state[0, 0] + radius*np.sin(yaw),
                          state[2, 0] + (state[10, 0] if aid % 2 else 0),
                          state[4, 0] - radius*np.cos(yaw), yaw])
        return dict(state=state, covariance=covariance, plates=np.array(poses))

    def h(self, X_state, armor_id):
        """2. 建立四块装甲板与车体中心之间的几何关系"""
        xc, yc, zc = X_state[0, 0], X_state[2, 0], X_state[4, 0]
        body_yaw = X_state[6, 0]
        r, dl, dh = X_state[8, 0], X_state[9, 0], X_state[10, 0]

        current_plate_yaw = body_yaw + armor_id * (np.pi / 2.0)

        is_side = (armor_id % 2 != 0)
        r_i = r + dl if is_side else r
        y_i = yc + dh if is_side else yc

        xa = xc + r_i * np.sin(current_plate_yaw)
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

    @staticmethod
    def residual(observed, predicted):
        residual = observed - predicted
        residual[[0, 1, 3], 0] = wrap_to_pi(residual[[0, 1, 3], 0])
        return residual

    def initialize(self, z):
        yaw, pitch, distance, plate_yaw = z[:, 0]
        x = distance * np.cos(pitch) * np.sin(yaw)
        y = distance * np.sin(pitch)
        zc = distance * np.cos(pitch) * np.cos(yaw)
        radius = 0.26
        self.X[:, 0] = [x - radius * np.sin(plate_yaw), 0, y, 0,
                        zc + radius * np.cos(plate_yaw), 0,
                        plate_yaw, 0, radius, 0, 0]
        self.P = np.diag([.1, 1., .1, 1., .1, 1., .05, 100., .01, .01, .01])
        self.is_initialized = True

    def associate(self, observations):
        candidates, diagnostics = [], []
        for index, obs in enumerate(observations):
            z = np.asarray(obs['Z_obs'], dtype=float)
            yaw, pitch, distance, plate_yaw = z[:, 0]
            position = distance * np.array([np.cos(pitch)*np.sin(yaw), np.sin(pitch), np.cos(pitch)*np.cos(yaw)])
            diagnostics.append(dict(observation_index=index, accepted=False, armor_id=-1,
                                    nis=np.nan, reason='geometry_gate', best_candidate_id=-1,
                                    distance_residual=np.nan))
            ids = range(4) if obs.get('armor_id') in (None, -1) else [obs['armor_id']]
            for armor_id in ids:
                if armor_id not in range(4):
                    continue
                predicted = self.h(self.X, armor_id)
                pyaw, ppitch, pdistance = predicted[:3, 0]
                point = pdistance * np.array([np.cos(ppitch)*np.sin(pyaw), np.sin(ppitch), np.cos(ppitch)*np.cos(pyaw)])
                error = np.linalg.norm(position - point)
                if error <= self.max_distance_error and abs(wrap_to_pi(plate_yaw-predicted[3, 0])) <= self.max_yaw_error:
                    candidates.append(dict(index=index, armor_id=armor_id, distance=error,
                                           residual=self.residual(z, predicted), predicted=predicted, Z_obs=z))
                    diagnostics[index]['reason'] = 'association_conflict'
        matches = []
        for candidate in sorted(candidates, key=lambda m: m['distance']):
            if any(candidate['index'] == m['index'] or candidate['armor_id'] == m['armor_id'] or
                   abs(wrap_to_pi(candidate['Z_obs'][3, 0] - m['Z_obs'][3, 0] -
                                 (candidate['armor_id'] - m['armor_id'])*np.pi/2)) > self.pair_yaw_tolerance
                   for m in matches):
                continue
            matches.append(candidate)
            diagnostics[candidate['index']].update(accepted=True, armor_id=candidate['armor_id'], reason='accepted')
            if len(matches) == 4:
                break
        return matches, diagnostics

    def update_multi(self, observations):
        matches, diagnostics = self.associate(observations)
        if not matches:
            return matches, diagnostics
        H = np.vstack([self.get_jacobian(self.X, m['armor_id']) for m in matches])
        residual = np.vstack([m['residual'] for m in matches])
        R = np.zeros((4*len(matches), 4*len(matches)))
        for i in range(len(matches)):
            R[4*i:4*i+4, 4*i:4*i+4] = self.R
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T
        self.X += K @ residual
        self.X[6, 0] = wrap_to_pi(self.X[6, 0])
        self.X[8, 0] = np.clip(self.X[8, 0], .20, .30)
        side_radius = np.clip(self.X[8, 0] + self.X[9, 0], .20, .30)
        self.X[9, 0] = side_radius - self.X[8, 0]
        I_KH = np.eye(11) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = (self.P + self.P.T) / 2
        return matches, diagnostics

    def update(self, Z_obs, armor_id=None):
        matches, _ = self.update_multi([dict(Z_obs=Z_obs, armor_id=armor_id)])
        return matches[0]['armor_id'] if matches else None
