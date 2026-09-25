"""11-state armor EKF. No file I/O or plotting dependencies."""
import numpy as np

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class ArmorEKF:
    def __init__(self, nis_gate=16.0, pair_yaw_tolerance=np.deg2rad(25),
                 max_distance_error=0.5, adaptive_noise=True):
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
        self.adaptive_noise = adaptive_noise

        self.nis_gate = nis_gate
        self.pair_yaw_tolerance = pair_yaw_tolerance
        self.max_distance_error = max_distance_error
        self.is_initialized = False

    def predict(self, dt):
        """1. 建立车体中心运动模型（过程噪声按 dt 缩放）"""
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and nonnegative")
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

    @staticmethod
    def valid_observation(z):
        return (z.shape == (4, 1) and np.isfinite(z).all()
                and z[2, 0] > 0 and abs(z[1, 0]) < np.pi / 2
                and abs(z[0, 0]) < np.pi / 2)

    def initialize(self, z):
        if not self.valid_observation(z):
            raise ValueError("Invalid initialization observation")
        yaw, pitch, distance, plate_yaw = z[:, 0]
        x = distance * np.cos(pitch) * np.sin(yaw)
        y = distance * np.sin(pitch)
        zc = distance * np.cos(pitch) * np.cos(yaw)
        radius = 0.26
        self.X[:, 0] = [x - radius * np.sin(plate_yaw), 0, y, 0,
                        zc + radius * np.cos(plate_yaw), 0,
                        plate_yaw, 0, radius, 0, 0]
        # Do not estimate angular speed from two potentially different plates.
        self.P = np.diag([.1, 1., .1, 1., .1, 1., .05, 100., .01, .01, .01])
        self.is_initialized = True

    def observation_noise(self, obs):
        """Bounded variance inflation from optional detector quality metadata.

        These are heuristic variance factors, not calibrated probabilities.
        Missing/invalid quality metadata preserves the legacy noise matrix.
        The horizontal incidence angle is plate_yaw + target_yaw because the
        solver's yaw sign is opposite the camera Y-axis rotation convention.
        """
        scales = np.ones(4)
        if not self.adaptive_noise:
            return self.R.copy(), scales

        def finite_number(key):
            try:
                value = float(obs.get(key, np.nan))
                return value if np.isfinite(value) else np.nan
            except (TypeError, ValueError):
                return np.nan

        score = finite_number('detection_score')
        error = finite_number('reprojection_error')
        score_valid = np.isfinite(score) and 0 <= score <= 1
        error_valid = np.isfinite(error) and error >= 0
        if not (score_valid or error_valid):
            return self.R.copy(), scales
        quality = 1.0 + ((1-score)**2 if score_valid else 0)
        quality *= 1.0 + ((min(error, 4.0)/2.0)**2 if error_valid else 0)
        scales[:] = min(quality, 3.0)
        z = np.asarray(obs['Z_obs'], dtype=float)
        grazing = np.sin(z[3, 0] + z[0, 0])**2
        scales[2] *= 1.0 + .5*grazing
        scales[3] *= 1.0 + grazing
        # D R D also preserves positive definiteness for a correlated base R.
        D = np.diag(np.sqrt(scales))
        return D @ self.R @ D, scales

    def innovation(self, z, armor_id, R=None):
        prediction = self.h(self.X, armor_id)
        residual = self.residual(z, prediction)
        H = self.get_jacobian(self.X, armor_id)
        S = H @ self.P @ H.T + (self.R if R is None else R)
        nis = float((residual.T @ np.linalg.solve(S, residual)).item())
        return nis, residual, H, prediction

    def find_best_armor_id(self, Z_obs):
        """Return None when every hypothesis fails the observation gates."""
        matches, _ = self.associate([{'Z_obs': Z_obs}])
        return matches[0]['armor_id'] if matches else None

    def associate(self, observations):
        """Joint assignment against one prior; unique IDs and consistent yaw gaps.

        Prefer the largest consistent set, then NIS plus a noise-volume penalty.
        Absolute range gating prevents a large covariance from admitting gross
        depth errors. Thresholds are configurable, not ground-truth guarantees.
        """
        candidates, diagnostics = [], []
        for index, obs in enumerate(observations):
            z = np.asarray(obs['Z_obs'], dtype=float)
            diagnostic = dict(observation_index=index, accepted=False,
                              armor_id=-1, nis=np.nan, reason='invalid',
                              best_candidate_id=-1, distance_residual=np.nan)
            options = []
            if self.valid_observation(z):
                R, scales = self.observation_noise({**obs, 'Z_obs': z})
                diagnostic.update(target_yaw_noise_scale=scales[0],
                                  target_pitch_noise_scale=scales[1],
                                  distance_noise_scale=scales[2], yaw_noise_scale=scales[3])
                ids = range(4) if obs.get('armor_id') is None else [obs['armor_id']]
                for armor_id in ids:
                    if armor_id not in range(4):
                        continue
                    nis, residual, H, predicted = self.innovation(z, armor_id, R)
                    base_S = H @ self.P @ H.T + self.R
                    S = H @ self.P @ H.T + R
                    # Inflating R reduces NIS. Include the covariance-volume
                    # penalty when choosing between conflicting observations.
                    # Subtract the base volume to retain legacy scores at R=R0.
                    cost = nis + np.linalg.slogdet(S)[1] - np.linalg.slogdet(base_S)[1]
                    if not np.isfinite(diagnostic['nis']) or nis < diagnostic['nis']:
                        diagnostic['nis'] = nis
                        diagnostic['best_candidate_id'] = armor_id
                        diagnostic['distance_residual'] = residual[2, 0]
                    if (nis <= self.nis_gate
                            and abs(residual[2, 0]) <= self.max_distance_error):
                        options.append(dict(index=index, armor_id=armor_id, nis=nis,
                                            residual=residual, H=H, predicted=predicted,
                                            Z_obs=z, R=R, association_cost=cost))
                diagnostic['reason'] = 'association_conflict' if options else 'innovation_gate'
            candidates.append(options)
            diagnostics.append(diagnostic)

        best, best_score = [], float('inf')
        def search(index, selected, used, score):
            nonlocal best, best_score
            if len(selected) + min(4 - len(used), len(candidates) - index) < len(best):
                return
            if index == len(candidates):
                if len(selected) > len(best) or (len(selected) == len(best) and score < best_score):
                    best, best_score = selected.copy(), score
                return
            for candidate in candidates[index]:
                aid = candidate['armor_id']
                if aid in used:
                    continue
                consistent = True
                for other in selected:
                    measured_gap = candidate['Z_obs'][3, 0] - other['Z_obs'][3, 0]
                    expected_gap = (aid - other['armor_id']) * np.pi / 2
                    if abs(wrap_to_pi(measured_gap - expected_gap)) > self.pair_yaw_tolerance:
                        consistent = False
                        break
                if consistent:
                    search(index + 1, selected + [candidate], used | {aid},
                           score + candidate['association_cost'])
            search(index + 1, selected, used, score)
        search(0, [], set(), 0.)
        for match in best:
            diagnostics[match['index']].update(accepted=True, armor_id=match['armor_id'],
                                               nis=match['nis'], reason='accepted',
                                               distance_residual=match['residual'][2, 0])
        return best, diagnostics

    def update_multi(self, observations):
        matches, diagnostics = self.associate(observations)
        if not matches:
            return matches, diagnostics  # Keep the predicted state AND covariance.
        # One stacked update: all associations, Jacobians and gates use the same prior.
        H = np.vstack([m['H'] for m in matches])
        residual = np.vstack([m['residual'] for m in matches])
        R = np.zeros((4*len(matches), 4*len(matches)))
        for i, match in enumerate(matches):
            R[4*i:4*i+4, 4*i:4*i+4] = match['R']
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T
        self.X += K @ residual
        self.X[6, 0] = wrap_to_pi(self.X[6, 0])
        self.X[8, 0] = np.clip(self.X[8, 0], .20, .30)
        # Both alternating radii must remain physically positive.
        side_radius = np.clip(self.X[8, 0] + self.X[9, 0], .20, .30)
        self.X[9, 0] = side_radius - self.X[8, 0]
        I_KH = np.eye(11) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = (self.P + self.P.T) / 2
        return matches, diagnostics

    def update(self, Z_obs, armor_id=None):
        matches, _ = self.update_multi([dict(Z_obs=Z_obs, armor_id=armor_id)])
        return matches[0]['armor_id'] if matches else None
