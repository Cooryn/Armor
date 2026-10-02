"""Independent NumPy reference for the six-state single-plate EKF."""
from pathlib import Path

import numpy as np
import pandas as pd

FPS = 30.0
INPUT_COLUMNS = ["frame_id", "timestamp", "x", "y", "z", "target_yaw", "target_pitch", "distance"]
ORIGIN_COLUMNS = ["base_reference_timestamp_ms", "base_origin_x_m", "base_origin_y_m", "base_origin_z_m"]


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class SinglePlateEKF:
    geometry_epsilon = 1e-6

    def __init__(self):
        self.state = np.zeros(6)
        self.P = np.eye(6) * 10.0
        self.R = np.diag([0.0016, 0.0016, 0.16])
        self.is_initialized = False

    @classmethod
    def valid_observation(cls, observation):
        z = np.asarray(observation)
        return (z.shape == (3,) and np.isfinite(z).all() and z[2] > cls.geometry_epsilon
                and abs(z[1]) < np.pi / 2 and z[2] * np.cos(z[1]) > cls.geometry_epsilon)

    @classmethod
    def valid_geometry(cls, state):
        return (np.isfinite(state).all() and np.hypot(state[0], state[4]) > cls.geometry_epsilon
                and np.hypot(np.hypot(state[0], state[4]), state[2]) > cls.geometry_epsilon)

    @staticmethod
    def h(state):
        x, y, z = state[::2]
        horizontal = np.hypot(x, z)
        return np.array([np.arctan2(x, z), np.arctan2(y, horizontal), np.hypot(horizontal, y)])

    @classmethod
    def jacobian(cls, state):
        if not cls.valid_geometry(state):
            raise ValueError("Singular single-plate observation geometry")
        x, y, z = state[::2]
        horizontal = np.hypot(x, z)
        distance = np.hypot(horizontal, y)
        # First differentiate in xyz, then place the columns into the six-state Jacobian.
        position_jacobian = np.array([
            [z / horizontal**2, 0.0, -x / horizontal**2],
            [-x * y / (horizontal * distance**2), horizontal / distance**2,
             -y * z / (horizontal * distance**2)],
            [x / distance, y / distance, z / distance],
        ])
        H = np.zeros((3, 6))
        H[:, ::2] = position_jacobian
        return H

    def initialize(self, observation):
        if not self.valid_observation(observation):
            raise ValueError("Invalid single-plate initialization observation")
        yaw, pitch, distance = observation
        horizontal = distance * np.cos(pitch)
        self.state = np.array([horizontal * np.sin(yaw), 0.0, distance * np.sin(pitch), 0.0,
                               horizontal * np.cos(yaw), 0.0])
        self.P = np.eye(6) * 10.0
        self.is_initialized = True

    def predict(self, dt):
        if not self.is_initialized or not np.isfinite(dt) or dt < 0:
            raise ValueError("Prediction requires initialization and finite nonnegative dt")
        F = np.eye(6)
        F[np.arange(0, 6, 2), np.arange(1, 6, 2)] = dt
        dt2 = dt * dt
        block = 0.01 * np.array([[dt2 * dt2 / 4, dt2 * dt / 2], [dt2 * dt / 2, dt2]])
        state = F @ self.state
        covariance = F @ self.P @ F.T + np.kron(np.eye(3), block)
        if not np.isfinite(state).all() or not np.isfinite(covariance).all():
            raise ValueError("Non-finite single-plate prediction")
        self.state, self.P = state, covariance

    def update(self, observation):
        if (not self.is_initialized or not self.valid_observation(observation)
                or not self.valid_geometry(self.state)):
            return False
        H = self.jacobian(self.state)
        residual = observation - self.h(self.state)
        residual[:2] = wrap_to_pi(residual[:2])
        S = H @ self.P @ H.T + self.R
        try:
            np.linalg.cholesky(S)
            K = np.linalg.solve(S, H @ self.P).T
        except np.linalg.LinAlgError:
            return False
        state = self.state + K @ residual
        A = np.eye(6) - K @ H
        covariance = A @ self.P @ A.T + K @ self.R @ K.T
        covariance = (covariance + covariance.T) / 2
        if not np.isfinite(state).all() or not np.isfinite(covariance).all():
            return False
        self.state, self.P = state, covariance
        return True


def run_predict(csv_input_path, output_dir, suffix="1"):
    data = pd.read_csv(csv_input_path)
    missing = set(INPUT_COLUMNS) - set(data)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    data[INPUT_COLUMNS] = data[INPUT_COLUMNS].apply(pd.to_numeric, errors="raise").astype(float)
    ids = data.frame_id.to_numpy()
    timestamps = data.timestamp.to_numpy()
    if (not np.isfinite(ids).all() or (ids < 0).any() or (ids != np.floor(ids)).any()
            or not np.isfinite(timestamps).all() or (timestamps < 0).any()):
        raise ValueError("Invalid frame_id or timestamp")
    base = False
    origin = {}
    if "coordinate_frame" in data and not data.empty:
        frames = set(data.coordinate_frame)
        if not frames <= {"camera", "base"} or len(frames) != 1:
            raise ValueError("Unknown or mixed input coordinate_frame")
        base = frames == {"base"}
        if base:
            if not set(ORIGIN_COLUMNS) <= set(data):
                raise ValueError("Missing base origin metadata")
            metadata = data[ORIGIN_COLUMNS].apply(pd.to_numeric, errors="raise").to_numpy()
            if (not np.isfinite(metadata).all() or not (metadata == metadata[0]).all()
                    or metadata[0, 0] < 0):
                raise ValueError("Missing or mixed base origin metadata")
            origin = dict(zip(ORIGIN_COLUMNS, metadata[0]))

    predictor = SinglePlateEKF()
    results = []
    last_timestamp = None
    for frame_id, group in data.groupby("frame_id"):
        valid = group[np.isfinite(group[["x", "y", "z"]]).all(axis=1)
                      & group.apply(lambda r: predictor.valid_observation(
                          r[["target_yaw", "target_pitch", "distance"]].to_numpy(dtype=float)), axis=1)]
        if valid.empty:
            continue
        row = valid.sort_values("distance", kind="stable").iloc[0]
        observation = row[["target_yaw", "target_pitch", "distance"]].to_numpy(dtype=float)
        dt = (row.timestamp - last_timestamp) / 1000 if last_timestamp is not None else 1 / FPS
        if dt <= 0:
            dt = 1 / FPS
        last_timestamp = row.timestamp
        if not predictor.is_initialized:
            predictor.initialize(observation)
            continue
        predictor.predict(dt)
        state = predictor.state
        predicted = predictor.h(state)
        results.append({
            "frame_id": frame_id,
            "predicted_x": state[0], "observed_x": row.x, "error_x": state[0] - row.x,
            "predicted_z": state[4], "observed_z": row.z, "error_z": state[4] - row.z,
            "predicted_yaw": predicted[0], "observed_yaw": observation[0],
            "error_yaw": wrap_to_pi(predicted[0] - observation[0]),
            "predicted_distance": predicted[2], "observed_distance": observation[2],
            "error_distance": predicted[2] - observation[2],
        })
        predictor.update(observation)

    if not results:
        print("No prediction exported: at least two detected frames are required.")
        return None
    result = pd.DataFrame(results)
    if base:
        result["coordinate_frame"] = "base"
        for name, value in origin.items():
            result[name] = value
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / f"prediction_result_{suffix}.csv"
    metrics_path = output / f"rmse_result_{suffix}.txt"
    result.to_csv(csv_path, index=False)
    with metrics_path.open("w", encoding="utf-8") as stream:
        for column, name, unit in zip(
                ["error_x", "error_z", "error_yaw", "error_distance"],
                ["RMSE_x", "RMSE_z", "RMSE_yaw", "RMSE_distance"], ["m", "m", "rad", "m"]):
            stream.write(f"{name}: {np.sqrt(np.mean(result[column]**2)):.6f} {unit}\n")
    print(f"已输出预测结果: {csv_path}")
    print(f"已计算RMSE: {metrics_path}")
    return result
