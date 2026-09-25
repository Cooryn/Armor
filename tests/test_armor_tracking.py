import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from predictor.predictor_armor import ArmorEKF, wrap_to_pi
from predictor.run_armor import run_predict_armor
from plot.video import compute_plate_position


def measurement(theta, armor_id=0):
    angle = theta + armor_id * np.pi / 2
    x, y, z = .1 + .26 * np.sin(angle), .2, 3 - .26 * np.cos(angle)
    return np.array([[np.arctan2(x, z)], [np.arctan2(y, np.hypot(x, z))],
                     [np.linalg.norm([x, y, z])], [wrap_to_pi(angle)]])


class ArmorTrackingTests(unittest.TestCase):
    def tracker(self):
        ekf = ArmorEKF()
        ekf.initialize(measurement(0))
        return ekf

    def test_pnp_yaw_sign_and_initialization(self):
        # Solver yaw for a pure camera-Y rotation is the negative rotation angle.
        rotation, _ = cv2.Rodrigues(np.array([0., -.4, 0.]))
        yaw = np.arctan2(rotation[2, 0], rotation[2, 2])
        position = np.array([.1, .2, 3.]) - .26 * rotation[:, 2]
        z = np.array([[np.arctan2(position[0], position[2])],
                      [np.arctan2(position[1], np.hypot(position[0], position[2]))],
                      [np.linalg.norm(position)], [yaw]])
        ekf = ArmorEKF()
        ekf.initialize(z)
        np.testing.assert_allclose(ekf.X[[0, 2, 4], 0], [.1, .2, 3.], atol=1e-12)
        np.testing.assert_allclose(ekf.h(ekf.X, 0), z, atol=1e-12)
        np.testing.assert_allclose(compute_plate_position(.1, .2, 3, yaw, .26, 0, 0, 0), position)

    def test_reject_outlier_without_changing_prior(self):
        ekf = self.tracker()
        ekf.predict(.1)
        x, p = ekf.X.copy(), ekf.P.copy()
        z = measurement(0)
        z[2, 0] += 2
        self.assertIsNone(ekf.update(z))
        np.testing.assert_array_equal(ekf.X, x)
        np.testing.assert_array_equal(ekf.P, p)

    def test_mahalanobis_gate_without_range_outlier(self):
        ekf = self.tracker()
        ekf.P *= .001
        z = measurement(0)
        z[1, 0] += .8
        self.assertIsNone(ekf.update(z))

    def test_unique_ids_and_pair_geometry(self):
        ekf = self.tracker()
        obs = [{'Z_obs': measurement(0)}, {'Z_obs': measurement(0)},
               {'Z_obs': measurement(0, 1)}]
        matches, diagnostics = ekf.update_multi(obs)
        self.assertEqual(len(matches), 2)
        self.assertEqual({m['armor_id'] for m in matches}, {0, 1})
        self.assertEqual(sum(d['accepted'] for d in diagnostics), 2)
        ekf = self.tracker()
        ekf.P *= 100  # Individually permissive gates must not bypass pair geometry.
        bad = measurement(0, 1)
        bad[3, 0] = 0
        matches, _ = ekf.associate([{'Z_obs': measurement(0), 'armor_id': 0},
                                  {'Z_obs': bad, 'armor_id': 1}])
        self.assertEqual(len(matches), 1)

    def test_order_independent_stacked_update(self):
        a, b = self.tracker(), self.tracker()
        obs = [{'Z_obs': measurement(.02)}, {'Z_obs': measurement(.02, 1)}]
        a.update_multi(obs)
        b.update_multi(obs[::-1])
        np.testing.assert_allclose(a.X, b.X, atol=1e-10)
        np.testing.assert_allclose(a.P, b.P, atol=1e-10)

    def test_rotation_switching_dropout_and_covariance(self):
        ekf = self.tracker()
        ekf.X[7, 0] = 2.
        for frame in range(1, 201):
            ekf.predict(.02)
            theta = 2 * frame * .02
            if not 60 <= frame < 80:
                # Select a physically front-facing plate; this naturally switches IDs.
                aid = min(range(4), key=lambda i: abs(wrap_to_pi(theta + i*np.pi/2)))
                self.assertEqual(ekf.update(measurement(theta, aid)), aid)
            else:
                ekf.update_multi([])
            self.assertGreater(np.linalg.eigvalsh(ekf.P).min(), -1e-10)
        np.testing.assert_allclose(ekf.X[[0, 2, 4], 0], [.1, .2, 3.], atol=1e-5)
        self.assertAlmostEqual(ekf.X[7, 0], 2., places=5)

    def test_invalid_observations(self):
        ekf = self.tracker()
        for z in (np.full((4, 1), np.nan), np.zeros((4, 1)), np.zeros((3, 1))):
            self.assertIsNone(ekf.update(z))

    def test_csv_gap_and_rejection_export(self):
        rows = []
        for frame, time in [(0, 0), (1, 100), (4, 400), (5, 500)]:
            z = measurement(0)
            if frame == 5:
                z[2, 0] += 2
            rows.append(dict(frame_id=frame, timestamp=time, x=.1, y=.2, z=2.74,
                             target_yaw=z[0, 0], target_pitch=z[1, 0],
                             distance=z[2, 0], armor_orientation_yaw=z[3, 0]))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'input.csv'
            pd.DataFrame(rows).to_csv(path, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_predict_armor(path, root)
            self.assertEqual(result.frame_id.tolist(), [1, 2, 3, 4, 5])
            self.assertEqual(result.timestamp.tolist(), [100, 200, 300, 400, 500])
            for frame in (2, 3, 5):
                row = result[result.frame_id == frame].iloc[0]
                self.assertEqual(row.status, 'prediction_only')
                self.assertTrue(np.isnan(row.err_distance))
            diagnostics = pd.read_csv(Path(root) / 'armor_observation_diagnostics_1.csv')
            self.assertFalse(diagnostics[diagnostics.frame_id == 5].accepted.iloc[0])


if __name__ == '__main__':
    unittest.main()
