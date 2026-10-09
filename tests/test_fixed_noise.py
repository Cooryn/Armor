"""Fixed covariance, independent stacked-update equations and CSV compatibility."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tests.reference.predictor.predictor_armor import ArmorEKF
from tests.reference.predictor.run_armor import run_predict_armor
from tests.test_armor_tracking import measurement


class FixedNoiseTests(unittest.TestCase):
    def tracker(self):
        b = ArmorEKF()
        b.initialize(measurement(0))
        b.predict(.03)
        return b

    def test_quality_metadata_does_not_change_update_or_gate(self):
        a, b = self.tracker(), self.tracker()
        z = measurement(.03)
        a.update_multi([dict(Z_obs=z)])
        b.update_multi([dict(Z_obs=z, detection_score=0, reprojection_error=100)])
        np.testing.assert_array_equal(a.X, b.X)
        np.testing.assert_array_equal(a.P, b.P)
        z[2, 0] += 2
        prior, covariance = b.X.copy(), b.P.copy()
        matches, diagnostics = b.update_multi([dict(Z_obs=z, detection_score=0, reprojection_error=100)])
        self.assertFalse(matches)
        self.assertEqual(diagnostics[0]['reason'], 'innovation_gate')
        np.testing.assert_array_equal(prior, b.X)
        np.testing.assert_array_equal(covariance, b.P)

    def test_conflict_uses_smallest_nis(self):
        b = self.tracker()
        z = measurement(0)
        noisy = z.copy()
        noisy[2, 0] += .1
        matches, _ = b.associate([dict(Z_obs=noisy, armor_id=0), dict(Z_obs=z, armor_id=0)])
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]['index'], 1)

    def test_stacked_update_matches_independent_equations(self):
        a, b = self.tracker(), self.tracker()
        a.R[0, 3] = a.R[3, 0] = .0002
        b.R = a.R.copy()
        obs = [dict(Z_obs=measurement(.02), armor_id=0), dict(Z_obs=measurement(.02, 1), armor_id=1)]
        matches, _ = a.associate(obs)
        self.assertEqual(len(matches), 2)
        H = np.vstack([m['H'] for m in matches])
        residual = np.vstack([m['residual'] for m in matches])
        R = np.kron(np.eye(len(matches)), a.R)
        K = np.linalg.solve(H @ a.P @ H.T + R, H @ a.P).T
        expected = a.X + K @ residual
        I_KH = np.eye(11) - K @ H
        expected_covariance = I_KH @ a.P @ I_KH.T + K @ R @ K.T
        a.update_multi(obs)
        b.update_multi(obs[::-1])
        np.testing.assert_allclose(a.X, expected, atol=1e-10)
        np.testing.assert_allclose(a.P, expected_covariance, atol=1e-10)
        np.testing.assert_allclose(a.X, b.X, atol=1e-10)
        np.testing.assert_allclose(a.P, b.P, atol=1e-10)
        self.assertGreater(np.linalg.eigvalsh(a.P).min(), 0)

    def test_csv_extra_quality_columns_are_ignored(self):
        z = measurement(.3)[:, 0]
        rows = [dict(frame_id=i, timestamp=i*1000/30, target_yaw=z[0], target_pitch=z[1],
                     distance=z[2], armor_orientation_yaw=z[3]) for i in range(3)]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/'raw.csv'
            pd.DataFrame(rows).to_csv(source, index=False)
            clean = run_predict_armor(source, root/'clean')
            data = pd.DataFrame(rows)
            data['detection_score'], data['reprojection_error'] = 'invalid', 'invalid'
            data.to_csv(source, index=False)
            quality = run_predict_armor(source, root/'quality')
            pd.testing.assert_frame_equal(clean, quality)
            diagnostics = pd.read_csv(root/'quality/armor_observation_diagnostics_1.csv')
            self.assertEqual((diagnostics.reason == 'accepted').sum(), 2)
            self.assertFalse(any('noise_scale' in column for column in diagnostics))


if __name__ == '__main__':
    unittest.main()
