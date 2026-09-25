"""Regression checks for observation-specific covariance and compatibility."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from predictor.predictor_armor import ArmorEKF
from predictor.run_armor import run_predict_armor
from tests.test_armor_tracking import measurement


class ObservationNoiseTests(unittest.TestCase):
    def tracker(self, adaptive=True):
        tracker = ArmorEKF(adaptive_noise=adaptive)
        tracker.initialize(measurement(0))
        tracker.predict(.03)
        return tracker

    def test_missing_invalid_quality_and_disable_preserve_base_noise(self):
        tracker = self.tracker()
        for metadata in ({}, {'detection_score': None, 'reprojection_error': 'bad'},
                         {'detection_score': 2, 'reprojection_error': -1},
                         {'detection_score': np.inf, 'reprojection_error': np.nan}):
            R, scales = tracker.observation_noise(dict(Z_obs=measurement(.5), **metadata))
            np.testing.assert_array_equal(R, tracker.R)
            np.testing.assert_array_equal(scales, np.ones(4))
        a, b = self.tracker(), self.tracker(False)
        for tracker in (a, b):
            tracker.update_multi([{'Z_obs': measurement(.03)}])
        np.testing.assert_allclose(a.X, b.X, atol=1e-12)
        np.testing.assert_allclose(a.P, b.P, atol=1e-12)
        R, _ = b.observation_noise(dict(Z_obs=measurement(.7), detection_score=.1, reprojection_error=9))
        np.testing.assert_array_equal(R, b.R)

    def test_bounded_variances_and_line_of_sight_sign(self):
        tracker = self.tracker()
        z = measurement(0)
        z[0, 0] = .4
        z[3, 0] = -.4  # Face-on relative to the viewing ray.
        face, face_scales = tracker.observation_noise(dict(Z_obs=z, detection_score=1, reprojection_error=0))
        np.testing.assert_allclose(face, tracker.R)
        z[3, 0] += np.pi/2
        side, side_scales = tracker.observation_noise(dict(Z_obs=z, detection_score=1, reprojection_error=0))
        self.assertGreater(side[3, 3], face[3, 3])
        self.assertEqual(side[0, 0], face[0, 0])
        R, scales = tracker.observation_noise(dict(Z_obs=z, detection_score=0, reprojection_error=1e6))
        self.assertTrue(np.all(scales >= 1))
        self.assertTrue(np.all(scales <= [3, 3, 4.5, 6]))
        self.assertGreater(np.linalg.eigvalsh(R).min(), 0)

    def test_lower_quality_has_less_influence_and_does_not_bypass_range_gate(self):
        a, b = self.tracker(), self.tracker()
        z = measurement(0)
        z[2, 0] += .15
        prior = a.X.copy()
        a.update_multi([dict(Z_obs=z, armor_id=0, detection_score=1, reprojection_error=0)])
        b.update_multi([dict(Z_obs=z, armor_id=0, detection_score=.2, reprojection_error=3)])
        self.assertLess(np.linalg.norm(b.X-prior), np.linalg.norm(a.X-prior))
        z[2, 0] += 2
        prior = b.X.copy()
        matches, diagnostics = b.update_multi([dict(Z_obs=z, detection_score=0, reprojection_error=100)])
        self.assertFalse(matches)
        self.assertEqual(diagnostics[0]['reason'], 'innovation_gate')
        np.testing.assert_array_equal(prior, b.X)

    def test_conflicting_low_quality_observation_not_favored_by_small_nis(self):
        tracker = self.tracker()
        z = measurement(0)
        matches, _ = tracker.associate([
            dict(Z_obs=z, armor_id=0, detection_score=.1, reprojection_error=3),
            dict(Z_obs=z, armor_id=0, detection_score=1, reprojection_error=0)])
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]['index'], 1)

    def test_heterogeneous_stacked_update_matches_independent_equations(self):
        a, b = self.tracker(), self.tracker()
        obs = [dict(Z_obs=measurement(.02), armor_id=0, detection_score=.4, reprojection_error=1.5),
               dict(Z_obs=measurement(.02, 1), armor_id=1, detection_score=.95, reprojection_error=.1)]
        matches, diagnostics = a.associate(obs)
        self.assertEqual(len(matches), 2)
        self.assertNotEqual(diagnostics[0]['distance_noise_scale'], diagnostics[1]['distance_noise_scale'])
        H = np.vstack([m['H'] for m in matches])
        residual = np.vstack([m['residual'] for m in matches])
        R = np.zeros((8, 8))
        for i, m in enumerate(matches):
            R[4*i:4*i+4, 4*i:4*i+4] = m['R']
        K = np.linalg.solve(H @ a.P @ H.T + R, H @ a.P).T
        expected = a.X + K @ residual
        a.update_multi(obs)
        b.update_multi(obs[::-1])
        np.testing.assert_allclose(a.X, expected, atol=1e-10)
        np.testing.assert_allclose(a.X, b.X, atol=1e-10)
        np.testing.assert_allclose(a.P, b.P, atol=1e-10)
        self.assertGreater(np.linalg.eigvalsh(a.P).min(), 0)

    def test_csv_quality_reaches_filter_and_diagnostic_export(self):
        z = measurement(.3)[:, 0]
        rows = [dict(frame_id=i, timestamp=i*1000/30, target_yaw=z[0], target_pitch=z[1],
                     distance=z[2], armor_orientation_yaw=z[3], detection_score=.4,
                     reprojection_error=1.5) for i in range(3)]
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'raw.csv'
            pd.DataFrame(rows).to_csv(source, index=False)
            for mode in (True, False):
                run_predict_armor(source, folder, adaptive_noise=mode)
                d = pd.read_csv(Path(folder)/'armor_observation_diagnostics_1.csv')
                scales = d.loc[d.reason == 'accepted', 'yaw_noise_scale']
                self.assertEqual(len(scales), 2)
                self.assertTrue((scales > 1).all() if mode else (scales == 1).all())


if __name__ == '__main__':
    unittest.main()
