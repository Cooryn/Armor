"""Future forecasting must be causal and must not advance the live tracker."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from tests.reference.predictor.predictor_armor import ArmorEKF, wrap_to_pi
from tests.reference.predictor.run_armor import run_predict_armor
from tests.test_armor_tracking import measurement


class ForecastTests(unittest.TestCase):
    def tracker(self):
        ekf = ArmorEKF()
        ekf.initialize(measurement(.2))
        ekf.X[[1, 3, 5, 7], 0] = [1., -.2, .3, 8.]
        ekf.X[6, 0] = np.pi-.1
        ekf.X[9, 0], ekf.X[10, 0] = .02, .04
        return ekf

    def test_forecast_four_plates_translation_rotation_and_no_mutation(self):
        ekf = self.tracker()
        X, P, F = ekf.X.copy(), ekf.P.copy(), ekf.F.copy()
        result = ekf.forecast(.05)
        state = result['state']
        np.testing.assert_allclose(state[[0, 2, 4], 0], X[[0, 2, 4], 0]+.05*X[[1, 3, 5], 0])
        self.assertAlmostEqual(state[6, 0], wrap_to_pi(X[6, 0]+8*.05))
        for aid, pose in enumerate(result['plates']):
            yaw = state[6, 0] + aid * np.pi / 2
            radius = state[8, 0] + (state[9, 0] if aid % 2 else 0)
            expected = [state[0, 0] + radius * np.sin(yaw),
                        state[2, 0] + (state[10, 0] if aid % 2 else 0),
                        state[4, 0] - radius * np.cos(yaw)]
            np.testing.assert_allclose(pose[:3], expected)
            self.assertAlmostEqual(pose[3], wrap_to_pi(state[6, 0]+aid*np.pi/2))
        for horizon in (.1, .05, 0.):
            ekf.forecast(horizon)
        np.testing.assert_array_equal(ekf.X, X)
        np.testing.assert_array_equal(ekf.P, P)
        np.testing.assert_array_equal(ekf.F, F)
        reference = self.tracker()
        reference.predict(.05)
        np.testing.assert_allclose(result['state'], reference.X)
        np.testing.assert_allclose(result['covariance'], reference.P)
        self.assertGreater(np.linalg.eigvalsh(result['covariance']).min(), 0)
        result['state'][:] = 0
        result['covariance'][:] = 0
        np.testing.assert_array_equal(ekf.X, X)
        np.testing.assert_array_equal(ekf.P, P)

    def test_zero_invalid_and_uninitialized_horizons(self):
        ekf = self.tracker()
        np.testing.assert_allclose(ekf.forecast(0)['state'], ekf.X)
        np.testing.assert_allclose(ekf.forecast(0)['covariance'], ekf.P)
        for value in (-.05, np.nan, np.inf):
            with self.assertRaises(ValueError):
                ekf.forecast(value)
            with self.assertRaises(ValueError):
                run_predict_armor('unused.csv', 'unused', prediction_horizon_ms=value)
        with self.assertRaises(ValueError):
            ArmorEKF().forecast()

    def test_runner_time_units_dropouts_and_live_state_independence(self):
        rows = []
        for frame in [0, 1, 2, 4, 5]:
            z = measurement(frame*.1)[:, 0]
            rows.append(dict(frame_id=frame,timestamp=frame*1000/30,
                target_yaw=z[0],target_pitch=z[1],distance=z[2],armor_orientation_yaw=z[3]))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root/'raw.csv'
            pd.DataFrame(rows).to_csv(raw,index=False)
            zero = run_predict_armor(raw,root/'zero',prediction_horizon_ms=0)
            lead = run_predict_armor(raw,root/'lead',prediction_horizon_ms=50)
            columns = [c for c in zero if not c.startswith('future_') and
                       c not in ('prediction_horizon_ms','prediction_timestamp')]
            pd.testing.assert_frame_equal(zero[columns],lead[columns])
            np.testing.assert_allclose(lead.prediction_timestamp-lead.timestamp,50)
            np.testing.assert_allclose(lead.future_xc,lead.xc+.05*lead.vxc)
            future = pd.read_csv(root/'lead/armor_future_prediction_1.csv')
            self.assertEqual(len(future),len(lead)*4)
            self.assertTrue((future.groupby('frame_id').armor_id.nunique()==4).all())
            self.assertTrue((future[future.frame_id==3].source_status=='prediction_only').all())
            np.testing.assert_allclose(future.prediction_timestamp-future.timestamp,50)
            self.assertGreater(future.prediction_timestamp.max(),lead.timestamp.max())

if __name__ == '__main__':
    unittest.main()
