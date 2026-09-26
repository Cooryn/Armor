import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import pandas as pd
from tests.reference.predictor.predictor import run_predict
from tests.reference.predictor.predictor_polar import run_predict_polar, PolarEKF
from tests.reference.predictor.run_armor import run_predict_armor
from plot.video import project, CAMERA_MATRIX, DISTORTION

COLUMNS = "frame_id timestamp x y z target_yaw target_pitch distance armor_orientation_yaw".split()

class RegressionTests(unittest.TestCase):
    def test_empty_and_single_frame(self):
        with tempfile.TemporaryDirectory() as root:
            csv = Path(root) / "input.csv"
            for rows in ([], [[0, 0, 0, 0, 3, 0, 0, 3, 0]]):
                pd.DataFrame(rows, columns=COLUMNS).to_csv(csv, index=False)
                for run in (run_predict, run_predict_polar, run_predict_armor):
                    with contextlib.redirect_stdout(io.StringIO()):
                        run(csv, root)
                self.assertEqual(list(Path(root).glob("*result*")), [])

    def test_projection_includes_distortion(self):
        expected, _ = cv2.projectPoints(np.array([[1., .5, 3.]]), np.zeros(3), np.zeros(3), CAMERA_MATRIX, DISTORTION)
        self.assertEqual(project(1., .5, 3.), tuple(expected.ravel().astype(int)))
        self.assertIsNone(project(0, 0, -1))

    def test_polar_error_is_prior(self):
        with tempfile.TemporaryDirectory() as root:
            csv = Path(root) / "input.csv"
            pd.DataFrame([[0, 0, 0, 0, 3, 0, 0, 3, 0], [1, 100, 0, 0, 4, 0, 0, 4, 0]], columns=COLUMNS).to_csv(csv, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                run_predict_polar(csv, root)
            result = pd.read_csv(Path(root) / "polar_prediction_result_1.csv")
            self.assertAlmostEqual(result.err_distance.iloc[0], -1.)

    def test_elapsed_time_uses_timestamps(self):
        from tests.reference.predictor.predictor import BasicPredictor
        calls = []
        original = BasicPredictor.predict
        def record(obj, dt):
            calls.append(dt)
            return original(obj, dt)
        with tempfile.TemporaryDirectory() as root:
            csv = Path(root) / "input.csv"
            pd.DataFrame([[0, 0, 0, 0, 3, 0, 0, 3, 0], [1, 100, 0, 0, 3, 0, 0, 3, 0]], columns=COLUMNS).to_csv(csv, index=False)
            with patch.object(BasicPredictor, "predict", record), contextlib.redirect_stdout(io.StringIO()):
                run_predict(csv, root)
        self.assertEqual(calls, [.1])

if __name__ == "__main__":
    unittest.main()
