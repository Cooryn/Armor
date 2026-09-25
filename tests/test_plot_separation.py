import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import pandas as pd

from predictor.run_armor import run_predict_armor
from plot.armor import plot_armor


class PlotSeparationTests(unittest.TestCase):
    def test_core_import_needs_no_plotting_or_csv_library(self):
        root = Path(__file__).resolve().parents[1]
        code = (
            'import sys; from predictor.predictor_armor import ArmorEKF; '
            'assert "matplotlib" not in sys.modules; '
            'assert "pandas" not in sys.modules; '
            'assert "cv2" not in sys.modules; ArmorEKF()'
        )
        subprocess.run([sys.executable, '-B', '-c', code], cwd=root, check=True)

    def test_runner_exports_only_data_and_plotter_preserves_it(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / 'pose.csv'
            pd.DataFrame([
                dict(frame_id=i, timestamp=i*100, x=0, y=0, z=3,
                     target_yaw=0, target_pitch=0, distance=3, armor_orientation_yaw=0)
                for i in [0, 1, 3]
            ]).to_csv(raw, index=False)
            results = root / 'data'
            images = root / 'images'
            run_predict_armor(raw, results)
            self.assertEqual({p.suffix for p in results.iterdir()}, {'.csv', '.txt'})
            snapshot = {p.name: p.read_bytes() for p in results.iterdir()}
            with contextlib.redirect_stdout(io.StringIO()):
                plot_armor(results / 'armor_prediction_result_1.csv', raw, images)
            self.assertEqual({p.name for p in images.iterdir()}, {
                'top_down_trajectory_1.png', 'body_yaw_curve_1.png',
                'folded_armor_yaw_curve_1.png', 'armor_prediction_error_curve_1.png'})
            for image in images.iterdir():
                self.assertIsNotNone(cv2.imread(str(image)))
            self.assertEqual(snapshot, {p.name: p.read_bytes() for p in results.iterdir()})


if __name__ == '__main__':
    unittest.main()
