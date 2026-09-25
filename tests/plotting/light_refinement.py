"""Reproduce the step-1 comparison against the saved pre-change snapshot.

Run: python -B -m tests.plotting.light_refinement
Outputs are diagnostics, not ground-truth accuracy measurements.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .detector_comparison import metrics

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'tests/outputs/light_refinement'
BASELINE = OUTPUT / 'before'
TITLE = 'Light endpoint refinement only; unchanged PnP and EKF'


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for suffix, frames, end_seconds in [(1, 815, 25), (2, 351, 10)]:
        for name, raw, result in [('before', BASELINE, BASELINE),
                                  ('after', ROOT/'data', ROOT/'results')]:
            prediction = pd.read_csv(result/f'armor_prediction_result_{suffix}.csv')
            segment = prediction[prediction.timestamp.between(2000, end_seconds*1000)]
            center = segment[['xc', 'yc', 'zc']].to_numpy()
            deviation = np.linalg.norm(center-np.median(center, axis=0), axis=1)
            diagnostics = pd.read_csv(result/f'armor_observation_diagnostics_{suffix}.csv')
            rows.append(dict(video=suffix, version=name, segment_start_s=2,
                segment_end_s=end_seconds,
                **metrics(raw/f'pose_raw_{suffix}.csv', frames),
                center_p95_cm=100*np.quantile(deviation, .95),
                angular_speed_std=segment.w.std(),
                nearest_accepted_distance_rmse_cm=100*np.sqrt(np.nanmean(segment.err_distance**2)),
                nearest_accepted_yaw_rmse_deg=np.degrees(np.sqrt(np.nanmean(segment.err_armor_yaw**2))),
                rejected_observations=int((~diagnostics.accepted).sum())))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT/'summary.csv', index=False)
    print(summary.to_string(index=False))

    fields = [('missing_frames', 'Frames without detection (full video)'),
              ('pair_yaw_error_over_30_deg', 'Double-plate inconsistency >30 deg (full video)'),
              ('center_p95_cm', 'Center deviation P95 (cm)'),
              ('angular_speed_std', 'Angular speed std (rad/s)'),
              ('nearest_accepted_distance_rmse_cm', 'Prior distance residual RMSE (cm)'),
              ('nearest_accepted_yaw_rmse_deg', 'Prior plate yaw residual RMSE (deg)')]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), layout='constrained')
    for ax, (field, title) in zip(axes.flat, fields):
        x = np.arange(2)
        for offset, name, color in [(-.18, 'before', '#8795a8'), (.18, 'after', '#167f9d')]:
            values = summary[summary.version == name][field].to_numpy()
            bars = ax.bar(x+offset, values, .36, label=name, color=color)
            ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
        ax.set_xticks(x, ['Video 1', 'Video 2'])
        ax.set_title(title, fontsize=11)
        ax.margins(y=.35)
        ax.legend(loc='upper left', ncol=2, framealpha=.95)
    fig.suptitle(TITLE+'\n'
                 'EKF windows: video 1 = 2–25 s, video 2 = 2–10 s; no ground truth', fontsize=13)
    fig.savefig(OUTPUT/'comparison.png', dpi=150)
    plt.close(fig)

    # Matched-frame inspection of actual detector outputs; no synthetic boxes.
    for suffix, frame_id, crop in [(1, 753, (430, 570, 850, 840)),
                                    (2, 41, (300, 500, 1000, 950))]:
        images = []
        for name, folder in [('Before', BASELINE), ('After', ROOT/'results')]:
            cap = cv2.VideoCapture(str(folder/f'video_{suffix}.mp4'))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                raise RuntimeError(f'Cannot read {name} video {suffix} frame {frame_id}')
            x1, y1, x2, y2 = crop
            image = frame[y1:y2, x1:x2].copy()
            cv2.putText(image, f'{name}: video {suffix}, frame {frame_id}', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
            images.append(image)
        cv2.imwrite(str(OUTPUT/f'frame_{suffix}_{frame_id}.jpg'), np.hstack(images))


if __name__ == '__main__':
    main()
