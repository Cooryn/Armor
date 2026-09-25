"""Compare saved pre-change observations with the current detector outputs.

These are consistency diagnostics, not annotated precision/recall or true pose errors.
Run from any directory with the repository Python environment.
"""
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'tests' / 'outputs' / 'detector_comparison'


def metrics(path, total_frames):
    data = pd.read_csv(path)
    pair_errors = []
    for _, group in data.groupby('frame_id'):
        if len(group) == 2:
            gap = abs(np.angle(np.exp(1j * np.diff(group.armor_orientation_yaw)))[0])
            pair_errors.append(abs(np.rad2deg(gap) - 90))
    return dict(observations=len(data), detected_frames=data.frame_id.nunique(),
                missing_frames=total_frames-data.frame_id.nunique(),
                double_detection_frames=len(pair_errors),
                pair_yaw_error_over_30_deg=sum(np.array(pair_errors) > 30),
                pair_yaw_error_median_deg=np.median(pair_errors) if pair_errors else np.nan)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for video, frames in [(1, 815), (2, 351)]:
        for version, folder in [('before', ROOT/'tests/outputs/detector_baseline'),
                                ('after', ROOT/'data')]:
            rows.append(dict(video=video, version=version,
                             **metrics(folder/f'pose_raw_{video}.csv', frames)))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT/'summary.csv', index=False)
    print(summary.to_string(index=False))

    # Actual rendered detector outputs at the same source frame, not synthetic boxes.
    crops = []
    for version, path in [('Before', ROOT/'tests/outputs/detector_baseline/video_1.mp4'),
                          ('After', ROOT/'results/video_1.mp4')]:
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 753)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f'Cannot read frame 753: {path}')
        crop = frame[570:840, 430:850].copy()
        cv2.putText(crop, f'{version}: frame 753', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2)
        crops.append(crop)
    cv2.imwrite(str(OUTPUT/'frame_753_before_after.jpg'), np.hstack(crops))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, field, title in zip(axes,
            ['missing_frames', 'pair_yaw_error_over_30_deg'],
            ['Frames without a detection', 'Two-plate yaw inconsistency >30 deg']):
        before = summary[summary.version == 'before'][field].to_numpy()
        after = summary[summary.version == 'after'][field].to_numpy()
        x = np.arange(2)
        ax.bar(x-.18, before, .36, label='Before')
        ax.bar(x+.18, after, .36, label='After')
        ax.set_xticks(x, ['Video 1', 'Video 2'])
        ax.set_title(title)
        ax.legend()
    fig.savefig(OUTPUT/'consistency_comparison.png', dpi=160)
    plt.close(fig)

    # Only compare EKF outputs produced using the same predictor and parameters.
    ekf_rows = []
    for video, lo, hi in [(1, 2, 25), (2, 2, 10)]:
        for version, folder in [('before', ROOT/'tests/outputs/detector_baseline'),
                                ('after', ROOT/'results')]:
            path = folder/f'armor_prediction_result_{video}.csv'
            if not path.exists():
                continue
            result = pd.read_csv(path)
            segment = result[result.timestamp.between(lo*1000, hi*1000)]
            diagnostics = pd.read_csv(folder/f'armor_observation_diagnostics_{video}.csv')
            center = segment[['xc', 'yc', 'zc']].to_numpy()
            deviation = np.linalg.norm(center-np.median(center, axis=0), axis=1)
            ekf_rows.append(dict(video=video, version=version,
                center_p95_m=np.quantile(deviation, .95), angular_speed_std=segment.w.std(),
                accepted=int((diagnostics.reason == 'accepted').sum()),
                rejected=int(diagnostics.reason.isin(
                    ['innovation_gate', 'association_conflict', 'invalid']).sum()),
                distance_prior_rmse_m=np.sqrt((segment.err_distance**2).mean()),
                yaw_prior_rmse_rad=np.sqrt((segment.err_armor_yaw**2).mean())))
    if ekf_rows:
        pd.DataFrame(ekf_rows).to_csv(OUTPUT/'ekf_consistency.csv', index=False)


if __name__ == '__main__':
    main()
