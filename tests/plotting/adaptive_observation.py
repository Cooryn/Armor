"""Compare quality-weighted EKF with step-2 baseline on identical raw CSVs."""
import numpy as np
import pandas as pd
from . import light_refinement as comparison


def main():
    comparison.OUTPUT = comparison.ROOT / 'tests/outputs/adaptive_observation'
    comparison.BASELINE = comparison.OUTPUT / 'before'
    comparison.TITLE = 'Adaptive observation noise; identical detector and PnP observations'
    for suffix in (1, 2):
        assert (comparison.BASELINE/f'pose_raw_{suffix}.csv').read_bytes() == (
            comparison.ROOT/f'data/pose_raw_{suffix}.csv').read_bytes(), 'Raw inputs changed'
    comparison.main()
    rows = []
    for suffix in (1, 2):
        for version, folder in [('before', comparison.BASELINE), ('after', comparison.ROOT/'results')]:
            d = pd.read_csv(folder/f'armor_observation_diagnostics_{suffix}.csv')
            p = pd.read_csv(folder/f'armor_prediction_result_{suffix}.csv')
            accepted = d[d.reason == 'accepted']
            tail = p[p.frame_id >= 330] if suffix == 2 else p.iloc[:0]
            stop = p[p.frame_id >= 320] if suffix == 2 else p.iloc[:0]
            small = (stop.w.abs() < .5).to_numpy()
            settled = next((int(stop.frame_id.iloc[i]) for i in range(max(0, len(stop)-4))
                            if small[i:i+5].all()), np.nan)
            rows.append(dict(video=suffix, version=version, accepted=len(accepted),
                rejected=int((~d.accepted).sum()), prediction_only=int((p.status == 'prediction_only').sum()),
                accepted_distance_rmse_cm=100*np.sqrt(np.nanmean(accepted.distance_residual**2)),
                stopped_tail_w_rms=np.sqrt(np.mean(tail.w**2)) if len(tail) else np.nan,
                first_5_frames_below_half_rad_s_after_frame320=settled,
                distance_scale_median=d.distance_noise_scale.median() if 'distance_noise_scale' in d else 1.,
                yaw_scale_median=d.yaw_noise_scale.median() if 'yaw_noise_scale' in d else 1.))
    detail = pd.DataFrame(rows)
    detail.to_csv(comparison.OUTPUT/'weighting_summary.csv', index=False)
    print(detail.to_string(index=False))


if __name__ == '__main__':
    main()
