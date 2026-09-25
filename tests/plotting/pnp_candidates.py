"""Compare step 2 with the saved step-1 baseline; EKF remains unchanged."""
import pandas as pd
from . import light_refinement as comparison


def main():
    comparison.OUTPUT = comparison.ROOT / 'tests/outputs/pnp_candidates'
    comparison.BASELINE = comparison.OUTPUT / 'before'
    comparison.TITLE = 'PnP candidate selection; unchanged detector and EKF'
    comparison.main()
    for suffix in (1, 2):
        raw = pd.read_csv(comparison.ROOT/f'data/pose_raw_{suffix}.csv')
        print(f'Video {suffix}: candidate counts = {raw.pnp_candidate_count.value_counts().to_dict()}, '
              f'temporal tie breaks = {raw.pnp_used_temporal.sum()}')


if __name__ == '__main__':
    main()
