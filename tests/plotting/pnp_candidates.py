"""Compare step 2 with the saved step-1 baseline; EKF remains unchanged."""
from . import light_refinement as comparison


def main():
    comparison.OUTPUT = comparison.ROOT / 'tests/outputs/pnp_candidates'
    comparison.BASELINE = comparison.OUTPUT / 'before'
    comparison.TITLE = 'PnP candidate selection; unchanged detector and EKF'
    comparison.main()


if __name__ == '__main__':
    main()
