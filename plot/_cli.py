"""Shared paths for standalone plotting commands."""
import argparse
from pathlib import Path


def parse_paths(description, default_suffix='2'):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--suffix', default=default_suffix)
    parser.add_argument('--data-dir', type=Path, default=root / 'data')
    parser.add_argument('--results-dir', type=Path, default=root / 'results')
    parser.add_argument('--output-dir', type=Path,
                        help='Image directory; defaults to --results-dir')
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.results_dir
    return args
