import argparse
from pathlib import Path


def parse_paths(description, default_suffix='2', model='single'):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--suffix', default=default_suffix)
    parser.add_argument('--source', choices=['video', 'camera'], help='Defaults to camera for suffix live, otherwise video')
    parser.add_argument('--model', choices=['single', 'armor'], default=model)
    parser.add_argument('--camera-mode', choices=['offline', 'mcu'], default='offline')
    parser.add_argument('--input-name', help='Video filename without extension; defaults to video_<suffix>')
    parser.add_argument('--data-dir', type=Path, help='Observation directory; defaults to the selected result directory')
    parser.add_argument('--results-dir', type=Path, help='Override the selected result directory')
    parser.add_argument('--output-dir', type=Path,
                        help='Image directory; defaults to a plots subdirectory')
    args = parser.parse_args()
    args.source = args.source or ('camera' if args.suffix == 'live' else 'video')
    if args.results_dir is None:
        input_name = (args.input_name or f'video_{args.suffix}') if args.source == 'video' else args.camera_mode
        args.results_dir = root / 'results' / args.source / args.model / input_name
    if args.data_dir is None:
        args.data_dir = args.results_dir
    if args.output_dir is None:
        args.output_dir = args.results_dir / 'plots'
    return args
