"""Compatibility entry point for tests.plotting.detector_comparison."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.plotting.detector_comparison import main, metrics

if __name__ == '__main__':
    main()
