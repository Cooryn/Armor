"""Opt-in integration test: switch source constants, rebuild, run and restore.

Run with --preview to also open a short OpenCV preview. Test videos/results are
kept under tests/outputs; the original source and default executable are restored.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.reference.predictor.video_predictor import run_video_predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, default=ROOT / 'tests/build')
    parser.add_argument('--work-dir', type=Path, default=ROOT / 'tests/outputs/armor-video')
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--guards-only', action='store_true', help='Only test preview, live-branch build and file I/O errors')
    parser.add_argument('--max-source-frames', type=int, help='Use short source clips for routine integration checks')
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'src/Armor.cpp'
    original = source.read_bytes()
    env = {k.upper(): v for k, v in os.environ.items()}
    env['PATH'] = str(Path(sys.base_prefix) / 'Library/bin') + os.pathsep + env.get('PATH', '')
    exe = ROOT / ('Armor.exe' if sys.platform == 'win32' else 'Armor')
    report = []

    def build(label):
        result = subprocess.run(['cmake', '--build', str(args.build_dir), '--config', 'Release', '--target', 'Armor'],
                                cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')
        (work / f'build-{label}.log').write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def configure(color, video, model, preview=False):
        text = original.decode('utf-8')
        models = {'SinglePlate': 'PREDICTOR_SINGLE_PLATE', 'Polar': 'PREDICTOR_POLAR', 'Armor': 'PREDICTOR_ARMOR'}
        values = {'target_color': f'ENEMY_{color.upper()}', 'video_file': json.dumps(video.relative_to(ROOT).as_posix()),
                  'video_camera_profile': '2' if 'video_2' in video.stem else '1',
                  'predictor_type': models[model], 'preview': 'true' if preview else 'false'}
        for key, value in values.items():
            text, count = re.subn(rf'(?m)^(constexpr [^\n]*\b{key}\s*=\s*)[^;]+;', lambda m: m[1] + value + ';', text)
            assert count == 1, key
        source.write_bytes(text.encode('utf-8'))

    def run_case(model, color, suffix, video, preview=False, blue_fixture=False):
        tag = model.lower() + color + suffix + ('preview' if preview else '') + ('fixture' if blue_fixture else '')
        case = work / tag
        case.mkdir(parents=True, exist_ok=True)
        # Select the explicit profile from the fixture name; the final token
        # is a unique output suffix to avoid overwriting formal results.
        input_video = case / f'video_{suffix}_{tag}.avi'
        if preview or blue_fixture or args.max_source_frames:
            cap = cv2.VideoCapture(str(video))
            writer = cv2.VideoWriter(str(input_video), cv2.VideoWriter_fourcc(*'MJPG'),
                                    cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))
            assert writer.isOpened()
            for _ in range(6 if preview else args.max_source_frames or 120):
                ok, frame = cap.read()
                assert ok
                # Both supplied videos contain red plates. Swap B/R channels
                # to exercise blue tracking on a known moving target as well.
                writer.write(frame[:, :, ::-1] if blue_fixture else frame)
            writer.release()
            cap.release()
        else:
            shutil.copyfile(video, input_video)
        configure(color, input_video, model, preview)
        print(tag, 'building and processing video', flush=True)
        build(tag)
        result = subprocess.run([str(exe)], cwd=work, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')
        (case / 'run.log').write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        raw_path = ROOT / 'data' / f'pose_raw_{tag}.csv'
        raw = pd.read_csv(raw_path)
        base_path = ROOT / 'data' / f'pose_base_{tag}.csv'
        base = pd.read_csv(base_path)
        assert base.coordinate_frame.eq('base').all()
        assert raw.coordinate_frame.eq('camera').all()
        if not preview:
            assert len(raw) > 0, f'{color} produced no observations'
        if blue_fixture:
            assert len(raw) > 10, 'blue fixture failed to produce a tracking sequence'
        cap = cv2.VideoCapture(str(input_video))
        count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        groups = {int(fid): [(r[['x', 'y', 'z']].to_numpy(float),
                             r[['target_yaw', 'target_pitch', 'distance', 'armor_orientation_yaw']].to_numpy(float))
                            for _, r in group.iterrows()] for fid, group in base.groupby('frame_id')}
        frames = [(i, i * 1000 / fps, groups.get(i, [])) for i in range(count)]
        reference = case / 'numpy'
        mode = {'SinglePlate': 'basic', 'Polar': 'polar', 'Armor': 'armor'}[model]
        run_video_predictor(frames, mode, reference)
        maximum = 0.
        for expected in reference.iterdir():
            if expected.name == 'geometry.csv':
                continue
            actual = ROOT / 'results' / expected.name.replace('_1.', f'_{tag}.')
            if expected.suffix == '.txt':
                assert expected.read_text() == actual.read_text(), actual
            else:
                x, y = pd.read_csv(expected), pd.read_csv(actual)
                assert y.coordinate_frame.eq('base').all()
                y = y.drop(columns=['coordinate_frame'])
                assert x.columns.tolist() == y.columns.tolist() and x.shape == y.shape, actual
                for col in x:
                    if pd.api.types.is_numeric_dtype(x[col]) and not pd.api.types.is_bool_dtype(x[col]):
                        xx, yy = x[col].to_numpy(float), y[col].to_numpy(float)
                        np.testing.assert_allclose(xx, yy, rtol=1e-7, atol=1e-8, equal_nan=True, err_msg=str(actual) + '/' + col)
                        finite = np.isfinite(xx) & np.isfinite(yy)
                        if finite.any(): maximum = max(maximum, float(np.max(np.abs(xx[finite] - yy[finite]))))
                    else:
                        pd.testing.assert_series_equal(x[col], y[col], check_dtype=False)
            shutil.move(str(actual), str(case / actual.name))
        overlay_path = ROOT / 'results' / (input_video.stem + '.avi')
        overlay = cv2.VideoCapture(str(overlay_path))
        assert overlay.isOpened() and int(overlay.get(cv2.CAP_PROP_FRAME_COUNT)) == count
        assert overlay.get(cv2.CAP_PROP_FPS) == fps
        ids = sorted(groups)
        fid = ids[len(ids) // 2] if ids else 0
        overlay.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, snapshot = overlay.read()
        assert ok and cv2.imwrite(str(case / 'overlay.png'), snapshot)
        overlay.release()
        shutil.move(str(overlay_path), str(case / 'overlay.avi'))
        shutil.move(str(raw_path), str(case / 'observations.csv'))
        shutil.move(str(base_path), str(case / 'base_observations.csv'))
        shutil.move(str(ROOT / 'data' / f'camera_pose_{tag}.csv'), str(case / 'camera_pose.csv'))
        shutil.move(str(ROOT / 'results' / f'control_target_{tag}.csv'), str(case / 'control_target.csv'))
        report.append(dict(model=model, color=color, frames=count, observations=len(raw), preview=preview,
                           blue_fixture=blue_fixture, snapshot_frame=fid, max_absolute_difference=maximum))
        print(tag, 'PASS', count, 'frames,', len(raw), 'observations', flush=True)

    try:
        if not args.guards_only:
            for model in ('SinglePlate', 'Polar', 'Armor'):
                for color, suffix in (('red', '1'), ('blue', '2')):
                    run_case(model, color, suffix, ROOT / 'assets/video' / f'video_{suffix}.avi')
                run_case(model, 'blue', '1', ROOT / 'assets/video/video_1.avi', blue_fixture=True)
            run_case('Armor', 'red', '2', ROOT / 'assets/video/video_2.avi')
        if args.preview:
            run_case('Armor', 'red', '1', ROOT / 'assets/video/video_1.avi', True)
        # Build the live branch without opening hardware or sending commands.
        configure('red', work / 'unused.avi', 'Armor')
        source.write_text(source.read_text(encoding='utf-8').replace('camera_source = CAMERA_VIDEO;', 'camera_source = CAMERA_OPENCV;'), encoding='utf-8')
        build('live-compile-only')
        configure('red', work / 'missing.avi', 'Armor')
        build('missing')
        rejection = subprocess.run([str(exe)], env=env, capture_output=True, text=True)
        assert rejection.returncode != 0 and 'Cannot open image source' in rejection.stderr
        # A valid video located in results would otherwise be overwritten by its
        # own output. Use a newly allocated file and verify it stays byte-identical.
        with tempfile.NamedTemporaryFile(dir=ROOT / 'results', prefix='armor_guard_', suffix='.avi', delete=False) as file:
            collision = Path(file.name)
        try:
            shutil.copyfile(ROOT / 'assets/video/video_1.avi', collision)
            before = collision.read_bytes()
            configure('red', collision, 'Armor')
            build('collision')
            rejection = subprocess.run([str(exe)], env=env, capture_output=True, text=True)
            assert rejection.returncode != 0 and 'must not overwrite' in rejection.stderr
            assert collision.read_bytes() == before
        finally:
            collision.unlink()
    finally:
        source.write_bytes(original)
        build('restored')
        assert source.read_bytes() == original
        print('Restored original source configuration and rebuilt Armor', flush=True)
    (work / 'verification.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
