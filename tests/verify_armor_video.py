"""Opt-in integration test: switch source constants, rebuild, run and restore.

All video runs display the OpenCV window. Test videos/results are
kept under tests/outputs; the original source and default executable are restored.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
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
    parser.add_argument('--guards-only', action='store_true', help='Only test live-branch build and file I/O errors')
    parser.add_argument('--max-source-frames', type=int, help='Use short source clips for routine integration checks')
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    program_root = work / 'program-output'
    source = ROOT / 'src/Armor.cpp'
    original = source.read_bytes()
    env = {k.upper(): v for k, v in os.environ.items()}
    env['PATH'] = str(Path(sys.base_prefix) / 'Library/bin') + os.pathsep + env.get('PATH', '')
    exe = ROOT / ('Armor.exe' if sys.platform == 'win32' else 'Armor')
    report = []

    def formal_snapshot():
        return {str(path.relative_to(ROOT)): (path.stat().st_size, path.stat().st_mtime_ns)
                for path in (ROOT / 'results').rglob('*') if path.is_file()}

    formal_before = formal_snapshot()

    def build(label):
        result = subprocess.run(['cmake', '--build', str(args.build_dir), '--config', 'Release', '--target', 'Armor'],
                                cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')
        (work / f'build-{label}.log').write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    def configure(color, video, model):
        text = original.decode('utf-8')
        models = {'SinglePlate': 'PREDICTOR_SINGLE_PLATE', 'Armor': 'PREDICTOR_ARMOR'}
        # Video must ignore both MCU switches, even when they are enabled.
        values = {'camera_source': 'VIDEO', 'connect_mcu': 'true', 'auto_aim_enabled': 'true',
                  'prediction_horizon_ms': '50',
                  'target_color': f'ENEMY_{color.upper()}', 'video_file': json.dumps(video.as_posix()),
                  'video_camera_profile': '2' if 'video_2' in video.stem else '1',
                  'predictor_type': models[model]}
        for key, value in values.items():
            text, count = re.subn(rf'(?m)^((?:const |CameraSource |PredictorType |bool )[^\n]*\b{key}\s*=\s*)[^;]+;', lambda m: m[1] + value + ';', text)
            assert count == 1, key
        text, count = re.subn(r'const std::filesystem::path root = ARMOR_PROJECT_ROOT;',
                             lambda _: 'const std::filesystem::path root = ' + json.dumps(program_root.as_posix()) + ';', text)
        assert count == 1, 'isolated program output root'
        source.write_bytes(text.encode('utf-8'))

    def run_case(model, color, suffix, video, blue_fixture=False):
        tag = model.lower() + color + suffix + ('fixture' if blue_fixture else '')
        case = work / tag
        case.mkdir(parents=True, exist_ok=True)
        input_video = case / f'video_{suffix}_{tag}.avi'
        if blue_fixture or args.max_source_frames:
            cap = cv2.VideoCapture(str(video))
            writer = cv2.VideoWriter(str(input_video), cv2.VideoWriter_fourcc(*'MJPG'),
                                    cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))
            assert writer.isOpened()
            for _ in range(args.max_source_frames or 120):
                ok, frame = cap.read()
                assert ok
                writer.write(frame[:, :, ::-1] if blue_fixture else frame)
            writer.release()
            cap.release()
        else:
            shutil.copyfile(video, input_video)
        configure(color, input_video, model)
        print(tag, 'building and processing video', flush=True)
        build(tag)
        result = subprocess.run([str(exe)], cwd=work, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')
        (case / 'run.log').write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        model_name = {'SinglePlate': 'single', 'Armor': 'armor'}[model]
        result_directory = program_root / 'results/video' / model_name / input_video.stem
        raw_path = result_directory / f'pose_raw_{tag}.csv'
        raw = pd.read_csv(raw_path)
        base_path = result_directory / f'pose_base_{tag}.csv'
        assert not base_path.exists()
        assert not (result_directory / f'camera_pose_{tag}.csv').exists()
        assert (result_directory / f'control_target_{tag}.csv').exists()
        assert raw.coordinate_frame.eq('camera').all()
        assert len(raw) > 0, f'{color} produced no observations'
        if blue_fixture:
            assert len(raw) > 10, 'blue fixture failed to produce a tracking sequence'
        cap = cv2.VideoCapture(str(input_video))
        count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        groups = {int(fid): [(r[['x', 'y', 'z']].to_numpy(float),
                             r[['target_yaw', 'target_pitch', 'distance', 'armor_orientation_yaw']].to_numpy(float))
                            for _, r in group.iterrows()] for fid, group in raw.groupby('frame_id')}
        frames = [(i, i * 1000 / fps, groups.get(i, [])) for i in range(count)]
        reference = case / 'numpy'
        mode = {'SinglePlate': 'basic', 'Armor': 'armor'}[model]
        run_video_predictor(frames, mode, reference)
        maximum = 0.
        for expected in reference.iterdir():
            if expected.name == 'geometry.csv':
                continue
            actual = result_directory / expected.name.replace('_1.', f'_{tag}.')
            if expected.suffix == '.txt':
                assert expected.read_text() == actual.read_text(), actual
            else:
                x, y = pd.read_csv(expected), pd.read_csv(actual)
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
        overlay_path = result_directory / (input_video.stem + '.avi')
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
        report.append(dict(model=model, color=color, frames=count, observations=len(raw),
                           blue_fixture=blue_fixture, snapshot_frame=fid, max_absolute_difference=maximum))
        print(tag, 'PASS', count, 'frames,', len(raw), 'observations', flush=True)

    try:
        if not args.guards_only:
            for model in ('SinglePlate', 'Armor'):
                for color, suffix in (('red', '1'), ('blue', '2')):
                    run_case(model, color, suffix, ROOT / 'assets/video' / f'video_{suffix}.avi')
                run_case(model, 'blue', '1', ROOT / 'assets/video/video_1.avi', blue_fixture=True)
            run_case('Armor', 'red', '2', ROOT / 'assets/video/video_2.avi')
        configure('red', work / 'unused.avi', 'Armor')
        source.write_text(source.read_text(encoding='utf-8').replace('camera_source = VIDEO;', 'camera_source = CAMERA;'), encoding='utf-8')
        build('live-compile-only')
        configure('red', work / 'missing.avi', 'Armor')
        build('missing')
        rejection = subprocess.run([str(exe)], env=env, capture_output=True, text=True)
        assert rejection.returncode != 0 and 'Cannot open image source' in rejection.stderr
        collision_directory = program_root / 'results/video/armor/collision'
        collision_directory.mkdir(parents=True, exist_ok=True)
        collision = collision_directory / 'collision.avi'
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
    assert formal_snapshot() == formal_before, 'integration tests modified formal results'


if __name__ == '__main__':
    main()
