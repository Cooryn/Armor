"""Capture chessboard images and export calibration for Armor's live camera."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import cv2 as cv
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MIN_VIEWS = 10


def find_corners(frame, pattern):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    found, corners = cv.findChessboardCorners(
        gray, pattern, cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE
    )
    if not found:
        return None
    return cv.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                           (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_MAX_ITER, 50, .001))


def read_image(path, size):
    frame = cv.imdecode(np.fromfile(path, dtype=np.uint8), cv.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Cannot read image: {path}")
    check_size(frame, size)
    return frame


def check_size(frame, size):
    actual = frame.shape[1], frame.shape[0]
    if actual != size:
        raise RuntimeError(f"Actual image size {actual}; expected {size}. Match Armor's capture mode.")


def calibrate(args, corners, paths, output):
    if len(corners) < MIN_VIEWS:
        raise RuntimeError(f"Need at least {MIN_VIEWS} detected views; aim for 25-30 varied views.")
    points = np.zeros((args.columns * args.rows, 3), np.float32)
    points[:, :2] = np.mgrid[0:args.columns, 0:args.rows].T.reshape(-1, 2) * args.square_mm
    rms, matrix, distortion, rotations, translations = cv.calibrateCamera(
        [points.copy() for _ in corners], corners, (args.width, args.height), None, None
    )
    errors = []
    for path, observed, rotation, translation in zip(paths, corners, rotations, translations):
        projected, _ = cv.projectPoints(points, rotation, translation, matrix, distortion)
        error = float(np.sqrt(np.mean(np.sum((observed - projected) ** 2, axis=2))))
        errors.append({"image": str(path), "rms_pixels": error})
    result = {
        "image_width": args.width, "image_height": args.height,
        "pattern_inner_corners": [args.columns, args.rows], "square_size_mm": args.square_mm,
        "view_count": len(corners), "camera_matrix": matrix.tolist(),
        "distortion_coefficients": distortion.ravel().tolist(),
        "distortion_order": ["k1", "k2", "p1", "p2", "k3"],
        "rms_pixels": float(rms), "per_view_errors": errors,
    }
    (output / "calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    storage = cv.FileStorage(str(output / "calibration.yml"), cv.FILE_STORAGE_WRITE)
    if not storage.isOpened():
        raise RuntimeError("Cannot write calibration.yml")
    try:
        storage.write("image_width", args.width)
        storage.write("image_height", args.height)
        storage.write("camera_matrix", matrix)
        storage.write("distortion_coefficients", distortion)
        storage.write("rms_pixels", float(rms))
    finally:
        storage.release()
    rows = [", ".join(format(float(v), ".17g") for v in row) for row in matrix]
    coeffs = ", ".join(format(float(v), ".17g") for v in distortion.ravel())
    cpp = (
        f"// OV2710, {args.width}x{args.height}; keep the calibrated focus setting.\n"
        "const cv::Mat live_camera_matrix = (cv::Mat_<double>(3, 3) <<\n"
        + ",\n".join("    " + row for row in rows) + ");\n"
        "const cv::Mat live_distortion = (cv::Mat_<double>(1, 5) <<\n"
        f"    {coeffs});\n"
    )
    (output / "camera_parameters.cpp.txt").write_text(cpp, encoding="utf-8")
    print(f"Calibration completed: {len(corners)} views, RMS = {rms:.4f} pixels")
    for view in sorted(errors, key=lambda item: item["rms_pixels"], reverse=True)[:5]:
        print(f"  {Path(view['image']).name}: {view['rms_pixels']:.4f} px")
    print(f"Results: {output}")


def capture(args, output):
    backend = cv.CAP_DSHOW if sys.platform == "win32" else cv.CAP_ANY
    stream = cv.VideoCapture(args.device, backend)
    window = "OV2710 calibration - SPACE: save | C: calculate | ESC: exit"
    corners, paths = [], []
    try:
        if not stream.isOpened():
            raise RuntimeError(f"Cannot open camera {args.device}. Connect OV2710 and close other camera apps.")
        for prop, value in (
            (cv.CAP_PROP_FRAME_WIDTH, args.width), (cv.CAP_PROP_FRAME_HEIGHT, args.height),
            (cv.CAP_PROP_FPS, args.fps), (cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG")),
        ):
            if not stream.set(prop, value):
                raise RuntimeError(f"Driver rejected capture setting {prop}={value}")
        cv.namedWindow(window, cv.WINDOW_NORMAL)
        cv.resizeWindow(window, 960, 540)
        print(f"Device {args.device}; MJPG {args.width}x{args.height}; {args.columns}x{args.rows} INNER corners")
        print(f"Measured square side: {args.square_mm:g} mm. SPACE: save, C: calculate, ESC: exit.")
        print(f"Images and results: {output}")
        while True:
            ok, frame = stream.read()
            if not ok:
                raise RuntimeError("Camera acquisition failed")
            check_size(frame, (args.width, args.height))
            detected = find_corners(frame, (args.columns, args.rows))
            canvas = frame.copy()
            if detected is not None:
                cv.drawChessboardCorners(canvas, (args.columns, args.rows), detected, True)
            status = "BOARD FOUND" if detected is not None else "Show the complete board"
            cv.putText(canvas, f"{status} | Saved: {len(corners)}", (15, 30), cv.FONT_HERSHEY_SIMPLEX,
                       .7, (0, 255, 0) if detected is not None else (0, 0, 255), 2)
            cv.imshow(window, canvas)
            key = cv.waitKey(1) & 0xff
            if key in (27, ord("q")):
                break
            try:
                if cv.getWindowProperty(window, cv.WND_PROP_VISIBLE) < 1:
                    break
            except cv.error:
                break
            if key == ord(" "):
                if detected is None:
                    print("Not saved: complete board was not detected")
                    continue
                path = output / f"view_{len(corners) + 1:03d}.png"
                encoded, data = cv.imencode(".png", frame)
                if not encoded:
                    raise RuntimeError("Cannot encode captured image")
                data.tofile(path)
                corners.append(detected.copy())
                paths.append(path)
                print(f"Saved {path.name}; total {len(corners)}")
            if key in (ord("c"), ord("C")):
                if len(corners) < MIN_VIEWS:
                    print("Collect at least 10 views first; 25-30 recommended")
                    continue
                calibrate(args, corners, paths, output)
                break
    finally:
        stream.release()
        cv.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--columns", type=int, default=9, help="INNER corner columns, not squares")
    parser.add_argument("--rows", type=int, default=6, help="INNER corner rows, not squares")
    parser.add_argument("--square-mm", type=float, default=25, help="Measured square side length")
    parser.add_argument("--images", type=Path, help="Use saved images instead of a live camera")
    parser.add_argument("--output", type=Path, default=ROOT / "data/calibration/ov2710")
    args = parser.parse_args()
    if min(args.width, args.height, args.fps, args.square_mm) <= 0 or min(args.columns, args.rows) < 2:
        parser.error("Dimensions, FPS and square side must be positive; corner counts must be at least 2")
    output = args.output.resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    if args.images is None:
        capture(args, output)
    else:
        corners, paths = [], []
        for path in sorted(args.images.iterdir()):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                continue
            detected = find_corners(read_image(path, (args.width, args.height)), (args.columns, args.rows))
            if detected is not None:
                corners.append(detected)
                paths.append(path.resolve())
            else:
                print(f"No complete chessboard: {path.name}")
        calibrate(args, corners, paths, output)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, cv.error) as error:
        print(f"Calibration failed: {error}", file=sys.stderr)
        sys.exit(1)
