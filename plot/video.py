"""Render frame-aligned EKF estimates and observation diagnostics for either video."""
import argparse
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

FX, FY = 1286.307063384126, 1288.1400736562441
CX, CY = 645.34450819155256, 483.6163720308021

ARMOR_COLORS = {
    0: (0, 255, 255),
    1: (255, 180, 0),
    2: (255, 0, 255),
    3: (0, 255, 100),
}
ARMOR_NAMES = {i: f"A{i}" for i in range(4)}

CENTER_COLOR = (255, 255, 255)  # 白色
OBS_COLOR = (0, 255, 255)       # 黄色 (实际检测)


CAMERA_MATRIX = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.]])
DISTORTION = np.array([-0.47562935060124745, 0.21831745829617311,
                       0.0004957613589406044, -0.00034617769548693592, 0.])

PROFILES = {
    '1': (CAMERA_MATRIX, DISTORTION, (1440, 1080)),
    '2': (np.array([[1711.311186, 0, 732.488057], [0, 1714.616882, 546.930868], [0, 0, 1.]]),
          np.array([-.119922, -.078593, .007511, -.028028, 0.]), (1280, 1024)),
}


def project(x, y, z, width=1440, height=1080, camera_matrix=CAMERA_MATRIX, distortion=DISTORTION):
    if z <= .1:
        return None
    points, _ = cv2.projectPoints(np.array([[x, y, z]], dtype=float),
                                  np.zeros(3), np.zeros(3), camera_matrix, distortion)
    u, v = points.ravel()
    if not (0 <= u < width and 0 <= v < height):
        return None
    return int(u), int(v)


def compute_plate_position(xc, yc, zc, body_yaw, r, dl, dh, armor_id):
    """用 EKF 状态计算第 armor_id 块装甲板的 3D 位置"""
    plate_yaw = body_yaw + armor_id * (np.pi / 2.0)
    is_side = (armor_id % 2 != 0)
    r_i = r + dl if is_side else r
    y_i = yc + dh if is_side else yc
    xa = xc + r_i * np.sin(plate_yaw)
    za = zc - r_i * np.cos(plate_yaw)
    ya = y_i
    return xa, ya, za


def compute_plate_corners(xc, yc, zc, body_yaw, r, dl, dh, armor_id):
    """Yaw-only model: TL, BL, BR, TR; same 135 x 56 mm geometry as Solver."""
    center = np.array(compute_plate_position(xc, yc, zc, body_yaw, r, dl, dh, armor_id))
    yaw = body_yaw + armor_id * np.pi / 2
    half_width = .135 / 2 * np.array([np.cos(yaw), 0., np.sin(yaw)])
    half_height = np.array([0., .056 / 2, 0.])
    return np.array([center-half_width-half_height, center-half_width+half_height,
                     center+half_width+half_height, center+half_width-half_height])


def draw_plate_outline(canvas, corners, camera_matrix, distortion, width, height, color, dashed=False):
    if np.any(corners[:, 2] <= .1):
        return []
    projected, _ = cv2.projectPoints(corners, np.zeros(3), np.zeros(3), camera_matrix, distortion)
    projected = projected.reshape(-1, 2)
    points = np.rint(projected).astype(int)
    visible = []
    for i in range(4):
        a, b = tuple(map(int, points[i])), tuple(map(int, points[(i+1) % 4]))
        ok, a, b = cv2.clipLine((0, 0, width, height), a, b)
        if ok:
            if dashed:
                start, end = np.array(a, dtype=float), np.array(b, dtype=float)
                length = np.linalg.norm(end-start)
                if length > 0:
                    direction = (end-start)/length
                    for offset in np.arange(0, length, 10):
                        p = tuple(np.rint(start+direction*offset).astype(int))
                        q = tuple(np.rint(start+direction*min(offset+6, length)).astype(int))
                        cv2.line(canvas, p, q, color, 2, cv2.LINE_AA)
            else:
                cv2.line(canvas, a, b, color, 2, cv2.LINE_AA)
            visible.extend([a, b])
    return visible


def draw_frame(frame, frame_id, fps, state, observations, diagnostics,
               camera_matrix, distortion, trail, camera_from_base):
    """Draw only the exact frame's state; never freeze a stale estimate over a gap."""
    height, width = frame.shape[:2]
    canvas = frame.copy()
    def pixel(position):
        position = camera_from_base[:, :3] @ np.asarray(position) + camera_from_base[:, 3]
        return project(*position, width, height, camera_matrix, distortion)
    status = 'NO STATE'
    plate_points = {}
    future_values = None
    if state is not None:
        values = np.array([state[name] for name in ['xc', 'yc', 'zc', 'body_yaw', 'r', 'dl', 'dh']])
        status = 'PREDICTION ONLY' if state['status'] == 'prediction_only' else 'UPDATED'
        center = pixel(values[:3])
        if center is not None:
            trail.append(center)
            if len(trail) > 1:
                cv2.polylines(canvas, [np.array(trail, np.int32)], False, (180,180,180), 1, cv2.LINE_AA)
            cv2.circle(canvas, center, 5, (255,255,255), 2, cv2.LINE_AA)
        for aid in range(4):
            color = ARMOR_COLORS[aid]
            corners = compute_plate_corners(*values, aid)
            corners = corners @ camera_from_base[:, :3].T + camera_from_base[:, 3]
            draw_plate_outline(canvas, corners, camera_matrix, distortion, width, height, color)
            point = pixel(compute_plate_position(*values, aid))
            if point is not None:
                plate_points[aid] = point
                color = ARMOR_COLORS[aid]
                cv2.circle(canvas, point, 5, color, 2, cv2.LINE_AA)
                cv2.putText(canvas, f'A{aid}', (point[0]+11, point[1]-9),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
                if center is not None:
                    cv2.line(canvas, center, point, color, 1, cv2.LINE_AA)
        if state['prediction_horizon_ms'] > 0:
            future_values = np.array([state[name] for name in
                ['future_xc', 'future_yc', 'future_zc', 'future_body_yaw', 'r', 'dl', 'dh']])
            future_center = pixel(future_values[:3])
            if future_center is not None:
                cv2.drawMarker(canvas, future_center, (255,255,255), cv2.MARKER_DIAMOND, 14, 2, cv2.LINE_AA)
            for aid in range(4):
                color = ARMOR_COLORS[aid]
                corners = compute_plate_corners(*future_values, aid)
                corners = corners @ camera_from_base[:, :3].T + camera_from_base[:, 3]
                draw_plate_outline(canvas, corners, camera_matrix, distortion, width, height, color, dashed=True)
                point = pixel(compute_plate_position(*future_values, aid))
                if point is not None:
                    cv2.drawMarker(canvas, point, color, cv2.MARKER_DIAMOND, 14, 2, cv2.LINE_AA)
                    cv2.putText(canvas, f'F{aid}', (point[0]+8, point[1]+16),
                                cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    if state is None:
        trail.clear()
    accepted, rejected, unknown = 0, 0, 0
    for index, obs in enumerate(observations):
        diagnostic = diagnostics.get(index)
        reason = diagnostic['reason'] if diagnostic else None
        accepted_flag = diagnostic['accepted'] if diagnostic else False
        is_accepted = str(accepted_flag).lower() == 'true'
        is_rejected = diagnostic is not None and reason in ('invalid', 'innovation_gate', 'association_conflict')
        aid = int(diagnostic['armor_id']) if is_accepted else -1
        color = ARMOR_COLORS[aid] if is_accepted else ((40,80,255) if is_rejected else (0,220,255))
        accepted += int(is_accepted); rejected += int(is_rejected)
        unknown += int(not is_accepted and not is_rejected)
        point = pixel([obs['x'], obs['y'], obs['z']])
        if point is None:
            continue
        marker = cv2.MARKER_TILTED_CROSS if is_rejected else cv2.MARKER_CROSS
        cv2.drawMarker(canvas, point, color, marker, 18, 2, cv2.LINE_AA)
        if is_accepted and aid in plate_points:
            cv2.line(canvas, point, plate_points[aid], color, 1, cv2.LINE_AA)
    lines = [f'REPLAY Armor frame {frame_id} {status}',
             'Camera XYZ (m): X right, Y down, Z forward']
    if not observations:
        lines.append('No armor detected')
    for index, obs in enumerate(observations):
        position = camera_from_base[:, :3] @ np.array([obs['x'], obs['y'], obs['z']]) + camera_from_base[:, 3]
        label = f'#{index+1}'
        lines.append(f'{label}  X={position[0]:.3f}  Y={position[1]:.3f}  Z={position[2]:.3f}')
        point = pixel([obs['x'], obs['y'], obs['z']])
        if point is not None:
            cv2.putText(canvas, label, point, cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(canvas, label, point, cv2.FONT_HERSHEY_SIMPLEX, .6, OBS_COLOR, 2, cv2.LINE_AA)
    if diagnostics:
        lines.append(f'Accepted: {accepted}  Rejected: {rejected}  Unclassified: {unknown}')
    if future_values is not None:
        lines.append(f'Forecast: +{state["prediction_horizon_ms"]:g} ms')
    for index, label in enumerate(lines):
        position = (20, 30 + index*28)
        cv2.putText(canvas, label, position, cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(canvas, label, position, cv2.FONT_HERSHEY_SIMPLEX, .6, OBS_COLOR, 2, cv2.LINE_AA)
    return canvas


def render_video(video_path, prediction_csv, raw_csv, output_path, camera_pose_csv, profile='1',
                 diagnostics_csv=None, start_frame=0, max_frames=None, trail_length=30):
    inputs = [Path(video_path), Path(prediction_csv), Path(raw_csv)]
    if diagnostics_csv is not None:
        inputs.append(Path(diagnostics_csv))
    inputs.append(Path(camera_pose_csv))
    output_path = Path(output_path)
    if any(output_path.resolve() == path.resolve() for path in inputs):
        raise ValueError('Output must not overwrite an input file')
    pred = pd.read_csv(prediction_csv)
    raw = pd.read_csv(raw_csv)
    poses = pd.read_csv(camera_pose_csv)
    transforms = {}
    for row in poses.to_dict('records'):
        w, x, y, z = [row[c] for c in ['qw', 'qx', 'qy', 'qz']]
        rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                             [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                             [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        translation = np.array([row[c] for c in ['x', 'y', 'z']])
        transforms[int(row['frame_id'])] = np.column_stack((rotation.T, -rotation.T @ translation))
    pred_lookup = {int(r['frame_id']): r for r in pred.to_dict('records')}
    raw_lookup = {int(fid): group.to_dict('records') for fid, group in raw.groupby('frame_id', sort=False)}
    diagnostic_lookup = {}
    if diagnostics_csv is not None:
        for row in pd.read_csv(diagnostics_csv).to_dict('records'):
            diagnostic_lookup.setdefault(int(row['frame_id']), {})[int(row['observation_index'])] = row
    camera_matrix, distortion, _ = PROFILES[str(profile)]
    cap = cv2.VideoCapture(str(video_path))
    writer = None
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot open video: {video_path}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        output_path.parent.mkdir(parents=True,exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
        if not writer.isOpened():
            raise RuntimeError(f'Cannot create output video: {output_path}')
        cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
        count = 0
        trail = deque(maxlen=trail_length)
        while start_frame + count < total and (max_frames is None or count < max_frames):
            ok,frame = cap.read()
            if not ok:
                raise RuntimeError("Video decoding failed before the end of the source")
            fid = start_frame+count
            canvas = draw_frame(frame, fid, fps, pred_lookup.get(fid), raw_lookup.get(fid, []),
                                diagnostic_lookup.get(fid, {}), camera_matrix, distortion, trail, transforms[fid])
            writer.write(canvas)
            count += 1
        if not count:
            raise RuntimeError('No video frames were decoded')
        print(f'Saved {count} frames: {output_path}')
        return count
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suffix', choices=['1','2'], default='1')
    parser.add_argument('--video', type=Path)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--raw', type=Path)
    parser.add_argument('--camera-poses', type=Path, help='Per-frame T_BC exported by Armor')
    parser.add_argument('--diagnostics', type=Path)
    parser.add_argument('--no-diagnostics', action='store_true')
    parser.add_argument('--camera-profile', choices=['1','2'])
    parser.add_argument('--data-dir', type=Path, default=root/'data')
    parser.add_argument('--results-dir', type=Path, default=root/'results')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--trail-length', type=int, default=30)
    args = parser.parse_args()
    if args.no_diagnostics and args.diagnostics:
        parser.error('--diagnostics and --no-diagnostics are mutually exclusive')
    diagnostics = args.diagnostics or args.results_dir/f'armor_observation_diagnostics_{args.suffix}.csv'
    if args.no_diagnostics:
        diagnostics = None
    try:
        predictions = args.predictions or args.results_dir/f'armor_prediction_result_{args.suffix}.csv'
        render_video(args.video or root/'assets/video'/f'video_{args.suffix}.avi',
                     predictions,
                     args.raw or args.data_dir/f'pose_base_{args.suffix}.csv',
                     args.output or (args.output_dir or args.results_dir)/f'armor_video_{args.suffix}.mp4',
                     args.camera_poses or args.data_dir/f'camera_pose_{args.suffix}.csv',
                     args.camera_profile or args.suffix, diagnostics,
                     args.start_frame,args.max_frames,args.trail_length)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
