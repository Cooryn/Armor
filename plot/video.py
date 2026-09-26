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
    0: (90, 100, 255),
    1: (255, 180, 40),
    2: (80, 230, 110),
    3: (20, 190, 255),
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


def project(x, y, z, width=1440, height=1080, camera_matrix=None, distortion=None):
    if not np.isfinite([x, y, z]).all() or z <= .1:
        return None
    camera_matrix = CAMERA_MATRIX if camera_matrix is None else camera_matrix
    distortion = DISTORTION if distortion is None else distortion
    points, _ = cv2.projectPoints(np.array([[x, y, z]], dtype=float),
                                  np.zeros(3), np.zeros(3), camera_matrix, distortion)
    u, v = points.ravel()
    if not np.isfinite([u, v]).all() or not (0 <= u < width and 0 <= v < height):
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
    if not np.isfinite(corners).all() or np.any(corners[:, 2] <= .1):
        return []
    projected, _ = cv2.projectPoints(corners, np.zeros(3), np.zeros(3), camera_matrix, distortion)
    projected = projected.reshape(-1, 2)
    if not np.isfinite(projected).all():
        return []
    points = np.rint(np.clip(projected, -1000000, 1000000)).astype(int)
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


def draw_cross(img, cx, cy, size=6, color=(255, 255, 255), thickness=2):
    """绘制十字标记"""
    cv2.line(img, (cx - size, cy), (cx + size, cy), color, thickness)
    cv2.line(img, (cx, cy - size), (cx, cy + size), color, thickness)


def load_table(path, required, unique=False):
    table = pd.read_csv(path)
    missing = set(required) - set(table.columns)
    if missing:
        raise ValueError(f'{path}: missing columns {sorted(missing)}')
    ids = pd.to_numeric(table.frame_id, errors='raise').to_numpy(dtype=float)
    if not np.isfinite(ids).all() or np.any(ids < 0) or np.any(ids % 1):
        raise ValueError(f'{path}: invalid frame IDs')
    table['frame_id'] = ids.astype(np.int64)
    if unique and table.frame_id.duplicated().any():
        raise ValueError(f'{path}: duplicate prediction frame IDs')
    return table


def draw_frame(frame, frame_id, fps, state, observations, diagnostics,
               camera_matrix, distortion, trail):
    """Draw only the exact frame's state; never freeze a stale estimate over a gap."""
    height, width = frame.shape[:2]
    canvas = np.zeros((height, width + 340, 3), dtype=np.uint8)
    canvas[:, :width] = frame
    canvas[:, width:] = (30, 27, 24)
    def pixel(position):
        return project(*position, width, height, camera_matrix, distortion)
    status = 'NO STATE'
    plate_points = {}
    detail_points = []
    future_values = None
    if state is not None:
        values = np.array([state[name] for name in ['xc', 'yc', 'zc', 'body_yaw', 'r', 'dl', 'dh']])
        if not np.isfinite(values).all():
            state = None
            status = 'INVALID STATE'
        else:
            status = 'PREDICTION ONLY' if state.get('status') == 'prediction_only' else 'UPDATED'
            center = pixel(values[:3])
            if center is not None:
                trail.append(center)
                detail_points.append(center)
                if len(trail) > 1:
                    cv2.polylines(canvas, [np.array(trail, np.int32)], False, (180,180,180), 1, cv2.LINE_AA)
                cv2.drawMarker(canvas, center, (255,255,255), cv2.MARKER_STAR, 15, 2)
            for aid in range(4):
                color = ARMOR_COLORS[aid]
                corners = compute_plate_corners(*values, aid)
                detail_points.extend(draw_plate_outline(canvas, corners, camera_matrix,
                                                        distortion, width, height, color))
                point = pixel(compute_plate_position(*values, aid))
                if point is not None:
                    plate_points[aid] = point
                    detail_points.append(point)
                    color = ARMOR_COLORS[aid]
                    cv2.circle(canvas, point, 8, color, 2, cv2.LINE_AA)
                    cv2.putText(canvas, f'A{aid}', (point[0]+11, point[1]-9),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
                    if center is not None:
                        cv2.line(canvas, center, point, color, 1, cv2.LINE_AA)
            if state.get('prediction_horizon_ms', 0) > 0:
                future_values = np.array([state.get(name, np.nan) for name in
                    ['future_xc', 'future_yc', 'future_zc', 'future_body_yaw', 'r', 'dl', 'dh']])
                if not np.isfinite(future_values).all():
                    future_values = None
                else:
                    future_center = pixel(future_values[:3])
                    if future_center is not None:
                        detail_points.append(future_center)
                        cv2.drawMarker(canvas, future_center, (255,255,255), cv2.MARKER_DIAMOND, 13, 1)
                    for aid in range(4):
                        color = ARMOR_COLORS[aid]
                        corners = compute_plate_corners(*future_values, aid)
                        detail_points.extend(draw_plate_outline(canvas, corners, camera_matrix,
                                                distortion, width, height, color, dashed=True))
                        point = pixel(compute_plate_position(*future_values, aid))
                        if point is not None:
                            detail_points.append(point)
                            cv2.drawMarker(canvas, point, color, cv2.MARKER_DIAMOND, 11, 1)
                            cv2.putText(canvas, f'F{aid}', (point[0]+8, point[1]+16),
                                        cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    if state is None:
        trail.clear()
    accepted, rejected, unknown = 0, 0, 0
    for index, obs in enumerate(observations):
        diagnostic = diagnostics.get(index)
        reason = diagnostic.get('reason') if diagnostic else None
        accepted_flag = diagnostic.get('accepted') if diagnostic else False
        is_accepted = str(accepted_flag).lower() == 'true'
        is_rejected = diagnostic is not None and reason in ('invalid', 'innovation_gate', 'association_conflict')
        aid = int(diagnostic['armor_id']) if is_accepted else -1
        color = ARMOR_COLORS.get(aid, (0,220,255)) if is_accepted else ((40,80,255) if is_rejected else (0,220,255))
        accepted += int(is_accepted); rejected += int(is_rejected)
        unknown += int(not is_accepted and not is_rejected)
        point = pixel([obs['x'], obs['y'], obs['z']])
        if point is None:
            continue
        detail_points.append(point)
        marker = cv2.MARKER_TILTED_CROSS if is_rejected else cv2.MARKER_CROSS
        cv2.drawMarker(canvas, point, color, marker, 18, 2, cv2.LINE_AA)
        if is_accepted and aid in plate_points:
            cv2.line(canvas, point, plate_points[aid], color, 1, cv2.LINE_AA)
    lines = [
        ('ARMOR TRACKING', (255,255,255)),
        (f'Frame {frame_id} | {frame_id/fps:.2f} s', (210,210,210)),
        (status, (80,220,100) if status == 'UPDATED' else (0,200,255)),
        ('', (0,0,0)),
        ('ESTIMATE (same-frame EKF)', (210,210,210)),
    ]
    if state is not None:
        lines += [(f'{name}: {state[name]:.3f} {unit}', (235,235,235))
                  for name,unit in [('xc','m'),('yc','m'),('zc','m'),('body_yaw','rad')]]
        if 'w' in state:
            lines.append((f'w: {state["w"]:.2f} rad/s', (235,235,235)))
        lines += [(f'r: {state["r"]:.3f} m', (235,235,235)),
                  (f'dl / dh: {state["dl"]:.3f} / {state["dh"]:.3f}', (235,235,235))]
        if future_values is not None:
            lines += [(f'FORECAST: +{state["prediction_horizon_ms"]:g} ms', (255,255,255)),
                      (f'Target time: {state["prediction_timestamp"]/1000:.3f} s', (210,210,210))]
        else:
            lines.append(('FORECAST: off / unavailable', (170,170,170)))
    lines += [('',(0,0,0)), (f'Observations: {len(observations)}', (230,230,230)),
              (f'Accepted: {accepted}   Rejected: {rejected}',(230,230,230)),
              (f'Unclassified: {unknown}', (0,220,255)), ('',(0,0,0)),
              ('LEGEND', (255,255,255)), ('Solid A0-A3: current estimate', (220,220,220)),
              ('Dashed F0-F3: future forecast', (220,220,220)),
              ('+ : observation', (220,220,220)), ('Red X: rejected observation', (40,80,255)),
              ('Yellow +: not classified', (0,220,255)), ('Star / trail: center', (220,220,220)),
              ('Diamond: future center/plate', (220,220,220)),
              ('A/F pairs: same relative ID', (170,170,170))]
    for index,(label,color) in enumerate(lines):
        y = 32 + index*27
        if y >= height-10:
            break
        cv2.putText(canvas, label, (width+16, y), cv2.FONT_HERSHEY_SIMPLEX, .53, color, 1, cv2.LINE_AA)
    if detail_points and height >= 1000:
        points = np.array(detail_points)
        low = points.min(axis=0) - 35
        high = points.max(axis=0) + 35
        midpoint = (low + high) / 2
        span = np.maximum(high-low, [200, 140])
        x0, y0 = np.maximum(midpoint-span/2, [0,0]).astype(int)
        x1, y1 = np.minimum(midpoint+span/2, [width,height]).astype(int)
        crop = canvas[y0:y1, x0:x1].copy()
        scale = min(308/crop.shape[1], 180/crop.shape[0])
        detail = cv2.resize(crop, (max(1,int(crop.shape[1]*scale)), max(1,int(crop.shape[0]*scale))))
        left, top = width+16+(308-detail.shape[1])//2, height-195
        canvas[top:top+detail.shape[0], left:left+detail.shape[1]] = detail
        cv2.putText(canvas, 'TARGET DETAIL', (width+16,height-210),
                    cv2.FONT_HERSHEY_SIMPLEX,.5,(220,220,220),1,cv2.LINE_AA)
    return canvas


def render_video(video_path, prediction_csv, raw_csv, output_path, profile='1',
                 diagnostics_csv=None, start_frame=0, max_frames=None, trail_length=30):
    if start_frame < 0 or (max_frames is not None and max_frames <= 0) or trail_length < 0:
        raise ValueError('Invalid frame range or trail length')
    inputs = [Path(video_path), Path(prediction_csv), Path(raw_csv)]
    if diagnostics_csv is not None:
        inputs.append(Path(diagnostics_csv))
    output_path = Path(output_path)
    if any(output_path.resolve() == path.resolve() for path in inputs):
        raise ValueError('Output must not overwrite an input file')
    pred = load_table(prediction_csv, ['frame_id','xc','yc','zc','body_yaw','r','dl','dh'], unique=True)
    if 'coordinate_frame' in pred and not pred.coordinate_frame.eq('camera').all():
        raise ValueError('This video renderer requires camera-frame states; base-frame states need a per-frame camera transform before projection.')
    forecast_columns = {'prediction_horizon_ms','prediction_timestamp',
                        'future_xc','future_yc','future_zc','future_body_yaw'}
    if forecast_columns.intersection(pred.columns):
        if not forecast_columns.issubset(pred.columns) or 'timestamp' not in pred:
            raise ValueError('Incomplete forecast columns; rerun predictor_armor.exe')
        if (not np.isfinite(pred[list(forecast_columns)+['timestamp']].to_numpy()).all()
                or (pred.prediction_horizon_ms < 0).any()
                or not np.allclose(pred.prediction_timestamp-pred.timestamp,
                                   pred.prediction_horizon_ms, rtol=0, atol=1e-6)):
            raise ValueError('Invalid forecast timestamp or values')
    raw = load_table(raw_csv, ['frame_id','x','y','z'])
    if 'coordinate_frame' in raw and not raw.coordinate_frame.eq('camera').all():
        raise ValueError('This video renderer requires camera-frame observations; do not project base-frame positions directly.')
    pred_lookup = {int(r['frame_id']): r for r in pred.to_dict('records')}
    raw_lookup = {int(fid): group.to_dict('records') for fid,group in raw.groupby('frame_id',sort=False)}
    diagnostic_lookup = {}
    if diagnostics_csv is not None:
        diag = load_table(diagnostics_csv, ['frame_id','observation_index','accepted','armor_id','reason'])
        if diag.duplicated(['frame_id','observation_index']).any():
            raise ValueError('Duplicate observation diagnostics')
        for row in diag.to_dict('records'):
            if not np.isfinite(row['observation_index']) or row['observation_index'] % 1:
                raise ValueError('Invalid observation index in diagnostics')
            if str(row['accepted']).lower() == 'true' and row['armor_id'] not in range(4):
                raise ValueError('Invalid accepted armor ID in diagnostics')
            fid, index = int(row['frame_id']), int(row['observation_index'])
            group = raw_lookup.get(fid, [])
            if index < 0 or index >= len(group):
                raise ValueError('Diagnostics do not match raw observation rows')
            if np.isfinite(row.get('observed_distance', np.nan)) and 'distance' in group[index]:
                if not np.isclose(row['observed_distance'], group[index]['distance'], rtol=1e-6, atol=1e-6):
                    raise ValueError('Diagnostics do not match raw observation distances')
            diagnostic_lookup.setdefault(fid,{})[index] = row
    camera_matrix, distortion, expected_size = PROFILES[str(profile)]
    cap = cv2.VideoCapture(str(video_path))
    writer = None
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot open video: {video_path}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('Invalid video frame rate')
        if (width,height) != expected_size:
            raise ValueError(f'Camera profile {profile} expects {expected_size}, video is {(width,height)}')
        if start_frame >= total:
            raise ValueError('Start frame is outside the video')
        if any(fid >= total for fid in pred_lookup) or any(fid >= total for fid in raw_lookup):
            raise ValueError('CSV frame IDs exceed video length')
        output_path.parent.mkdir(parents=True,exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width+340,height))
        if not writer.isOpened():
            raise RuntimeError(f'Cannot create output video: {output_path}')
        cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
        count = 0
        trail = deque(maxlen=trail_length)
        while max_frames is None or count < max_frames:
            ok,frame = cap.read()
            if not ok:
                break
            fid = start_frame+count
            canvas = draw_frame(frame,fid,fps,pred_lookup.get(fid),raw_lookup.get(fid,[]),
                                diagnostic_lookup.get(fid,{}),camera_matrix,distortion,trail)
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
    elif not diagnostics.exists() and args.diagnostics is None:
        print('No diagnostics CSV: observations will be marked unclassified.')
        diagnostics = None
    try:
        render_video(args.video or root/'assets/video'/f'video_{args.suffix}.avi',
                     args.predictions or args.results_dir/f'armor_prediction_result_{args.suffix}.csv',
                     args.raw or args.data_dir/f'pose_raw_{args.suffix}.csv',
                     args.output or (args.output_dir or args.results_dir)/f'armor_video_{args.suffix}.mp4',
                     args.camera_profile or args.suffix, diagnostics,
                     args.start_frame,args.max_frames,args.trail_length)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
