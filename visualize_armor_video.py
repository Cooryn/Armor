"""
将 EKF 预测的四块装甲板位置叠加绘制到原始视频上。
输出: results/armor_video_1.mp4
"""
import pandas as pd
import numpy as np
import cv2
import os
import sys

# ==========================================
# 相机内参 (video_1, 来自 C++ 标定)
# ==========================================
FX, FY = 1286.307063384126, 1288.1400736562441
CX, CY = 645.34450819155256, 483.6163720308021

# ==========================================
# 四块装甲板的颜色 (BGR for OpenCV)
# ==========================================
ARMOR_COLORS = {
    0: (0, 0, 255),     # 红 - 前板 (Front)
    1: (255, 0, 0),     # 蓝 - 右板 (Right)
    2: (0, 255, 0),     # 绿 - 后板 (Back)
    3: (0, 165, 255),   # 橙 - 左板 (Left)
}
ARMOR_NAMES = {0: "Front", 1: "Right", 2: "Back", 3: "Left"}

CENTER_COLOR = (255, 255, 255)  # 白色
OBS_COLOR = (0, 255, 255)       # 黄色 (实际检测)


def project(x, y, z):
    """3D 相机坐标 → 2D 图像坐标"""
    if z <= 0.1:
        return None
    u = FX * x / z + CX
    v = FY * y / z + CY
    if u < 0 or u >= 1440 or v < 0 or v >= 1080:
        return None
    return int(u), int(v)


def compute_plate_position(xc, yc, zc, body_yaw, r, dl, dh, armor_id):
    """用 EKF 状态计算第 armor_id 块装甲板的 3D 位置"""
    plate_yaw = body_yaw + armor_id * (np.pi / 2.0)
    is_side = (armor_id % 2 != 0)
    r_i = r + dl if is_side else r
    y_i = yc + dh if is_side else yc
    xa = xc - r_i * np.sin(plate_yaw)
    za = zc - r_i * np.cos(plate_yaw)
    ya = y_i
    return xa, ya, za


def draw_cross(img, cx, cy, size=6, color=(255, 255, 255), thickness=2):
    """绘制十字标记"""
    cv2.line(img, (cx - size, cy), (cx + size, cy), color, thickness)
    cv2.line(img, (cx, cy - size), (cx, cy + size), color, thickness)


def main():
    pred_csv = "results/armor_prediction_result_1.csv"
    raw_csv = "data/pose_raw_1.csv"
    video_path = "assets/video/video_1.avi"
    output_path = "results/armor_video_1.mp4"

    if not os.path.exists(pred_csv):
        print(f"错误: 找不到 {pred_csv}，请先运行 predictor/predictor_armor.py")
        sys.exit(1)
    if not os.path.exists(video_path):
        print(f"错误: 找不到视频 {video_path}")
        sys.exit(1)

    pred = pd.read_csv(pred_csv)
    raw = pd.read_csv(raw_csv)

    ekf_lookup = {}
    for _, row in pred.iterrows():
        ekf_lookup[int(row['frame_id'])] = row

    pnp_lookup = {}
    for _, row in raw.iterrows():
        fid = int(row['frame_id'])
        if fid not in pnp_lookup:
            pnp_lookup[fid] = []
        pnp_lookup[fid].append(row)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频: {total_frames} 帧, {fps} fps, {width}x{height}")

    os.makedirs("results", exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    last_ekf_row = None  # 用于缺失帧的向前填充
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id = frame_idx + 1
        ekf_row = ekf_lookup.get(frame_id)

        # 缺失帧：沿用上一帧的 EKF 状态（向前填充）
        if ekf_row is None and last_ekf_row is not None:
            ekf_row = last_ekf_row
        if ekf_row is not None:
            last_ekf_row = ekf_row

        overlay = frame.copy()

        cv2.putText(overlay, f"Frame {frame_id}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if ekf_row is not None:
            xc, yc, zc = ekf_row['xc'], ekf_row['yc'], ekf_row['zc']
            body_yaw = ekf_row['body_yaw']
            r, dl, dh = ekf_row['r'], ekf_row['dl'], ekf_row['dh']

            # 车体中心
            center_pt = project(xc, yc, zc)
            if center_pt:
                cv2.drawMarker(overlay, center_pt, CENTER_COLOR,
                               cv2.MARKER_STAR, 15, 2)

            # 车身朝向线
            if center_pt:
                tip_x = xc - r * np.sin(body_yaw)
                tip_z = zc - r * np.cos(body_yaw)
                tip_pt = project(tip_x, yc, tip_z)
                if tip_pt:
                    cv2.line(overlay, center_pt, tip_pt, (255, 255, 255), 2)

            # 四块预测装甲板
            for aid in range(4):
                xa, ya, za = compute_plate_position(xc, yc, zc, body_yaw, r, dl, dh, aid)
                pt = project(xa, ya, za)
                if pt:
                    color = ARMOR_COLORS[aid]
                    cv2.circle(overlay, pt, 8, color, -1)
                    cv2.circle(overlay, pt, 10, color, 2)
                    cv2.putText(overlay, f"A{aid}", (pt[0] + 12, pt[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 实际 PnP 检测位置
        pnp_dets = pnp_lookup.get(frame_id, [])
        for det in pnp_dets:
            pt = project(det['x'], det['y'], det['z'])
            if pt:
                if ekf_row is not None:
                    yaw_diff = (det['armor_orientation_yaw'] - ekf_row['body_yaw'] + np.pi) % (2 * np.pi) - np.pi
                    det_aid = int(round(yaw_diff / (np.pi / 2.0))) % 4
                else:
                    det_aid = 0
                det_color = ARMOR_COLORS.get(det_aid, OBS_COLOR)
                draw_cross(overlay, pt[0], pt[1], size=9, color=det_color, thickness=2)

        # 图例
        legend_x = width - 150
        legend_y = 40
        for aid in range(4):
            y = legend_y + aid * 25
            cv2.circle(overlay, (legend_x, y), 6, ARMOR_COLORS[aid], -1)
            cv2.putText(overlay, ARMOR_NAMES[aid],
                        (legend_x + 12, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        ARMOR_COLORS[aid], 2)

        cv2.drawMarker(overlay, (legend_x, legend_y + 110), CENTER_COLOR, cv2.MARKER_STAR, 10, 2)
        cv2.putText(overlay, "Center", (legend_x + 12, legend_y + 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, CENTER_COLOR, 2)
        draw_cross(overlay, legend_x, legend_y + 130, size=6, color=OBS_COLOR)
        cv2.putText(overlay, "PnP det", (legend_x + 12, legend_y + 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, OBS_COLOR, 2)

        if ekf_row is not None:
            info_lines = [
                f"body_yaw: {body_yaw:.2f} rad",
                f"r: {r:.3f}  dl: {dl:.3f}  dh: {dh:.3f}",
                f"中心: xc={xc:.3f} zc={zc:.3f}",
            ]
            for i, line in enumerate(info_lines):
                cv2.putText(overlay, line, (10, height - 60 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        writer.write(overlay)

        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"  处理中... {frame_idx}/{total_frames}")

    cap.release()
    writer.release()
    print(f"✅ 可视化视频已保存至: {output_path}")
    print(f"   共处理 {frame_idx} 帧")


if __name__ == "__main__":
    main()
