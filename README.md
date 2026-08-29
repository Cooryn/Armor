# RoboMaster Vision：装甲板检测与 EKF 预测

RoboMaster 视觉项目，包含两部分：

- **C++ 装甲板检测**：基于 OpenCV 的灯条提取、装甲板配对与 PnP 位姿解算。
- **Python EKF 预测**：基于 PnP 观测的扩展卡尔曼滤波，估计车体中心、朝向与底盘结构参数，并对目标进行一步预测。

## 特性

### 1. 装甲板检测与位姿解算（C++）

- 颜色提取（红 / 蓝）、灯条筛选、灯条配对成装甲板。
- `solvePnP` 解算装甲板在相机坐标系下的位姿 `[x, y, z]` 与朝向。
- 输出观测向量 `[x, y, z, target_yaw, target_pitch, distance, armor_orientation_yaw]`，供后续 EKF 使用。
- 内置 Debug 面板与滑动条，可实时调参。
- 视频模式自动导出带标注视频，并把逐帧位姿写入 CSV。

### 2. EKF 预测（Python）

三个递进版本的预测器：

1. **`predictor.py`** —— 基础 6 维匀速模型（CV）
   状态 `[x, vx, y, vy, z, vz]`，观测量为 PnP 解算出的 `[x, y, z]`。

2. **`predictor_polar.py`** —— 9 维极坐标模型
   状态 `[xc, vxc, yc, vyc, zc, vzc, body_yaw, w, r]`，观测量为
   `[target_yaw, target_pitch, distance, armor_orientation_yaw]`。估计车体旋转中心、朝向角与底盘半径 `r`。

3. **`predictor_armor.py`** —— 11 维完整模型（最终版）
   状态 `[xc, vxc, yc, vyc, zc, vzc, body_yaw, w, r, dl, dh]`，额外估计侧板相对底盘中心的横向偏移 `dl` 与高度偏移 `dh`。
   通过 **多重假设检验（MHT）** 在 4 块装甲板中自动选择最优 ID，使用 **Joseph 形式** 更新协方差以保证对称正定。

## 目录结构

```text
Armor/
├── CMakeLists.txt                  # C++ 构建脚本
├── include/                        # C++ 头文件
│   ├── input_stream.hpp            # 输入流抽象（视频 / 相机 / 图片）
│   ├── lightbar_detector.hpp       # 灯条检测与装甲板配对声明
│   └── solver.hpp                  # PnP 位姿解算声明
├── src/                            # C++ 实现
│   ├── main.cpp                    # 主程序：参数解析、Debug 面板、CSV/视频输出
│   ├── lightbar_detector.cpp       # 颜色提取、灯条筛选、装甲板配对与绘制
│   └── solver.cpp                  # solvePnP 位姿解算
├── predictor/                      # Python 预测器（三个递进版本）
│   ├── predictor.py                # 6 维 CV 模型 EKF
│   ├── predictor_polar.py          # 9 维极坐标 EKF
│   └── predictor_armor.py          # 11 维完整模型 EKF（最终版）
├── plot_raw_data.py                # 绘制原始位姿 / 偏航观测曲线
├── visualize_armor_video.py        # 将 EKF 预测的四块装甲板叠加到原视频
├── data/                           # C++ 导出的逐帧位姿 CSV（运行时生成）
│   ├── pose_raw_1.csv
│   └── pose_raw_2.csv
├── assets/                         # 输入素材
│   ├── image/                      # image_0.jpg ~ image_10.jpg
│   └── video/                      # video_1.avi、video_2.avi
└── results/                        # 运行输出（CSV / PNG / MP4 / TXT，运行时生成）
```

> `data/` 与 `results/` 为运行时生成的产物，已加入 `.gitignore`。

## 依赖

**C++ 检测：**

- C++17
- CMake ≥ 3.10
- OpenCV ≥ 4.0
- Eigen3
- `ffmpeg`（视频模式后处理，用于将 AVI 转码为 MP4）

**Python 预测：**

- Python 3.8+
- `numpy`
- `pandas`
- `matplotlib`
- `opencv-python`（仅 `visualize_armor_video.py` 需要）

## 构建与运行

### 1. 构建 C++ 检测程序

```bash
mkdir build && cd build
cmake ..
make -j4
./Armor
```

可执行文件 `Armor` 会输出到项目根目录。

### 2. 运行 C++ 检测

```bash
./Armor [运行模式] [目标颜色] [文件名]
```

| 参数 | 取值 | 默认 |
| --- | --- | --- |
| 运行模式 | `image` / `video` | `image` |
| 目标颜色 | `red` / `blue` | `red` |
| 文件名 | 如 `image_1.jpg`、`video_2.avi` | `image_1.jpg` / `video_1.avi` |

例如：

```bash
./Armor video red video_1.avi
./Armor image blue image_3.jpg
```

程序会先在当前目录查找输入文件，找不到则回退到 `./assets/<mode>/<filename>`。

**Debug 面板滑动条**（实时调参）：

- Gray Thresh / Color Thresh —— 高光与颜色阈值
- Max Angle Diff —— 灯条角度差上限
- Max Len Ratio(x10) —— 左右灯条长度比上限
- Min Aspect(x10) —— 装甲板宽高比下限
- Max Y Diff(x10) —— 左右灯条 Y 轴错位上限

**交互按键：**

- 图片模式：按 `S` 保存当前满意结果（PNG），按 `ESC` 退出。
- 视频模式：按 `ESC` 提前退出；正常结束时自动导出视频。

### 3. 运行 Python 预测

先由 C++ 视频模式生成 `data/pose_raw_*.csv`，再运行对应的 Python 脚本：

```bash
# 绘制原始位姿 / 偏航观测曲线
python plot_raw_data.py

# 基础 6 维 EKF
python predictor/predictor.py

# 9 维极坐标 EKF
python predictor/predictor_polar.py

# 11 维完整模型 EKF（最终版）
python predictor/predictor_armor.py

# 将 EKF 预测的四块装甲板叠加到原视频
python visualize_armor_video.py
```

脚本默认处理 `suffix = "2"`（即 `pose_raw_2.csv`），可在各脚本 `__main__` 中的 `suffix` 变量处切换。

## 数据链路

```text
assets/video/*.avi  ──►  C++ (Armor)  ──►  data/pose_raw_*.csv
                                                  │
                                   ┌──────────────┼───────────────┐
                                   ▼              ▼               ▼
                           predictor/predictor.py   predictor/predictor_polar.py   predictor/predictor_armor.py
                                   │              │               │
                                   ▼              ▼               ▼
                              results/*.csv / *.png / *.txt / *.mp4
```

C++ 导出的 CSV 列：

```text
frame_id, timestamp, x, y, z, target_yaw, target_pitch, distance, armor_orientation_yaw
```

## 输出文件说明

各预测脚本在 `results/` 下生成的文件以 `suffix` 区分（如 `_1`、`_2`）：

| 脚本 | 输出 |
| --- | --- |
| `plot_raw_data.py` | `pose_raw_curve_*.png`、`raw_yaw_curve_*.png` |
| `predictor.py` | `prediction_result_*.csv`、`rmse_result_*.txt`、`prediction_error_curve_*.png`、`top_down_trajectory_*.png` |
| `predictor_polar.py` | `polar_prediction_result_*.csv`、`polar_rmse_result_*.txt`、`polar_prediction_curve_*.png`、`polar_error_curve_*.png` |
| `predictor_armor.py` | `armor_prediction_result_*.csv`、`armor_rmse_result_*.txt`、`top_down_trajectory_*.png`、`body_yaw_curve_*.png`、`folded_armor_yaw_curve_*.png`、`armor_prediction_error_curve_*.png` |
| `visualize_armor_video.py` | `armor_video_1.mp4`（叠加四块预测装甲板与原视频） |

## 评估

- 基础预测器输出 `RMSE_x`、`RMSE_z`、`RMSE_yaw`、`RMSE_distance`。
- 极坐标与完整模型输出 `RMSE_target_yaw`、`RMSE_target_pitch`、`RMSE_distance`、`RMSE_armor_yaw`。
- 完整模型还会绘制俯视轨迹图（观测 vs 预测的装甲板位置），用于直观验证滤波跟踪与 MHT 的装甲板识别是否正确。

## 注意事项

- 相机内参在 [src/main.cpp](src/main.cpp) 中硬编码，`video_2` 使用单独一套内参，其余素材使用默认内参。
- `visualize_armor_video.py` 读取的是 `results/armor_prediction_result_1.csv`（`suffix = "1"`），运行前需先用 `predictor/predictor_armor.py` 处理对应的 `pose_raw_1.csv`。
