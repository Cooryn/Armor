# RoboMaster Vision：装甲板检测与 EKF 跟踪

跟踪入口为 `src/Armor.cpp` / `Armor.exe`。它通过模块接口完成：

```text
视频 / 摄像头不连接下位机：camera → detector → solver/PnP → EKF → output
摄像头连接下位机：        camera → detector → solver/PnP → pose → EKF → output
                                                          ↑       ↓ 自瞄开启
                                                serial STATE    gimbal → serial TARGET
```

支持 OpenCV 摄像头和视频回放。`run_video()` 和 `run_camera()` 分别实现各自的处理循环。视频及不连接下位机的摄像头直接将 PnP 相机系观测送入 EKF，不创建串口或云台姿态对象。摄像头连接下位机时，使用反馈姿态将观测转换到固定基座系后进入 EKF，可独立开启或关闭自瞄。默认使用 USB OV2710 摄像头（MJPG、1280×720、请求 30 FPS）、不连接下位机、红色检测、单板 EKF、100 ms 提前预测和预览，实时镜头参数使用 2026-10-12 的 OV2710 实测标定。

只检测入口为 `src/Armor_detect.cpp` / `Armor_detect.exe`：`main()` 按输入来源调用独立的 `run_video()` 或 `run_camera()`，两个函数分别实现 camera → detector::detect → solver/PnP → 检测框与坐标窗口的处理循环。视频使用视频标定并按源 FPS 播放，摄像头使用实时相机内参并实时显示；不运行 EKF、固定系转换或云台控制，不打开串口。

生产代码按正确输入契约执行：图像、时间戳、观测、标定、枚举和参数由调用方正确提供，不做重复的有限值/格式/范围校验，不补默认数据，不兼容旧 CSV，也不自动换算法或求解器。检测几何筛选、PnP 候选选择、EKF 创新门控、漏检时只预测及控制时效/限位属于算法行为，继续保留。设备、文件和窗口 I/O 失败直接报错。

## 1. 模块接口

| 文件 | 接口及职责 |
| --- | --- |
| `camera.hpp/.cpp` | `camera::open/next`：OpenCV 摄像头或视频，输出图像、帧号和图像时间 |
| `lightbar_detector.hpp/.cpp` | `detector::detect`：灯条轮廓、筛选与配对；`detector::draw_armors` 绘制检测框 |
| `solver.hpp/.cpp` | `solver::solve_frame`：PnP、候选筛选、短时姿态关联，输出相机系 tvec/rvec 与观测 |
| `pose_base.hpp/.cpp` | `pose::update/convert/at_angles`：用最新 STATE 生成变换矩阵、转换完整装甲板位姿；串口由 Armor 接收 |
| `predictor*.hpp/.cpp` | 两种 EKF 各自提供 `update_frame`：选板、逐帧滤波、误差和未来几何；Armor.cpp 直接选择调用 |
| `gimbal.hpp/.cpp` | `gimbal::solve`：选未来目标板，计算绝对关节角并判断有效性 |
| `serial.hpp/.cpp` | `serial::receive/send_target/send_mode`：MC02 基础通信和独立心跳 |
| `log_output.hpp/.cpp` | `log_output::open/write/finish`：观测、预测、控制 CSV 和 RMSE |
| `video_output.hpp` | 检测/预测绘制、窗口预览、录像和时间戳补帧 |
| `Armor.cpp` | 顶部配置及上述接口调用，不包含检测、PnP、EKF或坐标变换算法 |
| `Armor_detect.cpp` | 相机/视频、检测、PnP、窗口与录像；左上角显示相机系 XYZ 坐标 |

`camera_frames.hpp` 已合并为 `pose_base.hpp`，类名为 `pose`。`camera_tracking.hpp/.cpp`、其旧仿真控制与独立入口已移除。`pose_base.cpp` 只保留共享转换实现，不再包含独立 main 或离线命令行。CMake 生成 Armor 和 Armor_detect；测试程序仅在 `BUILD_TESTING=ON` 时构建。

## 2. 构建和运行

依赖 C++17、Visual Studio 2022 C++ 工具、CMake、OpenCV 开发库、Eigen 和 Threads。Python 的 NumPy/pandas/Matplotlib/OpenCV 用于参考测试与绘图；Python OpenCV 包不能代替 C++ 开发库。

激活安装了 OpenCV/Eigen 的 Conda 环境后，可运行 `setup.ps1` 配置 `.venv` 并构建。已有环境时：

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DBUILD_TESTING=OFF -DOpenCV_DIR="$env:CONDA_PREFIX/Library/cmake" -DCMAKE_PREFIX_PATH="$env:CONDA_PREFIX/Library"
cmake --build build --config Release --target Armor --parallel
.\Armor.exe
```

程序不接受命令行参数，修改 `src/Armor.cpp` 顶部配置并重新编译：

```cpp
CameraSource camera_source = CAMERA; // 视频回放改为 VIDEO
const int camera_device = 1; // 当前电脑的外接 AX USB2.0 摄像头
const int video_camera_profile = 2; // 显式选择视频 1 或视频 2 的标定
const char *const video_file = "assets/video/video_2.avi";
const EnemyColor target_color = ENEMY_RED; // 或 ENEMY_BLUE
PredictorType predictor_type = PREDICTOR_SINGLE_PLATE;
const bool connect_mcu = false; // 仅摄像头模式：通信与坐标转换同时开启/关闭
const bool auto_aim_enabled = false; // 仅摄像头连接下位机时生效
const double prediction_horizon_ms = 100;
const char *const serial_port = "COM3";
```

Armor 和 Armor_detect 的预览始终开启，按 Esc 或关闭窗口退出。

| 输入 | `connect_mcu` | `auto_aim_enabled` | 工作流程 |
| --- | --- | --- | --- |
| VIDEO | 任意 | 任意 | 相机系检测、预测与输出，不通信、不转换坐标 |
| CAMERA | false | 任意 | 独立的摄像头处理循环，使用相机系检测、预测与输出 |
| CAMERA | true | false | 通信、接收姿态、转换到基座系并预测；发送 IDLE 和无效目标 |
| CAMERA | true | true | 通信、坐标转换、预测及自瞄控制 |

开关在启动前修改并重新编译，不是运行中的按键切换。视频模式忽略两个通信/自瞄开关；摄像头未连接下位机时忽略自瞄开关。

摄像头按实际采集时间戳处理，使用 `camera_recording_fps = 30` 录制 MJPG AVI：按时间戳补帧/丢帧以保持实际播放时长，检测、EKF 和 CSV 仍逐帧处理。视频文件模式按源 FPS 播放并导出处理结果。

摄像头请求 1280×720、30 FPS，格式为 MJPG；当前仅面向 Windows，使用 DirectShow。配置顺序为宽度、高度、请求 FPS、最后 MJPG，随后检查最终分辨率和 FOURCC。OpenCV 4.13 的 DirectShow `get(CAP_PROP_FPS)` 返回缓存请求值，不代表实际供帧速度。2026-10-12 实机测得设备 1 纯取帧约 29.826 FPS，有红色目标的单板与四板 EKF 完整显示/录像流程分别约 29.827、29.829 FPS。进一步枚举该设备的 DirectShow 模式，MJPG 720p 标称约 31 FPS，其余 MJPG 模式最高 30 FPS，没有 60 FPS 模式；FFmpeg 请求 720p MJPG 60 FPS 被拒绝，手动短曝光也未突破约 30 FPS。诊断与原始日志保存在 `tests/outputs/camera-60fps-diagnosis-20261012/`，此前的时序数据保存在 `tests/outputs/camera-fps-20261012/`。60 FPS 录像不代表每秒采集 60 张新图。当前电脑的设备 0 是内置 `ASUS FHD webcam`，设备 1 是外接 `AX USB2.0`（OV2710）；若设备顺序变化，应修改设备号。两个入口正常运行时不在终端输出帧数、性能统计或录像路径，错误信息仍会显示。

实时模式在顶部直接配置 `live_camera_matrix`、`live_distortion`、`camera_calibration` 和 `gimbal_config`。实时内参传入 solver；视频调用 `use_video_profile(video_camera_profile)`，不再根据文件名或空矩阵自动选择标定。视频 1 为 1440×1080，视频 2 为 1280×1024。两个入口的实时内参和畸变系数来自 `data/calibration/ov2710/20261012T082954_089195Z/calibration.json`：1280×720、9×6 内角点、25 mm 格子、31 张有效照片，整体重投影 RMS 为 0.521 像素。保持标定时的调焦位置和采集分辨率；换镜头或重新调焦后应重新标定，并同步更新两个入口。照片在画面边缘的覆盖较少，后续可以补拍四周位置；实际测距精度仍需用尺寸和距离已知的装甲板验证。

Esc 或关闭窗口结束处理并完成已处理帧的输出。输入、窗口、串口或写入失败返回非零退出码。连接下位机时，正常退出会发送无效目标及 IDLE；异常退出停止通信，下位机原有目标/链路超时保持生效。

### 只检测运行

修改 `src/Armor_detect.cpp` 顶部的输入来源、设备号、视频路径、标定 profile 和颜色，编译后运行：

```powershell
cmake --build build --config Release --target Armor_detect
.\Armor_detect.exe
```

默认使用 USB 摄像头和红色检测，预览始终开启，镜头参数使用上述 OV2710 实测标定。摄像头模式直接取图，不需要下位机；该文件的 `live_camera_matrix` 和 `live_distortion` 与跟踪入口使用同一组参数。窗口左上角列出每块 PnP 成功装甲板的 XYZ，单位 m，X 向右、Y 向下、Z 向前。框旁序号对应当前帧的坐标行，不表示跨帧身份。没有观测时显示 No armor detected，不保留上一帧坐标。Esc 或关闭窗口退出并完成录像保存；视频按源 FPS 播放。此入口保存带检测框和坐标的 MJPG AVI，不生成预测、控制或 CSV。

检测入口的摄像头录像保存到 `results/camera/detect/offline/camera_live.avi`，按采集时间戳补帧/丢帧；视频回放结果保存到 `results/video/detect/<视频主干>/<视频主干>.avi`，按源 FPS 逐帧保存。

## 3. 每帧行为

主流程保留视频和摄像头两个入口。视频及不连接下位机的摄像头，在检测与 PnP 后直接使用相机系四维观测进行预测和绘制，并计算模拟控制目标。两个函数分别实现处理循环，分别使用视频标定与实时相机标定，窗口分别显示 REPLAY 和 LIVE。

摄像头连接下位机时，每帧先尝试接收最新 STATE 并更新变换矩阵，检测与 PnP 后转换完整 tvec/rvec、重算基座系四维观测并调用所选 EKF。自瞄开启时解算并发送控制目标；关闭时保留姿态接收、坐标转换和预测，控制状态为 disabled，发送无效目标。此分支同时保留相机系原始观测和固定系观测。

实时图像和 STATE 共用 `serial::monotonic_time_ms()` 的 PC steady clock。STATE 在完整包校验后打接收时间，图像在取帧成功后打时间；这些仍是接收/取帧近似时刻，未校准曝光与传输延迟。视频使用 `frame_id × 1000 / FPS`，不会混用实时反馈时间。

只有摄像头连接下位机时，Armor 启动才等待第一条 STATE，超过 100 ms 未收到即报错，再开始取帧。每帧直接尝试接收最新反馈，收到后调用 `pose_base.update(state)`；没有新反馈时沿用上一份矩阵，不等待额外反馈。图像转换、绘图和控制共用这份矩阵，不缓存历史姿态、不插值或按图像时间同步。最新反馈近似代表图像时刻的姿态，控制层仍按实际反馈时间判断是否过期。

每帧只推进一次滤波。预测器假设输入观测正确，不再重复检查观测距离和俯仰角范围。首次观测初始化，从后续帧持续记录状态至结束，包括漏检帧；误差使用更新前预测，画面使用更新后状态。单板选择最近观测，距离并列保留第一条；Armor 保留四板关联与联合更新。漏检或被拒绝时继续预测，控制层继续根据预测位置跟踪。单板保留预测状态接近竖直轴时跳过更新的除零保护。

视频和未连接下位机的摄像头也逐帧输出 `control_target_*.csv`，不受 `auto_aim_enabled` 影响。模拟假设云台零位、无安装偏移，使用未来板位置和现有角度限位求解 yaw/pitch；角度单位为弧度，右转 yaw、抬头 pitch 为正。该结果相对当前相机方向，不发送串口，也不模拟电机运动。`valid` 仅表示解算目标有效，不表示已经发送；真实连接模式仍按下位机姿态解算。

## 4. 坐标、标定和控制

只使用相机系 C 和基座系 B。相机系为 OpenCV 坐标：X 右、Y 下、Z 前。基座和底盘固定，B 的原点在云台 yaw 转轴中心；Y 竖直向下，Z 沿启动时相机视线在地面上的投影，X 向右。首条姿态反馈确定参考水平朝向 `yaw_ref`，初始 pitch 保留相对地面的倾角：

```text
R_BC = Ry(yaw - yaw_ref) Rx(pitch) Rz(roll) R_mount
p_BC = Ry(yaw - yaw_ref) [yaw_to_pitch + Rx(pitch) Rz(roll) camera_in_pitch]
p_BA = R_BC p_CA + p_BC
R_BA = R_BC R_CA
```

| 标定字段 | 定义 |
| --- | --- |
| `angle_zero_rad` | 原始 yaw/pitch/roll 零位，rad |
| `angle_direction` | 每轴 +1 或 -1，校正角 = direction × (raw − zero) |
| `yaw_to_pitch` | yaw 转轴到 pitch 转轴，m |
| `camera_in_pitch` | pitch 转轴到相机光心，m |
| `camera_to_pitch` | 相机到安装轴的单位四元数 |

yaw 向右转为正，pitch 抬头为正且水平时为 0；分别对应绕 +Y、+X 的正旋转。roll 沿用绕 +Z 的约定，当前串口将其设为 0。初始相机倾斜时，基座仍保持水平；参考朝向使用安装旋转修正后的实际光轴水平投影。相机安装位移和朝向尚未标定，当前零安装偏移、单位旋转只是临时近似，真实参数需填入 `camera_calibration`。

`pose::update(state)` 用下位机的弧度角更新 `base_from_camera_`，首次反馈固定水平参考朝向，原点始终在 yaw 转轴中心；私有 `transform` 计算旋转和平移。`convert(camera_poses)` 使用当前矩阵转换整帧的位置与完整旋转，不修改原始相机观测。`at_angles(raw_angles)` 供控制器计算候选关节角对应的矩阵，不修改当前姿态。装甲板水平朝向直接由变换后的法向计算 `atan2(-normal_x, normal_z)`。初始视线竖直时无法确定水平投影，初始化会报错。

观测为 `[atan2(x,z), atan2(y,hypot(x,z)), norm(position), armor_yaw]`，位置单位 m、角度 rad。`target_pitch` 随 Y 向下为正；它与控制 pitch 的符号不同。固定系 distance 表示固定原点到板的距离，不再表示当前相机距离。

gimbal tracks the nearest visible future plate using calibrated absolute yaw/pitch. Both observed updates and `prediction_only` frames can produce valid tracking commands. Uninitialized or empty predictions, stale image/pose data (100 ms), no forward target, or joint-limit violations remain invalid; nonconvergence raises an error.

`GimbalConfig` 默认角度范围 yaw ±45°、pitch ±30°，最大位置方差 1 m²，是待实物配置的初值；填写为实际下位机关节范围。它只决定控制有效性，不改变 EKF 参数或下位机闭环。MC02 原有目标 100 ms、链路 200 ms 超时继续由固件执行。

PnP 与绘制板尺寸为 135×56 mm。尺寸在 `solver.cpp` 顶部配置，绘制尺寸位于 `output.cpp`；换板型时需同步。灯条使用完整像素轮廓拟合，配对按检测得分贪心选择，单根灯条最多用于一块板。

## 5. EKF 模型与多板处理

### 5.1 状态量

完整模型包含 11 个状态：

```text
[xc, vxc, yc, vyc, zc, vzc, body_yaw, w, r, dl, dh]
```

| 状态 | 含义 |
| --- | --- |
| `xc, yc, zc` | 车体参考中心位置 |
| `vxc, vyc, vzc` | 中心速度 |
| `body_yaw, w` | 车体参考朝向与角速度 |
| `r` | A0、A2 到旋转中心的半径 |
| `dl` | A1、A3 相对 A0、A2 的半径差，其半径为 `r + dl` |
| `dh` | A1、A3 的 y 坐标相对参考中心的偏移，正值表示更低 |

四块板的朝向相差 90°。A0–A3 是跟踪器内部相对编号，不是识别出的车头、车尾或数字标签。模型允许中心平移，不强制小车定轴旋转。

### 5.2 同帧多板

代码按含义统一命名：单条观测向量为 `z`，观测集合为 `observations`，残差为 `residual`，装甲板编号为 `armor_id`；滤波器状态和协方差为 `X`、`P`。主流程对象使用 `camera`、`solver`、`pose_base`、`prediction`、`control`、`serial`、`output`。时间戳以 `_ms`、`_s` 区分单位。相同矩阵类型但用途不同的数据仍使用不同名称。

每帧所有观测一起处理，先预测一次，再联合关联和更新：

1. 将观测转换为三维位置，并尝试关联 A0–A3。
2. 使用三维位置距离和装甲板朝向差门限筛选候选。
3. 按距离从小到大贪心匹配；距离相同时按输入顺序、板号排序。
4. 每条观测和板号只能使用一次；不再搜索全局最优组合。
5. 将选中观测联合更新到同一个车体状态，每条观测使用相同的固定协方差。

初始化帧只使用最近的一条有效观测，将其定义为 A0；该帧其他观测不参与更新。当前流程没有多车分组，同帧输入默认属于同一辆小车。

### 5.3 固定观测噪声与漏检

使用固定观测协方差 `R = diag(0.0016, 0.0004, 0.16, 0.0784)`，观测顺序为目标方位角、目标俯仰角、距离、装甲板朝向；角度项单位为 rad²，距离项为 m²。对应的有效标准差为 `0.04 rad、0.02 rad、0.40 m、0.28 rad`。检测评分、重投影误差和水平观察角度不改变观测权重。

所有观测共用一个固定 `R`，CSV 中的检测评分和重投影误差仅供检测诊断。调参记录与可复现方法见 [固定噪声调参说明](tests/fixed_noise_tuning.md)。

| 构造参数 | 默认值 | 含义 |
| --- | --- | --- |
| `max_yaw_error` | 45°，单位为弧度 | 观测与预测装甲板的朝向差上限 |
| `max_distance_error` | `0.5` m | 观测与预测装甲板的三维位置距离上限 |

过程噪声、固定观测协方差和初始化设置位于 `src/predictor_armor.cpp`，构造参数的默认值在 `include/predictor_armor.hpp` 的接口声明中。调参时应同时检查残差、中心漂移、接受率和运动变化时的响应。

漏检或所有观测被拒绝时，EKF 只预测。Armor 按源视频时间逐帧调用，初始化后一直导出到视频结束，不插值时间或观测，也不在漏检期间隐藏其预测几何。控制层在漏检时继续根据预测位置跟踪，滤波仍持续预测。

### 5.4 C++ 调用

四板模型直接接收 `std::vector<Eigen::Vector4d>` 观测，板号由关联算法自动确定。时间戳使用普通 `double`，调用方提供同一时钟上的递增时间。pose 只接收类型化反馈，不再解析离线标定/遥测 CSV；安装参数由构造函数传入。投影和预测中心通过 `optional` 表达不可见目标与未初始化状态。

滤波参数保存在各 EKF 类中。Armor.cpp 使用顶部的 `predictor_type`，通过 `if` 直接调用 `predictor_single::update_frame` 或 `predictor_armor::update_frame`。每个逐帧接口实现在对应 cpp 中，不经过通用模型选择器。`predictor_single.cpp` 只实现基础单板 EKF；固定列导出与 RMSE 位于 `output.cpp` 的 `output`。PnP 标定在进入循环前显式设置。

两个预测器保持声明与实现分离，算法和各自的逐帧接口放在对应 `.cpp`。所有模块均无自定义 namespace 和 using 类型别名。`predictor_single.hpp` 只保留单板模型声明；公共 `PredictorType`、`PredictionGeometry` 和 `PredictionResult` 定义在 `log_output.hpp`。两个模型统一接收 `std::vector<Eigen::Vector4d>`，每条观测为 `[target_yaw, target_pitch, distance, armor_yaw]`，需要三维位置时由观测逆变换得到。四板模型不再包含单板头文件。

Each module has one functional class: `camera.hpp` -> `camera`, `lightbar_detector.hpp` -> `detector`, `solver.hpp` -> `solver`, `pose_base.hpp` -> `pose`, `predictor_single.hpp` -> `predictor_single`, `predictor_armor.hpp` -> `predictor_armor`, `gimbal.hpp` -> `gimbal`, `serial.hpp` -> `serial`, and `log_output.hpp` -> `log_output`. Necessary data structs remain separate; detected plates use `armor`. Stateless operations are static class methods (for example, `detector::detect`, `predictor_armor::h`, and `gimbal::solve`). `log_output` owns CSV/RMSE logs; `video_output` owns drawing, preview and video recording.

The frame API is `update_frame(timestamp_ms, observations, horizon_ms)`. `PredictionResult::values` holds at most 9 model-specific log values; the output module writes frame metadata, centers and errors directly from their existing sources. Armor result CSV omits `vxc/vyc/vzc/w`, matched `armor_id`, and observation/accepted/rejected counts. The future-plate CSV retains `armor_id` to identify its four plates. Plotting and video replay columns are retained.

两个模型的状态、协方差、观测和雅可比均使用固定尺寸 Eigen 类型：基础单板为 8 维状态 / 4 维观测，装甲板为 11 维状态 / 4 维单板观测。四板联合更新按实际关联数量组合 4–16 维观测，矩阵存储上限固定为 16；卡尔曼增益直接通过 Eigen LDLT 求解，不构造逆矩阵，也不切换备用求解器。两个模型统一使用 `residual` 处理角度残差、Joseph 形式更新协方差，并对协方差做对称化。单板保留解析雅可比，四板保留数值雅可比。向量初始化使用 Eigen 构造函数或 `<<`，对角矩阵使用 `asDiagonal()`。

```cpp
#include "predictor_armor.hpp"

predictor_armor ekf; // 默认使用固定观测噪声
Eigen::Vector4d z(0, 0, 3, 0); // yaw, pitch, distance, plate yaw
ekf.update_frame(0, {z}, 50);
const auto prediction = ekf.update_frame(30, {z}, 50);
const auto &future = prediction.future;
// 导出：output.write(...)、output.finish()。
```

旧 Python 实现保存在 `tests/reference/predictor/`，仅供回归对照；原 `import predictor...` 接口已移除。
绘图仍使用 Python，继续读取相同的 CSV 文件。

### 5.5 未来位姿预测

两个模型都在 `update_frame()` 内使用状态副本计算未来位姿，通过返回值的 `future` 获取结果。
四板的 `future.center` 为未来中心，`future.plates` 按 A0–A3 保存四块板的 `[x, y, z, yaw]`，单位为米和弧度。
未来计算不修改当前滤波状态和协方差，也不计算未使用的未来协方差。

主 EKF 仍按帧间时间更新；未来预测使用本帧更新后的状态，缺测时使用推进到本帧的状态。
预测目标时间等于源图像时间加提前量，不按整数帧取整；30 FPS 下的 50 ms 仍精确使用 0.05 秒。
预测可以超出视频末尾，这些时刻没有后续视频观测可供验证。

摄像头连接下位机时，EKF 观测、状态和未来预测使用固定基座坐标系，由 pose 在滤波前转换；视频及不连接下位机的摄像头直接使用相机坐标系。Armor 仍采用水平旋转模型，不估计板面俯仰、横滚。
未来预测误差应与目标时刻、同一板号和同一坐标系下的后续观测比较。


## 6. 输出与绘图

所有正式输出按输入来源、模型和输入/摄像头模式分目录。同一次运行的观测、预测、RMSE、叠加视频和控制日志保存在同一目录，不再写入 `results/` 根目录。项目不再使用根目录下的 `data/`；测试参考样本保存在 `tests/reference/recordings/`。

```text
results/
  video/
    single/video_1/       # 单板模型处理 video_1.avi
    armor/video_1/
    detect/video_2/       # Armor_detect 的检测录像
  camera/
    single/offline/       # 摄像头不连接下位机，相机系
    single/mcu/           # 摄像头连接下位机，基座系
    armor/offline/
```

视频目录使用完整输入文件主干，CSV 后缀使用主干最后一个下划线后的内容；摄像头使用 `live` 后缀和 `camera_live.avi`。以下文件名均相对本次运行的目录：

| 文件 | 内容 |
| --- | --- |
| `pose_raw_1.csv` / `pose_raw_live.csv` | 相机系 PnP，coordinate_frame=camera |
| `pose_base_live.csv` | 仅连接下位机：固定系位置、完整 rvec/四元数、四维观测和基座原点元数据 |
| `camera_pose_live.csv` | 仅连接下位机：每帧图像时间、实际反馈时间 `pose_timestamp_ms`、使用的 T_BC 旋转/平移及固定原点元数据，包括无观测帧 |
| `video_1.avi` / `camera_live.avi` | 检测框、当前与未来几何；相机系直接投影，基座系先逆变换再投影 |
| `control_target_*.csv` | yaw/pitch、valid、目标板号和状态，每帧一行；未连接时为相机零位下的模拟控制目标 |
| `prediction_result_*.csv` | SinglePlate 原 13 列，基座系另加 coordinate_frame |
| `armor_prediction_result_*.csv` | Armor 原 32 列，基座系另加 coordinate_frame |
| `armor_future_prediction_*.csv` | 未来四板，原 10 列，基座系另加 coordinate_frame |
| `*rmse_result_*.txt` | 原六位小数 RMSE 格式；没有有效残差为 nan |

同一输入、模型和摄像头模式的同名输出会覆盖；不同模型、视频和摄像头模式互不覆盖。摄像头连接下位机时，状态及未来文件的 coordinate_frame 为 base；pose_base/camera_pose 记录 `base_reference_timestamp_ms` 和 `base_reference_yaw_rad`，原点元数据 `base_origin_x_m/y_m/z_m` 固定为 0，表示 yaw 转轴中心，配套文件必须来自同一次运行。视频及不连接下位机的摄像头使用相机系预测 CSV 格式，输出 pose_raw、预测/RMSE、control_target 和叠加视频，不生成 pose_base 或 camera_pose。初始化帧不写预测状态行；全程无观测时预测状态文件只有表头，RMSE 为 nan，control_target 仍逐帧记录无效目标。

结果不按日期分目录，不创建 history；每次运行直接覆盖对应输入、模型和摄像头模式下的同名输出。集成测试和 NumPy 参考运行默认写入 `tests/outputs/`，不会修改正式结果。手动生成的曲线默认写入对应结果目录的 `plots/` 子目录，同名绘图也直接覆盖。

当前板用实线/圆点，未来板用虚线/菱形。投影会逆变换所有三维角点，不会只改位置或 yaw。模型仍只估计固定系水平板朝向，四板均显示，不模拟遮挡。残差是先验预测相对观测的误差，不代表真实定位精度。

```powershell
.\.venv\Scripts\python -B -m plot.raw --suffix 1
.\.venv\Scripts\python -B -m plot.armor --suffix 1
# SinglePlate 配置运行后：
.\.venv\Scripts\python -B -m plot.basic --suffix 1
# 当前默认的摄像头离线单板结果：
.\.venv\Scripts\python -B -m plot.basic --suffix live
# 连接下位机的摄像头结果：
.\.venv\Scripts\python -B -m plot.basic --suffix live --camera-mode mcu
```

`plot.raw` 查看相机系原始观测，可用 `--model single|armor` 选择结果目录；`plot.armor` 按结果的坐标标签选择对应观测，`plot.basic` 只读取预测 CSV 并输出误差曲线。`--results-dir` 可指定完整结果目录，`--data-dir` 可单独指定观测目录，默认与结果目录相同。视频叠加统一由 C++ 输出，Python 只生成分析曲线。更多参数见 [plot/README.md](plot/README.md)。

## 7. 两种预测模型

在 `src/Armor.cpp` 修改 `predictor_type` 选择模型，重新编译后运行同一个 Armor.exe。

| 模型 | 状态与观测 | 每帧更新 |
| --- | --- | --- |
| SinglePlate | `[x,vx,y,vy,z,vz,armor_yaw,yaw_rate]`；方位、俯仰、距离、装甲板朝向 | 最近有效板，距离相同保留第一条 |
| Armor | 11 维：中心/速度、yaw/w、r/dl/dh | 四板关联与联合更新 |

SinglePlate 使用匀速状态转移和非线性球坐标观测：

```text
yaw      = atan2(x, z)
pitch    = atan2(y, sqrt(x² + z²))
distance = sqrt(x² + y² + z²)
armor_yaw = state[6]
```

使用解析雅可比、LDLT 求增益、Joseph 协方差更新和完整 8×8 协方差。
`R=diag(0.000036,0.000025,0.01,0.0576)`，位置 `q_pos=60`、板朝向 `q_yaw=75`，每轴过程噪声块为
`q * [[dt⁴/4,dt³/2],[dt³/2,dt²]]`，初始协方差 `10I`。原点或竖直轴附近跳过观测更新。
朝向按 `armor_yaw += yaw_rate * dt` 推进，角度残差归一化到 `[-π,π)`。当前框使用滤波后的 yaw，未来框使用预测 yaw；漏检时继续推进位置和朝向。按同一块板连续跟踪，不做换板判断或姿态重置。仅预测 yaw，不预测完整三维 rvec。原有位置误差 CSV 和 RMSE 列保持不变。

2026-10-12 按用户要求恢复此前调整后的单板噪声参数（candidate `c083`）。100 ms 提前量下的噪声调参结果保留在 [单板噪声调参记录](tests/single_noise_tuning.md)。



Armor 的固定参数见第 5 节。保留既有几何定义，不重新解释 yaw 或半径。

两种模型都使用每帧的源时间间隔推进，输入观测由上游保证正确；无观测时只预测。
SinglePlate 的 13 列记录更新前预测，Armor 的中心状态记录更新后状态，
所有误差均来自更新前预测。RMSE 只累计有限残差，Armor 只累计最近被接受板的残差。


## 8. 验证

```powershell
powershell -ExecutionPolicy Bypass -File .\tests\run.ps1
.\.venv\Scripts\python -B tests/verify_armor_video.py --max-source-frames 90
```

原生测试覆盖协议拆包/CRC、检测/PnP、两种 EKF、完整姿态变换、角度跨界、固定原点、最新反馈及无新反馈时沿用矩阵、移动相机下固定目标、绝对光轴对准、漏检/限位/失效控制，以及帧号、CSV、视频输出。NumPy 对照验证全部结果列、未来几何和 RMSE。视频集成脚本临时修改配置并验证红/蓝及两种模型，最后恢复源码和默认程序。

此次软件验证日志集中于 `tests/outputs/armor-unified/`。尚未接入实际摄像头、USB-TTL 或 MC02 电机；软件几何验证不能代替实物标定、延迟测量和收发联调。测试接口与产物说明见 [tests/README.md](tests/README.md)。

## 9. 达妙 MC02 基础串口收发

`serial` 对接提供的 `rm_gimbal_26` 固件 `host_link.c`，使用 Windows 串口 API，固定 **921600 baud、8N1、无流控**。固件 USART10 引脚为 **PE3/TX → USB-TTL RX，PE2/RX ← USB-TTL TX，GND 共地**，使用 3.3V TTL。板上连接器位置和供电接法见 [达妙 MC02 官方资料](https://gitee.com/kit-miao/dm-mc02)。

接口位于 `include/serial.hpp`，实现全部放在 `src/serial.cpp`：

| 接口 | 作用 |
| --- | --- |
| `open("COM3")` | 打开实际串口，启动通信线程；COM3 仅为示例 |
| `close()` | 等待已排队数据发完，结束线程并关闭串口；通信失败在清理后报告，析构时自动清理 |
| `receive(state)` | 取走最新收到的姿态，有新样本返回 true，否则返回 false |
| `send_target(yaw, pitch, valid)` | 更新最新目标，后台约 40 Hz 持续发送；100 ms 未更新后自动清除有效位 |
| `send_mode(mode)` | 排队发送模式，0=IDLE、1=MANUAL、2=AUTO_AIM、3=STABILIZE |

`serial` 只保存最新姿态；一次读到多条 STATE 时返回最后一条。姿态包含 yaw、pitch、roll、mode、flags 和 PC 接收时间戳。角度单位 rad，时间单位 ms；所有实时 PC 时间戳应使用同一个 `serial::monotonic_time_ms()`，不混用录像帧时间。当前固件 yaw 是关节角、pitch 是 IMU 姿态角、roll=0，flags=0 是预留值，不作为姿态有效性标志。

协议为 `A5 CMD LEN PAYLOAD CRC8`，float32 小端。CRC-8/ATM 多项式 0x07、初值 0，校验 CMD、LEN 和 PAYLOAD。TARGET=1，载荷 9 字节；STATE=2，载荷 14 字节；MODE=3，载荷 1 字节；HEARTBEAT=4，载荷为空。接收端按 LEN + 4 分帧，载荷最多 24 字节，支持分包、连续包和双向 HEARTBEAT；心跳不刷新姿态时间戳。噪声、错误长度或 CRC 错误会重新寻找帧头。按 v1.0 忽略上行 roll（置零），flags 保留但不参与有效性判断。

后台线程串行写包，每 50 ms 独立发送心跳；读取等待 5 ms、写超时 20 ms。串口打开或收发失败通过接口抛出异常，并停止通信，重新连接由调用方显式 close/open。目标有效性、反馈新鲜度、角度限位、模式决策和日志由上层负责；TARGET 每约 25 ms 发送一次，尚无目标或最新目标超过 100 ms 未更新时发送 valid=0，防止重复发送延长陈旧目标的有效期。模块不自动重连，也不生成 CSV 或独立监视程序。

构建模块：

```powershell
cmake --build build --config Release --target serial
```

串口模块由 Armor 直接调用。视频回放及不连接下位机的摄像头不打开串口；摄像头连接下位机时，根据自瞄开关发送 AUTO_AIM 或 IDLE，每帧发送 TARGET（关闭自瞄时 valid=0），正常结束时发送无效目标与 IDLE 并完成待发写入。尚未连接 MC02 做实物收发或运动验证。

### 统一视频界面

SinglePlate、Armor 和 Armor_detect 统一使用左上角黄色文字、黑色描边及相同行距，显示模式、帧号、状态和各观测板的相机系 XYZ（m）。框旁序号对应当前帧观测。Armor 保留四板当前实线框和未来虚线框。

串口内部的 TARGET/MODE 在发送接口直接打包，STATE 在后台 run() 直接解析，CRC 为 cpp 内共用函数。已删除独立协议类、发送转发函数及 flush；close 请求停止后，线程发完剩余 MODE/TARGET 再退出，无需发送完成条件变量。
