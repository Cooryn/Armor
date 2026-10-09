# RoboMaster Vision：装甲板检测与 EKF 跟踪

跟踪入口为 `src/Armor.cpp` / `Armor.exe`。它通过模块接口完成：

```text
Camera → lightbar_detector → Solver/PnP → PoseBase → EKF → Gimbal → Serial
  图像                         ↑                                控制目标 ↓
                               └──── MC02 STATE 姿态反馈 ───────────────┘
                                 Output：CSV、RMSE、叠加视频及窗口
```

支持 OpenCV 摄像头和视频回放。实时图像使用下位机姿态转换到固定坐标系后进入 EKF；视频回放使用明确的零角度虚拟相机，不接现场串口。默认保留视频 2、红色检测、四板 Armor EKF、50 ms 提前预测和预览。三种 EKF 的状态模型、噪声参数和四板联合更新保持不变。

只检测入口为 `src/Armor_detect.cpp` / `Armor_detect.exe`：Camera → detectArmors → Solver/PnP → 检测框与坐标窗口，不运行 EKF、固定系转换或云台控制，不打开串口。

生产代码按正确输入契约执行：图像、时间戳、观测、标定、枚举和参数由调用方正确提供，不做重复的有限值/格式/范围校验，不补默认数据，不兼容旧 CSV，也不自动换算法或求解器。检测几何筛选、PnP 候选选择、EKF 创新门控、漏检时只预测及控制时效/限位属于算法行为，继续保留。设备、文件和窗口 I/O 失败直接报错。

## 1. 模块接口

| 文件 | 接口及职责 |
| --- | --- |
| `camera.hpp/.cpp` | `Camera::open/next`：OpenCV 摄像头或视频，输出图像、帧号和图像时间 |
| `lightbar_detector.hpp/.cpp` | `detectArmors`：灯条轮廓、筛选与配对；`drawArmors` 绘制检测框 |
| `solver.hpp/.cpp` | `Solver::solve_frame`：PnP、候选筛选、短时姿态关联，输出相机系 tvec/rvec 与观测 |
| `pose_base.hpp/.cpp` | `PoseBase::receive/synchronize/convert`：缓存 STATE、同步姿态、转换完整装甲板位姿 |
| `predictor*.hpp/.cpp` | 三种 EKF 各自提供 `update_frame`：选板、逐帧滤波、误差和未来几何；Armor.cpp 直接选择调用 |
| `gimbal.hpp/.cpp` | `solve_gimbal`：选未来目标板，计算绝对关节角并判断有效性 |
| `serial.hpp/.cpp` | `Serial::receive/send_target/send_mode`：MC02 基础通信和独立心跳 |
| `output.hpp/.cpp` | `Output::open/write/finish` 与 `PredictionOutput`：观测、预测、控制文件和 RMSE，逆变换绘制、视频和窗口 |
| `Armor.cpp` | 顶部配置及上述接口调用，不包含检测、PnP、EKF或坐标变换算法 |
| `Armor_detect.cpp` | 相机/视频、检测、PnP及窗口；左上角显示相机系 XYZ 坐标 |

`camera_frames.hpp` 已合并为 `pose_base.hpp`，类名为 `PoseBase`。`camera_tracking.hpp/.cpp`、其旧仿真控制与独立入口已移除。`pose_base.cpp` 只保留共享转换实现，不再包含独立 main 或离线命令行。CMake 生成 Armor 和 Armor_detect；测试程序仅在 `BUILD_TESTING=ON` 时构建。

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
CameraSource camera_source = VIDEO; // 或 CAMERA
const int camera_device = 0;
const double camera_recording_fps = 30;
const int video_camera_profile = 2; // 显式选择视频 1 或视频 2 的标定
const char *const video_file = "assets/video/video_2.avi";
const EnemyColor target_color = ENEMY_RED; // 或 ENEMY_BLUE
PredictorType predictor_type = PREDICTOR_ARMOR;
bool preview = true;
const double prediction_horizon_ms = 50;
const char *const serial_port = "COM3";
```

视频路径相对编译时的项目根目录解析。预览模式按源 FPS 播放视频；关闭预览时不限速。摄像头按实际取帧时刻处理，录制帧率显式使用 `camera_recording_fps`，不参与实时滤波时钟。

实时模式在顶部直接配置 `live_camera_matrix`、`live_distortion`、`camera_calibration` 和 `gimbal_config`。实时内参传入 Solver；视频调用 `use_video_profile(video_camera_profile)`，不再根据文件名或空矩阵自动选择标定。视频 1 为 1440×1080，视频 2 为 1280×1024。

Esc 或关闭窗口结束处理并完成已处理帧的输出。输入、窗口、串口或写入失败返回非零退出码。正常退出实时模式会发送无效目标及 IDLE；异常退出停止通信，下位机原有目标/链路超时保持生效。

### 只检测运行

修改 `src/Armor_detect.cpp` 顶部的输入来源、设备号、视频路径、标定 profile 和颜色，编译后运行：

```powershell
cmake --build build --config Release --target Armor_detect
.\Armor_detect.exe
```

默认使用视频 2、红色检测和预览。摄像头模式直接取图，不需要下位机；需先填写该文件的 `live_camera_matrix` 和 `live_distortion`。窗口左上角列出每块 PnP 成功装甲板的 XYZ，单位 m，X 向右、Y 向下、Z 向前。框旁序号对应当前帧的坐标行，不表示跨帧身份。没有观测时显示 No armor detected，不保留上一帧坐标。Esc 或关闭窗口退出；视频按源 FPS 播放。此入口只显示画面，不生成预测、控制、CSV 或录制视频；`preview=false` 可用于无窗口处理。

## 3. 每帧行为

先取图像、检测灯条与装甲板、解算相机系位姿，再用图像时刻的姿态转换完整 tvec/rvec、重算四维观测并调用所选 EKF。随后控制解算选择未来目标板，Armor 直接调用 `send_target`。Output 同时保留相机系原始观测和固定系观测。

实时图像和 STATE 共用 `monotonic_time_ms()` 的 PC steady clock。STATE 在完整包校验后打接收时间，图像在取帧成功后打时间；这些仍是接收/取帧近似时刻，未校准曝光与传输延迟。视频使用 `frame_id × 1000 / FPS`，不会混用实时反馈时间。

姿态需包围图像时刻，相邻样本间隔不超过 100 ms，不外推。启动时先接收 STATE，再取第一帧；每帧 `synchronize` 等待右侧反馈，单次等待超过 100 ms 或反馈无法覆盖图像时刻即报错。同步成功后直接转换、滤波、控制和输出，不再用空观测替代缺失姿态。

每帧只推进一次滤波。首次有效观测初始化，从后续帧持续记录状态至结束，包括漏检帧；误差使用更新前预测，画面使用更新后状态。单板/Polar 选择最近有效观测，距离并列保留第一条；Armor 保留四板关联与联合更新。漏检或被拒绝时继续预测，但控制层立即发送 valid=false。

## 4. 坐标、标定和控制

相机系 C 为 OpenCV 坐标：X 右、Y 下、Z 前。固定系 B 的轴沿零位定义，原点为第一条有效反馈时的相机光心。机械系 M 以 yaw 转轴为原点，固定原点在 M 中记为 `o_ref`：

```text
R_BC = Ry(yaw) Rx(pitch) Rz(roll) R_mount
p_MC = Ry(yaw) [yaw_to_pitch + Rx(pitch) Rz(roll) camera_in_pitch]
p_BC = p_MC - o_ref
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

yaw 绕 +Y 为正，pitch 绕 +X 为正，roll 绕 +Z 为正。原固件 yaw 是相对关节角、pitch 是 IMU 姿态角、roll=0；需要实物确认 pitch 与转轴定义的一致性。默认零偏、零安装偏移和单位旋转仅表示理想模型。

`PoseBase::push` 保留最近 256 条已消费反馈，首次原点保持不动。`at(t)` 要求调用前已覆盖该时刻；`to_base` 同时转换位置与完整旋转；`convert` 处理整帧且不修改原始相机观测。装甲板水平朝向直接由变换后的法向计算 `atan2(-normal_x, normal_z)`。

观测为 `[atan2(x,z), atan2(y,hypot(x,z)), norm(position), armor_yaw]`，位置单位 m、角度 rad。`target_pitch` 随 Y 向下为正；它与控制 pitch 的符号不同。固定系 distance 表示固定原点到板的距离，不再表示当前相机距离。

Gimbal 选择当前相机可见范围内最近的未来板，并用相同的安装外参、偏移、零位及方向求解光轴对准所需的绝对原始 yaw/pitch。它保持测得的 roll，使用二维 Newton 求解。无有效更新、数据年龄超过 100 ms、位置方差过大、无前方目标或超出绝对限位时发送 valid=false；数值求解不收敛直接报错。

`GimbalConfig` 默认角度范围 yaw ±45°、pitch ±30°，最大位置方差 1 m²，是待实物配置的初值；填写为实际下位机关节范围。它只决定控制有效性，不改变 EKF 参数或下位机闭环。MC02 原有目标 100 ms、链路 200 ms 超时继续由固件执行。

PnP 与绘制板尺寸为 135×56 mm。尺寸在 `solver.cpp` 顶部配置，绘制尺寸位于 `output.cpp` 和 `plot/video.py`；换板型时需同步。灯条使用完整像素轮廓拟合，配对按检测得分贪心选择，单根灯条最多用于一块板。

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

每帧所有观测一起处理，先预测一次，再联合关联和更新：

1. 将观测转换为三维位置，并尝试关联 A0–A3。
2. 使用三维位置距离和装甲板朝向差门限筛选候选。
3. 按距离从小到大贪心匹配；距离相同时按输入顺序、板号排序。
4. 每条观测和板号只能使用一次，板间朝向关系须一致；不再搜索全局最优组合。
5. 将选中观测联合更新到同一个车体状态，每条观测使用相同的固定协方差。

初始化帧只使用最近的一条有效观测，将其定义为 A0；该帧其他观测不参与更新。当前流程没有多车分组，同帧输入默认属于同一辆小车。

### 5.3 固定观测噪声与漏检

使用固定观测协方差 `R = diag(0.0016, 0.0016, 0.16, 0.0576)`，观测顺序为目标方位角、目标俯仰角、距离、装甲板朝向；角度项单位为 rad²，距离项为 m²。对应的有效标准差为 `0.04 rad、0.04 rad、0.40 m、0.24 rad`。检测评分、重投影误差和水平观察角度不改变观测权重。

所有观测共用一个固定 `R`，CSV 中的检测评分和重投影误差仅供检测诊断。调参记录与可复现方法见 [固定噪声调参说明](tests/fixed_noise_tuning.md)。

| 构造参数 | 默认值 | 含义 |
| --- | --- | --- |
| `max_yaw_error` | 45°，单位为弧度 | 观测与预测装甲板的朝向差上限 |
| `max_distance_error` | `0.5` m | 观测与预测装甲板的三维位置距离上限 |
| `pair_yaw_tolerance` | 25°，传参单位为弧度 | 板间朝向关系容差 |

过程噪声、固定观测协方差和初始化设置位于 `src/predictor_armor.cpp`，构造参数的默认值在 `include/predictor_armor.hpp` 的接口声明中。调参时应同时检查残差、中心漂移、接受率和运动变化时的响应。

漏检或所有观测被拒绝时，EKF 只预测。Armor 按源视频时间逐帧调用，初始化后一直导出到视频结束，不插值时间或观测，也不在漏检期间隐藏其预测几何。控制层在漏检时发送 valid=false，滤波仍持续预测。

### 5.4 C++ 调用

四板模型直接接收 `std::vector<Eigen::Vector4d>` 观测，板号由关联算法自动确定。时间戳使用普通 `double`，调用方提供同一时钟上的递增时间。PoseBase 只接收类型化反馈，不再解析离线标定/遥测 CSV；安装参数由构造函数传入。投影和预测中心通过 `optional` 表达不可见目标与未初始化状态。

滤波参数保存在各 EKF 类中。Armor.cpp 使用顶部的 `predictor_type`，通过 `if` 直接调用 `SinglePlateEKF::update_frame`、`PolarEKF::update_frame` 或 `ArmorEKF::update_frame`。每个逐帧接口实现在对应 cpp 中，不经过通用模型选择器。`predictor.cpp` 只实现基础单板 EKF；固定列导出与 RMSE 位于 `output.cpp` 的 `PredictionOutput`。PnP 标定在进入循环前显式设置。

三个预测器保持声明与实现分离，算法和各自的逐帧接口放在对应 `.cpp`。所有模块均无自定义 namespace 和 using 类型别名。`predictor.hpp` 保留基础单板声明和公共观测/结果数据结构，不持有其他滤波器或选择模型。固定系 Armor 调用需设置 `armor.base_frame = true`。

三个模型的状态、协方差、观测和雅可比均使用固定尺寸 Eigen 类型：基础单板为 6 维状态 / 3 维观测，极坐标为 9 维状态 / 4 维观测，装甲板为 11 维状态 / 4 维单板观测。四板联合更新按实际关联数量组合 4–16 维观测，矩阵存储上限固定为 16；卡尔曼增益 直接通过 Eigen LDLT 求解，不构造逆矩阵，也不切换备用求解器。向量初始化使用 Eigen 构造函数或 `<<`，对角矩阵使用 `asDiagonal()`。

```cpp
#include "predictor_armor.hpp"

ArmorEKF ekf; // 默认使用固定观测噪声
ekf.base_frame = true; // 本应用使用固定系观测
Eigen::Vector4d z(0, 0, 3, 0); // yaw, pitch, distance, plate yaw
ekf.initialize(z);
ekf.predict(.03);
ekf.update_multi({z});
const auto future = ekf.forecast(.05); // 不推进当前状态
// 在线预测：predictor.update(frame_id, timestamp_ms, observations)。
// 导出：output.write(...)、output.finish()。
```

旧 Python 实现保存在 `tests/reference/predictor/`，仅供回归对照；原 `import predictor...` 接口已移除。
绘图仍使用 Python，继续读取相同的 CSV 文件。

### 5.5 未来位姿预测

`ekf.forecast(0.05)` 从当前滤波状态推算 0.05 秒后的车体状态、协方差和四块板位姿，
返回 `Forecast` 结构体中的 `state`、`covariance`、`plates`。`plates` 为 4×4 数组，行对应 A0–A3，
列为 x、y、z、yaw，单位为米和弧度。该操作不修改主 EKF 的状态、协方差或时间推进矩阵。

主 EKF 仍按帧间时间更新；未来预测使用本帧更新后的状态，缺测时使用推进到本帧的状态。
预测目标时间等于源图像时间加提前量，不按整数帧取整；30 FPS 下的 50 ms 仍精确使用 0.05 秒。
预测可以超出视频末尾，这些时刻没有后续视频观测可供验证。

Armor 的 EKF 观测、状态和未来预测统一使用固定基座坐标系，由 PoseBase 在滤波前转换。EKF 仍采用水平旋转模型，不估计板面俯仰、横滚。
未来预测误差应与目标时刻、同一板号和同一坐标系下的后续观测比较。


## 6. 输出与绘图

视频 `video_1.avi` 的输出如下；实时输入使用 `live` 后缀和 `camera_live.avi`。

| 文件 | 内容 |
| --- | --- |
| `data/pose_raw_1.csv` | 相机系 PnP，coordinate_frame=camera |
| `data/pose_base_1.csv` | 固定系位置、完整 rvec/四元数、四维观测和基座原点元数据 |
| `data/camera_pose_1.csv` | 每帧同步标志与 T_BC 的旋转/平移及固定原点元数据，包括无观测帧 |
| `results/video_1.avi` | 检测框、当前与未来几何，全部逆变换至当前相机后投影 |
| `results/control_target_1.csv` | 绝对 yaw/pitch、valid、目标板号、状态和位置方差，每帧一行 |
| `results/prediction_result_1.csv` | SinglePlate 原 13 列加 coordinate_frame |
| `results/polar_prediction_result_1.csv` | Polar 原 15 列加 coordinate_frame |
| `results/armor_prediction_result_1.csv` | Armor 原 32 列加 coordinate_frame |
| `results/armor_future_prediction_1.csv` | 未来四板，原 10 列加 coordinate_frame |
| `results/*rmse_result_1.txt` | 原六位小数 RMSE 格式；没有有效残差为 nan |

同名输出会覆盖。AVI 使用完整输入主干，其他文件后缀为主干最后一个下划线后的内容。状态及未来文件的 coordinate_frame 均为 base，参考原点元数据在 pose_base/camera_pose 中，配套文件必须来自同一次运行。初始化帧不写状态行；全程无观测时状态文件只有表头，RMSE 为 nan。

当前板用实线/圆点，未来板用虚线/菱形。投影会逆变换所有三维角点，不会只改位置或 yaw。模型仍只估计固定系水平板朝向，四板均显示，不模拟遮挡。残差是先验预测相对观测的误差，不代表真实定位精度。

```powershell
.\.venv\Scripts\python -B -m plot.raw --suffix 1
.\.venv\Scripts\python -B -m plot.armor --suffix 1
.\.venv\Scripts\python -B -m plot.video --suffix 1
# SinglePlate / Polar 配置运行后：
.\.venv\Scripts\python -B -m plot.basic --suffix 1
.\.venv\Scripts\python -B -m plot.polar --suffix 1
```

`plot.raw` 查看相机系原始观测；`plot.basic/armor` 按结果的坐标标签选择固定系观测。`plot.video` 读取 pose_base 和 camera_pose，将整块板的角点逆变换后投影；旧无坐标标签的相机系 CSV 仍可绘制。可通过 `--camera-poses` 指定变换文件。实际摄像头录制的视频已由 C++ 直接叠加；离线绘图的相机 profile 必须与实际内参一致。更多参数见 [plot/README.md](plot/README.md)。

## 7. 三种预测模型

在 `src/Armor.cpp` 修改 `predictor_type` 选择模型，重新编译后运行同一个 Armor.exe。

| 模型 | 状态与观测 | 每帧更新 |
| --- | --- | --- |
| SinglePlate | `[x,vx,y,vy,z,vz]`；方位、俯仰、距离 | 最近有效板，距离相同保留第一条 |
| Polar | 9 维：中心/速度、yaw/w、等半径 | 最近有效板，距离相同保留第一条 |
| Armor | 11 维：中心/速度、yaw/w、r/dl/dh | 四板关联与联合更新 |

SinglePlate 使用匀速状态转移和非线性球坐标观测：

```text
yaw      = atan2(x, z)
pitch    = atan2(y, sqrt(x² + z²))
distance = sqrt(x² + y² + z²)
```

其解析雅可比、LDLT 求增益、Joseph 协方差更新和完整 6×6 协方差保持不变。
`R=diag(0.0016,0.0016,0.16)`，`q=0.01`，每轴过程噪声块为
`q * [[dt⁴/4,dt³/2],[dt³/2,dt²]]`，初始协方差 `10I`。原点或竖直轴附近跳过观测更新。
最近板选择可能切换实体板，不提供单板身份关联。

Polar 的板位置统一采用 `x = xc + r*sin(yaw)`、`z = zc - r*cos(yaw)`，与 PnP 朝向及 Armor 模型一致；初始化中心使用相反位移。


Polar 保留原 9 维模型、数值雅可比与固定噪声 `R=diag(0.005,0.005,0.05,0.05)`；
Armor 的固定参数见第 5 节。两种模型的既有几何定义也保留，不重新解释 yaw 或半径。

三种模型都使用每帧的源时间间隔推进。无效观测不更新；非法或不递增的帧号/时间戳明确报错。
SinglePlate 的 13 列记录更新前预测，Polar 的 15 列及 Armor 的中心状态记录更新后状态，
所有误差均来自更新前预测。RMSE 只累计有限残差，Armor 只累计最近被接受板的残差。


## 8. 验证

```powershell
powershell -ExecutionPolicy Bypass -File .\tests\run.ps1
.\.venv\Scripts\python -B tests/verify_armor_video.py --max-source-frames 90
```

原生测试覆盖协议拆包/CRC、检测/PnP、三种 EKF、完整姿态变换、跨界插值、固定原点、移动相机下固定目标、绝对光轴对准、漏检/限位/失效控制，以及帧号、CSV、视频输出。NumPy 对照验证全部结果列、未来几何和 RMSE。视频集成脚本临时修改配置并验证红/蓝及三种模型，最后恢复源码和默认程序。

此次软件验证日志集中于 `tests/outputs/armor-unified/`。尚未接入实际摄像头、USB-TTL 或 MC02 电机；软件几何验证不能代替实物标定、延迟测量和收发联调。测试接口与产物说明见 [tests/README.md](tests/README.md)。

## 9. 达妙 MC02 基础串口收发

`Serial` 对接提供的 `rm_gimbal_26` 固件 `host_link.c`，使用 Windows 串口 API，固定 **921600 baud、8N1、无流控**。固件 USART10 引脚为 **PE3/TX → USB-TTL RX，PE2/RX ← USB-TTL TX，GND 共地**，使用 3.3V TTL。板上连接器位置和供电接法见 [达妙 MC02 官方资料](https://gitee.com/kit-miao/dm-mc02)。

接口位于 `include/serial.hpp`，实现全部放在 `src/serial.cpp`：

| 接口 | 作用 |
| --- | --- |
| `open("COM3")` | 打开实际串口，启动通信线程；COM3 仅为示例 |
| `close()` | 等待已排队数据发完，结束线程并关闭串口；通信失败在清理后报告，析构时自动清理 |
| `receive(state)` | 取走最新收到的姿态，有新样本返回 true，否则返回 false |
| `send_target(yaw, pitch, valid)` | 按调用方给定的角度和有效标志排队发送目标，只保留最新待发目标 |
| `send_mode(mode)` | 排队发送模式，0=IDLE、1=MANUAL、2=AUTO_AIM、3=STABILIZE |

`Serial` 只保存最新姿态；一次读到多条 STATE 时返回最后一条。姿态包含 yaw、pitch、roll、mode、flags 和 PC 接收时间戳。角度单位 rad，时间单位 ms；所有实时 PC 时间戳应使用同一个 `monotonic_time_ms()`，不混用录像帧时间。当前固件 yaw 是关节角、pitch 是 IMU 姿态角、roll=0，flags=0 是预留值，不作为姿态有效性标志。

协议为 `A5 CMD LEN PAYLOAD CRC8`，float32 小端。CRC-8/ATM 多项式 0x07、初值 0，校验 CMD、LEN 和 PAYLOAD。TARGET=1，载荷 9 字节；STATE=2，载荷 14 字节；MODE=3，载荷 1 字节；HEARTBEAT=4，载荷为空。接收端只解析固定 18 字节 STATE 包，保留分包等待和连续包处理；帧头或 CRC 错误直接报错，不扫描噪声重找帧头，也不重复检查角度值。

后台线程串行写包，每 50 ms 独立发送心跳；读取等待 5 ms、写超时 20 ms。串口打开或收发失败通过接口抛出异常，并停止通信，重新连接由调用方显式 close/open。目标有效性、反馈新鲜度、角度限位、模式决策和日志由上层负责；模块不自动修改 valid、不自动重连，也不生成 CSV 或独立监视程序。

构建模块：

```powershell
cmake --build build --config Release --target serial
```

串口模块由 Armor 直接调用。视频回放不打开串口；实时模式发送 AUTO_AIM、每帧 TARGET，正常结束时发送无效目标与 IDLE 并完成待发写入。尚未连接 MC02 做实物收发或运动验证。

### 统一视频界面

SinglePlate、Polar、Armor、Armor_detect 和 Python 视频叠加统一使用左上角黄色文字、黑色描边及相同行距，显示模式、帧号、状态和各观测板的相机系 XYZ（m）。框旁序号对应当前帧观测。Polar/Armor 保留四板当前实线框和未来虚线框。Python 导出保持输入分辨率，不追加侧栏或放大图。

串口内部的 TARGET/MODE 在发送接口直接打包，STATE 在后台 run() 直接解析，CRC 为 cpp 内共用函数。已删除独立协议类、发送转发函数及 flush；close 请求停止后，线程发完剩余 MODE/TARGET 再退出，无需发送完成条件变量。
