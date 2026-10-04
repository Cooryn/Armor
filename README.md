# RoboMaster Vision：装甲板检测与 EKF 跟踪

本项目使用 C++ / OpenCV 从视频中检测装甲板，通过 PnP 解算装甲板位姿，再由 C++ / Eigen 扩展卡尔曼滤波器（EKF）估计车体中心、旋转朝向、角速度和装甲板几何参数。支持导出检测视频、原始观测、滤波结果、诊断曲线和状态叠加视频。

主要工作流面向单辆小车的四装甲板跟踪。目前有两个独立视频模块，由同一个 CMake 构建：`Armor.exe` 在同一个入口中完成识别、PnP、三种 EKF 中的一种和未来预测叠加；`camera_tracking.exe` 保留自己的 EKF 和相机云台控制流程。默认逐帧显示视频，模拟实时图像输入；也支持无窗口批处理。

目前没有下位机通信或真实云台反馈。手动、自动和归中控制作用于仿真云台，界面标注 `SIMULATION`；录像不会随仿真云台转动而改变。实时预测仍使用相机坐标系，固定基座系转换目前是单独的离线流程。

## 1. 功能与数据流程

| 模块 | 输入 | 功能 | 输出 |
| --- | --- | --- | --- |
| 装甲板检测 | 红色或蓝色装甲板视频 | 颜色提取、灯条筛选、端点定位、灯条配对 | 装甲板角点与检测质量 |
| PnP 解算 | 角点、相机标定、装甲板尺寸 | 平面姿态候选求解、筛选与短时关联 | 装甲板位置、朝向、重投影误差 |
| 装甲板 EKF | 逐帧 PnP 观测 | 多板关联、固定观测噪声、状态估计、漏检预测 | 车体状态、观测诊断、残差统计 |
| 实时预测与相机自瞄 | 视频逐帧检测结果 | EKF、50 ms 预测、手动/自动/归中、仿真光轴跟踪 | 实时窗口、预测 AVI、原始观测与控制 CSV |
| 图表与视频 | 原始观测、EKF 结果、原视频 | 绘制曲线与投影状态 | PNG 图表、MP4 视频 |

```text
输入视频 → Armor.exe → 检测/PnP → 所选 EKF → 当前与未来预测窗口 / AVI
                                 │                │
                                 ▼                ▼
                           原始观测 CSV     状态 / 残差 / RMSE
                                 │                │
                                 ▼                ▼
                              plot.raw     plot.basic / polar / armor / video

输入视频 → camera_tracking.exe → 检测/PnP → EKF → 仿真相机控制与独立视频输出
```

Armor 单线程按视频帧号和源 FPS 处理每一帧，不回读观测 CSV。EKF 使用 C++ 和 Eigen；
Python 的 `plot/` 读取导出文件绘图，修改图表不必重跑视频。

### 1.1 灯条配对策略

`Armor.exe` 与 `camera_tracking.exe` 共用 `src/lightbar_detector.cpp`。轮廓提取使用 `CHAIN_APPROX_NONE`，直接用完整像素轮廓拟合灯条轴线，不再对轮廓重采样。

候选灯条对先经过角度差、长度比、局部坐标系下的宽高比与上下错位筛选，再结合形状、中间灯条和灯条质量计算 `detection_score`。最终采用贪心匹配：

1. 按候选得分从高到低排序；同分时按排序后的左右灯条索引确定顺序。
2. 两根灯条均未使用时接受该候选，并标记两根灯条已使用。
3. 将最终装甲板按中心横坐标排序后交给 PnP 解算。

## 2. 目录与脚本职责

```text
Armor/
├── CMakeLists.txt                 C++ 构建配置
├── setup.ps1                      Python 环境配置与 C++ 构建
├── requirements.txt               Python 依赖
├── include/                       检测、预测、相机控制与坐标变换头文件
├── src/
│   ├── Armor.cpp                  源码配置及检测 → PnP → 预测 → 输出调用链
│   ├── camera_tracking.cpp        共享视频读写与预测绘制、自瞄显示与控制、离线控制转换
│   ├── pose_base.cpp              离线相机系到固定基座系位姿转换
│   ├── lightbar_detector.cpp      灯条检测、装甲板配对及检测结果绘制
│   ├── solver.cpp                 相机参数、逐帧 PnP、候选检查和短时姿态关联
│   ├── predictor.cpp              基础单板 EKF、在线预测及 CSV/RMSE 导出
│   ├── predictor_polar.cpp        极坐标 EKF 实现
│   └── predictor_armor.cpp        四装甲板联合 EKF 实现
├── plot/
│   ├── raw.py                     原始 PnP 观测曲线
│   ├── armor.py                   11 维装甲板 EKF 图表
│   ├── video.py                   原视频叠加观测、估计状态与装甲板边框
│   ├── basic.py                   6 维模型图表
│   ├── polar.py                   9 维模型图表
│   └── _cli.py                    静态绘图命令的公共路径参数
├── assets/image/                  历史样本（检测程序不再提供图片模式）
├── assets/video/                  输入视频
├── data/                          C++ 导出的原始观测 CSV
├── results/                       CSV、TXT、PNG、AVI 和 MP4 输出
├── tests/                         回归测试与离线评估工具
│   ├── plotting/                  测试用图像导出
│   ├── build/                     C++ 测试构建目录
│   └── outputs/                   测试日志、临时文件、评估产物
├── build/                         正式 C++ 构建目录
└── .venv/                         项目 Python 环境
```

默认构建输出三个程序到仓库根目录：`Armor.exe`、`camera_tracking.exe`、`pose_base.exe`。三个独立 predictor 可执行目标及其旧入口已移除。正式输出保存在 `data/`、`results/`；测试集中在 `tests/`。补充说明见 [绘图说明](plot/README.md)和[测试说明](tests/README.md)。

## 3. 环境配置

### 3.1 依赖

| 依赖 | 用途 |
| --- | --- |
| Miniconda / Python | 提供基础 Python 环境，创建项目 `.venv` |
| Visual Studio 2022 C++ 工具 | 编译 C++17，需要桌面 C++ 开发工具和 Windows SDK |
| CMake | 生成并构建 Visual Studio x64 工程 |
| OpenCV C++ 开发库 | 图像处理、视频读写、PnP 和 HighGUI 窗口；需要带 GUI 的开发库和 CMake 配置 |
| Eigen ≥ 3.3 | 三种预测器的向量、矩阵和线性求解；纯头文件库，当前验证版本为 5.0.1 |
| NumPy、pandas、Matplotlib、Python OpenCV | Python 参考回归、CSV 处理、图表与视频导出 |

Python 包版本要求见 [requirements.txt](requirements.txt)。仓库现有环境使用 Python 3.13 和 OpenCV 4.13。Python 的 `opencv-python` 包不代替 C++ OpenCV 开发库。

### 3.2 配置与构建

在 PowerShell 中进入仓库根目录，激活安装了 C++ OpenCV 和 Eigen 的 Conda 环境。若依赖安装在 `base` 中：

```powershell
conda activate base
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

`setup.ps1` 使用当前 `python` 创建可复用基础环境包的 `.venv`，安装 Python 依赖，从基础环境的 `Library/cmake` 查找 OpenCV，再通过根目录 CMake 构建上述三个 Release 程序。

脚本要求 `python`、`cmake` 已能在终端调用，不负责安装 Visual Studio、Conda 或 C++ OpenCV。已有 `.venv` 时会复用它；运行时应使用与其对应的 Conda 基础环境。

手动配置并编译 C++（没有 `build` 文件夹时 CMake 会创建）：

```powershell
$pythonBase = python -c "import sys; print(sys.base_prefix)"
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DBUILD_TESTING=OFF "-DOpenCV_DIR=$pythonBase/Library/cmake" "-DCMAKE_PREFIX_PATH=$pythonBase/Library"
cmake --build build --config Release --parallel
$env:PATH = "$pythonBase\Library\bin;$env:PATH"
```

以上路径适用于依赖安装在当前 Python 对应的 Conda 环境。其他安装方式请提供实际的 OpenCV、Eigen CMake 路径。配置完成后，只重新编译 C++：

```powershell
cmake --build build --config Release --parallel
```

Python 命令统一使用 `.\.venv\Scripts\python`，不要求额外激活 `.venv`。

只构建两个视频模块可使用 `cmake --build build --config Release --target Armor camera_tracking --parallel`。`--config Release` 选择 Release 配置，`--parallel` 启用并行编译；Visual Studio 工程使用该构建命令，不使用 `make`。

## 4. 配置、构建与运行 Armor

在 `src/Armor.cpp` 顶部集中修改：

```cpp
constexpr EnemyColor target_color = ENEMY_RED;
constexpr const char* video_file = "assets/video/video_2.avi";
constexpr PredictorType predictor_type = PREDICTOR_ARMOR;
constexpr bool preview = true;
constexpr double prediction_horizon_ms = 50;
```

| 参数 | 用法 |
| --- | --- |
| `target_color` | `ENEMY_RED` 或 `ENEMY_BLUE` |
| `video_file` | 相对项目根目录的视频路径；视频 2 使用 `assets/video/video_2.avi` |
| `predictor_type` | `PREDICTOR_SINGLE_PLATE`、`PREDICTOR_POLAR` 或 `PREDICTOR_ARMOR`，见第 10 节 |
| `preview` | `true` 按源 FPS 显示；`false` 无窗口、不限速处理 |
| `prediction_horizon_ms` | 非负、有限的提前量，单位 ms，默认 50 |

根据目标颜色选择 `ENEMY_RED` 或 `ENEMY_BLUE`；仓库现有两段视频的目标均为红色。修改配置后重新编译，再运行：

```powershell
cmake --build build --config Release --target Armor
.\Armor.exe
```

Armor 不接收命令行参数。输入与输出路径基于编译时的项目根目录，因此从其他工作目录启动也能找到配置视频。
按 Esc 或关闭窗口会结束处理，并完成已处理帧的 AVI、CSV 和 RMSE 导出。预览需要带 GUI 的 OpenCV。
输入、窗口创建和文件写入失败会返回非零退出码并报告原因。

每帧执行检测/PnP、写原始观测、推进/更新 EKF、绘制当前与未来位置，最后显示并写入视频。
首次有效观测仅初始化；状态行从下一帧起一直写到视频结束，包括连续漏检和末尾漏检。
无观测帧保留预测状态，观测及残差字段为空。全程没有有效观测或仅处理初始化帧时，
状态 CSV 只有表头，RMSE 为 `nan`。初始化帧仍可显示当前与未来几何，Armor 模型也记录初始化诊断。

单板模型绘制一个目标点；Polar 和 Armor 绘制中心及四块板。
圆点和实线表示更新后的当前状态，菱形和虚线表示未来预测。
误差由更新前预测计算，不能用当前画面的未来标记代替未来时刻的精度评估。
四块板都绘制，不模拟遮挡。

## 5. 绘图与独立相机跟踪程序

运行 Armor 后，可直接使用对应模型的绘图模块：

```powershell
.\.venv\Scripts\python -B -m plot.raw --suffix 1
.\.venv\Scripts\python -B -m plot.armor --suffix 1
.\.venv\Scripts\python -B -m plot.video --suffix 1
# SinglePlate 或 Polar 配置生成结果后：
.\.venv\Scripts\python -B -m plot.basic --suffix 1 --output-dir results/basic
.\.venv\Scripts\python -B -m plot.polar --suffix 1 --output-dir results/polar
```

静态图支持 `--suffix`、`--data-dir`、`--results-dir`、`--output-dir`。
`plot.video` 用于 Armor 模型，还支持 `--video`、`--predictions`、`--raw`、`--diagnostics`、
`--no-diagnostics`、`--camera-profile 1|2`、`--start-frame`、`--max-frames` 和 `--output`。
它按帧号对齐，缺少状态行时不沿用前一帧。详细参数见 [plot/README.md](plot/README.md)。

`camera_tracking.exe` 继续使用独立入口和原有命令行：

```powershell
.\camera_tracking.exe red video_1.avi
.\camera_tracking.exe blue video_2.avi --headless
```

该程序的仿真控制默认自动模式：`1` 为手动，`W/S`、`A/D` 调节 pitch/yaw；
`2` 自动跟踪，`3` 或 `C` 归中，Esc 或关窗结束。
没有真实电机、串口或云台反馈；录像不会随仿真云台转动而改变。
它的时间推进、丢失目标规则和输出保持原状，无须先运行 Armor。

## 6. 坐标、角度与物理尺寸

### 6.1 相机坐标系

位置以米为单位，原点在相机光心：`x` 向右、`y` 向下、`z` 向相机前方。PnP/EKF CSV 角度为弧度，角速度为弧度/秒；控制 CSV 中 `_deg`、`_dps` 字段分别使用度、度/秒，控制 pitch 向上为正。

| 字段 | 定义 |
| --- | --- |
| `x, y, z` | PnP 解算的装甲板中心位置 |
| `xc, yc, zc` | EKF 估计的车体旋转中心位置 |
| `target_yaw` | 装甲板中心的水平方位角：`atan2(x, z)` |
| `target_pitch` | 装甲板中心的俯仰方向角：`atan2(y, sqrt(x² + z²))`；因 y 向下，正值表示偏下 |
| `distance` | 相机到装甲板中心的三维距离 |
| `armor_orientation_yaw` | 装甲板自身的水平朝向角，来自 PnP 旋转矩阵 |
| `body_yaw` | EKF 车体参考相位，由初始化板号定义，不代表识别出的车头方向 |

`target_yaw` 表示“装甲板在哪里”，`armor_orientation_yaw` 表示“装甲板朝哪里”。对纯水平旋转，装甲板 yaw 为 `0°` 表示板面正对相机、左右灯条深度相同；正角度表示右灯条更远，负角度表示左灯条更远。俯仰和侧倾需结合完整姿态理解。

角度按 ±π 折回，跨越边界时可以出现跳变。`body_yaw_curve` 未做连续角度展开，不能直接把边界跳变当作运动异常。

### 6.2 物理尺寸与相机配置

PnP 与叠加边框采用灯条中心间距 **135 mm**、灯条长度 **56 mm**。实际装甲板应与模型尺寸一致。尺寸或标定不准确会影响距离、朝向和投影位置。

| 配置 | 配套视频尺寸 | C++ 选择方式 | 可视化选择方式 |
| --- | --- | --- | --- |
| `1` | 1440 × 1080 | 文件名主干不包含 `video_2` 时的默认参数 | `--camera-profile 1` |
| `2` | 1280 × 1024 | 文件名主干包含 `video_2` | `--camera-profile 2` |

相机内参与畸变参数位于 `src/solver.cpp`、`src/camera_tracking.cpp` 和 `plot/video.py`；物理板尺寸位于 `src/solver.cpp`、`src/camera_tracking.cpp` 的绘图实现和 `plot/video.py`。使用新相机、新分辨率或不同板型时，需要同步配置两个视频模块和离线可视化。`plot.video` 会检查画面尺寸，但相同分辨率本身不意味着相机标定相同。

## 7. EKF 模型与多板处理

### 7.1 状态量

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

### 7.2 同帧多板

每帧所有观测一起处理，先预测一次，再联合关联和更新：

1. 检查每条观测的有效性，并尝试关联 A0–A3。
2. 使用 NIS 和距离残差门限筛除不一致候选。
3. 要求同帧板号唯一，板间朝向差与板号关系一致。
4. 优先选择数量最多的一致观测集合，再比较总 NIS。
5. 将选中观测联合更新到同一个车体状态，每条观测使用相同的固定协方差。

初始化帧只使用最近的一条有效观测，将其定义为 A0；该帧其他观测不参与更新。当前流程没有多车分组，同帧输入默认属于同一辆小车。

### 7.3 固定观测噪声与漏检

使用固定观测协方差 `R = diag(0.0016, 0.0016, 0.16, 0.0576)`，观测顺序为目标方位角、目标俯仰角、距离、装甲板朝向；角度项单位为 rad²，距离项为 m²。对应的有效标准差为 `0.04 rad、0.04 rad、0.40 m、0.24 rad`。检测评分、重投影误差和水平观察角度不改变观测权重。

所有观测共用一个固定 `R`，CSV 中的检测评分和重投影误差仅供检测诊断。调参记录与可复现方法见 [固定噪声调参说明](tests/fixed_noise_tuning.md)。

| 构造参数 | 默认值 | 含义 |
| --- | --- | --- |
| `nis_gate` | `16.0` | 归一化创新平方门限 |
| `max_distance_error` | `0.5` m | 距离残差绝对上限 |
| `pair_yaw_tolerance` | 25°，传参单位为弧度 | 板间朝向关系容差 |

过程噪声、固定观测协方差和初始化设置位于 `src/predictor_armor.cpp`，构造参数的默认值在 `include/predictor_armor.hpp` 的接口声明中。调参时应同时检查残差、中心漂移、接受率和运动变化时的响应。

漏检或所有观测被拒绝时，EKF 只预测。Armor 按源视频时间逐帧调用，初始化后一直导出到视频结束，不插值时间或观测，也不在漏检期间隐藏其预测几何。独立的 camera_tracking 保留自己的目标丢失处理规则。

### 7.4 C++ 调用

观测的 `armor_id` 使用普通整数：`-1` 表示自动关联，`0～3` 表示指定板号，其余编号不参与关联。内部时间戳使用普通 `double`，`-1` 表示尚未设置，`0` 是有效时间；`camera_frames_load` 的参考时间默认 `-1`，表示使用第一条遥测。命令行显式传入的原点时间必须是有限的非负数。投影和预测中心仍通过 `optional` 表达无有效结果。

默认滤波参数直接保存在数据结构中，使用 `ArmorEKF ekf;` 或 `VideoPredictor predictor{PREDICTOR_ARMOR, 50};` 初始化，不再提供 `make_*` 工厂和只读取字段的查询函数。模型选择、最近板选择、更新及显示结果组装集中在 `video_predictor_update`；CSV 的开关文件、分隔符和换行操作放在对应导出函数内。PnP 视频标定在首次 `solver_solve_frame` 调用时按视频文件名选择；已有标定可用 `Solver{camera_matrix, distort_coeffs}` 直接设置。预测提前量和控制参数在首次使用时检查。

三个预测器保持声明与实现分离：`include/predictor*.hpp` 声明普通函数和只存数据的结构体，`src/predictor*.cpp` 实现算法。项目接口不使用自定义类、成员函数、嵌套类型或命名空间。`predictor.hpp/.cpp` 的 `video_predictor_update` 负责在线预测，`prediction_output_write/finish` 负责 CSV/RMSE 导出，均不依赖 OpenCV；CSV 辅助函数仅在对应 `.cpp` 内使用。`lightbar_detector.hpp/.cpp` 只负责检测及检测结果绘制；`solver.hpp/.cpp` 的 `solver_solve_frame` 负责相机参数选择、逐帧 PnP 和类型化观测。`camera_tracking.hpp/.cpp` 的 `video_input_*`、`video_output_*` 负责视频读写、窗口及预测画面。`Armor.cpp` 保留顶部配置和“检测 → PnP → 预测 → 输出”调用链，无新增模块文件。CMake 的 `predictor_filters` 包含滤波及预测结果导出，`armor_vision` 包含检测与 PnP，`armor_video` 复用现有相机文件的视频实现；共享库编译时排除相机程序的独立入口。自己的滤波目标可使用 `target_link_libraries(your_target PRIVATE predictor_filters)`。

三个模型的状态、协方差、观测和雅可比均使用固定尺寸 Eigen 类型：基础单板为 6 维状态 / 3 维观测，极坐标为 9 维状态 / 4 维观测，装甲板为 11 维状态 / 4 维单板观测。四板联合更新按实际关联数量组合 4–16 维观测，矩阵存储上限固定为 16；卡尔曼增益和 NIS 直接通过 Eigen LDLT 求解，不构造逆矩阵，也不切换备用求解器。向量初始化使用 Eigen 构造函数或 `<<`，对角矩阵使用 `asDiagonal()`。

```cpp
#include "predictor_armor.hpp"

ArmorEKF ekf; // 默认使用固定观测噪声
ArmorMeasurement z(0, 0, 3, 0); // yaw, pitch, distance, plate yaw
armor_initialize(ekf, z);
armor_predict(ekf, .03);
armor_update_multi(ekf, {{z}});
const auto future = armor_forecast(ekf, .05); // 不推进当前状态
// 在线预测：video_predictor_update(predictor, frame_id, timestamp_ms, observations)。
// 导出：prediction_output_write/finish，或 video_output_write/finish。
```

旧 Python 实现保存在 `tests/reference/predictor/`，仅供回归对照；原 `import predictor...` 接口已移除。
绘图仍使用 Python，继续读取相同的 CSV 文件。

### 7.5 未来位姿预测

`ekf.forecast(0.05)` 从当前滤波状态推算 0.05 秒后的车体状态、协方差和四块板位姿，
返回 `Forecast` 结构体中的 `state`、`covariance`、`plates`。`plates` 为 4×4 数组，行对应 A0–A3，
列为 x、y、z、yaw，单位为米和弧度。该操作不修改主 EKF 的状态、协方差或时间推进矩阵。

主 EKF 仍按帧间时间更新；未来预测使用本帧更新后的状态，缺测时使用推进到本帧的状态。
预测目标时间等于源图像时间加提前量，不按整数帧取整；30 FPS 下的 50 ms 仍精确使用 0.05 秒。
预测可以超出视频末尾，这些时刻没有后续视频观测可供验证。

Armor 的观测与预测使用相机坐标系。需要运动补偿的调用者可先使用第 12 节的位姿转换，再通过滤波 API 处理固定基座系观测。EKF 仍采用水平旋转模型，不估计板面俯仰、横滚。
未来预测误差应与目标时刻、同一板号和同一坐标系下的后续观测比较。

## 8. 输出文件与读法

以 `video_1.avi` 为例，Armor 总会写入 `results/video_1.avi` 和 `data/pose_raw_1.csv`，
视频包含检测框、PnP 位置与所选模型的当前/未来预测。模型输出如下：

| 模型 | 输出文件 |
| --- | --- |
| SinglePlate | `prediction_result_1.csv`（13 列）、`rmse_result_1.txt` |
| Polar | `polar_prediction_result_1.csv`（15 列）、`polar_rmse_result_1.txt` |
| Armor | `armor_prediction_result_1.csv`（32 列）、`armor_rmse_result_1.txt` |
| Armor 额外导出 | `armor_observation_diagnostics_1.csv`（11 列）、`armor_future_prediction_1.csv`（10 列） |

模型结果全部位于 `results/`。AVI 名取视频完整主干，CSV 后缀取主干最后一个下划线之后的部分；
没有下划线时使用完整主干。同名输出会覆盖，运行只生成当前所选模型的结果。
RMSE 为六位小数及原单位。无有效残差时各项为 `nan`。

camera_tracking 的独立输出仍是 `results/camera_prediction_video_1.avi`、
`data/camera_pose_raw_1.csv` 和 `results/camera_control_1.csv`。它不导出 Armor 的完整状态/诊断文件。

原始位置是装甲板中心；中心状态是 EKF 车体旋转中心。原始曲线连接全部检测观测，
同帧多板和换板会造成跳变。误差曲线与 RMSE 表示更新前预测和当前观测的残差，
并非相对于定位真值的误差。

### 8.1 原始 CSV：`data/pose_raw_*.csv`

| 字段 | 含义 / 单位 |
| --- | --- |
| `frame_id` | 源视频帧号，从 0 开始；同帧多块板各占一行 |
| `timestamp` | 源视频时间，毫秒，由帧号和帧率计算 |
| `x, y, z` | 装甲板中心相机坐标，m |
| `target_yaw, target_pitch` | 装甲板中心的方位角、俯仰方向角，rad |
| `distance` | 装甲板中心距离，m |
| `armor_orientation_yaw` | 装甲板自身朝向，rad |
| `detection_score` | 启发式检测质量分数，越大越好，不是正确识别概率 |
| `reprojection_error` | 四角点重投影的像素 RMSE |
| `pnp_candidate_count` | 通过有效性检查、去重后的 PnP 候选数量 |
| `pnp_used_temporal` | 是否利用短时时序提示选择了非最小重投影误差候选，0/1 |
| `rvec_x, rvec_y, rvec_z` | PnP 完整旋转向量的三个分量，rad |
| `coordinate_frame` | 两个视频模块的原始观测均为 `camera` |

无检测帧不会在原始 CSV 中写占位行。输入 EKF 的同帧观测必须具有相同时间戳，帧间时间戳必须递增。

### 8.2 EKF 结果 CSV

| 字段 | 含义 |
| --- | --- |
| `frame_id, timestamp` | 当前帧与源时间，时间单位仍为毫秒 |
| `prediction_horizon_ms, prediction_timestamp` | 未来预测提前量及目标时间，均为毫秒 |
| `future_xc, future_yc, future_zc, future_body_yaw` | 目标时刻的车体中心和参考朝向，m、rad |
| `xc, yc, zc`、`vxc, vyc, vzc` | 当前帧估计的中心位置和速度，m、m/s |
| `body_yaw, w` | 当前帧估计的参考朝向和角速度，rad、rad/s |
| `r, dl, dh` | 估计的几何参数，m |
| `xa, za` | 代表性装甲板的更新前预测位置，m，不是中心坐标 |
| `armor_id` | 代表性装甲板编号 |
| `pred_armor_yaw, obs_armor_yaw` | 代表性板的更新前预测朝向与当前观测朝向，rad |
| `err_target_yaw, err_target_pitch, err_distance, err_armor_yaw` | 更新前预测减观测；距离为 m，角度为 rad |
| `observation_count, accepted_count, rejected_count` | 当前帧观测、接受与拒绝数量 |
| `status` | `updated` 表示有观测更新；`prediction_only` 表示仅预测 |

中心状态是当前时刻估计，`xa/za` 与误差字段是更新前预测，两类字段的时刻含义不同。多板联合更新时，代表性字段只展示最近的一块被接受装甲板；所有被接受观测都参与更新。无接受观测时，观测角度和残差记为 `NaN`。

未来预测文件 `armor_future_prediction_*.csv` 包含 `frame_id`、`timestamp`、
`prediction_horizon_ms`、`prediction_timestamp`、`armor_id`、`x`、`y`、`z`、
`armor_orientation_yaw` 和 `source_status`。其中帧号与 `timestamp` 属于做出预测的源帧，
位置和朝向属于 `prediction_timestamp`；`source_status` 说明源帧是否有观测更新。

### 8.3 逐条观测诊断 CSV

诊断文件通过 `frame_id` 和 `observation_index` 定位原始 CSV 中同帧的第几条记录，索引从 0 开始。

| 字段 | 含义 |
| --- | --- |
| `accepted, armor_id` | 是否使用该观测，以及接受时关联的板号 |
| `nis` | 归一化创新平方；初始化等没有该计算的记录为空 |
| `best_candidate_id` | 用于候选诊断的最小 NIS 板号，未必是最终联合关联结果 |
| `observed_distance, observed_armor_yaw` | 原始距离和朝向观测 |
| `distance_residual` | 观测减预测的距离残差，与结果 CSV 的距离误差符号相反 |
| `reason` | 接受或未使用的原因，见下表 |

| `reason` 值 | 含义 |
| --- | --- |
| `accepted` | 已关联并参与更新 |
| `initialization` | 用作初始化种子 |
| `initialization_unused` | 初始化帧中的其他观测，未参与更新 |
| `invalid` | 观测形状、数值或范围无效 |
| `innovation_gate` | NIS 或距离残差门限未通过 |
| `association_conflict` | 存在独立候选，但未进入最终联合关联组合 |

初始化阶段没有卡尔曼更新，NIS 和距离残差可以为空。`accepted=False` 不一定代表误检，例如初始化帧的未使用观测。

### 8.4 残差与精度

误差图和 RMSE 文本统计每帧代表性被接受观测的更新前残差，角度差折回到 ±π。它们不是相对于真实位置或姿态的误差，也不是同帧所有观测的汇总，更不是 50 ms 未来预测误差。

残差小表示预测与检测结果较一致。评估时还需要查看拒绝率、连续漏检、中心漂移、角速度稳定性和停车响应；确认绝对精度需要实际尺寸、相机标定与外部真值。

### 8.5 实时仿真控制 CSV：`results/camera_control_*.csv`

| 字段 | 含义 |
| --- | --- |
| `frame_id,timestamp_ms` | 源帧编号和视频时间，ms |
| `mode` | `MANUAL`、`AUTO`、`CENTER` |
| `simulation_only` | 当前固定为 `1`，没有硬件反馈 |
| `target_valid` | 自动模式下是否存在有效预测跟踪目标；手动和归中为 `0` |
| `yaw_deg,pitch_deg` | 积分更新后的仿真云台角度，右/上为正 |
| `yaw_rate_dps,pitch_rate_dps` | 本帧仿真实际角速度，度/秒 |
| `control_valid` | 本帧控制是否有效；手动和归中不要求目标有效 |
| `target_x,target_y,target_z` | 最近一次自动选中的未来装甲板位置，m，位于输入视频的相机系；仅在 `target_valid=1` 时作为本帧目标使用 |

该文件与第 12 节的离线控制 CSV 格式不同。实时程序没有导出未来板号，当前选择规则是四块预测板中位于相机前方且距离最近的一块。

## 9. 离线视频画面说明

Armor 模型在 `prediction_only` 帧中不再挑选一块最近预测板填充关联结果：
`armor_id=-1`，`xa,za,pred_armor_yaw,obs_armor_yaw` 和观测残差留空。
车体状态和 `armor_future_prediction_*.csv` 的四块板仍按模型正常预测。
初始化诊断也记录实际时间戳、观测距离和板朝向，便于与原始数据对齐。

`armor_video_*.mp4` 保留原画面尺寸和帧率，右侧增加 340 像素信息栏，不复制音轨。支持的画面尺寸下，侧栏底部显示目标局部放大图。

| 标记 | 含义 |
| --- | --- |
| A0–A3 彩色实线边框与圆圈 | 当前帧估计的四块装甲板 |
| F0–F3 同色虚线边框与菱形 | 对应四块板的未来预测；与同编号 A 标记属于同一相对板号 |
| 彩色加号 | 已接受且关联到板号的原始观测 |
| 红色叉号 | 被拒绝的原始观测 |
| 黄色加号 | 未分类观测，例如未加载诊断文件 |
| 白色星形与轨迹 | 估计车体中心及近期轨迹 |
| 白色菱形 | 未来车体中心 |

侧栏显示帧号、源时间、状态、中心位置、角速度、结构参数、观测计数、预测提前量及目标时间。当前帧没有 EKF 记录时显示无状态，不沿用前一帧状态。旧 CSV 没有未来预测字段时只显示当前估计；提前量为 0 时也不重复绘制未来框。

彩色边框由对应时刻的状态和 135 × 56 mm 尺寸投影生成，包含相机畸变影响。它们不是 C++ 原始检测框。未来框绘制在当前画面上表示预期运动位置，不应与当前检测框直接比较精度。四块板全部显示，未模拟遮挡。

## 10. 三种预测模型

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
Armor 的固定参数见第 7 节。两种模型的既有几何定义也保留，不重新解释 yaw 或半径。

三种模型都使用每帧的源时间间隔推进。无效观测不更新；非法或不递增的帧号/时间戳明确报错。
SinglePlate 的 13 列记录更新前预测，Polar 的 15 列及 Armor 的中心状态记录更新后状态，
所有误差均来自更新前预测。RMSE 只累计有限残差，Armor 只累计最近被接受板的残差。

## 11. 测试与常见问题

```powershell
# Python 与 C++ 回归测试
powershell -ExecutionPolicy Bypass -File .\tests\run.ps1

# 只运行 Python 测试
powershell -ExecutionPolicy Bypass -File .\tests\run.ps1 -PythonOnly
```

测试覆盖灯条几何、PnP 姿态、时序匹配、EKF 关联与离群筛选、固定协方差、多帧 CSV 和可视化对齐。日志、临时文件与 C++ 测试构建分别位于 `tests/outputs/`、`tests/outputs/tmp/`、`tests/build/`。

相机相关测试还覆盖手动角度控制、归中、自动跟踪预测位置、丢失观测时保持仿真姿态，以及固定基座坐标变换。测试目标名 `camera_gimbal_tests` 保留，但不代表还有独立的 `camera_gimbal.exe` 程序。

| 现象 | 检查方法 |
| --- | --- |
| `cmake` 不可识别 | 确认 CMake 已安装，将其 `bin` 目录加入 PATH 后重新打开终端 |
| 找不到 OpenCV 或 DLL | 确认当前 Conda 环境包含 C++ 开发库、`Library/cmake` 配置，并将其 `Library/bin` 加入 PATH |
| Python 缺少依赖 | 使用项目 `.venv` 的解释器，检查 `setup.ps1` 是否成功完成 |
| 结果只有表头，RMSE 为 nan | 检查颜色、相机标定、有效观测与是否在初始化帧后继续处理 |
| 图中出现竖线或角度跳变 | 检查同帧多板、换板、角度折回，以及真实观测异常 |
| 可视化提示尺寸或诊断不匹配 | 使用同一轮、同编号的原视频、原始 CSV、EKF 结果和诊断文件，并核对相机配置 |
| 定轴旋转时估计中心仍漂移 | 检查角点、PnP 朝向、尺寸、标定、观测权重及模型参数；模型没有固定中心约束 |

## 12. 保留的离线工具：固定基座坐标与控制 CSV

`pose_base.exe` 保留离线坐标变换；原来的离线控制 CSV 转换已合入 `camera_tracking.exe --input ...`，与实时预测共用一个可执行文件，由根目录 CMake 构建。
前者输出基座系装甲板位姿；后者输出相机光轴跟踪偏差和角速度，不连接电机或串口。

本节不是启动两个实时视频模块的前置步骤。`pose_base.cpp` 和离线控制转换目前仍在仓库中，尚未删除；实时模式不读取这里的遥测或标定 CSV。进入离线控制转换时，`--input` 必须是 `camera_tracking.exe` 的第一个参数。

### 坐标与变换顺序

- **基座 B**：原点为参考时刻的相机光心；轴方向仍与基座零位一致，固定 X 向右、Y 向下、Z 向前。原点不随相机后续转动而移动。基座在本次跟踪中必须静止。
- **相机 C**：原点为光心，X 向图像右、Y 向图像下、Z 沿光轴向前。
- yaw 绕基座 +Y 旋转，正值使相机向右；pitch 绕 yaw 后的局部 +X 旋转，正值抬头；roll 绕 pitch 后的局部 +Z 旋转。
- `axis_*_m` 是 yaw 原点到 pitch 原点的向量，表达在 yaw 旋转后的坐标系中。
- `camera_*_m` 是 pitch 原点到光心的向量，表达在 pitch/roll 后的安装坐标系中。
- `mount_qw,qx,qy,qz` 将相机轴旋转到安装坐标系，四元数顺序为 **w,x,y,z**。

```text
R_BC = Ry(yaw) · Rx(pitch) · Rz(roll) · R_mount
t_MC = Ry(yaw) · [t_axis + Rx(pitch) · Rz(roll) · t_camera]
t_BC = t_MC(t) - t_MC(t_ref)
p_B  = R_BC · p_C + t_BC
R_BA = R_BC · R_CA
```

其中 M 为以 yaw 转轴为原点的机械参考系，仅用于外参计算；B 与 M 的轴方向一致。
`t_ref` 默认取遥测第一帧时间，也可在 `pose_base` 和 `camera_tracking --input` 中同时指定相同的 `--origin-time-ms 1000`（毫秒）。
参考时刻的相机位置在 B 系中为零，安装旋转和云台角度仍保留在 `R_BC` 中。

其中 A 为装甲板自身坐标系，PnP 的 `rvec/tvec` 给出 A 到 C 的变换。
原始观测先转到 B 再进入 EKF，不能先在运动相机系滤波、最后仅旋转输出坐标来代替运动补偿。

### 标定与云台反馈输入

必须显式提供两个数值 CSV，不自动假设真实云台角度或安装偏移为零。

`calibration.csv`：表头如下，后面仅一行真实标定值。长度单位米，安装四元数必须接近单位长度。

```csv
axis_x_m,axis_y_m,axis_z_m,camera_x_m,camera_y_m,camera_z_m,mount_qw,mount_qx,mount_qy,mount_qz
```

`telemetry.csv`：与视频共用时间原点，时间单位毫秒，角度单位度。即使没有 roll 轴，也需显式填写 `roll_deg=0`。

```csv
timestamp,yaw_deg,pitch_deg,roll_deg
```

时间戳必须严格递增。非采样时刻按相邻角度的最短方向插值，默认两反馈样本间隔不能超过 100 ms；
可用 `--sync-gap-ms` 调整。反馈必须覆盖输入全部时刻，不允许外推。
这些约定需要与实际编码器零位、符号和时钟校准一致。

### 固定基座系位姿转换

先由配置好的 Armor 生成相机系完整 PnP 位姿，再调用保留的转换工具：

```powershell
.\Armor.exe
.\pose_base.exe --suffix 1 --telemetry telemetry.csv --calibration calibration.csv
```

`data/pose_base_1.csv` 包含基座系位置、完整旋转向量、四元数、水平板朝向和原点元数据。
旧版只有 yaw 的 CSV 无法恢复完整姿态。转换工具拒绝混合或未知坐标系，并检查遥测覆盖。

统一 Armor 入口只处理相机系视频，不读取该 CSV。需要基座系 EKF 的调用者可使用
`ArmorEKF` API 并显式设置 `base_frame=true`；CSV 读取、逐帧时间轴和元数据传递由调用者负责。
`camera_tracking.exe --input ...` 保留离线控制转换，输入须是其要求的预测 CSV，
`--input` 必须放在第一个参数位置。旧有独立 predictor CSV 入口已经移除。
`plot.video` 仅接受相机系投影。

### 原点记录与后续重置接口

pose_base 的基座系 CSV 携带 `base_reference_timestamp_ms` 和 `base_origin_x_m/base_origin_y_m/base_origin_z_m`；后三项表示参考光心在机械系 M 中的位置，单位米。调用者必须保持参考原点一致；控制导出程序核对所用参考时间和原点。

`camera_frames_reset_origin(frames, timestamp_ms)` 可将固定原点改为指定时刻的光心，并返回旧固定系到新固定系的平移变换。轴方向、姿态和速度方向保持不变。当前离线流程通过相同的 `--origin-time-ms` 重新转换、预测、导出；在线重置时还需同步迁移 EKF 的位置状态或重新初始化跟踪器，本次尚未接入在线重置流程。

### 相机跟踪指令

`camera_tracking --input` 使用**当前帧估计中心**，经 `T_BC` 的逆变换得到相机系位置，再计算：

```text
yaw_error   = atan2(x_camera, z_camera)
pitch_error = -atan2(y_camera, hypot(x_camera, z_camera))
```

控制 CSV 每帧一行，字段为：

| 字段 | 含义 |
|---|---|
| `frame_id,timestamp` | 源帧编号、视频时间（ms） |
| `yaw_error_deg,pitch_error_deg` | 原始光轴角度偏差（度），右/上为正 |
| `yaw_offset_deg,pitch_offset_deg` | 限幅后的相对光轴偏差，不是电机绝对角度 |
| `yaw_rate_dps,pitch_rate_dps` | 比例控制、速度与加速度限制后的角速度（度/秒） |
| `target_valid,control_valid` | 目标可用、当前指令可用；执行端必须检查 `control_valid` |
| `angle_limited,speed_limited,acceleration_limited` | 是否发生限幅 |
| `status` | `initializing`、`tracking`、`no_observation`、`invalid_target` 或 `time_gap` |

默认参数：增益 2/s；yaw/pitch 偏差限幅 45°/30°；速度限制 60°/s、45°/s；加速度限制 180°/s²；
中心死区 0.2°；控制帧间隔上限 200 ms。使用 `camera_tracking.exe --input results/armor_prediction_result_1.csv --help` 查看离线参数；单独 `--help` 显示视频模式帮助。
这里的角度限幅是**相对光轴偏差限幅**，不代替实际关节行程限制。
首帧等待时基；没有接受观测、位置无效或时间间隔过大时立即输出零角速度并令 `control_valid=false`。
丢失目标的立即停止优先于加速度限幅。CSV 最后一行不会自动生成后续停止帧，未来硬件执行端必须另设命令超时。
输入是离线视频时，只验证指令计算，不代表已经完成真实电机闭环控制。

### 尚无硬件标定时的仿真验证

```powershell
.\.venv\Scripts\python.exe -B tests/camera_pipeline.py --work-dir tests/outputs/camera-demo-new
```

脚本生成明确标为 `SIMULATION_*` 的反馈和外参，以“目标固定、相机 yaw/pitch/roll 持续变化”验证坐标补偿，
再验证固定系 EKF、四板位姿和相机指令，以及改变参考原点后姿态不变、相机指令一致、混合原点被拒绝。另覆盖丢失目标、无效位置、缺少外参、反馈时间不覆盖和输入保护。
仿真配置仅用于测试，不作为真实平台标定。`tests/run.ps1` 会自动运行这些检查。
