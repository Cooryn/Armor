# RoboMaster Vision：装甲板检测与 EKF 跟踪

本项目使用 C++ / OpenCV 从图片或视频中检测装甲板，通过 PnP 解算装甲板位姿，再由 Python 扩展卡尔曼滤波器（EKF）估计车体中心、旋转朝向、角速度和装甲板几何参数。支持导出检测视频、原始观测、滤波结果、诊断曲线和状态叠加视频。

主要工作流面向单辆小车的四装甲板跟踪。默认采用无窗口批处理，适合离线检查检测与跟踪效果。

## 1. 功能与数据流程

| 模块 | 输入 | 功能 | 输出 |
| --- | --- | --- | --- |
| 装甲板检测 | 红色或蓝色装甲板图片、视频 | 颜色提取、灯条筛选、端点定位、灯条配对 | 装甲板角点与检测质量 |
| PnP 解算 | 角点、相机标定、装甲板尺寸 | 平面姿态候选求解、筛选与短时关联 | 装甲板位置、朝向、重投影误差 |
| 装甲板 EKF | 逐帧 PnP 观测 | 多板关联、质量加权、状态估计、漏检预测 | 车体状态、观测诊断、残差统计 |
| 图表与视频 | 原始观测、EKF 结果、原视频 | 绘制曲线与投影状态 | PNG 图表、MP4 视频 |

```text
输入视频 → Armor.exe → data/pose_raw_*.csv → predictor_armor.exe
              │                 │                    │
              ▼                 ▼                    ▼
         检测视频           plot.raw            EKF 结果与诊断
                                │                    │
                                ▼                    ▼
                           原始观测曲线       plot.armor / plot.video
                                                     │
                                                     ▼
                                               EKF 图表与视频
```

EKF 核心使用 C++ 和 Eigen；CSV 读写由 C++ 运行程序负责，Python 图表和视频导出由 `plot/` 负责。修改图表样式后可以直接重新绘图，不必重跑检测或滤波。

## 2. 目录与脚本职责

```text
Armor/
├── CMakeLists.txt                 C++ 构建配置
├── setup.ps1                      Python 环境配置与 C++ 构建
├── run.ps1                        C++ 检测运行入口
├── requirements.txt               Python 依赖
├── include/                       检测器、预测器与输入流头文件
├── src/
│   ├── main.cpp                   输入参数、相机参数、检测流程、文件导出
│   ├── lightbar_detector.cpp      灯条检测、端点定位、配对与绘制
│   ├── solver.cpp                 PnP 候选求解、有效性检查、短时姿态关联
│   ├── predictor.cpp              基础预测器，独立 main 与 CSV 流程
│   ├── predictor_polar.cpp        极坐标预测器，独立 main 与 CSV 流程
│   └── predictor_armor.cpp        装甲板预测器，独立 main 与 CSV 流程
├── plot/
│   ├── raw.py                     原始 PnP 观测曲线
│   ├── armor.py                   11 维装甲板 EKF 图表
│   ├── video.py                   原视频叠加观测、估计状态与装甲板边框
│   ├── basic.py                   6 维模型图表
│   ├── polar.py                   9 维模型图表
│   └── _cli.py                    静态绘图命令的公共路径参数
├── assets/image/                  输入图片
├── assets/video/                  输入视频
├── data/                          C++ 导出的原始观测 CSV
├── results/                       正式 CSV、TXT、PNG 和 MP4 输出
├── tests/                         回归测试与离线评估工具
│   ├── plotting/                  测试用图像导出
│   ├── build/                     C++ 测试构建目录
│   └── outputs/                   测试日志、临时文件、评估产物
├── build/                         正式 C++ 构建目录
└── .venv/                         项目 Python 环境
```

`Armor.exe` 构建后位于仓库根目录。正式输出保存在 `data/`、`results/`；测试脚本和测试产物集中在 `tests/`。补充说明见 [绘图说明](plot/README.md)和[测试说明](tests/README.md)。

## 3. 环境配置

### 3.1 依赖

| 依赖 | 用途 |
| --- | --- |
| Miniconda / Python | 提供基础 Python 环境，创建项目 `.venv` |
| Visual Studio 2022 C++ 工具 | 编译 C++17，需要桌面 C++ 开发工具和 Windows SDK |
| CMake | 生成并构建 Visual Studio x64 工程 |
| OpenCV C++ 开发库 | 图像处理、视频读写、PnP；需要头文件、链接库和 CMake 配置 |
| Eigen ≥ 3.3 | 三种预测器的向量、矩阵和线性求解；纯头文件库，当前验证版本为 5.0.1 |
| FFmpeg | 将检测视频转为 MP4，需要 `mpeg4` 编码器 |
| NumPy、pandas、Matplotlib、Python OpenCV | Python 参考回归、CSV 处理、图表与视频导出 |

Python 包版本要求见 [requirements.txt](requirements.txt)。仓库现有环境使用 Python 3.13 和 OpenCV 4.13。Python 的 `opencv-python` 包不代替 C++ OpenCV 开发库。

### 3.2 配置与构建

在 PowerShell 中进入仓库根目录，激活安装了 C++ OpenCV 和 FFmpeg 的 Conda 环境。若依赖安装在 `base` 中：

```powershell
conda activate base
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

`setup.ps1` 使用当前 `python` 创建可复用基础环境包的 `.venv`，安装 Python 依赖，从基础环境的 `Library/cmake` 查找 OpenCV，再通过根目录 CMake 构建 Release 版 `Armor.exe` 和三个预测程序，均输出至仓库根目录。

脚本要求 `python`、`cmake` 已能在终端调用，不负责安装 Visual Studio、Conda 或 C++ OpenCV。已有 `.venv` 时会复用它；运行时应使用与其对应的 Conda 基础环境。

只重新编译 C++：

```powershell
cmake --build build --config Release --parallel
```

Python 命令统一使用 `.\.venv\Scripts\python`，不要求额外激活 `.venv`。

## 4. 快速运行

以下命令在仓库根目录执行。同一编号的检测、滤波、绘图和视频必须配套使用；同名输出会被覆盖。

### 4.1 单段视频

```powershell
# 检测视频，导出原始观测
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Mode video -Color red -InputFile video_1.avi

# 运行装甲板 EKF
.\predictor_armor.exe --suffix 1

# 绘制原始数据、EKF 结果和视频
.\.venv\Scripts\python -B -m plot.raw --suffix 1
.\.venv\Scripts\python -B -m plot.armor --suffix 1
.\.venv\Scripts\python -B -m plot.video --suffix 1
```

处理第二段视频时，将输入改为 `video_2.avi`，所有 `--suffix` 改为 `2`。

### 4.2 批量处理两段视频

```powershell
foreach ($suffix in 1, 2) {
    powershell -ExecutionPolicy Bypass -File .\run.ps1 -Mode video -Color red -InputFile "video_$suffix.avi"
    if ($LASTEXITCODE -ne 0) { throw "检测失败：video_$suffix" }
    .\predictor_armor.exe --suffix $suffix
    if ($LASTEXITCODE -ne 0) { throw "EKF 运行失败：$suffix" }
    foreach ($module in 'raw', 'armor', 'video') {
        .\.venv\Scripts\python -B -m "plot.$module" --suffix $suffix
        if ($LASTEXITCODE -ne 0) { throw "导出失败：$module，编号 $suffix" }
    }
}
```

正常完成后，每段视频对应 `data/` 中 1 个原始 CSV 和 `results/` 中 12 个正式结果文件。

### 4.3 图片检测

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Mode image -Color red -InputFile image_1.jpg
```

图片模式保存 `results/image_1.png`，不产生用于时序 EKF 的观测 CSV。

## 5. 命令参数

### 5.1 检测程序

`run.ps1` 接受 `-Mode image|video`、`-Color red|blue`、`-InputFile`，默认值分别为 `video`、`red`、`video_2.avi`。脚本会将 Conda 的 `Library/bin` 加入本次运行的 DLL 搜索路径。

配置好 DLL 和 FFmpeg 路径后，也可直接运行：

```powershell
.\Armor.exe video red video_1.avi
.\Armor.exe image blue image_3.jpg
```

直接运行 `Armor.exe` 的默认模式是 `image`，颜色为 `red`。未指定文件时，图片模式使用 `image_1.jpg`，视频模式使用 `video_1.avi`。

程序先检查传入路径，找不到时从 `assets/<模式>/<文件名>` 查找。检测视频按源文件名输出，例如 `video_1.avi` 对应 `results/video_1.mp4`。CSV 编号取文件名主干最后一个下划线后的部分；无下划线时使用整个主干，例如 `sample.avi` 对应 `data/pose_raw_sample.csv`。

默认不打开窗口。安装了支持 GUI 的 C++ OpenCV 时，可传入第四个参数 `--gui`：

```powershell
.\Armor.exe video red video_1.avi --gui
```

GUI 中可调颜色阈值、灯条角度差、长度比、装甲板宽高比和灯条错位。图片模式按 `S` 保存、`Esc` 退出；视频模式按 `Esc` 提前结束。批处理不需要 GUI 支持。

### 5.2 装甲板 EKF

| 参数 | 含义 | 默认 |
| --- | --- | --- |
| `--suffix` | 输入、输出编号 | `1` |
| `--input` | 指定原始 CSV，优先于编号输入路径 | `data/pose_raw_<suffix>.csv` |
| `--output-dir` | 数值结果输出目录 | `results/` |
| `--fixed-noise` | 使用固定观测协方差，关闭质量加权 | 不启用，默认按质量加权 |
| `--prediction-horizon-ms` | 相对于图像时间的未来预测提前量，单位 ms；`0` 关闭提前预测 | `50` |

```powershell
.\predictor_armor.exe --input data/pose_raw_1.csv --suffix 1 --output-dir results/custom

# 固定权重的对照运行单独保存
.\predictor_armor.exe --suffix 1 --fixed-noise --output-dir tests/outputs/fixed_noise
```

默认预测图像采集时间之后 50 ms 的四块板位置与水平朝向。使用
`--prediction-horizon-ms 100` 可改为 100 ms，修改后需重新运行 `plot.video`。
这是固定预测提前量，不会人为暂停程序，也不自动测量处理、通信或云台响应延迟。

### 5.3 静态图表

`plot.raw`、`plot.armor`、`plot.basic`、`plot.polar` 都支持：

| 参数 | 含义 |
| --- | --- |
| `--suffix` | 数据编号；`armor` 默认 `1`，其余默认 `2` |
| `--data-dir` | 原始观测目录，默认 `data/` |
| `--results-dir` | 预测 CSV 所在目录，默认 `results/` |
| `--output-dir` | 图像输出目录，默认与结果目录相同 |

```powershell
.\.venv\Scripts\python -B -m plot.armor --suffix 1 --results-dir results/custom --output-dir results/custom
```

### 5.4 视频可视化

| 参数 | 含义 |
| --- | --- |
| `--suffix 1|2` | 数据编号，默认 `1`，同时决定默认相机配置 |
| `--video` | 原视频路径，默认 `assets/video/video_<suffix>.avi` |
| `--predictions`、`--raw` | 显式指定 EKF 结果和原始观测 CSV |
| `--diagnostics` | 显式指定逐条观测诊断 CSV |
| `--no-diagnostics` | 不加载诊断；不能与 `--diagnostics` 同用 |
| `--camera-profile 1|2` | 相机配置，默认与编号一致 |
| `--data-dir`、`--results-dir` | 原始 CSV 与 EKF 结果目录 |
| `--output-dir` | 视频输出目录，默认使用结果目录 |
| `--output` | 完整输出路径，优先于输出目录 |
| `--start-frame` | 起始帧，默认 `0` |
| `--max-frames` | 最多导出帧数，默认导出到结束 |
| `--trail-length` | 中心轨迹保留帧数，默认 `30`，`0` 关闭 |

只检查一小段视频：

```powershell
.\.venv\Scripts\python -B -m plot.video --suffix 1 --start-frame 100 --max-frames 60 --output tests/outputs/video_preview.mp4
```

## 6. 坐标、角度与物理尺寸

### 6.1 相机坐标系

位置以米为单位，原点在相机光心：`x` 向右、`y` 向下、`z` 向相机前方。CSV 角度为弧度，角速度为弧度/秒。

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

相机内参与畸变参数分别位于 `src/main.cpp` 和 `plot/video.py`；物理板尺寸位于 `src/solver.cpp` 和 `plot/video.py`。使用新相机、新分辨率或不同板型时，需要同步配置检测与可视化。可视化会检查画面尺寸，但相同分辨率本身不意味着相机标定相同。

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
4. 优先选择数量最多的一致观测集合，再比较关联代价。
5. 将选中观测联合更新到同一个车体状态，使用各自的观测协方差。

初始化帧只使用最近的一条有效观测，将其定义为 A0；该帧其他观测不参与更新。当前流程没有多车分组，同帧输入默认属于同一辆小车。

### 7.3 质量加权与漏检

默认根据检测评分、重投影误差和水平观察角度调整观测协方差。低分、高重投影误差的观测权重降低，侧视板的距离和朝向权重额外降低。倍率有上限，质量字段缺失或全部无效时使用基础固定协方差。这是启发式加权，不是对实际测量噪声的统计标定。

| 构造参数 | 默认值 | 含义 |
| --- | --- | --- |
| `nis_gate` | `16.0` | 归一化创新平方门限 |
| `max_distance_error` | `0.5` m | 距离残差绝对上限 |
| `pair_yaw_tolerance` | 25°，传参单位为弧度 | 板间朝向关系容差 |
| `adaptive_noise` | `True` | 是否启用质量加权 |

过程噪声、基础观测协方差和初始化设置位于 `include/predictor_armor.hpp`。调参时应同时检查残差、中心漂移、接受率和运动变化时的响应。

漏检或所有观测被拒绝时，EKF 只预测。CSV 中间的空缺帧按相邻时间戳恢复时间轴，不生成插值观测。输出从初始化帧之后开始，到最后一个原始观测帧结束，不推测 CSV 未记录的开头和尾部空帧。少于两帧可用观测时不会产生新的预测结果，已有同名文件不会因此自动删除。

### 7.4 C++ 调用

三个预测程序均可单独编译一个 `.cpp` 后运行，无需链接共用预测器库。也可在自己的程序中直接使用 Eigen 滤波器头文件：

```cpp
#include "predictor_armor.hpp"

predictor::ArmorEKF ekf; // 默认启用自适应观测噪声
// 纯滤波器：initialize / predict / update_multi / forecast
// CSV 批处理通过 predictor_armor.exe --suffix 1 运行。
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

旧相机系流程使用当前相机坐标系；第 12 节的流程先按云台姿态和安装外参转换到固定基座系，再进行滤波和未来预测。EKF 仍采用水平旋转模型，不估计板面俯仰、横滚。
未来预测误差应与目标时刻、同一板号和同一坐标系下的后续观测比较。

## 8. 输出文件与读法

下表中的 `*` 表示编号，例如 `1` 或 `2`。主流程每段视频在 `results/` 生成 12 个文件。

| 文件 | 内容与用途 |
| --- | --- |
| `video_*.mp4` | C++ 检测视频，显示通过 PnP 检查的装甲板边框 |
| `armor_video_*.mp4` | EKF 叠加视频，显示观测、估计车体中心、四块板及状态信息 |
| `pose_raw_curve_*.png` | 原始装甲板中心的 x、y、z、距离、目标俯仰角曲线 |
| `raw_yaw_curve_*.png` | 原始目标方位角与装甲板朝向角，均未经过 EKF |
| `top_down_trajectory_*.png` | X–Z 平面上的装甲板原始观测与更新前预测位置 |
| `body_yaw_curve_*.png` | EKF 车体参考朝向，角度按 ±π 折回 |
| `folded_armor_yaw_curve_*.png` | 当前选中板号的预测朝向与观测朝向对比，使用折回后的角度 |
| `armor_prediction_error_curve_*.png` | 目标方位、俯仰、距离、板朝向的更新前残差 |
| `armor_prediction_result_*.csv` | 每帧一行的 EKF 状态和代表性观测残差 |
| `armor_future_prediction_*.csv` | 每个源帧四行，记录四块板在目标时刻的未来位置和水平朝向 |
| `armor_observation_diagnostics_*.csv` | 每条观测一行的关联、接受情况、拒绝原因、权重倍率 |
| `armor_rmse_result_*.txt` | 预测结果 CSV 中四项残差的 RMSE |

原始曲线连接 CSV 中的所有观测，没有按物理板号分组。同帧多板会出现相同横坐标的多个点，换板也会产生跳变。**原始位置是装甲板中心，不是车体中心。**

俯视图的原始轨迹取每帧最近的原始观测；预测轨迹对应最近的被接受观测，全部被拒绝时使用最近预测板。两条曲线不保证每帧对应同一块物理板，不能把图上的距离直接作为定位误差。

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
| `target_yaw_noise_scale, target_pitch_noise_scale` | 方位、俯仰观测方差倍率 |
| `distance_noise_scale, yaw_noise_scale` | 距离、板朝向观测方差倍率 |
| `reason` | 接受或未使用的原因，见下表 |

| `reason` 值 | 含义 |
| --- | --- |
| `accepted` | 已关联并参与更新 |
| `initialization` | 用作初始化种子 |
| `initialization_unused` | 初始化帧中的其他观测，未参与更新 |
| `invalid` | 观测形状、数值或范围无效 |
| `innovation_gate` | NIS 或距离残差门限未通过 |
| `association_conflict` | 存在独立候选，但未进入最终联合关联组合 |

噪声倍率是相对于基础 `R` 的**方差倍率**，不是标准差倍率。初始化阶段没有观测更新，相应字段可以为空。`accepted=False` 不一定代表误检，例如初始化帧的未使用观测。

### 8.4 残差与精度

误差图和 RMSE 文本统计每帧代表性被接受观测的更新前残差，角度差折回到 ±π。它们不是相对于真实位置或姿态的误差，也不是同帧所有观测的汇总，更不是 50 ms 未来预测误差。

残差小表示预测与检测结果较一致。评估时还需要查看拒绝率、连续漏检、中心漂移、角速度稳定性和停车响应；确认绝对精度需要实际尺寸、相机标定与外部真值。

## 9. 视频画面说明

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

## 10. 其他预测模型

| 模型 | 状态 | 运行 / 绘图 |
| --- | --- | --- |
| 基础 6 维模型 | 位置和速度 `[x, vx, y, vy, z, vz]` | `include/predictor.hpp` / `plot.basic` |
| 极坐标 9 维模型 | 中心、速度、车体朝向、角速度、半径 | `include/predictor_polar.hpp` / `plot.polar` |

两个脚本的直接运行入口默认使用编号 `2`，编号在各自 `__main__` 中设置。也可调用 `run_predict` 或 `run_predict_polar` 显式传入 CSV、输出目录和编号。

```powershell
.\predictor.exe
.\.venv\Scripts\python -B -m plot.basic --suffix 2 --output-dir results/basic

.\predictor_polar.exe
.\.venv\Scripts\python -B -m plot.polar --suffix 2 --output-dir results/polar
```

基础模型导出 `prediction_result_*.csv`、`rmse_result_*.txt`；极坐标模型导出 `polar_prediction_result_*.csv`、`polar_rmse_result_*.txt`。基础模型与完整装甲板模型的俯视图同名，比较时应指定不同图像输出目录。

## 11. 测试与常见问题

```powershell
# Python 与 C++ 回归测试
powershell -ExecutionPolicy Bypass -File .\tests\run.ps1

# 只运行 Python 测试
powershell -ExecutionPolicy Bypass -File .\tests\run.ps1 -PythonOnly
```

测试覆盖灯条几何、PnP 姿态、时序匹配、EKF 关联与离群筛选、质量加权、协方差、多帧 CSV 和可视化对齐。日志、临时文件与 C++ 测试构建分别位于 `tests/outputs/`、`tests/outputs/tmp/`、`tests/build/`。

| 现象 | 检查方法 |
| --- | --- |
| `cmake` 不可识别 | 确认 CMake 已安装，将其 `bin` 目录加入 PATH 后重新打开终端 |
| 找不到 OpenCV 或 DLL | 确认当前 Conda 环境包含 C++ 开发库、`Library/cmake` 配置和运行库，通过 `run.ps1` 运行 |
| Python 缺少依赖 | 使用项目 `.venv` 的解释器，检查 `setup.ps1` 是否成功完成 |
| OpenCV 无法创建窗口 | 使用默认批处理；`--gui` 需要支持 GUI 的 C++ OpenCV |
| FFmpeg 转码失败 | 检查 `ffmpeg` 是否可调用及是否支持 `mpeg4`；失败时保留中间 AVI |
| 没有新的 EKF 文件 | 检查输入路径、有效帧数和终端提示，不要把已有同名文件当成本次结果 |
| 图中出现竖线或角度跳变 | 检查同帧多板、换板、角度折回，以及真实观测异常 |
| 可视化提示尺寸或诊断不匹配 | 使用同一轮、同编号的原视频、原始 CSV、EKF 结果和诊断文件，并核对相机配置 |
| 定轴旋转时估计中心仍漂移 | 检查角点、PnP 朝向、尺寸、标定、观测权重及模型参数；模型没有固定中心约束 |

## 12. 纯相机云台：固定基座坐标与控制 CSV

新增独立程序 `pose_base.exe` 和 `camera_gimbal.exe`，由同一个根目录 CMake 构建。
前者输出基座系装甲板位姿；后者输出相机光轴跟踪偏差和角速度，不连接电机或串口。

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
`t_ref` 默认取遥测第一帧时间，也可在 `pose_base` 和 `camera_gimbal` 中同时指定相同的 `--origin-time-ms 1000`（毫秒）。
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

### 固定基座系处理流程

在根目录执行，先重新运行检测，取得新增的 `rvec_x/rvec_y/rvec_z` 和 `coordinate_frame=camera`。
旧版只有 yaw 的原始 CSV 无法恢复完整三维姿态，不能直接用于完整位姿转换。

```powershell
cmake --build build --config Release --parallel
.\run.ps1 -Mode video -Color red -InputFile video_1.avi
.\pose_base.exe --suffix 1 --telemetry telemetry.csv --calibration calibration.csv
.\predictor_armor.exe --input data/pose_base_1.csv --suffix 1 --output-dir results/base
.\camera_gimbal.exe --input results/base/armor_prediction_result_1.csv --output results/base/camera_gimbal_1.csv --telemetry telemetry.csv --calibration calibration.csv
```

`data/pose_raw_1.csv` 保留原始相机测量，`data/pose_base_1.csv` 是正式基座系观测输出。
它包含基座系位置、完整旋转向量、完整四元数，以及用于水平 EKF 的板面法向水平投影角。
`predictor_armor` 自动识别 `coordinate_frame=base`，中心、速度、诊断角度和四板未来位姿均使用基座系；
基础、极坐标预测器也保留输入坐标系标签。混合或未知坐标系会报错。
基座系四板预测额外输出 `qw/qx/qy/qz`，它们来自水平旋转模型，仅描述估计的 yaw，并非完整姿态滤波。
旧 `plot.video` 仅支持相机系投影，会明确拒绝基座系 CSV，避免把固定系坐标直接画到相机图像上。

### 原点记录与后续重置接口

基座系 CSV 及预测结果携带 `base_reference_timestamp_ms` 和 `base_origin_x_m/base_origin_y_m/base_origin_z_m`；后三项表示参考光心在机械系 M 中的位置，单位米。预测器拒绝混合不同参考原点的数据，控制导出程序核对所用参考时间和原点。

`CameraFrames::reset_origin(timestamp_ms)` 可将固定原点改为指定时刻的光心，并返回旧固定系到新固定系的平移变换。轴方向、姿态和速度方向保持不变。当前离线流程通过相同的 `--origin-time-ms` 重新转换、预测、导出；在线重置时还需同步迁移 EKF 的位置状态或重新初始化跟踪器，本次尚未接入在线重置流程。

### 相机跟踪指令

`camera_gimbal` 使用**当前帧估计中心**，经 `T_BC` 的逆变换得到相机系位置，再计算：

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
中心死区 0.2°；控制帧间隔上限 200 ms。使用 `camera_gimbal.exe --help` 查看对应参数。
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
