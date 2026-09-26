# 绘图脚本说明

这里的脚本读取已有数据生成图像，不负责运行 EKF。默认输出到仓库 `results/`。
命令从仓库根目录执行，使用 `.\.venv\Scripts\python -m plot.<模块>`。

| 脚本 | 职责 | 输入 | 输出 |
| --- | --- | --- | --- |
| `raw.py` | 检查 C++ 导出的原始观测，尚未经过 EKF | `data/pose_raw_*.csv` | `pose_raw_curve_*.png`：位置、距离、俯仰；`raw_yaw_curve_*.png`：目标偏航和装甲板朝向 |
| `armor.py` | 展示完整装甲板 EKF 的结果 | `results/armor_prediction_result_*.csv` 和原始观测 CSV | 俯视轨迹、车体朝向、装甲板朝向对比、先验残差，共四张图 |
| `basic.py` | 展示基础 6 维预测器的结果 | `results/prediction_result_*.csv` 和原始观测 CSV | 误差曲线、俯视轨迹 |
| `polar.py` | 展示 9 维极坐标预测器的结果 | `results/polar_prediction_result_*.csv` | 中心位置、速度、朝向、半径等状态图，以及观测残差图 |
| `video.py` | 将 EKF 中心、四块板和原始观测投影到视频；显示筛选状态和目标放大图 | 对应编号的视频、原始观测、预测结果，可选观测诊断 CSV | `results/armor_video_*.mp4` |
| `_cli.py` | 为静态绘图脚本提供公共命令行参数和路径解析 | 命令行参数 | 不生成图像，不单独运行 |
| `__init__.py` | 将目录声明为 Python 包 | 无 | 不生成图像，不单独运行 |

当前主要调试装甲板检测与完整 EKF 时，使用 `raw.py`、`armor.py` 和 `video.py` 即可。
`basic.py`、`polar.py` 用于保留的旧预测器对照。

```powershell
# 直接查看原始检测数据，不需要先运行预测
.\.venv\Scripts\python -m plot.raw --suffix 1

# 先运行完整 EKF，再独立绘图
.\predictor_armor.exe --suffix 1
.\.venv\Scripts\python -m plot.armor --suffix 1

# 视频叠加：自动选择对应相机参数
.\.venv\Scripts\python -m plot.video --suffix 1
.\.venv\Scripts\python -m plot.video --suffix 2
```

`raw`、`armor`、`basic`、`polar` 支持：

- `--suffix 1` 或 `--suffix 2`：数据编号。`armor` 默认 1，其余默认 2。
- `--data-dir`：原始观测目录，默认 `data/`。
- `--results-dir`：预测结果目录，默认 `results/`。
- `--output-dir`：图像输出目录，默认与 `--results-dir` 相同。

`video.py` 同样支持上述参数，并支持 `--video`、`--predictions`、`--raw`、`--diagnostics`
指定文件，`--output` 指定输出文件，`--camera-profile 1|2` 明确指定相机标定。
相机配置分别要求原视频尺寸 1440×1080、1280×1024，尺寸不符时停止，避免错误投影。
输出保持原帧率和原画面尺寸，右侧追加 340 像素信息栏；不复制音轨。

```powershell
# 只检查第 745 帧起的 30 帧；测试片段集中放入 tests/outputs
.\.venv\Scripts\python -m plot.video --suffix 1 --start-frame 745 --max-frames 30 --output tests/outputs/video_preview.mp4
```

彩色实线 A0–A3 和圆圈表示当前帧的 EKF 估计装甲板，同色虚线 F0–F3 和菱形表示未来预测。
加号表示观测，红色叉号表示被拒绝的观测。白色星形为当前中心，白色菱形为未来中心。
边框按与 C++ PnP 相同的 135×56 mm 尺寸构建三维角点，随各板 yaw 旋转后带畸变投影；
局部放大图也显示边框。它是 EKF 几何模型的投影，不是原始检测框；当前模型不包含板面俯仰和横滚。
接收观测的板号来自诊断 CSV，不通过 yaw 临时猜测；缺少诊断时显示黄色未分类观测，
也可用 `--no-diagnostics` 显式关闭诊断标记。A0–A3 是跟踪器相对编号，不保证对应物理前后左右。
右侧显示帧号、源视频时间、更新/只预测/无状态、中心、角速度、结构参数及目标局部放大图。
中心轨迹默认保留 30 帧，`--trail-length 0` 可关闭。当前帧没有 EKF 记录时不沿用旧状态。
侧栏显示未来预测提前量和目标时间。提前量由 `predictor_armor.exe --prediction-horizon-ms`
设置，默认 50 ms；修改后需重新生成预测 CSV 再绘制视频。`0` 关闭提前预测框。
旧 CSV 缺少未来字段时只显示当前估计，字段不完整或时间不一致时停止导出并提示。
未来框叠加在当前画面上，仅表示预期位置，不应与当前观测直接计算未来预测误差。
四块板全部显示，未模拟遮挡。误差图和 RMSE 文本仍是当前帧更新前残差，不是未来预测误差。

`basic` 与 `armor` 的俯视图都叫 `top_down_trajectory_*.png`；同编号对照时请通过
`--output-dir` 分开保存，避免相互覆盖。

图像中的 EKF 误差是相对于观测的先验残差，不等同于真实定位误差。
检测前后对照工具属于测试，位于 `tests/compare_detector.py`，产物集中到 `tests/outputs/`。
