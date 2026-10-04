# 测试与诊断目录

固定观测噪声调参：先构建 `armor_noise_evaluation`，再运行
`.\.venv\Scripts\python -B -m tests.tune_fixed_noise`。方法和参数见 [调参说明](fixed_noise_tuning.md)。
`test_fixed_noise.py` 检查质量字段不影响更新、旧 CSV 兼容、离群门控、
固定协方差的联合更新、顺序不变性及诊断字段导出。

PnP 第 2 项对比：`.\.venv\Scripts\python -B -m tests.plotting.pnp_candidates`。
基线为第 1 项完成后的数据，产物位于 `outputs/pnp_candidates/`。
`test_detector.cpp` 同时检查畸变下的合成姿态、yaw 符号、匹配顺序不变性、
歧义关联禁用历史以及漏检/时间间隔后的历史清理。

所有测试脚本、检测前后对照、历史分析图片、临时试验和测试构建集中在这里。
正式程序输出仍位于仓库的 `data/`、`results/`。

原生测试直接初始化滤波和控制数据，调用更新接口检查参数错误；配置校验随首次更新执行。三种在线预测模型继续按同一帧时间轴与 NumPy 对照，验证合并辅助函数后状态、诊断、预测几何及 RMSE 保持一致。

标准库表示简化的回归覆盖：板号 `-1` 自动关联、指定板号 `0`、非法板号、时间 `0` 的重复帧检查和目标丢失计时，以及默认遥测原点、显式原点 `0` 和非法原点时间。

```text
tests/
  test_*.py                  Python 回归测试
  test_detector.cpp          C++ 检测回归测试
  run.ps1                    统一测试入口
  compare_detector.py        前后对照入口
  plotting/                  测试用图像导出
  build/                     C++ 测试构建和 CTest 内部日志（忽略提交）
  outputs/                   测试产物（忽略提交）
    detector_baseline/       修改前的数据、视频与源码快照
    detector_comparison/     对照统计与图片
    detector_trials/         中间参数试验
    motion_analysis/         运动一致性诊断图片
    legacy_build/            原 build 下的历史测试产物
    tmp/                     测试临时文件
```

在仓库根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\tests\run.ps1
# 只运行 Python 回归测试
powershell -ExecutionPolicy Bypass -File .\tests\run.ps1 -PythonOnly
# 根据保存的基线和当前正式输出生成对照
.\.venv\Scripts\python -B tests/compare_detector.py
```

正式 CMake 构建默认关闭 `BUILD_TESTING`；测试入口在 `tests/build` 单独开启。
对照工具写入 `tests/outputs/detector_comparison`，不向正式 `results` 添加测试图片。
# 灯条端点细化对比

运行 `.\.venv\Scripts\python -B -m tests.plotting.light_refinement`，
读取 `outputs/light_refinement/before/` 与当前 `data/`、`results/`，
在 `outputs/light_refinement/` 保存统计表、对比图和相同帧的检测截图。
EKF 统计窗口：视频 1 为 2–25 秒，视频 2 为 2–10 秒；
漏检和双板异常统计覆盖全视频。残差只展示每帧最近的被接受观测。

## 在线预测回归

`include/predictor*.hpp` 声明 EKF 接口，`src/predictor*.cpp` 实现算法。
`src/Armor.cpp` 只包含统一视频配置及模块调用链。
`predictor.hpp/.cpp` 提供 `video_predictor_*`、`prediction_output_*` 普通函数，状态使用只存数据的结构体，均不依赖 OpenCV；
`lightbar_detector.hpp/.cpp` 只负责检测与检测结果绘制，`solver.hpp/.cpp` 提供逐帧 PnP。
视频输入、窗口及预测画面输出放在现有 `camera_tracking.hpp/.cpp`。
程序通过 `predictor_filters`、`armor_vision`、`armor_video` 链接对应实现，无新增模块文件；
预测接口与导出测试只链接 `predictor_filters`，不再依赖图像模块。
三个独立 predictor 可执行目标及 CSV 命令行入口已移除。

`test_predictors.cpp` 验证解析雅可比、零创新、跨界角度、变步长轨迹、
协方差正定性，以及 Eigen 增益/NIS 与独立 LU 计算。
`test_video_predictor.cpp` 验证最近有效板、并列首条、四板联合更新、连续漏检、
末尾预测、空/单帧输入、非变异未来预测、非法时钟和写入错误；
测试通过公共头文件调用 `video_predictor_update` 和 `prediction_output_write/finish`，
不再包含入口 `.cpp` 或依赖排除入口的编译宏。
同一测试程序提供测试专用的 `--replay`，不作为正式 CSV 工具。

`reference/predictor/video_predictor.py` 使用 NumPy 按完整视频帧时钟重放，
`verify_predictor_equivalence.py` 对比 C++ 全部结果列、关联诊断、未来四板、
显示几何及六位小数 RMSE。两段真实记录和合成序列共 35 组案例。
其余 Python 参考 CSV runner 保留为历史回归夹具，不作为应用入口。

```powershell
ctest --test-dir tests/build -C Release --output-on-failure
.\.venv\Scripts\python.exe -B tests/verify_predictor_equivalence.py --python-root . --bin-dir tests/build/Release --work-dir tests/outputs/predictor-comparison-new
```

上述对照需要两份 `data/pose_raw_1.csv` / `pose_raw_2.csv`，
可依次配置并运行 Armor 生成。`run.ps1` 构建/运行原生测试、Python 测试、
在线 NumPy 对照和相机坐标/控制仿真。

视频集成验证单独运行，会临时修改顶部源码参数，分别构建三种模型及红/蓝检测。
蓝色轨迹夹具交换现有红色视频的 B/R 通道。检查完整帧数、源 FPS、全部输出列、
相机系标签及 RMSE，并保存叠加截图。结束时恢复原始配置并重建 Armor。

```powershell
.\.venv\Scripts\python.exe -B tests/verify_armor_video.py --preview
```

`--preview` 额外打开六帧 OpenCV 预览，不提供命令行视频/模型覆盖。
集成视频、CSV、截图和构建日志集中保存于 `tests/outputs/armor-video/`。

相机坐标/控制仿真通过测试专用 `camera_filter_replay` 调用 `ArmorEKF`（`base_frame=true`），
不依赖已删除的 predictor 可执行文件。Python 调用方按帧传递基座元数据，
并在控制导出端检查混合原点及错误原点；测试桥接程序只在 `BUILD_TESTING=ON` 时构建。
Polar 另有独立物理几何回归，检查 30° 朝向下中心、四板位置及静止未来预测，避免仅靠同公式的 NumPy 对照掩盖符号错误。
