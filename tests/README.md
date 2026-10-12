# 测试与诊断

正式入口为 Armor（检测与跟踪）和 Armor_detect（仅检测/PnP预览）。以下软件测试不连接实际摄像头、串口或电机，测试产物在 tests/outputs。Armor 的正式输出集中于 results，按 video/camera、single/armor、视频主干或 offline/mcu 分目录，同一次运行的观测与预测日志保存在一起。

```powershell
powershell -ExecutionPolicy Bypass -File .\tests\run.ps1
# 只运行 Python 测试
powershell -ExecutionPolicy Bypass -File .\tests\run.ps1 -PythonOnly
ctest --test-dir tests/build -C Release --output-on-failure
```

CMake 默认关闭 BUILD_TESTING；测试使用 tests/build 单独构建。run.ps1 包含原生回归、Python 回归、NumPy 预测对照和固定坐标控制回放。

生产源码假设调用方提供正确图像、递增时间戳、有效参数和标定；不再要求错误输入触发额外格式校验。算法门控、几何筛选、漏检行为、串口分包及实际 I/O 错误继续验证。历史 NumPy 离线参考仍保留自己的输入检查。

## 原生测试

| 文件 / CTest | 检查 |
| --- | --- |
| test_serial.cpp / serial_regressions | 模拟系统串口 I/O，运行真实后台线程：MC02 黄金字节、CRC、拆包/粘包、最新 STATE、错误传递、close 排空、析构及重开 |
| test_detector.cpp / detector_regressions | 灯条轮廓、配对、PnP 候选、畸变与 yaw 符号、时间关联、漏检历史清理 |
| test_predictors.cpp / predictor_api_regressions | 两种 EKF 的雅可比、零创新、角度跨界、变步长、协方差和 Eigen 求解 |
| test_camera_gimbal.cpp / pose_base_regressions | 最新 STATE 到完整位姿，yaw 转轴原点、初始光轴水平投影、保留地面 pitch、安装偏移/旋转、零位/方向、跨 pi 转动、长期保持初始朝向、固定系后方观测 |
| test_camera_tracking.cpp / armor_pipeline_regressions | 原跟踪测试迁移到统一接口：Camera 视频时钟、真实合成 PnP、移动相机固定目标、两种 EKF、连续漏检、绝对角度光轴对准、外参/零位/方向/限位/时效、CSV、逆变换绘制及 AVI |
| test_video_predictor.cpp / video_predictor_regressions | 最近有效板/并列首条、四板联合更新、空/单帧/连续漏检、非变异预测、输出列/RMSE及文件写入错误 |

原两个相机测试保留文件名，内容已迁移到新模块，生产代码不再包含 CameraTracking/CameraGimbal 或独立入口。camera_pipeline.py 调用 pipeline_tests 并核对两种模型输出坐标标签、帧数、固定目标位置、实际反馈时间、无新反馈时沿用矩阵及连续漏检时的预测跟踪。camera_filter_replay 保留为测试专用固定系 EKF 数据桥接程序，不是生产入口。

```powershell
.\.venv\Scripts\python -B tests/camera_pipeline.py --work-dir tests/outputs/pipeline-new --filter-bin-dir tests/build/Release
```

## 数值和视频对照

reference/predictor 中的 NumPy 实现仅用于测试。原生回放按配置直接调用两种模型各自的 update_frame，不使用生产模型分派器。verify_predictor_equivalence.py 按完整帧时间轴比较全部结果列、未来四板和 RMSE：两段真实记录及合成轨迹、最近板选择、漏检和角度跨界共 20 组案例。输入观测按接口约定保证正确，不再测试预测器对无效观测的过滤。它使用 reference/recordings/pose_raw_1.csv / pose_raw_2.csv 做直接模型对照，不依赖生产入口或正式输出。

```powershell
.\.venv\Scripts\python -B tests/verify_predictor_equivalence.py --python-root . --bin-dir tests/build/Release --work-dir tests/outputs/equivalence-new
.\.venv\Scripts\python -B tests/verify_armor_video.py --max-source-frames 90
```

视频集成脚本临时修改 Armor.cpp 顶部配置与运行根目录，将整个程序输出隔离到工作目录下的 program-output，明确选择 VIDEO 和 50 ms 预测，并将通信/自瞄开关设为 true，验证视频仍独立于下位机。分别构建两种模型和红/蓝检测，蓝色轨迹夹具交换红色视频的 B/R 通道。验证相机系观测、不生成基座/姿态/控制文件、固定列/RMSE、帧号、FPS和导出视频，保存叠加截图，并核对正式 results 完全未改变。finally 中恢复源码并重建默认 Armor。省略 --max-source-frames 使用完整源视频，--preview 额外测试短预览。

pipeline_tests 同时以录像帧模拟不连接下位机的摄像头输出，覆盖两种 EKF 的相机系预测、LIVE 输出路径及无基座/姿态/控制文件；该测试不打开物理摄像头。

Python 绘图测试使用当前固定系格式，验证完整板角点经 T_BC 逆变换，以及初始化前无状态帧不沿用旧状态。软件结果不代表实物曝光/串口延迟已经测量或机械标定已完成。

简化后的日志与结果：tests/outputs/fallback-removal。串口线程实际收发、OpenCV 实际摄像头和电机闭环仍待实物联调。

## 保留的诊断工具

检测阈值、暗小灯条与背景误检调参见 [detector_parameter_tuning.md](detector_parameter_tuning.md)，包含原生候选扫描、完整视频及样本前后对比。

单板响应与噪声调试见 [single_noise_tuning.md](single_noise_tuning.md)，包含完整视频观测、连续片段对比、30/60 FPS 单板运动与静态抖动验证。

固定噪声调参说明见 [fixed_noise_tuning.md](fixed_noise_tuning.md)。构建 armor_noise_evaluation 后运行 tests.tune_fixed_noise。test_fixed_noise.py 检查固定联合协方差、门控、输入质量字段不改变权重及诊断导出。

检测历史对比运行 tests/compare_detector.py；灯条端点与 PnP 候选分析分别为 tests.plotting.light_refinement 和 tests.plotting.pnp_candidates。这些诊断产物均在 outputs 中。
