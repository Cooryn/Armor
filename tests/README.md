# 测试与诊断目录

第 4 项观测加权对比：`.\.venv\Scripts\python -B -m tests.plotting.adaptive_observation`。
使用相同原始 CSV，对比第 2 项完成后的基线，产物位于 `outputs/adaptive_observation/`。
`weighting_summary.csv` 补充全部接受观测的距离残差、接受数量、权重倍率和停车尾段指标。
停车指标以第 320 帧为检查起点，不把它当作精确实测停车时刻。
`test_observation_noise.py` 检查权重上界、角度符号、旧输入兼容、离群门控、
不同权重的联合更新、顺序不变性及诊断字段导出。

PnP 第 2 项对比：`.\.venv\Scripts\python -B -m tests.plotting.pnp_candidates`。
基线为第 1 项完成后的数据，产物位于 `outputs/pnp_candidates/`。
`test_detector.cpp` 同时检查畸变下的合成姿态、yaw 符号、匹配顺序不变性、
歧义关联禁用历史以及漏检/时间间隔后的历史清理。

所有测试脚本、检测前后对照、历史分析图片、临时试验和测试构建集中在这里。
正式程序输出仍位于仓库的 `data/`、`results/`。

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
