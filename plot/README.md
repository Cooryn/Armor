# 绘图脚本说明

这里仅保留 CSV 分析曲线。叠加视频由 C++ `output` 直接保存为 AVI，不再通过 Python 重复生成。

| 命令 | 输入 | 输出图片 |
| --- | --- | --- |
| `plot.raw` | `pose_raw_*.csv` | `pose_raw_curve_*.png`、`raw_observation_yaw_*.png` |
| `plot.basic` | `prediction_result_*.csv` | `prediction_error_curve_*.png` |
| `plot.armor` | `armor_prediction_result_*.csv` 和对应观测 | `top_down_trajectory_*.png`、`body_yaw_estimate_*.png`、`armor_yaw_comparison_*.png`、`armor_prediction_error_curve_*.png` |

绘图分别位于 `raw.py`、`basic.py` 和 `armor.py`，公共参数由 `_cli.py` 解析。

从仓库根目录执行：

```powershell
.\.venv\Scripts\python -m plot.raw --suffix 1 --model armor
.\.venv\Scripts\python -m plot.armor --suffix 1
.\.venv\Scripts\python -m plot.basic --suffix live
.\.venv\Scripts\python -m plot.basic --suffix live --camera-mode mcu
```

公共参数：

- `--suffix`：CSV 后缀；armor 默认 1，其余默认 2，摄像头使用 live。
- `--source video|camera`：默认 live 选择 camera，其余选择 video。
- `--model single|armor`：basic/raw 默认 single，armor 默认 armor。
- `--camera-mode offline|mcu`：摄像头模式，默认 offline。
- `--input-name`：视频文件主干，默认 video_<suffix>。
- `--results-dir`：直接指定结果目录。
- `--data-dir`：指定观测目录，默认与结果目录相同。
- `--output-dir`：指定图片目录，默认结果目录下的 plots。

默认结果目录是 `results/video/<模型>/<视频名>/` 或 `results/camera/<模型>/offline|mcu/`。
同名图片会覆盖。

raw 始终读取相机系原始观测。armor 根据预测 CSV 的坐标系选择 pose_raw 或 pose_base；basic 只读取预测 CSV，不依赖观测文件。
误差图反映更新前预测与观测的偏差，不等同于真实定位误差。当前没有控制目标曲线。
