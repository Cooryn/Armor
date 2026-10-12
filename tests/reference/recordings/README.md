# 参考观测

这些历史观测 CSV 用作数值对照和噪声调参输入，正式运行不会覆盖它们。

- `pose_raw_1.csv`、`pose_raw_2.csv`：三种模型的 C++ / NumPy 数值对照及离线参考程序。
- `camera_pose_raw_1.csv`、`camera_pose_raw_2.csv`：历史相机系观测；固定噪声调参使用第一份。

正式运行生成的观测和预测统一保存在项目的 `results/` 下。
