# 单板噪声调试（2026-10-10）

## 2026-10-12：恢复调整后的参数

按用户最新要求，单板默认噪声恢复为此前调参选出的 candidate `c083`：`q_pos=60`、`q_yaw=75`、`R=diag(0.000036,0.000025,0.01,0.0576)`；独立 NumPy 参考同步恢复，保留 50/100 ms 运动和静态误差上界检查。

## 2026-10-12：恢复远程仓库参数（历史）

此前按用户要求，单板默认噪声曾恢复为 `origin/main` 中的 `q_pos=100`、`q_yaw=100`、`R=diag(0.000025,0.000025,0.01,0.0576)`；独立 NumPy 参考同步恢复。50/100 ms 运动和静态误差上界检查保留，移除仅用于要求上次调参优于旧参数的比较断言。此段为历史记录，当前参数以上一节为准。

## 2026-10-12：平衡静止和移动时的预测抖动（历史）

用户反馈静止和移动时都有抖动。此前参数主要解决跟随滞后，位置/朝向过程噪声均为 100；速度估计会响应观测波动，而当前入口的 100 ms 外推会放大速度波动。此次只调整单板固定 Q/R，并同步 NumPy 参考；匀速模型、初始化、最近板选择、PnP、四板参数和 100 ms 提前量不变。

| 参数 | 调整前 | 本次调参值 |
| --- | ---: | ---: |
| `q_pos` | 100 | 60 |
| `q_yaw` | 100 | 75 |
| 方位角观测方差（rad²） | 0.000025 | 0.000036 |
| 俯仰角观测方差（rad²） | 0.000025 | 0.000025 |
| 距离观测方差（m²） | 0.01 | 0.01 |
| 板朝向观测方差（rad²） | 0.0576 | 0.0576 |

使用生产 C++ `predictor_single::update_frame` 比较 160 组候选及旧参数。训练输入包括两段此前实机帧率测试中的稳定目标观测、正式目录中已有的运动记录，以及 30/60 FPS 的静止、慢速、快速、近距离、深度运动和漏检模拟。三个模拟噪声种子分别用于排序、候选保护筛选和最终独立噪声复核；第三组噪声未参与选择，但运动场景相同。真实记录缺帧时间戳仅在离线诊断中插值，没有进入生产代码。

先按静态预测误差、朝向误差、运动预测误差及真实静止记录的帧间变化综合排序，再检查运动响应，最终部署 candidate `c083`。没有将 q 降到 10–30，这些候选虽然更平滑，但运动跟随误差明显增大。

| 同一份摄像头观测回放，100 ms 预测 | 静止记录 1 | 静止记录 2 | 运动记录 |
| --- | ---: | ---: | ---: |
| 未来位置帧间变化 RMS | -25.51% | -24.53% | -15.25% |
| 速度增量 RMS | -27.78% | -26.52% | -20.12% |
| 角速度增量 RMS | -13.10% | -13.17% | -12.18% |

真实记录没有外部位置/朝向真值。帧间变化和速度增量包含真实运动，运动记录的下降不能单独当作精度提升；此处是原始观测回放，没有新采集实机数据。

最终独立噪声复核：静止目标的 100 ms 未来位置真值 RMSE 在 30/60 FPS 下由 35.01/24.08 mm 降到 31.09/20.84 mm，下降 11.20%/13.43%。快速运动位置预测 RMSE 由 73.71/72.69 mm 变为 82.45/83.41 mm；所有移动/漏检场景中最坏位置预测 RMSE 增幅为 14.76%，最坏朝向预测 RMSE 增幅为 9.09%。这是平滑与响应的取舍，不宣称所有误差改善。快速运动横向最佳时间平移滞后由 1/2 ms 变为 3/3 ms；这只是该模拟轨迹的相位指标，不是实机端到端延迟。

复现及产物：

```powershell
cmake --build build --config Release --target single_noise_evaluation --parallel 1 -- /nodeReuse:false
.\.venv\Scripts\python.exe -B -m tests.tune_single_stability
```

`tests/outputs/single-stability-20261012/` 保存输入快照、候选参数、ranking.csv、comparison.csv、validation.csv、holdout.csv、全部状态及 comparison.png。默认命令需要上述本地实机记录。`test_predictors.cpp` 增加 100 ms 静态抖动对照和 50/100 ms 运动响应保护。以下保留历史记录。

验证完成：`Armor.exe` 已重新构建；`predictor_api_regressions`、`video_predictor_regressions`、`armor_pipeline_regressions` 均通过。原生/NumPy 完整帧时间轴对照 20 组全部通过。新增评估器最后重建后的指标与 holdout.csv 一致。对比图见 [comparison.png](outputs/single-stability-20261012/comparison.png)。

## 2026-10-12：real.mp4 检查，未应用新参数

输入为用户提供的 `D:/DOWNLOADS/real.mp4`，手机拍摄且无对应内参和畸变标定。视频 720×1280、245 帧，容器 FPS 29.878，约 8.2 秒。没有使用现有摄像头或视频标定冒充手机标定，也没有据此修改正式单板 Q/R。

使用生产 C++ 红色灯条检测器逐帧处理：245 帧均有候选，142 帧为单候选，103 帧为双候选。人工检查首帧，两个框分别落在真实装甲板和桌面灯条反光上；其余双候选帧未逐一人工标注。该统计是检测结果，不是 PnP 成功率或跟踪准确率。

结果保存于 `tests/outputs/single-real/detections.csv` 和 `detection_preview.mp4`。评估工具新增不需要标定的 pixels 模式，并修复其对已删除 valid_observation 的调用：

```powershell
cmake --build tests/build --config Release --target single_video_evaluation
tests/build/Release/single_video_evaluation.exe D:/DOWNLOADS/real.mp4 pixels tests/outputs/single-real/detections.csv
```

进一步确定米制 EKF 噪声需要对应拍摄设置的手机标定及确认板尺寸，或实际摄像头生成的 pose_raw CSV。手机反光候选问题也应与噪声调参区分。以下为此前已部署参数的历史记录。

用户报告的是实时摄像头下、始终只有同一块装甲板的跟随滞后。本次只修改单板 EKF 的固定噪声，没有加入换板重置、修改关联规则或修改多板模型。入口的相机/视频、预测器选择保持原配置。

| 参数 | 原值 | 新值 |
| --- | ---: | ---: |
| 加速度过程方差 `process_noise` | 0.01 | 100 |
| 方位、俯仰观测方差（rad²） | 0.0016 | 0.000025 |
| 距离观测方差（m²） | 0.16 | 0.01 |

新观测标准差分别为 0.005 rad（约 0.286°）和 0.10 m。过程方差仍使用原有 `dt⁴/4, dt³/2, dt²` 离散匀速模型，未改变模型或时间戳算法。原来的低过程噪声配合高观测噪声，会在协方差收敛后显著压低对加减速的响应。参数集中在 `include/predictor_single.hpp`，独立 NumPy 参考同步更新。

## 视频证据及限制

原生检测器/PnP 完整处理 `video_1.avi` 的 815 帧和 `video_2.avi` 的 351 帧，分别有 710 和 351 帧包含有效观测。先保存旧参数基线，再用新参数重放；C++ 状态与 NumPy 对照通过（1e-9 容差）。

这两段素材都是旋转多板场景，不等同于用户的实时单板场景。整段最近板观测存在不连续，不能把提高响应后跨板外推误差增大解释为单板调参失败，也不能拿贴近当前观测当成未来预测精度。诊断因此把观测缺失、帧间方位跳变超过 0.035 rad、板朝向跳变超过 0.6 rad 的位置切为片段，并在每个片段起点重新初始化**两组**滤波器。这只是离线筛选规则，没有进入生产代码，也不保证真实板身份。

大多数运动片段只有 5–9 帧，视频 2 末尾还有约 1.3 秒连续片段。因此视频主要检查初始化、短时跟随及停止后的收敛，不能充分验证长时间运行的过程噪声。没有采集或声称完成实时摄像头验证。

50 ms 未来位置用相邻两帧观测线性插值；只统计当前时刻到插值终点都在同一片段的样本，不跨漏检。以下是真实观测残差，**不是位置真值误差**：

| 指标 | 视频 1 原→新 | 视频 2 原→新 |
| --- | ---: | ---: |
| 当前位置 RMSE | 0.02116→0.01490 m | 0.03521→0.02205 m |
| 片段内 50 ms 方位残差 RMSE | 1.621→1.521° | 1.766→1.475° |
| 片段内 50 ms 位置残差 RMSE | 0.08982→0.09179 m | 0.12026→0.11475 m |

视频 1 的未来三维位置残差略有增加，结果不是所有指标都改善。逐项数据见 `tests/outputs/single-noise/summary.json`；对比视频为 `comparison_1.mp4`、`comparison_2.mp4`，白色为观测、绿色为当前滤波位置、橙色为 50 ms 预测。视频上标明片段重启，不能把它当作整段生产运行效果。

## 同一块板的长时间响应验证

另以固定随机种子生成 12 秒连续单板运动：横向叠加 0.65/1.1 Hz 正弦，竖向 0.4 Hz，距离 0.35 Hz；独立检查静止板。观测标准差为方位 0.003 rad、俯仰 0.001 rad、距离 0.035 m，参考视频连续片段二阶差分量级，但不是相机噪声标定。剔除前 1 秒初始化过程。比较 36 组候选，以运动预测误差及静态抖动共同判断；选择较稳健的 0.005 rad / 0.10 m，而非只取仿真最低分。

| 仿真指标 | 30 FPS 原→新 | 60 FPS 原→新 |
| --- | ---: | ---: |
| 横向最佳时间平移滞后 | 305→1 ms | 313→2 ms |
| 运动 50 ms 真值 RMSE | 0.3257→0.0360 m | 0.3284→0.0369 m |
| 静止位置真值 RMSE | 0.00536→0.02073 m | 0.00470→0.01712 m |

时间平移按 1 ms 网格搜索，是该轨迹的滤波相位指标，**不是端到端摄像头延迟**。新参数的代价是静态抖动增加。真实光照、距离、PnP 标定以及摄像头排队延迟仍需实时单板录像确认。这些场景参与了调参，不能当作独立泛化测试。

## 复现

在已配置 OpenCV DLL PATH 的 PowerShell 中，从仓库根目录执行：

```powershell
cmake --build tests/build --config Release --target single_video_evaluation predictor_tests video_predictor_tests
New-Item -ItemType Directory -Force tests/outputs/single-noise
.\tests\build\Release\single_video_evaluation.exe assets/video/video_1.avi 1 tests/outputs/single-noise/observations_1.csv
.\tests\build\Release\single_video_evaluation.exe assets/video/video_2.avi 2 tests/outputs/single-noise/observations_2.csv
.\.venv\Scripts\python.exe -B -m tests.tune_single_noise --segments
.\.venv\Scripts\python.exe -B -m tests.evaluate_single_response
.\.venv\Scripts\python.exe -B -m tests.render_single_noise
ctest --test-dir tests/build -C Release -R 'predictor_api_regressions|video_predictor_regressions' --output-on-failure
```

视频扫描前 70% 用于候选排序，后 30% 单独导出；它们也被人工查看过，不是盲测集。原生回归新增 30/60 FPS 连续加减速与静止抖动上界。另运行 `verify_predictor_equivalence.py`，30 组原生/NumPy 对照全部通过；正式 `Armor.exe` 已重新构建。实际运行新参数时需选择 `PREDICTOR_SINGLE_PLATE`；当前 `Armor.cpp` 仍为原有 `VIDEO / PREDICTOR_ARMOR`。
