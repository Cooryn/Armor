#include <iostream>
#include <opencv2/opencv.hpp>
// 注意：如果后面 CMake 配置了 include 目录，这里直接写文件名即可；
// 如果没配置，可能需要写成 #include "../include/input_stream.hpp"
#include "input_stream.hpp"
#include "lightbar_detector.hpp"

int main()
{
    // ==========================================
    // 1. 运行与战局配置
    // ==========================================
    // 模式可填: "image" 或 "video"
    std::string run_mode = "video";

    // 输入和输出路径设置 (对应你创建的目录结构)
    std::string input_path = (run_mode == "video") ? "./assets/video_1_raw.mp4" : "./assets/image_1_raw.png";
    std::string output_path = (run_mode == "video") ? "./results/video_1_raw.mp4" : "./results/image_1_raw.png";

    // 战局配置：设置你需要打击的敌方颜色
    // 由于任务视频中大部分是蓝方机器人，我们这里默认设为 BLUE
    EnemyColor target_color = EnemyColor::BLUE;

    // 利用多态初始化输入流
    std::unique_ptr<InputStream> stream;
    if (run_mode == "video")
    {
        stream = std::make_unique<VideoFileStream>(input_path);
    }
    else if (run_mode == "image")
    {
        stream = std::make_unique<ImageStream>(input_path);
    }
    else
    {
        std::cerr << "[Error] 未知的运行模式！" << std::endl;
        return -1;
    }

    // ==========================================
    // 2. 录制器初始化 (针对视频流)
    // ==========================================
    cv::VideoWriter video_writer;
    cv::Mat frame;

    // 预读第一帧，获取分辨率以初始化 VideoWriter
    if (!stream->getFrame(frame))
    {
        std::cerr << "[Error] 无法读取第一帧，请检查 assets 路径！" << std::endl;
        return -1;
    }

    if (run_mode == "video")
    {
        int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        double fps = 30.0;
        video_writer.open(output_path, codec, fps, frame.size(), true);
        if (!video_writer.isOpened())
        {
            std::cerr << "[Warning] 视频写入器初始化失败！" << std::endl;
        }
        else
        {
            std::cout << "[Info] 正在处理视频，结果将保存至: " << output_path << std::endl;
        }
    }

    // ==========================================
    // 3. 核心流水线 (Pipeline)
    // ==========================================
    int frame_count = 0;

    do
    {
        if (frame.empty())
            continue;

        // --- 阶段 A：图像预处理 ---
        cv::Mat processed_frame;
        // 降低画面亮度，凸显发光的灯条
        cv::addWeighted(frame, 0.6, frame, 0.0, 0.0, processed_frame);

        // 提取指定颜色（红/蓝）区域掩码
        cv::Mat mask = extractColor(processed_frame, target_color);

        // --- 阶段 B：灯条提取 ---
        auto contours = extractContours(mask);
        auto lightBars = filterLightBars(contours, 2.0, 30.0);

        // --- 阶段 C：装甲板匹配 ---
        auto lightRects = getLightBarRects(lightBars);
        auto armors = matchArmors(lightRects);

        // --- 阶段 D：可视化渲染 ---
        // 注意：这里是在原图 frame 上绘制，而不是降低亮度的 processed_frame
        cv::Mat result = drawArmors(frame, armors);

        // ==========================================
        // [预留位置]：Task 2 PnP 位姿解算模块将插入在这里
        // 如果 armors 不为空，我们将提取 armors[i].vertices 送入 Solver
        // ==========================================

        // --- 结果输出与保存 ---
        cv::imshow("RoboMaster Vision System", result);

        if (run_mode == "video" && video_writer.isOpened())
        {
            video_writer.write(result);
        }
        else if (run_mode == "image")
        {
            cv::imwrite(output_path, result);
            std::cout << "[Info] 图片已保存至: " << output_path << std::endl;
        }

        // --- 运行控制 ---
        frame_count++;
        if (frame_count % 30 == 0)
            std::cout << "[Info] 已处理帧数: " << frame_count << std::endl;

        // 图片模式无限等待，视频模式按 30ms 延时播放
        int delay = (run_mode == "image") ? 0 : 30;
        if (cv::waitKey(delay) == 27)
        { // ESC 键退出
            std::cout << "[Info] 收到退出指令。" << std::endl;
            break;
        }

    } while (stream->getFrame(frame));

    // 清理现场
    if (video_writer.isOpened())
        video_writer.release();
    cv::destroyAllWindows();
    std::cout << "[Info] 运行结束。" << std::endl;

    return 0;
}