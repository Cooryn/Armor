#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <memory>
#include "input_stream.hpp"
#include "lightbar_detector.hpp"
#include "solver.hpp"

int main()
{
    std::filesystem::create_directories("./results");

    // ==========================================
    // 1. 运行参数配置
    // ==========================================
    std::string run_mode = "video";
    std::string input_path = (run_mode == "video") ? "./assets/video/video_1.avi" : "./assets/image/image_1.jpg";
    std::string output_path = (run_mode == "video") ? "./results/video/video_1.avi" : "./results/image/image_1.jpg";

    // 设置要打击的敌方颜色
    EnemyColor target_color = EnemyColor::RED;

    // ==========================================
    // 2. 初始化输入输出流
    // ==========================================
    std::unique_ptr<InputStream> stream;
    if (run_mode == "video")
        stream = std::make_unique<VideoFileStream>(input_path);
    else if (run_mode == "image")
        stream = std::make_unique<ImageStream>(input_path);
    else
        return -1;

    cv::VideoWriter video_writer;
    cv::Mat frame;

    if (!stream->getFrame(frame))
    {
        std::cerr << "[错误] 无法读取视频/图片，请检查路径: " << input_path << std::endl;
        return -1;
    }

    if (run_mode == "video")
    {
        video_writer.open(output_path, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), 30.0, frame.size(), true);
    }

    // ==========================================
    // 3. PnP 解算器初始化 (使用 video_1 的内参)
    // ==========================================
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 1286.307, 0, 645.344, 0, 1288.140, 483.616, 0, 0, 1);
    cv::Mat distort_coeffs = (cv::Mat_<double>(1, 5) << -0.4756, 0.2183, 0.00049, -0.00034, 0);
    Solver pnp_solver(camera_matrix, distort_coeffs);

    int frame_count = 0;
    std::cout << "[信息] 视觉流水线启动，按 ESC 退出..." << std::endl;

    // ==========================================
    // 4. 视觉处理主循环
    // ==========================================
    do
    {
        if (frame.empty())
            continue;

        // --- A. 图像预处理 ---
        cv::Mat processed_frame;
        cv::addWeighted(frame, 0.6, frame, 0.0, 0.0, processed_frame);

        // --- B. 2D 目标检测 ---
        cv::Mat mask = extractColor(processed_frame, target_color);
        auto contours = extractContours(mask);
        auto lightBars = filterLightBars(contours, 2.0, 30.0);
        auto lightRects = getLightBarRects(lightBars);
        auto armors = matchArmors(lightRects);

        // --- C. 可视化渲染基础信息 ---
        cv::Mat result = drawArmors(frame, armors);

        // --- D. 3D 位姿解算 (PnP) ---
        for (auto &armor : armors)
        {
            if (pnp_solver.solve(armor))
            {
                // 画 3D 轴
                pnp_solver.drawAxis(result, armor);
                // 打印 Z 轴距离和 Yaw 角
                cv::putText(result, cv::format("Z: %.2fm Yaw: %.1f deg", armor.tvec.at<double>(2), armor.yaw),
                            cv::Point(armor.vertices[0].x - 10, armor.vertices[0].y - 15),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
            }
        }

        // --- E. 输出呈现与保存 ---
        cv::imshow("RoboMaster Vision System", result);

        if (run_mode == "video" && video_writer.isOpened())
            video_writer.write(result);
        else if (run_mode == "image")
            cv::imwrite(output_path, result);

        frame_count++;
        if (frame_count % 30 == 0)
            std::cout << "[信息] 已处理帧数: " << frame_count << std::endl;

        // 正常播放速度：30ms 延时
        if (cv::waitKey((run_mode == "image") ? 0 : 30) == 27)
            break;

    } while (stream->getFrame(frame));

    // ==========================================
    // 5. 资源清理
    // ==========================================
    if (video_writer.isOpened())
        video_writer.release();
    cv::destroyAllWindows();
    std::cout << "[信息] 处理完成！结果保存在 " << output_path << std::endl;

    return 0;
}