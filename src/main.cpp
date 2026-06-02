#include <iostream>
#include <opencv2/opencv.hpp>
#include "input_stream.hpp"
#include "lightbar_detector.hpp"
#include "solver.hpp"
#include <filesystem>

int main()
{
    // 确保 results 文件夹存在
    std::filesystem::create_directories("./results");

    // 运行配置
    std::string run_mode = "video"; // "image" 或 "video"
    std::string input_path = (run_mode == "video") ? "./assets/video_1_raw.mp4" : "./assets/image_1_raw.png";
    std::string output_path = (run_mode == "video") ? "./results/video_1.mp4" : "./results/image_1.png";
    EnemyColor target_color = EnemyColor::BLUE;

    // 初始化流
    std::unique_ptr<InputStream> stream;
    if (run_mode == "video")
        stream = std::make_unique<VideoFileStream>(input_path);
    else if (run_mode == "image")
        stream = std::make_unique<ImageStream>(input_path);

    cv::VideoWriter video_writer;
    cv::Mat frame;

    if (!stream->getFrame(frame))
        return -1;

    if (run_mode == "video")
    {
        video_writer.open(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30.0, frame.size(), true);
    }

    // 初始化 PnP 解算器 (video_1 参数)
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 1286.307, 0, 645.344, 0, 1288.140, 483.616, 0, 0, 1);
    cv::Mat distort_coeffs = (cv::Mat_<double>(1, 5) << -0.4756, 0.2183, 0.00049, -0.00034, 0);
    Solver pnp_solver(camera_matrix, distort_coeffs);

    int frame_count = 0;
    do
    {
        if (frame.empty())
            continue;

        cv::Mat processed_frame;
        cv::addWeighted(frame, 0.6, frame, 0.0, 0.0, processed_frame);

        cv::Mat mask = extractColor(processed_frame, target_color);
        auto contours = extractContours(mask);
        auto lightBars = filterLightBars(contours, 2.0, 30.0);
        auto lightRects = getLightBarRects(lightBars);
        auto armors = matchArmors(lightRects);

        cv::Mat result = drawArmors(frame, armors);

        // PnP 解算与可视化
        for (auto &armor : armors)
        {
            if (pnp_solver.solve(armor))
            {
                pnp_solver.drawAxis(result, armor);
                std::string pose_text = cv::format("Z: %.2fm Yaw: %.1f", armor.tvec.at<double>(2), armor.yaw);
                cv::putText(result, pose_text, cv::Point(armor.vertices[0].x, armor.vertices[0].y - 10),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);
            }
        }

        cv::imshow("RoboMaster Vision", result);
        if (run_mode == "video" && video_writer.isOpened())
            video_writer.write(result);
        else if (run_mode == "image")
            cv::imwrite(output_path, result);

        if (cv::waitKey((run_mode == "image") ? 0 : 30) == 27)
            break;

    } while (stream->getFrame(frame));

    if (video_writer.isOpened())
        video_writer.release();
    return 0;
}