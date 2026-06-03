#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <memory>
#include <algorithm> // 用于字符串转小写
#include "input_stream.hpp"
#include "lightbar_detector.hpp"
#include "solver.hpp"

// 打印使用说明
void printHelp(const char *prog_name)
{
    std::cout << "====================================================\n"
              << "用法: " << prog_name << " [运行模式] [目标颜色]\n\n"
              << "参数说明:\n"
              << "  [运行模式] : 可选 'image' 或 'video' (默认: image)\n"
              << "  [目标颜色] : 可选 'red' 或 'blue' (默认: blue)\n\n"
              << "示例:\n"
              << "  " << prog_name << " image red   (处理图片，提取红方装甲板)\n"
              << "  " << prog_name << " video blue  (处理视频，提取蓝方装甲板)\n"
              << "====================================================\n";
}

int main(int argc, char **argv)
{
    // ==========================================
    // 1. 命令行参数解析
    // ==========================================
    // 默认参数
    std::string run_mode = "image";
    EnemyColor target_color = EnemyColor::BLUE;

    // 如果用户输入了 "-h" 或 "--help"，打印帮助信息并退出
    if (argc >= 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help"))
    {
        printHelp(argv[0]);
        return 0;
    }

    // 解析第 1 个参数：运行模式
    if (argc >= 2)
    {
        std::string mode_arg = argv[1];
        std::transform(mode_arg.begin(), mode_arg.end(), mode_arg.begin(), ::tolower); // 转小写
        if (mode_arg == "video" || mode_arg == "image")
        {
            run_mode = mode_arg;
        }
        else
        {
            std::cerr << "[错误] 未知的运行模式: " << mode_arg << "\n";
            printHelp(argv[0]);
            return -1;
        }
    }

    // 解析第 2 个参数：目标颜色
    if (argc >= 3)
    {
        std::string color_arg = argv[2];
        std::transform(color_arg.begin(), color_arg.end(), color_arg.begin(), ::tolower); // 转小写
        if (color_arg == "red")
        {
            target_color = EnemyColor::RED;
        }
        else if (color_arg == "blue")
        {
            target_color = EnemyColor::BLUE;
        }
        else
        {
            std::cerr << "[错误] 未知的目标颜色: " << color_arg << "\n";
            printHelp(argv[0]);
            return -1;
        }
    }

    std::cout << "[信息] 当前配置 -> 模式: " << run_mode
              << " | 打击颜色: " << (target_color == EnemyColor::RED ? "RED" : "BLUE") << std::endl;

    // ==========================================
    // 2. 动态路径配置
    // ==========================================
    std::string input_path = (run_mode == "video") ? "./assets/video/video_1.avi" : "./assets/image/image_1.jpg";
    std::string output_path = (run_mode == "video") ? "./results/video/video_1.mp4" : "./results/image/image_1.jpg";

    // 自动创建输出文件夹
    std::filesystem::path out_dir = std::filesystem::path(output_path).parent_path();
    std::filesystem::create_directories(out_dir);

    // ==========================================
    // 3. 初始化输入输出流
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
    // 4. PnP 解算器初始化
    // ==========================================
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 1286.307, 0, 645.344, 0, 1288.140, 483.616, 0, 0, 1);
    cv::Mat distort_coeffs = (cv::Mat_<double>(1, 5) << -0.4756, 0.2183, 0.00049, -0.00034, 0);
    Solver pnp_solver(camera_matrix, distort_coeffs);

    int frame_count = 0;
    std::cout << "[信息] 视觉流水线启动..." << std::endl;

    // ==========================================
    // 5. 视觉处理主循环
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
        auto lightBars = filterLightBars(contours, 1.5, 10.0); // 宽松参数
        auto lightRects = getValidLightRects(lightBars);
        auto armors = matchArmors(lightRects);

        // --- C. 可视化渲染基础信息 ---
        cv::Mat result = drawArmors(frame, armors);

        // --- D. 3D 位姿解算 (PnP) ---
        for (auto &armor : armors)
        {
            if (pnp_solver.solve(armor))
            {
                pnp_solver.drawAxis(result, armor);
                cv::putText(result, cv::format("Z: %.2fm Yaw: %.1f deg", armor.tvec.at<double>(2), armor.yaw),
                            cv::Point(armor.vertices[0].x - 10, armor.vertices[0].y - 15),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
            }
        }

        // --- E. 输出呈现与保存 ---
        cv::imshow("RoboMaster Vision System - Final", result);

        if (run_mode == "video" && video_writer.isOpened())
        {
            video_writer.write(result);
        }
        else if (run_mode == "image")
        {
            std::filesystem::path out_file(output_path);
            std::string stem = out_file.stem().string();
            std::string ext = out_file.extension().string();
            std::string dir = out_file.parent_path().string();

            // 生成中间过程图
            cv::Mat contours_img = drawAllContours(frame, contours);
            cv::imwrite(dir + "/" + stem + "_mask" + ext, mask);
            cv::imwrite(dir + "/" + stem + "_contours" + ext, contours_img);
            cv::imwrite(output_path, result);
            std::cout << "[信息] 已保存过程图: " << stem << "_mask" << ext << " 和 " << stem << "_contours" << ext << std::endl;
        }

        frame_count++;

        if (cv::waitKey((run_mode == "image") ? 0 : 30) == 27)
            break;

    } while (stream->getFrame(frame));

    if (video_writer.isOpened())
        video_writer.release();
    cv::destroyAllWindows();
    std::cout << "[信息] 处理完成！" << std::endl;

    return 0;
}