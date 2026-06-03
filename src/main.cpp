#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <memory>
#include <algorithm>
#include "input_stream.hpp"
#include "lightbar_detector.hpp"
#include "solver.hpp"

// ... (printHelp 函数保持不变) ...
void printHelp(const char *prog_name)
{
    std::cout << "====================================================\n"
              << "用法: " << prog_name << " [运行模式] [目标颜色]\n\n"
              << "参数说明:\n"
              << "  [运行模式] : 可选 'image' 或 'video' (默认: image)\n"
              << "  [目标颜色] : 可选 'red' 或 'blue' (默认: blue)\n\n"
              << "====================================================\n";
}

int main(int argc, char **argv)
{
    // ==========================================
    // 1. 命令行参数解析 (保持不变)
    // ==========================================
    std::string run_mode = "image";
    EnemyColor target_color = EnemyColor::RED; // 默认红色

    if (argc >= 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help"))
    {
        printHelp(argv[0]);
        return 0;
    }
    if (argc >= 2)
    {
        std::string mode_arg = argv[1];
        std::transform(mode_arg.begin(), mode_arg.end(), mode_arg.begin(), ::tolower);
        if (mode_arg == "video" || mode_arg == "image")
            run_mode = mode_arg;
    }
    if (argc >= 3)
    {
        std::string color_arg = argv[2];
        std::transform(color_arg.begin(), color_arg.end(), color_arg.begin(), ::tolower);
        if (color_arg == "red")
            target_color = EnemyColor::RED;
        else if (color_arg == "blue")
            target_color = EnemyColor::BLUE;
    }

    // ==========================================
    // 2. 动态路径配置 & 初始化输入流 (保持不变)
    // ==========================================
    std::string input_path = (run_mode == "video") ? "./assets/video/video_1.avi" : "./assets/image/image_1.jpg";
    std::string output_path = (run_mode == "video") ? "./results/video/video_1.mp4" : "./results/image/image_1.jpg";
    std::filesystem::create_directories(std::filesystem::path(output_path).parent_path());

    std::unique_ptr<InputStream> stream;
    if (run_mode == "video")
        stream = std::make_unique<VideoFileStream>(input_path);
    else if (run_mode == "image")
        stream = std::make_unique<ImageStream>(input_path);

    cv::Mat original_frame;
    if (!stream->getFrame(original_frame))
    {
        std::cerr << "[错误] 无法读取视频/图片数据，请检查路径！" << std::endl;
        return -1;
    }

    cv::VideoWriter video_writer;
    if (run_mode == "video")
    {
        // 【核心修复】：将 'a','v','c','1' 替换为最兼容的 'm','p','4','v'
        video_writer.open(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30.0, original_frame.size(), true);

        if (!video_writer.isOpened())
        {
            std::cerr << "[致命错误] OpenCV 无法创建视频，请检查是否缺少 ffmpeg 支持，或将后缀改为 .avi！" << std::endl;
        }
        else
        {
            std::cout << "[信息] 视频录制已就绪：" << output_path << std::endl;
        }
    }

    // ==========================================
    // 3. PnP 解算器初始化 (保持不变)
    // ==========================================
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 1286.307, 0, 645.344, 0, 1288.140, 483.616, 0, 0, 1);
    cv::Mat distort_coeffs = (cv::Mat_<double>(1, 5) << -0.4756, 0.2183, 0.00049, -0.00034, 0);
    Solver pnp_solver(camera_matrix, distort_coeffs);

    // ==========================================
    // 4. 【核心升级】：参数持久化与控制台
    // ==========================================
    cv::namedWindow("Debug Panel", cv::WINDOW_AUTOSIZE);

    // 设定“出厂默认值”
    int color_th = 20;
    int gray_th = 80;
    int min_area = 10;
    int min_angle = 55;

    std::string config_file = "./assets/config.yaml";

    // 【新增 A：开局读取配置】
    cv::FileStorage fs_read(config_file, cv::FileStorage::READ);
    if (fs_read.isOpened())
    {
        fs_read["color_th"] >> color_th;
        fs_read["gray_th"] >> gray_th;
        fs_read["min_area"] >> min_area;
        fs_read["min_angle"] >> min_angle;
        fs_read.release();
        std::cout << "[信息] 已成功加载本地配置 config.yaml！" << std::endl;
    }
    else
    {
        std::cout << "[信息] 未找到配置文件，将使用默认参数。" << std::endl;
    }

    // 绑定滑动条
    cv::createTrackbar("Color Diff", "Debug Panel", &color_th, 100);
    cv::createTrackbar("Gray Thresh", "Debug Panel", &gray_th, 255);
    cv::createTrackbar("Min Area", "Debug Panel", &min_area, 200);
    cv::createTrackbar("Min Angle", "Debug Panel", &min_angle, 90);

    // ==========================================
    // 5. 视觉处理主循环
    // ==========================================
    do
    {
        cv::Mat current_frame = original_frame.clone();

        cv::Mat mask = extractColor(current_frame, target_color, color_th, gray_th);
        auto contours = extractContours(mask);
        auto lightBars = filterLightBars(contours, 1.5, (double)min_area);
        auto lightRects = getValidLightRects(lightBars, (float)min_angle);
        auto armors = matchArmors(lightRects);

        cv::imshow("Mask Preview", mask);
        cv::Mat result = drawArmors(current_frame, armors);

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

        // --- E. 输出呈现 ---
        cv::imshow("RoboMaster Vision System - Final", result);

        // 如果是视频模式，不断把每一帧写入文件
        if (run_mode == "video")
        {
            if (video_writer.isOpened())
            {
                video_writer.write(result);
            }
            if (!stream->getFrame(original_frame))
                break; // 视频放完则结束
        }

        // 【核心修复】：加入 S 键保存功能
        char key = (char)cv::waitKey(30);
        if (key == 27)
            break; // 按 ESC 退出

        if (key == 's' || key == 'S')
        {
            // 当你调出了完美的参数，按下 'S' 键手动保存！
            std::filesystem::path out_file(output_path);
            std::string stem = out_file.stem().string();
            std::string ext = out_file.extension().string();
            std::string dir = out_file.parent_path().string();

            cv::imwrite(dir + "/" + stem + "_mask" + ext, mask);
            cv::imwrite(output_path, result);
            std::cout << "[成功] 已将完美调参结果保存至：" << output_path << std::endl;
        }

    } while (true);

    // ==========================================
    // 6. 【新增 B：剧终保存配置】
    // ==========================================
    cv::FileStorage fs_write(config_file, cv::FileStorage::WRITE);
    if (fs_write.isOpened())
    {
        fs_write << "color_th" << color_th;
        fs_write << "gray_th" << gray_th;
        fs_write << "min_area" << min_area;
        fs_write << "min_angle" << min_angle;
        fs_write.release();
        std::cout << "[信息] 参数已自动固化保存至 " << config_file << "，下次启动将自动生效！" << std::endl;
    }

    if (video_writer.isOpened())
        video_writer.release();
    cv::destroyAllWindows();
    return 0;
}