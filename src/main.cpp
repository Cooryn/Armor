#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <memory>
#include <algorithm>
#include <fstream>
#include "input_stream.hpp"
#include "lightbar_detector.hpp"
#include "solver.hpp"

void printHelp(const char *prog_name)
{
    std::cout << "====================================================\n"
              << "用法: " << prog_name << " [运行模式] [目标颜色]\n\n"
              << "参数说明:\n"
              << "  [运行模式] : 可选 'image' 或 'video' (默认: image)\n"
              << "  [目标颜色] : 可选 'red' 或 'blue' (默认: red)\n\n"
              << "====================================================\n";
}

int main(int argc, char **argv)
{
    // ==========================================
    // 1. 命令行参数解析
    // ==========================================
    std::string run_mode = "image";
    EnemyColor target_color = EnemyColor::RED;

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
    // 2. 动态路径配置 (严格适配考核标准)
    // ==========================================
    std::string input_path = (run_mode == "video") ? "./assets/video/video_1.avi" : "./assets/image/image_1.jpg";

    // 提取文件名 (如 "image_1" 或 "video_1")
    std::string stem = std::filesystem::path(input_path).stem().string();
    std::string out_dir = "./results/";
    std::filesystem::create_directories(out_dir);

    // 强制按考核要求设置后缀：图片必须是 .png，视频是 .mp4
    std::string ext = (run_mode == "video") ? ".mp4" : ".png";

    // 定义双路输出路径
    std::string raw_output_path = out_dir + stem + "_raw" + ext; // 对应档次1、2
    std::string final_output_path = out_dir + stem + ext;        // 对应档次3、4

    std::unique_ptr<InputStream> stream;
    if (run_mode == "video")
        stream = std::make_unique<VideoFileStream>(input_path);
    else if (run_mode == "image")
        stream = std::make_unique<ImageStream>(input_path);

    cv::Mat original_frame;
    if (!stream->getFrame(original_frame))
    {
        std::cerr << "[错误] 无法读取数据，请检查路径！" << std::endl;
        return -1;
    }

    // ==========================================
    // 3. 双路视频录制器初始化
    // ==========================================
    cv::VideoWriter writer_raw, writer_final;
    if (run_mode == "video")
    {
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        writer_raw.open(raw_output_path, fourcc, 30.0, original_frame.size(), true);
        writer_final.open(final_output_path, fourcc, 30.0, original_frame.size(), true);

        if (!writer_raw.isOpened() || !writer_final.isOpened())
        {
            std::cerr << "[致命错误] 无法创建视频输出流！" << std::endl;
        }
        else
        {
            std::cout << "[信息] 双路视频录制已就绪：\n  -> " << raw_output_path << "\n  -> " << final_output_path << std::endl;
        }
    }

    // ==========================================
    // 4. PnP 解算器初始化
    // ==========================================
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 1286.307, 0, 645.344, 0, 1288.140, 483.616, 0, 0, 1);
    cv::Mat distort_coeffs = (cv::Mat_<double>(1, 5) << -0.4756, 0.2183, 0.00049, -0.00034, 0);
    Solver pnp_solver(camera_matrix, distort_coeffs);

    // ==========================================
    // 5. 参数持久化与控制台
    // ==========================================
    cv::namedWindow("Debug Panel", cv::WINDOW_AUTOSIZE);
    int color_th = 20, gray_th = 80, min_area = 10, min_angle = 55;
    std::string config_file = "./assets/config.yaml";

    cv::FileStorage fs_read(config_file, cv::FileStorage::READ);
    if (fs_read.isOpened())
    {
        fs_read["color_th"] >> color_th;
        fs_read["gray_th"] >> gray_th;
        fs_read["min_area"] >> min_area;
        fs_read["min_angle"] >> min_angle;
        fs_read.release();
    }

    cv::createTrackbar("Color Diff", "Debug Panel", &color_th, 100);
    cv::createTrackbar("Gray Thresh", "Debug Panel", &gray_th, 255);
    cv::createTrackbar("Min Area", "Debug Panel", &min_area, 200);
    cv::createTrackbar("Min Angle", "Debug Panel", &min_angle, 90);

    // ==========================================
    // (新增) 交付 5：数据日志记录初始化
    // ==========================================
    std::string csv_path = "./results/pose_data.csv";
    std::ofstream csv_file(csv_path);
    if (csv_file.is_open())
    {
        csv_file << "Frame,X,Y,Z,Yaw\n";
    }
    int frame_count = 0; // 记录当前是第几帧

    // ==========================================
    // 6. 视觉处理主循环
    // ==========================================
    do
    {
        cv::Mat current_frame = original_frame.clone();

        // 提取与检测
        cv::Mat mask = extractColor(current_frame, target_color, color_th, gray_th);
        auto contours = extractContours(mask);
        auto lightBars = filterLightBars(contours, 1.5, (double)min_area);
        auto lightRects = getValidLightRects(lightBars, (float)min_angle);
        auto armors = matchArmors(lightRects);

        // --- 核心修改：双路渲染 ---
        // 渲染路线 1：Raw (仅含 2D 装甲板检测，用于交付档次 1 / 2)
        cv::Mat raw_result = drawArmors(current_frame, armors);

        // 渲染路线 2：Final (在 Raw 的基础上克隆，追加 3D 姿态文本，用于交付档次 3 / 4)
        cv::Mat final_result = raw_result.clone();

        int text_y_offset = 30;

        for (size_t i = 0; i < armors.size(); i++)
        {
            auto &armor = armors[i];
            if (pnp_solver.solve(armor))
            {
                double tx = armor.tvec.at<double>(0), ty = armor.tvec.at<double>(1), tz = armor.tvec.at<double>(2);
                double rx = armor.rvec.at<double>(0), ry = armor.rvec.at<double>(1), rz = armor.rvec.at<double>(2);

                std::string tvec_str = cv::format("tvec:  x %6.2f  y %6.2f  z %6.2f", tx, ty, tz);
                std::string rvec_str = cv::format("rvec:  x %6.2f  y %6.2f  z %6.2f", rx, ry, rz);

                cv::putText(final_result, tvec_str, cv::Point(20, text_y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 255), 2);
                text_y_offset += 30;
                cv::putText(final_result, rvec_str, cv::Point(20, text_y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 255), 2);
                text_y_offset += 40;
                if (csv_file.is_open() && run_mode == "video")
                {
                    csv_file << frame_count << ","
                             << tx << "," << ty << "," << tz << ","
                             << armor.yaw << "\n";
                }
            }
        }

        // --- 输出呈现 ---
        cv::imshow("RoboMaster Vision - Raw (2D)", raw_result);
        cv::imshow("RoboMaster Vision - Final (3D)", final_result);

        // 双路视频写入处理
        if (run_mode == "video")
        {
            if (writer_raw.isOpened())
                writer_raw.write(raw_result);
            if (writer_final.isOpened())
                writer_final.write(final_result);
            if (!stream->getFrame(original_frame))
                break;
        }

        // --- 快捷键与双路图片保存逻辑 ---
        char key = (char)cv::waitKey(30);
        if (key == 27)
            break;

        if (key == 's' || key == 'S')
        {
            // 严格以 .png 格式导出双图
            cv::imwrite(raw_output_path, raw_result);
            cv::imwrite(final_output_path, final_result);
            std::cout << "\n[考核交付就绪] 已成功导出：" << std::endl;
            std::cout << "  ✓ " << raw_output_path << std::endl;
            std::cout << "  ✓ " << final_output_path << std::endl;
        }

        frame_count++;
    } while (true);

    // ==========================================
    // 7. 剧终保存配置
    // ==========================================
    cv::FileStorage fs_write(config_file, cv::FileStorage::WRITE);
    if (fs_write.isOpened())
    {
        fs_write << "color_th" << color_th;
        fs_write << "gray_th" << gray_th;
        fs_write << "min_area" << min_area;
        fs_write << "min_angle" << min_angle;
        fs_write.release();
    }
    if (csv_file.is_open())
        csv_file.close();
    if (writer_raw.isOpened())
        writer_raw.release();
    if (writer_final.isOpened())
        writer_final.release();
    cv::destroyAllWindows();
    return 0;
}