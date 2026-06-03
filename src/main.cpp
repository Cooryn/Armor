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

namespace fs = std::filesystem;

void printHelp(const char *prog_name)
{
    std::cout << "====================================================\n"
              << "用法: " << prog_name << " [运行模式] [目标颜色] [文件名]\n\n"
              << "参数说明:\n"
              << "  [运行模式] : 'image' 或 'video' (默认: image)\n"
              << "  [目标颜色] : 'red' 或 'blue' (默认: red)\n"
              << "  [文件名]   : 例如 'image_1.jpg' 或 'video_2.avi'\n"
              << "====================================================\n";
}

int main(int argc, char **argv)
{
    // ==========================================
    // 1. 命令行参数灵活解析
    // ==========================================
    if (argc >= 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help"))
    {
        printHelp(argv[0]);
        return 0;
    }

    std::string run_mode = "image";
    EnemyColor target_color = EnemyColor::RED;
    std::string input_filename = "";

    if (argc >= 2)
        run_mode = argv[1];

    if (argc >= 3)
    {
        std::string color_arg = argv[2];
        std::transform(color_arg.begin(), color_arg.end(), color_arg.begin(), ::tolower);
        if (color_arg == "blue")
            target_color = EnemyColor::BLUE;
    }

    if (argc >= 4)
    {
        input_filename = argv[3]; // 获取你输入的 image_1.jpg
    }

    // 设置默认 fallback 文件
    if (input_filename.empty())
    {
        input_filename = (run_mode == "video") ? "video_1.avi" : "image_1.jpg";
    }

    // 【智能路径寻址】：如果你只输入了文件名，程序会自动去 assets 里找
    std::string input_path = input_filename;
    if (!fs::exists(input_path))
    {
        input_path = "./assets/" + run_mode + "/" + input_filename;
        if (!fs::exists(input_path))
        {
            std::cerr << "[致命错误] 找不到输入文件: " << input_filename << std::endl;
            std::cerr << "请确保文件存在于当前目录，或存在于 ./assets/" << run_mode << "/ 目录下。" << std::endl;
            return -1;
        }
    }

    std::cout << "[信息] 成功加载文件: " << input_path << " | 模式: " << run_mode << " | 目标: " << (target_color == EnemyColor::RED ? "红方" : "蓝方") << std::endl;

    // ==========================================
    // 2. 读取 YAML 配置与初始化
    // ==========================================
    int color_th = 20, gray_th = 80, min_area = 10, min_angle = 55;
    cv::FileStorage fs_read("./assets/config.yaml", cv::FileStorage::READ);
    if (fs_read.isOpened())
    {
        fs_read["color_th"] >> color_th;
        fs_read["gray_th"] >> gray_th;
        fs_read["min_area"] >> min_area;
        fs_read["min_angle"] >> min_angle;
        fs_read.release();
    }

    // 创建输出目录
    std::string stem = fs::path(input_path).stem().string();
    std::string out_dir = "./results/";
    fs::create_directories(out_dir);

    // 强制按考核标准设置输出格式
    std::string ext = (run_mode == "video") ? ".avi" : ".png";
    std::string raw_output_path = out_dir + stem + "_raw" + ext;
    std::string final_output_path = out_dir + stem + ext;

    // 初始化输入输出流
    std::unique_ptr<InputStream> stream;
    if (run_mode == "video")
        stream = std::make_unique<VideoFileStream>(input_path);
    else if (run_mode == "image")
        stream = std::make_unique<ImageStream>(input_path);

    cv::Mat original_frame;
    if (!stream->getFrame(original_frame))
        return -1;

    cv::VideoWriter writer_raw, writer_final;
    std::ofstream csv_file;
    if (run_mode == "video")
    {
        int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        writer_raw.open(raw_output_path, fourcc, 30.0, original_frame.size(), true);
        writer_final.open(final_output_path, fourcc, 30.0, original_frame.size(), true);

        // ==========================================
        // 【核心修改】：智能拼接 CSV 文件名
        // ==========================================
        // 假设 stem 是 "video_1" 或 "video_2"
        size_t pos = stem.find_last_of('_');
        // 截取最后下划线及后面的内容，得到 "_1" 或 "_2"
        std::string suffix = (pos != std::string::npos) ? stem.substr(pos) : "";

        // 完美拼凑成 "pose_data_1.csv"
        std::string csv_filename = "pose_data" + suffix + ".csv";

        csv_file.open(out_dir + csv_filename);
        if (csv_file.is_open())
            csv_file << "Frame,X,Y,Z,Yaw\n";
    }

    // ==========================================
    // 动态初始化 PnP 解算器 (根据文件名自动适配内参)
    // ==========================================
    cv::Mat camera_matrix, distort_coeffs;

    // 通过检测文件名 stem 是否包含 "_2"，来判断是不是第二组数据
    if (stem.find("_2") != std::string::npos)
    {
        // video_2 / image_2 的专属内参
        camera_matrix = (cv::Mat_<double>(3, 3) << 1711.311186, 0.000000, 732.488057,
                         0.000000, 1714.616882, 546.930868,
                         0.000000, 0.000000, 1.000000);
        distort_coeffs = (cv::Mat_<double>(1, 5) << -0.119922, -0.078593, 0.007511, -0.028028, 0.000000);
        std::cout << "加载 video_2 内参矩阵。" << std::endl;
    }
    else
    {
        // 默认 / video_1 / image_1 的专属内参
        camera_matrix = (cv::Mat_<double>(3, 3) << 1286.307063384126, 0, 645.34450819155256,
                         0, 1288.1400736562441, 483.6163720308021,
                         0, 0, 1);
        distort_coeffs = (cv::Mat_<double>(1, 5) << -0.47562935060124745, 0.21831745829617311, 0.0004957613589406044, -0.00034617769548693592, 0);
        std::cout << "加载 video_1 内参矩阵。" << std::endl;
    }

    Solver pnp_solver(camera_matrix, distort_coeffs);

    int frame_count = 0;

    // ==========================================
    // 3. 核心处理循环 (无滑动条，直接出图/出视频)
    // ==========================================
    do
    {
        cv::Mat frame = original_frame.clone();

        cv::Mat mask = extractColor(frame, target_color, color_th, gray_th);
        auto contours = extractContours(mask);
        auto lightBars = filterLightBars(contours, 1.5, (double)min_area);
        auto lightRects = getValidLightRects(lightBars, (float)min_angle);
        auto armors = matchArmors(lightRects);

        cv::Mat raw_result = drawArmors(frame, armors);
        cv::Mat final_result = raw_result.clone();

        int text_y_offset = 30;
        for (size_t i = 0; i < armors.size(); i++)
        {
            if (pnp_solver.solve(armors[i]))
            {
                double tx = armors[i].tvec.at<double>(0), ty = armors[i].tvec.at<double>(1), tz = armors[i].tvec.at<double>(2);
                double rx = armors[i].rvec.at<double>(0), ry = armors[i].rvec.at<double>(1), rz = armors[i].rvec.at<double>(2);

                cv::putText(final_result, cv::format("tvec:  x %6.2f  y %6.2f  z %6.2f", tx, ty, tz), cv::Point(20, text_y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 255), 2);
                text_y_offset += 30;
                cv::putText(final_result, cv::format("rvec:  x %6.2f  y %6.2f  z %6.2f", rx, ry, rz), cv::Point(20, text_y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 255), 2);
                text_y_offset += 40;

                if (csv_file.is_open() && run_mode == "video")
                {
                    csv_file << frame_count << "," << tx << "," << ty << "," << tz << "," << armors[i].yaw << "\n";
                }
            }
        }

        // --- 结果展示与保存 ---
        cv::imshow("Raw Detection", raw_result);
        cv::imshow("Final PnP", final_result);

        if (run_mode == "image")
        {
            // 如果是图片模式，处理完直接严格保存并等待键盘输入退出
            cv::imwrite(raw_output_path, raw_result);
            cv::imwrite(final_output_path, final_result);
            std::cout << "[成功] 图片已导出至 " << out_dir << std::endl;
            std::cout << "请在图像窗口按任意键退出..." << std::endl;
            cv::waitKey(0);
            break;
        }
        else
        {
            // 如果是视频模式，写入流中，并允许中途按 ESC 退出
            writer_raw.write(raw_result);
            writer_final.write(final_result);
            if (cv::waitKey(1) == 27)
            {
                std::cout << "[中止] 用户按下了 ESC 键退出视频。" << std::endl;
                break;
            }
            if (!stream->getFrame(original_frame))
                break; // 视频结束
            frame_count++;
        }

    } while (true);

    if (writer_raw.isOpened())
        writer_raw.release();
    if (writer_final.isOpened())
        writer_final.release();
    if (csv_file.is_open())
        csv_file.close();

    if (run_mode == "video")
    {
        std::cout << "[成功] 视频已导出至 " << out_dir << " (共处理 " << frame_count << " 帧)" << std::endl;
    }

    cv::destroyAllWindows();
    return 0;
}