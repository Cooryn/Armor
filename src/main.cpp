#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <memory>
#include <algorithm>
#include <fstream>
#include <cmath> // 引入 cmath 以使用 std::remainder
#include <chrono>  // 用于真实时间戳计时

#include "input_stream.hpp"
#include "lightbar_detector.hpp"
#include "solver.hpp"

// 注意：移除了 ekf_predictor.hpp

double deg2rad_0_to_2pi(double angle_deg)
{
    double res = std::fmod(angle_deg, 360.0);
    if (res < 0)
        res += 360.0;
    return res * (CV_PI / 180.0);
}

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
        input_filename = argv[3];
    }

    if (input_filename.empty())
    {
        input_filename = (run_mode == "video") ? "video_1.avi" : "image_1.jpg";
    }

    std::string input_path = input_filename;
    if (!fs::exists(input_path))
    {
        input_path = "./assets/" + run_mode + "/" + input_filename;
        if (!fs::exists(input_path))
        {
            std::cerr << "找不到输入文件: " << input_filename << std::endl;
            return -1;
        }
    }

    // ==========================================
    // 1. 初始化调参变量
    // ==========================================
    // 颜色与高光阈值
    int color_th = 70;
    int gray_th = 170;

    // 几何限制条件 (针对大侧角的放宽默认值)
    int min_area = 40;
    int min_angle = 55;

    int max_angle_diff = 10;
    int max_len_ratio_x10 = 20;    // 最大长度比 2.0 (滑动条为 20)
    int min_aspect_ratio_x10 = 8;  // 最小宽高比 0.8 (滑动条为 8)
    int max_y_diff_ratio_x10 = 8;

    // ==========================================
    // 2. 创建 Debug 控制面板窗口与滑动条
    // ==========================================
    cv::namedWindow("Debug Dashboard", cv::WINDOW_AUTOSIZE);

    cv::createTrackbar("Gray Thresh", "Debug Dashboard", &gray_th, 255);
    cv::createTrackbar("Color Thresh", "Debug Dashboard", &color_th, 255);
    cv::createTrackbar("Max Angle Diff", "Debug Dashboard", &max_angle_diff, 45);
    cv::createTrackbar("Max Len Ratio(x10)", "Debug Dashboard", &max_len_ratio_x10, 50);
    cv::createTrackbar("Min Aspect(x10)", "Debug Dashboard", &min_aspect_ratio_x10, 50);
    cv::createTrackbar("Max Y Diff(x10)", "Debug Dashboard", &max_y_diff_ratio_x10, 30);

    // 创建输出目录
    std::string stem = fs::path(input_path).stem().string();
    std::string out_dir = "./results/";
    fs::create_directories(out_dir);

    // 设置视频输出格式
    std::string ext = (run_mode == "video") ? ".avi" : ".png";
    std::string output_path = out_dir + stem + ext;

    // 初始化输入输出流
    std::unique_ptr<InputStream> stream;
    if (run_mode == "video")
        stream = std::make_unique<VideoFileStream>(input_path);
    else if (run_mode == "image")
        stream = std::make_unique<ImageStream>(input_path);

    cv::Mat original_frame;
    if (!stream->getFrame(original_frame))
        return -1;

    cv::VideoWriter writer;
    std::ofstream csv_file;
    if (run_mode == "video")
    {
        int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        writer.open(output_path, fourcc, 30.0, original_frame.size(), true);

        std::string suffix = stem.substr(stem.find_last_of('_'));
        std::string csv_filename = "pose_raw" + suffix + ".csv";

        // 确保 ./data/ 目录存在
        fs::create_directories("./data/");
        csv_file.open("./data/" + csv_filename);
        if (csv_file.is_open())
        {
            csv_file << "frame_id,timestamp,x,y,z,target_yaw,target_pitch,distance,armor_orientation_yaw\n";
        }
    }

    cv::Mat camera_matrix, distort_coeffs;

    if (stem.find("video_2") != std::string::npos)
    {
        camera_matrix = (cv::Mat_<double>(3, 3) << 1711.311186, 0.000000, 732.488057,
                         0.000000, 1714.616882, 546.930868,
                         0.000000, 0.000000, 1.000000);
        distort_coeffs = (cv::Mat_<double>(1, 5) << -0.119922, -0.078593, 0.007511, -0.028028, 0.000000);
    }
    else
    {
        camera_matrix = (cv::Mat_<double>(3, 3) << 1286.307063384126, 0, 645.34450819155256,
                         0, 1288.1400736562441, 483.6163720308021,
                         0, 0, 1);
        distort_coeffs = (cv::Mat_<double>(1, 5) << -0.47562935060124745, 0.21831745829617311, 0.0004957613589406044, -0.00034617769548693592, 0);
    }

    Solver pnp_solver(camera_matrix, distort_coeffs);
    int frame_count = 0;

    // 用于计算真实时间戳（替代硬编码的 frame_count * 33.33）
    auto video_start_time = std::chrono::steady_clock::now();

    do
    {
        cv::Mat frame = original_frame.clone();

        cv::Mat mask = extractColor(frame, target_color, color_th, gray_th);
        auto contours = extractContours(mask);
        auto lightBars = filterLightBars(contours, 1.5, (double)min_area);
        auto lightRects = getValidLightRects(lightBars, (float)min_angle);
        // 将整型参数还原为浮点数传入匹配函数
        auto armors = matchArmors(lightRects,
                                  max_angle_diff,
                                  max_len_ratio_x10 / 10.0f,
                                  min_aspect_ratio_x10 / 10.0f,
                                  max_y_diff_ratio_x10 / 10.0f);

        cv::Mat raw_result = drawArmors(frame, armors);
        cv::Mat final_result = raw_result.clone();

        int text_y_offset = 30;
        std::vector<Armor> valid_armors;

        for (size_t i = 0; i < armors.size(); i++)
        {
            if (pnp_solver.solve(armors[i]))
            {
                valid_armors.push_back(armors[i]);

                double tx = armors[i].tvec.at<double>(0), ty = armors[i].tvec.at<double>(1), tz = armors[i].tvec.at<double>(2);
                double rx = armors[i].rvec.at<double>(0), ry = armors[i].rvec.at<double>(1), rz = armors[i].rvec.at<double>(2);

                cv::putText(final_result, cv::format("Armor[%zu] tvec: x %5.2f y %5.2f z %5.2f", i, tx, ty, tz),
                            cv::Point(20, text_y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
                text_y_offset += 25;

                cv::putText(final_result, cv::format("Armor[%zu] rvec: x %5.2f y %5.2f z %5.2f", i, rx, ry, rz),
                            cv::Point(20, text_y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
                text_y_offset += 30;
            }
        }

        if (!valid_armors.empty())
        {
            // 使用真实时间戳（毫秒），而非硬编码的固定帧间隔
            auto now = std::chrono::steady_clock::now();
            double timestamp = std::chrono::duration<double, std::milli>(now - video_start_time).count();

            // 🚀 核心修改：不再只取 [0]，而是遍历所有合法的装甲板
            for (size_t i = 0; i < valid_armors.size(); i++)
            {
                Armor current_armor = valid_armors[i];

                // 1. 获取 PnP 算出的原始 x, y, z
                double tx = current_armor.tvec.at<double>(0);
                double ty = current_armor.tvec.at<double>(1);
                double tz = current_armor.tvec.at<double>(2);
                double armor_yaw_rad = current_armor.yaw * CV_PI / 180.0;

                // 2. 解算纯几何维度的目标朝向
                double target_yaw = std::atan2(tx, tz);
                double target_pitch = std::atan2(ty, std::sqrt(tx * tx + tz * tz));
                double distance = std::sqrt(tx * tx + ty * ty + tz * tz);

                // 将角度约束在 [-PI, PI]
                double armor_orientation_yaw = std::remainder(armor_yaw_rad, 2.0 * CV_PI);

                // 3. 写入 CSV (同一个 frame_count 会被写入多次，占多行)
                if (csv_file.is_open() && run_mode == "video")
                {
                    csv_file << frame_count << ","
                             << timestamp << ","
                             << tx << ","
                             << ty << ","
                             << tz << ","
                             << target_yaw << ","
                             << target_pitch << ","
                             << distance << ","
                             << armor_orientation_yaw << "\n";
                }
            }
        }

        cv::imshow("Armor Tracking", final_result);

        if (run_mode == "image")
        {
            // 使用 30ms 延时刷新窗口，让滑动条能够实时响应
            char key = (char)cv::waitKey(30);

            if (key == 27) // 按下 ESC 键退出
            {
                std::cout << "按下了 ESC 键，退出图片 Debug。" << std::endl;
                break;
            }
            else if (key == 's' || key == 'S') // 🌟 进阶技巧：按下 S 键保存当前满意结果
            {
                cv::imwrite(output_path, final_result);
                std::cout << "✅ 当前满意的参数结果已保存至: " << output_path << std::endl;
                std::cout << "当前参数: Gray=" << gray_th << " AngleDiff=" << max_angle_diff
                          << " Aspect=" << min_aspect_ratio_x10 / 10.0f << std::endl;
            }
            // 注意：这里没有 break，程序会自动回到 do-while 开头，用新参数重新处理这同一张图片
        }
        else
        {
            writer.write(final_result);
            if (cv::waitKey(1) == 27)
            {
                std::cout << "按下了 ESC 键退出视频。" << std::endl;
                break;
            }
            if (!stream->getFrame(original_frame))
                break;
            frame_count++;
        }

    } while (true);

    if (writer.isOpened())
        writer.release();
    if (csv_file.is_open())
        csv_file.close();

    // 视频后处理：使用 FFmpeg 转码为 MP4，并删除原 AVI 文件
    if (run_mode == "video")
    {
        std::string mp4 = out_dir + stem + ".mp4";
        std::string cmd_final = "ffmpeg -y -i " + output_path + " -c:v libx264 " + mp4 + " -loglevel quiet";

        int ret = system(cmd_final.c_str());

        if (ret == 0)
        {
            std::remove(output_path.c_str());
        }
        std::cout << "视频已导出至 " << out_dir << " (共处理 " << frame_count << " 帧)" << std::endl;
    }

    cv::destroyAllWindows();
    return 0;
}