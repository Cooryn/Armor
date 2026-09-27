#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <cctype>
#include <algorithm>
#include <fstream>
#include <cmath> // 引入 cmath 以使用 std::remainder
#include <iomanip>
#include <chrono>

#include "lightbar_detector.hpp"
#include "solver.hpp"

namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    if (argc == 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        std::cout << "Usage: " << argv[0] << " [red|blue] [video file] [--headless]\n"
                  << "Live preview at source FPS; Esc or close window to stop. --headless disables preview.\n"
                  << "Defaults: red video_1.avi.\n";
        std::cout << "Outputs: results/*.avi and data/pose_raw_*.csv\n";
        return 0;
    }
    if (argc > 4 || (argc == 4 && std::string(argv[3]) != "--headless")) {
        std::cerr << "Usage: " << argv[0] << " [red|blue] [video file] [--headless]" << std::endl;
        return 1;
    }
    const bool preview = argc < 4;
    std::string color = argc >= 2 ? argv[1] : "red";
    std::transform(color.begin(), color.end(), color.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (color != "red" && color != "blue") {
        std::cerr << "Color must be red or blue" << std::endl;
        return 1;
    }
    const EnemyColor target_color = color == "red" ? EnemyColor::RED : EnemyColor::BLUE;
    const std::string input_filename = argc >= 3 ? argv[2] : "video_1.avi";
    std::string input_path = input_filename;
    if (!fs::exists(input_path)) input_path = (fs::path("assets/video") / input_filename).string();
    std::string extension = fs::path(input_path).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (extension != ".avi" && extension != ".mp4" && extension != ".mov" &&
        extension != ".mkv" && extension != ".m4v" && extension != ".webm" &&
        extension != ".mpg" && extension != ".mpeg" && extension != ".wmv") {
        std::cerr << "Unsupported video file extension: " << extension << std::endl;
        return 1;
    }
    cv::VideoCapture stream(input_path);
    cv::Mat original_frame;
    if (!stream.isOpened() || !stream.read(original_frame)) {
        std::cerr << "Cannot read video: " << input_path << std::endl;
        return 1;
    }
    const double source_fps = stream.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(source_fps) || source_fps <= 0) {
        std::cerr << "Invalid source video FPS" << std::endl;
        return 1;
    }
    const char *window = "Armor live preview - Esc to stop";
    if (preview) {
        try {
            cv::namedWindow(window, cv::WINDOW_NORMAL);
            cv::resizeWindow(window, 960, 720);
        } catch (const cv::Exception &error) {
            std::cerr << "Cannot create OpenCV window; use --headless. " << error.what() << std::endl;
            return 1;
        }
    }
    constexpr int color_th = 70, gray_th = 170, min_area = 40;
    constexpr float min_angle = 55, max_angle_diff = 20;
    std::string stem = fs::path(input_path).stem().string();
    std::string out_dir = "./results/";
    fs::create_directories(out_dir);

    std::string output_path = out_dir + stem + ".avi";

    cv::VideoWriter writer;
    std::ofstream csv_file;
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    cv::Size output_size = original_frame.size();
    writer.open(output_path, fourcc, source_fps, output_size, true);
    if (!writer.isOpened()) { std::cerr << "Cannot open output video" << std::endl; return 1; }

    const auto separator = stem.find_last_of('_');
    std::string suffix = separator == std::string::npos ? "_" + stem : stem.substr(separator);
    std::string csv_filename = "pose_raw" + suffix + ".csv";

    // 确保 ./data/ 目录存在
    fs::create_directories("./data/");
    csv_file.open("./data/" + csv_filename);
    if (!csv_file.is_open()) { std::cerr << "Cannot open output CSV" << std::endl; return 1; }
    csv_file << std::setprecision(15);
    csv_file << "frame_id,timestamp,x,y,z,target_yaw,target_pitch,distance,armor_orientation_yaw,detection_score,reprojection_error,pnp_candidate_count,pnp_used_temporal,rvec_x,rvec_y,rvec_z,coordinate_frame\n";

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

    // Frame timestamps use the source video clock.

    do
    {
        const auto frame_start = std::chrono::steady_clock::now();
        cv::Mat frame = original_frame.clone();

        cv::Mat mask = extractColor(frame, target_color, color_th, gray_th);
        auto contours = extractContours(mask);
        std::vector<float> light_quality;
        auto lightRects = getValidLightRects(contours, (float)min_angle, &light_quality, 1.5, min_area);
        auto armors = matchArmors(lightRects, max_angle_diff, 2.f, .8f, .8f, 3.1f, .35f, light_quality);

        cv::Mat final_result = frame.clone();

        int text_y_offset = 30;
        std::vector<Armor> valid_armors;
        const double source_time = frame_count/source_fps;
        const auto yaw_hints = pnp_solver.yawHints(armors, source_time);

        for (size_t i = 0; i < armors.size(); i++)
        {
            bool solved = pnp_solver.solve(armors[i], yaw_hints[i]);
            if (solved)
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

        pnp_solver.finishFrame(valid_armors, source_time);
        drawArmors(final_result, valid_armors);
        if (!valid_armors.empty())
        {
            // Source video time in milliseconds.
            double timestamp = frame_count * 1000.0 / source_fps;

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
                csv_file << frame_count << ","
                         << timestamp << ","
                         << tx << ","
                         << ty << ","
                         << tz << ","
                         << target_yaw << ","
                         << target_pitch << ","
                         << distance << ","
                         << armor_orientation_yaw << ","
                         << current_armor.detection_score << ","
                         << current_armor.reprojection_error << ","
                         << current_armor.pnp_candidate_count << ","
                         << current_armor.pnp_used_temporal << ","
                         << current_armor.rvec.at<double>(0) << ","
                         << current_armor.rvec.at<double>(1) << ","
                         << current_armor.rvec.at<double>(2) << ",camera\n";

            }
        }


        writer.write(final_result);
        ++frame_count;
        if (preview) {
            cv::imshow(window, final_result);
            const auto deadline = frame_start + std::chrono::duration<double>(1.0 / source_fps);
            bool stop = false;
            do {
                const double remaining_ms = std::chrono::duration<double, std::milli>(
                    deadline - std::chrono::steady_clock::now()).count();
                const int delay = static_cast<int>(std::clamp(std::ceil(remaining_ms), 1.0, 10.0));
                stop = (cv::waitKey(delay) & 0xff) == 27;
                if (!stop) {
                    // Qt destroys its receiver when the last window is closed.
                    try { stop = cv::getWindowProperty(window, cv::WND_PROP_VISIBLE) < 1; }
                    catch (const cv::Exception &) { stop = true; }
                }
                if (stop) break;
            } while (std::chrono::steady_clock::now() < deadline);
            if (stop) break;
        }
        if (!stream.read(original_frame)) break;

    } while (true);

    writer.release();
    csv_file.close();
    if (preview) cv::destroyAllWindows();

    std::cout << "视频已导出至 " << out_dir << " (共处理 " << frame_count << " 帧)" << std::endl;

    return 0;
}
