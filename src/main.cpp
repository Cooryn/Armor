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

class VehicleTracker
{
public:
    bool is_initialized = false;

    cv::Point3f vehicle_center;
    double vehicle_yaw = 0.0;

    void update(const std::vector<Armor> &detected_armors)
    {
        if (detected_armors.empty())
            return;

        cv::Point3f sum_center(0, 0, 0);

        for (const auto &armor : detected_armors)
        {
            cv::Mat rmat;
            cv::Rodrigues(armor.rvec, rmat);
            cv::Mat offset = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 0.25);
            cv::Mat center_mat = armor.tvec + rmat * offset;

            sum_center.x += center_mat.at<double>(0);
            sum_center.y += center_mat.at<double>(1);
            sum_center.z += center_mat.at<double>(2);
        }

        vehicle_center.x = sum_center.x / detected_armors.size();
        vehicle_center.y = sum_center.y / detected_armors.size();
        vehicle_center.z = sum_center.z / detected_armors.size();

        double current_armor_yaw = detected_armors[0].yaw;

        if (!is_initialized)
        {
            vehicle_yaw = current_armor_yaw;
            is_initialized = true;
        }
        else
        {
            double diff = current_armor_yaw - vehicle_yaw;
            int face_offset = std::round(diff / 90.0);
            vehicle_yaw = current_armor_yaw - (face_offset * 90.0);
        }
    }
};

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

    int color_th = 70, gray_th = 250, min_area = 40, min_angle = 55;

    // 创建输出目录
    std::string stem = fs::path(input_path).stem().string();
    std::string out_dir = "./results/";
    fs::create_directories(out_dir);

    // 设置输出格式
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

        std::string suffix = stem.substr(stem.find_last_of('_'));

        std::string csv_filename = "pose_data" + suffix + ".csv";

        csv_file.open(out_dir + csv_filename);
        if (csv_file.is_open())
            csv_file << "Frame,X,Z,Yaw\n";
    }

    cv::Mat camera_matrix, distort_coeffs;

    if (stem.find("video_2") != std::string::npos)
    {
        camera_matrix = (cv::Mat_<double>(3, 3) << 1711.311186, 0.000000, 732.488057,
                         0.000000, 1714.616882, 546.930868,
                         0.000000, 0.000000, 1.000000);
        distort_coeffs = (cv::Mat_<double>(1, 5) << -0.119922, -0.078593, 0.007511, -0.028028, 0.000000);
    }
    else if (stem.find("video_1") != std::string::npos)
    {
        camera_matrix = (cv::Mat_<double>(3, 3) << 1286.307063384126, 0, 645.34450819155256,
                         0, 1288.1400736562441, 483.6163720308021,
                         0, 0, 1);
        distort_coeffs = (cv::Mat_<double>(1, 5) << -0.47562935060124745, 0.21831745829617311, 0.0004957613589406044, -0.00034617769548693592, 0);
    }
    else if (stem.find("image") != std::string::npos)
    {
        camera_matrix = (cv::Mat_<double>(3, 3) << 1286.307063384126, 0, 645.34450819155256,
                         0, 1288.1400736562441, 483.6163720308021,
                         0, 0, 1);
        distort_coeffs = (cv::Mat_<double>(1, 5) << -0.47562935060124745, 0.21831745829617311, 0.0004957613589406044, -0.00034617769548693592, 0);
    }

    Solver pnp_solver(camera_matrix, distort_coeffs);

    int frame_count = 0;
    VehicleTracker vehicle_tracker;

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
            std::sort(valid_armors.begin(), valid_armors.end(), [](const Armor &a, const Armor &b)
                      { return a.tvec.at<double>(2) < b.tvec.at<double>(2); });

            vehicle_tracker.update(valid_armors);

            cv::putText(final_result, cv::format("Yaw: %.1f", vehicle_tracker.vehicle_yaw), cv::Point(20, text_y_offset + 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);

            if (csv_file.is_open() && run_mode == "video")
            {
                csv_file << frame_count << ","
                         << vehicle_tracker.vehicle_center.x << ","
                         << vehicle_tracker.vehicle_center.z << ","
                         << vehicle_tracker.vehicle_yaw << "\n";
            }
        }

        cv::imshow("Armor", final_result);

        if (run_mode == "image")
        {
            cv::imwrite(raw_output_path, raw_result);
            cv::imwrite(final_output_path, final_result);
            std::cout << "图片已导出至 " << out_dir << std::endl;
            std::cout << "按任意键退出..." << std::endl;
            cv::waitKey(0);
            break;
        }
        else
        {
            writer_raw.write(raw_result);
            writer_final.write(final_result);
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

    if (writer_raw.isOpened())
        writer_raw.release();
    if (writer_final.isOpened())
        writer_final.release();
    if (csv_file.is_open())
        csv_file.close();

    if (run_mode == "video")
    {
        std::string raw_mp4 = out_dir + stem + "_raw.mp4";
        std::string final_mp4 = out_dir + stem + ".mp4";
        std::string cmd_raw = "ffmpeg -y -i " + raw_output_path + " -c:v libx264 " + raw_mp4 + " -loglevel quiet";
        std::string cmd_final = "ffmpeg -y -i " + final_output_path + " -c:v libx264 " + final_mp4 + " -loglevel quiet";

        int ret1 = system(cmd_raw.c_str());
        int ret2 = system(cmd_final.c_str());

        if (ret1 == 0 && ret2 == 0)
        {
            std::remove(raw_output_path.c_str());
            std::remove(final_output_path.c_str());
        }
    }

    if (run_mode == "video")
    {
        std::cout << "视频已导出至 " << out_dir << " (共处理 " << frame_count << " 帧)" << std::endl;
    }

    cv::destroyAllWindows();
    return 0;
}