#include "camera.hpp"
#include "lightbar_detector.hpp"
#include "solver.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <utility>

CameraSource camera_source = CAMERA;
const int camera_device = 1; // AX USB2.0（外接 OV2710），0 为笔记本内置摄像头
const double camera_recording_fps = 60;
const int video_camera_profile = 2;
const char *const video_file = "assets/video/video_2.avi";
const EnemyColor target_color = ENEMY_RED;
bool preview = true;

// 临时使用视频 2 的 1280×1024 标定，OV2710 标定完成后替换。
const cv::Mat live_camera_matrix = (cv::Mat_<double>(3, 3) << 1711.311186, 0, 732.488057,
                                    0, 1714.616882, 546.930868,
                                    0, 0, 1);
const cv::Mat live_distortion = (cv::Mat_<double>(1, 5) << -.119922, -.078593, .007511, -.028028, 0);

const char *const window = "Armor detection - Esc to stop";

void run_video(const std::filesystem::path &root)
{
    ::camera camera;
    camera.open(VIDEO, root / std::filesystem::u8path(video_file));
    ::solver solver;
    solver.use_video_profile(video_camera_profile);
    if (preview)
    {
        cv::namedWindow(window, cv::WINDOW_NORMAL);
        cv::resizeWindow(window, 960, 720);
    }
    std::int64_t processed = 0;
    bool running = true;
    do
    {
        const auto frame_start = std::chrono::steady_clock::now();
        auto detections = detector::detect(camera.image, target_color);
        const auto poses = solver.solve_frame(std::move(detections), camera.timestamp_ms);
        cv::Mat canvas = camera.image.clone();
        detector::draw_armors(canvas, poses.armors);
        detector::draw_video_info(canvas, poses.armors, cv::format("REPLAY Detect frame %lld", static_cast<long long>(camera.frame_id)));
        ++processed;
        if (preview)
        {
            cv::imshow(window, canvas);
            const auto deadline = frame_start + std::chrono::duration<double>(1 / camera.fps);
            do
            {
                const double remaining = std::chrono::duration<double, std::milli>(deadline - std::chrono::steady_clock::now()).count();
                if ((cv::waitKey(static_cast<int>(std::clamp(std::ceil(remaining), 1.0, 10.0))) & 0xff) == 27)
                {
                    running = false;
                    break;
                }
                try
                {
                    running = cv::getWindowProperty(window, cv::WND_PROP_VISIBLE) >= 1;
                }
                catch (const cv::Exception &error)
                {
                    if (error.code != cv::Error::StsNullPtr)
                        throw;
                    running = false;
                }
            } while (running && std::chrono::steady_clock::now() < deadline);
        }
    } while (running && camera.next());
    if (preview)
        cv::destroyAllWindows();
    std::cout << "Processed " << processed << " frames.\n";
}

void run_camera()
{
    ::camera camera;
    camera.open(CAMERA, {}, camera_device, camera_recording_fps);
    ::solver solver(live_camera_matrix, live_distortion);
    if (preview)
    {
        cv::namedWindow(window, cv::WINDOW_NORMAL);
        cv::resizeWindow(window, 960, 720);
    }
    std::int64_t processed = 0;
    bool running = true;
    do
    {
        auto detections = detector::detect(camera.image, target_color);
        const auto poses = solver.solve_frame(std::move(detections), camera.timestamp_ms);
        cv::Mat canvas = camera.image.clone();
        detector::draw_armors(canvas, poses.armors);
        detector::draw_video_info(canvas, poses.armors, cv::format("LIVE Detect frame %lld", static_cast<long long>(camera.frame_id)));
        ++processed;
        if (preview)
        {
            cv::imshow(window, canvas);
            if ((cv::waitKey(1) & 0xff) == 27)
                running = false;
            else
            {
                try
                {
                    running = cv::getWindowProperty(window, cv::WND_PROP_VISIBLE) >= 1;
                }
                catch (const cv::Exception &error)
                {
                    if (error.code != cv::Error::StsNullPtr)
                        throw;
                    running = false;
                }
            }
        }
    } while (running && camera.next());
    if (preview)
        cv::destroyAllWindows();
    std::cout << "Processed " << processed << " frames.\n";
}

int main()
{
    try
    {
        const std::filesystem::path root = ARMOR_PROJECT_ROOT;
        if (camera_source == VIDEO)
            run_video(root);
        else
            run_camera();
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Armor detection failed: " << error.what() << '\n';
        return 1;
    }
}
