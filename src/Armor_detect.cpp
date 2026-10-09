#include "camera.hpp"
#include "lightbar_detector.hpp"
#include "solver.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <utility>

// Edit this configuration and rebuild. Video paths are relative to the project root.
constexpr CameraSource camera_source = CAMERA_VIDEO;
constexpr int camera_device = 0;
constexpr double camera_recording_fps = 30;
constexpr int video_camera_profile = 2;
constexpr const char *video_file = "assets/video/video_2.avi";
constexpr EnemyColor target_color = ENEMY_RED;
constexpr bool preview = true;

// Fill the actual intrinsics before selecting CAMERA_OPENCV.
const cv::Mat live_camera_matrix;
const cv::Mat live_distortion;

int main()
{
    try
    {
        const std::filesystem::path root = ARMOR_PROJECT_ROOT;
        constexpr bool live = camera_source == CAMERA_OPENCV;
        constexpr const char *window = "Armor detection - Esc to stop";
        Camera camera;
        camera.open({camera_source, root / std::filesystem::u8path(video_file), camera_device, camera_recording_fps});
        Solver pnp(live ? live_camera_matrix : cv::Mat(), live ? live_distortion : cv::Mat());
        if constexpr (!live)
            pnp.use_video_profile(video_camera_profile);
        if constexpr (preview)
        {
            cv::namedWindow(window, cv::WINDOW_NORMAL);
            cv::resizeWindow(window, 960, 720);
        }
        std::int64_t processed = 0;
        bool running = true;
        do
        {
            const auto frame_start = std::chrono::steady_clock::now();
            auto detections = detectArmors(camera.image_, target_color);
            const auto poses = pnp.solve_frame(std::move(detections), camera.timestamp_ms);
            cv::Mat canvas = camera.image_.clone();
            drawArmors(canvas, poses.armors);
            drawVideoInfo(canvas, poses.armors, cv::format("%s Detect frame %lld",
                live ? "LIVE" : "REPLAY", static_cast<long long>(camera.frame_id_)));
            ++processed;
            if constexpr (preview)
            {
                cv::imshow(window, canvas);
                const auto deadline = frame_start + std::chrono::duration<double>(live ? 0 : 1 / camera.fps_);
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
                        // A normally closed Win32 window is reported as StsNullPtr.
                        if (error.code != cv::Error::StsNullPtr)
                            throw;
                        running = false;
                    }
                } while (running && std::chrono::steady_clock::now() < deadline);
            }
        } while (running && camera.next());
        if constexpr (preview)
            cv::destroyAllWindows();
        std::cout << "Processed " << processed << " frames.\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Armor detection failed: " << error.what() << '\n';
        return 1;
    }
}
