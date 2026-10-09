#include "camera.hpp"
#include "lightbar_detector.hpp"
#include "solver.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <utility>

CameraSource camera_source = VIDEO;
const int camera_device = 0;
const double camera_recording_fps = 30;
const int video_camera_profile = 2;
const char *const video_file = "assets/video/video_2.avi";
const EnemyColor target_color = ENEMY_RED;
bool preview = true;

const cv::Mat live_camera_matrix;
const cv::Mat live_distortion;

int main()
{
    try
    {
        const std::filesystem::path root = ARMOR_PROJECT_ROOT;
        const char *const window = "Armor detection - Esc to stop";
        Camera camera;
        camera.open({camera_source, root / std::filesystem::u8path(video_file), camera_device, camera_recording_fps});
        Solver pnp(camera_source == CAMERA ? live_camera_matrix : cv::Mat(), camera_source == CAMERA ? live_distortion : cv::Mat());
        if (camera_source == VIDEO)
            pnp.use_video_profile(video_camera_profile);
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
            auto detections = detectArmors(camera.image_, target_color);
            const auto poses = pnp.solve_frame(std::move(detections), camera.timestamp_ms);
            cv::Mat canvas = camera.image_.clone();
            drawArmors(canvas, poses.armors);
            drawVideoInfo(canvas, poses.armors, cv::format("%s Detect frame %lld",
                camera_source == CAMERA ? "LIVE" : "REPLAY", static_cast<long long>(camera.frame_id_)));
            ++processed;
            if (preview)
            {
                cv::imshow(window, canvas);
                const auto deadline = frame_start + std::chrono::duration<double>(camera_source == CAMERA ? 0 : 1 / camera.fps_);
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
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Armor detection failed: " << error.what() << '\n';
        return 1;
    }
}
