#include "camera.hpp"
#include "video_output.hpp"
#include "lightbar_detector.hpp"
#include "solver.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <utility>

CameraSource camera_source = VIDEO;
const int camera_device = 1;
const double camera_recording_fps = 30;
const int video_camera_profile = 2;
const char *const video_file = "assets/video/video_2.avi";
const EnemyColor target_color = ENEMY_RED;

// OV2710实测标定1280×720
const cv::Mat live_camera_matrix = (cv::Mat_<double>(3, 3) << 724.1270087027805, 0, 634.8734753435976,
                                    0, 724.7744702772803, 361.56645765182736,
                                    0, 0, 1);
const cv::Mat live_distortion = (cv::Mat_<double>(1, 5) << 0.19096982316205274, -0.5174000457589348,
                                -0.0016604452786579244, -0.0009416271376343901, 0.5296904922932862);


void run_video(const std::filesystem::path &root)
{
    ::camera camera;
    camera.open(VIDEO, root / std::filesystem::u8path(video_file));
    ::solver solver;
    solver.use_video_profile(video_camera_profile);
    video_output recording(root / "results" / (camera.source == CAMERA ? "camera" : "video") / "detect" /
                              (camera.source == CAMERA ? "offline" : camera.stem), camera);
    bool running = true;
    do
    {
        const auto frame_start = std::chrono::steady_clock::now();
        auto detections = detector::detect(camera.image, target_color);
        const auto poses = solver.solve_frame(std::move(detections), camera.timestamp_ms);
        running = recording.write(camera, solver, poses, frame_start);
    } while (running && camera.next());
    recording.finish();
}

void run_camera(const std::filesystem::path &root)
{
    ::camera camera;
    camera.open(CAMERA, {}, camera_device, camera_recording_fps);
    ::solver solver(live_camera_matrix, live_distortion);
    video_output recording(root / "results" / (camera.source == CAMERA ? "camera" : "video") / "detect" /
                              (camera.source == CAMERA ? "offline" : camera.stem), camera);
    bool running = true;
    do
    {
        const auto frame_start = std::chrono::steady_clock::now();
        auto detections = detector::detect(camera.image, target_color);
        const auto poses = solver.solve_frame(std::move(detections), camera.timestamp_ms);
        running = recording.write(camera, solver, poses, frame_start);
    } while (running && camera.next());
    recording.finish();
}

int main()
{
    try
    {
        const std::filesystem::path root = ARMOR_PROJECT_ROOT;
        if (camera_source == VIDEO)
            run_video(root);
        else
            run_camera(root);
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Armor detection failed: " << error.what() << '\n';
        return 1;
    }
}
